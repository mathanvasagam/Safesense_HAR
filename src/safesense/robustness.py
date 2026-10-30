"""Deterministic, zero-padded robustness scenarios for SafeSense inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .safesense import (
    FRAME_COUNT,
    LAG_SHIFTS,
    SAFESENSE_CONDITIONAL_GATE_THRESHOLD,
    SAFESENSE_CONDITIONAL_SEEDS,
    SAFESENSE_CONDITIONAL_WEIGHTS,
    SAFESENSE_EARLY_EXIT_IMU_CHANNEL,
    SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT,
    SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD,
    SAFESENSE_EARLY_EXIT_SEEDS,
    SAFESENSE_EARLY_EXIT_SKELETON_CHANNEL,
    SafeSense,
    SafeSenseBranchExpert,
    SafeSenseConditionalEnsemble,
    SafeSenseEarlyExitEnsemble,
    _feature_arrays,
    _metrics,
    parameter_count,
    safesense_splits,
)


class RobustnessError(ValueError):
    """Raised for invalid robustness inputs or scenarios."""


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    category: str
    modality: str | None = None
    shift: int = 0
    loss_fraction: float = 0.0
    noise_std: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def scenario_specs() -> tuple[CorruptionSpec, ...]:
    specs = [CorruptionSpec("clean", "clean")]
    for modality in ("skeleton", "imu"):
        for shift in (-4, -2, 2, 4):
            specs.append(CorruptionSpec(f"{modality}_shift_{shift:+d}", "shift", modality, shift=shift))
        for fraction in (0.1, 0.3, 0.5):
            specs.append(CorruptionSpec(f"{modality}_frame_loss_{int(fraction * 100)}", "frame_loss", modality, loss_fraction=fraction))
        for noise in (0.1, 0.3, 0.5):
            specs.append(CorruptionSpec(f"{modality}_noise_{noise}", "noise", modality, noise_std=noise))
    specs.extend((
        CorruptionSpec("missing_skeleton", "missing", "skeleton"),
        CorruptionSpec("missing_imu", "missing", "imu"),
        CorruptionSpec("missing_both", "missing", "both"),
        CorruptionSpec("imu_composite_shift_+4_loss_30_noise_0.3", "composite", "imu", shift=4, loss_fraction=0.3, noise_std=0.3),
    ))
    return tuple(specs)


def corrective_imu_shift(spec: CorruptionSpec) -> int:
    """Return the IMU correction needed to align with the skeleton anchor."""
    if not spec.shift:
        return 0
    return int(spec.shift) if spec.modality == "skeleton" else -int(spec.shift)


def base_temporal_masks(fused: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(fused)
    if values.ndim != 5 or values.shape[1:] != (1, FRAME_COUNT, 22, 3):
        raise RobustnessError(f"expected fused array [N,1,128,22,3], got {values.shape}")
    sequence = values[:, 0]
    # Official preprocessing repeats/pads skeleton frames to 128. The appended
    # IMU nodes retain the only reliable clean temporal validity signal.
    clean_mask = np.any(sequence[:, :, 20:] != 0, axis=(2, 3))
    return {
        "skeleton": clean_mask.copy(),
        "imu": clean_mask.copy(),
    }


def _shift(values: np.ndarray, amount: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if amount >= 0:
        if amount < values.shape[1]:
            shifted[:, amount:] = values[:, : values.shape[1] - amount]
    elif -amount < values.shape[1]:
        shifted[:, : amount] = values[:, -amount:]
    return shifted


def _apply_shift(values: np.ndarray, mask: np.ndarray, amount: int) -> tuple[np.ndarray, np.ndarray]:
    return _shift(values, amount), _shift(mask.astype(np.float32)[..., None], amount)[..., 0].astype(bool)


def training_noise_scales(fused: np.ndarray, masks: dict[str, np.ndarray], train_indices: Sequence[int]) -> dict[str, float]:
    values = np.asarray(fused)[:, 0]
    indices = np.asarray(train_indices, dtype=np.int64)
    scales: dict[str, float] = {}
    for name, slc in (("skeleton", slice(0, 20)), ("imu", slice(20, 22))):
        valid_values = values[indices][:, :, slc][masks[name][indices]]
        if valid_values.size == 0:
            raise RobustnessError(f"no valid training values for {name} noise scale")
        scales[name] = float(valid_values.std(dtype=np.float64)) or 1.0
    return scales


def _frame_loss(mask: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    result = mask.copy()
    for row in range(mask.shape[0]):
        valid = np.flatnonzero(mask[row])
        count = min(len(valid), int(round(len(valid) * fraction)))
        if count:
            result[row, rng.choice(valid, size=count, replace=False)] = False
    return result


def apply_corruption(
    fused: np.ndarray,
    spec: CorruptionSpec,
    *,
    seed: int,
    scales: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(fused, dtype=np.float32).copy()
    masks = base_temporal_masks(values)
    rng = np.random.default_rng(seed)
    modalities = ("skeleton", "imu") if spec.modality == "both" else ((spec.modality,) if spec.modality else ())
    if spec.name == "clean":
        return values, masks
    for modality in modalities:
        slc = slice(0, 20) if modality == "skeleton" else slice(20, 22)
        if spec.name.startswith("missing"):
            values[:, 0, :, slc] = 0
            masks[modality][:] = False
            continue
        branch = values[:, 0, :, slc]
        branch_mask = masks[modality]
        if spec.shift:
            branch, branch_mask = _apply_shift(branch, branch_mask, spec.shift)
        if spec.loss_fraction:
            old_mask = branch_mask
            branch_mask = _frame_loss(old_mask, spec.loss_fraction, rng)
            branch[old_mask & ~branch_mask] = 0
        if spec.noise_std:
            noise = rng.normal(0.0, scales[modality] * spec.noise_std, size=branch.shape).astype(np.float32)
            branch = np.where(branch_mask[..., None, None], branch + noise, branch)
        values[:, 0, :, slc] = branch
        masks[modality] = branch_mask
    return values, masks


def aggregate_metrics(records: Sequence[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise RobustnessError("cannot aggregate empty robustness records")
    by_category: dict[str, list[float]] = {}
    for record in records:
        if record["scenario"] == "clean":
            continue
        by_category.setdefault(str(record["category"]), []).append(float(record["accuracy"]))
    category_means = {category: float(np.mean(values)) for category, values in by_category.items()}
    return {
        "category_accuracy_mean": category_means,
        "equal_category_corrupted_accuracy_mean": float(np.mean(list(category_means.values()))) if category_means else None,
    }


def select_fusion_final_checkpoints(result_paths: Sequence[str | Path]) -> dict[int, str]:
    """Select only recorded final weights; retained-best files are never accepted."""
    selected: dict[int, str] = {}
    for result_path in result_paths:
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        seed = int(result["seed"])
        final_path = result.get("final_checkpoint") or result.get("weights_path")
        if not final_path:
            raise RobustnessError(f"missing final Fusion-GCN checkpoint for seed {seed}")
        selected[seed] = str(final_path)
    if set(selected) != {11, 22, 33}:
        raise RobustnessError("Fusion-GCN robustness requires final checkpoints for seeds 11, 22, 33")
    return selected


def write_robustness_json(path: str | Path, result: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fusion_final_model(repo_dir: str | Path, checkpoint: str | Path, device: "torch.device") -> object:
    """Construct the official model from a runtime checkout and load FINAL weights."""
    import sys
    import torch

    from .fusion_gcn import apply_runtime_compatibility

    root = Path(repo_dir).resolve()
    apply_runtime_compatibility(root)
    paths = [str(root), str(root / "torch_src")]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    from datasets.utd_mhad.constants import center_joint, skeleton_edges
    from models.mmargcn.mmargcn import Model
    from util.graph import Graph

    model = Model(
        {"skeleton": (1, 128, 22, 3)},
        27,
        Graph(skeleton_edges, center_joint=center_joint),
        mode="skeleton_imu_spatial_fusion",
        num_imu_joints=2,
        imu_enhanced_mode="append_center",
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def _load_safesense_model(checkpoint: str | Path, device: "torch.device", *, safesense_variant: str = "full") -> object:
    import torch

    model = SafeSense(variant=safesense_variant).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def _predict_batches(model: object, arrays: Sequence[np.ndarray], device: "torch.device", *, mask: np.ndarray | None = None, batch_size: int = 64) -> np.ndarray:
    import torch

    count = len(arrays[0])
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            values = [torch.from_numpy(array[start:stop]).to(device) for array in arrays]
            if mask is not None:
                values.append(torch.from_numpy(mask[start:stop]).to(device))
            output = model(*values)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(predictions)


def _predict_safesense_batches_with_lag(
    model: object,
    skeleton: np.ndarray,
    imu: np.ndarray,
    mask: np.ndarray,
    device: "torch.device",
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    predictions: list[np.ndarray] = []
    lag_predictions: list[np.ndarray] = []
    has_lag = False
    with torch.inference_mode():
        for start in range(0, len(skeleton), batch_size):
            stop = min(start + batch_size, len(skeleton))
            skeleton_batch = torch.from_numpy(skeleton[start:stop]).to(device)
            imu_batch = torch.from_numpy(imu[start:stop]).to(device)
            mask_batch = torch.from_numpy(mask[start:stop]).to(device)
            output = model(skeleton_batch, imu_batch, mask_batch)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
            lag_logits = getattr(model, "lag_logits", None)
            if lag_logits is not None:
                has_lag = True
                lag_predictions.append(lag_logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(predictions), (np.concatenate(lag_predictions) if has_lag else None)


def run_safesense_development_robustness(
    prepared_dir: str | Path,
    safesense_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    corruption_seed: int = 20260814,
    model_loader: Callable[[str | Path, "torch.device"], object] | None = None,
) -> dict[str, object]:
    """Evaluate subject-7 robustness using only the prepared odd-subject partition."""
    import torch

    prepared = Path(prepared_dir)
    run_dir = Path(safesense_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((run_dir / "development.json").read_text(encoding="utf-8"))
    if recorded.get("workflow") != "development":
        raise RobustnessError("development robustness requires a development artifact")
    seeds = tuple(int(seed) for seed in recorded.get("seeds", ()))
    if not seeds:
        raise RobustnessError("development artifact has no seeds")
    variant = str(recorded.get("variant", "full"))

    fused = np.load(prepared / "skeleton_train_features.npy")
    labels = np.load(prepared / "train_labels.npy").astype(np.int64, copy=False)
    splits = safesense_splits(prepared)
    if np.any(splits.development_train >= len(fused)) or np.any(splits.development_valid >= len(fused)):
        raise RobustnessError("development robustness attempted to leave the prepared training partition")
    masks = base_temporal_masks(fused)
    scales = training_noise_scales(fused, masks, splits.development_train)
    scenarios = scenario_specs()
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    expected = {int(item["seed"]): float(item["best_validation_accuracy"]) for item in recorded["seed_results"]}

    stats_by_seed: dict[int, dict[str, np.ndarray]] = {}
    models: dict[int, object] = {}
    for seed in seeds:
        seed_dir = run_dir / f"seed-{seed}"
        with np.load(seed_dir / "stats.npz") as archive:
            stats_by_seed[seed] = {name: np.array(archive[name]) for name in ("skeleton_mean", "skeleton_std", "imu_mean", "imu_std")}
        loader = model_loader or (lambda path, dev: _load_safesense_model(path, dev, safesense_variant=variant))
        models[seed] = loader(seed_dir / "checkpoint.pt", selected_device)

    records: list[dict[str, object]] = []
    lag_records: list[dict[str, object]] = []
    for spec in scenarios:
        corrupted, corrupted_masks = apply_corruption(fused, spec, seed=corruption_seed, scales=scales)
        skeleton, imu, _ = _feature_arrays(corrupted)
        pair_mask = np.stack((corrupted_masks["skeleton"], corrupted_masks["imu"]), axis=-1).astype(np.float32)
        valid = splits.development_valid
        for seed in seeds:
            stats = stats_by_seed[seed]
            skeleton_input = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
            imu_input = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
            skeleton_input *= pair_mask[..., 0, None]
            imu_input *= pair_mask[..., 1, None]
            predictions, lag_predictions = _predict_safesense_batches_with_lag(
                models[seed], skeleton_input[valid], imu_input[valid], pair_mask[valid], selected_device
            )
            metrics = _metrics(predictions, labels[valid])
            if spec.name == "clean" and abs(float(metrics["accuracy"]) - expected[seed]) > 1e-6:
                raise RobustnessError(f"clean development accuracy mismatch for seed {seed}")
            records.append({"model": "SafeSense", "variant": variant, "seed": seed, "scenario": spec.name, "category": spec.category, **metrics})
            if lag_predictions is not None and spec.shift:
                corrective = corrective_imu_shift(spec)
                target_index = LAG_SHIFTS.index(corrective)
                predicted_shifts = np.asarray([LAG_SHIFTS[int(index)] for index in lag_predictions], dtype=np.int64)
                lag_records.append({
                    "seed": seed,
                    "scenario": spec.name,
                    "applied_shift": int(spec.shift),
                    "corrective_shift": corrective,
                    "exact_match_accuracy": float(np.mean(lag_predictions == target_index)),
                    "mean_absolute_shift_error": float(np.mean(np.abs(predicted_shifts - corrective))),
                })

    aggregate = _model_aggregates(records)["SafeSense"]
    lag_summary = None
    if lag_records:
        lag_summary = {
            "exact_match_accuracy_mean": float(np.mean([item["exact_match_accuracy"] for item in lag_records])),
            "mean_absolute_shift_error_mean": float(np.mean([item["mean_absolute_shift_error"] for item in lag_records])),
        }
    result: dict[str, object] = {
        "status": "completed",
        "workflow": "development_robustness",
        "variant": variant,
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "even_subject_arrays_loaded": False,
        "corruption_seed": corruption_seed,
        "noise_scales": scales,
        "scenario_specs": [spec.to_dict() for spec in scenarios],
        "metrics": records,
        "model_aggregate": aggregate,
        "lag_recovery": lag_summary,
        "lag_records": lag_records,
    }
    write_robustness_json(output / "result.json", result)
    return result


def _predict_conditional_batches(
    model: SafeSenseConditionalEnsemble,
    skeleton: np.ndarray,
    imu: np.ndarray,
    mask: np.ndarray,
    device: "torch.device",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    predictions: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(skeleton), batch_size):
            stop = min(start + batch_size, len(skeleton))
            logits = model(
                torch.from_numpy(skeleton[start:stop]).to(device),
                torch.from_numpy(imu[start:stop]).to(device),
                torch.from_numpy(mask[start:stop]).to(device),
            )
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            route = model.auxiliary_route
            if route is None:
                raise RobustnessError("conditional ensemble did not expose its auxiliary route")
            routes.append(route.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(routes)


def run_safesense_conditional_ensemble_development(
    prepared_dir: str | Path,
    safesense_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    corruption_seed: int = 20260814,
    model_loader: Callable[[str | Path, "torch.device"], object] | None = None,
) -> dict[str, object]:
    """Evaluate seed-11 primary plus fault-routed seed-22 on subject 7 only."""
    import torch

    prepared, run_dir, output = Path(prepared_dir), Path(safesense_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((run_dir / "development.json").read_text(encoding="utf-8"))
    if recorded.get("workflow") != "development" or recorded.get("variant") != "robust":
        raise RobustnessError("conditional ensemble requires a robust development artifact")
    control_seeds = tuple(int(seed) for seed in recorded.get("seeds", ()))
    if control_seeds != (11, 22, 33):
        raise RobustnessError("conditional ensemble control requires robust seeds 11, 22, 33")

    fused = np.load(prepared / "skeleton_train_features.npy")
    labels = np.load(prepared / "train_labels.npy").astype(np.int64, copy=False)
    splits = safesense_splits(prepared)
    if np.any(splits.development_train >= len(fused)) or np.any(splits.development_valid >= len(fused)):
        raise RobustnessError("conditional ensemble attempted to leave the prepared training partition")
    masks = base_temporal_masks(fused)
    scales = training_noise_scales(fused, masks, splits.development_train)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    shared_stats: dict[str, np.ndarray] | None = None
    models: dict[int, object] = {}
    checkpoint_paths: dict[int, Path] = {}
    for seed in control_seeds:
        seed_dir = run_dir / f"seed-{seed}"
        with np.load(seed_dir / "stats.npz") as archive:
            stats = {name: np.array(archive[name]) for name in ("skeleton_mean", "skeleton_std", "imu_mean", "imu_std")}
        if shared_stats is None:
            shared_stats = stats
        elif any(not np.array_equal(shared_stats[name], stats[name]) for name in shared_stats):
            raise RobustnessError("conditional checkpoints do not share identical train-only statistics")
        checkpoint = seed_dir / "checkpoint.pt"
        checkpoint_paths[seed] = checkpoint
        loader = model_loader or (lambda path, dev: _load_safesense_model(path, dev, safesense_variant="robust"))
        models[seed] = loader(checkpoint, selected_device)
    if shared_stats is None:
        raise RobustnessError("conditional ensemble did not load normalization statistics")
    if not all(isinstance(models[seed], torch.nn.Module) for seed in control_seeds):
        raise RobustnessError("conditional ensemble models must be torch modules")
    candidate = SafeSenseConditionalEnsemble(
        models[SAFESENSE_CONDITIONAL_SEEDS[0]],
        models[SAFESENSE_CONDITIONAL_SEEDS[1]],
        weights=SAFESENSE_CONDITIONAL_WEIGHTS,
        gate_threshold=SAFESENSE_CONDITIONAL_GATE_THRESHOLD,
    ).to(selected_device).eval()

    records: list[dict[str, object]] = []
    route_records: list[dict[str, object]] = []
    valid = splits.development_valid
    for spec in scenario_specs():
        corrupted, corrupted_masks = apply_corruption(fused, spec, seed=corruption_seed, scales=scales)
        skeleton, imu, _ = _feature_arrays(corrupted)
        pair_mask = np.stack((corrupted_masks["skeleton"], corrupted_masks["imu"]), axis=-1).astype(np.float32)
        skeleton_input = ((skeleton - shared_stats["skeleton_mean"]) / shared_stats["skeleton_std"]).astype(np.float32)
        imu_input = ((imu - shared_stats["imu_mean"]) / shared_stats["imu_std"]).astype(np.float32)
        skeleton_input *= pair_mask[..., 0, None]
        imu_input *= pair_mask[..., 1, None]
        for seed in control_seeds:
            predictions, _ = _predict_safesense_batches_with_lag(
                models[seed], skeleton_input[valid], imu_input[valid], pair_mask[valid], selected_device
            )
            records.append({"model": "SafeSense robust control", "seed": seed, "scenario": spec.name, "category": spec.category, **_metrics(predictions, labels[valid])})
        predictions, routes = _predict_conditional_batches(
            candidate, skeleton_input[valid], imu_input[valid], pair_mask[valid], selected_device
        )
        records.append({"model": "SafeSense conditional ensemble", "scenario": spec.name, "category": spec.category, **_metrics(predictions, labels[valid])})
        route_records.append({"scenario": spec.name, "category": spec.category, "auxiliary_trigger_rate": float(np.mean(routes))})

    aggregates = _model_aggregates(records)
    control = aggregates["SafeSense robust control"]
    ensemble = aggregates["SafeSense conditional ensemble"]
    category_accuracy = {key: float(ensemble["category_accuracy_mean"][key]) >= float(value) for key, value in control["category_accuracy_mean"].items()}
    category_macro = {key: float(ensemble["category_macro_f1_mean"][key]) >= float(value) for key, value in control["category_macro_f1_mean"].items()}
    scalar_checks = {
        "clean_accuracy": float(ensemble["clean_accuracy_mean"]) >= float(control["clean_accuracy_mean"]),
        "clean_macro_f1": float(ensemble["clean_macro_f1_mean"]) >= float(control["clean_macro_f1_mean"]),
        "equal_category_accuracy": float(ensemble["equal_category_corrupted_accuracy_mean"]) >= float(control["equal_category_corrupted_accuracy_mean"]),
        "equal_category_macro_f1": float(ensemble["equal_category_corrupted_macro_f1_mean"]) >= float(control["equal_category_corrupted_macro_f1_mean"]),
    }
    all_passed = all(scalar_checks.values()) and all(category_accuracy.values()) and all(category_macro.values())
    by_category: dict[str, list[float]] = {}
    for item in route_records:
        by_category.setdefault(str(item["category"]), []).append(float(item["auxiliary_trigger_rate"]))
    route_summary = {key: float(np.mean(value)) for key, value in by_category.items()}
    result: dict[str, object] = {
        "status": "completed",
        "workflow": "conditional_ensemble_development_robustness",
        "variant": "robust",
        "primary_seed": SAFESENSE_CONDITIONAL_SEEDS[0],
        "auxiliary_seed": SAFESENSE_CONDITIONAL_SEEDS[1],
        "conditional_weights": list(SAFESENSE_CONDITIONAL_WEIGHTS),
        "gate_threshold": SAFESENSE_CONDITIONAL_GATE_THRESHOLD,
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "even_subject_arrays_loaded": False,
        "corruption_seed": corruption_seed,
        "control_aggregate": control,
        "conditional_aggregate": ensemble,
        "auxiliary_trigger_rate_by_category": route_summary,
        "route_records": route_records,
        "no_regression_checks": {**scalar_checks, "category_accuracy": category_accuracy, "category_macro_f1": category_macro, "all_passed": all_passed},
        "single_model_parameter_count": parameter_count(models[11]),
        "conditional_parameter_count": parameter_count(candidate),
        "checkpoint_bytes_fp32_total": int(checkpoint_paths[11].stat().st_size + checkpoint_paths[22].stat().st_size),
        "metrics": records,
    }
    write_robustness_json(output / "result.json", result)
    return result


def _aggregate_no_regression_checks(
    candidate: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    category_accuracy = {
        key: float(candidate["category_accuracy_mean"][key]) >= float(value)
        for key, value in reference["category_accuracy_mean"].items()
    }
    category_macro = {
        key: float(candidate["category_macro_f1_mean"][key]) >= float(value)
        for key, value in reference["category_macro_f1_mean"].items()
    }
    scalar = {
        "clean_accuracy": float(candidate["clean_accuracy_mean"]) >= float(reference["clean_accuracy_mean"]),
        "clean_macro_f1": float(candidate["clean_macro_f1_mean"]) >= float(reference["clean_macro_f1_mean"]),
        "equal_category_accuracy": float(candidate["equal_category_corrupted_accuracy_mean"]) >= float(reference["equal_category_corrupted_accuracy_mean"]),
        "equal_category_macro_f1": float(candidate["equal_category_corrupted_macro_f1_mean"]) >= float(reference["equal_category_corrupted_macro_f1_mean"]),
    }
    return {
        **scalar,
        "category_accuracy": category_accuracy,
        "category_macro_f1": category_macro,
        "all_passed": all(scalar.values()) and all(category_accuracy.values()) and all(category_macro.values()),
    }


def _predict_early_exit_batches(
    model: SafeSenseEarlyExitEnsemble | object,
    skeleton: np.ndarray,
    imu: np.ndarray,
    mask: np.ndarray,
    device: "torch.device",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    predictions: list[np.ndarray] = []
    routes: list[np.ndarray] = []
    skeleton_choices: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(skeleton), batch_size):
            stop = min(start + batch_size, len(skeleton))
            logits = model(
                torch.from_numpy(skeleton[start:stop]).to(device),
                torch.from_numpy(imu[start:stop]).to(device),
                torch.from_numpy(mask[start:stop]).to(device),
            )
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            route = getattr(model, "auxiliary_route", None)
            use_skeleton = getattr(model, "use_skeleton", None)
            if route is None or use_skeleton is None:
                raise RobustnessError("early-exit candidate did not expose routing diagnostics")
            routes.append(route.cpu().numpy())
            skeleton_choices.append(use_skeleton.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(routes), np.concatenate(skeleton_choices)


def run_safesense_early_exit_development(
    prepared_dir: str | Path,
    safesense_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    corruption_seed: int = 20260814,
    model_loader: Callable[[str | Path, "torch.device"], object] | None = None,
    candidate_loader: Callable[[str | Path, str | Path, "torch.device"], object] | None = None,
    reference_aggregate: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate the sparse-precheck branch-expert early exit using odd subjects only."""
    import torch

    prepared, run_dir, output = Path(prepared_dir), Path(safesense_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((run_dir / "development.json").read_text(encoding="utf-8"))
    if recorded.get("workflow") != "development" or recorded.get("variant") != "robust":
        raise RobustnessError("early-exit evaluation requires a robust development artifact")
    control_seeds = tuple(int(seed) for seed in recorded.get("seeds", ()))
    if control_seeds != (11, 22, 33):
        raise RobustnessError("early-exit control requires robust seeds 11, 22, 33")

    fused = np.load(prepared / "skeleton_train_features.npy")
    labels = np.load(prepared / "train_labels.npy").astype(np.int64, copy=False)
    splits = safesense_splits(prepared)
    if np.any(splits.development_train >= len(fused)) or np.any(splits.development_valid >= len(fused)):
        raise RobustnessError("early-exit evaluation attempted to leave the prepared training partition")
    masks = base_temporal_masks(fused)
    scales = training_noise_scales(fused, masks, splits.development_train)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    shared_stats: dict[str, np.ndarray] | None = None
    models: dict[int, object] = {}
    checkpoints: dict[int, Path] = {}
    loader = model_loader or (lambda path, dev: _load_safesense_model(path, dev, safesense_variant="robust"))
    for seed in control_seeds:
        seed_dir = run_dir / f"seed-{seed}"
        with np.load(seed_dir / "stats.npz") as archive:
            stats = {name: np.array(archive[name]) for name in ("skeleton_mean", "skeleton_std", "imu_mean", "imu_std")}
        if shared_stats is None:
            shared_stats = stats
        elif any(not np.array_equal(shared_stats[name], stats[name]) for name in shared_stats):
            raise RobustnessError("early-exit checkpoints do not share identical train-only statistics")
        checkpoint = seed_dir / "checkpoint.pt"
        checkpoints[seed] = checkpoint
        models[seed] = loader(checkpoint, selected_device)
    if shared_stats is None:
        raise RobustnessError("early-exit evaluation did not load normalization statistics")

    if candidate_loader is not None:
        candidate = candidate_loader(checkpoints[SAFESENSE_EARLY_EXIT_SEEDS[0]], checkpoints[SAFESENSE_EARLY_EXIT_SEEDS[1]], selected_device)
        branch_checkpoint_bytes = None
        auxiliary_parameter_count = None
    else:
        if not isinstance(models[SAFESENSE_EARLY_EXIT_SEEDS[0]], SafeSense):
            raise RobustnessError("default early-exit construction requires a SafeSense primary model")
        auxiliary = SafeSenseBranchExpert.from_robust_checkpoint(checkpoints[SAFESENSE_EARLY_EXIT_SEEDS[1]]).to(selected_device).eval()
        candidate = SafeSenseEarlyExitEnsemble(
            models[SAFESENSE_EARLY_EXIT_SEEDS[0]],
            auxiliary,
            primary_weight=SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT,
            roughness_threshold=SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD,
        ).to(selected_device).eval()
        branch_checkpoint = output / "seed-22-branch-expert.pt"
        torch.save({name: value.detach().cpu() for name, value in auxiliary.state_dict().items()}, branch_checkpoint)
        branch_checkpoint_bytes = branch_checkpoint.stat().st_size
        auxiliary_parameter_count = parameter_count(auxiliary)

    records: list[dict[str, object]] = []
    route_records: list[dict[str, object]] = []
    valid = splits.development_valid
    for spec in scenario_specs():
        corrupted, corrupted_masks = apply_corruption(fused, spec, seed=corruption_seed, scales=scales)
        skeleton, imu, _ = _feature_arrays(corrupted)
        pair_mask = np.stack((corrupted_masks["skeleton"], corrupted_masks["imu"]), axis=-1).astype(np.float32)
        skeleton_input = ((skeleton - shared_stats["skeleton_mean"]) / shared_stats["skeleton_std"]).astype(np.float32)
        imu_input = ((imu - shared_stats["imu_mean"]) / shared_stats["imu_std"]).astype(np.float32)
        skeleton_input *= pair_mask[..., 0, None]
        imu_input *= pair_mask[..., 1, None]
        for seed in control_seeds:
            predictions, _ = _predict_safesense_batches_with_lag(
                models[seed], skeleton_input[valid], imu_input[valid], pair_mask[valid], selected_device
            )
            records.append({
                "model": "SafeSense robust control", "seed": seed, "scenario": spec.name,
                "category": spec.category, **_metrics(predictions, labels[valid])
            })
        predictions, routes, use_skeleton = _predict_early_exit_batches(
            candidate, skeleton_input[valid], imu_input[valid], pair_mask[valid], selected_device
        )
        records.append({
            "model": "SafeSense early exit", "scenario": spec.name, "category": spec.category,
            **_metrics(predictions, labels[valid])
        })
        route_records.append({
            "scenario": spec.name, "category": spec.category,
            "auxiliary_trigger_rate": float(np.mean(routes)),
            "skeleton_choice_rate": float(np.mean(use_skeleton[routes])) if bool(np.any(routes)) else None,
        })

    aggregates = _model_aggregates(records)
    control = aggregates["SafeSense robust control"]
    early_exit = aggregates["SafeSense early exit"]
    control_checks = _aggregate_no_regression_checks(early_exit, control)
    reference_checks = _aggregate_no_regression_checks(early_exit, reference_aggregate) if reference_aggregate is not None else None
    by_category: dict[str, list[float]] = {}
    for item in route_records:
        by_category.setdefault(str(item["category"]), []).append(float(item["auxiliary_trigger_rate"]))
    route_summary = {key: float(np.mean(value)) for key, value in by_category.items()}
    deployable_checkpoint_bytes = None
    if branch_checkpoint_bytes is not None:
        deployable_checkpoint_bytes = int(checkpoints[SAFESENSE_EARLY_EXIT_SEEDS[0]].stat().st_size + branch_checkpoint_bytes)
    result: dict[str, object] = {
        "status": "completed",
        "workflow": "early_exit_branch_expert_development_robustness",
        "variant": "robust",
        "primary_seed": SAFESENSE_EARLY_EXIT_SEEDS[0],
        "auxiliary_seed": SAFESENSE_EARLY_EXIT_SEEDS[1],
        "primary_probability_weight": SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT,
        "roughness_threshold": SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD,
        "roughness_skeleton_channel": SAFESENSE_EARLY_EXIT_SKELETON_CHANNEL,
        "roughness_imu_channel": SAFESENSE_EARLY_EXIT_IMU_CHANNEL,
        "roughness_threshold_source": "subjects 1,3,5 synthetic clean-vs-0.1-noise midpoint using the frozen seed-11 train-only normalization stats; channel pair (skeleton 9, IMU 0) ranked first by source-only separation margin and then verified on subject 7",
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "even_subject_arrays_loaded": False,
        "corruption_seed": corruption_seed,
        "control_aggregate": control,
        "early_exit_aggregate": early_exit,
        "auxiliary_trigger_rate_by_category": route_summary,
        "route_records": route_records,
        "no_regression_vs_robust_control": control_checks,
        "no_regression_vs_retained_conditional": reference_checks,
        "single_model_parameter_count": parameter_count(models[11]) if isinstance(models[11], torch.nn.Module) else None,
        "auxiliary_parameter_count": auxiliary_parameter_count,
        "early_exit_parameter_count": parameter_count(candidate) if isinstance(candidate, torch.nn.Module) else None,
        "deployable_checkpoint_bytes_fp32": deployable_checkpoint_bytes,
        "metrics": records,
    }
    write_robustness_json(output / "result.json", result)
    return result


def run_safesense_early_exit_posthoc(
    prepared_dir: str | Path,
    safesense_dir: str | Path,
    fusion_repo_dir: str | Path,
    fusion_results_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    corruption_seed: int = 20260814,
) -> dict[str, object]:
    """Evaluate the frozen development-selected early-exit wrapper on even subjects once."""
    import torch

    prepared = Path(prepared_dir)
    run_dir = Path(safesense_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if recorded.get("workflow") != "final" or recorded.get("variant") != "robust":
        raise RobustnessError("early-exit post-hoc evaluation requires the frozen robust final artifact")
    if int(recorded.get("temporal_epoch", 0)) != 33 or int(recorded.get("adapter_epoch", 0)) != 27:
        raise RobustnessError("early-exit post-hoc evaluation requires frozen temporal-33/adapter-27 checkpoints")
    if tuple(int(seed) for seed in recorded.get("seeds", ())) != (11, 22, 33):
        raise RobustnessError("early-exit post-hoc evaluation requires final seeds 11, 22, and 33")

    train = np.load(prepared / "skeleton_train_features.npy")
    held_out = np.load(prepared / "skeleton_val_features.npy")
    fused = np.concatenate((train, held_out), axis=0)
    labels = np.concatenate((np.load(prepared / "train_labels.npy"), np.load(prepared / "val_labels.npy")))
    masks = base_temporal_masks(fused)
    splits = safesense_splits(prepared)
    full_training = np.concatenate((splits.development_train, splits.development_valid))
    scales = training_noise_scales(fused, masks, full_training)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    primary_seed, auxiliary_seed = SAFESENSE_EARLY_EXIT_SEEDS
    primary_dir = run_dir / f"seed-{primary_seed}"
    auxiliary_dir = run_dir / f"seed-{auxiliary_seed}"
    with np.load(primary_dir / "stats.npz") as archive:
        stats = {name: np.array(archive[name]) for name in ("skeleton_mean", "skeleton_std", "imu_mean", "imu_std")}
    with np.load(auxiliary_dir / "stats.npz") as archive:
        auxiliary_stats = {name: np.array(archive[name]) for name in ("skeleton_mean", "skeleton_std", "imu_mean", "imu_std")}
    if any(not np.array_equal(stats[name], auxiliary_stats[name]) for name in stats):
        raise RobustnessError("final seed-11 and seed-22 checkpoints do not share identical train-only statistics")

    primary_checkpoint = primary_dir / "checkpoint.pt"
    auxiliary_checkpoint = auxiliary_dir / "checkpoint.pt"
    candidate = SafeSenseEarlyExitEnsemble(
        _load_safesense_model(primary_checkpoint, selected_device, safesense_variant="robust"),
        SafeSenseBranchExpert.from_robust_checkpoint(auxiliary_checkpoint).to(selected_device).eval(),
        primary_weight=SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT,
        roughness_threshold=SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD,
    ).to(selected_device).eval()
    branch_checkpoint = output / "seed-22-branch-expert.pt"
    torch.save({name: value.detach().cpu() for name, value in candidate.auxiliary.state_dict().items()}, branch_checkpoint)

    fusion_result_paths = [
        Path(fusion_results_dir) / f"seed-{seed}" / f"fusion_gcn_reproduction_seed_{seed}.json"
        for seed in (11, 22, 33)
    ]
    fusion_checkpoints = select_fusion_final_checkpoints(fusion_result_paths)
    fusion_models = {
        seed: load_fusion_final_model(fusion_repo_dir, fusion_checkpoints[seed], selected_device)
        for seed in (11, 22, 33)
    }

    scenarios = scenario_specs()
    records: list[dict[str, object]] = []
    route_records: list[dict[str, object]] = []
    candidate_predictions = np.empty((len(scenarios), len(splits.official_test)), dtype=np.int64)
    fusion_predictions = np.empty((3, len(scenarios), len(splits.official_test)), dtype=np.int64)
    test_slice = splits.official_test
    for scenario_index, spec in enumerate(scenarios):
        corrupted, corrupted_masks = apply_corruption(fused, spec, seed=corruption_seed, scales=scales)
        skeleton, imu, _ = _feature_arrays(corrupted)
        pair_mask = np.stack((corrupted_masks["skeleton"], corrupted_masks["imu"]), axis=-1).astype(np.float32)
        skeleton_input = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
        imu_input = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
        skeleton_input *= pair_mask[..., 0, None]
        imu_input *= pair_mask[..., 1, None]
        predictions, routes, use_skeleton = _predict_early_exit_batches(
            candidate, skeleton_input[test_slice], imu_input[test_slice], pair_mask[test_slice], selected_device
        )
        candidate_predictions[scenario_index] = predictions
        records.append({
            "model": "SafeSense early exit", "scenario": spec.name, "category": spec.category,
            **_metrics(predictions, labels[test_slice]),
        })
        route_records.append({
            "scenario": spec.name,
            "category": spec.category,
            "auxiliary_trigger_rate": float(np.mean(routes)),
            "skeleton_choice_rate": float(np.mean(use_skeleton[routes])) if bool(np.any(routes)) else None,
        })
        for seed_index, seed in enumerate((11, 22, 33)):
            predictions = _predict_batches(fusion_models[seed], (corrupted[test_slice],), selected_device)
            fusion_predictions[seed_index, scenario_index] = predictions
            records.append({
                "model": "Fusion-GCN", "seed": seed, "scenario": spec.name, "category": spec.category,
                **_metrics(predictions, labels[test_slice]),
            })

    aggregates = _model_aggregates(records)
    route_by_category: dict[str, list[float]] = {}
    for item in route_records:
        route_by_category.setdefault(str(item["category"]), []).append(float(item["auxiliary_trigger_rate"]))
    route_summary = {key: float(np.mean(values)) for key, values in route_by_category.items()}
    prediction_artifact = output / "predictions.npz"
    np.savez(
        prediction_artifact,
        labels=labels[test_slice],
        safesense_predictions=candidate_predictions,
        fusion_predictions=fusion_predictions,
        fusion_seeds=np.asarray((11, 22, 33)),
        scenarios=np.asarray([spec.name for spec in scenarios]),
    )
    result: dict[str, object] = {
        "status": "completed",
        "workflow": "early_exit_branch_expert_posthoc_even_subject_robustness",
        "selection_frozen_before_even_subject_evaluation": True,
        "posthoc_test_subjects": [2, 4, 6, 8],
        "training_subjects": [1, 3, 5, 7],
        "primary_seed": primary_seed,
        "auxiliary_seed": auxiliary_seed,
        "primary_probability_weight": SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT,
        "roughness_threshold": SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD,
        "roughness_skeleton_channel": SAFESENSE_EARLY_EXIT_SKELETON_CHANNEL,
        "roughness_imu_channel": SAFESENSE_EARLY_EXIT_IMU_CHANNEL,
        "corruption_seed": corruption_seed,
        "noise_scale_training_subjects": [1, 3, 5, 7],
        "scenario_specs": [spec.to_dict() for spec in scenarios],
        "model_aggregates": aggregates,
        "auxiliary_trigger_rate_by_category": route_summary,
        "route_records": route_records,
        "early_exit_parameter_count": parameter_count(candidate),
        "auxiliary_parameter_count": parameter_count(candidate.auxiliary),
        "deployable_checkpoint_bytes_fp32": int(primary_checkpoint.stat().st_size + branch_checkpoint.stat().st_size),
        "fusion_checkpoint_policy": "final_weights_only_no_test_selected_checkpoint",
        "metrics": records,
        "prediction_artifact": str(prediction_artifact),
    }
    write_robustness_json(output / "result.json", result)
    return result


def _model_aggregates(records: Sequence[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for model_name in sorted({str(record["model"]) for record in records}):
        model_records = [record for record in records if record["model"] == model_name]
        clean = [float(record["accuracy"]) for record in model_records if record["scenario"] == "clean"]
        clean_macro = [float(record["macro_f1"]) for record in model_records if record["scenario"] == "clean"]
        categories: dict[str, list[float]] = {}
        macro_categories: dict[str, list[float]] = {}
        for record in model_records:
            if record["scenario"] != "clean":
                categories.setdefault(str(record["category"]), []).append(float(record["accuracy"]))
                macro_categories.setdefault(str(record["category"]), []).append(float(record["macro_f1"]))
        category_means = {key: float(np.mean(value)) for key, value in categories.items()}
        macro_category_means = {key: float(np.mean(value)) for key, value in macro_categories.items()}
        result[model_name] = {
            "clean_accuracy_mean": float(np.mean(clean)),
            "clean_macro_f1_mean": float(np.mean(clean_macro)),
            "category_accuracy_mean": category_means,
            "category_macro_f1_mean": macro_category_means,
            "equal_category_corrupted_accuracy_mean": float(np.mean(list(category_means.values()))),
            "equal_category_corrupted_macro_f1_mean": float(np.mean(list(macro_category_means.values()))),
        }
    return result


def _recorded_final_clean_accuracies(safesense_dir: str | Path, variant: str) -> dict[int, float] | None:
    """Read final-run clean accuracies so robustness validates the exact frozen run."""
    result_path = Path(safesense_dir) / "result.json"
    if not result_path.is_file():
        return None
    recorded = json.loads(result_path.read_text(encoding="utf-8"))
    if recorded.get("workflow") != "final":
        raise RobustnessError(f"expected a final SafeSense result at {result_path}")
    if str(recorded.get("variant", "full")) != variant:
        raise RobustnessError(
            f"SafeSense result variant {recorded.get('variant')!r} does not match requested {variant!r}"
        )
    accuracies = {int(item["seed"]): float(item["accuracy"]) for item in recorded.get("seed_results", ())}
    if set(accuracies) != {11, 22, 33}:
        raise RobustnessError("SafeSense final result must contain clean accuracies for seeds 11, 22, and 33")
    return accuracies


def run_robustness_evaluation(
    prepared_dir: str | Path,
    safesense_dir: str | Path,
    fusion_repo_dir: str | Path,
    fusion_results_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    corruption_seed: int = 20260814,
    expected_clean_accuracy: dict[int, float] | None = None,
    expected_fusion_accuracy: dict[int, float] | None = None,
    safesense_variant: str = "full",
    safesense_model_loader: Callable[[str | Path, "torch.device"], object] | None = None,
    fusion_model_loader: Callable[[str | Path, str | Path, "torch.device"], object] | None = None,
) -> dict[str, object]:
    """Evaluate both frozen models on one deterministic corrupted-array suite."""
    import torch

    prepared = Path(prepared_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = np.load(prepared / "skeleton_train_features.npy")
    held_out = np.load(prepared / "skeleton_val_features.npy")
    fused = np.concatenate((train, held_out), axis=0)
    masks = base_temporal_masks(fused)
    splits = safesense_splits(prepared)
    full_training = np.concatenate((splits.development_train, splits.development_valid))
    scales = training_noise_scales(fused, masks, full_training)
    scenarios = scenario_specs()
    if expected_clean_accuracy is None:
        expected_clean_accuracy = _recorded_final_clean_accuracies(safesense_dir, safesense_variant)
        if expected_clean_accuracy is None:
            expected_clean_accuracy = (
                {11: 0.9395348837209302, 22: 0.9372093023255814, 33: 0.9255813953488372}
                if safesense_variant == "classprior"
                else {11: 0.9302325581395349, 22: 0.9046511627906977, 33: 0.9162790697674419}
            )
    expected_fusion_accuracy = expected_fusion_accuracy or {11: 0.9279, 22: 0.9442, 33: 0.9326}
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    fusion_result_paths = [Path(fusion_results_dir) / f"seed-{seed}" / f"fusion_gcn_reproduction_seed_{seed}.json" for seed in (11, 22, 33)]
    fusion_checkpoints = select_fusion_final_checkpoints(fusion_result_paths)
    result: dict[str, object] = {
        "status": "completed",
        "corruption_seed": corruption_seed,
        "sign_convention": "positive shift moves a frame toward a later index; zero padding, never circular",
        "noise_scales": scales,
        "noise_scale_training_count": int(len(full_training)),
        "noise_scale_training_subjects": [1, 3, 5, 7],
        "scenario_specs": [spec.to_dict() for spec in scenarios],
        "corruption_order": ["shift", "frame_loss", "noise"],
        "fusion_checkpoint_policy": "final_weights_only_no_test_selected_checkpoint",
        "prediction_axis_order": ["model", "seed", "scenario", "sample"],
        "models": ["SafeSense", "Fusion-GCN"],
        "safesense_variant": safesense_variant,
        "safesense": [],
        "fusion_gcn": {"repo": str(fusion_repo_dir), "results": str(fusion_results_dir), "macro_f1_source": "newly computed by this evaluator"},
    }
    all_predictions = np.empty((2, 3, len(scenarios), len(splits.official_test)), dtype=np.int64)
    labels = np.concatenate((np.load(prepared / "train_labels.npy"), np.load(prepared / "val_labels.npy")))
    seeds = (11, 22, 33)
    safe_stats: dict[int, dict[str, np.ndarray]] = {}
    safe_models: dict[int, object] = {}
    fusion_models: dict[int, object] = {}
    for seed in seeds:
        seed_dir = Path(safesense_dir) / f"seed-{seed}"
        stats_file = seed_dir / "stats.npz"
        checkpoint = seed_dir / "checkpoint.pt"
        if not stats_file.is_file() or not checkpoint.is_file():
            raise RobustnessError(f"missing SafeSense final checkpoint artifacts for seed {seed}: {seed_dir}")
        with np.load(stats_file) as archive:
            safe_stats[seed] = {name: np.array(archive[name]) for name in ("skeleton_mean", "skeleton_std", "imu_mean", "imu_std")}
        loader = safesense_model_loader or (lambda path, dev: _load_safesense_model(path, dev, safesense_variant=safesense_variant))
        safe_models[seed] = loader(checkpoint, selected_device)
        fusion_loader = fusion_model_loader or load_fusion_final_model
        fusion_models[seed] = fusion_loader(fusion_repo_dir, fusion_checkpoints[seed], selected_device)
    records: list[dict[str, object]] = []
    for scenario_index, spec in enumerate(scenarios):
        corrupted, corrupted_masks = apply_corruption(fused, spec, seed=corruption_seed, scales=scales)
        skeleton, imu, _ = _feature_arrays(corrupted)
        pair_mask = np.stack((corrupted_masks["skeleton"], corrupted_masks["imu"]), axis=-1).astype(np.float32)
        for seed_index, seed in enumerate(seeds):
            stats = safe_stats[seed]
            skeleton_input = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
            imu_input = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
            skeleton_input *= pair_mask[..., 0, None]
            imu_input *= pair_mask[..., 1, None]
            test_slice = splits.official_test
            safe_predictions = _predict_batches(safe_models[seed], (skeleton_input[test_slice], imu_input[test_slice]), selected_device, mask=pair_mask[test_slice])
            fusion_predictions = _predict_batches(fusion_models[seed], (corrupted[test_slice],), selected_device)
            all_predictions[0, seed_index, scenario_index] = safe_predictions
            all_predictions[1, seed_index, scenario_index] = fusion_predictions
            for model_name, predictions in (("SafeSense", safe_predictions), ("Fusion-GCN", fusion_predictions)):
                metrics = _metrics(predictions, labels[test_slice])
                if spec.name == "clean":
                    expected = expected_clean_accuracy[seed] if model_name == "SafeSense" else expected_fusion_accuracy[seed]
                    tolerance = 1e-6 if model_name == "SafeSense" else 0.00005
                    if abs(float(metrics["accuracy"]) - expected) > tolerance:
                        raise RobustnessError(f"clean {model_name} accuracy mismatch for seed {seed}")
                records.append({"model": model_name, "seed": seed, "scenario": spec.name, "category": spec.category, **metrics})
    result["metrics"] = records
    result["model_aggregates"] = _model_aggregates(records)
    prediction_artifact = output / "predictions.npz"
    np.savez(prediction_artifact, labels=labels[splits.official_test], predictions=all_predictions, model_names=np.asarray(result["models"]), seeds=np.asarray(seeds), scenarios=np.asarray([spec.name for spec in scenarios]))
    result["prediction_artifact"] = str(prediction_artifact)
    write_robustness_json(output / "result.json", result)
    return result
