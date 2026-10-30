"""Train standard comparison baselines and evaluate the frozen 25-scenario protocol.

This script is reporting-only.  It never changes or selects the SafeSense model.
All baseline hyperparameters are fixed before official-even evaluation:

* train on all odd UTD-MHAD subjects (1,3,5,7);
* clean supervised cross-entropy only;
* 60 fixed epochs, AdamW, lr=1e-3, weight_decay=1e-4, batch=32;
* seeds 11,22,33;
* equal SWA of completed epochs 45..60;
* train-only normalization statistics;
* one frozen evaluation on even subjects (2,4,6,8) and the exact preserved
  25-scenario corruption suite (seed 20260814).

The baselines deliberately represent common fusion strategies rather than
paper-specific reproductions:
  - skeleton-only temporal CNN;
  - IMU-only temporal CNN;
  - frame-level early-fusion temporal CNN;
  - decision-level late-fusion dual temporal CNN.

They complement, rather than replace, the separately reproduced Fusion-GCN
external baseline.
"""
from __future__ import annotations

import argparse
import base64
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from safesense.robustness import (
    apply_corruption,
    base_temporal_masks,
    scenario_specs,
    training_noise_scales,
)
from safesense.safesense import (
    IMU_INPUTS,
    NUM_CLASSES,
    SKELETON_INPUTS,
    SafeSenseTCNBranch,
    _feature_arrays,
    _metrics,
    parameter_count,
    safesense_splits,
    set_safesense_seed,
)

SEEDS = (11, 22, 33)
EPOCHS = 60
SWA_START = 45
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CORRUPTION_SEED = 20260814


def _masked_pool(value: Tensor, mask: Tensor) -> Tensor:
    """Masked mean+max over [B,C,T], returning [B,2C]."""
    valid = mask.float().clamp(0, 1)
    channels = value.transpose(1, 2)  # [B,T,C]
    count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (channels * valid.unsqueeze(-1)).sum(dim=1) / count
    fill = torch.finfo(channels.dtype).min
    maximum = channels.masked_fill(~valid.bool().unsqueeze(-1), fill).max(dim=1).values
    has_any = valid.any(dim=1, keepdim=True)
    maximum = torch.where(has_any, maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class SkeletonTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = SafeSenseTCNBranch(SKELETON_INPUTS, 96)
        self.head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, NUM_CLASSES))

    def forward(self, skeleton: Tensor, imu: Tensor, mask: Tensor) -> Tensor:  # noqa: ARG002
        sk_mask = mask[..., 0] if mask.ndim == 3 else mask
        return self.head(_masked_pool(self.branch(skeleton.transpose(1, 2)), sk_mask))


class ImuTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = SafeSenseTCNBranch(IMU_INPUTS, 96)
        self.head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, NUM_CLASSES))

    def forward(self, skeleton: Tensor, imu: Tensor, mask: Tensor) -> Tensor:  # noqa: ARG002
        imu_mask = mask[..., 1] if mask.ndim == 3 else mask
        return self.head(_masked_pool(self.branch(imu.transpose(1, 2)), imu_mask))


class EarlyFusionTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = SafeSenseTCNBranch(SKELETON_INPUTS + IMU_INPUTS, 96)
        self.head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, NUM_CLASSES))

    def forward(self, skeleton: Tensor, imu: Tensor, mask: Tensor) -> Tensor:
        if mask.ndim == 2:
            pool_mask = mask
        else:
            pool_mask = mask.any(dim=-1).float()
        fused = torch.cat((skeleton, imu), dim=-1)
        return self.head(_masked_pool(self.branch(fused.transpose(1, 2)), pool_mask))


class LateFusionTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skeleton_branch = SafeSenseTCNBranch(SKELETON_INPUTS, 96)
        self.imu_branch = SafeSenseTCNBranch(IMU_INPUTS, 96)
        self.skeleton_head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, NUM_CLASSES))
        self.imu_head = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, NUM_CLASSES))

    def forward(self, skeleton: Tensor, imu: Tensor, mask: Tensor) -> Tensor:
        if mask.ndim == 2:
            sk_mask = imu_mask = mask
        else:
            sk_mask, imu_mask = mask[..., 0], mask[..., 1]
        sk_logits = self.skeleton_head(_masked_pool(self.skeleton_branch(skeleton.transpose(1, 2)), sk_mask))
        imu_logits = self.imu_head(_masked_pool(self.imu_branch(imu.transpose(1, 2)), imu_mask))
        available = torch.stack((sk_mask.any(dim=1), imu_mask.any(dim=1)), dim=1).float()
        denominator = available.sum(dim=1, keepdim=True)
        weights = available / denominator.clamp_min(1.0)
        no_modalities = denominator.squeeze(1) <= 0
        if bool(no_modalities.any()):
            weights = weights.clone()
            weights[no_modalities] = 0.5
        return sk_logits * weights[:, 0:1] + imu_logits * weights[:, 1:2]


BASELINES = {
    "Skeleton-TCN": SkeletonTCN,
    "IMU-TCN": ImuTCN,
    "EarlyFusion-TCN": EarlyFusionTCN,
    "LateFusion-TCN": LateFusionTCN,
}


@dataclass(frozen=True)
class Arrays:
    skeleton: np.ndarray
    imu: np.ndarray
    pair_mask: np.ndarray
    labels: np.ndarray


def _stats(skeleton: np.ndarray, imu: np.ndarray, mask: np.ndarray, train: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, values, channel in (("skeleton", skeleton, 0), ("imu", imu, 1)):
        valid = mask[train, :, channel].astype(bool)
        flat = values[train][valid]
        mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = std
    return result


def _normalize(skeleton: np.ndarray, imu: np.ndarray, pair_mask: np.ndarray, stats: dict[str, np.ndarray]) -> Arrays:
    sk = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
    im = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    sk *= pair_mask[..., 0, None]
    im *= pair_mask[..., 1, None]
    return Arrays(sk, im, pair_mask.astype(np.float32), np.empty(0, dtype=np.int64))


def _swa_update(average: dict[str, Tensor] | None, model: nn.Module, count: int) -> dict[str, Tensor]:
    current = model.state_dict()
    if average is None:
        return {name: value.detach().cpu().clone() for name, value in current.items()}
    for name, value in current.items():
        value_cpu = value.detach().cpu()
        if torch.is_floating_point(value_cpu):
            average[name].add_((value_cpu - average[name]) / float(count))
        else:
            average[name].copy_(value_cpu)
    return average


def _batch_indices(size: int, seed: int, epoch: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(size)
    for start in range(0, size, BATCH_SIZE):
        yield order[start : start + BATCH_SIZE]


def train_model(
    model_factory: type[nn.Module],
    arrays: Arrays,
    train_indices: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_safesense_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sk = torch.from_numpy(arrays.skeleton[train_indices])
    im = torch.from_numpy(arrays.imu[train_indices])
    mask = torch.from_numpy(arrays.pair_mask[train_indices])
    labels = torch.from_numpy(arrays.labels[train_indices].astype(np.int64))
    swa: dict[str, Tensor] | None = None
    swa_count = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for local in _batch_indices(len(train_indices), seed, epoch):
            bsk = sk[local].to(device)
            bim = im[local].to(device)
            bmask = mask[local].to(device)
            by = labels[local].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bsk, bim, bmask)
            loss = nn.functional.cross_entropy(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch >= SWA_START:
            swa_count += 1
            swa = _swa_update(swa, model, swa_count)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"seed={seed} epoch={epoch}/{EPOCHS} loss={np.mean(losses):.6f} swa_states={swa_count}", flush=True)
    if swa is None or swa_count != EPOCHS - SWA_START + 1:
        raise RuntimeError("SWA state count mismatch")
    model.load_state_dict(swa)
    model.eval()
    return model


@torch.inference_mode()
def predict(model: nn.Module, arrays: Arrays, indices: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        rows = indices[start : start + batch_size]
        logits = model(
            torch.from_numpy(arrays.skeleton[rows]).to(device),
            torch.from_numpy(arrays.imu[rows]).to(device),
            torch.from_numpy(arrays.pair_mask[rows]).to(device),
        )
        outputs.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def category_aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    category_accuracy: dict[str, float] = {}
    category_f1: dict[str, float] = {}
    for category in ("shift", "frame_loss", "noise", "missing", "composite"):
        rows = [row for row in records if row["category"] == category]
        category_accuracy[category] = float(np.mean([float(row["accuracy"]) for row in rows]))
        category_f1[category] = float(np.mean([float(row["macro_f1"]) for row in rows]))
    return {
        "category_accuracy_mean": category_accuracy,
        "category_macro_f1_mean": category_f1,
        "equal_category_corrupted_accuracy_mean": float(np.mean(list(category_accuracy.values()))),
        "equal_category_corrupted_macro_f1_mean": float(np.mean(list(category_f1.values()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default="cache/fusion-gcn")
    parser.add_argument("--output", default="research/matched-standard-baseline-results.json")
    parser.add_argument("--predictions", default="research/matched-standard-baseline-predictions.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    prepared = Path(args.prepared)
    train_raw = np.load(prepared / "skeleton_train_features.npy")
    test_raw = np.load(prepared / "skeleton_val_features.npy")
    fused = np.concatenate((train_raw, test_raw), axis=0)
    labels = np.concatenate((np.load(prepared / "train_labels.npy"), np.load(prepared / "val_labels.npy"))).astype(np.int64)
    splits = safesense_splits(prepared)
    full_train = np.concatenate((splits.development_train, splits.development_valid))
    if len(full_train) != 431 or len(splits.official_test) != 430:
        raise RuntimeError("frozen UTD split count mismatch")

    base_masks = base_temporal_masks(fused)
    clean_pair = np.stack((base_masks["skeleton"], base_masks["imu"]), axis=-1).astype(np.float32)
    skeleton, imu, _ = _feature_arrays(fused)
    stats = _stats(skeleton, imu, clean_pair, full_train)
    clean = _normalize(skeleton, imu, clean_pair, stats)
    clean = Arrays(clean.skeleton, clean.imu, clean.pair_mask, labels)
    scales = training_noise_scales(fused, base_masks, full_train)
    specs = scenario_specs()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    result: dict[str, object] = {
        "date": "2026-08-16",
        "status": "completed_reporting_baselines",
        "purpose": "standard matched comparison baselines; not used to tune or select SafeSense",
        "protocol": {
            "dataset": "UTD-MHAD",
            "train_subjects": [1, 3, 5, 7],
            "test_subjects": [2, 4, 6, 8],
            "train_samples": 431,
            "test_samples": 430,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "swa_start_epoch": SWA_START,
            "swa_state_count": EPOCHS - SWA_START + 1,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "training_loss": "clean cross-entropy only",
            "selection": "none; fixed recipe and fixed epoch/SWA window before official-even evaluation",
            "normalization": "all-odd train-only statistics",
            "corruption_seed": CORRUPTION_SEED,
            "scenario_count": len(specs),
            "scenario_source": "src/safesense/robustness.py::scenario_specs",
            "noise_scales": scales,
        },
        "baseline_definitions": {
            "Skeleton-TCN": "177-D skeleton engineered features -> 96-wide residual depthwise TCN -> masked mean+max -> linear classifier",
            "IMU-TCN": "12-D inertial engineered features -> 96-wide residual depthwise TCN -> masked mean+max -> linear classifier",
            "EarlyFusion-TCN": "frame-level concatenation of normalized skeleton+IMU features -> 96-wide residual depthwise TCN -> masked mean+max -> classifier",
            "LateFusion-TCN": "independent 96-wide skeleton and IMU TCN classifiers -> equal decision fusion over available modalities",
        },
        "models": {},
    }
    all_predictions = np.empty((len(BASELINES), len(SEEDS), len(specs), len(splits.official_test)), dtype=np.uint8)

    for model_index, (name, factory) in enumerate(BASELINES.items()):
        seed_records: list[dict[str, object]] = []
        print(f"=== {name} params={parameter_count(factory())} ===", flush=True)
        for seed_index, seed in enumerate(SEEDS):
            print(f"training {name} seed={seed}", flush=True)
            model = train_model(factory, clean, full_train, seed=seed, device=device)
            records: list[dict[str, object]] = []
            for scenario_index, spec in enumerate(specs):
                corrupted, corrupted_masks = apply_corruption(fused, spec, seed=CORRUPTION_SEED, scales=scales)
                sk, im, _ = _feature_arrays(corrupted)
                pair = np.stack((corrupted_masks["skeleton"], corrupted_masks["imu"]), axis=-1).astype(np.float32)
                normalized = _normalize(sk, im, pair, stats)
                scenario_arrays = Arrays(normalized.skeleton, normalized.imu, normalized.pair_mask, labels)
                preds = predict(model, scenario_arrays, splits.official_test, device)
                all_predictions[model_index, seed_index, scenario_index] = preds.astype(np.uint8)
                metrics = _metrics(preds, labels[splits.official_test])
                records.append({"scenario": spec.name, "category": spec.category, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
            clean_record = records[0]
            aggregate = category_aggregate(records)
            seed_records.append({
                "seed": seed,
                "clean_accuracy": clean_record["accuracy"],
                "clean_macro_f1": clean_record["macro_f1"],
                **aggregate,
                "scenarios": records,
            })
            print(
                f"evaluated {name} seed={seed} clean={float(clean_record['accuracy']):.6f} "
                f"robust={float(aggregate['equal_category_corrupted_accuracy_mean']):.6f}",
                flush=True,
            )
        result["models"][name] = {
            "parameters": parameter_count(factory()),
            "seed_results": seed_records,
            "clean_accuracy_mean": float(np.mean([float(row["clean_accuracy"]) for row in seed_records])),
            "clean_macro_f1_mean": float(np.mean([float(row["clean_macro_f1"]) for row in seed_records])),
            "equal_category_corrupted_accuracy_mean": float(np.mean([float(row["equal_category_corrupted_accuracy_mean"]) for row in seed_records])),
            "equal_category_corrupted_macro_f1_mean": float(np.mean([float(row["equal_category_corrupted_macro_f1_mean"]) for row in seed_records])),
            "category_accuracy_mean": {
                category: float(np.mean([float(row["category_accuracy_mean"][category]) for row in seed_records]))
                for category in ("shift", "frame_loss", "noise", "missing", "composite")
            },
        }

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prediction_record = {
        "date": "2026-08-16",
        "model_names": list(BASELINES),
        "seeds": list(SEEDS),
        "scenarios": [spec.name for spec in specs],
        "labels_dtype": "uint8",
        "labels_shape": [len(splits.official_test)],
        "labels_base64": base64.b64encode(labels[splits.official_test].astype(np.uint8).tobytes()).decode("ascii"),
        "predictions_dtype": "uint8",
        "predictions_shape": list(all_predictions.shape),
        "prediction_axis_order": ["model", "seed", "scenario", "sample"],
        "predictions_base64": base64.b64encode(all_predictions.tobytes()).decode("ascii"),
    }
    Path(args.predictions).write_text(json.dumps(prediction_record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {args.predictions}", flush=True)


if __name__ == "__main__":
    main()
