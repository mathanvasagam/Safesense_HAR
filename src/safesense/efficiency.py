"""Model-only FP32 latency benchmark helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .robustness import base_temporal_masks, load_fusion_final_model
from .safesense import (
    SAFESENSE_CONDITIONAL_SEEDS,
    SAFESENSE_EARLY_EXIT_SEEDS,
    SafeSense,
    SafeSenseBranchExpert,
    SafeSenseEarlyExitEnsemble,
    _feature_arrays,
    load_safesense_conditional_ensemble,
    load_safesense_early_exit_ensemble,
    safesense_splits,
)


def benchmark_model(
    model: nn.Module,
    inputs: Sequence[Tensor],
    *,
    device: str = "cuda",
    warmups: int = 200,
    iterations: int = 1000,
    checkpoint_bytes: int | None = None,
) -> dict[str, object]:
    """Measure only the forward call, synchronizing CUDA around each sample."""
    if warmups < 0 or iterations < 1:
        raise ValueError("warmups must be non-negative and iterations positive")
    selected = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(selected).eval()
    values = tuple(value.to(selected) for value in inputs)
    with torch.inference_mode():
        for _ in range(warmups):
            model(*values)
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        samples: list[float] = []
        for _ in range(iterations):
            if selected.type == "cuda":
                torch.cuda.synchronize(selected)
            started = time.perf_counter()
            model(*values)
            if selected.type == "cuda":
                torch.cuda.synchronize(selected)
            samples.append((time.perf_counter() - started) * 1000)
    return {
        "device": str(selected),
        "warmups": warmups,
        "iterations": iterations,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes_fp32": checkpoint_bytes,
        "median_ms_model_only": float(np.median(samples)),
        "p95_ms_model_only": float(np.percentile(samples, 95)),
    }


def checkpoint_bytes(path: str | Path) -> int:
    return Path(path).stat().st_size


def _cuda_graph_repair_internal_gaps(value: Tensor, mask: Tensor, *, radius: int = 3) -> Tensor:
    """Branch-free equivalent of SafeSense internal-gap repair for graph capture."""
    from .safesense import _zero_padded_shift_tensor

    valid = mask.float().clamp(0, 1)
    left = torch.zeros_like(valid, dtype=torch.bool)
    right = torch.zeros_like(valid, dtype=torch.bool)
    for offset in range(1, radius + 1):
        left |= _zero_padded_shift_tensor(valid, offset, time_dim=1).bool()
        right |= _zero_padded_shift_tensor(valid, -offset, time_dim=1).bool()
    repair = (valid <= 0) & left & right
    channels = value.transpose(1, 2)
    weights = valid.unsqueeze(1)
    kernel = 2 * radius + 1
    numerator = torch.nn.functional.avg_pool1d(channels * weights, kernel, stride=1, padding=radius) * kernel
    denominator = torch.nn.functional.avg_pool1d(weights, kernel, stride=1, padding=radius) * kernel
    filled = numerator / denominator.clamp_min(1.0)
    return torch.where(repair.unsqueeze(1), filled, channels).transpose(1, 2)


def _cuda_graph_safe_valid(valid: Tensor) -> Tensor:
    """Branch-free equivalent of FactorizedTimeModalityBlock._safe_valid."""
    safe = valid.clone()
    missing = ~safe.any(dim=1)
    safe[:, 0] = safe[:, 0] | missing
    return safe


class CudaGraphBatch1Inference(nn.Module):
    """Replay a frozen batch-1 CUDA forward with static device input buffers."""

    def __init__(self, model: nn.Module, example_inputs: Sequence[Tensor]):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph inference requires CUDA")
        values = tuple(example_inputs)
        if not values or any(value.device.type != "cuda" or value.shape[0] != 1 for value in values):
            raise ValueError("CUDA Graph inference requires non-empty CUDA batch-1 inputs")
        devices = {value.device for value in values}
        if len(devices) != 1:
            raise ValueError("CUDA Graph inputs must share one device")
        self.model = model.eval()
        self._device = values[0].device
        self._static_inputs = tuple(value.detach().clone() for value in values)
        self._graph = torch.cuda.CUDAGraph()

        from . import safesense as safesense_module

        original_repair = safesense_module._repair_internal_gaps
        original_safe_valid = safesense_module.FactorizedTimeModalityBlock.__dict__["_safe_valid"]
        patch_safesense = isinstance(self.model, safesense_module.SafeSenseFactorized)
        if patch_safesense:
            safesense_module._repair_internal_gaps = _cuda_graph_repair_internal_gaps
            safesense_module.FactorizedTimeModalityBlock._safe_valid = staticmethod(_cuda_graph_safe_valid)
        try:
            with torch.inference_mode():
                for _ in range(20):
                    self.model(*self._static_inputs)
            torch.cuda.synchronize(self._device)
            with torch.inference_mode(), torch.cuda.graph(self._graph):
                self._static_output = self.model(*self._static_inputs)
        finally:
            if patch_safesense:
                safesense_module._repair_internal_gaps = original_repair
                safesense_module.FactorizedTimeModalityBlock._safe_valid = original_safe_valid

    def forward(self, *inputs: Tensor):
        if len(inputs) != len(self._static_inputs):
            raise ValueError("CUDA Graph input count changed after capture")
        for static, value in zip(self._static_inputs, inputs):
            if value.shape != static.shape or value.dtype != static.dtype or value.device != static.device:
                raise ValueError("CUDA Graph input shape, dtype, and device must match capture")
            static.copy_(value)
        self._graph.replay()
        return self._static_output


def benchmark_conditional_ensemble_development(
    robust_run_dir: str | Path,
    prepared_dir: str | Path,
    fusion_repo_dir: str | Path,
    fusion_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    warmups: int = 200,
    iterations: int = 1000,
) -> dict[str, object]:
    """Benchmark clean fast path and missing-modality routed path on subject 7."""
    selected = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir, prepared = Path(robust_run_dir), Path(prepared_dir)
    primary = run_dir / f"seed-{SAFESENSE_CONDITIONAL_SEEDS[0]}/checkpoint.pt"
    auxiliary = run_dir / f"seed-{SAFESENSE_CONDITIONAL_SEEDS[1]}/checkpoint.pt"
    conditional = load_safesense_conditional_ensemble(primary, auxiliary)
    single = SafeSense(variant="robust")
    single.load_state_dict(torch.load(primary, map_location="cpu", weights_only=True))
    fusion = load_fusion_final_model(fusion_repo_dir, fusion_checkpoint, selected)

    fused = np.load(prepared / "skeleton_train_features.npy")
    index = int(safesense_splits(prepared).development_valid[0])
    raw_sample = fused[index:index + 1]
    skeleton, imu, _ = _feature_arrays(raw_sample)
    masks = base_temporal_masks(raw_sample)
    pair_mask = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
    with np.load(run_dir / f"seed-{SAFESENSE_CONDITIONAL_SEEDS[0]}/stats.npz") as stats:
        skeleton = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
        imu = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    skeleton *= pair_mask[..., 0, None]
    imu *= pair_mask[..., 1, None]
    clean_inputs = (torch.from_numpy(skeleton), torch.from_numpy(imu), torch.from_numpy(pair_mask))
    missing_imu = imu.copy(); missing_imu[:] = 0
    missing_mask = pair_mask.copy(); missing_mask[..., 1] = 0
    routed_inputs = (torch.from_numpy(skeleton), torch.from_numpy(missing_imu), torch.from_numpy(missing_mask))
    checkpoint_total = checkpoint_bytes(primary) + checkpoint_bytes(auxiliary)
    clean_result = benchmark_model(conditional, clean_inputs, device=device, warmups=warmups, iterations=iterations, checkpoint_bytes=checkpoint_total)
    with torch.inference_mode():
        conditional(*(value.to(selected) for value in clean_inputs))
    clean_route = conditional.auxiliary_route
    routed_result = benchmark_model(conditional, routed_inputs, device=device, warmups=warmups, iterations=iterations, checkpoint_bytes=checkpoint_total)
    with torch.inference_mode():
        conditional(*(value.to(selected) for value in routed_inputs))
    routed_route = conditional.auxiliary_route
    single_result = benchmark_model(single, clean_inputs, device=device, warmups=warmups, iterations=iterations, checkpoint_bytes=checkpoint_bytes(primary))
    fusion_result = benchmark_model(fusion, (torch.from_numpy(raw_sample),), device=device, warmups=warmups, iterations=iterations, checkpoint_bytes=checkpoint_bytes(fusion_checkpoint))
    if clean_route is None or bool(clean_route.any()):
        raise RuntimeError("conditional clean benchmark unexpectedly invoked the auxiliary model")
    if routed_route is None or not bool(routed_route.all()):
        raise RuntimeError("conditional missing-modality benchmark did not invoke the auxiliary model")
    result = {
        "status": "completed",
        "device": str(selected), "batch_size": 1, "dtype": "float32", "warmups": warmups, "iterations": iterations,
        "models": {
            "SafeSense conditional clean": clean_result,
            "SafeSense conditional routed": routed_result,
            "SafeSense robust seed 11": single_result,
            "Fusion-GCN": fusion_result,
        },
        "fusion_to_clean_latency_ratio": float(fusion_result["median_ms_model_only"]) / float(clean_result["median_ms_model_only"]),
        "fusion_to_routed_latency_ratio": float(fusion_result["median_ms_model_only"]) / float(routed_result["median_ms_model_only"]),
        "routed_to_clean_latency_ratio": float(routed_result["median_ms_model_only"]) / float(clean_result["median_ms_model_only"]),
        "clean_auxiliary_triggered": False,
        "routed_auxiliary_triggered": True,
        "latency_scope": "model-only forward; excludes preprocessing and end-to-end latency",
        "input_source": "first subject-7 development sample from prepared odd-subject arrays",
        "even_subject_arrays_loaded": False,
    }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    write_efficiency_result(output / "result.json", result)
    return result


def benchmark_early_exit_development(
    robust_run_dir: str | Path,
    prepared_dir: str | Path,
    fusion_repo_dir: str | Path,
    fusion_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    warmups: int = 200,
    iterations: int = 1000,
) -> dict[str, object]:
    """Benchmark sparse-precheck clean fast path and worst routed skeleton-expert path."""
    selected = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir, prepared, output = Path(robust_run_dir), Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary_path = run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[0]}/checkpoint.pt"
    auxiliary_source = run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[1]}/checkpoint.pt"
    candidate = load_safesense_early_exit_ensemble(primary_path, auxiliary_source)
    auxiliary = candidate.auxiliary
    auxiliary_checkpoint = output / "seed-22-branch-expert.pt"
    torch.save({name: value.detach().cpu() for name, value in auxiliary.state_dict().items()}, auxiliary_checkpoint)
    fusion = load_fusion_final_model(fusion_repo_dir, fusion_checkpoint, selected)

    fused = np.load(prepared / "skeleton_train_features.npy")
    index = int(safesense_splits(prepared).development_valid[0])
    raw_sample = fused[index:index + 1]
    skeleton, imu, _ = _feature_arrays(raw_sample)
    masks = base_temporal_masks(raw_sample)
    pair_mask = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
    with np.load(run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[0]}/stats.npz") as stats:
        skeleton = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
        imu = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    skeleton *= pair_mask[..., 0, None]
    imu *= pair_mask[..., 1, None]
    clean_inputs = (torch.from_numpy(skeleton), torch.from_numpy(imu), torch.from_numpy(pair_mask))

    # Missing IMU is the worst routed path: both extracted experts execute their wider skeleton branches.
    missing_imu = imu.copy()
    missing_imu[:] = 0
    missing_mask = pair_mask.copy()
    missing_mask[..., 1] = 0
    routed_inputs = (torch.from_numpy(skeleton), torch.from_numpy(missing_imu), torch.from_numpy(missing_mask))

    checkpoint_total = checkpoint_bytes(primary_path) + checkpoint_bytes(auxiliary_checkpoint)
    clean_result = benchmark_model(
        candidate, clean_inputs, device=device, warmups=warmups, iterations=iterations, checkpoint_bytes=checkpoint_total
    )
    with torch.inference_mode():
        candidate(*(value.to(selected) for value in clean_inputs))
    clean_route = candidate.auxiliary_route
    routed_result = benchmark_model(
        candidate, routed_inputs, device=device, warmups=warmups, iterations=iterations, checkpoint_bytes=checkpoint_total
    )
    with torch.inference_mode():
        candidate(*(value.to(selected) for value in routed_inputs))
    routed_route = candidate.auxiliary_route
    routed_use_skeleton = candidate.use_skeleton
    fusion_result = benchmark_model(
        fusion, (torch.from_numpy(raw_sample),), device=device, warmups=warmups, iterations=iterations,
        checkpoint_bytes=checkpoint_bytes(fusion_checkpoint),
    )
    if clean_route is None or bool(clean_route.any()):
        raise RuntimeError("early-exit clean benchmark unexpectedly routed away from the full primary")
    if routed_route is None or not bool(routed_route.all()):
        raise RuntimeError("early-exit missing-IMU benchmark did not take the routed path")
    if routed_use_skeleton is None or not bool(routed_use_skeleton.all()):
        raise RuntimeError("early-exit missing-IMU benchmark did not select the skeleton experts")

    clean_median_beats = float(clean_result["median_ms_model_only"]) < float(fusion_result["median_ms_model_only"])
    clean_p95_beats = float(clean_result["p95_ms_model_only"]) < float(fusion_result["p95_ms_model_only"])
    routed_median_beats = float(routed_result["median_ms_model_only"]) < float(fusion_result["median_ms_model_only"])
    routed_p95_beats = float(routed_result["p95_ms_model_only"]) < float(fusion_result["p95_ms_model_only"])
    result = {
        "status": "completed",
        "device": str(selected), "batch_size": 1, "dtype": "float32",
        "warmups": warmups, "iterations": iterations,
        "models": {
            "SafeSense early-exit clean": clean_result,
            "SafeSense early-exit routed worst-case": routed_result,
            "Fusion-GCN": fusion_result,
        },
        "auxiliary_parameters": sum(parameter.numel() for parameter in auxiliary.parameters()),
        "auxiliary_checkpoint_bytes_fp32": checkpoint_bytes(auxiliary_checkpoint),
        "fusion_to_clean_median_ratio": float(fusion_result["median_ms_model_only"]) / float(clean_result["median_ms_model_only"]),
        "fusion_to_routed_median_ratio": float(fusion_result["median_ms_model_only"]) / float(routed_result["median_ms_model_only"]),
        "clean_median_beats_fusion": clean_median_beats,
        "clean_p95_beats_fusion": clean_p95_beats,
        "routed_median_beats_fusion": routed_median_beats,
        "routed_p95_beats_fusion": routed_p95_beats,
        "all_latency_checks_passed": clean_median_beats and clean_p95_beats and routed_median_beats and routed_p95_beats,
        "clean_auxiliary_triggered": False,
        "routed_auxiliary_triggered": True,
        "routed_uses_skeleton_experts": True,
        "latency_scope": "model-only forward; excludes preprocessing and end-to-end latency",
        "input_source": "first subject-7 development sample from prepared odd-subject arrays",
        "even_subject_arrays_loaded": False,
    }
    write_efficiency_result(output / "result.json", result)
    return result


def benchmark_early_exit_posthoc(
    robust_final_dir: str | Path,
    prepared_dir: str | Path,
    fusion_repo_dir: str | Path,
    fusion_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    warmups: int = 200,
    iterations: int = 1000,
) -> dict[str, object]:
    """Benchmark frozen early-exit clean and routed paths on the official even-subject partition."""
    selected = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir, prepared, output = Path(robust_final_dir), Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary_path = run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[0]}/checkpoint.pt"
    auxiliary_source = run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[1]}/checkpoint.pt"
    candidate = load_safesense_early_exit_ensemble(primary_path, auxiliary_source)
    auxiliary_checkpoint = output / "seed-22-branch-expert.pt"
    torch.save(
        {name: value.detach().cpu() for name, value in candidate.auxiliary.state_dict().items()},
        auxiliary_checkpoint,
    )
    fusion = load_fusion_final_model(fusion_repo_dir, fusion_checkpoint, selected)

    train = np.load(prepared / "skeleton_train_features.npy")
    held_out = np.load(prepared / "skeleton_val_features.npy")
    fused = np.concatenate((train, held_out), axis=0)
    index = int(safesense_splits(prepared).official_test[0])
    raw_sample = fused[index:index + 1]
    skeleton, imu, _ = _feature_arrays(raw_sample)
    masks = base_temporal_masks(raw_sample)
    pair_mask = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
    with np.load(run_dir / f"seed-{SAFESENSE_EARLY_EXIT_SEEDS[0]}/stats.npz") as stats:
        skeleton = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
        imu = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    skeleton *= pair_mask[..., 0, None]
    imu *= pair_mask[..., 1, None]
    clean_inputs = (torch.from_numpy(skeleton), torch.from_numpy(imu), torch.from_numpy(pair_mask))

    missing_imu = imu.copy()
    missing_imu[:] = 0
    missing_mask = pair_mask.copy()
    missing_mask[..., 1] = 0
    routed_inputs = (torch.from_numpy(skeleton), torch.from_numpy(missing_imu), torch.from_numpy(missing_mask))

    checkpoint_total = checkpoint_bytes(primary_path) + checkpoint_bytes(auxiliary_checkpoint)
    clean_result = benchmark_model(
        candidate, clean_inputs, device=device, warmups=warmups, iterations=iterations,
        checkpoint_bytes=checkpoint_total,
    )
    with torch.inference_mode():
        candidate(*(value.to(selected) for value in clean_inputs))
    clean_route = candidate.auxiliary_route
    routed_result = benchmark_model(
        candidate, routed_inputs, device=device, warmups=warmups, iterations=iterations,
        checkpoint_bytes=checkpoint_total,
    )
    with torch.inference_mode():
        candidate(*(value.to(selected) for value in routed_inputs))
    routed_route = candidate.auxiliary_route
    routed_use_skeleton = candidate.use_skeleton
    fusion_result = benchmark_model(
        fusion, (torch.from_numpy(raw_sample),), device=device, warmups=warmups, iterations=iterations,
        checkpoint_bytes=checkpoint_bytes(fusion_checkpoint),
    )
    if clean_route is None or bool(clean_route.any()):
        raise RuntimeError("frozen early-exit clean post-hoc sample unexpectedly routed")
    if routed_route is None or not bool(routed_route.all()):
        raise RuntimeError("frozen early-exit routed post-hoc sample did not route")
    if routed_use_skeleton is None or not bool(routed_use_skeleton.all()):
        raise RuntimeError("missing-IMU post-hoc latency sample did not select skeleton experts")

    result = {
        "status": "completed",
        "workflow": "early_exit_branch_expert_posthoc_even_subject_efficiency",
        "selection_frozen_before_even_subject_evaluation": True,
        "device": str(selected),
        "batch_size": 1,
        "dtype": "float32",
        "warmups": warmups,
        "iterations": iterations,
        "models": {
            "SafeSense early-exit clean": clean_result,
            "SafeSense early-exit routed worst-case": routed_result,
            "Fusion-GCN": fusion_result,
        },
        "auxiliary_parameters": sum(parameter.numel() for parameter in candidate.auxiliary.parameters()),
        "auxiliary_checkpoint_bytes_fp32": checkpoint_bytes(auxiliary_checkpoint),
        "fusion_to_clean_median_ratio": float(fusion_result["median_ms_model_only"]) / float(clean_result["median_ms_model_only"]),
        "fusion_to_routed_median_ratio": float(fusion_result["median_ms_model_only"]) / float(routed_result["median_ms_model_only"]),
        "clean_median_beats_fusion": float(clean_result["median_ms_model_only"]) < float(fusion_result["median_ms_model_only"]),
        "clean_p95_beats_fusion": float(clean_result["p95_ms_model_only"]) < float(fusion_result["p95_ms_model_only"]),
        "routed_median_beats_fusion": float(routed_result["median_ms_model_only"]) < float(fusion_result["median_ms_model_only"]),
        "routed_p95_beats_fusion": float(routed_result["p95_ms_model_only"]) < float(fusion_result["p95_ms_model_only"]),
        "latency_scope": "model-only forward; excludes preprocessing and end-to-end latency",
        "input_source": "first official even-subject test sample",
        "posthoc_test_subjects": [2, 4, 6, 8],
    }
    write_efficiency_result(output / "result.json", result)
    return result


def write_efficiency_result(path: str | Path, result: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def benchmark_frozen_models(
    safesense_checkpoint: str | Path,
    prepared_dir: str | Path,
    safesense_stats_path: str | Path,
    fusion_repo_dir: str | Path,
    fusion_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    safesense_variant: str = "full",
) -> dict[str, object]:
    selected = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    safe = SafeSense(variant=safesense_variant)
    safe.load_state_dict(torch.load(safesense_checkpoint, map_location="cpu", weights_only=True))
    fusion = load_fusion_final_model(fusion_repo_dir, fusion_checkpoint, selected)
    fused = np.concatenate((
        np.load(Path(prepared_dir) / "skeleton_train_features.npy"),
        np.load(Path(prepared_dir) / "skeleton_val_features.npy"),
    ), axis=0)
    test_index = int(safesense_splits(prepared_dir).official_test[0])
    raw_sample = fused[test_index:test_index + 1]
    skeleton, imu, _ = _feature_arrays(raw_sample)
    masks = base_temporal_masks(raw_sample)
    with np.load(safesense_stats_path) as stats:
        skeleton = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32)
        imu = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32)
    pair_mask = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
    skeleton *= pair_mask[..., 0, None]
    imu *= pair_mask[..., 1, None]
    safe_inputs = (torch.from_numpy(skeleton), torch.from_numpy(imu), torch.from_numpy(pair_mask))
    fusion_inputs = (torch.from_numpy(raw_sample),)
    safe_result = benchmark_model(safe, safe_inputs, device=device, warmups=200, iterations=1000, checkpoint_bytes=checkpoint_bytes(safesense_checkpoint))
    fusion_result = benchmark_model(fusion, fusion_inputs, device=device, warmups=200, iterations=1000, checkpoint_bytes=checkpoint_bytes(fusion_checkpoint))
    result = {
        "status": "completed",
        "device": str(selected),
        "batch_size": 1,
        "dtype": "float32",
        "warmups": 200,
        "iterations": 1000,
        "models": {"SafeSense": safe_result, "Fusion-GCN": fusion_result},
        "safesense_variant": safesense_variant,
        "latency_ratio_fusion_to_safesense": float(fusion_result["median_ms_model_only"]) / float(safe_result["median_ms_model_only"]),
        "latency_ratio_safesense_to_fusion": float(safe_result["median_ms_model_only"]) / float(fusion_result["median_ms_model_only"]),
        "latency_scope": "model-only forward; excludes preprocessing and end-to-end latency",
        "input_source": "first official even-subject test sample from prepared fused arrays",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_efficiency_result(output / "result.json", result)
    return result
