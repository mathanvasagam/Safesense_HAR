"""SafeSense skeleton/IMU model and reproducible training harness."""

from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 27
FRAME_COUNT = 128
SKELETON_INPUTS = 177
IMU_INPUTS = 12
SAFESENSE_VARIANTS = ("full", "equal", "skeleton", "imu", "classwise", "classprior", "aligned", "temporal", "robust")
EXPERT_AUXILIARY_CE_WEIGHT = 0.2
LAG_AUXILIARY_CE_WEIGHT = 0.2
TEMPORAL_CONSISTENCY_WEIGHT = 0.15
TEMPORAL_CONSISTENCY_TEMPERATURE = 2.0
TEMPORAL_CLEAN_VIEW_WEIGHT = 0.65
ROBUST_RECONSTRUCTION_WEIGHT = 0.05
ROBUST_FEATURE_DISTILL_WEIGHT = 0.10
ROBUST_GATE_AUXILIARY_WEIGHT = 0.10
ROBUST_TOPOLOGY_AUXILIARY_WEIGHT = 0.15
ROBUST_TEMPORAL_CORRUPTED_WEIGHT = 0.20
ROBUST_STRUCTURED_CORRUPTED_WEIGHT = 0.15
ROBUST_ADAPTER_CLEAN_TOLERANCE = 0.003
ROBUST_ADAPTER_CLEAN_SCORE_WEIGHT = 0.10
ROBUST_EFFECTIVE_GATE_CENTER = 0.15
ROBUST_EFFECTIVE_GATE_SHARPNESS = 20.0
SAFESENSE_CONDITIONAL_SEEDS = (11, 22)
SAFESENSE_CONDITIONAL_WEIGHTS = (0.5, 0.5)
SAFESENSE_CONDITIONAL_GATE_THRESHOLD = ROBUST_EFFECTIVE_GATE_CENTER
SAFESENSE_EARLY_EXIT_SEEDS = (11, 22)
SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT = 0.40
SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD = 1.8728979055100403
SAFESENSE_EARLY_EXIT_SKELETON_CHANNEL = 9
SAFESENSE_EARLY_EXIT_IMU_CHANNEL = 0
LAG_SHIFTS = tuple(range(-4, 5))
ROBUST_TRAINING_PROBABILITY = 0.5
SUBJECT_STYLE_CLASSIFICATION_WEIGHT = 0.20
SUBJECT_STYLE_CONSISTENCY_WEIGHT = 0.10
FACTORIZED_CLEAN_WEIGHT = 0.55
FACTORIZED_TEMPORAL_WEIGHT = 0.15
FACTORIZED_STRUCTURED_WEIGHT = 0.15
FACTORIZED_STYLE_WEIGHT = 0.15
FACTORIZED_EXPERT_AUXILIARY_WEIGHT = 0.15
FACTORIZED_ALIGNMENT_WEIGHT = 0.15
FACTORIZED_CONSISTENCY_WEIGHT = 0.10
# Official UTD-MHAD Kinect-1 edges, zero-based.
BONE_EDGES = ((0, 1), (2, 1), (4, 1), (8, 1), (3, 2), (12, 3), (16, 3),
              (5, 4), (6, 5), (7, 6), (9, 8), (10, 9), (11, 10), (13, 12),
              (14, 13), (15, 14), (17, 16), (18, 17), (19, 18))


class SafeSenseError(ValueError):
    """Raised for invalid SafeSense artifacts or configurations."""


class _ProgressReporter:
    """Write periodic progress to stdout and a repo/volume-local log file."""

    def __init__(self, path: Path, *, interval_seconds: float = 30.0):
        if interval_seconds <= 0:
            raise SafeSenseError("progress interval must be positive")
        self.path = path
        self.interval_seconds = float(interval_seconds)
        self.started = time.monotonic()
        self._state = "initializing"
        self._state_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="safesense-progress", daemon=True)

    def _emit(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        elapsed = time.monotonic() - self.started
        line = f"[safesense-progress] {stamp} elapsed={elapsed:.1f}s {message}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()

    def start(self, state: str) -> None:
        self.update(state, emit=True)
        self._thread.start()

    def update(self, state: str, *, emit: bool = False) -> None:
        with self._state_lock:
            self._state = state
        if emit:
            self._emit(f"state={state}")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            with self._state_lock:
                state = self._state
            self._emit(f"heartbeat state={state}")

    def close(self, status: str = "completed") -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._state_lock:
            state = self._state
        self._emit(f"status={status} state={state}")


@dataclass(frozen=True)
class SafeSenseTrainingConfig:
    seed: int = 11
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SafeSenseSplits:
    development_train: np.ndarray
    development_valid: np.ndarray
    official_test: np.ndarray


def set_safesense_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _zero_padded_shift_tensor(value: Tensor, amount: int, *, time_dim: int) -> Tensor:
    """Shift a tensor along one temporal dimension with zero padding."""
    if not -value.shape[time_dim] < amount < value.shape[time_dim]:
        return torch.zeros_like(value)
    shifted = torch.zeros_like(value)
    source = [slice(None)] * value.ndim
    destination = [slice(None)] * value.ndim
    if amount >= 0:
        source[time_dim] = slice(0, value.shape[time_dim] - amount if amount else None)
        destination[time_dim] = slice(amount, None)
    else:
        source[time_dim] = slice(-amount, None)
        destination[time_dim] = slice(0, amount)
    shifted[tuple(destination)] = value[tuple(source)]
    return shifted


def _corrective_lag_target(applied_shift: int) -> int:
    corrective = -int(applied_shift)
    if corrective not in LAG_SHIFTS:
        raise ValueError(f"applied shift must map into {LAG_SHIFTS}, got {applied_shift}")
    return LAG_SHIFTS.index(corrective)


def _corrective_lag_target_for_modality(applied_shift: int, modality: str) -> int:
    """Return the corrective IMU-shift class for a shifted input modality."""
    if modality not in ("skeleton", "imu"):
        raise ValueError("modality must be skeleton or imu")
    corrective = int(applied_shift) if modality == "skeleton" else -int(applied_shift)
    if corrective not in LAG_SHIFTS:
        raise ValueError(f"applied shift must map into {LAG_SHIFTS}, got {applied_shift}")
    return LAG_SHIFTS.index(corrective)


def _repair_internal_gaps(value: Tensor, mask: Tensor, *, radius: int = 3) -> Tensor:
    """Fill only short internal missing runs from nearby valid frames; preserve boundaries."""
    if value.ndim != 3 or mask.ndim != 2 or tuple(value.shape[:2]) != tuple(mask.shape):
        raise ValueError("value/mask must have shapes [B,T,F] and [B,T]")
    valid = mask.float().clamp(0, 1)
    left = torch.zeros_like(valid, dtype=torch.bool)
    right = torch.zeros_like(valid, dtype=torch.bool)
    for offset in range(1, radius + 1):
        left |= _zero_padded_shift_tensor(valid, offset, time_dim=1).bool()
        right |= _zero_padded_shift_tensor(valid, -offset, time_dim=1).bool()
    repair = (valid <= 0) & left & right
    if not bool(repair.any()):
        return value
    channels = value.transpose(1, 2)
    weights = valid.unsqueeze(1)
    kernel = 2 * radius + 1
    numerator = nn.functional.avg_pool1d(channels * weights, kernel, stride=1, padding=radius) * kernel
    denominator = nn.functional.avg_pool1d(weights, kernel, stride=1, padding=radius) * kernel
    filled = numerator / denominator.clamp_min(1.0)
    repaired = torch.where(repair.unsqueeze(1), filled, channels)
    return repaired.transpose(1, 2)


def _sanitize_masked_motion_features(
    skeleton: Tensor,
    imu: Tensor,
    skeleton_mask: Tensor,
    imu_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Remove derivative artifacts across invalid temporal transitions."""
    skeleton = skeleton.clone()
    imu = imu.clone()
    skeleton_valid = skeleton_mask.float().clamp(0, 1)
    imu_valid = imu_mask.float().clamp(0, 1)
    skeleton = skeleton * skeleton_valid.unsqueeze(-1)
    imu = imu * imu_valid.unsqueeze(-1)
    skeleton_transition = skeleton_valid.clone()
    imu_transition = imu_valid.clone()
    if skeleton.shape[1] > 1:
        skeleton_transition[:, 1:] = skeleton_valid[:, 1:] * skeleton_valid[:, :-1]
        imu_transition[:, 1:] = imu_valid[:, 1:] * imu_valid[:, :-1]
    skeleton[:, :, 60:120] = skeleton[:, :, 60:120] * skeleton_transition.unsqueeze(-1)
    imu[:, :, 6:12] = imu[:, :, 6:12] * imu_transition.unsqueeze(-1)
    return skeleton, imu


def _pair_mask(mask: Tensor, shape: tuple[int, int]) -> Tensor:
    if mask.ndim == 2 and tuple(mask.shape) == shape:
        return torch.stack((mask, mask), dim=-1)
    if mask.ndim == 3 and tuple(mask.shape[:2]) == shape and mask.shape[-1] == 2:
        return mask.clone()
    raise ValueError("mask must have shape [B,T] or [B,T,2]")


def _drop_frames(value: Tensor, mask: Tensor, fraction: float) -> tuple[Tensor, Tensor]:
    valid = torch.nonzero(mask > 0, as_tuple=False).flatten()
    count = min(int(round(valid.numel() * fraction)), int(valid.numel()))
    if count:
        selected = valid[torch.randperm(valid.numel(), device=value.device)[:count]]
        value[selected] = 0
        mask[selected] = 0
    return value, mask


def _augment_aligned_batch(
    skeleton: Tensor,
    imu: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply deterministic-under-seed synthetic corruption for aligned training."""
    skeleton = skeleton.clone()
    imu = imu.clone()
    pair_mask = _pair_mask(mask, tuple(skeleton.shape[:2])).float()
    batch_size = skeleton.shape[0]
    lag_targets = torch.full((batch_size,), _corrective_lag_target(0), dtype=torch.long, device=skeleton.device)
    lag_supervision = torch.zeros(batch_size, dtype=torch.bool, device=skeleton.device)
    corrupt = torch.rand(batch_size, device=skeleton.device) < ROBUST_TRAINING_PROBABILITY
    categories = torch.randint(0, 5, (batch_size,), device=skeleton.device)
    fractions = (0.1, 0.3, 0.5)
    noise_levels = (0.1, 0.3, 0.5)

    for row in range(batch_size):
        if not bool(corrupt[row]):
            lag_supervision[row] = bool(pair_mask[row, :, 0].any() and pair_mask[row, :, 1].any())
            continue
        category = int(categories[row])
        modality = int(torch.randint(0, 2, (1,), device=skeleton.device))
        if category == 0:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            imu[row] = _zero_padded_shift_tensor(imu[row], applied, time_dim=0)
            pair_mask[row, :, 1] = _zero_padded_shift_tensor(pair_mask[row, :, 1], applied, time_dim=0)
            lag_targets[row] = _corrective_lag_target(applied)
            lag_supervision[row] = bool(pair_mask[row, :, 0].any() and pair_mask[row, :, 1].any())
        elif category == 1:
            fraction = fractions[int(torch.randint(0, len(fractions), (1,), device=skeleton.device))]
            target = skeleton[row] if modality == 0 else imu[row]
            _drop_frames(target, pair_mask[row, :, modality], fraction)
        elif category == 2:
            noise_std = noise_levels[int(torch.randint(0, len(noise_levels), (1,), device=skeleton.device))]
            target = skeleton[row] if modality == 0 else imu[row]
            valid = pair_mask[row, :, modality].bool().unsqueeze(-1)
            target.add_(torch.randn_like(target) * noise_std * valid)
        elif category == 3:
            if modality == 0:
                skeleton[row].zero_()
            else:
                imu[row].zero_()
            pair_mask[row, :, modality] = 0
        else:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            imu[row] = _zero_padded_shift_tensor(imu[row], applied, time_dim=0)
            pair_mask[row, :, 1] = _zero_padded_shift_tensor(pair_mask[row, :, 1], applied, time_dim=0)
            _drop_frames(imu[row], pair_mask[row, :, 1], 0.3)
            valid = pair_mask[row, :, 1].bool().unsqueeze(-1)
            imu[row].add_(torch.randn_like(imu[row]) * 0.3 * valid)
            lag_targets[row] = _corrective_lag_target(applied)
            lag_supervision[row] = bool(pair_mask[row, :, 0].any() and pair_mask[row, :, 1].any())

    availability = torch.stack((pair_mask[..., 0].any(dim=1), pair_mask[..., 1].any(dim=1)), dim=1).float()
    return skeleton, imu, pair_mask, availability, lag_targets, lag_supervision


def _augment_temporal_batch(
    skeleton: Tensor,
    imu: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create one corrupted view per sample while retaining a separate clean view."""
    skeleton = skeleton.clone()
    imu = imu.clone()
    pair_mask = _pair_mask(mask, tuple(skeleton.shape[:2])).float()
    batch_size = skeleton.shape[0]
    lag_targets = torch.full((batch_size,), _corrective_lag_target(0), dtype=torch.long, device=skeleton.device)
    categories = torch.randint(0, 5, (batch_size,), device=skeleton.device)
    fractions = (0.1, 0.3, 0.5)
    noise_levels = (0.1, 0.3, 0.5)

    for row in range(batch_size):
        category = int(categories[row])
        modality_index = int(torch.randint(0, 2, (1,), device=skeleton.device))
        modality = "skeleton" if modality_index == 0 else "imu"
        if category == 0:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            target = skeleton[row] if modality == "skeleton" else imu[row]
            target.copy_(_zero_padded_shift_tensor(target, applied, time_dim=0))
            pair_mask[row, :, modality_index] = _zero_padded_shift_tensor(pair_mask[row, :, modality_index], applied, time_dim=0)
            lag_targets[row] = _corrective_lag_target_for_modality(applied, modality)
        elif category == 1:
            fraction = fractions[int(torch.randint(0, len(fractions), (1,), device=skeleton.device))]
            target = skeleton[row] if modality == "skeleton" else imu[row]
            _drop_frames(target, pair_mask[row, :, modality_index], fraction)
        elif category == 2:
            noise_std = noise_levels[int(torch.randint(0, len(noise_levels), (1,), device=skeleton.device))]
            target = skeleton[row] if modality == "skeleton" else imu[row]
            valid = pair_mask[row, :, modality_index].bool().unsqueeze(-1)
            target.add_(torch.randn_like(target) * noise_std * valid)
        elif category == 3:
            if modality == "skeleton":
                skeleton[row].zero_()
            else:
                imu[row].zero_()
            pair_mask[row, :, modality_index] = 0
        else:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            imu[row] = _zero_padded_shift_tensor(imu[row], applied, time_dim=0)
            pair_mask[row, :, 1] = _zero_padded_shift_tensor(pair_mask[row, :, 1], applied, time_dim=0)
            _drop_frames(imu[row], pair_mask[row, :, 1], 0.3)
            valid = pair_mask[row, :, 1].bool().unsqueeze(-1)
            imu[row].add_(torch.randn_like(imu[row]) * 0.3 * valid)
            lag_targets[row] = _corrective_lag_target(applied)

    availability = torch.stack((pair_mask[..., 0].any(dim=1), pair_mask[..., 1].any(dim=1)), dim=1).float()
    lag_supervision = availability.bool().all(dim=1)
    return skeleton, imu, pair_mask, availability, lag_targets, lag_supervision


def _rebuild_engineered_features(skeleton_positions: Tensor, imu_values: Tensor) -> tuple[Tensor, Tensor]:
    """Rebuild SafeSense engineered features from raw skeleton positions and IMU channels."""
    if skeleton_positions.ndim != 3 or skeleton_positions.shape[-1] != 60:
        raise ValueError("skeleton_positions must have shape [B,T,60]")
    if imu_values.ndim != 3 or imu_values.shape[-1] != 6:
        raise ValueError("imu_values must have shape [B,T,6]")
    skeleton_difference = torch.zeros_like(skeleton_positions)
    imu_difference = torch.zeros_like(imu_values)
    if skeleton_positions.shape[1] > 1:
        skeleton_difference[:, 1:] = skeleton_positions[:, 1:] - skeleton_positions[:, :-1]
        imu_difference[:, 1:] = imu_values[:, 1:] - imu_values[:, :-1]
    joints = skeleton_positions.reshape(*skeleton_positions.shape[:2], 20, 3)
    bones = torch.stack([joints[:, :, child] - joints[:, :, parent] for child, parent in BONE_EDGES], dim=2)
    skeleton = torch.cat((skeleton_positions, skeleton_difference, bones.flatten(2)), dim=-1)
    imu = torch.cat((imu_values, imu_difference), dim=-1)
    return skeleton, imu


def _augment_subject_style_batch(
    skeleton: Tensor,
    imu: Tensor,
    mask: Tensor,
    stats: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply label-preserving subject/sensor style changes in raw physical space."""
    pair_mask = _pair_mask(mask, tuple(skeleton.shape[:2])).float()
    skeleton_positions = (skeleton * stats["skeleton_std"] + stats["skeleton_mean"])[:, :, :60].clone()
    imu_values = (imu * stats["imu_std"] + stats["imu_mean"])[:, :, :6].clone()
    joints = skeleton_positions.reshape(*skeleton_positions.shape[:2], 20, 3)
    valid = pair_mask[..., 0].float()
    root = joints[:, :, 1]
    anchor = (root * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    scale = 0.85 + 0.30 * torch.rand((len(joints), 1, 1, 1), device=joints.device, dtype=joints.dtype)
    yaw = (torch.rand(len(joints), device=joints.device, dtype=joints.dtype) * 2.0 - 1.0) * (15.0 * np.pi / 180.0)
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    rotation = torch.zeros((len(joints), 3, 3), device=joints.device, dtype=joints.dtype)
    rotation[:, 0, 0] = cosine; rotation[:, 0, 2] = sine
    rotation[:, 1, 1] = 1.0
    rotation[:, 2, 0] = -sine; rotation[:, 2, 2] = cosine
    centered = joints - anchor[:, None, None, :]
    transformed = torch.einsum("bij,btkj->btki", rotation, centered) * scale + anchor[:, None, None, :]
    translation_scale = stats["skeleton_noise_scale"] * 0.10
    translation = torch.randn((len(joints), 1, 1, 3), device=joints.device, dtype=joints.dtype) * translation_scale
    transformed = transformed + translation
    joints.copy_(torch.where(valid[..., None, None].bool(), transformed, joints))

    imu_triplets = imu_values.reshape(*imu_values.shape[:2], 2, 3)
    imu_rotated = torch.einsum("bij,btkj->btki", rotation, imu_triplets)
    gain = 0.90 + 0.20 * torch.rand((len(joints), 1, 2, 1), device=joints.device, dtype=joints.dtype)
    bias = torch.randn((len(joints), 1, 2, 1), device=joints.device, dtype=joints.dtype) * stats["imu_noise_scale"] * 0.03
    imu_styled = imu_rotated * gain + bias
    imu_valid = pair_mask[..., 1, None, None].bool()
    imu_triplets.copy_(torch.where(imu_valid, imu_styled, imu_triplets))

    rebuilt_skeleton, rebuilt_imu = _rebuild_engineered_features(joints.flatten(2), imu_triplets.flatten(2))
    rebuilt_skeleton = (rebuilt_skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]
    rebuilt_imu = (rebuilt_imu - stats["imu_mean"]) / stats["imu_std"]
    rebuilt_skeleton *= pair_mask[..., 0, None]
    rebuilt_imu *= pair_mask[..., 1, None]
    availability = torch.stack((pair_mask[..., 0].any(dim=1), pair_mask[..., 1].any(dim=1)), dim=1).float()
    return rebuilt_skeleton, rebuilt_imu, pair_mask, availability


def _augment_structured_temporal_batch(
    skeleton: Tensor,
    imu: Tensor,
    mask: Tensor,
    stats: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Corrupt raw physical channels, then rebuild/renormalize engineered features.

    This mirrors the robustness evaluator more closely than perturbing already-engineered
    standardized channels. Reconstruction supervision is emitted only for noise/frame-loss
    cases that a local denoiser can reasonably invert.
    """
    pair_mask = _pair_mask(mask, tuple(skeleton.shape[:2])).float()
    skeleton_positions = skeleton * stats["skeleton_std"] + stats["skeleton_mean"]
    imu_values = imu * stats["imu_std"] + stats["imu_mean"]
    skeleton_positions = skeleton_positions[:, :, :60].clone()
    imu_values = imu_values[:, :, :6].clone()
    batch_size = skeleton.shape[0]
    lag_targets = torch.full((batch_size,), _corrective_lag_target(0), dtype=torch.long, device=skeleton.device)
    reconstruction_supervision = torch.zeros((batch_size, 2), dtype=torch.bool, device=skeleton.device)
    correction_targets = torch.zeros((batch_size, 2), dtype=torch.float32, device=skeleton.device)
    categories = torch.randint(0, 5, (batch_size,), device=skeleton.device)
    fractions = (0.1, 0.3, 0.5)
    noise_levels = (0.1, 0.3, 0.5)

    for row in range(batch_size):
        category = int(categories[row])
        modality_index = int(torch.randint(0, 2, (1,), device=skeleton.device))
        modality = "skeleton" if modality_index == 0 else "imu"
        target = skeleton_positions[row] if modality == "skeleton" else imu_values[row]
        if category == 0:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            target.copy_(_zero_padded_shift_tensor(target, applied, time_dim=0))
            pair_mask[row, :, modality_index] = _zero_padded_shift_tensor(pair_mask[row, :, modality_index], applied, time_dim=0)
            lag_targets[row] = _corrective_lag_target_for_modality(applied, modality)
        elif category == 1:
            fraction = fractions[int(torch.randint(0, len(fractions), (1,), device=skeleton.device))]
            _drop_frames(target, pair_mask[row, :, modality_index], fraction)
            reconstruction_supervision[row, modality_index] = True
        elif category == 2:
            level = noise_levels[int(torch.randint(0, len(noise_levels), (1,), device=skeleton.device))]
            scale = stats[f"{modality}_noise_scale"] * level
            valid = pair_mask[row, :, modality_index].bool().unsqueeze(-1)
            target.add_(torch.randn_like(target) * scale * valid)
            reconstruction_supervision[row, modality_index] = True
            correction_targets[row, modality_index] = 1.0
        elif category == 3:
            target.zero_()
            pair_mask[row, :, modality_index] = 0
        else:
            magnitude = int(torch.randint(1, 5, (1,), device=skeleton.device))
            direction = -1 if int(torch.randint(0, 2, (1,), device=skeleton.device)) == 0 else 1
            applied = magnitude * direction
            imu_values[row] = _zero_padded_shift_tensor(imu_values[row], applied, time_dim=0)
            pair_mask[row, :, 1] = _zero_padded_shift_tensor(pair_mask[row, :, 1], applied, time_dim=0)
            _drop_frames(imu_values[row], pair_mask[row, :, 1], 0.3)
            valid = pair_mask[row, :, 1].bool().unsqueeze(-1)
            imu_values[row].add_(torch.randn_like(imu_values[row]) * stats["imu_noise_scale"] * 0.3 * valid)
            lag_targets[row] = _corrective_lag_target(applied)
            correction_targets[row, 1] = 1.0

    rebuilt_skeleton, rebuilt_imu = _rebuild_engineered_features(skeleton_positions, imu_values)
    rebuilt_skeleton = (rebuilt_skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]
    rebuilt_imu = (rebuilt_imu - stats["imu_mean"]) / stats["imu_std"]
    rebuilt_skeleton = rebuilt_skeleton * pair_mask[..., 0, None]
    rebuilt_imu = rebuilt_imu * pair_mask[..., 1, None]
    availability = torch.stack((pair_mask[..., 0].any(dim=1), pair_mask[..., 1].any(dim=1)), dim=1).float()
    lag_supervision = availability.bool().all(dim=1)
    return rebuilt_skeleton, rebuilt_imu, pair_mask, availability, lag_targets, lag_supervision, reconstruction_supervision, correction_targets


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(f1)),
        "per_class_f1": [float(value) for value in f1],
        "confusion_matrix": confusion.tolist(),
    }


def _load_manifest(prepared_dir: Path) -> list[dict[str, str]]:
    path = prepared_dir / "manifest.csv"
    if not path.is_file():
        raise SafeSenseError(f"missing Fusion-GCN manifest: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def safesense_odd_fold_splits(prepared_dir: str | Path, validation_subject: int = 7) -> SafeSenseSplits:
    if validation_subject not in (1, 3, 5, 7):
        raise SafeSenseError("odd-subject validation fold must be one of 1, 3, 5, 7")
    rows = _load_manifest(Path(prepared_dir))
    subjects = np.asarray([int(row["subject"]) for row in rows], dtype=np.int64)
    official_train_rows = np.flatnonzero(np.isin(subjects, (1, 3, 5, 7)))
    official_test_rows = np.flatnonzero(np.isin(subjects, (2, 4, 6, 8)))
    # prepare-fusion writes official train arrays first and held-out arrays second;
    # translate manifest order into that array order without changing the manifest.
    train_positions = np.full(len(rows), -1, dtype=np.int64)
    test_positions = np.full(len(rows), -1, dtype=np.int64)
    train_positions[official_train_rows] = np.arange(len(official_train_rows))
    test_positions[official_test_rows] = len(official_train_rows) + np.arange(len(official_test_rows))
    array_positions = np.where(train_positions >= 0, train_positions, test_positions)
    train_subjects = tuple(subject for subject in (1, 3, 5, 7) if subject != validation_subject)
    splits = SafeSenseSplits(
        development_train=array_positions[np.isin(subjects, train_subjects)],
        development_valid=array_positions[subjects == validation_subject],
        official_test=array_positions[np.isin(subjects, (2, 4, 6, 8))],
    )
    if len(rows) == 861:
        if len(splits.development_train) + len(splits.development_valid) != 431 or len(splits.official_test) != 430:
            raise SafeSenseError("manifest does not match the frozen UTD-MHAD subject protocol")
    if np.intersect1d(splits.development_train, splits.development_valid).size:
        raise SafeSenseError("odd-subject development train and validation folds overlap")
    if np.intersect1d(np.concatenate((splits.development_train, splits.development_valid)), splits.official_test).size:
        raise SafeSenseError("development and official test splits overlap")
    return splits


def safesense_splits(prepared_dir: str | Path) -> SafeSenseSplits:
    return safesense_odd_fold_splits(prepared_dir, validation_subject=7)


def _feature_arrays(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if array.ndim != 5 or array.shape[1:] != (1, FRAME_COUNT, 22, 3):
        raise SafeSenseError(f"expected official Fusion-GCN array [N,1,128,22,3], got {array.shape}")
    sequence = np.asarray(array[:, 0], dtype=np.float32)
    skeleton = sequence[:, :, :20]
    imu = sequence[:, :, 20:]
    skeleton_positions = skeleton.reshape(len(array), FRAME_COUNT, 60)
    skeleton_difference = np.zeros_like(skeleton_positions)
    skeleton_difference[:, 1:] = skeleton_positions[:, 1:] - skeleton_positions[:, :-1]
    bones = np.stack([skeleton[:, :, child] - skeleton[:, :, parent] for child, parent in BONE_EDGES], axis=2)
    skeleton_features = np.concatenate((skeleton_positions, skeleton_difference, bones.reshape(len(array), FRAME_COUNT, 57)), axis=-1)
    imu_features = imu.reshape(len(array), FRAME_COUNT, 6)
    imu_difference = np.zeros_like(imu_features)
    imu_difference[:, 1:] = imu_features[:, 1:] - imu_features[:, :-1]
    imu_features = np.concatenate((imu_features, imu_difference), axis=-1)
    temporal_mask = (np.abs(imu_features[:, :, :6]).sum(axis=-1) > 0).astype(np.float32)
    return skeleton_features, imu_features, temporal_mask


def _standard_stats(features: np.ndarray, indices: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = features[indices]
    valid = mask[indices].astype(bool)
    flattened = values[valid]
    if flattened.size == 0:
        raise SafeSenseError("training split has no valid IMU frames")
    mean = flattened.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flattened.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _raw_noise_scale(features: np.ndarray, indices: np.ndarray, mask: np.ndarray, width: int) -> np.ndarray:
    positions = np.asarray(indices, dtype=np.int64)
    values = features[positions, :, :width]
    valid = mask[positions].astype(bool)
    flattened = values[valid]
    if flattened.size == 0:
        raise SafeSenseError("training split has no valid raw values for corruption scaling")
    scale = float(flattened.std(dtype=np.float64)) or 1.0
    return np.asarray(scale, dtype=np.float32)


class SafeSenseDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, skeleton: np.ndarray, imu: np.ndarray, mask: np.ndarray, labels: np.ndarray, indices: Sequence[int], stats: dict[str, np.ndarray]):
        self.skeleton = torch.from_numpy(((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32, copy=False))
        self.imu = torch.from_numpy(((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32, copy=False))
        self.mask = torch.from_numpy(mask.astype(np.float32, copy=False))
        if self.mask.ndim == 2:
            skeleton_mask = imu_mask = self.mask
        elif self.mask.ndim == 3 and self.mask.shape[-1] == 2:
            skeleton_mask, imu_mask = self.mask[..., 0], self.mask[..., 1]
        else:
            raise SafeSenseError("mask must have shape [N,T] or [N,T,2]")
        self.skeleton = self.skeleton * skeleton_mask.unsqueeze(-1)
        self.imu = self.imu * imu_mask.unsqueeze(-1)
        self.availability = torch.stack((skeleton_mask.any(dim=1), imu_mask.any(dim=1)), dim=1).float()
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augmentation_stats = {
            name: torch.as_tensor(stats[name], dtype=torch.float32)
            for name in (
                "skeleton_mean", "skeleton_std", "imu_mean", "imu_std",
                "skeleton_noise_scale", "imu_noise_scale",
            )
        }

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        index = int(self.indices[position])
        return self.skeleton[index], self.imu[index], self.mask[index], self.availability[index], self.labels[index]



def load_safesense_data(prepared_dir: str | Path, train_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    prepared = Path(prepared_dir)
    required = ("skeleton_train_features.npy", "train_labels.npy", "skeleton_val_features.npy", "val_labels.npy")
    missing = [name for name in required if not (prepared / name).is_file()]
    if missing:
        raise SafeSenseError(f"missing prepared SafeSense arrays: {missing}")
    train_features = np.load(prepared / "skeleton_train_features.npy")
    val_features = np.load(prepared / "skeleton_val_features.npy")
    train_labels = np.load(prepared / "train_labels.npy")
    val_labels = np.load(prepared / "val_labels.npy")
    train_skeleton, train_imu, train_mask = _feature_arrays(train_features)
    val_skeleton, val_imu, val_mask = _feature_arrays(val_features)
    features = np.concatenate((train_skeleton, val_skeleton))
    imu = np.concatenate((train_imu, val_imu))
    mask = np.concatenate((train_mask, val_mask))
    labels = np.concatenate((train_labels, val_labels)).astype(np.int64, copy=False)
    # The prepared arrays are official train then held-out order, matching manifest.csv.
    if len(labels) != len(_load_manifest(prepared)):
        raise SafeSenseError("prepared arrays and manifest lengths differ")
    skeleton_mean, skeleton_std = _standard_stats(features, np.asarray(train_indices), mask)
    imu_mean, imu_std = _standard_stats(imu, np.asarray(train_indices), mask)
    stats = {
        "skeleton_mean": skeleton_mean,
        "skeleton_std": skeleton_std,
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "skeleton_noise_scale": _raw_noise_scale(features, np.asarray(train_indices), mask, 60),
        "imu_noise_scale": _raw_noise_scale(imu, np.asarray(train_indices), mask, 6),
    }
    return features, imu, mask, labels, stats


def _load_partition(prepared_dir: Path, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one prepared partition; development never opens the held-out arrays."""
    if partition not in ("train", "val"):
        raise ValueError(partition)
    features = np.load(prepared_dir / f"skeleton_{partition}_features.npy")
    labels = np.load(prepared_dir / f"{partition}_labels.npy").astype(np.int64, copy=False)
    skeleton, imu, mask = _feature_arrays(features)
    return skeleton, imu, mask, labels


def _canonicalize_skeleton_features(skeleton: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Canonicalize initial translation, yaw, and body scale while retaining relative motion."""
    if skeleton.ndim != 3 or skeleton.shape[-1] != SKELETON_INPUTS:
        raise SafeSenseError("canonicalization expects SafeSense skeleton features [N,T,177]")
    skeleton_mask = mask.astype(bool, copy=False) if mask.ndim == 2 else mask[..., 0].astype(bool, copy=False)
    positions = skeleton[:, :, :60].reshape(len(skeleton), FRAME_COUNT, 20, 3).astype(np.float32, copy=True)
    canonical = np.zeros_like(positions)
    for row, value in enumerate(positions):
        valid = np.flatnonzero(skeleton_mask[row])
        if valid.size == 0:
            continue
        origin = value[int(valid[0]), 0].copy()
        centered = value - origin[None, None, :]
        shoulder = centered[valid, 8] - centered[valid, 4]
        shoulder_mean = shoulder.mean(axis=0)
        yaw = float(np.arctan2(float(shoulder_mean[2]), float(shoulder_mean[0])))
        c, sn = float(np.cos(yaw)), float(np.sin(yaw))
        rotation = np.asarray(((c, 0.0, sn), (0.0, 1.0, 0.0), (-sn, 0.0, c)), dtype=np.float32)
        rotated = centered @ rotation.T
        widths = np.linalg.norm(value[valid, 8] - value[valid, 4], axis=1)
        torsos = np.linalg.norm(value[valid, 2] - value[valid, 0], axis=1)
        scales = np.concatenate((widths, torsos))
        scales = scales[np.isfinite(scales) & (scales > 1e-6)]
        scale = float(np.median(scales)) if scales.size else 1.0
        canonical[row] = rotated / max(scale, 1e-6)
        canonical[row, ~skeleton_mask[row]] = 0.0
    flat = canonical.reshape(len(skeleton), FRAME_COUNT, 60)
    difference = np.zeros_like(flat)
    difference[:, 1:] = flat[:, 1:] - flat[:, :-1]
    bones = np.stack([canonical[:, :, child] - canonical[:, :, parent] for child, parent in BONE_EDGES], axis=2)
    rebuilt = np.concatenate((flat, difference, bones.reshape(len(skeleton), FRAME_COUNT, 57)), axis=-1)
    rebuilt *= skeleton_mask[..., None]
    return rebuilt.astype(np.float32, copy=False)


def _development_data(prepared_dir: Path, splits: SafeSenseSplits, *, body_canonicalization: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    skeleton, imu, mask, labels = _load_partition(prepared_dir, "train")
    if body_canonicalization:
        skeleton = _canonicalize_skeleton_features(skeleton, mask)
    indices = np.concatenate((splits.development_train, splits.development_valid))
    if np.any(indices >= len(labels)):
        raise SafeSenseError("development subjects are not contained in the prepared training partition")
    skeleton_mean, skeleton_std = _standard_stats(skeleton, splits.development_train, mask)
    imu_mean, imu_std = _standard_stats(imu, splits.development_train, mask)
    return skeleton, imu, mask, labels, {
        "skeleton_mean": skeleton_mean,
        "skeleton_std": skeleton_std,
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "skeleton_noise_scale": _raw_noise_scale(skeleton, splits.development_train, mask, 60),
        "imu_noise_scale": _raw_noise_scale(imu, splits.development_train, mask, 6),
    }


class ResidualDepthwiseTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * 2
        groups = max(1, min(16, channels))
        while channels % groups:
            groups -= 1
        self.depthwise = nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        update = self.depthwise(value)
        update = self.pointwise(update)
        return value + self.dropout(self.activation(self.norm(update)))


class SafeSenseTCNBranch(nn.Module):
    def __init__(self, input_size: int, width: int, dropout: float = 0.15, dilations: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.input = nn.Conv1d(input_size, width, 1)
        self.blocks = nn.Sequential(*(ResidualDepthwiseTCNBlock(width, dilation, dropout) for dilation in dilations))

    def forward(self, value: Tensor) -> Tensor:
        return self.blocks(self.input(value))


class ResidualTemporalDenoiser(nn.Module):
    """Multi-scale residual temporal cleaner for gated noisy-modality repair."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        groups = max(1, min(16, hidden))
        while hidden % groups:
            groups -= 1
        self.project_in = nn.Conv1d(channels * 3, hidden, 1)
        self.depthwise_short = nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden)
        self.depthwise_medium = nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2, groups=hidden)
        self.norm_short = nn.GroupNorm(groups, hidden)
        self.norm_medium = nn.GroupNorm(groups, hidden)
        self.project_out = nn.Conv1d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.tensor(-1.2))
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, value: Tensor) -> Tensor:
        smooth3 = nn.functional.avg_pool1d(value, 3, stride=1, padding=1)
        smooth7 = nn.functional.avg_pool1d(value, 7, stride=1, padding=3)
        hidden = self.project_in(torch.cat((value, smooth3, smooth7), dim=1))
        short = nn.functional.gelu(self.norm_short(self.depthwise_short(hidden)))
        medium = nn.functional.gelu(self.norm_medium(self.depthwise_medium(hidden + short)))
        update = self.project_out(hidden + short + medium)
        return value + torch.sigmoid(self.scale) * update


class TopologyGraphBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.self_linear = nn.Linear(channels, channels)
        self.neighbor_linear = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        neighbors = torch.einsum("ij,btjc->btic", adjacency, nodes)
        update = self.self_linear(nodes) + self.neighbor_linear(neighbors)
        return nodes + self.dropout(nn.functional.gelu(self.norm(update)))


class SkeletonTopologyBranch(nn.Module):
    """Explicit fixed-topology joint encoder feeding a compact temporal branch."""

    def __init__(self):
        super().__init__()
        adjacency = torch.eye(20, dtype=torch.float32)
        for child, parent in BONE_EDGES:
            adjacency[child, parent] = 1.0
            adjacency[parent, child] = 1.0
        degree = adjacency.sum(dim=1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        self.register_buffer("adjacency", adjacency)
        self.input = nn.Sequential(nn.Linear(6, 96), nn.LayerNorm(96), nn.GELU())
        self.blocks = nn.ModuleList((TopologyGraphBlock(96), TopologyGraphBlock(96), TopologyGraphBlock(96)))
        self.joint_attention = nn.Linear(96, 1)
        self.temporal = SafeSenseTCNBranch(96, 96, dilations=(1, 2, 4))

    def forward(self, skeleton: Tensor, mask: Tensor) -> Tensor:
        positions = skeleton[:, :, :60].reshape(*skeleton.shape[:2], 20, 3)
        motion = skeleton[:, :, 60:120].reshape(*skeleton.shape[:2], 20, 3)
        nodes = self.input(torch.cat((positions, motion), dim=-1))
        for block in self.blocks:
            nodes = block(nodes, self.adjacency)
        scores = self.joint_attention(nodes).squeeze(-1)
        weights = torch.softmax(scores, dim=2)
        pooled = (nodes * weights.unsqueeze(-1)).sum(dim=2)
        pooled = pooled * mask.unsqueeze(-1)
        return self.temporal(pooled.transpose(1, 2))


class SafeSense(nn.Module):
    def __init__(
        self,
        *,
        variant: str = "full",
        equal_weights: bool | None = None,
        single_modality: str | None = None,
        subject_style_training: bool = False,
    ):
        super().__init__()
        if equal_weights is not None or single_modality is not None:
            # Keep the original constructor spelling for callers of the smoke path.
            if equal_weights and single_modality is not None:
                raise ValueError("equal_weights and single_modality cannot both be set")
            variant = "equal" if equal_weights else (single_modality or "full")
        if variant not in SAFESENSE_VARIANTS:
            raise ValueError(f"variant must be one of {SAFESENSE_VARIANTS}")
        self.variant = variant
        self.subject_style_training = bool(subject_style_training)
        if self.subject_style_training and variant != "temporal":
            raise ValueError("subject-style training is supported only by the temporal variant")
        dilations = (1, 2, 4, 8) if variant == "aligned" else (1, 2, 4)
        self.skeleton_branch = SafeSenseTCNBranch(SKELETON_INPUTS, 96, dilations=dilations)
        self.imu_branch = SafeSenseTCNBranch(IMU_INPUTS, 48, dilations=dilations)
        self.skeleton_projection = nn.Linear(192, 96)
        self.imu_projection = nn.Linear(96, 96)
        self._last_expert_logits: tuple[Tensor, Tensor] | None = None
        self._last_lag_logits: Tensor | None = None
        self._last_temporal_lag_logits: Tensor | None = None
        self._last_lag_candidate_mask: Tensor | None = None
        self._last_temporal_embedding: Tensor | None = None
        self._last_denoised_inputs: tuple[Tensor, Tensor] | None = None
        self._last_correction_gate: Tensor | None = None
        self._last_effective_correction_gate: Tensor | None = None
        self._last_robust_topology_logits: Tensor | None = None
        if variant in ("classwise", "classprior", "aligned", "temporal", "robust"):
            self.skeleton_expert = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, NUM_CLASSES))
            self.imu_expert = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, NUM_CLASSES))
            if variant == "classwise":
                self.classwise_router = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, NUM_CLASSES * 2))
            else:
                self.classprior_router = nn.Parameter(torch.zeros(NUM_CLASSES, 2))
            if variant == "aligned":
                self.skeleton_alignment = nn.Conv1d(96, 32, 1, bias=False)
                self.imu_alignment = nn.Conv1d(48, 32, 1, bias=False)
                self.lag_bias = nn.Parameter(torch.zeros(len(LAG_SHIFTS)))
                with torch.no_grad():
                    self.lag_bias[_corrective_lag_target(0)] = 0.25
                self.reliability = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 2))
            if variant in ("temporal", "robust"):
                self.temporal_skeleton_query = nn.Conv1d(96, 64, 1, bias=False)
                self.temporal_imu_key = nn.Conv1d(48, 64, 1, bias=False)
                self.temporal_imu_value = nn.Conv1d(48, 96, 1)
                self.temporal_shift_bias = nn.Parameter(torch.zeros(len(LAG_SHIFTS)))
                with torch.no_grad():
                    self.temporal_shift_bias[_corrective_lag_target(0)] = 0.5
                self.temporal_lag_head = nn.Sequential(
                    nn.Linear(2 * len(LAG_SHIFTS), 64),
                    nn.GELU(),
                    nn.Linear(64, len(LAG_SHIFTS)),
                )
                nn.init.zeros_(self.temporal_lag_head[-1].weight)
                nn.init.zeros_(self.temporal_lag_head[-1].bias)
                self.temporal_reliability = nn.Sequential(
                    nn.Linear(384, 64),
                    nn.GELU(),
                    nn.Linear(64, 2),
                )
                nn.init.zeros_(self.temporal_reliability[-1].weight)
                nn.init.zeros_(self.temporal_reliability[-1].bias)
                self.temporal_reliability_strength = nn.Parameter(torch.tensor(-2.0))
                self.temporal_fusion_projection = nn.Sequential(
                    nn.Conv1d(384, 192, 1),
                    nn.GroupNorm(16, 192),
                    nn.GELU(),
                )
                self.temporal_attention_norm = nn.LayerNorm(192)
                self.temporal_attention = nn.MultiheadAttention(192, 4, dropout=0.1, batch_first=True)
                self.temporal_attention_dropout = nn.Dropout(0.1)
                self.temporal_ffn_norm = nn.LayerNorm(192)
                self.temporal_ffn = nn.Sequential(
                    nn.Linear(192, 384),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(384, 192),
                    nn.Dropout(0.1),
                )
                self.temporal_classifier = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, NUM_CLASSES))
                nn.init.normal_(self.temporal_classifier[-1].weight, std=1e-3)
                nn.init.zeros_(self.temporal_classifier[-1].bias)
                self.temporal_logit_scale = nn.Parameter(torch.tensor(0.25))
            if variant == "robust":
                self.robust_skeleton_denoiser = ResidualTemporalDenoiser(SKELETON_INPUTS, 128)
                self.robust_imu_denoiser = ResidualTemporalDenoiser(IMU_INPUTS, 64)
                self.robust_topology = SkeletonTopologyBranch()
                self.robust_topology_strength = nn.Parameter(torch.tensor(-1.4))
                self.robust_topology_classifier = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, NUM_CLASSES))
                nn.init.normal_(self.robust_topology_classifier[-1].weight, std=1e-3)
                nn.init.zeros_(self.robust_topology_classifier[-1].bias)
                self.robust_topology_logit_scale = nn.Parameter(torch.tensor(0.05))
                self.robust_quality_router = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 2))
                nn.init.zeros_(self.robust_quality_router[-1].weight)
                nn.init.zeros_(self.robust_quality_router[-1].bias)
                self.robust_quality_strength = nn.Parameter(torch.tensor(-1.5))
                self.robust_fault_penalty = nn.Parameter(torch.tensor(5.0))
                self.robust_correction_gate = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 2))
                nn.init.zeros_(self.robust_correction_gate[-1].weight)
                nn.init.constant_(self.robust_correction_gate[-1].bias, -2.5)
            self.classifier = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, NUM_CLASSES))
        else:
            self.gate = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 2))
            self.classifier = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, NUM_CLASSES))

    @staticmethod
    def _pool(sequence: Tensor, mask: Tensor) -> Tensor:
        valid = mask.unsqueeze(1).bool()
        masked = sequence.masked_fill(~valid, 0.0)
        count = valid.sum(dim=-1).clamp_min(1)
        mean = masked.sum(dim=-1) / count
        maximum = sequence.masked_fill(~valid, -torch.inf).amax(dim=-1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=1)

    @staticmethod
    def _modality_masks(mask: Tensor, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        if mask.ndim == 2 and tuple(mask.shape) == shape:
            return mask, mask
        if mask.ndim == 3 and tuple(mask.shape[:2]) == shape and mask.shape[-1] == 2:
            return mask[..., 0], mask[..., 1]
        raise ValueError("mask must have shape [B,T] or [B,T,2]")

    @staticmethod
    def _quality_features(skeleton: Tensor, imu: Tensor, skeleton_mask: Tensor, imu_mask: Tensor) -> Tensor:
        def roughness(value: Tensor, valid: Tensor, width: int) -> tuple[Tensor, Tensor]:
            base = value[:, :, :width]
            if base.shape[1] < 3:
                zeros = torch.zeros(base.shape[0], device=base.device, dtype=base.dtype)
                return zeros, zeros
            delta = base[:, 1:] - base[:, :-1]
            transition = (valid[:, 1:] > 0) & (valid[:, :-1] > 0)
            first = (delta.abs().mean(dim=-1) * transition).sum(dim=1) / transition.sum(dim=1).clamp_min(1)
            jerk = delta[:, 1:] - delta[:, :-1]
            triple = transition[:, 1:] & transition[:, :-1]
            second = (jerk.abs().mean(dim=-1) * triple).sum(dim=1) / triple.sum(dim=1).clamp_min(1)
            return torch.log1p(first), torch.log1p(second)

        skeleton_first, skeleton_second = roughness(skeleton, skeleton_mask, 60)
        imu_first, imu_second = roughness(imu, imu_mask, 6)
        return torch.stack((
            skeleton_mask.float().mean(dim=1), imu_mask.float().mean(dim=1),
            skeleton_first, imu_first, skeleton_second, imu_second,
        ), dim=1)

    def _align_imu(self, skeleton_sequence: Tensor, imu_sequence: Tensor, skeleton_mask: Tensor, imu_mask: Tensor) -> tuple[Tensor, Tensor]:
        skeleton_shared = nn.functional.normalize(self.skeleton_alignment(skeleton_sequence), dim=1)
        imu_shared = nn.functional.normalize(self.imu_alignment(imu_sequence), dim=1)
        shifted_sequences = []
        shifted_shared = []
        shifted_masks = []
        for shift in LAG_SHIFTS:
            shifted_sequences.append(_zero_padded_shift_tensor(imu_sequence, shift, time_dim=2))
            shifted_shared.append(_zero_padded_shift_tensor(imu_shared, shift, time_dim=2))
            shifted_masks.append(_zero_padded_shift_tensor(imu_mask, shift, time_dim=1))
        sequence_candidates = torch.stack(shifted_sequences, dim=1)
        shared_candidates = torch.stack(shifted_shared, dim=1)
        mask_candidates = torch.stack(shifted_masks, dim=1).float()
        overlap = skeleton_mask[:, None, :].float() * mask_candidates
        cosine = (skeleton_shared[:, None, :, :] * shared_candidates).sum(dim=2)
        overlap_count = overlap.sum(dim=-1)
        scores = (cosine * overlap).sum(dim=-1) / overlap_count.clamp_min(1.0)
        scores = torch.where(overlap_count > 0, scores, torch.zeros_like(scores))
        lag_logits = scores * 5.0 + self.lag_bias.unsqueeze(0)
        lag_weights = torch.softmax(lag_logits, dim=1)
        aligned_sequence = (sequence_candidates * lag_weights[:, :, None, None]).sum(dim=1)
        aligned_mask_score = (mask_candidates * lag_weights[:, :, None]).sum(dim=1)
        aligned_mask = (aligned_mask_score > 0.25).float()
        self._last_lag_logits = lag_logits
        return aligned_sequence, aligned_mask

    def _local_temporal_align_imu(
        self,
        skeleton_sequence: Tensor,
        imu_sequence: Tensor,
        skeleton_mask: Tensor,
        imu_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Locally align IMU features to the skeleton timeline over corrective shifts."""
        skeleton_query = nn.functional.normalize(self.temporal_skeleton_query(skeleton_sequence), dim=1)
        imu_key = nn.functional.normalize(self.temporal_imu_key(imu_sequence), dim=1)
        imu_value = self.temporal_imu_value(imu_sequence)
        key_candidates = []
        value_candidates = []
        mask_candidates = []
        for correction in LAG_SHIFTS:
            key_candidates.append(_zero_padded_shift_tensor(imu_key, correction, time_dim=2))
            value_candidates.append(_zero_padded_shift_tensor(imu_value, correction, time_dim=2))
            mask_candidates.append(_zero_padded_shift_tensor(imu_mask, correction, time_dim=1))
        candidate_keys = torch.stack(key_candidates, dim=1)
        candidate_values = torch.stack(value_candidates, dim=1)
        candidate_masks = torch.stack(mask_candidates, dim=1).float()
        raw_logits = (skeleton_query[:, None] * candidate_keys).sum(dim=2) * 6.0
        raw_logits = raw_logits + self.temporal_shift_bias[None, :, None]
        candidate_valid = candidate_masks > 0
        overlap = skeleton_mask[:, None, :].bool() & candidate_valid
        overlap_count = overlap.sum(dim=-1)
        mean_logits = (raw_logits * overlap.float()).sum(dim=-1) / overlap_count.clamp_min(1)
        mean_logits = torch.where(
            overlap_count > 0,
            mean_logits,
            self.temporal_shift_bias.unsqueeze(0).expand_as(mean_logits),
        )
        maximum_logits = raw_logits.masked_fill(~overlap, -torch.inf).amax(dim=-1)
        maximum_logits = torch.where(torch.isfinite(maximum_logits), maximum_logits, mean_logits)
        global_logits = mean_logits + self.temporal_lag_head(torch.cat((mean_logits, maximum_logits), dim=1))
        # The supervised global lag head is diagnostic/auxiliary. Actual alignment
        # remains driven by direct fine-grained cross-modal correlation so an
        # imperfect sequence-level lag prediction cannot override local evidence.
        local_logits = raw_logits.masked_fill(~candidate_valid, -1e4) + 0.5 * mean_logits[:, :, None]
        local_weights = torch.softmax(local_logits, dim=1)
        aligned_imu = (candidate_values * local_weights[:, :, None, :]).sum(dim=1)
        aligned_mask_score = (candidate_masks * local_weights).sum(dim=1)
        aligned_mask = (aligned_mask_score > 0.2).float()
        self._last_lag_logits = global_logits
        self._last_temporal_lag_logits = raw_logits.permute(0, 2, 1)
        self._last_lag_candidate_mask = candidate_valid.permute(0, 2, 1)
        return aligned_imu, aligned_mask

    def _temporal_fusion_logits(
        self,
        skeleton_sequence: Tensor,
        imu_sequence: Tensor,
        skeleton_mask: Tensor,
        imu_mask: Tensor,
    ) -> Tensor:
        aligned_imu, aligned_mask = self._local_temporal_align_imu(
            skeleton_sequence, imu_sequence, skeleton_mask, imu_mask
        )
        fused = torch.cat(
            (
                skeleton_sequence,
                aligned_imu,
                (skeleton_sequence - aligned_imu).abs(),
                skeleton_sequence * aligned_imu,
            ),
            dim=1,
        )
        tokens = self.temporal_fusion_projection(fused).transpose(1, 2)
        valid = (skeleton_mask > 0) | (aligned_mask > 0)
        safe_valid = valid.clone()
        all_missing = ~safe_valid.any(dim=1)
        if bool(all_missing.any()):
            safe_valid[all_missing, 0] = True
        normalized = self.temporal_attention_norm(tokens)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attended, _ = self.temporal_attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=~safe_valid,
                need_weights=False,
            )
        tokens = tokens + self.temporal_attention_dropout(attended)
        tokens = tokens + self.temporal_ffn(self.temporal_ffn_norm(tokens))
        pooled = self._pool(tokens.transpose(1, 2), valid.float())
        self._last_temporal_embedding = pooled
        return self.temporal_classifier(pooled)

    def forward(self, skeleton: Tensor, imu: Tensor, mask: Tensor | None = None, availability: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if skeleton.ndim != 3 or skeleton.shape[-1] != SKELETON_INPUTS:
            raise ValueError(f"skeleton must have shape [B,T,{SKELETON_INPUTS}]")
        if imu.ndim != 3 or imu.shape[-1] != IMU_INPUTS:
            raise ValueError(f"imu must have shape [B,T,{IMU_INPUTS}]")
        if mask is None:
            mask = torch.ones(skeleton.shape[:2], device=skeleton.device)
        skeleton_mask, imu_mask = self._modality_masks(mask, tuple(skeleton.shape[:2]))
        if availability is None:
            availability = torch.stack((skeleton_mask.any(dim=1), imu_mask.any(dim=1)), dim=1).float()
        quality_features = None
        if self.variant in ("temporal", "robust"):
            skeleton, imu = _sanitize_masked_motion_features(skeleton, imu, skeleton_mask, imu_mask)
            skeleton = _repair_internal_gaps(skeleton, skeleton_mask)
            imu = _repair_internal_gaps(imu, imu_mask)
        denoised_skeleton = denoised_imu = None
        robust_topology_logits = None
        if self.variant == "robust":
            quality_features = self._quality_features(skeleton, imu, skeleton_mask, imu_mask)
            denoised_skeleton = self.robust_skeleton_denoiser(skeleton.transpose(1, 2)).transpose(1, 2)
            denoised_imu = self.robust_imu_denoiser(imu.transpose(1, 2)).transpose(1, 2)
            correction_gate = torch.sigmoid(self.robust_correction_gate(quality_features))
            effective_gate = correction_gate * torch.sigmoid(
                ROBUST_EFFECTIVE_GATE_SHARPNESS * (correction_gate - ROBUST_EFFECTIVE_GATE_CENTER)
            )
            self._last_correction_gate = correction_gate
            self._last_effective_correction_gate = effective_gate
            self._last_denoised_inputs = (denoised_skeleton, denoised_imu)
            # Keep the frozen temporal backbone input-identical to the temporal control.
            # Denoising/topology operate only as a sidecar expert so lag/alignment cannot
            # regress because the robustness adapter rewrote shared temporal features.
        else:
            self._last_denoised_inputs = None
            self._last_correction_gate = None
            self._last_effective_correction_gate = None
        skeleton_sequence = self.skeleton_branch(skeleton.transpose(1, 2))
        imu_sequence = self.imu_branch(imu.transpose(1, 2))
        if self.variant == "robust":
            if denoised_skeleton is None or self._last_correction_gate is None:
                raise SafeSenseError("robust specialist inputs were not produced")
            topology_sequence = self.robust_topology(denoised_skeleton, skeleton_mask)
            if self._last_effective_correction_gate is None:
                raise SafeSenseError("robust effective correction gate was not produced")
            topology_embedding = self._pool(topology_sequence, skeleton_mask)
            robust_topology_logits = self.robust_topology_classifier(topology_embedding)
            self._last_robust_topology_logits = robust_topology_logits
        if self.variant == "aligned":
            imu_sequence, imu_mask = self._align_imu(skeleton_sequence, imu_sequence, skeleton_mask, imu_mask)
        elif self.variant not in ("temporal", "robust"):
            self._last_lag_logits = None
            self._last_temporal_lag_logits = None
            self._last_lag_candidate_mask = None
            self._last_temporal_embedding = None
        skeleton_embedding = self.skeleton_projection(self._pool(skeleton_sequence, skeleton_mask))
        imu_embedding = self.imu_projection(self._pool(imu_sequence, imu_mask))
        availability = availability.float().clamp(0, 1)
        skeleton_embedding = skeleton_embedding * availability[:, 0:1]
        imu_embedding = imu_embedding * availability[:, 1:2]
        gate_input = torch.cat((skeleton_embedding, imu_embedding, (skeleton_embedding - imu_embedding).abs(), skeleton_embedding * imu_embedding), dim=1)
        if self.variant == "aligned":
            reliability_logits = self.reliability(gate_input).masked_fill(availability <= 0, -1e9)
            weights = torch.softmax(reliability_logits, dim=1)
        elif self.variant in ("temporal", "robust"):
            reliability_logits = self.temporal_reliability(gate_input)
            fault_penalty = None
            if self.variant == "robust":
                if quality_features is None or self._last_effective_correction_gate is None:
                    raise SafeSenseError("robust quality/fault features were not produced")
                specialist_strength = self._last_effective_correction_gate.mean(dim=1, keepdim=True)
                reliability_logits = reliability_logits + specialist_strength * torch.sigmoid(self.robust_quality_strength) * self.robust_quality_router(quality_features)
                fault_penalty = nn.functional.softplus(self.robust_fault_penalty)
                reliability_logits = reliability_logits - fault_penalty * self._last_effective_correction_gate
            reliability_logits = reliability_logits.masked_fill(availability <= 0, -1e9)
            reliability_strength = torch.sigmoid(self.temporal_reliability_strength)
            if self.variant == "robust" and fault_penalty is not None and self._last_effective_correction_gate is not None:
                fault_availability = availability * torch.exp(-fault_penalty * self._last_effective_correction_gate)
                base_weights = fault_availability / fault_availability.sum(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                base_weights = availability / availability.sum(dim=1, keepdim=True).clamp_min(1.0)
            adaptive_weights = torch.softmax(reliability_logits, dim=1)
            weights = (1.0 - reliability_strength) * base_weights + reliability_strength * adaptive_weights
        elif self.variant in ("equal", "classwise", "classprior"):
            weights = availability.clone()
        elif self.variant in ("skeleton", "imu"):
            weights = torch.zeros_like(availability)
            requested = 0 if self.variant == "skeleton" else 1
            weights[:, requested] = availability[:, requested]
        else:
            gate_logits = self.gate(gate_input).masked_fill(availability <= 0, -1e9)
            weights = torch.softmax(gate_logits, dim=1)
        available_count = weights.sum(dim=1, keepdim=True)
        normalized = weights / available_count.clamp_min(1e-6)
        # A missing single requested modality stays missing; the full model uses a
        # neutral finite fallback only when neither modality is available.
        if self.variant in ("skeleton", "imu"):
            weights = torch.where(available_count > 0, normalized, torch.zeros_like(weights))
        else:
            weights = torch.where(available_count > 0, normalized, torch.full_like(weights, 0.5))
        embedding = skeleton_embedding * weights[:, 0:1] + imu_embedding * weights[:, 1:2]
        if self.variant in ("classwise", "classprior", "aligned", "temporal", "robust"):
            joint_logits = self.classifier(embedding)
            skeleton_logits = self.skeleton_expert(skeleton_embedding) * availability[:, 0:1]
            imu_logits = self.imu_expert(imu_embedding) * availability[:, 1:2]
            if self.variant == "classwise":
                router_logits = self.classwise_router(gate_input).reshape(-1, NUM_CLASSES, 2)
            elif self.variant == "aligned":
                router_logits = self.classprior_router.unsqueeze(0) + reliability_logits[:, None, :]
            elif self.variant in ("temporal", "robust"):
                learned_reliability = reliability_logits
                router_logits = self.classprior_router.unsqueeze(0) + reliability_strength * learned_reliability[:, None, :]
                if self.variant == "robust" and fault_penalty is not None and self._last_effective_correction_gate is not None:
                    router_logits = router_logits - fault_penalty * self._last_effective_correction_gate[:, None, :]
            else:
                router_logits = self.classprior_router.unsqueeze(0).expand(skeleton.shape[0], -1, -1)
            router_logits = router_logits.masked_fill(availability[:, None, :] <= 0, -1e9)
            class_weights = torch.softmax(router_logits, dim=-1)
            expert_logits = skeleton_logits * class_weights[:, :, 0] + imu_logits * class_weights[:, :, 1]
            self._last_expert_logits = (skeleton_logits, imu_logits)
            baseline_logits = 0.5 * (joint_logits + expert_logits)
            if self.variant in ("temporal", "robust"):
                temporal_logits = self._temporal_fusion_logits(
                    skeleton_sequence, imu_sequence, skeleton_mask, imu_mask
                )
                baseline_logits = baseline_logits + torch.tanh(self.temporal_logit_scale) * temporal_logits
            if self.variant == "robust":
                if robust_topology_logits is None:
                    raise SafeSenseError("robust topology logits were not produced")
                topology_scale = torch.tanh(self.robust_topology_logit_scale) * (1.0 + 0.1 * torch.tanh(self.robust_topology_strength))
                skeleton_count = skeleton_mask.float().sum(dim=1, keepdim=True)
                imu_count = imu_mask.float().sum(dim=1, keepdim=True)
                relative_skeleton_coverage = torch.where(
                    imu_count > 0,
                    (skeleton_count / imu_count.clamp_min(1.0)).clamp(max=1.0),
                    torch.ones_like(skeleton_count),
                )
                topology_coverage = (
                    torch.ones_like(relative_skeleton_coverage)
                    if self.training else relative_skeleton_coverage.square()
                )
                baseline_logits = baseline_logits + availability[:, 0:1] * topology_coverage * topology_scale * robust_topology_logits
            return baseline_logits, class_weights.mean(dim=1)
        return self.classifier(embedding), weights

    @property
    def expert_logits(self) -> tuple[Tensor, Tensor] | None:
        return self._last_expert_logits

    @property
    def lag_logits(self) -> Tensor | None:
        return self._last_lag_logits

    @property
    def temporal_lag_logits(self) -> Tensor | None:
        return self._last_temporal_lag_logits

    @property
    def lag_candidate_mask(self) -> Tensor | None:
        return self._last_lag_candidate_mask

    @property
    def temporal_embedding(self) -> Tensor | None:
        return self._last_temporal_embedding

    @property
    def denoised_inputs(self) -> tuple[Tensor, Tensor] | None:
        return self._last_denoised_inputs

    @property
    def correction_gate(self) -> Tensor | None:
        return self._last_correction_gate

    @property
    def effective_correction_gate(self) -> Tensor | None:
        return self._last_effective_correction_gate

    @property
    def robust_topology_logits(self) -> Tensor | None:
        return self._last_robust_topology_logits


class EfficientTemporalReducer(nn.Module):
    """Reduce 128-frame modality streams to 32 tokens with depthwise separable convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, min(16, channels))
        while channels % groups:
            groups -= 1
        self.steps = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 5, stride=2, padding=2, groups=channels),
                nn.Conv1d(channels, channels, 1),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
            )
            for _ in range(2)
        ])

    def forward(self, sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        valid = mask.float()
        for step in self.steps:
            sequence = step(sequence)
            valid = nn.functional.max_pool1d(valid.unsqueeze(1), 3, stride=2, padding=1).squeeze(1)
            valid = (valid > 0).float()
            sequence = sequence * valid.unsqueeze(1)
        return sequence, valid


class FactorizedTimeModalityBlock(nn.Module):
    """Alternate efficient attention over time and the two sensor modalities."""

    def __init__(self, width: int = 96, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.modality_norm = nn.LayerNorm(width)
        self.modality_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _safe_valid(valid: Tensor) -> Tensor:
        safe = valid.clone()
        missing = ~safe.any(dim=1)
        if bool(missing.any()):
            safe[missing, 0] = True
        return safe

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        batch, modalities, steps, width = tokens.shape
        temporal = tokens.reshape(batch * modalities, steps, width)
        temporal_valid = valid.reshape(batch * modalities, steps)
        normalized = self.temporal_norm(temporal)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            temporal_update, _ = self.temporal_attention(
                normalized, normalized, normalized,
                key_padding_mask=~self._safe_valid(temporal_valid),
                need_weights=False,
            )
        temporal = temporal + self.dropout(temporal_update)
        temporal = temporal * temporal_valid.unsqueeze(-1)
        tokens = temporal.reshape(batch, modalities, steps, width)

        modality = tokens.permute(0, 2, 1, 3).reshape(batch * steps, modalities, width)
        modality_valid = valid.permute(0, 2, 1).reshape(batch * steps, modalities)
        normalized = self.modality_norm(modality)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            modality_update, _ = self.modality_attention(
                normalized, normalized, normalized,
                key_padding_mask=~self._safe_valid(modality_valid),
                need_weights=False,
            )
        modality = modality + self.dropout(modality_update)
        modality = modality * modality_valid.unsqueeze(-1)
        tokens = modality.reshape(batch, steps, modalities, width).permute(0, 2, 1, 3)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens * valid.unsqueeze(-1)


class SafeSenseFactorized(nn.Module):
    """Graph-aware dual-modality model with factorized time/modality attention."""

    variant = "factorized"
    subject_style_training = True

    def __init__(self, width: int = 96, blocks: int = 2):
        super().__init__()
        self.skeleton_flat = SafeSenseTCNBranch(SKELETON_INPUTS, 64)
        self.skeleton_graph = SkeletonTopologyBranch()
        self.skeleton_fuse = nn.Sequential(
            nn.Conv1d(160, width, 1),
            nn.GroupNorm(16, width),
            nn.GELU(),
        )
        self.imu_branch = SafeSenseTCNBranch(IMU_INPUTS, 64)
        self.imu_project = nn.Sequential(
            nn.Conv1d(64, width, 1),
            nn.GroupNorm(16, width),
            nn.GELU(),
        )
        self.skeleton_reduce = EfficientTemporalReducer(width)
        self.imu_reduce = EfficientTemporalReducer(width)
        self.factorized_blocks = nn.ModuleList(FactorizedTimeModalityBlock(width) for _ in range(blocks))
        self.classifier = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, NUM_CLASSES))
        self.skeleton_expert = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, NUM_CLASSES))
        self.imu_expert = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, NUM_CLASSES))
        self._last_expert_logits: tuple[Tensor, Tensor] | None = None
        self._last_alignment_embeddings: tuple[Tensor, Tensor] | None = None

    @staticmethod
    def _masked_mean(tokens: Tensor, valid: Tensor) -> Tensor:
        weights = valid.float().unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        skeleton: Tensor,
        imu: Tensor,
        mask: Tensor | None = None,
        availability: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if skeleton.ndim != 3 or skeleton.shape[-1] != SKELETON_INPUTS:
            raise ValueError(f"skeleton must have shape [B,T,{SKELETON_INPUTS}]")
        if imu.ndim != 3 or imu.shape[-1] != IMU_INPUTS:
            raise ValueError(f"imu must have shape [B,T,{IMU_INPUTS}]")
        if mask is None:
            mask = torch.ones(skeleton.shape[:2], device=skeleton.device)
        skeleton_mask, imu_mask = SafeSense._modality_masks(mask, tuple(skeleton.shape[:2]))
        if availability is None:
            availability = torch.stack((skeleton_mask.any(dim=1), imu_mask.any(dim=1)), dim=1).float()
        availability = availability.float().clamp(0, 1)
        skeleton, imu = _sanitize_masked_motion_features(skeleton, imu, skeleton_mask, imu_mask)
        skeleton = _repair_internal_gaps(skeleton, skeleton_mask)
        imu = _repair_internal_gaps(imu, imu_mask)

        flat_sequence = self.skeleton_flat(skeleton.transpose(1, 2))
        graph_sequence = self.skeleton_graph(skeleton, skeleton_mask)
        skeleton_sequence = self.skeleton_fuse(torch.cat((flat_sequence, graph_sequence), dim=1))
        imu_sequence = self.imu_project(self.imu_branch(imu.transpose(1, 2)))
        skeleton_sequence, skeleton_mask = self.skeleton_reduce(skeleton_sequence, skeleton_mask)
        imu_sequence, imu_mask = self.imu_reduce(imu_sequence, imu_mask)
        skeleton_mask = skeleton_mask * availability[:, 0:1]
        imu_mask = imu_mask * availability[:, 1:2]

        tokens = torch.stack((skeleton_sequence.transpose(1, 2), imu_sequence.transpose(1, 2)), dim=1)
        valid = torch.stack((skeleton_mask.bool(), imu_mask.bool()), dim=1)
        for block in self.factorized_blocks:
            tokens = block(tokens, valid)

        skeleton_tokens, imu_tokens = tokens[:, 0], tokens[:, 1]
        skeleton_embedding = SafeSense._pool(skeleton_tokens.transpose(1, 2), skeleton_mask)
        imu_embedding = SafeSense._pool(imu_tokens.transpose(1, 2), imu_mask)
        skeleton_logits = self.skeleton_expert(skeleton_embedding) * availability[:, 0:1]
        imu_logits = self.imu_expert(imu_embedding) * availability[:, 1:2]
        self._last_expert_logits = (skeleton_logits, imu_logits)
        self._last_alignment_embeddings = (
            self._masked_mean(skeleton_tokens, skeleton_mask),
            self._masked_mean(imu_tokens, imu_mask),
        )

        modality_count = valid.float().sum(dim=1)
        fused_tokens = (tokens * valid.unsqueeze(-1)).sum(dim=1) / modality_count.unsqueeze(-1).clamp_min(1.0)
        fused_mask = valid.any(dim=1).float()
        fused_embedding = SafeSense._pool(fused_tokens.transpose(1, 2), fused_mask)
        logits = self.classifier(fused_embedding)
        weights = availability / availability.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = torch.where(availability.sum(dim=1, keepdim=True) > 0, weights, torch.full_like(weights, 0.5))
        return logits, weights

    @property
    def expert_logits(self) -> tuple[Tensor, Tensor] | None:
        return self._last_expert_logits

    @property
    def alignment_embeddings(self) -> tuple[Tensor, Tensor] | None:
        return self._last_alignment_embeddings



class SafeSenseConditionalEnsemble(nn.Module):
    """Run a second frozen robust expert only for detected noise or missing modality."""

    def __init__(
        self,
        primary: nn.Module,
        auxiliary: nn.Module,
        *,
        weights: Sequence[float] = SAFESENSE_CONDITIONAL_WEIGHTS,
        gate_threshold: float = SAFESENSE_CONDITIONAL_GATE_THRESHOLD,
    ):
        super().__init__()
        if len(weights) != 2:
            raise ValueError("conditional ensemble requires exactly two weights")
        weight_array = np.asarray(tuple(weights), dtype=np.float32)
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array <= 0):
            raise ValueError("conditional ensemble weights must be finite and strictly positive")
        if not np.isfinite(gate_threshold) or not 0.0 <= gate_threshold <= 1.0:
            raise ValueError("conditional ensemble gate threshold must be in [0, 1]")
        weight_array = weight_array / weight_array.sum()
        self.primary = primary
        self.auxiliary = auxiliary
        self.register_buffer("ensemble_weights", torch.from_numpy(weight_array))
        self.gate_threshold = float(gate_threshold)
        self._last_auxiliary_route: Tensor | None = None

    @staticmethod
    def _modality_masks(mask: Tensor, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        if mask.ndim == 2 and tuple(mask.shape) == shape:
            return mask, mask
        if mask.ndim == 3 and tuple(mask.shape[:2]) == shape and mask.shape[-1] == 2:
            return mask[..., 0], mask[..., 1]
        raise ValueError("mask must have shape [B,T] or [B,T,2]")

    @staticmethod
    def _logits(output: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        return output[0] if isinstance(output, (tuple, list)) else output

    def forward(
        self,
        skeleton: Tensor,
        imu: Tensor,
        mask: Tensor,
        availability: Tensor | None = None,
    ) -> Tensor:
        primary_output = self.primary(skeleton, imu, mask, availability) if availability is not None else self.primary(skeleton, imu, mask)
        primary_logits = self._logits(primary_output)
        effective_gate = getattr(self.primary, "effective_correction_gate", None)
        if effective_gate is None:
            raise SafeSenseError("conditional ensemble primary model did not expose an effective correction gate")
        skeleton_mask, imu_mask = self._modality_masks(mask, tuple(skeleton.shape[:2]))
        skeleton_available = skeleton_mask.bool().any(dim=1)
        imu_available = imu_mask.bool().any(dim=1)
        both_available = skeleton_available & imu_available
        masks_match = skeleton_mask.bool().eq(imu_mask.bool()).all(dim=1)
        fault_detected = effective_gate.amax(dim=1) > self.gate_threshold
        route = (~both_available) | (masks_match & fault_detected)
        self._last_auxiliary_route = route.detach()

        primary_log_prob = nn.functional.log_softmax(primary_logits, dim=1)
        if not bool(route.any()):
            return primary_log_prob

        routed_availability = availability[route] if availability is not None else None
        auxiliary_output = (
            self.auxiliary(skeleton[route], imu[route], mask[route], routed_availability)
            if routed_availability is not None
            else self.auxiliary(skeleton[route], imu[route], mask[route])
        )
        auxiliary_logits = self._logits(auxiliary_output)
        auxiliary_log_prob = nn.functional.log_softmax(auxiliary_logits, dim=1)
        weights = self.ensemble_weights.to(dtype=primary_log_prob.dtype)
        blended = torch.logsumexp(
            torch.stack((
                primary_log_prob[route] + weights[0].log(),
                auxiliary_log_prob + weights[1].log(),
            ), dim=0),
            dim=0,
        )
        output = primary_log_prob.clone()
        output[route] = blended
        return output

    @property
    def auxiliary_route(self) -> Tensor | None:
        return self._last_auxiliary_route


def load_safesense_conditional_ensemble(
    primary_checkpoint: str | Path,
    auxiliary_checkpoint: str | Path,
    *,
    weights: Sequence[float] = SAFESENSE_CONDITIONAL_WEIGHTS,
    gate_threshold: float = SAFESENSE_CONDITIONAL_GATE_THRESHOLD,
) -> SafeSenseConditionalEnsemble:
    """Load the frozen seed-11/22 robust experts used by the conditional route."""
    models = []
    for checkpoint in (primary_checkpoint, auxiliary_checkpoint):
        model = SafeSense(variant="robust")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
    return SafeSenseConditionalEnsemble(
        models[0], models[1], weights=weights, gate_threshold=gate_threshold
    )


class SafeSenseBranchExpert(nn.Module):
    """Exact skeleton/IMU expert subset extracted from a frozen robust SafeSense checkpoint."""

    def __init__(self):
        super().__init__()
        self.skeleton_branch = SafeSenseTCNBranch(SKELETON_INPUTS, 96, dilations=(1, 2, 4))
        self.imu_branch = SafeSenseTCNBranch(IMU_INPUTS, 48, dilations=(1, 2, 4))
        self.skeleton_projection = nn.Linear(192, 96)
        self.imu_projection = nn.Linear(96, 96)
        self.skeleton_expert = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, NUM_CLASSES))
        self.imu_expert = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, NUM_CLASSES))

    @classmethod
    def from_robust_state_dict(cls, state: dict[str, Tensor]) -> "SafeSenseBranchExpert":
        model = cls()
        for name, module in (
            ("skeleton_branch", model.skeleton_branch),
            ("imu_branch", model.imu_branch),
            ("skeleton_projection", model.skeleton_projection),
            ("imu_projection", model.imu_projection),
            ("skeleton_expert", model.skeleton_expert),
            ("imu_expert", model.imu_expert),
        ):
            prefix = f"{name}."
            subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            module.load_state_dict(subset)
        return model

    @classmethod
    def from_robust_checkpoint(cls, checkpoint: str | Path) -> "SafeSenseBranchExpert":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return cls.from_robust_state_dict(state)

    def forward_preprocessed(
        self,
        skeleton: Tensor,
        imu: Tensor,
        skeleton_mask: Tensor,
        imu_mask: Tensor,
        use_skeleton: Tensor,
    ) -> Tensor:
        output = skeleton.new_zeros((len(skeleton), NUM_CLASSES))
        if bool(use_skeleton.any()):
            selected = use_skeleton
            sequence = self.skeleton_branch(skeleton[selected].transpose(1, 2))
            embedding = self.skeleton_projection(SafeSense._pool(sequence, skeleton_mask[selected]))
            available = skeleton_mask[selected].bool().any(dim=1).float()
            output[selected] = self.skeleton_expert(embedding) * available[:, None]
        use_imu = ~use_skeleton
        if bool(use_imu.any()):
            sequence = self.imu_branch(imu[use_imu].transpose(1, 2))
            embedding = self.imu_projection(SafeSense._pool(sequence, imu_mask[use_imu]))
            available = imu_mask[use_imu].bool().any(dim=1).float()
            output[use_imu] = self.imu_expert(embedding) * available[:, None]
        return output

    def forward(self, skeleton: Tensor, imu: Tensor, mask: Tensor, use_skeleton: Tensor) -> Tensor:
        skeleton_mask, imu_mask = SafeSenseEarlyExitEnsemble._modality_masks(mask, tuple(skeleton.shape[:2]))
        skeleton, imu = _sanitize_masked_motion_features(skeleton, imu, skeleton_mask, imu_mask)
        skeleton = _repair_internal_gaps(skeleton, skeleton_mask)
        imu = _repair_internal_gaps(imu, imu_mask)
        return self.forward_preprocessed(skeleton, imu, skeleton_mask, imu_mask, use_skeleton)


class SafeSenseEarlyExitEnsemble(nn.Module):
    """Use full seed-11 SafeSense on clean-like inputs and tiny branch experts on routed faults."""

    def __init__(
        self,
        primary: SafeSense,
        auxiliary: SafeSenseBranchExpert,
        *,
        primary_weight: float = SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT,
        roughness_threshold: float = SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD,
    ):
        super().__init__()
        if not 0.0 < primary_weight < 1.0:
            raise ValueError("early-exit primary weight must be in (0, 1)")
        if not np.isfinite(roughness_threshold) or roughness_threshold <= 0:
            raise ValueError("early-exit roughness threshold must be finite and positive")
        self.primary = primary
        self.auxiliary = auxiliary
        self.primary_weight = float(primary_weight)
        self.roughness_threshold = float(roughness_threshold)
        self._last_auxiliary_route: Tensor | None = None
        self._last_use_skeleton: Tensor | None = None
        self._last_roughness: Tensor | None = None

    @staticmethod
    def _modality_masks(mask: Tensor, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        if mask.ndim == 2 and tuple(mask.shape) == shape:
            return mask, mask
        if mask.ndim == 3 and tuple(mask.shape[:2]) == shape and mask.shape[-1] == 2:
            return mask[..., 0], mask[..., 1]
        raise ValueError("mask must have shape [B,T] or [B,T,2]")

    @staticmethod
    def _sparse_roughness(value: Tensor, valid: Tensor, channel: int) -> Tensor:
        sampled = value[:, :, channel]
        if sampled.shape[1] < 2:
            return value.new_zeros(value.shape[0])
        delta = (sampled[:, 1:] - sampled[:, :-1]).abs()
        transition = valid[:, 1:].bool() & valid[:, :-1].bool()
        return (delta * transition).sum(dim=1) / transition.sum(dim=1).clamp_min(1)

    def _primary_expert_logits(
        self,
        skeleton: Tensor,
        imu: Tensor,
        skeleton_mask: Tensor,
        imu_mask: Tensor,
        use_skeleton: Tensor,
    ) -> Tensor:
        output = skeleton.new_zeros((len(skeleton), NUM_CLASSES))
        if bool(use_skeleton.any()):
            selected = use_skeleton
            sequence = self.primary.skeleton_branch(skeleton[selected].transpose(1, 2))
            embedding = self.primary.skeleton_projection(SafeSense._pool(sequence, skeleton_mask[selected]))
            available = skeleton_mask[selected].bool().any(dim=1).float()
            output[selected] = self.primary.skeleton_expert(embedding) * available[:, None]
        use_imu = ~use_skeleton
        if bool(use_imu.any()):
            sequence = self.primary.imu_branch(imu[use_imu].transpose(1, 2))
            embedding = self.primary.imu_projection(SafeSense._pool(sequence, imu_mask[use_imu]))
            available = imu_mask[use_imu].bool().any(dim=1).float()
            output[use_imu] = self.primary.imu_expert(embedding) * available[:, None]
        return output

    def forward(
        self,
        skeleton: Tensor,
        imu: Tensor,
        mask: Tensor,
        availability: Tensor | None = None,
    ) -> Tensor:
        skeleton_mask, imu_mask = self._modality_masks(mask, tuple(skeleton.shape[:2]))
        skeleton_available = skeleton_mask.bool().any(dim=1)
        imu_available = imu_mask.bool().any(dim=1)
        both_available = skeleton_available & imu_available
        masks_match = skeleton_mask.bool().eq(imu_mask.bool()).all(dim=1)
        skeleton_roughness = self._sparse_roughness(
            skeleton, skeleton_mask, SAFESENSE_EARLY_EXIT_SKELETON_CHANNEL
        )
        imu_roughness = self._sparse_roughness(imu, imu_mask, SAFESENSE_EARLY_EXIT_IMU_CHANNEL)
        roughness = torch.stack((skeleton_roughness, imu_roughness), dim=1)
        route = (~both_available) | (masks_match & (roughness.amax(dim=1) > self.roughness_threshold))
        use_skeleton = torch.where(
            skeleton_available & ~imu_available,
            torch.ones_like(skeleton_available),
            torch.where(
                imu_available & ~skeleton_available,
                torch.zeros_like(skeleton_available),
                skeleton_roughness <= imu_roughness,
            ),
        )
        self._last_auxiliary_route = route.detach()
        self._last_use_skeleton = use_skeleton.detach()
        self._last_roughness = roughness.detach()

        probabilities = skeleton.new_full((len(skeleton), NUM_CLASSES), 1.0 / NUM_CLASSES)
        fast_path = ~route
        if bool(fast_path.any()):
            fast_availability = availability[fast_path] if availability is not None else None
            primary_output = (
                self.primary(skeleton[fast_path], imu[fast_path], mask[fast_path], fast_availability)
                if fast_availability is not None
                else self.primary(skeleton[fast_path], imu[fast_path], mask[fast_path])
            )
            primary_logits = primary_output[0] if isinstance(primary_output, (tuple, list)) else primary_output
            probabilities[fast_path] = torch.softmax(primary_logits, dim=1)

        routed_available = route & (skeleton_available | imu_available)
        if bool(routed_available.any()):
            selected = routed_available
            clean_skeleton, clean_imu = _sanitize_masked_motion_features(
                skeleton[selected], imu[selected], skeleton_mask[selected], imu_mask[selected]
            )
            clean_skeleton = _repair_internal_gaps(clean_skeleton, skeleton_mask[selected])
            clean_imu = _repair_internal_gaps(clean_imu, imu_mask[selected])
            selected_modality = use_skeleton[selected]
            primary_logits = self._primary_expert_logits(
                clean_skeleton, clean_imu, skeleton_mask[selected], imu_mask[selected], selected_modality
            )
            auxiliary_logits = self.auxiliary.forward_preprocessed(
                clean_skeleton, clean_imu, skeleton_mask[selected], imu_mask[selected], selected_modality
            )
            probabilities[selected] = (
                self.primary_weight * torch.softmax(primary_logits, dim=1)
                + (1.0 - self.primary_weight) * torch.softmax(auxiliary_logits, dim=1)
            )
        return probabilities.clamp_min(1e-12).log()

    @property
    def auxiliary_route(self) -> Tensor | None:
        return self._last_auxiliary_route

    @property
    def use_skeleton(self) -> Tensor | None:
        return self._last_use_skeleton

    @property
    def roughness(self) -> Tensor | None:
        return self._last_roughness


def load_safesense_early_exit_ensemble(
    primary_checkpoint: str | Path,
    auxiliary_robust_checkpoint: str | Path,
    *,
    primary_weight: float = SAFESENSE_EARLY_EXIT_PRIMARY_WEIGHT,
    roughness_threshold: float = SAFESENSE_EARLY_EXIT_ROUGHNESS_THRESHOLD,
) -> SafeSenseEarlyExitEnsemble:
    primary = SafeSense(variant="robust")
    primary.load_state_dict(torch.load(primary_checkpoint, map_location="cpu", weights_only=True))
    primary.eval()
    auxiliary = SafeSenseBranchExpert.from_robust_checkpoint(auxiliary_robust_checkpoint).eval()
    return SafeSenseEarlyExitEnsemble(
        primary, auxiliary, primary_weight=primary_weight, roughness_threshold=roughness_threshold
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _loader(dataset: SafeSenseDataset, config: SafeSenseTrainingConfig, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _expert_auxiliary_loss(experts: tuple[Tensor, Tensor], labels: Tensor) -> Tensor:
    return 0.5 * sum(nn.functional.cross_entropy(expert, labels) for expert in experts)


def _temporal_lag_loss(
    model: SafeSense,
    lag_targets: Tensor,
    lag_supervision: Tensor,
    mask: Tensor,
) -> Tensor:
    global_logits = model.lag_logits
    temporal_logits = model.temporal_lag_logits
    candidate_mask = model.lag_candidate_mask
    if global_logits is None or temporal_logits is None or candidate_mask is None:
        raise SafeSenseError("temporal lag outputs were not produced")
    if not bool(lag_supervision.any()):
        return global_logits.sum() * 0.0
    shifted = lag_targets != _corrective_lag_target(0)
    sample_weights = torch.where(shifted, torch.ones_like(lag_targets, dtype=torch.float32), torch.full_like(lag_targets, 0.25, dtype=torch.float32))
    sample_weights = sample_weights * lag_supervision.float()
    global_ce = nn.functional.cross_entropy(global_logits, lag_targets, reduction="none")
    global_loss = (global_ce * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)

    batch_size, frame_count, _ = temporal_logits.shape
    target_grid = lag_targets[:, None].expand(batch_size, frame_count)
    target_candidate = candidate_mask.gather(2, target_grid.unsqueeze(-1)).squeeze(-1)
    skeleton_mask, _ = SafeSense._modality_masks(mask, (batch_size, frame_count))
    local_valid = target_candidate & skeleton_mask.bool() & lag_supervision[:, None]
    local_ce = nn.functional.cross_entropy(
        temporal_logits.reshape(-1, len(LAG_SHIFTS)),
        target_grid.reshape(-1),
        reduction="none",
    ).reshape(batch_size, frame_count)
    local_weights = local_valid.float() * sample_weights[:, None]
    local_loss = (local_ce * local_weights).sum() / local_weights.sum().clamp_min(1.0)
    return 0.75 * local_loss + 0.25 * global_loss


def _temporal_consistency_loss(clean_logits: Tensor, corrupted_logits: Tensor) -> Tensor:
    temperature = TEMPORAL_CONSISTENCY_TEMPERATURE
    target = torch.softmax(clean_logits.detach() / temperature, dim=1)
    prediction = torch.log_softmax(corrupted_logits / temperature, dim=1)
    return nn.functional.kl_div(prediction, target, reduction="batchmean") * (temperature * temperature)


def _robust_reconstruction_loss(
    denoised: tuple[Tensor, Tensor],
    clean_skeleton: Tensor,
    clean_imu: Tensor,
    clean_mask: Tensor,
    supervision: Tensor,
) -> Tensor:
    skeleton_mask, imu_mask = SafeSense._modality_masks(clean_mask, tuple(clean_skeleton.shape[:2]))
    total = clean_skeleton.sum() * 0.0
    denominator = clean_skeleton.new_tensor(0.0)
    for index, (prediction, target, valid, width) in enumerate((
        (denoised[0], clean_skeleton, skeleton_mask, 60),
        (denoised[1], clean_imu, imu_mask, 6),
    )):
        selected = supervision[:, index, None] & valid.bool()
        if bool(selected.any()):
            per_feature = nn.functional.smooth_l1_loss(
                prediction[:, :, :width], target[:, :, :width], reduction="none", beta=1.0
            )
            per_frame = per_feature.mean(dim=-1)
            total = total + (per_frame * selected.float()).sum()
            denominator = denominator + selected.float().sum()
    return total / denominator.clamp_min(1.0)


def _robust_feature_distill_loss(clean_embedding: Tensor, corrupted_embedding: Tensor) -> Tensor:
    target = nn.functional.normalize(clean_embedding.detach(), dim=1)
    prediction = nn.functional.normalize(corrupted_embedding, dim=1)
    return (1.0 - (target * prediction).sum(dim=1)).mean()


def _robust_topology_auxiliary_loss(logits: Tensor, labels: Tensor, availability: Tensor) -> Tensor:
    per_sample = nn.functional.cross_entropy(logits, labels, reduction="none")
    weights = availability[:, 0].float()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


def _epoch(
    model: SafeSense,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    sample_total = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    augmentation_stats = None
    if training and (model.variant == "robust" or getattr(model, "subject_style_training", False)):
        dataset = loader.dataset
        if not isinstance(dataset, SafeSenseDataset):
            raise SafeSenseError("robust training requires SafeSenseDataset augmentation statistics")
        augmentation_stats = {name: value.to(device) for name, value in dataset.augmentation_stats.items()}
    with context:
        for skeleton, imu, mask, availability, labels in loader:
            skeleton, imu, mask, availability, labels = (value.to(device) for value in (skeleton, imu, mask, availability, labels))
            lag_targets = lag_supervision = None
            if training and model.variant == "robust":
                if augmentation_stats is None:
                    raise SafeSenseError("robust augmentation statistics are unavailable")
                clean_logits, _ = model(skeleton, imu, mask, availability)
                clean_experts = model.expert_logits
                clean_embedding = model.temporal_embedding
                clean_topology_logits = model.robust_topology_logits
                if clean_experts is None or clean_embedding is None or clean_topology_logits is None:
                    raise SafeSenseError("clean robust representations were not produced")

                temporal_corrupted = _augment_temporal_batch(skeleton, imu, mask)
                temporal_skeleton, temporal_imu, temporal_mask, temporal_availability, temporal_lag_targets, temporal_lag_supervision = temporal_corrupted
                temporal_logits, _ = model(temporal_skeleton, temporal_imu, temporal_mask, temporal_availability)
                temporal_experts = model.expert_logits
                temporal_topology_logits = model.robust_topology_logits
                if temporal_experts is None or temporal_topology_logits is None:
                    raise SafeSenseError("temporal-corrupted robust experts were not produced")
                temporal_lag_loss = _temporal_lag_loss(
                    model, temporal_lag_targets, temporal_lag_supervision, temporal_mask
                )

                structured = _augment_structured_temporal_batch(skeleton, imu, mask, augmentation_stats)
                structured_skeleton, structured_imu, structured_mask, structured_availability, lag_targets, lag_supervision, reconstruction_supervision, correction_targets = structured
                structured_logits, _ = model(structured_skeleton, structured_imu, structured_mask, structured_availability)
                structured_experts = model.expert_logits
                structured_embedding = model.temporal_embedding
                denoised = model.denoised_inputs
                correction_gate = model.correction_gate
                structured_topology_logits = model.robust_topology_logits
                if structured_experts is None or structured_embedding is None or denoised is None or correction_gate is None or structured_topology_logits is None:
                    raise SafeSenseError("structured-corrupted robust representations were not produced")
                structured_lag_loss = _temporal_lag_loss(model, lag_targets, lag_supervision, structured_mask)

                clean_weight = TEMPORAL_CLEAN_VIEW_WEIGHT
                temporal_weight = ROBUST_TEMPORAL_CORRUPTED_WEIGHT
                structured_weight = ROBUST_STRUCTURED_CORRUPTED_WEIGHT
                if abs(clean_weight + temporal_weight + structured_weight - 1.0) > 1e-6:
                    raise SafeSenseError("robust tri-view classification weights must sum to one")
                loss = (
                    clean_weight * nn.functional.cross_entropy(clean_logits, labels)
                    + temporal_weight * nn.functional.cross_entropy(temporal_logits, labels)
                    + structured_weight * nn.functional.cross_entropy(structured_logits, labels)
                )
                loss = loss + EXPERT_AUXILIARY_CE_WEIGHT * (
                    clean_weight * _expert_auxiliary_loss(clean_experts, labels)
                    + temporal_weight * _expert_auxiliary_loss(temporal_experts, labels)
                    + structured_weight * _expert_auxiliary_loss(structured_experts, labels)
                )
                loss = loss + ROBUST_TOPOLOGY_AUXILIARY_WEIGHT * (
                    clean_weight * _robust_topology_auxiliary_loss(clean_topology_logits, labels, availability)
                    + temporal_weight * _robust_topology_auxiliary_loss(temporal_topology_logits, labels, temporal_availability)
                    + structured_weight * _robust_topology_auxiliary_loss(structured_topology_logits, labels, structured_availability)
                )
                corrupted_total = temporal_weight + structured_weight
                temporal_fraction = temporal_weight / corrupted_total
                structured_fraction = structured_weight / corrupted_total
                loss = loss + TEMPORAL_CONSISTENCY_WEIGHT * (
                    temporal_fraction * _temporal_consistency_loss(clean_logits, temporal_logits)
                    + structured_fraction * _temporal_consistency_loss(clean_logits, structured_logits)
                )
                loss = loss + LAG_AUXILIARY_CE_WEIGHT * (
                    temporal_fraction * temporal_lag_loss + structured_fraction * structured_lag_loss
                )
                loss = loss + ROBUST_FEATURE_DISTILL_WEIGHT * _robust_feature_distill_loss(clean_embedding, structured_embedding)
                loss = loss + ROBUST_RECONSTRUCTION_WEIGHT * _robust_reconstruction_loss(
                    denoised, skeleton, imu, mask, reconstruction_supervision
                )
                loss = loss + ROBUST_GATE_AUXILIARY_WEIGHT * nn.functional.binary_cross_entropy(
                    correction_gate, correction_targets
                )
                logits = clean_logits
            elif training and model.variant == "temporal":
                clean_logits, _ = model(skeleton, imu, mask, availability)
                clean_experts = model.expert_logits
                if clean_experts is None:
                    raise SafeSenseError("clean temporal expert logits were not produced")
                corrupted = _augment_temporal_batch(skeleton, imu, mask)
                corrupted_skeleton, corrupted_imu, corrupted_mask, corrupted_availability, lag_targets, lag_supervision = corrupted
                corrupted_logits, _ = model(
                    corrupted_skeleton, corrupted_imu, corrupted_mask, corrupted_availability
                )
                corrupted_experts = model.expert_logits
                if corrupted_experts is None:
                    raise SafeSenseError("corrupted temporal expert logits were not produced")
                clean_weight = TEMPORAL_CLEAN_VIEW_WEIGHT
                corrupted_weight = 1.0 - clean_weight
                loss = (
                    clean_weight * nn.functional.cross_entropy(clean_logits, labels)
                    + corrupted_weight * nn.functional.cross_entropy(corrupted_logits, labels)
                )
                loss = loss + EXPERT_AUXILIARY_CE_WEIGHT * (
                    clean_weight * _expert_auxiliary_loss(clean_experts, labels)
                    + corrupted_weight * _expert_auxiliary_loss(corrupted_experts, labels)
                )
                loss = loss + TEMPORAL_CONSISTENCY_WEIGHT * _temporal_consistency_loss(
                    clean_logits, corrupted_logits
                )
                if lag_targets is not None and lag_supervision is not None:
                    corrupted_lag_loss = _temporal_lag_loss(
                        model, lag_targets, lag_supervision, corrupted_mask
                    )
                    loss = loss + LAG_AUXILIARY_CE_WEIGHT * corrupted_lag_loss
                if getattr(model, "subject_style_training", False):
                    if augmentation_stats is None:
                        raise SafeSenseError("subject-style training requires train-only augmentation statistics")
                    styled_skeleton, styled_imu, styled_mask, styled_availability = _augment_subject_style_batch(
                        skeleton, imu, mask, augmentation_stats
                    )
                    styled_logits, _ = model(styled_skeleton, styled_imu, styled_mask, styled_availability)
                    styled_experts = model.expert_logits
                    if styled_experts is None:
                        raise SafeSenseError("subject-style temporal experts were not produced")
                    loss = loss + SUBJECT_STYLE_CLASSIFICATION_WEIGHT * (
                        nn.functional.cross_entropy(styled_logits, labels)
                        + EXPERT_AUXILIARY_CE_WEIGHT * _expert_auxiliary_loss(styled_experts, labels)
                    )
                    loss = loss + SUBJECT_STYLE_CONSISTENCY_WEIGHT * _temporal_consistency_loss(
                        clean_logits, styled_logits
                    )
                logits = clean_logits
            else:
                if training and model.variant == "aligned":
                    skeleton, imu, mask, availability, lag_targets, lag_supervision = _augment_aligned_batch(skeleton, imu, mask)
                logits, _ = model(skeleton, imu, mask, availability)
                loss = nn.functional.cross_entropy(logits, labels)
                if training and model.variant in ("classwise", "classprior", "aligned"):
                    experts = model.expert_logits
                    if experts is None:
                        raise SafeSenseError("expert logits were not produced")
                    loss = loss + EXPERT_AUXILIARY_CE_WEIGHT * _expert_auxiliary_loss(experts, labels)
                if training and model.variant == "aligned" and lag_targets is not None and lag_supervision is not None and lag_supervision.any():
                    lag_logits = model.lag_logits
                    if lag_logits is None:
                        raise SafeSenseError("lag logits were not produced")
                    per_sample = nn.functional.cross_entropy(lag_logits, lag_targets, reduction="none")
                    shifted = lag_targets != _corrective_lag_target(0)
                    lag_weights = torch.where(shifted, torch.ones_like(per_sample), torch.full_like(per_sample, 0.25))
                    lag_weights = lag_weights * lag_supervision.float()
                    lag_loss = (per_sample * lag_weights).sum() / lag_weights.sum().clamp_min(1.0)
                    loss = loss + LAG_AUXILIARY_CE_WEIGHT * lag_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = int(labels.shape[0])
            loss_total += float(loss.detach().cpu()) * batch_size
            sample_total += batch_size
            predictions.append(logits.detach().argmax(1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    return loss_total / max(sample_total, 1), np.concatenate(predictions), np.concatenate(targets)


def _factorized_alignment_loss(model: SafeSenseFactorized, availability: Tensor) -> Tensor:
    embeddings = model.alignment_embeddings
    if embeddings is None:
        raise SafeSenseError("factorized alignment embeddings were not produced")
    both = (availability[:, 0] > 0) & (availability[:, 1] > 0)
    if not bool(both.any()):
        return embeddings[0].sum() * 0.0
    skeleton = nn.functional.normalize(embeddings[0][both], dim=1)
    imu = nn.functional.normalize(embeddings[1][both], dim=1)
    return (1.0 - (skeleton * imu).sum(dim=1)).mean()


def _epoch_factorized(
    model: SafeSenseFactorized,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    sample_total = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    augmentation_stats = None
    if training:
        dataset = loader.dataset
        if not isinstance(dataset, SafeSenseDataset):
            raise SafeSenseError("factorized training requires SafeSenseDataset augmentation statistics")
        augmentation_stats = {name: value.to(device) for name, value in dataset.augmentation_stats.items()}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for skeleton, imu, mask, availability, labels in loader:
            skeleton, imu, mask, availability, labels = (
                value.to(device) for value in (skeleton, imu, mask, availability, labels)
            )
            clean_logits, _ = model(skeleton, imu, mask, availability)
            clean_experts = model.expert_logits
            if clean_experts is None:
                raise SafeSenseError("factorized clean expert logits were not produced")
            if training:
                if augmentation_stats is None:
                    raise SafeSenseError("factorized augmentation statistics are unavailable")
                temporal = _augment_temporal_batch(skeleton, imu, mask)
                temporal_logits, _ = model(temporal[0], temporal[1], temporal[2], temporal[3])
                temporal_experts = model.expert_logits
                structured = _augment_structured_temporal_batch(skeleton, imu, mask, augmentation_stats)
                structured_logits, _ = model(structured[0], structured[1], structured[2], structured[3])
                structured_experts = model.expert_logits
                styled = _augment_subject_style_batch(skeleton, imu, mask, augmentation_stats)
                styled_logits, _ = model(styled[0], styled[1], styled[2], styled[3])
                styled_experts = model.expert_logits
                if temporal_experts is None or structured_experts is None or styled_experts is None:
                    raise SafeSenseError("factorized augmented expert logits were not produced")
                style_alignment = _factorized_alignment_loss(model, styled[3])
                # Recompute the clean view last so the stored alignment embeddings match it.
                clean_logits, _ = model(skeleton, imu, mask, availability)
                clean_experts = model.expert_logits
                if clean_experts is None:
                    raise SafeSenseError("factorized clean expert logits were not restored")
                clean_alignment = _factorized_alignment_loss(model, availability)

                loss = (
                    FACTORIZED_CLEAN_WEIGHT * nn.functional.cross_entropy(clean_logits, labels)
                    + FACTORIZED_TEMPORAL_WEIGHT * nn.functional.cross_entropy(temporal_logits, labels)
                    + FACTORIZED_STRUCTURED_WEIGHT * nn.functional.cross_entropy(structured_logits, labels)
                    + FACTORIZED_STYLE_WEIGHT * nn.functional.cross_entropy(styled_logits, labels)
                )
                loss = loss + FACTORIZED_EXPERT_AUXILIARY_WEIGHT * (
                    FACTORIZED_CLEAN_WEIGHT * _expert_auxiliary_loss(clean_experts, labels)
                    + FACTORIZED_TEMPORAL_WEIGHT * _expert_auxiliary_loss(temporal_experts, labels)
                    + FACTORIZED_STRUCTURED_WEIGHT * _expert_auxiliary_loss(structured_experts, labels)
                    + FACTORIZED_STYLE_WEIGHT * _expert_auxiliary_loss(styled_experts, labels)
                )
                loss = loss + FACTORIZED_ALIGNMENT_WEIGHT * (0.75 * clean_alignment + 0.25 * style_alignment)
                loss = loss + FACTORIZED_CONSISTENCY_WEIGHT * (
                    _temporal_consistency_loss(clean_logits, temporal_logits)
                    + _temporal_consistency_loss(clean_logits, structured_logits)
                    + _temporal_consistency_loss(clean_logits, styled_logits)
                ) / 3.0
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                loss = nn.functional.cross_entropy(clean_logits, labels)
            batch_size = int(labels.shape[0])
            loss_total += float(loss.detach().cpu()) * batch_size
            sample_total += batch_size
            predictions.append(clean_logits.detach().argmax(1).cpu().numpy())
            targets.append(labels.cpu().numpy())
    return loss_total / max(sample_total, 1), np.concatenate(predictions), np.concatenate(targets)



def runtime_environment() -> dict[str, str | None]:
    import platform

    result: dict[str, str | None] = {"python": platform.python_version(), "cuda": None, "gpu": None}
    for name in ("numpy", "scipy", "torch"):
        try:
            module = __import__(name)
            result[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - diagnostic only
            result[name] = f"unavailable: {error}"
    if torch.cuda.is_available():
        result["cuda"] = str(torch.version.cuda)
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def _development_seed(prepared: Path, splits: SafeSenseSplits, config: SafeSenseTrainingConfig, variant: str, device: torch.device, artifact_dir: Path | None = None) -> tuple[int, dict[str, object]]:
    skeleton, imu, mask, labels, stats = _development_data(prepared, splits)
    train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
    valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
    model = SafeSense(variant=variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_epoch, best_accuracy, best_macro_f1, bad_epochs = 1, -1.0, -1.0, 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _epoch(
            model,
            _loader(train, config, True, config.seed + epoch),
            device,
            optimizer,
        )
        valid_loss, valid_predictions, valid_targets = _epoch(model, _loader(valid, config, False, config.seed), device, None)
        metrics = _metrics(valid_predictions, valid_targets)
        accuracy = float(metrics["accuracy"])
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss, "validation_accuracy": accuracy, "validation_macro_f1": float(metrics["macro_f1"])})
        if accuracy > best_accuracy + config.min_delta:
            best_epoch, best_accuracy, best_macro_f1, bad_epochs = epoch, accuracy, float(metrics["macro_f1"]), 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if best_state is None:
            raise SafeSenseError("development run did not produce a best checkpoint")
        torch.save(best_state, artifact_dir / "checkpoint.pt")
        np.savez(artifact_dir / "stats.npz", **stats)
    return best_epoch, {"seed": config.seed, "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy, "best_validation_macro_f1": best_macro_f1, "history": history}


def _initialize_robust_adapter(model: SafeSense, temporal_checkpoint: str | Path) -> list[str]:
    """Load a frozen temporal backbone and leave only robust adapter parameters trainable."""
    temporal_state = torch.load(temporal_checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(temporal_state, strict=False)
    if incompatible.unexpected_keys:
        raise SafeSenseError(f"unexpected temporal checkpoint keys: {incompatible.unexpected_keys}")
    if any(not name.startswith("robust_") for name in incompatible.missing_keys):
        raise SafeSenseError(f"temporal checkpoint is missing shared SafeSense weights: {incompatible.missing_keys}")
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("robust_")
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise SafeSenseError("robust adapter has no trainable parameters")
    return trainable


def _evaluate_robust_adapter_validation(
    model: SafeSense,
    loader: DataLoader,
    device: torch.device,
    augmentation_stats: dict[str, Tensor],
    *,
    seed: int,
) -> dict[str, float]:
    """Evaluate fixed temporal and structured corrupted views without changing training RNG state."""
    model.eval()
    temporal_predictions: list[np.ndarray] = []
    structured_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            for skeleton, imu, mask, availability, labels in loader:
                skeleton, imu, mask, availability, labels = (
                    value.to(device) for value in (skeleton, imu, mask, availability, labels)
                )
                temporal = _augment_temporal_batch(skeleton, imu, mask)
                temporal_logits, _ = model(temporal[0], temporal[1], temporal[2], temporal[3])
                structured = _augment_structured_temporal_batch(skeleton, imu, mask, augmentation_stats)
                structured_logits, _ = model(structured[0], structured[1], structured[2], structured[3])
                temporal_predictions.append(temporal_logits.argmax(dim=1).cpu().numpy())
                structured_predictions.append(structured_logits.argmax(dim=1).cpu().numpy())
                targets.append(labels.cpu().numpy())
    target_array = np.concatenate(targets)
    temporal_accuracy = float(np.mean(np.concatenate(temporal_predictions) == target_array))
    structured_accuracy = float(np.mean(np.concatenate(structured_predictions) == target_array))
    return {
        "temporal_corrupted_accuracy": temporal_accuracy,
        "structured_corrupted_accuracy": structured_accuracy,
        "robust_selection_accuracy": 0.5 * (temporal_accuracy + structured_accuracy),
    }


def _development_robust_adapter_seed(
    prepared: Path,
    splits: SafeSenseSplits,
    config: SafeSenseTrainingConfig,
    device: torch.device,
    temporal_checkpoint: str | Path,
    artifact_dir: Path,
) -> tuple[int, dict[str, object]]:
    skeleton, imu, mask, labels, stats = _development_data(prepared, splits)
    train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
    valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
    temporal_base = SafeSense(variant="temporal").to(device)
    temporal_base.load_state_dict(torch.load(temporal_checkpoint, map_location=device, weights_only=True))
    _, base_predictions, base_targets = _epoch(
        temporal_base, _loader(valid, config, False, config.seed), device, None
    )
    base_metrics = _metrics(base_predictions, base_targets)
    base_clean_accuracy = float(base_metrics["accuracy"])
    del temporal_base

    model = SafeSense(variant="robust").to(device)
    trainable = _initialize_robust_adapter(model, temporal_checkpoint)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    augmentation_stats = {name: value.to(device) for name, value in train.augmentation_stats.items()}
    best_epoch, best_accuracy, best_macro_f1, best_selection_score, bad_epochs = 1, -1.0, -1.0, -1.0, 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, _, _ = _epoch(model, _loader(train, config, True, config.seed + epoch), device, optimizer)
        valid_loader = _loader(valid, config, False, config.seed)
        valid_loss, valid_predictions, valid_targets = _epoch(model, valid_loader, device, None)
        metrics = _metrics(valid_predictions, valid_targets)
        accuracy = float(metrics["accuracy"])
        robust_metrics = _evaluate_robust_adapter_validation(
            model,
            _loader(valid, config, False, config.seed),
            device,
            augmentation_stats,
            seed=20260814 + config.seed,
        )
        clean_safe = accuracy >= base_clean_accuracy - ROBUST_ADAPTER_CLEAN_TOLERANCE
        selection_score = (
            robust_metrics["robust_selection_accuracy"] + ROBUST_ADAPTER_CLEAN_SCORE_WEIGHT * accuracy
            if clean_safe else -1.0
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "validation_accuracy": accuracy,
            "validation_macro_f1": float(metrics["macro_f1"]),
            **robust_metrics,
            "clean_safe": float(clean_safe),
            "selection_score": selection_score,
        })
        if selection_score > best_selection_score + config.min_delta:
            best_epoch = epoch
            best_accuracy = accuracy
            best_macro_f1 = float(metrics["macro_f1"])
            best_selection_score = selection_score
            bad_epochs = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if best_state is None:
        raise SafeSenseError("robust adapter development run did not produce a best checkpoint")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, artifact_dir / "checkpoint.pt")
    np.savez(artifact_dir / "stats.npz", **stats)
    return best_epoch, {
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_macro_f1,
        "history": history,
        "temporal_checkpoint": str(temporal_checkpoint),
        "base_clean_validation_accuracy": base_clean_accuracy,
        "selection_metric": "mean deterministic temporal/structured corrupted accuracy + 0.10 * clean accuracy, subject to clean floor",
        "clean_tolerance": ROBUST_ADAPTER_CLEAN_TOLERANCE,
        "best_selection_score": best_selection_score,
        "frozen_shared_parameters": True,
        "trainable_adapter_parameter_names": trainable,
    }


def _select_shared_adapter_epoch(records: Sequence[dict[str, object]]) -> tuple[int, float, float, float]:
    """Choose one shared adapter epoch by mean robust score with all seed clean floors satisfied."""
    if not records:
        raise SafeSenseError("shared adapter-epoch selection requires seed records")
    histories = []
    for record in records:
        history = {int(item["epoch"]): item for item in record.get("history", [])}
        if not history:
            raise SafeSenseError("shared adapter-epoch selection requires validation histories")
        histories.append(history)
    common_epochs = sorted(set.intersection(*(set(history) for history in histories)))
    candidates = []
    for epoch in common_epochs:
        rows = [history[epoch] for history in histories]
        if not all(bool(row.get("clean_safe", 0.0)) for row in rows):
            continue
        mean_score = float(np.mean([float(row["selection_score"]) for row in rows]))
        mean_robust = float(np.mean([float(row["robust_selection_accuracy"]) for row in rows]))
        mean_clean = float(np.mean([float(row["validation_accuracy"]) for row in rows]))
        candidates.append((mean_score, mean_robust, mean_clean, -epoch, epoch))
    if not candidates:
        raise SafeSenseError("no shared adapter epoch satisfies every seed clean floor")
    mean_score, mean_robust, mean_clean, _, epoch = max(candidates)
    return int(epoch), mean_score, mean_robust, mean_clean


def _select_clean_first_adapter_epoch(records: Sequence[dict[str, object]]) -> tuple[int, float, float, float]:
    """Choose one shared epoch after enforcing every seed's temporal clean floor."""
    if not records:
        raise SafeSenseError("clean-first adapter selection requires seed records")
    histories: list[dict[int, dict[str, object]]] = []
    for record in records:
        history = {int(item["epoch"]): item for item in record.get("history", [])}
        if not history:
            raise SafeSenseError("clean-first adapter selection requires validation histories")
        histories.append(history)
    common_epochs = sorted(set.intersection(*(set(history) for history in histories)))
    candidates: list[tuple[float, float, float, int, int]] = []
    for epoch in common_epochs:
        rows = [history[epoch] for history in histories]
        if not all(bool(row.get("clean_safe", False)) for row in rows):
            continue
        mean_robust = float(np.mean([float(row["robust_selection_accuracy"]) for row in rows]))
        mean_clean = float(np.mean([float(row["validation_accuracy"]) for row in rows]))
        mean_f1 = float(np.mean([float(row["validation_macro_f1"]) for row in rows]))
        candidates.append((mean_robust, mean_clean, mean_f1, -epoch, epoch))
    if not candidates:
        raise SafeSenseError("no shared adapter epoch preserves every seed temporal clean floor")
    mean_robust, mean_clean, mean_f1, _, epoch = max(candidates)
    return int(epoch), mean_robust, mean_clean, mean_f1


def run_safesense_robust_adapter_development(
    prepared_dir: str | Path,
    output_dir: str | Path,
    temporal_run_dir: str | Path,
    *,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    device: str = "cpu",
) -> dict[str, object]:
    """Train only robust adapters around frozen per-seed temporal development backbones."""
    if not tuple(seeds):
        raise SafeSenseError("robust adapter development requires at least one seed")
    started = time.perf_counter()
    prepared, output, temporal_root = Path(prepared_dir), Path(output_dir), Path(temporal_run_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_splits(prepared)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    records = []
    for seed in seeds:
        seed_config = config or SafeSenseTrainingConfig(seed=int(seed))
        if config is not None:
            seed_config = SafeSenseTrainingConfig(**{**config.to_dict(), "seed": int(seed)})
        checkpoint = temporal_root / f"seed-{seed}" / "checkpoint.pt"
        if not checkpoint.is_file():
            raise SafeSenseError(f"missing temporal development checkpoint for robust adapter seed {seed}: {checkpoint}")
        set_safesense_seed(seed_config.seed)
        _, record = _development_robust_adapter_seed(
            prepared, splits, seed_config, selected_device, checkpoint, output / f"seed-{seed}"
        )
        records.append(record)
    best_epochs = [int(record["best_epoch"]) for record in records]
    frozen_epoch, shared_score, shared_robust_accuracy, shared_clean_accuracy = _select_shared_adapter_epoch(records)
    model = SafeSense(variant="robust")
    adapter_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters() if name.startswith("robust_")
    )
    result = {
        "model": "SafeSense",
        "workflow": "development",
        "status": "completed",
        "variant": "robust",
        "training_mode": "frozen_temporal_backbone_robust_adapter",
        "temporal_run_dir": str(temporal_root),
        "seeds": [int(seed) for seed in seeds],
        "best_epochs": best_epochs,
        "frozen_epoch": frozen_epoch,
        "epoch_rule": "shared epoch maximizing mean robustness-aware score across seeds with every clean floor satisfied; earliest epoch tie-break",
        "frozen_epoch_mean_selection_score": shared_score,
        "frozen_epoch_mean_robust_selection_accuracy": shared_robust_accuracy,
        "frozen_epoch_mean_clean_accuracy": shared_clean_accuracy,
        "config": (config or SafeSenseTrainingConfig()).to_dict(),
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "split_counts": {"development_train": int(len(splits.development_train)), "development_validation": int(len(splits.development_valid))},
        "parameter_count": parameter_count(model),
        "adapter_trainable_parameter_count": adapter_parameters,
        "shared_temporal_backbone_frozen": True,
        "checkpoint_selection": "development robustness-aware with a hard clean-accuracy floor",
        "adapter_clean_tolerance": ROBUST_ADAPTER_CLEAN_TOLERANCE,
        "adapter_clean_score_weight": ROBUST_ADAPTER_CLEAN_SCORE_WEIGHT,
        "effective_gate_center": ROBUST_EFFECTIVE_GATE_CENTER,
        "effective_gate_sharpness": ROBUST_EFFECTIVE_GATE_SHARPNESS,
        "clean_view_weight": TEMPORAL_CLEAN_VIEW_WEIGHT,
        "robust_temporal_corrupted_weight": ROBUST_TEMPORAL_CORRUPTED_WEIGHT,
        "robust_structured_corrupted_weight": ROBUST_STRUCTURED_CORRUPTED_WEIGHT,
        "reconstruction_weight": ROBUST_RECONSTRUCTION_WEIGHT,
        "feature_distill_weight": ROBUST_FEATURE_DISTILL_WEIGHT,
        "correction_gate_auxiliary_weight": ROBUST_GATE_AUXILIARY_WEIGHT,
        "topology_auxiliary_weight": ROBUST_TOPOLOGY_AUXILIARY_WEIGHT,
        "lag_auxiliary_ce_weight": LAG_AUXILIARY_CE_WEIGHT,
        "consistency_weight": TEMPORAL_CONSISTENCY_WEIGHT,
        "seed_results": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
        "auxiliary_ce_weight_applied": EXPERT_AUXILIARY_CE_WEIGHT,
    }
    (output / "development.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_safesense_robust_adapter_fixed_epoch(
    prepared_dir: str | Path,
    output_dir: str | Path,
    temporal_run_dir: str | Path,
    *,
    adapter_epoch: int,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    device: str = "cpu",
) -> dict[str, object]:
    """Replay one shared robust-adapter epoch budget around frozen temporal development bases."""
    if adapter_epoch < 1 or not tuple(seeds):
        raise SafeSenseError("fixed robust adapter requires a positive epoch and at least one seed")
    started = time.perf_counter()
    prepared, output, temporal_root = Path(prepared_dir), Path(output_dir), Path(temporal_run_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_splits(prepared)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    records: list[dict[str, object]] = []
    trainable_names: list[str] | None = None
    for seed in seeds:
        base_config = config or SafeSenseTrainingConfig(seed=int(seed))
        seed_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "seed": int(seed), "epochs": adapter_epoch})
        checkpoint = temporal_root / f"seed-{seed}" / "checkpoint.pt"
        if not checkpoint.is_file():
            raise SafeSenseError(f"missing temporal development checkpoint for fixed adapter seed {seed}: {checkpoint}")
        set_safesense_seed(seed_config.seed)
        skeleton, imu, mask, labels, stats = _development_data(prepared, splits)
        train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
        valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
        temporal_base = SafeSense(variant="temporal").to(selected_device)
        temporal_base.load_state_dict(torch.load(checkpoint, map_location=selected_device, weights_only=True))
        _, base_predictions, base_targets = _epoch(
            temporal_base, _loader(valid, seed_config, False, seed_config.seed), selected_device, None
        )
        base_clean_accuracy = float(_metrics(base_predictions, base_targets)["accuracy"])
        del temporal_base

        model = SafeSense(variant="robust").to(selected_device)
        trainable = _initialize_robust_adapter(model, checkpoint)
        trainable_names = trainable
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=seed_config.learning_rate,
            weight_decay=seed_config.weight_decay,
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, adapter_epoch + 1):
            train_loss, _, _ = _epoch(
                model,
                _loader(train, seed_config, True, seed_config.seed + epoch),
                selected_device,
                optimizer,
            )
            history.append({"epoch": epoch, "train_loss": train_loss})
        valid_loss, valid_predictions, valid_targets = _epoch(
            model, _loader(valid, seed_config, False, seed_config.seed), selected_device, None
        )
        metrics = _metrics(valid_predictions, valid_targets)
        augmentation_stats = {name: value.to(selected_device) for name, value in train.augmentation_stats.items()}
        robust_metrics = _evaluate_robust_adapter_validation(
            model,
            _loader(valid, seed_config, False, seed_config.seed),
            selected_device,
            augmentation_stats,
            seed=20260814 + seed_config.seed,
        )
        artifact_dir = output / f"seed-{seed}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, artifact_dir / "checkpoint.pt")
        np.savez(artifact_dir / "stats.npz", **stats)
        records.append({
            "seed": int(seed),
            "best_epoch": adapter_epoch,
            "best_validation_accuracy": float(metrics["accuracy"]),
            "best_validation_macro_f1": float(metrics["macro_f1"]),
            "validation_loss": valid_loss,
            "history": history,
            **robust_metrics,
            "clean_safe": bool(float(metrics["accuracy"]) >= base_clean_accuracy - ROBUST_ADAPTER_CLEAN_TOLERANCE),
            "temporal_checkpoint": str(checkpoint),
            "base_clean_validation_accuracy": base_clean_accuracy,
            "checkpoint_policy": f"fixed shared robust adapter epoch {adapter_epoch}",
            "frozen_shared_parameters": True,
            "trainable_adapter_parameter_names": trainable,
        })
    model = SafeSense(variant="robust")
    adapter_parameters = sum(parameter.numel() for name, parameter in model.named_parameters() if name.startswith("robust_"))
    effective_config = config or SafeSenseTrainingConfig()
    result = {
        "model": "SafeSense",
        "workflow": "development",
        "status": "completed",
        "variant": "robust",
        "training_mode": "fixed_epoch_frozen_temporal_backbone_robust_adapter",
        "temporal_run_dir": str(temporal_root),
        "seeds": [int(seed) for seed in seeds],
        "best_epochs": [adapter_epoch for _ in seeds],
        "frozen_epoch": adapter_epoch,
        "epoch_rule": "pre-selected shared robust adapter epoch",
        "config": {**effective_config.to_dict(), "epochs": adapter_epoch},
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "split_counts": {"development_train": int(len(splits.development_train)), "development_validation": int(len(splits.development_valid))},
        "parameter_count": parameter_count(model),
        "adapter_trainable_parameter_count": adapter_parameters,
        "shared_temporal_backbone_frozen": True,
        "inference_only_topology_coverage_gate": "squared relative skeleton/IMU valid-frame coverage",
        "adapter_clean_tolerance": ROBUST_ADAPTER_CLEAN_TOLERANCE,
        "adapter_clean_score_weight": ROBUST_ADAPTER_CLEAN_SCORE_WEIGHT,
        "effective_gate_center": ROBUST_EFFECTIVE_GATE_CENTER,
        "effective_gate_sharpness": ROBUST_EFFECTIVE_GATE_SHARPNESS,
        "clean_view_weight": TEMPORAL_CLEAN_VIEW_WEIGHT,
        "robust_temporal_corrupted_weight": ROBUST_TEMPORAL_CORRUPTED_WEIGHT,
        "robust_structured_corrupted_weight": ROBUST_STRUCTURED_CORRUPTED_WEIGHT,
        "reconstruction_weight": ROBUST_RECONSTRUCTION_WEIGHT,
        "feature_distill_weight": ROBUST_FEATURE_DISTILL_WEIGHT,
        "correction_gate_auxiliary_weight": ROBUST_GATE_AUXILIARY_WEIGHT,
        "topology_auxiliary_weight": ROBUST_TOPOLOGY_AUXILIARY_WEIGHT,
        "lag_auxiliary_ce_weight": LAG_AUXILIARY_CE_WEIGHT,
        "consistency_weight": TEMPORAL_CONSISTENCY_WEIGHT,
        "seed_results": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
        "auxiliary_ce_weight_applied": EXPERT_AUXILIARY_CE_WEIGHT,
        "trainable_adapter_parameter_names": trainable_names or [],
    }
    (output / "development.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_safesense_odd_fold_temporal_history(
    prepared_dir: str | Path, output_dir: str | Path, *, validation_subject: int,
    max_epoch: int = 60, config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33), subject_style_training: bool = True,
    body_canonicalization: bool = True, device: str = "cpu",
) -> dict[str, object]:
    """Record temporal validation history on one odd fold without opening even arrays."""
    if validation_subject not in (1, 3, 5, 7) or max_epoch < 1 or not tuple(seeds):
        raise SafeSenseError("invalid odd-fold temporal-history request")
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_odd_fold_splits(prepared, validation_subject)
    skeleton, imu, mask, labels, stats = _development_data(prepared, splits, body_canonicalization=body_canonicalization)
    train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
    valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    base_config = config or SafeSenseTrainingConfig()
    records = []
    for seed in seeds:
        seed_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "seed": int(seed), "epochs": int(max_epoch)})
        set_safesense_seed(seed_config.seed)
        model = SafeSense(variant="temporal", subject_style_training=subject_style_training).to(selected_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay)
        history = []
        for epoch in range(1, max_epoch + 1):
            train_loss, _, _ = _epoch(model, _loader(train, seed_config, True, seed_config.seed + epoch), selected_device, optimizer)
            validation_loss, predictions, targets = _epoch(model, _loader(valid, seed_config, False, seed_config.seed), selected_device, None)
            metrics = _metrics(predictions, targets)
            history.append({"epoch": epoch, "train_loss": float(train_loss), "validation_loss": float(validation_loss), "validation_accuracy": float(metrics["accuracy"]), "validation_macro_f1": float(metrics["macro_f1"])})
        best = max(history, key=lambda item: (item["validation_accuracy"], item["validation_macro_f1"], -item["epoch"]))
        records.append({"seed": int(seed), "best_epoch": int(best["epoch"]), "best_validation_accuracy": float(best["validation_accuracy"]), "best_validation_macro_f1": float(best["validation_macro_f1"]), "history": history})
        del model, optimizer
        if selected_device.type == "cuda": torch.cuda.empty_cache()
    shared_rows = []
    for epoch in range(1, max_epoch + 1):
        rows = [record["history"][epoch - 1] for record in records]
        shared_rows.append({"epoch": epoch, "mean_accuracy": float(np.mean([row["validation_accuracy"] for row in rows])), "min_seed_accuracy": float(np.min([row["validation_accuracy"] for row in rows])), "mean_macro_f1": float(np.mean([row["validation_macro_f1"] for row in rows]))})
    selected = max(shared_rows, key=lambda item: (item["mean_accuracy"], item["min_seed_accuracy"], item["mean_macro_f1"], -item["epoch"]))
    result = {"model": "SafeSense", "workflow": "odd_subject_temporal_epoch_history", "status": "completed", "validation_subject": int(validation_subject), "training_subjects": [subject for subject in (1, 3, 5, 7) if subject != validation_subject], "official_even_subjects_loaded": False, "body_canonicalization": bool(body_canonicalization), "subject_style_training": bool(subject_style_training), "max_epoch": int(max_epoch), "seeds": [int(seed) for seed in seeds], "parameter_count": parameter_count(SafeSense(variant="temporal")), "seed_results": records, "shared_epoch_history": shared_rows, "selected_shared_epoch": int(selected["epoch"]), "selected_shared_mean_accuracy": float(selected["mean_accuracy"]), "selected_shared_min_seed_accuracy": float(selected["min_seed_accuracy"]), "selected_shared_mean_macro_f1": float(selected["mean_macro_f1"])}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_safesense_odd_fold_adapter_history(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_subject: int,
    temporal_epoch: int = 55,
    max_adapter_epoch: int = 40,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    subject_style_training: bool = True,
    body_canonicalization: bool = True,
    device: str = "cpu",
) -> dict[str, object]:
    """Train temporal once, then record clean-first robust-adapter history on one odd fold."""
    if validation_subject not in (1, 3, 5, 7) or temporal_epoch < 1 or max_adapter_epoch < 1 or not tuple(seeds):
        raise SafeSenseError("invalid odd-fold adapter-history request")
    started = time.perf_counter()
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_odd_fold_splits(prepared, validation_subject)
    skeleton, imu, mask, labels, stats = _development_data(
        prepared, splits, body_canonicalization=body_canonicalization
    )
    train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
    valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    base_config = config or SafeSenseTrainingConfig()
    records: list[dict[str, object]] = []
    corruption_seed_offset = validation_subject * 1000

    for seed in seeds:
        temporal_config = SafeSenseTrainingConfig(
            **{**base_config.to_dict(), "seed": int(seed), "epochs": int(temporal_epoch)}
        )
        set_safesense_seed(temporal_config.seed)
        temporal = SafeSense(variant="temporal", subject_style_training=subject_style_training).to(selected_device)
        temporal_optimizer = torch.optim.AdamW(
            temporal.parameters(), lr=temporal_config.learning_rate, weight_decay=temporal_config.weight_decay
        )
        for epoch in range(1, temporal_epoch + 1):
            _epoch(
                temporal,
                _loader(train, temporal_config, True, temporal_config.seed + epoch),
                selected_device,
                temporal_optimizer,
            )
        temporal_loss, temporal_predictions, temporal_targets = _epoch(
            temporal, _loader(valid, temporal_config, False, temporal_config.seed), selected_device, None
        )
        temporal_metrics = _metrics(temporal_predictions, temporal_targets)
        augmentation_stats = {name: value.to(selected_device) for name, value in train.augmentation_stats.items()}
        temporal_robust_metrics = _evaluate_robust_adapter_validation(
            temporal,
            _loader(valid, temporal_config, False, temporal_config.seed),
            selected_device,
            augmentation_stats,
            seed=20260814 + temporal_config.seed + corruption_seed_offset,
        )

        artifact_dir = output / "artifacts" / f"seed-{seed}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        temporal_checkpoint = artifact_dir / "temporal-checkpoint.pt"
        torch.save(
            {name: value.detach().cpu() for name, value in temporal.state_dict().items()},
            temporal_checkpoint,
        )
        np.savez(artifact_dir / "stats.npz", **stats)
        del temporal, temporal_optimizer
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

        robust = SafeSense(variant="robust").to(selected_device)
        trainable = _initialize_robust_adapter(robust, temporal_checkpoint)
        adapter_config = SafeSenseTrainingConfig(
            **{**base_config.to_dict(), "seed": int(seed), "epochs": int(max_adapter_epoch)}
        )
        robust_optimizer = torch.optim.AdamW(
            [parameter for parameter in robust.parameters() if parameter.requires_grad],
            lr=adapter_config.learning_rate,
            weight_decay=adapter_config.weight_decay,
        )
        history: list[dict[str, object]] = []
        for epoch in range(1, max_adapter_epoch + 1):
            train_loss, _, _ = _epoch(
                robust,
                _loader(train, adapter_config, True, adapter_config.seed + epoch),
                selected_device,
                robust_optimizer,
            )
            validation_loss, predictions, targets = _epoch(
                robust, _loader(valid, adapter_config, False, adapter_config.seed), selected_device, None
            )
            metrics = _metrics(predictions, targets)
            robust_metrics = _evaluate_robust_adapter_validation(
                robust,
                _loader(valid, adapter_config, False, adapter_config.seed),
                selected_device,
                augmentation_stats,
                seed=20260814 + adapter_config.seed + corruption_seed_offset,
            )
            clean_accuracy = float(metrics["accuracy"])
            clean_safe = clean_accuracy >= float(temporal_metrics["accuracy"]) - ROBUST_ADAPTER_CLEAN_TOLERANCE
            history.append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "validation_accuracy": clean_accuracy,
                "validation_macro_f1": float(metrics["macro_f1"]),
                **robust_metrics,
                "clean_safe": bool(clean_safe),
                "clean_accuracy_delta_vs_temporal": clean_accuracy - float(temporal_metrics["accuracy"]),
                "robust_selection_delta_vs_temporal": float(robust_metrics["robust_selection_accuracy"]) - float(temporal_robust_metrics["robust_selection_accuracy"]),
            })
        best_clean_safe: dict[str, object] | None = None
        clean_safe_rows = [row for row in history if bool(row["clean_safe"])]
        if clean_safe_rows:
            best_clean_safe = max(
                clean_safe_rows,
                key=lambda row: (
                    float(row["robust_selection_accuracy"]),
                    float(row["validation_accuracy"]),
                    float(row["validation_macro_f1"]),
                    -int(row["epoch"]),
                ),
            )
        records.append({
            "seed": int(seed),
            "temporal_epoch": int(temporal_epoch),
            "base_clean_validation_loss": float(temporal_loss),
            "base_clean_validation_accuracy": float(temporal_metrics["accuracy"]),
            "base_clean_validation_macro_f1": float(temporal_metrics["macro_f1"]),
            "base_temporal_corrupted_accuracy": float(temporal_robust_metrics["temporal_corrupted_accuracy"]),
            "base_structured_corrupted_accuracy": float(temporal_robust_metrics["structured_corrupted_accuracy"]),
            "base_robust_selection_accuracy": float(temporal_robust_metrics["robust_selection_accuracy"]),
            "history": history,
            "best_clean_safe_epoch": int(best_clean_safe["epoch"]) if best_clean_safe is not None else None,
            "best_clean_safe_robust_selection_accuracy": float(best_clean_safe["robust_selection_accuracy"]) if best_clean_safe is not None else None,
            "best_clean_safe_validation_accuracy": float(best_clean_safe["validation_accuracy"]) if best_clean_safe is not None else None,
            "best_clean_safe_validation_macro_f1": float(best_clean_safe["validation_macro_f1"]) if best_clean_safe is not None else None,
            "best_clean_safe_improves_temporal_robustness": bool(
                best_clean_safe is not None
                and float(best_clean_safe["robust_selection_accuracy"]) > float(temporal_robust_metrics["robust_selection_accuracy"]) + base_config.min_delta
            ),
            "temporal_checkpoint": str(temporal_checkpoint),
            "frozen_shared_parameters": True,
            "trainable_adapter_parameter_names": trainable,
        })
        del robust, robust_optimizer
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    shared_selection: dict[str, object] | None = None
    shared_selection_error: str | None = None
    if len(records) > 1:
        try:
            selected_epoch, mean_robust, mean_clean, mean_f1 = _select_clean_first_adapter_epoch(records)
            mean_base_robust = float(np.mean([float(record["base_robust_selection_accuracy"]) for record in records]))
            shared_selection = {
                "epoch": int(selected_epoch),
                "mean_robust_selection_accuracy": mean_robust,
                "mean_clean_accuracy": mean_clean,
                "mean_macro_f1": mean_f1,
                "mean_temporal_base_robust_selection_accuracy": mean_base_robust,
                "improves_mean_temporal_robustness": bool(mean_robust > mean_base_robust + base_config.min_delta),
            }
        except SafeSenseError as exc:
            shared_selection_error = str(exc)

    robust_model = SafeSense(variant="robust")
    adapter_parameters = sum(
        parameter.numel() for name, parameter in robust_model.named_parameters() if name.startswith("robust_")
    )
    result = {
        "model": "SafeSense",
        "workflow": "odd_subject_canonical_adapter_epoch_history",
        "status": "completed",
        "validation_subject": int(validation_subject),
        "training_subjects": [subject for subject in (1, 3, 5, 7) if subject != validation_subject],
        "official_even_subjects_loaded": False,
        "selection_prohibition": "even subjects 2,4,6,8 are not opened or used by this workflow",
        "temporal_epoch": int(temporal_epoch),
        "max_adapter_epoch": int(max_adapter_epoch),
        "subject_style_training": bool(subject_style_training),
        "body_canonicalization": bool(body_canonicalization),
        "seeds": [int(seed) for seed in seeds],
        "parameter_count": parameter_count(robust_model),
        "adapter_trainable_parameter_count": int(adapter_parameters),
        "adapter_clean_tolerance": ROBUST_ADAPTER_CLEAN_TOLERANCE,
        "selection_rule": "eligible only when every seed is clean-safe vs temporal55; maximize mean robust-selection accuracy, then mean clean accuracy, then mean macro-F1, then earliest epoch",
        "robustness_gain_rule": "reject the full adapter if the selected clean-safe epoch does not strictly improve mean robustness-selection over the temporal55 base",
        "seed_results": records,
        "shared_selection": shared_selection,
        "shared_selection_error": shared_selection_error,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _update_swa_state(model: nn.Module, swa_state: dict[str, Tensor] | None, count: int) -> tuple[dict[str, Tensor], int]:
    state = model.state_dict()
    if swa_state is None:
        return {name: value.detach().clone() for name, value in state.items()}, 1
    next_count = count + 1
    with torch.no_grad():
        for name, value in state.items():
            detached = value.detach()
            if torch.is_floating_point(detached):
                swa_state[name].add_((detached - swa_state[name]) / float(next_count))
            else:
                swa_state[name].copy_(detached)
    return swa_state, next_count


def run_safesense_factorized_swa_odd_fold(
    prepared_dir: str | Path, output_dir: str | Path, *, validation_subject: int, epochs: int = 60, swa_start_epoch: int = 45,
    config: SafeSenseTrainingConfig | None = None, seeds: Sequence[int] = (11,22,33), progress_interval_seconds: float = 30.0, device: str = "cpu",
) -> dict[str, object]:
    """Train the original factorized model and evaluate late-epoch averaged weights."""
    if validation_subject not in (1,3,5,7) or epochs < 1 or not 1 <= swa_start_epoch <= epochs or not tuple(seeds):
        raise SafeSenseError("invalid factorized SWA odd-fold request")
    prepared,output=Path(prepared_dir),Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    progress=_ProgressReporter(output/"progress.log",interval_seconds=progress_interval_seconds)
    progress.start(f"run-start factorized-swa validation_subject={validation_subject} seeds={[int(seed) for seed in seeds]} epochs={epochs} swa_start={swa_start_epoch}")
    try:
        splits=safesense_odd_fold_splits(prepared,validation_subject); skeleton,imu,mask,labels,stats=_development_data(prepared,splits)
        train=SafeSenseDataset(skeleton,imu,mask,labels,splits.development_train,stats); valid=SafeSenseDataset(skeleton,imu,mask,labels,splits.development_valid,stats)
        selected_device=torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu"); base_config=config or SafeSenseTrainingConfig()
        records=[]
        for seed_index,seed in enumerate(seeds,start=1):
            seed_config=SafeSenseTrainingConfig(**{**base_config.to_dict(),"seed":int(seed),"epochs":int(epochs)})
            progress.update(f"factorized-swa validation_subject={validation_subject} seed={seed_index}/{len(seeds)} value={seed} training-start",emit=True)
            set_safesense_seed(seed_config.seed); model=SafeSenseFactorized().to(selected_device); optimizer=torch.optim.AdamW(model.parameters(),lr=seed_config.learning_rate,weight_decay=seed_config.weight_decay)
            swa_state=None; swa_count=0
            for epoch in range(1,epochs+1):
                train_loss,_,_=_epoch_factorized(model,_loader(train,seed_config,True,seed_config.seed+epoch),selected_device,optimizer)
                if epoch >= swa_start_epoch: swa_state,swa_count=_update_swa_state(model,swa_state,swa_count)
                progress.update(f"factorized-swa validation_subject={validation_subject} seed={seed_index}/{len(seeds)} value={seed} epoch={epoch}/{epochs} train_loss={train_loss:.6f} swa_count={swa_count}",emit=epoch==1 or epoch==epochs or epoch%5==0)
            if swa_state is None: raise SafeSenseError("factorized SWA did not collect any states")
            model.load_state_dict(swa_state)
            valid_loss,predictions,targets=_epoch_factorized(model,_loader(valid,seed_config,False,seed_config.seed),selected_device,None); metrics=_metrics(predictions,targets)
            augmentation_stats={name:value.to(selected_device) for name,value in train.augmentation_stats.items()}
            robust_metrics=_evaluate_robust_adapter_validation(model,_loader(valid,seed_config,False,seed_config.seed),selected_device,augmentation_stats,seed=20260815+seed_config.seed+validation_subject*1000)
            artifact_dir=output/"artifacts"/f"seed-{int(seed)}"; artifact_dir.mkdir(parents=True,exist_ok=True); torch.save({name:value.detach().cpu() for name,value in model.state_dict().items()},artifact_dir/"checkpoint.pt"); np.savez(artifact_dir/"stats.npz",**stats)
            record={"seed":int(seed),"validation_loss":float(valid_loss),"validation_accuracy":float(metrics["accuracy"]),"validation_macro_f1":float(metrics["macro_f1"]),**robust_metrics,"swa_state_count":int(swa_count),"checkpoint":str(artifact_dir/"checkpoint.pt")}; records.append(record)
            progress.update(f"factorized-swa validation_subject={validation_subject} seed={seed_index}/{len(seeds)} value={seed} evaluated clean_accuracy={record['validation_accuracy']:.6f} macro_f1={record['validation_macro_f1']:.6f} robust_selection_accuracy={record['robust_selection_accuracy']:.6f}",emit=True)
            del model,optimizer
            if selected_device.type=="cuda": torch.cuda.empty_cache()
        clean=[r["validation_accuracy"] for r in records]; f1=[r["validation_macro_f1"] for r in records]; robust=[r["robust_selection_accuracy"] for r in records]
        result={"model":"SafeSense","workflow":"factorized_late_swa_odd_fold_full_training","status":"completed","validation_subject":int(validation_subject),"training_subjects":[x for x in (1,3,5,7) if x!=validation_subject],"epochs":int(epochs),"swa_start_epoch":int(swa_start_epoch),"swa_epochs":int(epochs-swa_start_epoch+1),"seeds":[int(seed) for seed in seeds],"official_even_subjects_loaded":False,"parameter_count":parameter_count(SafeSenseFactorized()),"training_definition":"original factorized clean+temporal+structured+subject-style objective; equal average of model weights after epochs 45..60; no development checkpoint selection","progress_interval_seconds":float(progress_interval_seconds),"seed_results":records,"aggregate":{"clean_accuracy_mean":float(np.mean(clean)),"clean_accuracy_min_seed":float(np.min(clean)),"clean_macro_f1_mean":float(np.mean(f1)),"robust_selection_accuracy_mean":float(np.mean(robust)),"robust_selection_accuracy_min_seed":float(np.min(robust))},"environment":runtime_environment()}
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); progress.update(f"run-completed factorized-swa validation_subject={validation_subject} clean_mean={result['aggregate']['clean_accuracy_mean']:.6f} clean_min_seed={result['aggregate']['clean_accuracy_min_seed']:.6f} robust_mean={result['aggregate']['robust_selection_accuracy_mean']:.6f}",emit=True); progress.close(); return result
    except Exception:
        progress.close(status="failed"); raise



def run_safesense_factorized_swa_final(
    prepared_dir: str | Path, output_dir: str | Path, *, epochs: int = 60, swa_start_epoch: int = 45,
    seed: int = 22, config: SafeSenseTrainingConfig | None = None, progress_interval_seconds: float = 30.0, device: str = "cpu",
) -> dict[str, object]:
    """Train the frozen seed-22 late-SWA design on all odd subjects, then evaluate even subjects once."""
    if epochs < 1 or not 1 <= swa_start_epoch <= epochs or seed != 22:
        raise SafeSenseError("invalid frozen factorized SWA final request")
    prepared, output = Path(prepared_dir), Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    progress = _ProgressReporter(output / "progress.log", interval_seconds=progress_interval_seconds)
    progress.start(f"run-start factorized-swa-final seed={seed} epochs={epochs} swa_start={swa_start_epoch}")
    try:
        skeleton, imu, mask, labels = _load_partition(prepared, "train")
        train_indices = np.arange(len(labels), dtype=np.int64)
        skeleton_mean, skeleton_std = _standard_stats(skeleton, train_indices, mask)
        imu_mean, imu_std = _standard_stats(imu, train_indices, mask)
        stats = {
            "skeleton_mean": skeleton_mean, "skeleton_std": skeleton_std, "imu_mean": imu_mean, "imu_std": imu_std,
            "skeleton_noise_scale": _raw_noise_scale(skeleton, train_indices, mask, 60),
            "imu_noise_scale": _raw_noise_scale(imu, train_indices, mask, 6),
        }
        train = SafeSenseDataset(skeleton, imu, mask, labels, train_indices, stats)
        selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        base_config = config or SafeSenseTrainingConfig()
        seed_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "seed": seed, "epochs": epochs})
        set_safesense_seed(seed); model = SafeSenseFactorized().to(selected_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay)
        swa_state=None; swa_count=0; history=[]
        for epoch in range(1, epochs + 1):
            train_loss, _, _ = _epoch_factorized(model, _loader(train, seed_config, True, seed + epoch), selected_device, optimizer)
            if epoch >= swa_start_epoch: swa_state, swa_count = _update_swa_state(model, swa_state, swa_count)
            history.append({"epoch": epoch, "train_loss": float(train_loss)})
            progress.update(f"factorized-swa-final seed={seed} epoch={epoch}/{epochs} train_loss={train_loss:.6f} swa_count={swa_count}", emit=epoch==1 or epoch==epochs or epoch%5==0)
        if swa_state is None: raise SafeSenseError("final SWA did not collect any states")
        model.load_state_dict(swa_state)
        progress.update("training-completed; loading official even-subject partition", emit=True)
        test_skeleton, test_imu, test_mask, test_labels = _load_partition(prepared, "val")
        test_indices = np.arange(len(test_labels), dtype=np.int64)
        test = SafeSenseDataset(test_skeleton, test_imu, test_mask, test_labels, test_indices, stats)
        test_loss, predictions, targets = _epoch_factorized(model, _loader(test, seed_config, False, seed), selected_device, None)
        metrics = _metrics(predictions, targets)
        augmentation_stats={name:value.to(selected_device) for name,value in train.augmentation_stats.items()}
        robust_metrics=_evaluate_robust_adapter_validation(model,_loader(test,seed_config,False,seed),selected_device,augmentation_stats,seed=20260815+seed+9000)
        artifact_dir=output / "seed-22"; artifact_dir.mkdir(parents=True,exist_ok=True)
        torch.save({name:value.detach().cpu() for name,value in model.state_dict().items()}, artifact_dir / "checkpoint.pt")
        np.savez(artifact_dir / "stats.npz", **stats); np.save(artifact_dir / "predictions.npy", predictions); np.save(artifact_dir / "labels.npy", targets)
        (artifact_dir / "history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
        result={"model":"SafeSense","workflow":"factorized_late_swa_frozen_final","status":"completed","seed":22,"epochs":epochs,"swa_start_epoch":swa_start_epoch,"swa_state_count":swa_count,"training_subjects":[1,3,5,7],"official_test_subjects":[2,4,6,8],"official_test_loaded_after_training":True,"parameter_count":parameter_count(model),"test_loss":float(test_loss),"accuracy":float(metrics["accuracy"]),"macro_f1":float(metrics["macro_f1"]),**robust_metrics,"checkpoint":str(artifact_dir/"checkpoint.pt"),"stats":str(artifact_dir/"stats.npz"),"environment":runtime_environment()}
        (output / "result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        progress.update(f"final-evaluated clean_accuracy={result['accuracy']:.6f} macro_f1={result['macro_f1']:.6f} robust_selection_accuracy={result['robust_selection_accuracy']:.6f}", emit=True)
        progress.close(); return result
    except Exception:
        progress.close(status="failed"); raise



def run_safesense_factorized_odd_fold(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_subject: int,
    epochs: int = 60,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    progress_interval_seconds: float = 30.0,
    device: str = "cpu",
) -> dict[str, object]:
    """Fully train and immediately evaluate the factorized model on one held-out odd subject."""
    if validation_subject not in (1, 3, 5, 7) or epochs < 1 or not tuple(seeds):
        raise SafeSenseError("invalid factorized odd-fold request")
    if progress_interval_seconds <= 0:
        raise SafeSenseError("progress interval must be positive")
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress = _ProgressReporter(output / "progress.log", interval_seconds=progress_interval_seconds)
    progress.start(
        f"run-start validation_subject={validation_subject} seeds={[int(seed) for seed in seeds]} epochs={epochs}"
    )
    try:
        splits = safesense_odd_fold_splits(prepared, validation_subject)
        skeleton, imu, mask, labels, stats = _development_data(prepared, splits)
        train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
        valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
        selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        base_config = config or SafeSenseTrainingConfig()
        seed_results: list[dict[str, object]] = []
        for seed_index, seed in enumerate(seeds, start=1):
            seed_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "seed": int(seed), "epochs": int(epochs)})
            progress.update(
                f"validation_subject={validation_subject} seed={seed_index}/{len(seeds)} value={int(seed)} training-start",
                emit=True,
            )
            set_safesense_seed(seed_config.seed)
            model = SafeSenseFactorized().to(selected_device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay
            )
            for epoch in range(1, epochs + 1):
                train_loss, _, _ = _epoch_factorized(
                    model,
                    _loader(train, seed_config, True, seed_config.seed + epoch),
                    selected_device,
                    optimizer,
                )
                progress.update(
                    f"validation_subject={validation_subject} seed={seed_index}/{len(seeds)} value={int(seed)} "
                    f"epoch={epoch}/{epochs} train_loss={float(train_loss):.6f}",
                    emit=epoch == 1 or epoch == epochs or epoch % 5 == 0,
                )
            valid_loss, valid_predictions, valid_targets = _epoch_factorized(
                model, _loader(valid, seed_config, False, seed_config.seed), selected_device, None
            )
            metrics = _metrics(valid_predictions, valid_targets)
            augmentation_stats = {name: value.to(selected_device) for name, value in train.augmentation_stats.items()}
            robust_metrics = _evaluate_robust_adapter_validation(
                model,
                _loader(valid, seed_config, False, seed_config.seed),
                selected_device,
                augmentation_stats,
                seed=20260814 + seed_config.seed + validation_subject * 1000,
            )
            artifact_dir = output / "artifacts" / f"seed-{int(seed)}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {name: value.detach().cpu() for name, value in model.state_dict().items()},
                artifact_dir / "checkpoint.pt",
            )
            np.savez(artifact_dir / "stats.npz", **stats)
            record = {
                "seed": int(seed),
                "validation_loss": float(valid_loss),
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_macro_f1": float(metrics["macro_f1"]),
                **robust_metrics,
                "checkpoint": str(artifact_dir / "checkpoint.pt"),
            }
            seed_results.append(record)
            progress.update(
                f"validation_subject={validation_subject} seed={seed_index}/{len(seeds)} value={int(seed)} evaluated "
                f"clean_accuracy={record['validation_accuracy']:.6f} macro_f1={record['validation_macro_f1']:.6f} "
                f"robust_selection_accuracy={record['robust_selection_accuracy']:.6f}",
                emit=True,
            )
            del model, optimizer
            if selected_device.type == "cuda":
                torch.cuda.empty_cache()

        clean = [float(item["validation_accuracy"]) for item in seed_results]
        f1 = [float(item["validation_macro_f1"]) for item in seed_results]
        robust = [float(item["robust_selection_accuracy"]) for item in seed_results]
        result = {
            "model": "SafeSense",
            "workflow": "factorized_graph_time_modality_odd_fold_full_training",
            "status": "completed",
            "validation_subject": int(validation_subject),
            "training_subjects": [subject for subject in (1, 3, 5, 7) if subject != validation_subject],
            "training_count": int(len(splits.development_train)),
            "validation_count": int(len(splits.development_valid)),
            "epochs": int(epochs),
            "epoch_policy": f"fixed {epochs}-epoch budget; no promotion from training curves",
            "seeds": [int(seed) for seed in seeds],
            "official_even_subjects_loaded": False,
            "selection_prohibition": "even subjects 2,4,6,8 are not opened or used by this workflow",
            "architecture": {
                "skeleton": "64-wide flat residual TCN plus fixed-topology graph encoder, fused to 96 channels",
                "imu": "64-wide residual TCN projected to 96 channels",
                "temporal_reduction": "two depthwise-separable stride-2 stages, 128 frames to 32 tokens",
                "fusion": "two factorized time-then-modality attention blocks at width 96",
                "training": "clean + temporal corruption + structured corruption + subject-style views with cross-modal alignment",
            },
            "loss_weights": {
                "clean": FACTORIZED_CLEAN_WEIGHT,
                "temporal": FACTORIZED_TEMPORAL_WEIGHT,
                "structured": FACTORIZED_STRUCTURED_WEIGHT,
                "subject_style": FACTORIZED_STYLE_WEIGHT,
                "expert_auxiliary": FACTORIZED_EXPERT_AUXILIARY_WEIGHT,
                "cross_modal_alignment": FACTORIZED_ALIGNMENT_WEIGHT,
                "consistency": FACTORIZED_CONSISTENCY_WEIGHT,
            },
            "parameter_count": parameter_count(SafeSenseFactorized()),
            "progress_interval_seconds": float(progress_interval_seconds),
            "progress_log": str(output / "progress.log"),
            "seed_results": seed_results,
            "aggregate": {
                "clean_accuracy_mean": float(np.mean(clean)),
                "clean_accuracy_min_seed": float(np.min(clean)),
                "clean_macro_f1_mean": float(np.mean(f1)),
                "robust_selection_accuracy_mean": float(np.mean(robust)),
                "robust_selection_accuracy_min_seed": float(np.min(robust)),
            },
            "environment": runtime_environment(),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        progress.update(
            f"run-completed validation_subject={validation_subject} "
            f"clean_mean={result['aggregate']['clean_accuracy_mean']:.6f} "
            f"clean_min_seed={result['aggregate']['clean_accuracy_min_seed']:.6f} "
            f"robust_mean={result['aggregate']['robust_selection_accuracy_mean']:.6f}",
            emit=True,
        )
        progress.close()
        return result
    except Exception:
        progress.close(status="failed")
        raise


def run_safesense_odd_fold_two_stage_crossval(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    temporal_epoch: int = 33,
    adapter_epoch: int = 27,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    validation_subjects: Sequence[int] = (1, 3, 5, 7),
    subject_style_training: bool = False,
    body_canonicalization: bool = False,
    device: str = "cpu",
) -> dict[str, object]:
    """Replay frozen temporal+robust budgets across leave-one-odd-subject-out folds only."""
    if temporal_epoch < 1 or adapter_epoch < 1 or not tuple(seeds):
        raise SafeSenseError("odd-fold two-stage cross-validation requires positive epochs and seeds")
    folds = tuple(int(subject) for subject in validation_subjects)
    if set(folds) != set(validation_subjects) or any(subject not in (1, 3, 5, 7) for subject in folds):
        raise SafeSenseError("validation subjects must be unique odd UTD-MHAD subjects")
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    base_config = config or SafeSenseTrainingConfig()
    fold_results: list[dict[str, object]] = []
    for validation_subject in folds:
        splits = safesense_odd_fold_splits(prepared, validation_subject)
        train_subjects = [subject for subject in (1, 3, 5, 7) if subject != validation_subject]
        skeleton, imu, mask, labels, stats = _development_data(prepared, splits, body_canonicalization=body_canonicalization)
        train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
        valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
        seed_results: list[dict[str, object]] = []
        for seed in seeds:
            seed_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "seed": int(seed), "epochs": temporal_epoch})
            set_safesense_seed(seed_config.seed)
            temporal = SafeSense(variant="temporal", subject_style_training=subject_style_training).to(selected_device)
            temporal_optimizer = torch.optim.AdamW(temporal.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay)
            for epoch in range(1, temporal_epoch + 1):
                _epoch(temporal, _loader(train, seed_config, True, seed_config.seed + epoch), selected_device, temporal_optimizer)
            _, temporal_predictions, temporal_targets = _epoch(
                temporal, _loader(valid, seed_config, False, seed_config.seed), selected_device, None
            )
            temporal_metrics = _metrics(temporal_predictions, temporal_targets)
            temporal_state = {name: value.detach().cpu().clone() for name, value in temporal.state_dict().items()}

            robust = SafeSense(variant="robust").to(selected_device)
            incompatible = robust.load_state_dict(temporal_state, strict=False)
            if incompatible.unexpected_keys or any(not name.startswith("robust_") for name in incompatible.missing_keys):
                raise SafeSenseError("odd-fold temporal state is incompatible with robust adapter")
            for name, parameter in robust.named_parameters():
                parameter.requires_grad = name.startswith("robust_")
            robust_optimizer = torch.optim.AdamW(
                [parameter for parameter in robust.parameters() if parameter.requires_grad],
                lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay,
            )
            adapter_config = SafeSenseTrainingConfig(**{**seed_config.to_dict(), "epochs": adapter_epoch})
            for epoch in range(1, adapter_epoch + 1):
                _epoch(robust, _loader(train, adapter_config, True, adapter_config.seed + epoch), selected_device, robust_optimizer)
            valid_loss, predictions, targets = _epoch(
                robust, _loader(valid, adapter_config, False, adapter_config.seed), selected_device, None
            )
            metrics = _metrics(predictions, targets)
            augmentation_stats = {name: value.to(selected_device) for name, value in train.augmentation_stats.items()}
            robust_metrics = _evaluate_robust_adapter_validation(
                robust, _loader(valid, adapter_config, False, adapter_config.seed), selected_device, augmentation_stats,
                seed=20260814 + adapter_config.seed + validation_subject * 1000,
            )
            seed_results.append({
                "seed": int(seed),
                "temporal_validation_accuracy": float(temporal_metrics["accuracy"]),
                "temporal_validation_macro_f1": float(temporal_metrics["macro_f1"]),
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_macro_f1": float(metrics["macro_f1"]),
                "validation_loss": float(valid_loss),
                **robust_metrics,
                "clean_safe_vs_temporal": bool(float(metrics["accuracy"]) >= float(temporal_metrics["accuracy"]) - ROBUST_ADAPTER_CLEAN_TOLERANCE),
            })
            del temporal, robust, temporal_optimizer, robust_optimizer
            if selected_device.type == "cuda":
                torch.cuda.empty_cache()
        fold_clean = [float(item["validation_accuracy"]) for item in seed_results]
        fold_f1 = [float(item["validation_macro_f1"]) for item in seed_results]
        fold_robust = [float(item["robust_selection_accuracy"]) for item in seed_results]
        fold_results.append({
            "validation_subject": validation_subject,
            "training_subjects": train_subjects,
            "training_count": int(len(splits.development_train)),
            "validation_count": int(len(splits.development_valid)),
            "seed_results": seed_results,
            "clean_accuracy_mean": float(np.mean(fold_clean)),
            "clean_accuracy_min_seed": float(np.min(fold_clean)),
            "clean_macro_f1_mean": float(np.mean(fold_f1)),
            "robust_selection_accuracy_mean": float(np.mean(fold_robust)),
        })
    fold_clean_means = [float(item["clean_accuracy_mean"]) for item in fold_results]
    fold_f1_means = [float(item["clean_macro_f1_mean"]) for item in fold_results]
    fold_robust_means = [float(item["robust_selection_accuracy_mean"]) for item in fold_results]
    all_seed_clean = [float(seed["validation_accuracy"]) for fold in fold_results for seed in fold["seed_results"]]
    result = {
        "model": "SafeSense",
        "workflow": "odd_subject_leave_one_out_two_stage_cross_validation",
        "status": "completed",
        "variant": "robust",
        "temporal_epoch": int(temporal_epoch),
        "adapter_epoch": int(adapter_epoch),
        "subject_style_training": bool(subject_style_training),
        "subject_style_classification_weight": SUBJECT_STYLE_CLASSIFICATION_WEIGHT if subject_style_training else 0.0,
        "subject_style_consistency_weight": SUBJECT_STYLE_CONSISTENCY_WEIGHT if subject_style_training else 0.0,
        "body_canonicalization": bool(body_canonicalization),
        "body_canonicalization_definition": "first-valid-frame joint-0 origin + sample shoulder/torso scale + mean joint-8-minus-joint-4 yaw alignment" if body_canonicalization else None,
        "seeds": [int(seed) for seed in seeds],
        "validation_subjects": list(folds),
        "official_even_subjects_loaded": False,
        "selection_prohibition": "even subjects 2,4,6,8 are not opened or used by this workflow",
        "parameter_count": parameter_count(SafeSense(variant="robust")),
        "fold_results": fold_results,
        "aggregate": {
            "clean_accuracy_mean_across_folds": float(np.mean(fold_clean_means)),
            "clean_accuracy_worst_fold": float(np.min(fold_clean_means)),
            "clean_accuracy_std_across_folds": float(np.std(fold_clean_means, ddof=0)),
            "clean_accuracy_worst_seed_fold": float(np.min(all_seed_clean)),
            "clean_macro_f1_mean_across_folds": float(np.mean(fold_f1_means)),
            "robust_selection_accuracy_mean_across_folds": float(np.mean(fold_robust_means)),
            "robust_selection_accuracy_worst_fold": float(np.min(fold_robust_means)),
        },
        "selection_recommendation": "future candidates must improve mean and worst-fold clean accuracy without reducing mean/worst-fold robust selection accuracy",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_safesense_development_fixed_epoch(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    frozen_epoch: int,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    variant: str = "temporal",
    device: str = "cpu",
) -> dict[str, object]:
    """Train only subjects 1,3,5 for a fixed epoch count, then evaluate subject 7."""
    if frozen_epoch < 1 or variant not in SAFESENSE_VARIANTS or not tuple(seeds):
        raise SafeSenseError("invalid fixed development epoch, variant, or seed list")
    started = time.perf_counter()
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_splits(prepared)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    skeleton, imu, mask, labels, stats = _development_data(prepared, splits)
    train = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_train, stats)
    valid = SafeSenseDataset(skeleton, imu, mask, labels, splits.development_valid, stats)
    base_config = config or SafeSenseTrainingConfig()
    effective_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "epochs": int(frozen_epoch)})
    records = []
    for seed in seeds:
        seed_config = SafeSenseTrainingConfig(**{**effective_config.to_dict(), "seed": int(seed)})
        set_safesense_seed(seed_config.seed)
        model = SafeSense(variant=variant).to(selected_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay)
        history = []
        for epoch in range(frozen_epoch):
            train_loss, _, _ = _epoch(
                model,
                _loader(train, seed_config, True, seed_config.seed + epoch + 1),
                selected_device,
                optimizer,
            )
            history.append({"epoch": epoch + 1, "train_loss": train_loss})
        validation_loss, predictions, targets = _epoch(
            model, _loader(valid, seed_config, False, seed_config.seed), selected_device, None
        )
        metrics = _metrics(predictions, targets)
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), seed_dir / "checkpoint.pt")
        np.savez(seed_dir / "stats.npz", **stats)
        records.append({
            "seed": int(seed),
            "best_epoch": int(frozen_epoch),
            "best_validation_accuracy": float(metrics["accuracy"]),
            "best_validation_macro_f1": float(metrics["macro_f1"]),
            "validation_loss": validation_loss,
            "history": history,
            "checkpoint_policy": "fixed development epoch; no per-seed validation checkpoint selection",
        })
    result = {
        "model": "SafeSense",
        "workflow": "development",
        "status": "completed",
        "variant": variant,
        "training_mode": "fixed_epoch_development_control",
        "seeds": [int(seed) for seed in seeds],
        "best_epochs": [int(frozen_epoch) for _ in seeds],
        "frozen_epoch": int(frozen_epoch),
        "epoch_rule": "pre-selected fixed epoch",
        "config": effective_config.to_dict(),
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "split_counts": {"development_train": int(len(splits.development_train)), "development_validation": int(len(splits.development_valid))},
        "parameter_count": parameter_count(SafeSense(variant=variant)),
        "seed_results": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
    }
    (output / "development.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _select_shared_frozen_epoch(records: Sequence[dict[str, object]]) -> tuple[int, float, float]:
    """Choose one shared epoch by mean validation performance across all seeds."""
    if not records:
        raise SafeSenseError("shared frozen-epoch selection requires seed records")
    histories = []
    for record in records:
        history = {int(item["epoch"]): item for item in record.get("history", [])}
        if not history:
            raise SafeSenseError("shared frozen-epoch selection requires validation histories")
        histories.append(history)
    common_epochs = sorted(set.intersection(*(set(history) for history in histories)))
    if not common_epochs:
        raise SafeSenseError("seed validation histories have no common epoch")
    candidates = []
    for epoch in common_epochs:
        mean_accuracy = float(np.mean([float(history[epoch]["validation_accuracy"]) for history in histories]))
        mean_macro_f1 = float(np.mean([float(history[epoch]["validation_macro_f1"]) for history in histories]))
        candidates.append((mean_accuracy, mean_macro_f1, -epoch, epoch))
    mean_accuracy, mean_macro_f1, _, selected_epoch = max(candidates)
    return int(selected_epoch), mean_accuracy, mean_macro_f1


def run_safesense_development(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    variant: str = "full",
    device: str = "cpu",
) -> dict[str, object]:
    """Select one epoch using only subjects 1,3,5 train and 7 validation."""
    if variant not in SAFESENSE_VARIANTS or not tuple(seeds):
        raise SafeSenseError("invalid development variant or empty seed list")
    started = time.perf_counter()
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_splits(prepared)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    records = []
    for seed in seeds:
        seed_config = config or SafeSenseTrainingConfig(seed=int(seed))
        if config is not None:
            seed_config = SafeSenseTrainingConfig(**{**config.to_dict(), "seed": int(seed)})
        set_safesense_seed(seed_config.seed)
        _, record = _development_seed(prepared, splits, seed_config, variant, selected_device, output / f"seed-{seed_config.seed}")
        records.append(record)
    best_epochs = tuple(int(record["best_epoch"]) for record in records)
    if variant == "temporal":
        frozen_epoch, frozen_mean_accuracy, frozen_mean_macro_f1 = _select_shared_frozen_epoch(records)
        epoch_rule = "shared epoch maximizing mean validation accuracy across seeds; macro-F1 tie-break; earliest epoch tie-break"
    else:
        frozen_epoch = int(np.median(best_epochs))
        frozen_mean_accuracy = None
        frozen_mean_macro_f1 = None
        epoch_rule = "median of per-seed best validation epochs"
    result = {
        "model": "SafeSense",
        "workflow": "development",
        "status": "completed",
        "variant": variant,
        "seeds": [int(seed) for seed in seeds],
        "best_epochs": list(best_epochs),
        "frozen_epoch": frozen_epoch,
        "epoch_rule": epoch_rule,
        "frozen_epoch_mean_validation_accuracy": frozen_mean_accuracy,
        "frozen_epoch_mean_validation_macro_f1": frozen_mean_macro_f1,
        "config": (config or SafeSenseTrainingConfig()).to_dict(),
        "development_train_subjects": [1, 3, 5],
        "development_validation_subjects": [7],
        "split_counts": {"development_train": int(len(splits.development_train)), "development_validation": int(len(splits.development_valid))},
        "parameter_count": parameter_count(SafeSense(variant=variant)),
        "robust_training": variant in ("aligned", "temporal", "robust"),
        "lag_auxiliary_ce_weight": LAG_AUXILIARY_CE_WEIGHT if variant in ("aligned", "temporal", "robust") else 0.0,
        "consistency_weight": TEMPORAL_CONSISTENCY_WEIGHT if variant in ("temporal", "robust") else 0.0,
        "clean_view_weight": TEMPORAL_CLEAN_VIEW_WEIGHT if variant in ("temporal", "robust") else 1.0,
        "reconstruction_weight": ROBUST_RECONSTRUCTION_WEIGHT if variant == "robust" else 0.0,
        "feature_distill_weight": ROBUST_FEATURE_DISTILL_WEIGHT if variant == "robust" else 0.0,
        "correction_gate_auxiliary_weight": ROBUST_GATE_AUXILIARY_WEIGHT if variant == "robust" else 0.0,
        "topology_auxiliary_weight": ROBUST_TOPOLOGY_AUXILIARY_WEIGHT if variant == "robust" else 0.0,
        "robust_temporal_corrupted_weight": ROBUST_TEMPORAL_CORRUPTED_WEIGHT if variant == "robust" else 0.0,
        "robust_structured_corrupted_weight": ROBUST_STRUCTURED_CORRUPTED_WEIGHT if variant == "robust" else 0.0,
        "structured_corruption": variant == "robust",
        "seed_results": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
        "auxiliary_ce_weight_applied": EXPERT_AUXILIARY_CE_WEIGHT if variant in ("classwise", "classprior", "aligned", "temporal", "robust") else 0.0,
    }
    (output / "development.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Keep the historical smoke artifact name; this JSON contains validation-only data.
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_safesense_robust_two_stage_final(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    temporal_epoch: int = 33,
    adapter_epoch: int = 27,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    subject_style_training: bool = False,
    device: str = "cpu",
) -> dict[str, object]:
    """Train the frozen two-stage SafeSense policy on odd subjects, then evaluate even subjects once."""
    if temporal_epoch < 1 or adapter_epoch < 1 or not tuple(seeds):
        raise SafeSenseError("two-stage final requires positive temporal/adapter epochs and at least one seed")
    seed_list = tuple(int(seed) for seed in seeds)
    started = time.perf_counter()
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_splits(prepared)
    full_train = np.concatenate((splits.development_train, splits.development_valid))
    features, imu, mask, labels, stats = load_safesense_data(prepared, full_train)
    train = SafeSenseDataset(features, imu, mask, labels, full_train, stats)
    test = SafeSenseDataset(features, imu, mask, labels, splits.official_test, stats)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    base_config = config or SafeSenseTrainingConfig()
    training_records: list[dict[str, object]] = []
    for seed in seed_list:
        seed_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "seed": seed})
        set_safesense_seed(seed)
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        temporal = SafeSense(variant="temporal", subject_style_training=subject_style_training).to(selected_device)
        temporal_optimizer = torch.optim.AdamW(
            temporal.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay
        )
        temporal_history: list[dict[str, float]] = []
        for epoch in range(1, temporal_epoch + 1):
            loss, _, _ = _epoch(
                temporal,
                _loader(train, seed_config, True, seed + epoch),
                selected_device,
                temporal_optimizer,
            )
            temporal_history.append({"epoch": epoch, "train_loss": loss})
        temporal_checkpoint = seed_dir / "temporal-checkpoint.pt"
        torch.save({name: value.detach().cpu() for name, value in temporal.state_dict().items()}, temporal_checkpoint)
        del temporal, temporal_optimizer

        robust = SafeSense(variant="robust").to(selected_device)
        trainable = _initialize_robust_adapter(robust, temporal_checkpoint)
        adapter_optimizer = torch.optim.AdamW(
            [parameter for parameter in robust.parameters() if parameter.requires_grad],
            lr=seed_config.learning_rate,
            weight_decay=seed_config.weight_decay,
        )
        adapter_history: list[dict[str, float]] = []
        for epoch in range(1, adapter_epoch + 1):
            loss, _, _ = _epoch(
                robust,
                _loader(train, seed_config, True, seed + epoch),
                selected_device,
                adapter_optimizer,
            )
            adapter_history.append({"epoch": epoch, "train_loss": loss})
        checkpoint = seed_dir / "checkpoint.pt"
        torch.save({name: value.detach().cpu() for name, value in robust.state_dict().items()}, checkpoint)
        np.savez(seed_dir / "stats.npz", **stats)
        (seed_dir / "temporal-history.json").write_text(json.dumps(temporal_history, indent=2) + "\n", encoding="utf-8")
        (seed_dir / "adapter-history.json").write_text(json.dumps(adapter_history, indent=2) + "\n", encoding="utf-8")
        training_records.append({
            "seed": seed,
            "temporal_epoch": temporal_epoch,
            "adapter_epoch": adapter_epoch,
            "temporal_checkpoint": str(temporal_checkpoint),
            "checkpoint": str(checkpoint),
            "shared_temporal_backbone_frozen_during_adapter_training": True,
            "trainable_adapter_parameter_names": trainable,
        })
        del robust, adapter_optimizer

    # The design and both epoch budgets are frozen before this loop touches held-out labels.
    seed_results: list[dict[str, object]] = []
    for record in training_records:
        seed = int(record["seed"])
        seed_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "seed": seed})
        model = SafeSense(variant="robust").to(selected_device)
        model.load_state_dict(torch.load(record["checkpoint"], map_location=selected_device, weights_only=True))
        test_loss, predictions, targets = _epoch(
            model, _loader(test, seed_config, False, seed), selected_device, None
        )
        metrics = _metrics(predictions, targets)
        seed_dir = output / f"seed-{seed}"
        np.save(seed_dir / "predictions.npy", predictions)
        np.save(seed_dir / "labels.npy", targets)
        np.save(seed_dir / "confusion.npy", np.asarray(metrics["confusion_matrix"], dtype=np.int64))
        seed_results.append({"seed": seed, "test_loss": test_loss, **metrics, "checkpoint": str(record["checkpoint"])})

    model = SafeSense(variant="robust")
    adapter_parameters = sum(parameter.numel() for name, parameter in model.named_parameters() if name.startswith("robust_"))
    result = {
        "model": "SafeSense",
        "workflow": "final",
        "status": "completed",
        "variant": "robust",
        "training_mode": "two_stage_fixed_temporal_and_adapter",
        "seeds": list(seed_list),
        "temporal_epoch": temporal_epoch,
        "adapter_epoch": adapter_epoch,
        "subject_style_training": bool(subject_style_training),
        "subject_style_classification_weight": SUBJECT_STYLE_CLASSIFICATION_WEIGHT if subject_style_training else 0.0,
        "subject_style_consistency_weight": SUBJECT_STYLE_CONSISTENCY_WEIGHT if subject_style_training else 0.0,
        "epoch_rule": "development-frozen shared epochs; no held-out checkpoint selection",
        "config": base_config.to_dict(),
        "training_subjects": [1, 3, 5, 7],
        "official_test_subjects": [2, 4, 6, 8],
        "split_counts": {"training": int(len(full_train)), "official_test": int(len(splits.official_test))},
        "parameter_count": parameter_count(model),
        "adapter_trainable_parameter_count": adapter_parameters,
        "shared_temporal_backbone_frozen_during_adapter_training": True,
        "inference_only_topology_coverage_gate": "squared relative skeleton/IMU valid-frame coverage",
        "clean_view_weight": TEMPORAL_CLEAN_VIEW_WEIGHT,
        "robust_temporal_corrupted_weight": ROBUST_TEMPORAL_CORRUPTED_WEIGHT,
        "robust_structured_corrupted_weight": ROBUST_STRUCTURED_CORRUPTED_WEIGHT,
        "reconstruction_weight": ROBUST_RECONSTRUCTION_WEIGHT,
        "feature_distill_weight": ROBUST_FEATURE_DISTILL_WEIGHT,
        "correction_gate_auxiliary_weight": ROBUST_GATE_AUXILIARY_WEIGHT,
        "topology_auxiliary_weight": ROBUST_TOPOLOGY_AUXILIARY_WEIGHT,
        "lag_auxiliary_ce_weight": LAG_AUXILIARY_CE_WEIGHT,
        "consistency_weight": TEMPORAL_CONSISTENCY_WEIGHT,
        "training_records": training_records,
        "seed_results": seed_results,
        "aggregate": {
            "accuracy_mean": float(np.mean([item["accuracy"] for item in seed_results])),
            "accuracy_std_population": float(np.std([item["accuracy"] for item in seed_results], ddof=0)),
            "macro_f1_mean": float(np.mean([item["macro_f1"] for item in seed_results])),
            "macro_f1_std_population": float(np.std([item["macro_f1"] for item in seed_results], ddof=0)),
        },
        "stats": str(output / f"seed-{seed_list[0]}" / "stats.npz"),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
        "auxiliary_ce_weight_applied": EXPERT_AUXILIARY_CE_WEIGHT,
        "held_out_evaluation_after_all_training_completed": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_safesense_final(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    frozen_epoch: int,
    config: SafeSenseTrainingConfig | None = None,
    seeds: Sequence[int] = (11, 22, 33),
    variant: str = "full",
    device: str = "cpu",
) -> dict[str, object]:
    """Train odd subjects for a supplied epoch count, then evaluate even subjects."""
    if frozen_epoch < 1 or variant not in SAFESENSE_VARIANTS or not tuple(seeds):
        raise SafeSenseError("invalid frozen epoch, variant, or empty seed list")
    seed_list = tuple(int(seed) for seed in seeds)
    started = time.perf_counter()
    prepared, output = Path(prepared_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = safesense_splits(prepared)
    full_train = np.concatenate((splits.development_train, splits.development_valid))
    features, imu, mask, labels, stats = load_safesense_data(prepared, full_train)
    train = SafeSenseDataset(features, imu, mask, labels, full_train, stats)
    test = SafeSenseDataset(features, imu, mask, labels, splits.official_test, stats)
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    base_config = config or SafeSenseTrainingConfig()
    effective_config = SafeSenseTrainingConfig(**{**base_config.to_dict(), "epochs": int(frozen_epoch)})
    seed_results = []
    parameter_total = None
    for seed in seed_list:
        seed_config = SafeSenseTrainingConfig(**{**effective_config.to_dict(), "seed": int(seed)})
        set_safesense_seed(seed_config.seed)
        model = SafeSense(variant=variant).to(selected_device)
        parameter_total = parameter_count(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=seed_config.learning_rate, weight_decay=seed_config.weight_decay)
        history = []
        for epoch in range(frozen_epoch):
            loss, _, _ = _epoch(
                model,
                _loader(train, seed_config, True, seed_config.seed + epoch + 1),
                selected_device,
                optimizer,
            )
            history.append({"epoch": epoch + 1, "train_loss": loss})
        test_loss, predictions, targets = _epoch(model, _loader(test, seed_config, False, seed_config.seed), selected_device, None)
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), seed_dir / "checkpoint.pt")
        np.save(seed_dir / "predictions.npy", predictions)
        np.save(seed_dir / "labels.npy", targets)
        np.save(seed_dir / "confusion.npy", np.asarray(_metrics(predictions, targets)["confusion_matrix"], dtype=np.int64))
        np.savez(seed_dir / "stats.npz", **stats)
        (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        seed_results.append({"seed": int(seed), "test_loss": test_loss, **_metrics(predictions, targets), "checkpoint": str(seed_dir / "checkpoint.pt")})
    result = {
        "model": "SafeSense",
        "workflow": "final",
        "status": "completed",
        "variant": variant,
        "seeds": list(seed_list),
        "frozen_epoch": int(frozen_epoch),
        "config": effective_config.to_dict(),
        "training_subjects": [1, 3, 5, 7],
        "official_test_subjects": [2, 4, 6, 8],
        "split_counts": {"training": int(len(full_train)), "official_test": int(len(splits.official_test))},
        "parameter_count": parameter_total,
        "seed_results": seed_results,
        "robust_training": variant == "temporal",
        "lag_auxiliary_ce_weight": LAG_AUXILIARY_CE_WEIGHT if variant == "temporal" else 0.0,
        "consistency_weight": TEMPORAL_CONSISTENCY_WEIGHT if variant == "temporal" else 0.0,
        "clean_view_weight": TEMPORAL_CLEAN_VIEW_WEIGHT if variant == "temporal" else 1.0,
        "aggregate": {
            "accuracy_mean": float(np.mean([item["accuracy"] for item in seed_results])),
            "accuracy_std_population": float(np.std([item["accuracy"] for item in seed_results], ddof=0)),
            "macro_f1_mean": float(np.mean([item["macro_f1"] for item in seed_results])),
            "macro_f1_std_population": float(np.std([item["macro_f1"] for item in seed_results], ddof=0)),
        },
        "stats": str(output / f"seed-{seed_list[0]}" / "stats.npz"),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": runtime_environment(),
        "auxiliary_ce_weight_applied": EXPERT_AUXILIARY_CE_WEIGHT if variant in ("classwise", "classprior", "aligned", "temporal", "robust") else 0.0,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_safesense_experiment(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    config: SafeSenseTrainingConfig | None = None,
    variant: str = "full",
    device: str = "cpu",
) -> dict[str, object]:
    """Backward-compatible development-only entry point; it never evaluates even subjects."""
    return run_safesense_development(prepared_dir, output_dir, config=config, seeds=(config.seed if config else 11,), variant=variant, device=device)


def safesense_smoke(prepared_dir: str | Path, output_dir: str | Path, *, variant: str = "full", device: str = "cpu") -> dict[str, object]:
    result = run_safesense_development(prepared_dir, output_dir, config=SafeSenseTrainingConfig(epochs=1, patience=1), seeds=(11,), variant=variant, device=device)
    result["accuracy"] = result["seed_results"][0]["best_validation_accuracy"]
    return result


def evaluate_safesense(prepared_dir: str | Path, run_dir: str | Path, *, device: str = "cpu") -> dict[str, object]:
    run = Path(run_dir)
    result_path = run / "result.json"
    if not result_path.is_file():
        raise SafeSenseError(f"completed SafeSense artifacts are missing from {run}")
    recorded = json.loads(result_path.read_text(encoding="utf-8"))
    first_seed = int(recorded.get("seeds", [11])[0])
    checkpoint = run / "checkpoint.pt"
    if not checkpoint.is_file():
        checkpoint = run / f"seed-{first_seed}" / "checkpoint.pt"
    stats_path = run / "stats.npz"
    if not stats_path.is_file():
        stats_path = run / f"seed-{first_seed}" / "stats.npz"
    if not checkpoint.is_file() or not stats_path.is_file() or recorded.get("workflow") != "final":
        raise SafeSenseError(f"a final SafeSense artifact set is required for evaluation: {run}")
    prepared = Path(prepared_dir)
    splits = safesense_splits(prepared)
    full_train = np.concatenate((splits.development_train, splits.development_valid))
    features, imu, mask, labels, _ = load_safesense_data(prepared, full_train)
    stats_archive = np.load(stats_path)
    stats = {name: stats_archive[name] for name in ("skeleton_mean", "skeleton_std", "imu_mean", "imu_std")}
    config = SafeSenseTrainingConfig(**{key: value for key, value in recorded["config"].items() if key in SafeSenseTrainingConfig.__dataclass_fields__})
    selected_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = SafeSense(variant=recorded.get("variant", "full")).to(selected_device)
    model.load_state_dict(torch.load(checkpoint, map_location=selected_device, weights_only=True))
    dataset = SafeSenseDataset(features, imu, mask, labels, splits.official_test, stats)
    loss, predictions, targets = _epoch(model, _loader(dataset, config, False, config.seed), selected_device, None)
    return {"status": "completed", "test_loss": loss, **_metrics(predictions, targets), "source_result": str(result_path)}
