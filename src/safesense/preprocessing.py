"""Deterministic UTD-MHAD feature extraction."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import numpy as np
import cv2
from scipy.io import loadmat


FRAME_COUNT = 32
DEPTH_FEATURES = 92
SKELETON_FEATURES = 60
IMU_FEATURES = 6


class PreprocessingError(ValueError):
    """Raised when a UTD-MHAD sample cannot be converted safely."""


def _nearest_index(value: float) -> int:
    """Round non-negative indices half-up, avoiding platform tie differences."""
    return int(math.floor(value + 0.5))


def select_visual_indices(skeleton: np.ndarray, count: int = FRAME_COUNT) -> np.ndarray:
    """Select motion-ranked frames, then sample the ranked frames chronologically."""
    if skeleton.ndim != 3 or skeleton.shape[:2] != (20, 3):
        raise PreprocessingError(f"expected skeleton [20,3,T], got {skeleton.shape}")
    frames = skeleton.shape[2]
    if frames < 1:
        raise PreprocessingError("skeleton sequence has no frames")
    if frames == 1:
        return np.zeros(count, dtype=np.int64)

    # Eq. (5): one Frobenius norm over all joints and coordinates per transition.
    displacement = np.linalg.norm(np.diff(skeleton, axis=2), axis=(0, 1))
    candidate_count = min(50, frames - 1)
    ranked = np.argsort(-displacement, kind="stable")[:candidate_count]
    chronological = np.sort(ranked)
    positions = np.linspace(0, len(chronological) - 1, count)
    selected_positions = np.array([_nearest_index(float(position)) for position in positions])
    return chronological[selected_positions].astype(np.int64, copy=False)


def map_visual_to_imu(visual_indices: np.ndarray, visual_frames: int, imu_frames: int) -> np.ndarray:
    """Map visual frame indices to IMU frames by endpoint-preserving proportional rounding."""
    if visual_frames < 1 or imu_frames < 1:
        raise PreprocessingError("both visual and IMU sequences need at least one frame")
    indices = np.asarray(visual_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= visual_frames):
        raise PreprocessingError("visual index is outside the visual sequence")
    if visual_frames == 1:
        return np.zeros(indices.shape, dtype=np.int64)
    scale = (imu_frames - 1) / (visual_frames - 1)
    mapped = np.array([_nearest_index(float(index) * scale) for index in indices], dtype=np.int64)
    return np.clip(mapped, 0, imu_frames - 1)


def adaptive_average_pool_1d(values: np.ndarray, output_size: int) -> np.ndarray:
    """Apply the deterministic start-floor/end-ceil adaptive average-pool rule."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0 or output_size < 1:
        raise PreprocessingError("adaptive pooling requires non-empty input and positive output size")
    pooled = np.empty(output_size, dtype=np.float32)
    for output_index in range(output_size):
        start = math.floor(output_index * values.size / output_size)
        end = math.ceil((output_index + 1) * values.size / output_size)
        pooled[output_index] = values[start:end].mean(dtype=np.float64)
    return pooled


def skeleton_features(skeleton: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return raw selected skeleton frames flattened to [32,60]."""
    if skeleton.ndim != 3 or skeleton.shape[:2] != (20, 3):
        raise PreprocessingError(f"expected skeleton [20,3,T], got {skeleton.shape}")
    if not np.isfinite(skeleton).all():
        raise PreprocessingError("skeleton contains non-finite values")
    if np.any(indices < 0) or np.any(indices >= skeleton.shape[2]):
        raise PreprocessingError("skeleton index is outside the sequence")
    selected = np.asarray(skeleton[:, :, indices], dtype=np.float32).transpose(2, 0, 1)
    return selected.reshape(len(indices), SKELETON_FEATURES).astype(np.float32, copy=False)


def _depth_frame_feature(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return pooled HOG and validity for one depth frame."""
    finite = np.isfinite(frame)
    if not np.any(finite):
        return np.zeros(DEPTH_FEATURES, dtype=np.float32), False
    values = np.nan_to_num(np.asarray(frame, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(np.min(values[finite])), float(np.max(values[finite]))
    if high <= low:
        return np.zeros(DEPTH_FEATURES, dtype=np.float32), False
    normalized = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if not np.any(binary):
        binary = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    if not np.any(binary):
        return np.zeros(DEPTH_FEATURES, dtype=np.float32), False
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(DEPTH_FEATURES, dtype=np.float32), False
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if width < 1 or height < 1:
        return np.zeros(DEPTH_FEATURES, dtype=np.float32), False
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    cropped = cv2.bitwise_and(normalized, normalized, mask=mask)[y : y + height, x : x + width]
    resized = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_LINEAR)
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    descriptor = cv2.HOGDescriptor((16, 16), (16, 16), (8, 8), (8, 8), 10).compute(resized)
    if descriptor is None or descriptor.size == 0:
        return np.zeros(DEPTH_FEATURES, dtype=np.float32), False
    return adaptive_average_pool_1d(descriptor, DEPTH_FEATURES), True


def depth_features(depth: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract pooled HOG features and a valid-frame mask from [240,320,T]."""
    if depth.ndim != 3 or depth.shape[:2] != (240, 320):
        raise PreprocessingError(f"expected depth [240,320,T], got {depth.shape}")
    features = np.zeros((len(indices), DEPTH_FEATURES), dtype=np.float32)
    mask = np.zeros(len(indices), dtype=np.float32)
    for output_index, frame_index in enumerate(np.asarray(indices, dtype=np.int64)):
        if frame_index < 0 or frame_index >= depth.shape[2]:
            raise PreprocessingError("depth index is outside the sequence")
        features[output_index], valid = _depth_frame_feature(depth[:, :, frame_index])
        mask[output_index] = float(valid)
    return features, mask


def _load_sample(cache_dir: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for modality, variable in (("depth", "d_depth"), ("skeleton", "d_skel"), ("inertial", "d_iner")):
        path = cache_dir / row[modality]
        try:
            payload = loadmat(path)
            value = np.asarray(payload[variable])
        except (KeyError, OSError, ValueError) as error:
            raise PreprocessingError(f"cannot load {modality} sample {row['sample_id']}: {error}") from error
        arrays.append(value)
    depth, skeleton, inertial = arrays
    if depth.shape[:2] != (240, 320) or depth.ndim != 3:
        raise PreprocessingError(f"{row['sample_id']}: invalid depth shape {depth.shape}")
    if skeleton.shape[:2] != (20, 3) or skeleton.ndim != 3:
        raise PreprocessingError(f"{row['sample_id']}: invalid skeleton shape {skeleton.shape}")
    if inertial.ndim != 2 or inertial.shape[1] != IMU_FEATURES:
        raise PreprocessingError(f"{row['sample_id']}: invalid inertial shape {inertial.shape}")
    if depth.shape[2] != skeleton.shape[2]:
        raise PreprocessingError(f"{row['sample_id']}: depth/skeleton frame count mismatch")
    if not np.isfinite(depth).all() or not np.isfinite(skeleton).all() or not np.isfinite(inertial).all():
        raise PreprocessingError(f"{row['sample_id']}: non-finite depth, skeleton, or inertial values")
    return depth, skeleton, inertial


def preprocess_dataset(cache_dir: str | Path) -> Path:
    """Create the literal-feature cache from an existing manifest."""
    cache_path = Path(cache_dir)
    for stale_path in (
        cache_path / "features.npz",
        cache_path / "features.npz.tmp",
        cache_path / "preprocessing.json",
        cache_path / "preprocessing.json.tmp",
    ):
        if stale_path.exists():
            stale_path.unlink()
    manifest_path = cache_path / "manifest.csv"
    if not manifest_path.is_file():
        raise PreprocessingError(f"missing manifest: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise PreprocessingError("manifest is empty")

    raw_depth = np.zeros((len(rows), FRAME_COUNT, DEPTH_FEATURES), dtype=np.float32)
    raw_skeleton = np.zeros((len(rows), FRAME_COUNT, SKELETON_FEATURES), dtype=np.float32)
    raw_inertial = np.zeros((len(rows), FRAME_COUNT, IMU_FEATURES), dtype=np.float32)
    depth_mask = np.zeros((len(rows), FRAME_COUNT), dtype=np.float32)
    skeleton_mask = np.ones((len(rows), FRAME_COUNT), dtype=np.float32)
    inertial_mask = np.ones((len(rows), FRAME_COUNT), dtype=np.float32)
    visual_indices = np.zeros((len(rows), FRAME_COUNT), dtype=np.int64)
    imu_indices = np.zeros((len(rows), FRAME_COUNT), dtype=np.int64)

    for row_index, row in enumerate(rows):
        depth, skeleton, inertial = _load_sample(cache_path, row)
        selected = select_visual_indices(skeleton)
        mapped = map_visual_to_imu(selected, skeleton.shape[2], inertial.shape[0])
        raw_depth[row_index], depth_mask[row_index] = depth_features(depth, selected)
        raw_skeleton[row_index] = skeleton_features(skeleton, selected)
        raw_inertial[row_index] = np.asarray(inertial[mapped], dtype=np.float32)
        visual_indices[row_index] = selected
        imu_indices[row_index] = mapped

    payload = {
        "schema_version": 2,
        "dataset": "UTD-MHAD",
        "sample_count": len(rows),
        "frame_selection": "Eq. (5) global Frobenius displacement between consecutive skeleton frames; top min(50,T-1), chronological, 32 nearest-position samples",
        "visual_index_rounding": "nearest integer, half-up; repeated indices are retained when fewer than 32 candidates exist",
        "imu_mapping": "endpoint-preserving proportional mapping round-half-up: round(i*(Timu-1)/(Tvisual-1)); clamp to [0,Timu-1]; inferred because released files contain no timestamps",
        "skeleton": "raw selected 20x3 coordinates flattened to 60 features; no centering, body scaling, or z-score normalization",
        "depth": "full-frame uint8 normalization, Otsu with adaptive-threshold fallback, 3x3 open/close, largest external contour, masked crop, resize 64x64, normalized OpenCV HOG win16/block16/stride8/cell8/10 bins, adaptive average pool to 92",
        "invalid_depth": "empty foreground produces zero HOG and a zero depth mask",
        "natural_lag_claim": False,
    }
    feature_path = cache_path / "features.npz"
    temporary = feature_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=np.array([row["sample_id"] for row in rows]),
            actions=np.array([int(row["action"]) for row in rows], dtype=np.int64),
            subjects=np.array([int(row["subject"]) for row in rows], dtype=np.int64),
            trials=np.array([int(row["trial"]) for row in rows], dtype=np.int64),
            raw_depth=raw_depth,
            raw_skeleton=raw_skeleton,
            raw_inertial=raw_inertial,
            depth=raw_depth,
            skeleton=raw_skeleton,
            inertial=raw_inertial,
            depth_mask=depth_mask,
            skeleton_mask=skeleton_mask,
            inertial_mask=inertial_mask,
            visual_indices=visual_indices,
            imu_indices=imu_indices,
        )
    temporary.replace(feature_path)
    metadata_path = cache_path / "preprocessing.json"
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metadata_temporary.replace(metadata_path)
    return feature_path
