import csv
from pathlib import Path

import numpy as np
from scipy.io import savemat

from safesense.preprocessing import (
    PreprocessingError,
    adaptive_average_pool_1d,
    depth_features,
    map_visual_to_imu,
    preprocess_dataset,
    select_visual_indices,
    skeleton_features,
)


def _skeleton(frames: int = 40) -> np.ndarray:
    value = np.zeros((20, 3, frames), dtype=np.float64)
    value[3] = np.array([[1.0], [2.0], [3.0]])
    value[0] = value[3] + np.array([[0.0], [0.0], [2.0]])
    value[13] = value[3] + np.array([[0.0], [0.0], [-1.0]])
    value[17] = value[3] + np.array([[0.0], [0.0], [-1.0]])
    value[0, 0, 12:] += np.arange(frames - 12)
    return value


def test_motion_selection_is_deterministic_and_chronological() -> None:
    skeleton = _skeleton()
    first = select_visual_indices(skeleton)
    second = select_visual_indices(skeleton)
    assert np.array_equal(first, second)
    assert first.shape == (32,)
    assert np.all(np.diff(first) >= 0)
    assert np.all(first < skeleton.shape[2] - 1)


def test_imu_mapping_is_monotonic_and_preserves_endpoints() -> None:
    source = np.array([0, 2, 9], dtype=np.int64)
    mapped = map_visual_to_imu(source, visual_frames=10, imu_frames=17)
    assert np.array_equal(mapped, [0, 4, 16])
    assert np.all(np.diff(mapped) >= 0)


def test_adaptive_pooling_is_fixed_size() -> None:
    assert np.allclose(adaptive_average_pool_1d(np.arange(10), 4), [1.0, 3.0, 6.0, 8.0])


def test_empty_depth_frame_has_zero_feature_and_invalid_mask() -> None:
    features, mask = depth_features(np.zeros((240, 320, 1), dtype=np.uint16), np.zeros(32, dtype=np.int64))
    assert features.shape == (32, 92)
    assert np.all(features == 0)
    assert np.all(mask == 0)


def test_depth_foreground_produces_pooled_opencv_hog() -> None:
    depth = np.zeros((240, 320, 1), dtype=np.uint16)
    depth[40:200, 100:220, 0] = np.arange(160, dtype=np.uint16)[:, None] + 500
    features, mask = depth_features(depth, np.zeros(32, dtype=np.int64))
    assert features.shape == (32, 92)
    assert np.all(mask == 1)
    assert np.any(features != 0)


def test_skeleton_features_are_raw_and_flattened() -> None:
    skeleton = _skeleton(frames=1)
    features = skeleton_features(skeleton, np.zeros(32, dtype=np.int64))
    reshaped = features.reshape(32, 20, 3)
    assert np.allclose(reshaped[:, 3], [1.0, 2.0, 3.0])
    assert np.allclose(reshaped[:, 0], [1.0, 2.0, 5.0])


def _write_fixture_dataset(root: Path, validation_value: float = 2.0, test_value: float = 3.0) -> Path:
    cache = root / "cache"
    cache.mkdir(parents=True)
    extracted = cache / "extracted"
    rows = []
    for index, (sample_id, split, value) in enumerate(
        (("a1_s1_t1", "train", 1.0), ("a1_s7_t1", "validation", validation_value), ("a1_s2_t1", "test", test_value))
    ):
        action, subject, trial = 1, (1 if index == 0 else 7 if index == 1 else 2), 1
        skeleton = _skeleton(frames=4)
        skeleton[0, 0] += value
        depth = np.zeros((240, 320, 4), dtype=np.uint16)
        depth[50:180, 80:220] = int(100 + value)
        inertial = np.full((8, 6), value, dtype=np.float64)
        paths = {}
        for modality, variable, array in (("depth", "d_depth", depth), ("skeleton", "d_skel", skeleton), ("inertial", "d_iner", inertial)):
            directory = extracted / modality
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"{sample_id}_{modality}.mat"
            savemat(directory / filename, {variable: array})
            paths[modality] = f"extracted/{modality}/{filename}"
        rows.append({"sample_id": sample_id, "action": action, "subject": subject, "trial": trial, **paths, "official_split": split, "development_split": split})
    with (cache / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return cache


def test_literal_features_are_not_standardized(tmp_path: Path) -> None:
    cache = _write_fixture_dataset(tmp_path / "first")
    preprocess_dataset(cache)
    with np.load(cache / "features.npz") as payload:
        assert np.allclose(payload["inertial"][0], 1.0)
        assert np.allclose(payload["inertial"][1], 2.0)
        assert np.allclose(payload["inertial"][2], 3.0)
        assert np.array_equal(payload["skeleton"], payload["raw_skeleton"])
        assert np.array_equal(payload["depth"], payload["raw_depth"])


def test_fixture_cache_shapes_dtypes_and_order(tmp_path: Path) -> None:
    cache = _write_fixture_dataset(tmp_path / "fixture")
    preprocess_dataset(cache)
    with np.load(cache / "features.npz") as payload:
        assert payload["sample_ids"].tolist() == ["a1_s1_t1", "a1_s7_t1", "a1_s2_t1"]
        assert payload["depth"].shape == (3, 32, 92)
        assert payload["skeleton"].shape == (3, 32, 60)
        assert payload["inertial"].shape == (3, 32, 6)
        assert payload["depth"].dtype == np.float32
        assert np.isfinite(payload["depth"]).all()
        assert np.isfinite(payload["skeleton"]).all()
        assert np.isfinite(payload["inertial"]).all()


def test_fixture_preprocessing_rerun_is_identical(tmp_path: Path) -> None:
    cache = _write_fixture_dataset(tmp_path / "rerun")
    preprocess_dataset(cache)
    with np.load(cache / "features.npz") as payload:
        first = {name: payload[name].copy() for name in ("depth", "skeleton", "inertial", "visual_indices", "imu_indices")}
    preprocess_dataset(cache)
    with np.load(cache / "features.npz") as payload:
        assert all(np.array_equal(values, payload[name]) for name, values in first.items())


def test_failed_rerun_removes_stale_feature_outputs(tmp_path: Path) -> None:
    cache = _write_fixture_dataset(tmp_path / "failed-rerun")
    preprocess_dataset(cache)
    (cache / "extracted/depth/a1_s1_t1_depth.mat").unlink()
    with np.testing.assert_raises(PreprocessingError):
        preprocess_dataset(cache)
    assert not (cache / "features.npz").exists()
    assert not (cache / "preprocessing.json").exists()
