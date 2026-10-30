from pathlib import Path

import numpy as np

from safesense.baseline import FeatureDataset, _metrics


def test_feature_dataset_reads_cached_modalities_and_zero_based_labels(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    arrays = {
        "depth": np.zeros((2, 32, 92), dtype=np.float32),
        "skeleton": np.zeros((2, 32, 60), dtype=np.float32),
        "inertial": np.zeros((2, 32, 6), dtype=np.float32),
        "depth_mask": np.ones((2, 32), dtype=np.float32),
        "skeleton_mask": np.ones((2, 32), dtype=np.float32),
        "inertial_mask": np.ones((2, 32), dtype=np.float32),
        "actions": np.array([1, 27]),
    }
    np.savez(path, **arrays)
    sample = FeatureDataset(path, [1])[0]
    assert sample[0].shape == (32, 92)
    assert sample[1].shape == (32, 60)
    assert sample[2].shape == (32, 6)
    assert sample[-1].item() == 26


def test_metrics_returns_accuracy_macro_f1_and_confusion_matrix() -> None:
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    metrics = _metrics(predictions, targets, num_classes=2)
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert len(metrics["per_class"]) == 2
    assert 0 < metrics["macro_f1"] < 1

