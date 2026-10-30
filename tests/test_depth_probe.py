import numpy as np
import torch

from safesense.depth_probe import DepthHOGProbe, depth_train_stats, paired_outcomes


def test_depth_probe_forward_shape_and_masked_frames() -> None:
    torch.manual_seed(1)
    model = DepthHOGProbe().eval()
    values = torch.randn(2, 32, 92)
    mask = torch.ones(2, 32)
    mask[:, -4:] = 0
    altered = values.clone()
    altered[:, -4:] = 1000
    with torch.no_grad():
        first = model(values, mask)
        second = model(altered, mask)
    assert first.shape == (2, 27)
    assert torch.allclose(first, second, atol=1e-6)


def test_depth_train_stats_use_training_indices_only() -> None:
    depth = np.zeros((3, 2, 2), dtype=np.float32)
    depth[0] = ((1, 3), (5, 7))
    depth[1] = ((3, 5), (7, 9))
    depth[2] = 1000
    mask = np.ones((3, 2), dtype=np.float32)
    mean, std = depth_train_stats(depth, mask, [0, 1])
    np.testing.assert_allclose(mean, np.array([4.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(std, np.sqrt(np.array([5.0, 5.0], dtype=np.float32)))


def test_paired_outcomes_counts_complementarity() -> None:
    targets = np.array([0, 1, 2, 3, 4])
    safesense = np.array([0, 1, 9, 9, 4])
    depth = np.array([0, 9, 2, 9, 8])
    result = paired_outcomes(safesense, depth, targets)
    assert result["both_correct"] == 1
    assert result["safesense_only_correct"] == 2
    assert result["depth_only_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["oracle_union_accuracy"] == 0.8
    assert result["oracle_gain_over_safesense_percentage_points"] == 20.0
    assert result["depth_recovery_rate_of_safesense_errors"] == 0.5
