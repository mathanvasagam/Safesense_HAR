import torch
from torch import nn

import safesense.efficiency as efficiency
from safesense.efficiency import benchmark_model
from safesense.safesense import FactorizedTimeModalityBlock, _repair_internal_gaps


def test_efficiency_benchmark_reports_model_only_metrics() -> None:
    result = benchmark_model(nn.Identity(), (torch.ones(1, 2),), device="cpu", warmups=1, iterations=3)
    assert result["warmups"] == 1
    assert result["iterations"] == 3
    assert result["parameters"] == 0
    assert result["checkpoint_bytes_fp32"] is None
    assert result["median_ms_model_only"] >= 0
    assert result["p95_ms_model_only"] >= result["median_ms_model_only"]


def test_cuda_graph_capture_helpers_are_math_equivalent() -> None:
    values = torch.arange(3 * 8 * 5, dtype=torch.float32).reshape(3, 8, 5)
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    eager = _repair_internal_gaps(values, mask, radius=2)
    graph_safe = efficiency._cuda_graph_repair_internal_gaps(values, mask, radius=2)
    assert torch.equal(eager, graph_safe)

    valid = mask.bool()
    assert torch.equal(
        FactorizedTimeModalityBlock._safe_valid(valid),
        efficiency._cuda_graph_safe_valid(valid),
    )


def test_dual_efficiency_result_uses_both_checkpoint_sizes_and_ratios(tmp_path, monkeypatch) -> None:
    safe_path, fusion_path = tmp_path / "safe.pt", tmp_path / "fusion.pt"
    safe_path.write_bytes(b"safe")
    fusion_path.write_bytes(b"fusion")
    seen_variants = []
    monkeypatch.setattr(efficiency, "SafeSense", lambda **kwargs: (seen_variants.append(kwargs["variant"]) or nn.Identity()))
    monkeypatch.setattr(efficiency.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(efficiency, "load_fusion_final_model", lambda *args: nn.Identity())
    calls = iter((1.0, 2.0))
    monkeypatch.setattr(efficiency, "benchmark_model", lambda *args, **kwargs: {
        "median_ms_model_only": next(calls), "p95_ms_model_only": 3.0, "parameters": 0,
        "checkpoint_bytes_fp32": kwargs["checkpoint_bytes"],
    })
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    import numpy as np
    values = np.zeros((2, 1, 128, 22, 3), dtype=np.float32)
    values[:, 0, :2, 20:] = 1
    np.save(prepared / "skeleton_train_features.npy", values[:1])
    np.save(prepared / "skeleton_val_features.npy", values[1:])
    (prepared / "manifest.csv").write_text("sample_id,action,subject,trial,split\na1,1,1,1,train\na2,1,2,1,test\n")
    np.save(prepared / "train_labels.npy", np.array([0]))
    np.save(prepared / "val_labels.npy", np.array([0]))
    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, skeleton_mean=np.zeros(177), skeleton_std=np.ones(177), imu_mean=np.zeros(12), imu_std=np.ones(12))
    result = efficiency.benchmark_frozen_models(safe_path, prepared, stats_path, tmp_path / "runtime", fusion_path, tmp_path / "out", device="cpu", safesense_variant="classprior")
    assert result["status"] == "completed"
    assert result["models"]["SafeSense"]["checkpoint_bytes_fp32"] == 4
    assert result["models"]["Fusion-GCN"]["checkpoint_bytes_fp32"] == 6
    assert result["latency_ratio_fusion_to_safesense"] == 2.0
    assert result["safesense_variant"] == "classprior"
    assert seen_variants == ["classprior"]
