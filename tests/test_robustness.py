import numpy as np
import json

from safesense.robustness import (
    CorruptionSpec,
    aggregate_metrics,
    apply_corruption,
    base_temporal_masks,
    corrective_imu_shift,
    scenario_specs,
    run_safesense_development_robustness,
    run_safesense_conditional_ensemble_development,
    run_safesense_early_exit_development,
    select_fusion_final_checkpoints,
    training_noise_scales,
)


def _fused() -> np.ndarray:
    values = np.zeros((2, 1, 128, 22, 3), dtype=np.float32)
    values[0, 0, :8, :20, 0] = np.arange(8)[:, None]
    values[0, 0, :8, 20:, 1] = np.arange(8)[:, None] + 1
    values[1] = values[0]
    return values


def test_scenario_enumeration_and_zero_padded_shift_both_signs() -> None:
    assert len(scenario_specs()) == 25
    values = _fused()
    masks = base_temporal_masks(values)
    scales = training_noise_scales(values, masks, [0])
    plus, _ = apply_corruption(values, CorruptionSpec("shift", "shift", "skeleton", shift=2), seed=3, scales=scales)
    minus, _ = apply_corruption(values, CorruptionSpec("shift", "shift", "skeleton", shift=-2), seed=3, scales=scales)
    np.testing.assert_array_equal(plus[0, 0, 2, 0, 0], values[0, 0, 0, 0, 0])
    np.testing.assert_array_equal(minus[0, 0, 0, 0, 0], values[0, 0, 2, 0, 0])
    assert plus[0, 0, :2, :20].sum() == 0
    assert minus[0, 0, -2:, :20].sum() == 0
    np.testing.assert_array_equal(masks["skeleton"], masks["imu"])


def test_lag_target_respects_skeleton_anchor_sign_convention() -> None:
    assert corrective_imu_shift(CorruptionSpec("s", "shift", "skeleton", shift=4)) == 4
    assert corrective_imu_shift(CorruptionSpec("i", "shift", "imu", shift=4)) == -4
    assert corrective_imu_shift(CorruptionSpec("c", "composite", "imu", shift=-2)) == 2


def test_frame_loss_and_noise_are_deterministic_and_train_scaled() -> None:
    values = _fused()
    masks = base_temporal_masks(values)
    scales = training_noise_scales(values, masks, [0])
    spec = CorruptionSpec("noise", "noise", "imu", noise_std=0.3)
    first, first_masks = apply_corruption(values, spec, seed=9, scales=scales)
    second, second_masks = apply_corruption(values, spec, seed=9, scales=scales)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_masks["imu"], second_masks["imu"])
    loss = CorruptionSpec("loss", "frame_loss", "skeleton", loss_fraction=0.5)
    values[0, 0, 20, 0, 0] = 7
    lost, lost_masks = apply_corruption(values, loss, seed=9, scales=scales)
    assert lost_masks["skeleton"].sum() == 8
    assert np.any(lost[:, 0, :, :20] == 0)
    assert lost[0, 0, 20, 0, 0] == 7


def test_aggregate_metrics_uses_equal_category_means() -> None:
    result = aggregate_metrics([
        {"scenario": "clean", "category": "clean", "accuracy": 1.0},
        {"scenario": "a", "category": "shift", "accuracy": 0.8},
        {"scenario": "b", "category": "shift", "accuracy": 0.6},
        {"scenario": "c", "category": "noise", "accuracy": 0.4},
    ])
    assert result["category_accuracy_mean"] == {"shift": 0.7, "noise": 0.4}
    assert result["equal_category_corrupted_accuracy_mean"] == 0.55


def test_fusion_checkpoint_selection_uses_final_not_retained(tmp_path) -> None:
    paths = []
    for seed in (11, 22, 33):
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(json.dumps({"seed": seed, "final_checkpoint": f"final-{seed}.pt", "best_retained_checkpoint": f"best-{seed}.pt"}))
        paths.append(path)
    assert select_fusion_final_checkpoints(paths) == {seed: f"final-{seed}.pt" for seed in (11, 22, 33)}


def test_development_robustness_never_requires_even_subject_arrays(tmp_path) -> None:
    import csv
    import torch

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader()
        writer.writerows([
            {"sample_id": "a1_s1_t1", "action": 1, "subject": 1, "trial": 1, "split": "train"},
            {"sample_id": "a1_s7_t1", "action": 1, "subject": 7, "trial": 1, "split": "train"},
            {"sample_id": "a1_s2_t1", "action": 1, "subject": 2, "trial": 1, "split": "test"},
        ])
    values = np.zeros((2, 1, 128, 22, 3), dtype=np.float32)
    values[:, 0, :8, :20, 0] = 1
    values[:, 0, :8, 20:, 1] = 1
    np.save(prepared / "skeleton_train_features.npy", values)
    np.save(prepared / "train_labels.npy", np.array([0, 0], dtype=np.int64))

    run = tmp_path / "run"
    run.mkdir()
    run.joinpath("development.json").write_text(json.dumps({
        "workflow": "development",
        "variant": "classprior",
        "seeds": [11, 22, 33],
        "seed_results": [
            {"seed": seed, "best_validation_accuracy": 1.0} for seed in (11, 22, 33)
        ],
    }))
    for seed in (11, 22, 33):
        seed_dir = run / f"seed-{seed}"
        seed_dir.mkdir()
        np.savez(seed_dir / "stats.npz", skeleton_mean=np.zeros(177), skeleton_std=np.ones(177), imu_mean=np.zeros(12), imu_std=np.ones(12))
        (seed_dir / "checkpoint.pt").write_bytes(b"fake")

    class FakeModel:
        lag_logits = None

        def __call__(self, skeleton, imu, mask):
            return torch.zeros((len(skeleton), 27)), torch.full((len(skeleton), 2), 0.5)

    result = run_safesense_development_robustness(
        prepared,
        run,
        tmp_path / "robust",
        model_loader=lambda *_: FakeModel(),
    )
    assert result["status"] == "completed"
    assert result["even_subject_arrays_loaded"] is False
    assert result["model_aggregate"]["clean_accuracy_mean"] == 1.0
    assert not (prepared / "skeleton_val_features.npy").exists()


def test_conditional_ensemble_development_never_requires_even_arrays(tmp_path) -> None:
    import csv
    import torch
    prepared = tmp_path / "prepared"; prepared.mkdir()
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split")); writer.writeheader()
        writer.writerows([
            {"sample_id":"a1_s1_t1","action":1,"subject":1,"trial":1,"split":"train"},
            {"sample_id":"a1_s7_t1","action":1,"subject":7,"trial":1,"split":"train"},
            {"sample_id":"a1_s2_t1","action":1,"subject":2,"trial":1,"split":"test"},
        ])
    values=np.zeros((2,1,128,22,3),dtype=np.float32); values[:,0,:8,:20,0]=1; values[:,0,:8,20:,1]=1
    np.save(prepared/"skeleton_train_features.npy",values); np.save(prepared/"train_labels.npy",np.array([0,0],dtype=np.int64))
    run=tmp_path/"run"; run.mkdir(); (run/"development.json").write_text(json.dumps({"workflow":"development","variant":"robust","seeds":[11,22,33],"seed_results":[]}))
    for seed in (11,22,33):
        d=run/f"seed-{seed}"; d.mkdir(); np.savez(d/"stats.npz",skeleton_mean=np.zeros(177),skeleton_std=np.ones(177),imu_mean=np.zeros(12),imu_std=np.ones(12)); (d/"checkpoint.pt").write_bytes(b"fake")
    class Fake(torch.nn.Module):
        def __init__(self): super().__init__(); self.register_parameter("dummy",torch.nn.Parameter(torch.zeros(1))); self._gate=None
        def forward(self,skeleton,imu,mask,availability=None):
            self._gate=skeleton.new_zeros((len(skeleton),2)); logits=skeleton.new_zeros((len(skeleton),27)); return logits, skeleton.new_full((len(skeleton),2),0.5)
        @property
        def effective_correction_gate(self): return self._gate
        @property
        def lag_logits(self): return None
    result=run_safesense_conditional_ensemble_development(prepared,run,tmp_path/"out",model_loader=lambda *_: Fake())
    assert result["even_subject_arrays_loaded"] is False
    assert result["no_regression_checks"]["all_passed"] is True
    assert result["conditional_aggregate"]["clean_accuracy_mean"] == 1.0
    assert not (prepared/"skeleton_val_features.npy").exists()


def test_early_exit_development_never_requires_even_arrays(tmp_path) -> None:
    import csv
    import torch

    prepared = tmp_path / "prepared-early"
    prepared.mkdir()
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader()
        writer.writerows([
            {"sample_id": "a1_s1_t1", "action": 1, "subject": 1, "trial": 1, "split": "train"},
            {"sample_id": "a1_s7_t1", "action": 1, "subject": 7, "trial": 1, "split": "train"},
            {"sample_id": "a1_s2_t1", "action": 1, "subject": 2, "trial": 1, "split": "test"},
        ])
    values = np.zeros((2, 1, 128, 22, 3), dtype=np.float32)
    values[:, 0, :8, :20, 0] = 1
    values[:, 0, :8, 20:, 1] = 1
    np.save(prepared / "skeleton_train_features.npy", values)
    np.save(prepared / "train_labels.npy", np.array([0, 0], dtype=np.int64))

    run = tmp_path / "run-early"
    run.mkdir()
    (run / "development.json").write_text(json.dumps({
        "workflow": "development", "variant": "robust", "seeds": [11, 22, 33], "seed_results": []
    }))
    for seed in (11, 22, 33):
        directory = run / f"seed-{seed}"
        directory.mkdir()
        np.savez(directory / "stats.npz", skeleton_mean=np.zeros(177), skeleton_std=np.ones(177), imu_mean=np.zeros(12), imu_std=np.ones(12))
        (directory / "checkpoint.pt").write_bytes(b"fake")

    class FakeControl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
        def forward(self, skeleton, imu, mask, availability=None):
            return skeleton.new_zeros((len(skeleton), 27)), skeleton.new_full((len(skeleton), 2), 0.5)
        @property
        def lag_logits(self): return None

    class FakeCandidate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
            self._route = None
            self._use = None
        def forward(self, skeleton, imu, mask):
            self._route = torch.zeros(len(skeleton), dtype=torch.bool, device=skeleton.device)
            self._use = torch.ones(len(skeleton), dtype=torch.bool, device=skeleton.device)
            return skeleton.new_zeros((len(skeleton), 27))
        @property
        def auxiliary_route(self): return self._route
        @property
        def use_skeleton(self): return self._use

    result = run_safesense_early_exit_development(
        prepared, run, tmp_path / "out-early",
        model_loader=lambda *_: FakeControl(),
        candidate_loader=lambda *_: FakeCandidate(),
    )
    assert result["even_subject_arrays_loaded"] is False
    assert result["no_regression_vs_robust_control"]["all_passed"] is True
    assert result["early_exit_aggregate"]["clean_accuracy_mean"] == 1.0
    assert not (prepared / "skeleton_val_features.npy").exists()


def test_completed_dual_model_artifact_has_dense_axes(tmp_path) -> None:
    import csv
    from safesense.robustness import run_robustness_evaluation

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader()
        writer.writerows([
            {"sample_id": "a1_s1_t1", "action": 1, "subject": 1, "trial": 1, "split": "train"},
            {"sample_id": "a1_s7_t1", "action": 2, "subject": 7, "trial": 1, "split": "train"},
            {"sample_id": "a1_s2_t1", "action": 3, "subject": 2, "trial": 1, "split": "test"},
        ])
    values = _fused()
    np.save(prepared / "skeleton_train_features.npy", values[:2])
    np.save(prepared / "train_labels.npy", np.array([0, 1]))
    np.save(prepared / "skeleton_val_features.npy", values[1:2])
    np.save(prepared / "val_labels.npy", np.array([2]))
    safe_dir, fusion_dir = tmp_path / "safe", tmp_path / "fusion"
    for seed in (11, 22, 33):
        (safe_dir / f"seed-{seed}").mkdir(parents=True)
        np.savez(safe_dir / f"seed-{seed}" / "stats.npz", skeleton_mean=np.zeros(177), skeleton_std=np.ones(177), imu_mean=np.zeros(12), imu_std=np.ones(12))
        (safe_dir / f"seed-{seed}" / "checkpoint.pt").write_bytes(b"fake")
        (fusion_dir / f"seed-{seed}").mkdir(parents=True)
        (fusion_dir / f"seed-{seed}" / f"fusion_gcn_reproduction_seed_{seed}.json").write_text(json.dumps({"seed": seed, "final_checkpoint": f"final-{seed}.pt"}))

    (safe_dir / "result.json").write_text(json.dumps({
        "workflow": "final",
        "variant": "temporal",
        "seed_results": [{"seed": seed, "accuracy": 0.0} for seed in (11, 22, 33)],
    }))

    class FakeModel:
        def __call__(self, *values):
            import torch
            return torch.zeros((len(values[0]), 27)), None

        def eval(self):
            return self

    result = run_robustness_evaluation(
        prepared, safe_dir, tmp_path / "runtime", fusion_dir, tmp_path / "out",
        expected_fusion_accuracy={11: 0.0, 22: 0.0, 33: 0.0},
        safesense_variant="temporal",
        safesense_model_loader=lambda *_: FakeModel(),
        fusion_model_loader=lambda *_: FakeModel(),
    )
    assert result["status"] == "completed"
    assert result["safesense_variant"] == "temporal"
    with np.load(tmp_path / "out" / "predictions.npz") as archive:
        assert archive["predictions"].shape == (2, 3, 25, 1)
        assert archive["labels"].shape == (1,)
