import csv
import json
from pathlib import Path

import numpy as np
import torch

from safesense.safesense import (
    IMU_INPUTS,
    LAG_SHIFTS,
    SKELETON_INPUTS,
    SafeSenseTrainingConfig,
    SafeSense,
    SafeSenseBranchExpert,
    SafeSenseConditionalEnsemble,
    SafeSenseEarlyExitEnsemble,
    _augment_aligned_batch,
    _augment_temporal_batch,
    _augment_structured_temporal_batch,
    _canonicalize_skeleton_features,
    _corrective_lag_target,
    _corrective_lag_target_for_modality,
    _development_data,
    _initialize_robust_adapter,
    _repair_internal_gaps,
    _sanitize_masked_motion_features,
    _select_clean_first_adapter_epoch,
    _select_shared_adapter_epoch,
    _select_shared_frozen_epoch,
    _update_swa_state,
    _zero_padded_shift_tensor,
    safesense_smoke,
    safesense_splits,
    safesense_odd_fold_splits,
    run_safesense_odd_fold_adapter_history,
    run_safesense_factorized_odd_fold,
    run_safesense_odd_fold_two_stage_crossval,
    load_safesense_data,
    parameter_count,
    run_safesense_development,
    run_safesense_development_fixed_epoch,
    run_safesense_robust_adapter_development,
    run_safesense_robust_adapter_fixed_epoch,
    run_safesense_robust_two_stage_final,
    run_safesense_final,
)


def _prepared(root: Path) -> Path:
    root.mkdir()
    rows = [
        {"sample_id": "a1_s1_t1", "action": "1", "subject": "1", "trial": "1", "split": "train"},
        {"sample_id": "a1_s2_t1", "action": "2", "subject": "2", "trial": "1", "split": "test"},
        {"sample_id": "a1_s7_t1", "action": "3", "subject": "7", "trial": "1", "split": "train"},
    ]
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    values = np.zeros((3, 1, 128, 22, 3), dtype=np.float32)
    values[0, 0, :4, 2] = 1
    values[1, 0, :5, 2] = 2
    values[2, 0, :6, 2] = 3
    values[0, 0, :4, 20:] = 1
    values[1, 0, :5, 20:] = 2
    values[2, 0, :6, 20:] = 3
    # Official prepared order is odd-subject train first, then even-subject val.
    np.save(root / "skeleton_train_features.npy", values[[0, 2]])
    np.save(root / "train_labels.npy", np.array([0, 2], dtype=np.int64))
    np.save(root / "skeleton_val_features.npy", values[[1]])
    np.save(root / "val_labels.npy", np.array([1], dtype=np.int64))
    return root


def test_safesense_splits_translate_manifest_order_and_masks(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    splits = safesense_splits(prepared)
    assert splits.development_train.tolist() == [0]
    assert splits.development_valid.tolist() == [1]
    assert splits.official_test.tolist() == [2]
    skeleton, imu, mask, labels, stats = load_safesense_data(prepared, splits.development_train)
    assert skeleton.shape == (3, 128, SKELETON_INPUTS)
    assert imu.shape == (3, 128, IMU_INPUTS)
    assert mask.sum(axis=1).tolist() == [4.0, 6.0, 5.0]
    assert labels.tolist() == [0, 2, 1]
    assert stats["skeleton_mean"].shape == (SKELETON_INPUTS,)
    assert np.allclose(stats["imu_mean"][:6], 1.0)


def test_safesense_model_missing_modalities_has_finite_outputs_and_gradients() -> None:
    model = SafeSense()
    skeleton = torch.randn(2, 128, SKELETON_INPUTS)
    imu = torch.randn(2, 128, IMU_INPUTS)
    mask = torch.zeros(2, 128)
    availability = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits, weights = model(skeleton, imu, mask, availability)
    assert logits.shape == (2, 27)
    assert weights.shape == (2, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    loss = logits.square().mean() + weights.square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert parameter_count(model) < 1_728_316


def test_safesense_separate_masks_keep_available_branch_active() -> None:
    torch.manual_seed(4)
    model = SafeSense()
    skeleton = torch.randn(2, 8, SKELETON_INPUTS)
    imu = torch.randn(2, 8, IMU_INPUTS)
    separate = torch.zeros(2, 8, 2)
    separate[0, :, 0] = 1
    separate[1, :, 1] = 1
    logits, weights = model(skeleton, imu, separate)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    all_missing, all_weights = model(skeleton, imu, torch.zeros_like(separate))
    assert torch.isfinite(all_missing).all()
    assert torch.isfinite(all_weights).all()


def test_safesense_state_dict_shape_compatibility() -> None:
    original = SafeSense()
    restored = SafeSense()
    restored.load_state_dict(original.state_dict())
    assert set(original.state_dict()) == set(restored.state_dict())
    assert set(SafeSense(variant="full").state_dict()) == set(original.state_dict())
    assert not any("classwise" in key or "expert" in key for key in original.state_dict())


def test_safesense_classwise_router_experts_and_missing_modalities() -> None:
    torch.manual_seed(8)
    model = SafeSense(variant="classwise")
    skeleton = torch.randn(3, 8, SKELETON_INPUTS)
    imu = torch.randn(3, 8, IMU_INPUTS)
    mask = torch.zeros(3, 8, 2)
    mask[0, :, 0] = 1
    mask[1, :, 1] = 1
    mask[2] = 1
    logits, weights = model(skeleton, imu, mask)
    assert logits.shape == (3, 27)
    assert weights.shape == (3, 2)
    assert torch.isfinite(logits).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0], [0.0, 1.0], weights[2].tolist()]))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    loss = logits.square().mean()
    loss.backward()
    assert model.expert_logits is not None
    for name in ("skeleton_expert", "imu_expert", "classwise_router"):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in getattr(model, name).parameters())
    assert parameter_count(model) < 150_000
    all_missing, all_missing_weights = model(skeleton[:1], imu[:1], torch.zeros(1, 8, 2))
    assert torch.isfinite(all_missing).all() and torch.isfinite(all_missing_weights).all()
    torch.testing.assert_close(all_missing_weights, torch.full((1, 2), 0.5))


def test_safesense_classprior_is_tiny_stable_and_finite() -> None:
    torch.manual_seed(9)
    model = SafeSense(variant="classprior")
    skeleton = torch.randn(3, 8, SKELETON_INPUTS)
    imu = torch.randn(3, 8, IMU_INPUTS)
    mask = torch.ones(3, 8, 2)
    mask[0, :, 0] = 0
    mask[1, :, 1] = 0
    logits, weights = model(skeleton, imu, mask)
    assert logits.shape == (3, 27) and weights.shape == (3, 2)
    assert torch.isfinite(logits).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(weights[:2], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    (logits.square().mean()).backward()
    assert model.classprior_router.grad is not None and torch.isfinite(model.classprior_router.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for module in (model.skeleton_expert, model.imu_expert) for parameter in module.parameters())
    assert parameter_count(model) < parameter_count(SafeSense(variant="classwise"))
    assert parameter_count(model) < 125_000
    missing, missing_weights = model(skeleton[:1], imu[:1], torch.zeros(1, 8, 2))
    assert torch.isfinite(missing).all() and torch.isfinite(missing_weights).all()


def test_safesense_aligned_zero_padding_and_inverse_lag_targets() -> None:
    values = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6, 1)
    plus = _zero_padded_shift_tensor(values, 2, time_dim=1)
    minus = _zero_padded_shift_tensor(values, -2, time_dim=1)
    torch.testing.assert_close(plus[:, :2], torch.zeros(1, 2, 1))
    torch.testing.assert_close(plus[:, 2:], values[:, :4])
    torch.testing.assert_close(minus[:, :4], values[:, 2:])
    torch.testing.assert_close(minus[:, -2:], torch.zeros(1, 2, 1))
    assert LAG_SHIFTS[_corrective_lag_target(4)] == -4
    assert LAG_SHIFTS[_corrective_lag_target(-3)] == 3
    assert LAG_SHIFTS[_corrective_lag_target(0)] == 0


def test_safesense_aligned_outputs_lag_logits_and_finite_gradients() -> None:
    torch.manual_seed(19)
    model = SafeSense(variant="aligned")
    skeleton = torch.randn(3, 16, SKELETON_INPUTS)
    imu = torch.randn(3, 16, IMU_INPUTS)
    mask = torch.ones(3, 16, 2)
    mask[0, :, 1] = 0
    logits, weights = model(skeleton, imu, mask)
    assert logits.shape == (3, 27)
    assert weights.shape == (3, 2)
    assert model.lag_logits is not None and model.lag_logits.shape == (3, len(LAG_SHIFTS))
    assert torch.isfinite(logits).all() and torch.isfinite(model.lag_logits).all()
    loss = logits.square().mean() + model.lag_logits.square().mean()
    loss.backward()
    for module in (model.skeleton_alignment, model.imu_alignment, model.reliability):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    assert parameter_count(model) > parameter_count(SafeSense(variant="classprior"))


def test_safesense_aligned_corruption_is_seed_deterministic() -> None:
    skeleton = torch.ones(12, 32, SKELETON_INPUTS)
    imu = torch.ones(12, 32, IMU_INPUTS)
    mask = torch.ones(12, 32)
    torch.manual_seed(77)
    first = _augment_aligned_batch(skeleton, imu, mask)
    torch.manual_seed(77)
    second = _augment_aligned_batch(skeleton, imu, mask)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)
    assert first[2].shape == (12, 32, 2)
    assert first[4].shape == first[5].shape == (12,)


def test_safesense_temporal_targets_and_internal_gap_repair() -> None:
    assert LAG_SHIFTS[_corrective_lag_target_for_modality(4, "imu")] == -4
    assert LAG_SHIFTS[_corrective_lag_target_for_modality(4, "skeleton")] == 4
    assert LAG_SHIFTS[_corrective_lag_target_for_modality(-3, "skeleton")] == -3
    values = torch.tensor([[[1.0], [2.0], [0.0], [4.0], [5.0], [0.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
    repaired = _repair_internal_gaps(values, mask, radius=2)
    torch.testing.assert_close(repaired[:, :2], values[:, :2])
    assert repaired[0, 2, 0] > 0
    assert repaired[0, 5, 0] == 0


def test_safesense_temporal_outputs_dense_lag_attention_and_finite_gradients() -> None:
    torch.manual_seed(29)
    model = SafeSense(variant="temporal")
    skeleton = torch.randn(3, 16, SKELETON_INPUTS)
    imu = torch.randn(3, 16, IMU_INPUTS)
    mask = torch.ones(3, 16, 2)
    mask[0, 5:8, 0] = 0
    mask[1, 7:10, 1] = 0
    mask[2] = 0
    logits, weights = model(skeleton, imu, mask)
    assert logits.shape == (3, 27)
    assert weights.shape == (3, 2)
    assert model.lag_logits is not None and model.lag_logits.shape == (3, len(LAG_SHIFTS))
    assert model.temporal_lag_logits is not None and model.temporal_lag_logits.shape == (3, 16, len(LAG_SHIFTS))
    assert model.lag_candidate_mask is not None and model.lag_candidate_mask.shape == (3, 16, len(LAG_SHIFTS))
    assert torch.isfinite(logits).all() and torch.isfinite(model.lag_logits).all()
    loss = logits.square().mean() + model.lag_logits.square().mean() + model.temporal_lag_logits.square().mean()
    loss.backward()
    for module in (
        model.temporal_skeleton_query,
        model.temporal_imu_key,
        model.temporal_imu_value,
        model.temporal_lag_head,
        model.temporal_reliability,
        model.temporal_fusion_projection,
        model.temporal_attention,
        model.temporal_ffn,
        model.temporal_classifier,
    ):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    assert model.temporal_logit_scale.grad is not None and torch.isfinite(model.temporal_logit_scale.grad)
    assert model.temporal_reliability_strength.grad is not None and torch.isfinite(model.temporal_reliability_strength.grad)
    assert parameter_count(model) > parameter_count(SafeSense(variant="classprior"))


def test_safesense_temporal_sanitizes_motion_across_missing_transitions() -> None:
    skeleton = torch.ones(1, 6, SKELETON_INPUTS)
    imu = torch.ones(1, 6, IMU_INPUTS)
    skeleton[:, :, 60:120] = 7
    imu[:, :, 6:12] = 9
    skeleton_mask = torch.ones(1, 6)
    imu_mask = torch.ones(1, 6)
    skeleton_mask[:, 2] = 0
    imu_mask[:, 4] = 0
    clean_skeleton, clean_imu = _sanitize_masked_motion_features(
        skeleton, imu, skeleton_mask, imu_mask
    )
    assert clean_skeleton[0, 2].abs().sum() == 0
    assert clean_skeleton[0, 3, 60:120].abs().sum() == 0
    assert clean_skeleton[0, 4, 60:120].abs().sum() > 0
    assert clean_imu[0, 4].abs().sum() == 0
    assert clean_imu[0, 5, 6:12].abs().sum() == 0


def test_safesense_temporal_global_lag_head_is_auxiliary_only() -> None:
    torch.manual_seed(30)
    model = SafeSense(variant="temporal").eval()
    skeleton = torch.randn(2, 16, SKELETON_INPUTS)
    imu = torch.randn(2, 16, IMU_INPUTS)
    mask = torch.ones(2, 16, 2)
    with torch.no_grad():
        before_logits, _ = model(skeleton, imu, mask)
        before_lag = model.lag_logits.clone()
        model.temporal_lag_head[-1].bias.add_(25.0 * torch.arange(len(LAG_SHIFTS)))
        after_logits, _ = model(skeleton, imu, mask)
        after_lag = model.lag_logits.clone()
    torch.testing.assert_close(after_logits, before_logits)
    assert not torch.allclose(after_lag, before_lag)


def test_safesense_temporal_corruption_is_seed_deterministic() -> None:
    skeleton = torch.ones(24, 32, SKELETON_INPUTS)
    imu = torch.ones(24, 32, IMU_INPUTS)
    mask = torch.ones(24, 32)
    torch.manual_seed(91)
    first = _augment_temporal_batch(skeleton, imu, mask)
    torch.manual_seed(91)
    second = _augment_temporal_batch(skeleton, imu, mask)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)
    assert first[2].shape == (24, 32, 2)
    assert first[4].shape == first[5].shape == (24,)
    assert torch.any(first[0] != skeleton) or torch.any(first[1] != imu)


def test_safesense_structured_corruption_rebuilds_physical_features_deterministically() -> None:
    torch.manual_seed(101)
    skeleton = torch.randn(48, 24, SKELETON_INPUTS)
    imu = torch.randn(48, 24, IMU_INPUTS)
    mask = torch.ones(48, 24, 2)
    stats = {
        "skeleton_mean": torch.zeros(SKELETON_INPUTS),
        "skeleton_std": torch.ones(SKELETON_INPUTS),
        "imu_mean": torch.zeros(IMU_INPUTS),
        "imu_std": torch.ones(IMU_INPUTS),
        "skeleton_noise_scale": torch.tensor(1.0),
        "imu_noise_scale": torch.tensor(1.0),
    }
    torch.manual_seed(102)
    first = _augment_structured_temporal_batch(skeleton, imu, mask, stats)
    torch.manual_seed(102)
    second = _augment_structured_temporal_batch(skeleton, imu, mask, stats)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)
    corrupted_skeleton, corrupted_imu, pair_mask = first[:3]
    assert first[6].shape == (48, 2)
    assert first[7].shape == (48, 2)
    assert torch.all((first[7] == 0) | (first[7] == 1))
    assert first[7].sum() > 0
    skeleton_transition = pair_mask[:, 1:, 0].bool() & pair_mask[:, :-1, 0].bool()
    imu_transition = pair_mask[:, 1:, 1].bool() & pair_mask[:, :-1, 1].bool()
    expected_skeleton_delta = corrupted_skeleton[:, 1:, :60] - corrupted_skeleton[:, :-1, :60]
    expected_imu_delta = corrupted_imu[:, 1:, :6] - corrupted_imu[:, :-1, :6]
    torch.testing.assert_close(
        corrupted_skeleton[:, 1:, 60:120][skeleton_transition],
        expected_skeleton_delta[skeleton_transition],
    )
    torch.testing.assert_close(
        corrupted_imu[:, 1:, 6:12][imu_transition],
        expected_imu_delta[imu_transition],
    )
    joints = corrupted_skeleton[:, :, :60].reshape(48, 24, 20, 3)
    valid = pair_mask[:, :, 0].bool()
    for edge_index, (child, parent) in enumerate(((0, 1), (2, 1), (4, 1))):
        bone = corrupted_skeleton[:, :, 120 + edge_index * 3:123 + edge_index * 3]
        torch.testing.assert_close(bone[valid], (joints[:, :, child] - joints[:, :, parent])[valid])


def test_safesense_robust_topology_denoising_is_finite_trainable_and_compact() -> None:
    torch.manual_seed(103)
    model = SafeSense(variant="robust")
    skeleton = torch.randn(3, 20, SKELETON_INPUTS)
    imu = torch.randn(3, 20, IMU_INPUTS)
    mask = torch.ones(3, 20, 2)
    mask[0, 4:9, 0] = 0
    mask[1, :, 1] = 0
    mask[2] = 0
    logits, weights = model(skeleton, imu, mask)
    assert logits.shape == (3, 27) and weights.shape == (3, 2)
    assert torch.isfinite(logits).all() and torch.isfinite(weights).all()
    assert model.temporal_embedding is not None and model.temporal_embedding.shape == (3, 384)
    assert model.denoised_inputs is not None
    assert model.correction_gate is not None and model.correction_gate.shape == (3, 2)
    assert torch.isfinite(model.correction_gate).all()
    assert torch.all((model.correction_gate > 0) & (model.correction_gate < 1))
    assert model.effective_correction_gate is not None
    assert model.effective_correction_gate.shape == model.correction_gate.shape
    assert torch.all(model.effective_correction_gate <= model.correction_gate)
    loss = logits.square().mean() + model.temporal_embedding.square().mean()
    loss.backward()
    assert model.robust_topology_strength.grad is not None
    assert model.robust_quality_router[-1].weight.grad is not None
    assert model.robust_skeleton_denoiser.project_out.weight.grad is not None
    assert model.robust_correction_gate[-1].weight.grad is not None
    assert all(torch.isfinite(value.grad).all() for value in (
        model.robust_topology_strength,
        model.robust_quality_router[-1].weight,
        model.robust_skeleton_denoiser.project_out.weight,
        model.robust_correction_gate[-1].weight,
    ))
    assert parameter_count(SafeSense(variant="temporal")) < parameter_count(model) < 1_036_990


def test_safesense_robust_topology_residual_scales_with_skeleton_coverage(tmp_path: Path) -> None:
    torch.manual_seed(106)
    temporal = SafeSense(variant="temporal").eval()
    checkpoint = tmp_path / "temporal-coverage.pt"
    torch.save(temporal.state_dict(), checkpoint)
    robust = SafeSense(variant="robust").eval()
    _initialize_robust_adapter(robust, checkpoint)
    skeleton = torch.randn(2, 20, SKELETON_INPUTS)
    imu = torch.randn(2, 20, IMU_INPUTS)
    full_mask = torch.ones(2, 20, 2)
    half_mask = full_mask.clone()
    half_mask[:, 10:, 0] = 0
    equally_padded_mask = full_mask.clone()
    equally_padded_mask[:, 10:] = 0
    with torch.no_grad():
        robust.robust_correction_gate[-1].weight.zero_()
        robust.robust_correction_gate[-1].bias.fill_(-20.0)
        robust.robust_quality_router[-1].weight.zero_()
        robust.robust_quality_router[-1].bias.zero_()
        robust.robust_topology_classifier[-1].weight.zero_()
        robust.robust_topology_classifier[-1].bias.fill_(1.0)
        robust.robust_topology_logit_scale.fill_(0.4)
        robust.robust_topology_strength.zero_()
        temporal_full, _ = temporal(skeleton, imu, full_mask)
        robust_full, _ = robust(skeleton, imu, full_mask)
        temporal_half, _ = temporal(skeleton, imu, half_mask)
        robust_half, _ = robust(skeleton, imu, half_mask)
        temporal_padded, _ = temporal(skeleton, imu, equally_padded_mask)
        robust_padded, _ = robust(skeleton, imu, equally_padded_mask)
    full_delta = robust_full - temporal_full
    half_delta = robust_half - temporal_half
    padded_delta = robust_padded - temporal_padded
    torch.testing.assert_close(half_delta, 0.25 * full_delta, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(padded_delta, full_delta, rtol=1e-5, atol=1e-5)


def test_safesense_robust_fault_prior_routes_away_from_detected_noise() -> None:
    torch.manual_seed(105)
    model = SafeSense(variant="robust").eval()
    skeleton = torch.randn(2, 20, SKELETON_INPUTS)
    imu = torch.randn(2, 20, IMU_INPUTS)
    mask = torch.ones(2, 20, 2)
    with torch.no_grad():
        model.temporal_reliability[-1].weight.zero_()
        model.temporal_reliability[-1].bias.zero_()
        model.robust_quality_router[-1].weight.zero_()
        model.robust_quality_router[-1].bias.zero_()
        model.classprior_router.zero_()
        model.robust_correction_gate[-1].weight.zero_()
        model.robust_correction_gate[-1].bias.fill_(-10.0)
        _, clean_weights = model(skeleton, imu, mask)
        model.robust_correction_gate[-1].bias.copy_(torch.tensor([10.0, -10.0]))
        _, skeleton_fault_weights = model(skeleton, imu, mask)
    assert clean_weights[:, 0].mean() > 0.45
    assert skeleton_fault_weights[:, 0].mean() < 0.05
    assert skeleton_fault_weights[:, 1].mean() > 0.95


def test_safesense_robust_adapter_loads_and_freezes_temporal_backbone(tmp_path: Path) -> None:
    torch.manual_seed(104)
    temporal = SafeSense(variant="temporal")
    checkpoint = tmp_path / "temporal.pt"
    torch.save(temporal.state_dict(), checkpoint)
    robust = SafeSense(variant="robust")
    trainable = _initialize_robust_adapter(robust, checkpoint)
    assert trainable and all(name.startswith("robust_") for name in trainable)
    assert all(parameter.requires_grad == name.startswith("robust_") for name, parameter in robust.named_parameters())
    torch.testing.assert_close(robust.skeleton_branch.input.weight, temporal.skeleton_branch.input.weight)
    torch.testing.assert_close(robust.temporal_classifier[-1].weight, temporal.temporal_classifier[-1].weight)
    temporal.eval()
    robust.eval()
    skeleton = torch.randn(2, 20, SKELETON_INPUTS)
    imu = torch.randn(2, 20, IMU_INPUTS)
    mask = torch.ones(2, 20, 2)
    with torch.no_grad():
        temporal(skeleton, imu, mask)
        temporal_lag = temporal.lag_logits.clone()
        robust(skeleton, imu, mask)
        robust_lag = robust.lag_logits.clone()
    torch.testing.assert_close(robust_lag, temporal_lag)

    missing_skeleton = mask.clone()
    missing_skeleton[:, :, 0] = 0
    with torch.no_grad():
        before_missing, _ = robust(skeleton, imu, missing_skeleton)
        robust.robust_topology_classifier[-1].bias.add_(100.0)
        after_missing, _ = robust(skeleton, imu, missing_skeleton)
    torch.testing.assert_close(after_missing, before_missing)
    assert sum(parameter.numel() for parameter in robust.parameters() if parameter.requires_grad) < 250_000
    assert sum(parameter.numel() for parameter in robust.parameters()) < 800_000


def test_conditional_ensemble_routes_only_fault_or_missing_inputs() -> None:
    class Primary(torch.nn.Module):
        def __init__(self):
            super().__init__(); self._gate = None
        def forward(self, skeleton, imu, mask, availability=None):
            self._gate = skeleton.new_full((len(skeleton), 2), 0.01)
            high = skeleton[:, 0, 0] > 0.5
            self._gate[high] = 0.8
            logits = skeleton.new_zeros((len(skeleton), 27)); logits[:, 0] = 3.0; logits[:, 1] = 1.0
            return logits, skeleton.new_full((len(skeleton), 2), 0.5)
        @property
        def effective_correction_gate(self): return self._gate
    class Auxiliary(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = 0; self.rows = 0
        def forward(self, skeleton, imu, mask, availability=None):
            self.calls += 1; self.rows += len(skeleton)
            logits = skeleton.new_zeros((len(skeleton), 27)); logits[:, 1] = 3.0
            return logits, skeleton.new_full((len(skeleton), 2), 0.5)

    primary, auxiliary = Primary(), Auxiliary()
    model = SafeSenseConditionalEnsemble(primary, auxiliary, gate_threshold=0.15)
    skeleton = torch.zeros(3, 8, SKELETON_INPUTS); imu = torch.zeros(3, 8, IMU_INPUTS)
    mask = torch.ones(3, 8, 2)
    skeleton[1, 0, 0] = 1.0  # high gate + equal masks => route
    mask[2, :, 1] = 0.0      # missing IMU => route regardless of gate
    output = model(skeleton, imu, mask)
    assert model.auxiliary_route.tolist() == [False, True, True]
    assert auxiliary.calls == 1 and auxiliary.rows == 2
    primary_prob = torch.softmax(torch.tensor([3.0, 1.0] + [0.0] * 25), dim=0)
    auxiliary_prob = torch.softmax(torch.tensor([0.0, 3.0] + [0.0] * 25), dim=0)
    torch.testing.assert_close(output[0].exp(), primary_prob)
    torch.testing.assert_close(output[1].exp(), 0.5 * primary_prob + 0.5 * auxiliary_prob)

    auxiliary.calls = auxiliary.rows = 0
    shifted_mask = torch.ones(1, 8, 2); shifted_mask[:, :2, 1] = 0
    shifted_skeleton = torch.ones(1, 8, SKELETON_INPUTS)
    shifted_imu = torch.zeros(1, 8, IMU_INPUTS)
    model(shifted_skeleton, shifted_imu, shifted_mask)
    assert model.auxiliary_route.tolist() == [False]
    assert auxiliary.calls == 0


def test_branch_expert_exactly_matches_frozen_robust_expert_logits() -> None:
    torch.manual_seed(230)
    robust = SafeSense(variant="robust").eval()
    expert = SafeSenseBranchExpert.from_robust_state_dict(robust.state_dict()).eval()
    skeleton = torch.randn(4, 32, SKELETON_INPUTS)
    imu = torch.randn(4, 32, IMU_INPUTS)
    mask = torch.ones(4, 32, 2)
    use_skeleton = torch.tensor([True, False, True, False])
    with torch.no_grad():
        robust(skeleton, imu, mask)
        expected_experts = robust.expert_logits
        assert expected_experts is not None
        expected = torch.where(use_skeleton[:, None], expected_experts[0], expected_experts[1])
        actual = expert(skeleton, imu, mask, use_skeleton)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)
    robust_state = robust.state_dict()
    for name, value in expert.state_dict().items():
        torch.testing.assert_close(value, robust_state[name], rtol=0.0, atol=0.0)
    assert parameter_count(expert) == 89_622


def test_early_exit_sparse_route_preserves_clean_primary_and_routes_noise_missing() -> None:
    torch.manual_seed(231)
    primary = SafeSense(variant="robust").eval()
    auxiliary_source = SafeSense(variant="robust").eval()
    auxiliary = SafeSenseBranchExpert.from_robust_state_dict(auxiliary_source.state_dict()).eval()
    model = SafeSenseEarlyExitEnsemble(primary, auxiliary).eval()

    clean_skeleton = torch.zeros(1, 128, SKELETON_INPUTS)
    clean_imu = torch.zeros(1, 128, IMU_INPUTS)
    full_mask = torch.ones(1, 128, 2)
    with torch.no_grad():
        primary_logits, _ = primary(clean_skeleton, clean_imu, full_mask)
        output = model(clean_skeleton, clean_imu, full_mask)
    assert model.auxiliary_route is not None and model.auxiliary_route.tolist() == [False]
    torch.testing.assert_close(output.exp(), torch.softmax(primary_logits, dim=1))

    noisy_imu = clean_imu.clone()
    noisy_imu[:, ::16, :3] = torch.arange(8, dtype=torch.float32)[None, :, None] * 20.0
    with torch.no_grad():
        model(clean_skeleton, noisy_imu, full_mask)
    assert model.auxiliary_route.tolist() == [True]
    assert model.use_skeleton is not None and model.use_skeleton.tolist() == [True]

    mismatch_mask = full_mask.clone()
    mismatch_mask[:, :8, 0] = 0
    with torch.no_grad():
        model(clean_skeleton, noisy_imu, mismatch_mask)
    assert model.auxiliary_route.tolist() == [False]

    missing_imu = full_mask.clone()
    missing_imu[:, :, 1] = 0
    with torch.no_grad():
        model(clean_skeleton, clean_imu, missing_imu)
    assert model.auxiliary_route.tolist() == [True]
    assert model.use_skeleton.tolist() == [True]
    assert parameter_count(model) == 805_105


def test_safesense_variants_respect_requested_availability() -> None:
    skeleton = torch.randn(2, 8, SKELETON_INPUTS)
    imu = torch.randn(2, 8, IMU_INPUTS)
    availability = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    for variant, requested in (("skeleton", 0), ("imu", 1)):
        _, weights = SafeSense(variant=variant)(skeleton, imu, availability=availability)
        assert torch.equal(weights[0], torch.tensor([0.0, 0.0]) if requested == 0 else torch.tensor([0.0, 1.0]))
        assert torch.equal(weights[1], torch.zeros(2))


def test_safesense_shared_adapter_epoch_requires_clean_safe_and_maximizes_mean_score() -> None:
    records = [
        {"history": [
            {"epoch": 1, "selection_score": 0.80, "robust_selection_accuracy": 0.70, "validation_accuracy": 0.90, "clean_safe": 1.0},
            {"epoch": 2, "selection_score": 0.92, "robust_selection_accuracy": 0.82, "validation_accuracy": 0.91, "clean_safe": 1.0},
            {"epoch": 3, "selection_score": 0.99, "robust_selection_accuracy": 0.90, "validation_accuracy": 0.80, "clean_safe": 0.0},
        ]},
        {"history": [
            {"epoch": 1, "selection_score": 0.84, "robust_selection_accuracy": 0.74, "validation_accuracy": 0.91, "clean_safe": 1.0},
            {"epoch": 2, "selection_score": 0.90, "robust_selection_accuracy": 0.80, "validation_accuracy": 0.90, "clean_safe": 1.0},
            {"epoch": 3, "selection_score": 1.00, "robust_selection_accuracy": 0.91, "validation_accuracy": 0.81, "clean_safe": 1.0},
        ]},
    ]
    epoch, score, robust_accuracy, clean_accuracy = _select_shared_adapter_epoch(records)
    assert epoch == 2
    assert abs(score - 0.91) < 1e-12
    assert abs(robust_accuracy - 0.81) < 1e-12
    assert abs(clean_accuracy - 0.905) < 1e-12


def test_safesense_clean_first_adapter_epoch_uses_robust_clean_f1_then_earliest() -> None:
    records = [
        {"history": [
            {"epoch": 1, "robust_selection_accuracy": 0.80, "validation_accuracy": 0.91, "validation_macro_f1": 0.90, "clean_safe": True},
            {"epoch": 2, "robust_selection_accuracy": 0.80, "validation_accuracy": 0.92, "validation_macro_f1": 0.90, "clean_safe": True},
            {"epoch": 3, "robust_selection_accuracy": 0.95, "validation_accuracy": 0.93, "validation_macro_f1": 0.93, "clean_safe": False},
            {"epoch": 4, "robust_selection_accuracy": 0.80, "validation_accuracy": 0.92, "validation_macro_f1": 0.91, "clean_safe": True},
            {"epoch": 5, "robust_selection_accuracy": 0.80, "validation_accuracy": 0.92, "validation_macro_f1": 0.91, "clean_safe": True},
        ]},
        {"history": [
            {"epoch": 1, "robust_selection_accuracy": 0.82, "validation_accuracy": 0.91, "validation_macro_f1": 0.90, "clean_safe": True},
            {"epoch": 2, "robust_selection_accuracy": 0.82, "validation_accuracy": 0.92, "validation_macro_f1": 0.90, "clean_safe": True},
            {"epoch": 3, "robust_selection_accuracy": 0.96, "validation_accuracy": 0.93, "validation_macro_f1": 0.93, "clean_safe": True},
            {"epoch": 4, "robust_selection_accuracy": 0.82, "validation_accuracy": 0.92, "validation_macro_f1": 0.91, "clean_safe": True},
            {"epoch": 5, "robust_selection_accuracy": 0.82, "validation_accuracy": 0.92, "validation_macro_f1": 0.91, "clean_safe": True},
        ]},
    ]
    epoch, robust_accuracy, clean_accuracy, macro_f1 = _select_clean_first_adapter_epoch(records)
    assert epoch == 4
    assert abs(robust_accuracy - 0.81) < 1e-12
    assert abs(clean_accuracy - 0.92) < 1e-12
    assert abs(macro_f1 - 0.91) < 1e-12


def test_safesense_shared_temporal_epoch_maximizes_mean_validation_accuracy() -> None:
    records = [
        {"history": [
            {"epoch": 1, "validation_accuracy": 0.80, "validation_macro_f1": 0.79},
            {"epoch": 2, "validation_accuracy": 0.90, "validation_macro_f1": 0.88},
            {"epoch": 3, "validation_accuracy": 0.90, "validation_macro_f1": 0.89},
        ]},
        {"history": [
            {"epoch": 1, "validation_accuracy": 0.90, "validation_macro_f1": 0.88},
            {"epoch": 2, "validation_accuracy": 0.90, "validation_macro_f1": 0.90},
            {"epoch": 3, "validation_accuracy": 0.90, "validation_macro_f1": 0.89},
        ]},
    ]
    epoch, accuracy, macro_f1 = _select_shared_frozen_epoch(records)
    assert epoch == 2
    assert accuracy == 0.90
    assert macro_f1 == 0.89


def test_safesense_development_isolated_and_freezes_median_epoch(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    result = run_safesense_development(
        prepared,
        tmp_path / "development",
        config=SafeSenseTrainingConfig(epochs=1, patience=1, batch_size=1),
        seeds=(11, 22, 33),
    )
    assert result["seeds"] == [11, 22, 33]
    assert result["frozen_epoch"] == 1
    assert result["split_counts"] == {"development_train": 1, "development_validation": 1}
    assert "official_test_subjects" not in result
    assert "test_loss" not in result
    assert "best_validation_macro_f1" in result["seed_results"][0]
    assert all((tmp_path / "development" / f"seed-{seed}" / "checkpoint.pt").is_file() for seed in (11, 22, 33))


def test_safesense_fixed_epoch_replays_development_training_trajectory(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    config = SafeSenseTrainingConfig(epochs=1, patience=1, batch_size=1)
    selected = run_safesense_development(
        prepared,
        tmp_path / "selected",
        config=config,
        seeds=(11,),
        variant="temporal",
    )
    fixed = run_safesense_development_fixed_epoch(
        prepared,
        tmp_path / "fixed",
        frozen_epoch=1,
        config=config,
        seeds=(11,),
        variant="temporal",
    )
    assert fixed["seed_results"][0]["best_validation_accuracy"] == selected["seed_results"][0]["best_validation_accuracy"]
    assert fixed["seed_results"][0]["best_validation_macro_f1"] == selected["seed_results"][0]["best_validation_macro_f1"]
    selected_state = torch.load(tmp_path / "selected/seed-11/checkpoint.pt", map_location="cpu", weights_only=True)
    fixed_state = torch.load(tmp_path / "fixed/seed-11/checkpoint.pt", map_location="cpu", weights_only=True)
    assert selected_state.keys() == fixed_state.keys()
    for name in selected_state:
        torch.testing.assert_close(fixed_state[name], selected_state[name])


def test_safesense_fixed_adapter_replays_adapter_development_trajectory(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    config = SafeSenseTrainingConfig(epochs=1, patience=1, batch_size=1)
    temporal_dir = tmp_path / "temporal-base"
    run_safesense_development_fixed_epoch(
        prepared,
        temporal_dir,
        frozen_epoch=1,
        config=config,
        seeds=(11,),
        variant="temporal",
    )
    selected_dir = tmp_path / "adapter-selected"
    run_safesense_robust_adapter_development(
        prepared,
        selected_dir,
        temporal_dir,
        config=config,
        seeds=(11,),
    )
    fixed_dir = tmp_path / "adapter-fixed"
    fixed = run_safesense_robust_adapter_fixed_epoch(
        prepared,
        fixed_dir,
        temporal_dir,
        adapter_epoch=1,
        config=config,
        seeds=(11,),
    )
    assert fixed["frozen_epoch"] == 1
    assert fixed["training_mode"] == "fixed_epoch_frozen_temporal_backbone_robust_adapter"
    selected_state = torch.load(selected_dir / "seed-11/checkpoint.pt", map_location="cpu", weights_only=True)
    fixed_state = torch.load(fixed_dir / "seed-11/checkpoint.pt", map_location="cpu", weights_only=True)
    assert selected_state.keys() == fixed_state.keys()
    for name in selected_state:
        torch.testing.assert_close(fixed_state[name], selected_state[name])


def test_safesense_fixed_epoch_development_never_opens_even_subject_arrays(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    (prepared / "skeleton_val_features.npy").unlink()
    (prepared / "val_labels.npy").unlink()
    result = run_safesense_development_fixed_epoch(
        prepared,
        tmp_path / "fixed-development",
        frozen_epoch=1,
        config=SafeSenseTrainingConfig(batch_size=1),
        seeds=(11,),
        variant="temporal",
    )
    assert result["training_mode"] == "fixed_epoch_development_control"
    assert result["frozen_epoch"] == 1
    assert result["development_train_subjects"] == [1, 3, 5]
    assert result["development_validation_subjects"] == [7]
    assert "official_test_subjects" not in result
    assert result["seed_results"][0]["checkpoint_policy"].startswith("fixed development epoch")


def test_safesense_two_stage_final_uses_fixed_temporal_and_adapter_epochs(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    result = run_safesense_robust_two_stage_final(
        prepared,
        tmp_path / "two-stage-final",
        temporal_epoch=1,
        adapter_epoch=1,
        config=SafeSenseTrainingConfig(batch_size=1),
        seeds=(11,),
    )
    assert result["training_mode"] == "two_stage_fixed_temporal_and_adapter"
    assert result["temporal_epoch"] == 1
    assert result["adapter_epoch"] == 1
    assert result["held_out_evaluation_after_all_training_completed"] is True
    assert result["training_subjects"] == [1, 3, 5, 7]
    assert result["official_test_subjects"] == [2, 4, 6, 8]
    assert result["seed_results"][0]["seed"] == 11
    assert (tmp_path / "two-stage-final/seed-11/temporal-checkpoint.pt").is_file()
    assert (tmp_path / "two-stage-final/seed-11/checkpoint.pt").is_file()


def test_safesense_final_uses_supplied_frozen_epoch_for_each_seed(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    result = run_safesense_final(
        prepared,
        tmp_path / "final",
        frozen_epoch=1,
        config=SafeSenseTrainingConfig(epochs=120, batch_size=2, learning_rate=2e-3, weight_decay=3e-4),
        seeds=(11, 22),
        variant="equal",
    )
    assert result["frozen_epoch"] == 1
    assert result["variant"] == "equal"
    assert result["config"]["epochs"] == 1
    assert result["config"]["batch_size"] == 2
    assert result["config"]["learning_rate"] == 2e-3
    assert result["training_subjects"] == [1, 3, 5, 7]
    assert [item["seed"] for item in result["seed_results"]] == [11, 22]
    assert set(result["aggregate"]) == {
        "accuracy_mean", "accuracy_std_population", "macro_f1_mean", "macro_f1_std_population"
    }
    assert result["aggregate"]["accuracy_std_population"] >= 0
    assert all((tmp_path / "final" / f"seed-{seed}" / "checkpoint.pt").is_file() for seed in (11, 22))


def test_safesense_one_epoch_smoke_is_reproducible(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    first = safesense_smoke(prepared, tmp_path / "first", device="cpu")
    second = safesense_smoke(prepared, tmp_path / "second", device="cpu")
    assert first["status"] == second["status"] == "completed"
    assert first["accuracy"] == second["accuracy"]
    assert json.loads((tmp_path / "first/result.json").read_text())["config"]["epochs"] == 1


def test_safesense_classwise_training_is_deterministic_and_records_auxiliary_loss(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path / "prepared")
    config = SafeSenseTrainingConfig(epochs=1, patience=1, batch_size=1)
    first = run_safesense_development(prepared, tmp_path / "first", config=config, seeds=(11,), variant="classwise")
    second = run_safesense_development(prepared, tmp_path / "second", config=config, seeds=(11,), variant="classwise")
    assert first["auxiliary_ce_weight_applied"] == second["auxiliary_ce_weight_applied"] == 0.2
    assert first["seed_results"][0]["best_validation_accuracy"] == second["seed_results"][0]["best_validation_accuracy"]


def _canonicalization_fixture() -> tuple[np.ndarray, np.ndarray]:
    skeleton = np.zeros((1, 128, SKELETON_INPUTS), dtype=np.float32)
    mask = np.zeros((1, 128), dtype=np.float32)
    mask[:, :4] = 1.0
    offsets = np.asarray(
        [[0.04 * joint, 0.03 * (joint % 5), 0.02 * (joint % 3)] for joint in range(20)],
        dtype=np.float32,
    )
    offsets[0] = (0.0, 0.0, 0.0)
    offsets[2] = (0.0, 1.0, 0.0)
    offsets[4] = (-1.0, 0.6, 0.0)
    offsets[8] = (1.0, 0.6, 0.0)
    positions = skeleton[:, :, :60].reshape(1, 128, 20, 3)
    for frame in range(4):
        root = np.asarray((0.12 * frame, 0.01 * frame, 0.05 * frame), dtype=np.float32)
        positions[0, frame] = offsets + root
        positions[0, frame, 7, 1] += 0.08 * frame
    positions[0, 4:] = 37.0
    return skeleton, mask


def test_body_canonicalization_preserves_shape_mask_and_rebuilds_features() -> None:
    skeleton, mask = _canonicalization_fixture()
    canonical = _canonicalize_skeleton_features(skeleton, mask)
    assert canonical.shape == skeleton.shape
    assert canonical.dtype == np.float32
    assert np.isfinite(canonical).all()
    assert np.count_nonzero(canonical[:, 4:]) == 0
    assert np.linalg.norm(canonical[0, 1, 60:120]) > 0
    assert np.linalg.norm(canonical[0, :4, 120:]) > 0


def test_body_canonicalization_is_translation_yaw_and_scale_invariant() -> None:
    skeleton, mask = _canonicalization_fixture()
    reference = _canonicalize_skeleton_features(skeleton, mask)
    positions = skeleton[:, :, :60].reshape(1, 128, 20, 3)

    translated = skeleton.copy()
    translated[:, :, :60] = (positions + np.asarray((4.5, -2.0, 7.25), dtype=np.float32)).reshape(1, 128, 60)
    np.testing.assert_allclose(_canonicalize_skeleton_features(translated, mask), reference, rtol=2e-5, atol=2e-5)

    angle = 0.73
    c, sn = np.cos(angle), np.sin(angle)
    rotation = np.asarray(((c, 0.0, sn), (0.0, 1.0, 0.0), (-sn, 0.0, c)), dtype=np.float32)
    yawed = skeleton.copy()
    yawed[:, :, :60] = (positions @ rotation.T).reshape(1, 128, 60)
    np.testing.assert_allclose(_canonicalize_skeleton_features(yawed, mask), reference, rtol=2e-5, atol=2e-5)

    scaled = skeleton.copy()
    scaled[:, :, :60] = (positions * np.float32(2.75)).reshape(1, 128, 60)
    np.testing.assert_allclose(_canonicalize_skeleton_features(scaled, mask), reference, rtol=2e-5, atol=2e-5)


def test_body_canonicalization_stats_ignore_validation_subject(tmp_path: Path) -> None:
    prepared = tmp_path / "canonical-stats"
    prepared.mkdir()
    rows = [
        {"sample_id": f"a1_s{subject}_t1", "action": 1, "subject": subject, "trial": 1, "split": "train"}
        for subject in (1, 3, 5, 7)
    ]
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader()
        writer.writerows(rows)
    values = np.zeros((4, 1, 128, 22, 3), dtype=np.float32)
    for sample in range(4):
        for frame in range(4):
            root = np.asarray((0.1 * frame, 0.0, 0.04 * frame), dtype=np.float32)
            values[sample, 0, frame, :20] = np.asarray(
                [[0.04 * joint, 0.03 * (joint % 5), 0.02 * (joint % 3)] for joint in range(20)],
                dtype=np.float32,
            ) + root
            values[sample, 0, frame, 0] = root
            values[sample, 0, frame, 2] = root + np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
            values[sample, 0, frame, 4] = root + np.asarray((-1.0, 0.6, 0.0), dtype=np.float32)
            values[sample, 0, frame, 8] = root + np.asarray((1.0, 0.6, 0.0), dtype=np.float32)
            values[sample, 0, frame, 20:] = 1.0
    np.save(prepared / "skeleton_train_features.npy", values)
    np.save(prepared / "train_labels.npy", np.zeros(4, dtype=np.int64))
    splits = safesense_odd_fold_splits(prepared, 7)
    skeleton_before, _, _, _, stats_before = _development_data(prepared, splits, body_canonicalization=True)

    changed = values.copy()
    changed[3, 0, :4, 7, 1] += 100.0
    np.save(prepared / "skeleton_train_features.npy", changed)
    skeleton_after, _, _, _, stats_after = _development_data(prepared, splits, body_canonicalization=True)

    assert not np.allclose(skeleton_before[splits.development_valid], skeleton_after[splits.development_valid])
    for name in ("skeleton_mean", "skeleton_std", "skeleton_noise_scale", "imu_mean", "imu_std", "imu_noise_scale"):
        np.testing.assert_allclose(stats_before[name], stats_after[name], rtol=0, atol=0)


def test_odd_fold_splits_cover_only_odd_development_subjects(tmp_path) -> None:
    import csv

    prepared = tmp_path / "odd-fold-splits"
    prepared.mkdir()
    rows = []
    for subject in (1, 3, 5, 7, 2, 4, 6, 8):
        rows.append({"sample_id": f"a1_s{subject}_t1", "action": 1, "subject": subject, "trial": 1, "split": "train" if subject % 2 else "test"})
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader(); writer.writerows(rows)
    for validation_subject in (1, 3, 5, 7):
        splits = safesense_odd_fold_splits(prepared, validation_subject)
        assert len(splits.development_train) == 3
        assert len(splits.development_valid) == 1
        assert len(splits.official_test) == 4
        assert np.intersect1d(splits.development_train, splits.development_valid).size == 0
        assert np.intersect1d(np.concatenate((splits.development_train, splits.development_valid)), splits.official_test).size == 0


def test_odd_fold_crossval_never_requires_even_arrays(tmp_path) -> None:
    import csv

    prepared = tmp_path / "odd-fold-run"
    prepared.mkdir()
    rows = []
    for i, subject in enumerate((1, 3, 5, 7, 2)):
        rows.append({"sample_id": f"a1_s{subject}_t1", "action": 1, "subject": subject, "trial": 1, "split": "train" if subject % 2 else "test"})
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader(); writer.writerows(rows)
    values = np.zeros((4, 1, 128, 22, 3), dtype=np.float32)
    values[:, 0, :16, :20, 0] = 1.0
    values[:, 0, :16, 20:, 1] = 1.0
    np.save(prepared / "skeleton_train_features.npy", values)
    np.save(prepared / "train_labels.npy", np.zeros(4, dtype=np.int64))
    result = run_safesense_odd_fold_two_stage_crossval(
        prepared, tmp_path / "odd-fold-out", temporal_epoch=1, adapter_epoch=1,
        seeds=(11,), validation_subjects=(7,), subject_style_training=True, body_canonicalization=True,
        config=SafeSenseTrainingConfig(epochs=1, patience=1, batch_size=4), device="cpu",
    )
    assert result["official_even_subjects_loaded"] is False
    assert result["subject_style_training"] is True
    assert result["body_canonicalization"] is True
    assert result["validation_subjects"] == [7]
    assert result["fold_results"][0]["training_subjects"] == [1, 3, 5]
    assert not (prepared / "skeleton_val_features.npy").exists()


def test_factorized_odd_fold_full_training_never_requires_even_arrays_and_logs_progress(tmp_path) -> None:
    prepared = tmp_path / "factorized-odd-fold"
    prepared.mkdir()
    rows = [
        {"sample_id": f"a1_s{subject}_t1", "action": 1, "subject": subject, "trial": 1, "split": "train" if subject % 2 else "test"}
        for subject in (1, 3, 5, 7, 2)
    ]
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader()
        writer.writerows(rows)
    values = np.zeros((4, 1, 128, 22, 3), dtype=np.float32)
    values[:, 0, :16, :20, 0] = 1.0
    values[:, 0, :16, 20:, 1] = 1.0
    np.save(prepared / "skeleton_train_features.npy", values)
    np.save(prepared / "train_labels.npy", np.zeros(4, dtype=np.int64))
    output = tmp_path / "factorized-out"
    result = run_safesense_factorized_odd_fold(
        prepared,
        output,
        validation_subject=7,
        epochs=1,
        seeds=(11,),
        progress_interval_seconds=0.05,
        config=SafeSenseTrainingConfig(epochs=1, patience=1, batch_size=4),
        device="cpu",
    )
    assert result["official_even_subjects_loaded"] is False
    assert result["validation_subject"] == 7
    assert result["training_subjects"] == [1, 3, 5]
    assert result["epochs"] == 1
    assert len(result["seed_results"]) == 1
    progress = (output / "progress.log").read_text()
    assert "run-start" in progress
    assert "run-completed" in progress
    assert "heartbeat" in progress
    assert not (prepared / "skeleton_val_features.npy").exists()


def test_update_swa_state_averages_completed_model_states() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    state, count = _update_swa_state(model, None, 0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    state, count = _update_swa_state(model, state, count)
    assert count == 2
    assert torch.allclose(state["weight"], torch.full_like(state["weight"], 2.0))



def test_odd_fold_adapter_history_never_requires_even_arrays(tmp_path) -> None:
    prepared = tmp_path / "odd-fold-adapter-history"
    prepared.mkdir()
    rows = []
    for subject in (1, 3, 5, 7, 2):
        rows.append({"sample_id": f"a1_s{subject}_t1", "action": 1, "subject": subject, "trial": 1, "split": "train" if subject % 2 else "test"})
    with (prepared / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "action", "subject", "trial", "split"))
        writer.writeheader()
        writer.writerows(rows)
    values = np.zeros((4, 1, 128, 22, 3), dtype=np.float32)
    values[:, 0, :16, :20, 0] = 1.0
    values[:, 0, :16, 20:, 1] = 1.0
    np.save(prepared / "skeleton_train_features.npy", values)
    np.save(prepared / "train_labels.npy", np.zeros(4, dtype=np.int64))
    result = run_safesense_odd_fold_adapter_history(
        prepared,
        tmp_path / "adapter-history-out",
        validation_subject=7,
        temporal_epoch=1,
        max_adapter_epoch=1,
        seeds=(11,),
        subject_style_training=True,
        body_canonicalization=True,
        config=SafeSenseTrainingConfig(epochs=1, patience=1, batch_size=4),
        device="cpu",
    )
    assert result["official_even_subjects_loaded"] is False
    assert result["temporal_epoch"] == 1
    assert result["max_adapter_epoch"] == 1
    assert result["shared_selection"] is None
    assert len(result["seed_results"][0]["history"]) == 1
    assert all(name.startswith("robust_") for name in result["seed_results"][0]["trainable_adapter_parameter_names"])
    assert not (prepared / "skeleton_val_features.npy").exists()
