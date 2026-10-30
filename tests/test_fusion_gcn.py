import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.spatial.transform import Rotation

from safesense.fusion_gcn import (
    FusionGCNError,
    _checkpoint_records,
    _retained_checkpoint,
    _normalize_skeleton,
    _official_pad_skeleton,
    pair_fusion_samples,
    prepare_fusion_gcn_dataset,
    run_fusion_gcn,
    write_runtime_config,
)


def _reference_pad(raw: np.ndarray) -> np.ndarray:
    padded = np.zeros((128, 20, 3), dtype=np.float32)
    padded[: len(raw)] = raw
    valid = np.sum(padded, axis=(1, 2)) != 0
    if not np.any(valid):
        return padded
    if not valid[0]:
        non_null = padded[valid].copy()
        padded.fill(0)
        padded[: len(non_null)] = non_null
        valid = np.sum(padded, axis=(1, 2)) != 0
    last = int(np.flatnonzero(valid)[-1]) + 1
    if last < len(padded):
        repeats = int(np.ceil((len(padded) - last) / last))
        padded[last:] = np.concatenate([padded[:last]] * repeats, axis=0)[: len(padded) - last]
    return padded


def _reference_normalize(raw: np.ndarray) -> np.ndarray:
    sequence = _reference_pad(raw)
    joint_mask = np.sum(sequence, axis=-1, keepdims=True) != 0
    sequence = (sequence - sequence[:, 2:3]) * joint_mask
    for source, target in (
        (sequence[0, 2] - sequence[0, 3], (0, 0, 1)),
        (sequence[0, 8] - sequence[0, 4], (1, 0, 0)),
    ):
        source_norm = np.linalg.norm(source)
        cross = np.cross(source, target)
        sine = np.linalg.norm(cross)
        cosine = np.clip(np.dot(source / source_norm, target), -1.0, 1.0)
        if source_norm > 1e-6 and sine > 1e-6:
            rotation = Rotation.from_rotvec(cross / sine * np.arccos(cosine))
            for frame_data in sequence:
                if np.sum(frame_data) != 0:
                    frame_data[:] = rotation.apply(frame_data)
    return sequence.astype(np.float32)


def _write_sample(root: Path, sample_id: str, *, include_depth: bool = False) -> None:
    skeleton_dir = root / "Skeleton"
    inertial_dir = root / "Inertial"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    inertial_dir.mkdir(parents=True, exist_ok=True)
    skeleton = np.zeros((20, 3, 3), dtype=np.float32)
    skeleton[1] = np.array([[1.0], [2.0], [3.0]])
    skeleton[2] = np.array([[1.0], [2.0], [4.0]])
    skeleton[3] = np.array([[1.0], [2.0], [2.0]])
    inertial = np.arange(24, dtype=np.float32).reshape(4, 6)
    savemat(skeleton_dir / f"{sample_id}_skeleton.mat", {"d_skel": skeleton})
    savemat(inertial_dir / f"{sample_id}_inertial.mat", {"d_iner": inertial})
    if include_depth:
        depth_dir = root / "Depth"
        depth_dir.mkdir(exist_ok=True)
        savemat(depth_dir / f"{sample_id}_depth.mat", {"d_depth": np.zeros((1, 1, 1))})


def test_pairing_is_numeric_and_ignores_depth(tmp_path: Path) -> None:
    _write_sample(tmp_path, "a2_s2_t1", include_depth=True)
    _write_sample(tmp_path, "a1_s1_t1")

    samples = pair_fusion_samples(tmp_path)

    assert [sample.sample_id for sample in samples] == ["a1_s1_t1", "a2_s2_t1"]
    assert samples[0].key == (1, 1, 1)
    assert all("depth" not in sample.to_dict() for sample in samples)


def test_pairing_rejects_unmatched_modalities(tmp_path: Path) -> None:
    _write_sample(tmp_path, "a1_s1_t1")
    (tmp_path / "Inertial" / "a1_s1_t1_inertial.mat").unlink()

    with pytest.raises(FusionGCNError, match="no inertial MAT files"):
        pair_fusion_samples(tmp_path)


def test_prepare_writes_official_center_node_arrays_and_audit(tmp_path: Path) -> None:
    _write_sample(tmp_path / "input", "a2_s2_t1")
    _write_sample(tmp_path / "input", "a1_s1_t1")
    output = tmp_path / "prepared"

    audit_path = prepare_fusion_gcn_dataset(tmp_path / "input", output, expected_count=2)

    train = np.load(output / "skeleton_train_features.npy")
    assert train.shape == (1, 1, 128, 22, 3)
    assert np.isfinite(train).all()
    assert np.load(output / "train_labels.npy").tolist() == [0]
    assert np.load(output / "val_labels.npy").tolist() == [1]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["upstream_source_in_project"] is False
    assert audit["configuration"]["imu_enhanced_mode"] == "append_center"
    assert audit["excluded_modalities"] == ["RGB", "depth"]


def test_runtime_config_preserves_official_epochs_and_smoke_override(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    np.save(prepared / "skeleton_train_features.npy", np.zeros((1, 1, 128, 22, 3), dtype=np.float32))
    np.save(prepared / "train_labels.npy", np.zeros(1, dtype=np.int64))
    np.save(prepared / "skeleton_val_features.npy", np.zeros((1, 1, 128, 22, 3), dtype=np.float32))
    np.save(prepared / "val_labels.npy", np.zeros(1, dtype=np.int64))

    reproduce = write_runtime_config(prepared, tmp_path / "reproduce")
    smoke = write_runtime_config(prepared, tmp_path / "smoke", smoke=True)
    seed_11 = write_runtime_config(prepared, tmp_path / "seed-11", seed=11)
    seed_22 = write_runtime_config(prepared, tmp_path / "seed-22", seed=22)

    assert "epochs: 60" in reproduce.read_text(encoding="utf-8")
    assert "epochs: 1" in smoke.read_text(encoding="utf-8")
    assert "append_center" in reproduce.read_text(encoding="utf-8")
    assert seed_11 != seed_22
    assert "fixed_seed: 11" in seed_11.read_text(encoding="utf-8")
    assert "fixed_seed: 22" in seed_22.read_text(encoding="utf-8")


def test_retained_checkpoint_parser_selects_best_filename() -> None:
    records = _checkpoint_records(
        [
            Path("training_seed_11/checkpoints/checkpoint_60_0.9023.pt"),
            Path("training_seed_11/checkpoints/checkpoint_42_0.9233.pt"),
        ]
    )
    best = _retained_checkpoint(records)
    assert best is not None
    assert best["epoch"] == 42
    assert best["accuracy"] == 0.9233
    assert str(best["path"]).endswith("checkpoint_42_0.9233.pt")


def test_normalization_matches_independent_upstream_reference_and_axes() -> None:
    raw = np.zeros((4, 20, 3), dtype=np.float32)
    for frame in range(4):
        center = np.array([0.5 * frame, 0.1, 0.2], dtype=np.float32)
        raw[frame, 2] = center
        raw[frame, 3] = center + [0.0, 0.0, -1.0]
        raw[frame, 4] = center + [-1.0, 0.0, 0.0]
        raw[frame, 8] = center + [1.0, 0.0, 0.0]
    raw[0, 2] = [0.0, 0.1, 0.2]

    actual = _normalize_skeleton(_official_pad_skeleton(raw))
    expected = _reference_normalize(raw)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual[:4, 2], 0.0, atol=1e-6)
    assert actual[1, 8, 0] - actual[1, 4, 0] > 0


def test_runtime_result_extracts_final_accuracy_session_and_compatibility_patches(tmp_path: Path) -> None:
    repo = tmp_path / "runtime"
    for relative in (
        "torch_src/session",
        "torch_src/models/mmargcn",
        "util",
        "datasets/utd_mhad",
    ):
        (repo / relative).mkdir(parents=True)
    (repo / "torch_src/session_helper.py").write_text("from tune_config import make_tune_config\n")
    (repo / "torch_src/session/__init__.py").write_text(
        "from . import session, debugging, evaluation, training, tuning, profiling\n"
    )
    (repo / "torch_src/progress.py").write_text("tensorboard: bool = True\n")
    (repo / "torch_src/models/mmargcn/rgb_feature_models.py").write_text("import torchvision.models as models\n")
    (repo / "util/graph.py").write_text("np.bool np.int np.float np.integer np.floating np.bool_\n")
    (repo / "datasets/utd_mhad/constants.py").write_text("value = np.int(1)\n")
    (repo / "root_runtime_module.py").write_text("MARKER = 'repo-root-imported'\n")
    (repo / "torch_src/main.py").write_text(
        "import pathlib, re, sys\n"
        "import root_runtime_module\n"
        "assert root_runtime_module.MARKER == 'repo-root-imported'\n"
        "config = pathlib.Path(sys.argv[2]).read_text()\n"
        "out = pathlib.Path(re.search(r'out_path: (.*)', config).group(1).strip())\n"
        "weights = out / 'training_fake' / 'checkpoints'\n"
        "weights.mkdir(parents=True)\n"
        "(weights / 'checkpoint_0_0.5000.pt').write_bytes(b'x')\n"
        "(weights / 'checkpoint_1_0.6000.pt').write_bytes(b'x')\n"
        "(weights / 'training_fake_weights.pt').write_bytes(b'x')\n"
        "print('Model - Trainable parameters: 123')\n"
        "print('validation_accuracy: 0.7500')\n"
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    np.save(prepared / "skeleton_train_features.npy", np.zeros((1, 1, 128, 22, 3), dtype=np.float32))
    np.save(prepared / "train_labels.npy", np.zeros(1, dtype=np.int64))
    np.save(prepared / "skeleton_val_features.npy", np.zeros((1, 1, 128, 22, 3), dtype=np.float32))
    np.save(prepared / "val_labels.npy", np.zeros(1, dtype=np.int64))
    output = tmp_path / "result"

    result = run_fusion_gcn(repo, prepared, output, smoke=True)

    assert result["status"] == "completed"
    assert result["final_validation_accuracy"] == 0.75
    assert result["parameter_count"] == 123
    assert result["session_directory"].endswith("training_fake")
    assert result["weights_path"].endswith("training_fake_weights.pt")
    assert result["best_retained_validation_accuracy"] == 0.6
    assert result["best_retained_checkpoint_epoch_index"] == 1
    assert result["best_retained_training_epoch"] == 2
    assert result["best_retained_checkpoint"].endswith("checkpoint_1_0.6000.pt")
    assert (output / "fusion_gcn_reproduction_seed_1.json").is_file()
    assert result["session_name"] == "fusion_gcn_seed_1"
    assert result["result_path"].endswith("fusion_gcn_reproduction_seed_1.json")
    assert "numpy_removed_aliases_across_runtime_sources" in result["compatibility_patches"]
    graph_source = (repo / "util/graph.py").read_text()
    assert "np.int " not in graph_source
    assert "np.float " not in graph_source
    assert "np.bool " not in graph_source
    assert "np.integer" in graph_source
    assert "np.floating" in graph_source
    assert "np.bool_" in graph_source
    assert "np.int" not in (repo / "datasets/utd_mhad/constants.py").read_text()
    assert {"python", "numpy", "scipy", "torch", "cuda", "gpu"} <= result["environment"].keys()
