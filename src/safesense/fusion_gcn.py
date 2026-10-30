"""Compatibility adapter for the public Fusion-GCN runtime.

The upstream repository is intentionally not copied into this project. This
module prepares UTD-MHAD skeleton/IMU inputs and builds commands for a checkout
provided at run time.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import date
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation


FUSION_GCN_REPOSITORY = "https://github.com/mduhme/fusion-gcn"
FUSION_GCN_DATASET = "UTD-MHAD"
FUSION_GCN_CONFIG = {
    "model": "mmargcn",
    "mode": "skeleton_imu_spatial_fusion",
    "num_imu_joints": 2,
    "imu_enhanced_mode": "append_center",
    "center_joint": 1,
    "normalization_origin_joint": 2,
    "optimizer": "ADAM",
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "scheduler": "cawr",
    "scheduler_T_0": 20,
    "batch_size": 8,
    "epochs": 60,
    "seed": 1,
    "evaluation": "final_epoch_weights",
}


def fusion_gcn_config(seed: int = 1, *, smoke: bool = False) -> dict[str, object]:
    """Return the official config with only the run seed/epoch count overridden."""
    if not isinstance(seed, int):
        raise FusionGCNError("Fusion-GCN seed must be an integer")
    config = dict(FUSION_GCN_CONFIG)
    config["seed"] = seed
    if smoke:
        config["epochs"] = 1
    return config

_MAT_RE = re.compile(
    r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_(?P<modality>skeleton|inertial)\.mat$",
    re.IGNORECASE,
)
_MAX_SKELETON_FRAMES = 128
_COMPATIBILITY_PATCHES = (
    "lazy_tuning_import_without_ray",
    "exclude_tuning_session_package_import",
    "numpy_removed_aliases_across_runtime_sources",
    "disable_tensorboard_server_launch",
    "optional_torchvision_for_rgb_unused_path",
)


class FusionGCNError(ValueError):
    """Raised when Fusion-GCN inputs or a runtime checkout are invalid."""


@dataclass(frozen=True)
class FusionSample:
    sample_id: str
    action: int
    subject: int
    trial: int
    skeleton: str
    inertial: str

    @property
    def key(self) -> tuple[int, int, int]:
        return self.action, self.subject, self.trial

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _sample_id(action: int, subject: int, trial: int) -> str:
    return f"a{action}_s{subject}_t{trial}"


def _modality_roots(data_dir: Path, modality: str) -> tuple[Path, ...]:
    names = (modality, modality.title(), modality.upper())
    roots = tuple(data_dir / name for name in names if (data_dir / name).is_dir())
    return roots or (data_dir,)


def _find_members(data_dir: Path, modality: str) -> dict[str, tuple[Path, int, int, int]]:
    found: dict[str, tuple[Path, int, int, int]] = {}
    for root in _modality_roots(data_dir, modality):
        for path in sorted(root.rglob("*.mat")):
            match = _MAT_RE.fullmatch(path.name)
            if match is None or match.group("modality").lower() != modality:
                continue
            action = int(match.group("action"))
            subject = int(match.group("subject"))
            trial = int(match.group("trial"))
            if not 1 <= action <= 27 or not 1 <= subject <= 8 or not 1 <= trial <= 4:
                raise FusionGCNError(f"out-of-range {modality} sample: {path.name}")
            sample_id = _sample_id(action, subject, trial)
            if sample_id in found:
                raise FusionGCNError(f"duplicate {modality} sample: {sample_id}")
            found[sample_id] = path, action, subject, trial
    if not found:
        raise FusionGCNError(f"no {modality} MAT files found under {data_dir}")
    return found


def pair_fusion_samples(data_dir: str | Path) -> list[FusionSample]:
    """Pair only skeleton and inertial MAT files in numeric sample order."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FusionGCNError(f"Fusion-GCN data directory does not exist: {root}")
    skeleton = _find_members(root, "skeleton")
    inertial = _find_members(root, "inertial")
    if set(skeleton) != set(inertial):
        missing_skeleton = sorted(set(inertial) - set(skeleton))
        missing_inertial = sorted(set(skeleton) - set(inertial))
        raise FusionGCNError(
            f"skeleton/IMU sample IDs differ (missing_skeleton={missing_skeleton}, "
            f"missing_inertial={missing_inertial})"
        )
    samples: list[FusionSample] = []
    for sample_id in sorted(skeleton, key=lambda value: tuple(int(part[1:]) for part in value.split("_"))):
        _, action, subject, trial = skeleton[sample_id]
        imu_path, imu_action, imu_subject, imu_trial = inertial[sample_id]
        if (action, subject, trial) != (imu_action, imu_subject, imu_trial):
            raise FusionGCNError(f"metadata mismatch for {sample_id}")
        samples.append(
            FusionSample(
                sample_id=sample_id,
                action=action,
                subject=subject,
                trial=trial,
                skeleton=str(skeleton[sample_id][0].resolve()),
                inertial=str(imu_path.resolve()),
            )
        )
    return samples


def _nearest_indices(source_length: int, target_length: int) -> np.ndarray:
    if source_length < 1 or target_length < 1:
        raise FusionGCNError("sequence lengths must be positive")
    if source_length == 1 or target_length == 1:
        return np.zeros(target_length, dtype=np.int64)
    return np.rint(np.arange(target_length) * (source_length - 1) / (target_length - 1)).astype(np.int64)


def _rotate_vector_pair(sequence: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match upstream ``scipy.spatial.transform.Rotation`` ordering/casting."""
    source_norm = float(np.linalg.norm(source))
    if source_norm <= 1e-6:
        return sequence
    rotation_axis = np.cross(source, target)
    rotation_angle = float(np.arccos(np.dot(source / source_norm, target)))
    if np.abs(rotation_axis).sum() < 1e-6 or np.abs(rotation_angle) < 1e-6:
        return sequence
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation = Rotation.from_rotvec(rotation_axis * rotation_angle)
    result = sequence.copy()
    for frame in result:
        if np.sum(frame) != 0:
            frame[:] = rotation.apply(frame)
    return result


def _normalize_skeleton(sequence: np.ndarray) -> np.ndarray:
    """Match upstream UTD-MHAD skeleton normalization without importing it."""
    if sequence.shape != (_MAX_SKELETON_FRAMES, 20, 3):
        raise FusionGCNError(f"expected padded skeleton [128,20,3], got {sequence.shape}")
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    valid = np.any(sequence != 0, axis=(1, 2))
    if not np.any(valid):
        return sequence
    first = int(np.flatnonzero(valid)[0])
    last = int(np.flatnonzero(valid)[-1])
    sequence[:first] = sequence[first]
    sequence[last + 1 :] = sequence[last]
    joint_mask = np.sum(sequence, axis=-1, keepdims=True) != 0
    sequence = (sequence - sequence[:, 2:3]) * joint_mask
    sequence = _rotate_vector_pair(sequence, sequence[0, 2] - sequence[0, 3], (0, 0, 1))
    sequence = _rotate_vector_pair(sequence, sequence[0, 8] - sequence[0, 4], (1, 0, 0))
    return sequence


def _official_pad_skeleton(raw: np.ndarray) -> np.ndarray:
    """Replicate the upstream loader/``pad_null_frames`` behavior."""
    if raw.shape[0] > 128:
        raise FusionGCNError(f"skeleton exceeds 128 frames: {raw.shape[0]}")
    sequence = np.zeros((128, 20, 3), dtype=np.float32)
    sequence[: len(raw)] = raw
    valid = np.sum(sequence, axis=(1, 2)) != 0
    if not np.any(valid):
        return sequence
    if not valid[0]:
        non_null = sequence[valid].copy()
        sequence.fill(0)
        sequence[: len(non_null)] = non_null
        valid = np.sum(sequence, axis=(1, 2)) != 0
    last = int(np.flatnonzero(valid)[-1]) + 1
    if last < len(sequence):
        repeats = int(np.ceil((len(sequence) - last) / last))
        sequence[last:] = np.concatenate([sequence[:last]] * repeats, axis=0)[: len(sequence) - last]
    return sequence


def _load_fused_sample(sample: FusionSample) -> np.ndarray:
    try:
        skeleton = np.asarray(loadmat(sample.skeleton)["d_skel"], dtype=np.float32).transpose(2, 0, 1)
        inertial = np.asarray(loadmat(sample.inertial)["d_iner"], dtype=np.float32)
    except (KeyError, OSError, ValueError) as error:
        raise FusionGCNError(f"cannot load {sample.sample_id}: {error}") from error
    if skeleton.ndim != 3 or skeleton.shape[1:] != (20, 3):
        raise FusionGCNError(f"{sample.sample_id}: invalid skeleton shape {skeleton.shape}")
    if inertial.ndim != 2 or inertial.shape[1] != 6:
        raise FusionGCNError(f"{sample.sample_id}: invalid inertial shape {inertial.shape}")
    if not np.isfinite(skeleton).all() or not np.isfinite(inertial).all():
        raise FusionGCNError(f"{sample.sample_id}: non-finite skeleton or inertial values")
    if skeleton.shape[0] > _MAX_SKELETON_FRAMES:
        raise FusionGCNError(f"{sample.sample_id}: skeleton exceeds 128 frames")
    target = skeleton.shape[0]
    skeleton_padded = _normalize_skeleton(_official_pad_skeleton(skeleton))
    imu_padded = np.zeros((128, 2, 3), dtype=np.float32)
    imu_padded[:target] = inertial[_nearest_indices(len(inertial), target)].reshape(target, 2, 3)
    fused = np.zeros((1, 128, 22, 3), dtype=np.float32)
    fused[0, :, :20] = skeleton_padded
    fused[0, :, 20:] = imu_padded
    return fused


def _split(subject: int) -> str:
    return "train" if subject in {1, 3, 5, 7} else "test"


def _write_csv(path: Path, samples: Sequence[FusionSample]) -> None:
    fields = ("sample_id", "action", "subject", "trial", "skeleton", "inertial", "split")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = sample.to_dict()
            row["split"] = _split(sample.subject)
            writer.writerow(row)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_fusion_gcn_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int | None = 861,
) -> Path:
    """Create official-compatible arrays and the initial reproduction audit."""
    samples = pair_fusion_samples(data_dir)
    if expected_count is not None and len(samples) != expected_count:
        raise FusionGCNError(f"expected {expected_count} paired samples, found {len(samples)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = [sample for sample in samples if _split(sample.subject) == "train"]
    test = [sample for sample in samples if _split(sample.subject) == "test"]
    if not train or not test:
        raise FusionGCNError("Fusion-GCN preparation requires both train and held-out subject samples")
    # The upstream MultiModalDataset names its held-out split ``val``.
    for split_name, split_samples in (("train", train), ("val", test)):
        arrays = np.stack([_load_fused_sample(sample) for sample in split_samples])
        labels = np.asarray([sample.action - 1 for sample in split_samples], dtype=np.int64)
        np.save(output / f"skeleton_{split_name}_features.npy", arrays)
        np.save(output / f"{split_name}_labels.npy", labels)
    _write_csv(output / "manifest.csv", samples)
    audit = {
        "dataset": FUSION_GCN_DATASET,
        "status": "prepared",
        "upstream_repository": FUSION_GCN_REPOSITORY,
        "upstream_source_in_project": False,
        "paired_by": ["action", "subject", "trial"],
        "excluded_modalities": ["RGB", "depth"],
        "sample_count": len(samples),
        "train_count": len(train),
        "test_count": len(test),
        "feature_shape": [1, 128, 22, 3],
        "labels_zero_based": True,
        "configuration": FUSION_GCN_CONFIG,
        "manifest": "manifest.csv",
        "runtime_command": "python torch_src/main.py -f <generated-config.yaml>",
    }
    audit_path = output / "fusion_gcn_audit.json"
    _write_json(audit_path, audit)
    return audit_path


def validate_runtime_checkout(repo_dir: str | Path) -> Path:
    root = Path(repo_dir)
    required = (root / "torch_src" / "main.py", root / "torch_src" / "models" / "mmargcn")
    if not root.is_dir() or not all(path.exists() for path in required):
        raise FusionGCNError(
            f"runtime checkout is missing Fusion-GCN files: {root}; "
            f"clone {FUSION_GCN_REPOSITORY} outside this project"
        )
    return root


def runtime_environment() -> dict[str, str | None]:
    """Collect versions lazily so preparation tests do not import torch."""
    import importlib
    import platform

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "torch"):
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # pragma: no cover - runtime image diagnostics
            versions[name] = f"unavailable: {error}"
    versions["cuda"] = None
    versions["gpu"] = None
    try:
        torch = importlib.import_module("torch")
        cuda_version = getattr(torch.version, "cuda", None)
        versions["cuda"] = str(cuda_version) if cuda_version is not None else None
        if torch.cuda.is_available():
            versions["gpu"] = str(torch.cuda.get_device_name(0))
    except Exception as error:  # pragma: no cover - runtime image diagnostics
        versions["cuda"] = f"unavailable: {error}"
    return versions


def apply_runtime_compatibility(repo_dir: str | Path) -> list[str]:
    """Patch only unused/obsolete runtime edges in an external checkout."""
    root = validate_runtime_checkout(repo_dir)
    patches: list[str] = []

    def replace(path: Path, old: str, new: str, name: str) -> None:
        text = path.read_text(encoding="utf-8")
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            patches.append(name)
        elif new in text:
            patches.append(name)
        else:
            raise FusionGCNError(f"cannot apply compatibility patch {name}: {path}")

    replace(
        root / "torch_src" / "session_helper.py",
        "from tune_config import make_tune_config",
        "def make_tune_config(_base_config):\n    raise RuntimeError('Fusion-GCN tuning requires optional Ray dependencies')",
        "lazy_tuning_import_without_ray",
    )
    replace(
        root / "torch_src" / "session" / "__init__.py",
        "from . import session, debugging, evaluation, training, tuning, profiling",
        "from . import session, debugging, evaluation, training, profiling",
        "exclude_tuning_session_package_import",
    )
    source_roots = (root / "torch_src", root / "util", root / "datasets" / "utd_mhad")
    alias_changed = False
    for source_root in source_roots:
        if not source_root.is_dir():
            continue
        for path in source_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            updated = re.sub(r"np\.bool(?![A-Za-z0-9_])", "bool", source)
            updated = re.sub(r"np\.int(?![A-Za-z0-9_])", "int", updated)
            updated = re.sub(r"np\.float(?![A-Za-z0-9_])", "float", updated)
            if updated != source:
                path.write_text(updated, encoding="utf-8")
                alias_changed = True
    if alias_changed or all(
        not any(
            re.search(rf"np\.{name}(?![A-Za-z0-9_])", path.read_text(encoding="utf-8"))
            for path in root_dir.rglob("*.py")
        )
        for root_dir in source_roots if root_dir.is_dir()
        for name in ("bool", "int", "float")
    ):
        patches.append("numpy_removed_aliases_across_runtime_sources")
    replace(
        root / "torch_src" / "progress.py",
        "tensorboard: bool = True",
        "tensorboard: bool = False",
        "disable_tensorboard_server_launch",
    )
    replace(
        root / "torch_src" / "models" / "mmargcn" / "rgb_feature_models.py",
        "import torchvision.models as models",
        "try:\n    import torchvision.models as models\nexcept Exception:  # RGB is unused by the center-node run\n    models = None",
        "optional_torchvision_for_rgb_unused_path",
    )
    return patches


def write_runtime_config(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    smoke: bool = False,
    seed: int = 1,
) -> Path:
    prepared = Path(prepared_dir).resolve()
    output = Path(output_dir).resolve()
    required = (
        "skeleton_train_features.npy",
        "train_labels.npy",
        "skeleton_val_features.npy",
        "val_labels.npy",
    )
    missing = [name for name in required if not (prepared / name).is_file()]
    if missing:
        raise FusionGCNError(f"prepared Fusion-GCN artifact is missing {missing}: {prepared}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / (f"fusion_gcn_smoke_seed_{seed}.yaml" if smoke else f"fusion_gcn_reproduce_seed_{seed}.yaml")
    official_config = fusion_gcn_config(seed, smoke=smoke)
    config = (
        f"input_data:\n  - path: {prepared}\n    loader: NumpyDatasetLoader\n    loader_args:\n"
        f"      in_memory: True\nout_path: {output}\nmodel: mmargcn\ndataset: UTD-MHAD\n"
        f"session_type: training\nfixed_seed: {seed}\nmode: skeleton_imu_spatial_fusion\nmodel_args:\n"
        f"  num_imu_joints: 2\n  imu_enhanced_mode: append_center\nbase_lr: 0.001\noptimizer: ADAM\n"
        f"optimizer_args:\n  weight_decay: 0.01\nlr_scheduler: cawr\nlr_scheduler_args:\n  T_0: 20\n"
        f"batch_size: 8\nepochs: {official_config['epochs']}\n"
    )
    config_path.write_text(config, encoding="utf-8")
    return config_path


def fusion_gcn_command(repo_dir: str | Path, config_path: str | Path) -> list[str]:
    validate_runtime_checkout(repo_dir)
    return [sys.executable, "torch_src/main.py", "-f", str(Path(config_path).resolve())]


def _checkpoint_records(paths: Sequence[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        name = path.stem
        checkpoint_match = re.search(r"checkpoint[_-](\d+)[_-](0?\.[0-9]+|[0-9]+(?:\.[0-9]+)?)", name, flags=re.IGNORECASE)
        epoch_match = checkpoint_match or re.search(r"(?:epoch|ep)[_-]?(\d+)", name, flags=re.IGNORECASE)
        accuracy_matches = re.findall(
            r"(?:val(?:idation)?(?:[_-]?(?:accuracy|acc))?|accuracy)[=_-]?([0-9]+(?:\.[0-9]+)?)",
            name,
            flags=re.IGNORECASE,
        )
        accuracy = float(checkpoint_match.group(2)) if checkpoint_match else (float(accuracy_matches[-1]) if accuracy_matches else None)
        if accuracy is not None and accuracy > 1:
            accuracy /= 100
        records.append(
            {
                "path": path,
                "epoch": int(epoch_match.group(1)) if epoch_match else None,
                "accuracy": accuracy,
            }
        )
    return records


def _retained_checkpoint(records: Sequence[dict[str, object]]) -> dict[str, object] | None:
    scored = [record for record in records if record["accuracy"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda record: (float(record["accuracy"]), int(record["epoch"] or -1), str(record["path"])))


def run_fusion_gcn(
    repo_dir: str | Path,
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    smoke: bool = False,
    seed: int = 1,
) -> dict[str, object]:
    """Run upstream training from an external checkout and record compact output."""
    started = time.perf_counter()
    config_path = write_runtime_config(prepared_dir, output_dir, smoke=smoke, seed=seed)
    compatibility_patches = apply_runtime_compatibility(repo_dir)
    command = fusion_gcn_command(repo_dir, config_path)
    repo_root = Path(repo_dir).resolve()
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else os.pathsep.join((str(repo_root), existing_pythonpath))
    )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    output = Path(output_dir)
    stdout = completed.stdout
    accuracy_matches = re.findall(r"validation_accuracy:\s*([0-9]+(?:\.[0-9]+)?)", stdout, flags=re.IGNORECASE)
    final_accuracy = float(accuracy_matches[-1]) if accuracy_matches else None
    parameter_matches = re.findall(r"Trainable parameters:\s*([0-9,]+)", stdout, flags=re.IGNORECASE)
    parameter_count = int(parameter_matches[-1].replace(",", "")) if parameter_matches else None
    weights = sorted(output.rglob("*_weights.pt"))
    checkpoints = sorted(output.rglob("checkpoint_*.pt"))
    checkpoint_records = _checkpoint_records(checkpoints)
    final_record = max(
        checkpoint_records,
        key=lambda record: (int(record["epoch"] is not None), int(record["epoch"] or -1), str(record["path"])),
        default=None,
    )
    best_record = _retained_checkpoint(checkpoint_records)
    if final_accuracy is None and final_record is not None:
        final_accuracy = final_record["accuracy"]
    session_dir = str(Path(final_record["path"]).parent.parent) if final_record else None
    status = "completed" if completed.returncode == 0 else "failed"
    error = None
    if completed.returncode == 0 and final_accuracy is None:
        status = "failed"
        error = "training exited 0 but final validation accuracy was not found in stdout"
    elif completed.returncode == 0 and session_dir is None:
        status = "failed"
        error = "training exited 0 but final session weights were not found"
    result_path = output / f"fusion_gcn_reproduction_seed_{seed}.json"
    result: dict[str, object] = {
        "status": status,
        "return_code": completed.returncode,
        "smoke": smoke,
        "seed": seed,
        "session_name": f"fusion_gcn_seed_{seed}",
        "output_directory": str(output),
        "result_path": str(result_path),
        "command": command,
        "upstream_repository": FUSION_GCN_REPOSITORY,
        "access_date": date.today().isoformat(),
        "configuration": fusion_gcn_config(seed, smoke=smoke),
        "compatibility_patches": compatibility_patches,
        "environment": runtime_environment(),
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count,
        "final_validation_accuracy": final_accuracy,
        "best_retained_validation_accuracy": best_record["accuracy"] if best_record else None,
        "best_retained_checkpoint_epoch_index": best_record["epoch"] if best_record else None,
        "best_retained_training_epoch": int(best_record["epoch"]) + 1 if best_record and best_record["epoch"] is not None else None,
        "best_retained_checkpoint": str(best_record["path"]) if best_record else None,
        "final_checkpoint": str(weights[-1]) if weights else None,
        "session_directory": session_dir,
        "weights_path": str(weights[-1]) if weights else None,
        "error": error,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    _write_json(result_path, result)
    if error is not None:
        raise FusionGCNError(error)
    return result


def runtime_smoke(repo_dir: str | Path, prepared_dir: str | Path, output_dir: str | Path, *, seed: int = 1) -> dict[str, object]:
    return run_fusion_gcn(repo_dir, prepared_dir, output_dir, smoke=True, seed=seed)


def runtime_reproduce(repo_dir: str | Path, prepared_dir: str | Path, output_dir: str | Path, *, seed: int = 1) -> dict[str, object]:
    return run_fusion_gcn(repo_dir, prepared_dir, output_dir, smoke=False, seed=seed)
