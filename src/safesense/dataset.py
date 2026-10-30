"""UTD-MHAD archive extraction and sample indexing."""

from __future__ import annotations

import json
import csv
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KNOWN_ABSENT_IDS = frozenset({"a8_s1_t4", "a23_s6_t4", "a27_s8_t4"})
MODALITIES = ("depth", "skeleton", "inertial")
OFFICIAL_TRAIN_SUBJECTS = frozenset({1, 3, 5, 7})
DEVELOPMENT_TRAIN_SUBJECTS = frozenset({1, 3, 5})
DEVELOPMENT_VALIDATION_SUBJECTS = frozenset({7})
ARCHIVE_NAMES = {modality: f"{modality.title()}.zip" for modality in MODALITIES}
SAMPLE_RE = re.compile(
    r"^a(?P<action>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_(?P<modality>depth|skeleton|inertial)\.mat$"
)


class DatasetError(ValueError):
    """Raised when the UTD-MHAD archives do not form a valid dataset."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    action: int
    subject: int
    trial: int
    depth: str
    skeleton: str
    inertial: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "sample_id": self.sample_id,
            "action": self.action,
            "subject": self.subject,
            "trial": self.trial,
            "depth": self.depth,
            "skeleton": self.skeleton,
            "inertial": self.inertial,
        }


def _sample_id(action: int, subject: int, trial: int) -> str:
    return f"a{action}_s{subject}_t{trial}"


def _parse_member(member: str, modality: str) -> tuple[str, int, int, int] | None:
    match = SAMPLE_RE.fullmatch(Path(member).name)
    if match is None or match.group("modality") != modality:
        return None
    action = int(match.group("action"))
    subject = int(match.group("subject"))
    trial = int(match.group("trial"))
    if not 1 <= action <= 27 or not 1 <= subject <= 8 or not 1 <= trial <= 4:
        raise DatasetError(f"out-of-range {modality} member: {member}")
    return _sample_id(action, subject, trial), action, subject, trial


def _copy_archive_members(raw_dir: Path, extracted_dir: Path, modality: str) -> dict[str, tuple[int, int, int]]:
    """Extract canonical members, flattening archive folders by modality."""
    archive = raw_dir / ARCHIVE_NAMES[modality]
    if not archive.is_file():
        raise DatasetError(f"missing archive: {archive}")
    destination = extracted_dir / modality
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen: dict[str, tuple[int, int, int]] = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parsed = _parse_member(info.filename, modality)
            if parsed is None:
                continue
            sample_id, action, subject, trial = parsed
            if sample_id in seen:
                raise DatasetError(f"duplicate {modality} sample: {sample_id}")
            output = destination / Path(info.filename).name
            temporary = output.with_suffix(output.suffix + ".tmp")
            with zf.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(output)
            seen[sample_id] = (action, subject, trial)
    return seen


def _expected_ids() -> set[str]:
    return {
        _sample_id(action, subject, trial)
        for action in range(1, 28)
        for subject in range(1, 9)
        for trial in range(1, 5)
    } - set(KNOWN_ABSENT_IDS)


def _validate_members(
    members: dict[str, dict[str, tuple[int, int, int]]],
    *,
    expected_count: int | None,
    expected_absent_ids: Iterable[str] | None,
) -> list[Sample]:
    if set(members) != set(MODALITIES):
        raise DatasetError("all three modalities are required")
    sample_sets = {modality: set(members[modality]) for modality in MODALITIES}
    if sample_sets["depth"] != sample_sets["skeleton"] or sample_sets["depth"] != sample_sets["inertial"]:
        raise DatasetError("modalities do not contain the same sample IDs")
    sample_ids = sample_sets["depth"]
    if expected_count is not None and len(sample_ids) != expected_count:
        raise DatasetError(f"expected {expected_count} complete triplets, found {len(sample_ids)}")
    absent = set(expected_absent_ids or ())
    if expected_count == 861 and absent == set(KNOWN_ABSENT_IDS) and sample_ids != _expected_ids():
        missing = sorted(_expected_ids() - sample_ids)
        extras = sorted(sample_ids - _expected_ids())
        raise DatasetError(f"unexpected UTD-MHAD sample IDs (missing={missing}, extras={extras})")
    samples: list[Sample] = []
    for sample_id in sorted(sample_ids, key=lambda value: tuple(int(part[1:]) for part in value.split("_"))):
        action, subject, trial = members["depth"][sample_id]
        for modality in MODALITIES:
            if members[modality][sample_id] != (action, subject, trial):
                raise DatasetError(f"metadata mismatch for {sample_id}")
        samples.append(
            Sample(
                sample_id=sample_id,
                action=action,
                subject=subject,
                trial=trial,
                depth=f"extracted/depth/{sample_id}_depth.mat",
                skeleton=f"extracted/skeleton/{sample_id}_skeleton.mat",
                inertial=f"extracted/inertial/{sample_id}_inertial.mat",
            )
        )
    return samples


def _split_labels(subject: int) -> tuple[str, str]:
    """Return the official and development labels for a subject."""
    official = "train" if subject in OFFICIAL_TRAIN_SUBJECTS else "test"
    if subject in DEVELOPMENT_TRAIN_SUBJECTS:
        development = "train"
    elif subject in DEVELOPMENT_VALIDATION_SUBJECTS:
        development = "validation"
    else:
        development = "test"
    return official, development


MANIFEST_FIELDS = (
    "sample_id",
    "action",
    "subject",
    "trial",
    "depth",
    "skeleton",
    "inertial",
    "official_split",
    "development_split",
)


def write_manifest(samples: Iterable[Sample], cache_dir: str | Path) -> Path:
    """Write the deterministic, relative-path UTD-MHAD split manifest."""
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.csv"
    rows: list[dict[str, int | str]] = []
    for sample in samples:
        for modality in MODALITIES:
            relative_path = Path(getattr(sample, modality))
            if relative_path.is_absolute():
                raise DatasetError(f"manifest path must be relative: {relative_path}")
            if not (cache_path / relative_path).is_file():
                raise DatasetError(f"manifest path does not exist: {relative_path}")
        official_split, development_split = _split_labels(sample.subject)
        row = sample.to_dict()
        row["official_split"] = official_split
        row["development_split"] = development_split
        rows.append(row)

    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(manifest_path)
    return manifest_path


def prepare_dataset(
    raw_dir: str | Path,
    cache_dir: str | Path,
    *,
    expected_count: int | None = 861,
    expected_absent_ids: Iterable[str] | None = KNOWN_ABSENT_IDS,
) -> Path:
    """Extract the three raw archives and write ``cache_dir/index.json``."""
    raw_path = Path(raw_dir)
    cache_path = Path(cache_dir)
    index_path = cache_path / "index.json"
    manifest_path = cache_path / "manifest.csv"
    for stale_path in (
        index_path,
        manifest_path,
        cache_path / "features.npz",
        cache_path / "features.npz.tmp",
        cache_path / "preprocessing.json",
        cache_path / "preprocessing.json.tmp",
    ):
        if stale_path.exists():
            stale_path.unlink()
    extracted_path = cache_path / "extracted"
    extracted_path.mkdir(parents=True, exist_ok=True)
    extracted_members = {
        modality: _copy_archive_members(raw_path, extracted_path, modality)
        for modality in MODALITIES
    }
    samples = _validate_members(
        extracted_members,
        expected_count=expected_count,
        expected_absent_ids=expected_absent_ids,
    )
    payload = {
        "dataset": "UTD-MHAD",
        "sample_count": len(samples),
        "samples": [sample.to_dict() for sample in samples],
    }
    temporary_index = index_path.with_suffix(".json.tmp")
    temporary_index.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_index.replace(index_path)
    write_manifest(samples, cache_path)
    return index_path
