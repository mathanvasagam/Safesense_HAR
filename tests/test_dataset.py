import json
import csv
import zipfile
from pathlib import Path

import pytest

from safesense.dataset import DatasetError, prepare_dataset


def _make_archive(path: Path, modality: str, names: list[str]) -> None:
    with zipfile.ZipFile(path / f"{modality.title()}.zip", "w") as archive:
        archive.writestr(f"{modality.title()}/", "")
        archive.writestr("__MACOSX/._ignored", "metadata")
        for name in names:
            archive.writestr(f"{modality.title()}/{name}_{modality}.mat", f"{modality}:{name}")


def test_prepare_extracts_sorted_complete_triplet(tmp_path: Path) -> None:
    names = ["a2_s1_t1", "a1_s1_t1"]
    for modality in ("depth", "skeleton", "inertial"):
        _make_archive(tmp_path, modality, names)
    index_path = prepare_dataset(tmp_path, tmp_path / "cache", expected_count=2, expected_absent_ids=())
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 2
    assert [sample["sample_id"] for sample in payload["samples"]] == names[::-1]
    assert (tmp_path / "cache/extracted/depth/a1_s1_t1_depth.mat").read_text() == "depth:a1_s1_t1"
    assert "ignored" not in index_path.read_text(encoding="utf-8")
    with (tmp_path / "cache/manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    assert [row["sample_id"] for row in manifest] == names[::-1]
    assert manifest[0]["official_split"] == "train"
    assert manifest[0]["development_split"] == "train"
    prepare_dataset(tmp_path, tmp_path / "cache", expected_count=2, expected_absent_ids=())
    assert len(list((tmp_path / "cache/extracted/depth").glob("*.mat"))) == 2


def test_real_protocol_manifest_counts_and_subject_boundaries(tmp_path: Path) -> None:
    names = [
        f"a{action}_s{subject}_t{trial}"
        for action in range(1, 28)
        for subject in range(1, 9)
        for trial in range(1, 5)
        if f"a{action}_s{subject}_t{trial}" not in {"a8_s1_t4", "a23_s6_t4", "a27_s8_t4"}
    ]
    for modality in ("depth", "skeleton", "inertial"):
        _make_archive(tmp_path, modality, names)

    prepare_dataset(tmp_path, tmp_path / "cache")
    with (tmp_path / "cache/manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 861
    assert {row["official_split"] for row in rows} == {"train", "test"}
    assert {row["development_split"] for row in rows} == {"train", "validation", "test"}
    assert sum(row["official_split"] == "train" for row in rows) == 431
    assert sum(row["official_split"] == "test" for row in rows) == 430
    assert sum(row["development_split"] == "train" for row in rows) == 323
    assert sum(row["development_split"] == "validation" for row in rows) == 108
    assert sum(row["development_split"] == "test" for row in rows) == 430
    for row in rows:
        assert not Path(row["depth"]).is_absolute()
        assert not Path(row["skeleton"]).is_absolute()
        assert not Path(row["inertial"]).is_absolute()
        assert int(row["subject"]) in range(1, 9)
        assert row["official_split"] == ("train" if int(row["subject"]) in {1, 3, 5, 7} else "test")
        assert row["development_split"] == (
            "train" if int(row["subject"]) in {1, 3, 5}
            else "validation" if int(row["subject"]) == 7
            else "test"
        )
        for modality in ("depth", "skeleton", "inertial"):
            assert (tmp_path / "cache" / row[modality]).is_file()


def test_prepare_rejects_incomplete_triplets(tmp_path: Path) -> None:
    _make_archive(tmp_path, "depth", ["a1_s1_t1"])
    _make_archive(tmp_path, "skeleton", ["a1_s1_t1"])
    _make_archive(tmp_path, "inertial", [])
    with pytest.raises(DatasetError, match="same sample IDs"):
        prepare_dataset(tmp_path, tmp_path / "cache", expected_count=1, expected_absent_ids=())


def test_prepare_rejects_duplicate_member(tmp_path: Path) -> None:
    for modality in ("depth", "skeleton", "inertial"):
        with zipfile.ZipFile(tmp_path / f"{modality.title()}.zip", "w") as archive:
            archive.writestr(f"{modality.title()}/a1_s1_t1_{modality}.mat", "one")
            archive.writestr(f"other/a1_s1_t1_{modality}.mat", "two")
    with pytest.raises(DatasetError, match="duplicate depth sample"):
        prepare_dataset(tmp_path, tmp_path / "cache", expected_count=1, expected_absent_ids=())


def test_failed_rerun_invalidates_previous_index(tmp_path: Path) -> None:
    for modality in ("depth", "skeleton", "inertial"):
        _make_archive(tmp_path, modality, ["a1_s1_t1"])
    cache_dir = tmp_path / "cache"
    prepare_dataset(tmp_path, cache_dir, expected_count=1, expected_absent_ids=())
    assert (cache_dir / "index.json").exists()
    for name in ("features.npz", "features.npz.tmp", "preprocessing.json", "preprocessing.json.tmp"):
        (cache_dir / name).write_text("stale")

    _make_archive(tmp_path, "inertial", [])
    with pytest.raises(DatasetError):
        prepare_dataset(tmp_path, cache_dir, expected_count=1, expected_absent_ids=())
    assert not (cache_dir / "index.json").exists()
    assert not (cache_dir / "manifest.csv").exists()
    assert not (cache_dir / "features.npz").exists()
    assert not (cache_dir / "features.npz.tmp").exists()
    assert not (cache_dir / "preprocessing.json").exists()
    assert not (cache_dir / "preprocessing.json.tmp").exists()
