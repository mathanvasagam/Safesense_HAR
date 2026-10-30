import csv
import json

from safesense.logging import append_csv, write_json


def test_write_json_creates_parent(tmp_path) -> None:
    path = tmp_path / "run" / "config.json"
    write_json(path, {"seed": 11, "model": "baseline"})
    assert json.loads(path.read_text()) == {"model": "baseline", "seed": 11}


def test_append_csv_writes_one_header(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    columns = ("seed", "accuracy")
    append_csv(path, {"seed": 11, "accuracy": 0.5}, columns)
    append_csv(path, {"seed": 22, "accuracy": 0.6}, columns)
    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"seed": "11", "accuracy": "0.5"},
            {"seed": "22", "accuracy": "0.6"},
        ]
