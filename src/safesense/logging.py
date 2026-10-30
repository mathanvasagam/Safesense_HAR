"""Tiny JSON and CSV writers for reproducible experiment records."""

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_json(path: str | Path, value: Any) -> None:
    """Write one indented JSON document, creating its parent directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_csv(path: str | Path, row: Mapping[str, Any], fieldnames: Iterable[str]) -> None:
    """Append a row to a CSV file and write its header when the file is new."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames)
    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})
