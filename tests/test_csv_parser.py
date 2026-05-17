"""CSV parser tests — synthesizes a save-event-list-style CSV and walks it."""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import pytest

from pix_mcp import csv_parser


def _write_csv(tmp_path: Path, rows: list[list[str]]) -> Path:
    f = tmp_path / "events.csv"
    with f.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r)
    return f


def test_sniff_header_separates_counters(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        [
            ["Queue ID", "Queue", "Global ID", "Name", "GPU Duration", "Vertex Count"],
            ["0", "Direct", "1", "Present", "1.2", "0"],
        ],
    )
    header = csv_parser.sniff_header(csv_path)
    assert "Queue ID" in header.columns
    assert "Name" in header.columns
    assert "GPU Duration" in header.counter_columns
    assert "Vertex Count" in header.counter_columns
    # baselines never end up in counter list
    for hint in ("Queue", "Queue ID", "Name", "Global ID"):
        assert hint not in header.counter_columns


def test_iter_rows_normalizes_keys(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        [
            ["Queue ID", "Queue", "Global ID", "Name", "GPU Duration"],
            ["0", "Direct", "1", "DrawInstanced", "0.05"],
            ["0", "Direct", "2", "Present", ""],
        ],
    )
    rows = list(csv_parser.iter_rows(csv_path))
    assert len(rows) == 2
    assert rows[0]["queue_id"] == "0"
    assert rows[0]["queue"] == "Direct"
    assert rows[0]["global_id"] == "1"
    assert rows[0]["name"] == "DrawInstanced"
    assert rows[0]["gpu_duration"] == "0.05"
    assert rows[1]["gpu_duration"] == ""


def test_iter_rows_callback(tmp_path: Path) -> None:
    csv_path = _write_csv(
        tmp_path,
        [
            ["Queue ID", "Queue", "Global ID", "Name"],
            ["0", "Direct", "10", "BeginEvent: Shadow"],
            ["0", "Direct", "11", "DrawIndexedInstanced"],
            ["0", "Direct", "12", "EndEvent"],
        ],
    )
    collected: list[dict[str, str]] = []
    n = csv_parser.parse_rows(csv_path, callback=collected.append)
    assert n == 3
    names = [r["name"] for r in collected]
    assert names == ["BeginEvent: Shadow", "DrawIndexedInstanced", "EndEvent"]
