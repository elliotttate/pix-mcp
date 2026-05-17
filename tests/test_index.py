"""SQLite indexer tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pix_mcp import index


def _build(tmp_path: Path) -> tuple[Path, Path]:
    wpix = tmp_path / "scene.wpix"
    wpix.write_bytes(b"dummy")
    csv_path = tmp_path / "scene.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Queue ID", "Queue", "Global ID", "Name", "GPU Duration"])
        # Frame 1
        w.writerow(["0", "Direct", "1", "ExecuteCommandLists", ""])
        w.writerow(["0", "Direct", "2", "BeginEvent: ShadowPass", ""])
        w.writerow(["0", "Direct", "3", "BeginEvent: Cascade0", ""])
        w.writerow(["0", "Direct", "4", "DrawIndexedInstanced", "0.10"])
        w.writerow(["0", "Direct", "5", "DrawIndexedInstanced", "0.30"])
        w.writerow(["0", "Direct", "6", "EndEvent", ""])
        w.writerow(["0", "Direct", "7", "EndEvent", ""])
        w.writerow(["0", "Direct", "8", "BeginEvent: GBuffer", ""])
        w.writerow(["0", "Direct", "9", "ResourceBarrier", ""])
        w.writerow(["0", "Direct", "10", "DrawInstanced", "1.20"])
        w.writerow(["0", "Direct", "11", "DispatchMesh", "2.50"])
        w.writerow(["0", "Direct", "12", "CopyResource", ""])
        w.writerow(["0", "Direct", "13", "EndEvent", ""])
        w.writerow(["0", "Direct", "14", "Present", ""])
        # Compute queue
        w.writerow(["1", "Compute", "15", "Dispatch", "0.05"])
        w.writerow(["1", "Compute", "16", "Dispatch", "0.07"])
    return wpix, csv_path


def test_build_index_counts(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    stats = index.build_index(wpix, csv_path)
    assert stats.event_count == 16
    # 3 markers: ShadowPass, Cascade0, GBuffer
    assert stats.marker_count == 3
    assert stats.queue_count == 2
    assert "gpu_duration" in stats.counter_columns


def test_event_type_classification(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    index.build_index(wpix, csv_path)
    counts = index.event_type_counts(wpix)
    assert counts.get("draw", 0) == 3  # 2x DrawIndexed + 1x DrawInstanced
    assert counts.get("dispatch", 0) == 3  # DispatchMesh + 2x compute Dispatch
    assert counts.get("begin_marker", 0) == 3
    assert counts.get("end_marker", 0) == 3
    assert counts.get("barrier", 0) == 1
    assert counts.get("copy", 0) == 1
    assert counts.get("present", 0) == 1
    assert counts.get("execute_command_lists", 0) == 1


def test_markers_and_inside_marker_query(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    index.build_index(wpix, csv_path)

    markers = index.list_markers(wpix)
    names = {m["name"]: m for m in markers}
    assert "ShadowPass" in names
    assert names["ShadowPass"]["start_global_id"] == 2
    assert names["ShadowPass"]["end_global_id"] == 7
    assert names["Cascade0"]["start_global_id"] == 3
    assert names["Cascade0"]["end_global_id"] == 6
    assert names["GBuffer"]["start_global_id"] == 8
    assert names["GBuffer"]["end_global_id"] == 13

    # Draws inside ShadowPass should pick up both cascade draws.
    rows = index.find_events(wpix, marker_name="ShadowPass", event_type="draw")
    gids = [r["global_id"] for r in rows]
    assert gids == [4, 5]


def test_get_event_and_counters(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    index.build_index(wpix, csv_path)
    ev = index.get_event(wpix, 11, include_raw=True)
    assert ev is not None
    assert ev["name"] == "DispatchMesh"
    assert ev["event_type"] == "dispatch"
    assert ev["counters"]["gpu_duration"] == "2.50"
    assert ev["raw"]["queue"] == "Direct"


def test_top_by_counter_sorts_descending(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    index.build_index(wpix, csv_path)
    rows = index.top_by_counter(wpix, "gpu_duration", n=5)
    durations = [r["counter_value"] for r in rows]
    assert durations == sorted(durations, reverse=True)
    # The most expensive event in our fixture is the DispatchMesh at gid 11.
    assert rows[0]["global_id"] == 11


def test_raw_sql_refuses_writes(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    index.build_index(wpix, csv_path)
    with pytest.raises(ValueError):
        index.raw_sql(wpix, "DELETE FROM events")
    with pytest.raises(ValueError):
        index.raw_sql(wpix, "UPDATE events SET name='x'")
    with pytest.raises(ValueError):
        index.raw_sql(wpix, "DROP TABLE events")


def test_raw_sql_allows_selects(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    index.build_index(wpix, csv_path)
    rows = index.raw_sql(
        wpix,
        "SELECT name FROM events WHERE event_type = ? ORDER BY global_id",
        params=["draw"],
    )
    assert [r["name"] for r in rows] == [
        "DrawIndexedInstanced",
        "DrawIndexedInstanced",
        "DrawInstanced",
    ]


def test_index_idempotent_via_status(tmp_path: Path) -> None:
    wpix, csv_path = _build(tmp_path)
    index.build_index(wpix, csv_path)
    status = index.index_status(wpix, csv_path)
    assert status["exists"]
    assert status["up_to_date"]
    assert status["event_count"] == 16
