"""Parser for the CSV produced by ``pixtool save-event-list``.

The CSV's columns are not fixed: queue id, name and global id are always
present, but any counter selected via ``--counters`` / ``--counter-groups`` is
appended as an extra column. We parse it incrementally so very large captures
(millions of rows) don't blow up memory — emit row dicts to a callback.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


# Columns that PIX always writes — the indexer relies on these.
ALWAYS_PRESENT_HINTS = ("Global ID", "Name", "Queue", "Queue ID")


@dataclass
class EventListHeader:
    columns: list[str]
    counter_columns: list[str]  # everything after the "always present" baseline


def _normalize_column(name: str) -> str:
    """Turn 'Global ID' into 'global_id', counters into snake_case-ish keys."""
    n = name.strip()
    n = re.sub(r"[^A-Za-z0-9]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_").lower()
    return n or "col"


def sniff_header(path: Path | str, *, encoding: str = "utf-8") -> EventListHeader:
    """Read just the first row to learn the schema."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return EventListHeader(columns=[], counter_columns=[])
    cols = [c.strip() for c in first]
    baseline = set()
    for hint in ALWAYS_PRESENT_HINTS:
        for c in cols:
            if c.lower() == hint.lower():
                baseline.add(c)
                break
    counter_cols = [c for c in cols if c not in baseline]
    return EventListHeader(columns=cols, counter_columns=counter_cols)


def iter_rows(
    path: Path | str, *, encoding: str = "utf-8"
) -> Iterator[dict[str, str]]:
    """Yield each event row as a dict keyed by normalized column name."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        norm = [_normalize_column(c) for c in reader.fieldnames]
        for row in reader:
            yield {
                norm_key: (row.get(orig_key) or "").strip()
                for norm_key, orig_key in zip(norm, reader.fieldnames)
            }


def parse_rows(
    path: Path | str,
    *,
    callback: Callable[[dict[str, str]], None],
    encoding: str = "utf-8",
) -> int:
    """Stream the CSV through ``callback``. Returns row count."""
    count = 0
    for row in iter_rows(path, encoding=encoding):
        callback(row)
        count += 1
    return count


def header_columns(path: Path | str, *, encoding: str = "utf-8") -> list[str]:
    """Return the normalized column names from the CSV header."""
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return []
    return [_normalize_column(c) for c in first]
