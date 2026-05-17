"""SQLite-backed index over a PIX capture's event list.

Goals:
  - Build once per capture (keyed by .wpix path + mtime), reuse forever.
  - Fine-grained queries (by global_id, marker, queue, event type, counter
    thresholds) without re-running pixtool every time.
  - Variable schema: counter columns are stored as JSON in ``counters`` so the
    DB doesn't need migration when a different ``--counters`` flag is used.
  - Markers are derived from PIX 'BeginEvent' / 'EndEvent' rows (depth-tracked)
    so we can answer "what's the global-id range for marker X?".

The DB lives next to the .wpix as ``<wpix>.pixmcp.db`` by default, or under
the configured cache dir if the wpix folder isn't writable.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import csv_parser
from .config import get_settings


SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS events (
    global_id INTEGER PRIMARY KEY,
    queue_id INTEGER,
    queue TEXT,
    name TEXT,
    event_type TEXT,           -- 'draw', 'dispatch', 'copy', 'begin_marker', 'end_marker', 'barrier', ...
    parent_marker_id INTEGER,  -- index into markers table (NULL at top level)
    depth INTEGER,
    counters TEXT,             -- JSON object of counter_name -> value
    raw TEXT                   -- JSON object of all CSV columns (for escape-hatch queries)
);

CREATE INDEX IF NOT EXISTS idx_events_name ON events(name);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_queue ON events(queue_id);
CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_marker_id);

CREATE TABLE IF NOT EXISTS markers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    start_global_id INTEGER,
    end_global_id INTEGER,
    parent_id INTEGER,
    depth INTEGER,
    queue_id INTEGER,
    queue TEXT
);

CREATE INDEX IF NOT EXISTS idx_markers_name ON markers(name);
CREATE INDEX IF NOT EXISTS idx_markers_range ON markers(start_global_id, end_global_id);

CREATE TABLE IF NOT EXISTS queues (
    queue_id INTEGER PRIMARY KEY,
    name TEXT,
    event_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS counter_columns (
    name TEXT PRIMARY KEY
);
"""


# Heuristics for classifying event names — covers the common D3D12 calls and
# pixtool's marker representation. We treat anything starting with 'BeginEvent'
# / 'EndEvent' as a marker, draw/dispatch/copy via name prefix, and 'Barrier'
# explicitly.
_EVENT_TYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^BeginEvent\b", re.I), "begin_marker"),
    (re.compile(r"^EndEvent\b", re.I), "end_marker"),
    (re.compile(r"\bSetMarker\b", re.I), "marker"),
    (re.compile(r"^(DrawInstanced|DrawIndexedInstanced|Draw)\b", re.I), "draw"),
    (re.compile(r"\bExecuteIndirect\b", re.I), "execute_indirect"),
    (re.compile(r"^Dispatch(Mesh|Rays)?\b", re.I), "dispatch"),
    (re.compile(r"^Copy(Resource|TextureRegion|BufferRegion|Tiles)?\b", re.I), "copy"),
    (re.compile(r"^Resolve(Subresource|QueryData)\b", re.I), "resolve"),
    (re.compile(r"\bClear(RenderTargetView|DepthStencilView|UnorderedAccessView)", re.I), "clear"),
    (re.compile(r"\bResourceBarrier\b", re.I), "barrier"),
    (re.compile(r"\bPresent\b", re.I), "present"),
    (re.compile(r"\bExecuteCommandLists\b", re.I), "execute_command_lists"),
    (re.compile(r"\bBuildRaytracingAccelerationStructure\b", re.I), "build_as"),
]


def classify_event(name: str) -> str:
    if not name:
        return "unknown"
    for pat, label in _EVENT_TYPE_RULES:
        if pat.search(name):
            return label
    return "other"


@dataclass
class IndexStats:
    event_count: int
    marker_count: int
    queue_count: int
    counter_columns: list[str]
    db_path: Path
    source_wpix: Path
    source_csv: Path


def _index_db_path(wpix_path: Path) -> Path:
    """Pick a writable location for the .pixmcp.db sidecar."""
    primary = wpix_path.with_suffix(wpix_path.suffix + ".pixmcp.db")
    try:
        primary.parent.mkdir(parents=True, exist_ok=True)
        if os.access(primary.parent, os.W_OK):
            return primary
    except Exception:
        pass
    cache = get_settings().ensure_cache_dir()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(wpix_path))
    return cache / f"{safe}.pixmcp.db"


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def index_status(wpix_path: Path, csv_path: Path) -> dict[str, Any]:
    """Report whether the index is current for the given wpix+csv pair."""
    db_path = _index_db_path(wpix_path)
    if not db_path.is_file():
        return {"exists": False, "db_path": str(db_path)}
    try:
        with _connect(db_path) as conn:
            try:
                conn.execute("SELECT 1 FROM events LIMIT 1")
            except sqlite3.OperationalError:
                return {"exists": False, "db_path": str(db_path), "reason": "no events table"}
            schema_v = _get_meta(conn, "schema_version")
            csv_mtime = _get_meta(conn, "csv_mtime")
            wpix_mtime = _get_meta(conn, "wpix_mtime")
            stored_csv = _get_meta(conn, "csv_path")
            event_count = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
    except sqlite3.DatabaseError as exc:
        return {"exists": False, "db_path": str(db_path), "reason": str(exc)}

    fresh_csv = csv_path.is_file() and str(int(csv_path.stat().st_mtime)) == csv_mtime
    fresh_wpix = wpix_path.is_file() and str(int(wpix_path.stat().st_mtime)) == wpix_mtime
    return {
        "exists": True,
        "db_path": str(db_path),
        "schema_version": int(schema_v) if schema_v else None,
        "csv_path": stored_csv,
        "csv_fresh": bool(fresh_csv),
        "wpix_fresh": bool(fresh_wpix),
        "event_count": event_count,
        "up_to_date": (
            fresh_csv
            and fresh_wpix
            and (int(schema_v) if schema_v else -1) == SCHEMA_VERSION
        ),
    }


def build_index(
    wpix_path: Path,
    csv_path: Path,
    *,
    rebuild: bool = False,
) -> IndexStats:
    """Build (or rebuild) the SQLite index from a save-event-list CSV."""
    db_path = _index_db_path(wpix_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"event-list CSV not found: {csv_path}")
    if rebuild and db_path.exists():
        db_path.unlink()

    header = csv_parser.sniff_header(csv_path)
    counter_columns = [csv_parser._normalize_column(c) for c in header.counter_columns]

    with _connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM markers")
        conn.execute("DELETE FROM queues")
        conn.execute("DELETE FROM counter_columns")
        for c in counter_columns:
            conn.execute("INSERT OR IGNORE INTO counter_columns(name) VALUES(?)", (c,))

        # Per-queue marker stacks for depth tracking.
        marker_stacks: dict[int, list[int]] = {}
        queue_event_counts: dict[int, str] = {}  # queue_id -> queue name (we accumulate counts at end)
        queue_count_acc: dict[int, int] = {}

        baseline_cols = {"global_id", "queue", "queue_id", "name"}

        def _coerce_int(s: str) -> int | None:
            if not s:
                return None
            try:
                return int(s)
            except ValueError:
                try:
                    return int(s, 0)
                except ValueError:
                    return None

        inserted = 0
        for row in csv_parser.iter_rows(csv_path):
            gid = _coerce_int(row.get("global_id", ""))
            if gid is None:
                continue
            queue_id = _coerce_int(row.get("queue_id", "")) or 0
            queue_name = row.get("queue", "") or ""
            name = row.get("name", "") or ""
            etype = classify_event(name)

            stack = marker_stacks.setdefault(queue_id, [])
            parent_marker_id = stack[-1] if stack else None
            depth = len(stack)

            counters_obj: dict[str, str] = {}
            raw_obj: dict[str, str] = {}
            for k, v in row.items():
                raw_obj[k] = v
                if k not in baseline_cols:
                    counters_obj[k] = v

            conn.execute(
                "INSERT OR REPLACE INTO events"
                "(global_id, queue_id, queue, name, event_type, parent_marker_id, depth, counters, raw) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    gid,
                    queue_id,
                    queue_name,
                    name,
                    etype,
                    parent_marker_id,
                    depth,
                    json.dumps(counters_obj, separators=(",", ":")),
                    json.dumps(raw_obj, separators=(",", ":")),
                ),
            )
            inserted += 1
            queue_count_acc[queue_id] = queue_count_acc.get(queue_id, 0) + 1
            queue_event_counts[queue_id] = queue_name

            if etype == "begin_marker":
                # The CSV usually carries the marker's label embedded in `name`,
                # e.g. "BeginEvent: ShadowPass" — strip the prefix.
                label = re.sub(r"^BeginEvent[: ]\s*", "", name, flags=re.I).strip()
                cur = conn.execute(
                    "INSERT INTO markers"
                    "(name, start_global_id, end_global_id, parent_id, depth, queue_id, queue) "
                    "VALUES (?, ?, NULL, ?, ?, ?, ?)",
                    (label or name, gid, parent_marker_id, depth, queue_id, queue_name),
                )
                marker_id = cur.lastrowid
                stack.append(marker_id)
                # Patch the event row's parent to its own marker so children pick it up.
                conn.execute(
                    "UPDATE events SET parent_marker_id = ? WHERE global_id = ?",
                    (parent_marker_id, gid),
                )
            elif etype == "end_marker":
                if stack:
                    marker_id = stack.pop()
                    conn.execute(
                        "UPDATE markers SET end_global_id = ? WHERE id = ?",
                        (gid, marker_id),
                    )

        for qid, qname in queue_event_counts.items():
            conn.execute(
                "INSERT OR REPLACE INTO queues(queue_id, name, event_count) VALUES(?, ?, ?)",
                (qid, qname, queue_count_acc.get(qid, 0)),
            )

        _set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        _set_meta(conn, "csv_path", str(csv_path))
        _set_meta(conn, "csv_mtime", str(int(csv_path.stat().st_mtime)))
        _set_meta(conn, "wpix_path", str(wpix_path))
        if wpix_path.is_file():
            _set_meta(conn, "wpix_mtime", str(int(wpix_path.stat().st_mtime)))
        _set_meta(conn, "counter_columns", json.dumps(counter_columns))

    with _connect(db_path) as conn:
        ev_count = conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        m_count = conn.execute("SELECT COUNT(*) AS c FROM markers").fetchone()["c"]
        q_count = conn.execute("SELECT COUNT(*) AS c FROM queues").fetchone()["c"]

    return IndexStats(
        event_count=ev_count,
        marker_count=m_count,
        queue_count=q_count,
        counter_columns=counter_columns,
        db_path=db_path,
        source_wpix=wpix_path,
        source_csv=csv_path,
    )


# ---- query helpers ----------------------------------------------------------


def open_db(wpix_path: Path) -> sqlite3.Connection:
    """Open an existing index (read-only-ish). Raises if it doesn't exist."""
    db_path = _index_db_path(wpix_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"index DB not found for {wpix_path} — call build_index first.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "global_id": row["global_id"],
        "queue_id": row["queue_id"],
        "queue": row["queue"],
        "name": row["name"],
        "event_type": row["event_type"],
        "parent_marker_id": row["parent_marker_id"],
        "depth": row["depth"],
        "counters": json.loads(row["counters"]) if row["counters"] else {},
    }


def get_event(wpix_path: Path, global_id: int, *, include_raw: bool = False) -> dict[str, Any] | None:
    with open_db(wpix_path) as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE global_id = ?", (global_id,)
        ).fetchone()
        if not row:
            return None
        out = _row_to_event(row)
        if include_raw and row["raw"]:
            out["raw"] = json.loads(row["raw"])
        return out


def find_events(
    wpix_path: Path,
    *,
    name_like: str | None = None,
    event_type: str | None = None,
    queue_id: int | None = None,
    marker_name: str | None = None,
    between: tuple[int, int] | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM events WHERE 1=1"
    params: list[Any] = []
    if name_like:
        sql += " AND name LIKE ?"
        params.append(name_like)
    if event_type:
        sql += " AND event_type = ?"
        params.append(event_type)
    if queue_id is not None:
        sql += " AND queue_id = ?"
        params.append(queue_id)
    if between:
        sql += " AND global_id BETWEEN ? AND ?"
        params.extend(between)
    if marker_name:
        # join to markers to bound by their range
        sql = (
            "SELECT e.* FROM events e "
            "JOIN markers m ON e.global_id BETWEEN m.start_global_id AND COALESCE(m.end_global_id, 1<<62) "
            "WHERE m.name LIKE ?"
        )
        params = [marker_name]
        if name_like:
            sql += " AND e.name LIKE ?"
            params.append(name_like)
        if event_type:
            sql += " AND e.event_type = ?"
            params.append(event_type)
        if queue_id is not None:
            sql += " AND e.queue_id = ?"
            params.append(queue_id)
        if between:
            sql += " AND e.global_id BETWEEN ? AND ?"
            params.extend(between)
    sql += " ORDER BY global_id LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with open_db(wpix_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]


def list_markers(
    wpix_path: Path,
    *,
    name_like: str | None = None,
    queue_id: int | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM markers WHERE 1=1"
    params: list[Any] = []
    if name_like:
        sql += " AND name LIKE ?"
        params.append(name_like)
    if queue_id is not None:
        sql += " AND queue_id = ?"
        params.append(queue_id)
    sql += " ORDER BY start_global_id LIMIT ?"
    params.append(limit)
    with open_db(wpix_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def list_queues(wpix_path: Path) -> list[dict[str, Any]]:
    with open_db(wpix_path) as conn:
        rows = conn.execute("SELECT * FROM queues ORDER BY queue_id").fetchall()
        return [dict(r) for r in rows]


def event_type_counts(wpix_path: Path) -> dict[str, int]:
    with open_db(wpix_path) as conn:
        rows = conn.execute(
            "SELECT event_type, COUNT(*) AS c FROM events GROUP BY event_type ORDER BY c DESC"
        ).fetchall()
        return {r["event_type"]: r["c"] for r in rows}


def top_by_counter(
    wpix_path: Path, counter: str, *, n: int = 25, event_type: str | None = None
) -> list[dict[str, Any]]:
    """Sort events by a counter value (numeric). The counter must be a column we indexed."""
    json_path = f"$.{counter}"
    sql = (
        "SELECT global_id, queue, name, event_type, "
        "       CAST(json_extract(counters, ?) AS REAL) AS counter_value "
        "FROM events WHERE json_extract(counters, ?) IS NOT NULL"
    )
    params: list[Any] = [json_path, json_path]
    if event_type:
        sql += " AND event_type = ?"
        params.append(event_type)
    sql += " ORDER BY counter_value DESC LIMIT ?"
    params.append(n)
    with open_db(wpix_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def raw_sql(wpix_path: Path, sql: str, params: list[Any] | None = None, *, limit: int = 1000) -> list[dict[str, Any]]:
    """Escape hatch — run an arbitrary SELECT against the index.

    Refuses anything that isn't read-only.
    """
    stripped = sql.strip().rstrip(";")
    lowered = stripped.lower()
    if not lowered.startswith(("select", "with", "explain")):
        raise ValueError("raw_sql only accepts SELECT / WITH / EXPLAIN statements.")
    forbidden = re.compile(
        r"\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace)\b", re.I
    )
    if forbidden.search(stripped):
        raise ValueError("raw_sql refuses write/DDL statements.")
    with open_db(wpix_path) as conn:
        cur = conn.execute(stripped, params or [])
        rows = cur.fetchmany(limit)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in rows]
