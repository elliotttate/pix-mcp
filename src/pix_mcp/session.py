"""Session manager for pix-mcp.

Two kinds of sessions:

* **CaptureSession** — a loaded .wpix file. Owns its on-disk SQLite index and
  any cached C++-export parse. Stateless w.r.t. pixtool processes; each pixtool
  invocation re-opens the wpix.

* **LaunchSession** — a long-running background ``pixtool launch <exe>``
  process. Keeps the game running so the user can hit F11 to capture; the MCP
  reports its stdout/stderr on demand.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cpp_export, index
from .pixtool import BackgroundProcess


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    return s[:48] or "session"


@dataclass
class CaptureSession:
    handle: str
    wpix_path: Path
    csv_path: Path | None = None      # last save-event-list output we know about
    cpp_export_dir: Path | None = None
    cpp_export_parsed: cpp_export.CppExport | None = None
    cpp_export_fingerprint: str | None = None
    indexed: bool = False
    indexed_at: float | None = None
    last_index_stats: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "wpix_path": str(self.wpix_path),
            "csv_path": str(self.csv_path) if self.csv_path else None,
            "cpp_export_dir": str(self.cpp_export_dir) if self.cpp_export_dir else None,
            "cpp_call_count": (
                self.cpp_export_parsed.call_count if self.cpp_export_parsed else None
            ),
            "indexed": self.indexed,
            "indexed_at": self.indexed_at,
            "index_stats": self.last_index_stats,
        }

    def ensure_cpp_export(self) -> cpp_export.CppExport:
        if self.cpp_export_dir is None:
            raise RuntimeError(
                "no C++ export attached to this session. "
                "Call export_to_cpp first (or attach_cpp_export)."
            )
        fp = cpp_export.fingerprint_dir(self.cpp_export_dir)
        if self.cpp_export_parsed is None or self.cpp_export_fingerprint != fp:
            self.cpp_export_parsed = cpp_export.parse_export(self.cpp_export_dir)
            self.cpp_export_fingerprint = fp
        return self.cpp_export_parsed


@dataclass
class LaunchSession:
    handle: str
    exe: str
    args: list[str]
    bg: BackgroundProcess
    notes: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "exe": self.exe,
            "args": list(self.args),
            "pid": self.bg.proc.pid,
            "running": self.bg.is_running(),
            "returncode": self.bg.returncode(),
            "uptime_sec": round(time.monotonic() - self.bg.started_at, 2),
            "notes": self.notes,
            "cmdline": self.bg.cmdline,
        }


class SessionManager:
    """Thread-safe registry of capture + launch sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._captures: dict[str, CaptureSession] = {}
        self._launches: dict[str, LaunchSession] = {}

    # ---- capture sessions ----------------------------------------------

    def open_capture(self, wpix_path: Path, *, handle: str | None = None) -> CaptureSession:
        wpix_path = wpix_path.resolve()
        if not wpix_path.is_file():
            raise FileNotFoundError(f"wpix not found: {wpix_path}")
        with self._lock:
            # If we already have a session for this path, return it.
            for sess in self._captures.values():
                if sess.wpix_path == wpix_path:
                    return sess
            h = handle or f"cap_{_slug(wpix_path.stem)}_{secrets.token_hex(3)}"
            sess = CaptureSession(handle=h, wpix_path=wpix_path)
            self._captures[h] = sess
            return sess

    def get_capture(self, handle: str) -> CaptureSession:
        with self._lock:
            sess = self._captures.get(handle)
            if sess is None:
                # Allow resolution by exact wpix path too.
                p = Path(handle)
                if p.is_file():
                    for s in self._captures.values():
                        if s.wpix_path == p.resolve():
                            return s
                raise KeyError(f"capture session not found: {handle}")
            return sess

    def list_captures(self) -> list[CaptureSession]:
        with self._lock:
            return list(self._captures.values())

    def close_capture(self, handle: str) -> bool:
        with self._lock:
            return self._captures.pop(handle, None) is not None

    def mark_indexed(self, sess: CaptureSession, stats: index.IndexStats) -> None:
        sess.indexed = True
        sess.indexed_at = time.time()
        sess.last_index_stats = {
            "event_count": stats.event_count,
            "marker_count": stats.marker_count,
            "queue_count": stats.queue_count,
            "counter_columns": stats.counter_columns,
            "db_path": str(stats.db_path),
        }

    # ---- launch sessions -----------------------------------------------

    def register_launch(
        self,
        bg: BackgroundProcess,
        *,
        exe: str,
        args: list[str],
        notes: str = "",
        handle: str | None = None,
    ) -> LaunchSession:
        with self._lock:
            h = handle or f"run_{_slug(Path(exe).stem)}_{secrets.token_hex(3)}"
            sess = LaunchSession(handle=h, exe=exe, args=list(args), bg=bg, notes=notes)
            self._launches[h] = sess
            return sess

    def get_launch(self, handle: str) -> LaunchSession:
        with self._lock:
            sess = self._launches.get(handle)
            if sess is None:
                raise KeyError(f"launch session not found: {handle}")
            return sess

    def list_launches(self) -> list[LaunchSession]:
        with self._lock:
            return list(self._launches.values())

    def stop_launch(self, handle: str, *, timeout: float = 5.0) -> int:
        with self._lock:
            sess = self._launches.pop(handle, None)
        if sess is None:
            raise KeyError(f"launch session not found: {handle}")
        return sess.bg.terminate(timeout=timeout)

    def reap(self) -> list[str]:
        """Remove exited launch sessions; return their handles."""
        reaped: list[str] = []
        with self._lock:
            for h, sess in list(self._launches.items()):
                if not sess.bg.is_running():
                    del self._launches[h]
                    reaped.append(h)
        return reaped


# Process-wide singleton — the MCP server uses this.
_sessions: SessionManager | None = None


def get_sessions() -> SessionManager:
    global _sessions
    if _sessions is None:
        _sessions = SessionManager()
    return _sessions
