"""pix-mcp MCP server.

Exposes pixtool.exe as MCP tools, plus an indexing layer that turns
``save-event-list`` CSV and ``export-to-cpp`` output into fast, fine-grained
queries (event lookup by global ID, marker ranges, state-at-event bindings, etc.).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import cpp_export, csv_parser, index
from .config import default_pix_captures_dir, get_settings, reload_settings
from .pixtool import (
    PixCommand,
    PixtoolError,
    PixtoolResult,
    cmd,
    run_pixtool_async,
    start_background_pixtool,
)
from .session import CaptureSession, get_sessions


mcp = FastMCP(
    "pix-mcp",
    instructions=(
        "MCP server for Microsoft PIX (pixtool.exe). Use pix_environment first to confirm "
        "the pixtool install. For live capture, pix_capture_launched runs the whole "
        "launch→capture→save pipeline in one call; pix_launch_background keeps a game "
        "running so you can press F11 in-app to capture. For analysis, pix_open_capture "
        "registers a .wpix, pix_index_events builds a queryable SQLite index, and "
        "pix_state_at_event answers 'what's bound at root param X of event Y' by parsing "
        "the export-to-cpp output."
    ),
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _resolve_wpix(capture: str) -> tuple[CaptureSession | None, Path]:
    """Accept either a session handle or a .wpix path."""
    sessions = get_sessions()
    # 1) Handle?
    try:
        sess = sessions.get_capture(capture)
        return sess, sess.wpix_path
    except KeyError:
        pass
    # 2) Path?
    p = Path(capture).expanduser()
    if p.is_file():
        return None, p.resolve()
    raise FileNotFoundError(
        f"'{capture}' is neither an open capture handle nor a path to a .wpix file."
    )


def _open_chain(wpix: Path, *, remote: str | None = None) -> list[PixCommand]:
    """Standard prefix for any wpix-targeting pixtool command chain."""
    opts: dict[str, Any] = {}
    if remote:
        opts["remote"] = remote
    return [cmd("open-capture", str(wpix), **opts)]


def _pixresult_to_dict(r: PixtoolResult, *, tail: int = 4000) -> dict[str, Any]:
    return {
        "returncode": r.returncode,
        "duration_sec": round(r.duration_sec, 3),
        "cmdline": r.cmdline,
        "stdout_tail": r.stdout[-tail:],
        "stderr_tail": r.stderr[-tail:],
        "ok": r.ok,
    }


async def _run_chain(
    chain: list[PixCommand],
    *,
    check: bool = True,
    timeout: float | None = None,
    log_file: Path | str | None = None,
) -> PixtoolResult:
    try:
        return await run_pixtool_async(chain, timeout=timeout, log_file=log_file, check=check)
    except (PixtoolError, FileNotFoundError) as exc:
        raise RuntimeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# environment / diagnostics
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_environment(refresh: bool = False) -> dict[str, Any]:
    """Report the resolved pixtool.exe path, cache dir, captures dir, and version banner.

    Args:
        refresh: re-discover pixtool (use after installing/upgrading PIX).
    """
    s = reload_settings() if refresh else get_settings()
    info: dict[str, Any] = {
        "pixtool_path": str(s.pixtool_path) if s.pixtool_path else None,
        "pixtool_found": s.pixtool_path is not None and s.pixtool_path.is_file(),
        "cache_dir": str(s.cache_dir),
        "captures_dir": str(s.captures_dir),
        "captures_dir_exists": s.captures_dir.is_dir(),
    }
    if info["pixtool_found"]:
        # pixtool --help spits the banner + command list; capture it briefly.
        try:
            proc = subprocess.run(
                [str(s.pixtool_path), "--help"],
                capture_output=True,
                text=True,
                timeout=15,
                encoding="utf-8",
                errors="replace",
            )
            head = (proc.stdout or proc.stderr or "").splitlines()[:4]
            info["pixtool_banner"] = "\n".join(head)
        except Exception as exc:
            info["pixtool_banner_error"] = str(exc)
    return info


@mcp.tool()
async def pix_raw(
    args: list[str],
    *,
    timeout: float | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Escape hatch: run pixtool with arbitrary args (no chaining, no parsing).

    The first argument should be a pixtool subcommand (e.g. ``open-capture``).
    Useful for commands the typed tools don't expose, or for chaining beyond
    what the typed API supports.

    Example::

        pix_raw(args=["open-capture", "C:/cap.wpix", "list-counters"])
    """
    s = get_settings()
    pixtool = s.require_pixtool()
    cmdline = [str(pixtool), *args]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmdline,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout if timeout is not None else s.pixtool_timeout_sec,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"pixtool timed out: {exc}") from exc
    dt = time.monotonic() - t0
    return {
        "returncode": proc.returncode,
        "duration_sec": round(dt, 3),
        "cmdline": cmdline,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


# ---------------------------------------------------------------------------
# one-shot live capture workflows
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_capture_launched(
    exe: str,
    output_wpix: str,
    *,
    args: str | None = None,
    working_directory: str | None = None,
    setenv: dict[str, str] | None = None,
    frames: int = 1,
    remote: str | None = None,
    capture_from_start: bool = False,
    open_after: bool = False,
    use_d3d11on12: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    """One-shot: launch ``exe`` under PIX, take a GPU capture, save it to ``output_wpix``.

    This is the simplest path for capturing a non-interactive workload (e.g.
    a benchmark/demo). For interactive use where you want to press F11 in-game,
    use ``pix_launch_background``.
    """
    chain: list[PixCommand] = []
    if frames and frames > 1:
        chain.append(cmd("set-gpu-capture-parameters", frames=frames))
    launch_opts: dict[str, Any] = {}
    if args:
        launch_opts["command-line"] = args
    if working_directory:
        launch_opts["working-directory"] = working_directory
    if remote:
        launch_opts["remote"] = remote
    if capture_from_start:
        launch_opts["captureFromStart"] = True
    if use_d3d11on12:
        launch_opts["force11on12"] = True
    chain.append(cmd("launch", exe, **launch_opts))
    if setenv:
        for k, v in setenv.items():
            chain.append(cmd("launch", **{"setenv": f"{k}={v}"}))  # type: ignore[arg-type]
    take_opts: dict[str, Any] = {"frames": frames} if frames and frames > 1 else {}
    if open_after:
        take_opts["open"] = True
    chain.append(cmd("take-capture", **take_opts))
    chain.append(cmd("save-capture", output_wpix))
    result = await _run_chain(chain, timeout=timeout)
    return {
        "result": _pixresult_to_dict(result),
        "output_wpix": output_wpix,
    }


@mcp.tool()
async def pix_capture_attached(
    pid: int,
    output_wpix: str,
    *,
    frames: int = 1,
    remote: str | None = None,
    open_after: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    """One-shot: attach to a running process, take a capture, save it.

    The target process must already be running with a D3D12 swapchain. PIX's
    capture engine attaches via WinPix services.
    """
    chain: list[PixCommand] = []
    if frames and frames > 1:
        chain.append(cmd("set-gpu-capture-parameters", frames=frames))
    attach_opts: dict[str, Any] = {}
    if remote:
        attach_opts["remote"] = remote
    chain.append(cmd("attach", pid, **attach_opts))
    take_opts: dict[str, Any] = {"frames": frames} if frames and frames > 1 else {}
    if open_after:
        take_opts["open"] = True
    chain.append(cmd("take-capture", **take_opts))
    chain.append(cmd("save-capture", output_wpix))
    result = await _run_chain(chain, timeout=timeout)
    return {
        "result": _pixresult_to_dict(result),
        "output_wpix": output_wpix,
    }


@mcp.tool()
async def pix_capture_programmatic(
    exe: str,
    output_wpix: str,
    *,
    args: str | None = None,
    working_directory: str | None = None,
    until_exit: bool = False,
    open_after: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Launch ``exe`` and wait for the app to trigger a programmatic capture
    (via the PIX programmatic-capture API).
    """
    chain: list[PixCommand] = []
    launch_opts: dict[str, Any] = {}
    if args:
        launch_opts["command-line"] = args
    if working_directory:
        launch_opts["working-directory"] = working_directory
    chain.append(cmd("launch", exe, **launch_opts))
    prog_opts: dict[str, Any] = {}
    if open_after:
        prog_opts["open"] = True
    if until_exit:
        prog_opts["until-exit"] = True
    chain.append(cmd("programmatic-capture", **prog_opts))
    chain.append(cmd("save-capture", output_wpix))
    result = await _run_chain(chain, timeout=timeout)
    return {"result": _pixresult_to_dict(result), "output_wpix": output_wpix}


# ---------------------------------------------------------------------------
# background launch (interactive F11 workflow)
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_launch_background(
    exe: str,
    *,
    args: str | None = None,
    working_directory: str | None = None,
    setenv: dict[str, str] | None = None,
    capture_key: str = "F11",
    frames: int = 1,
    remote: str | None = None,
    capture_from_start: bool = False,
    use_d3d11on12: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    """Start ``pixtool launch <exe>`` in the background so the user can press
    the capture key (default F11) inside the game. Returns a handle the caller
    uses to query status or terminate.

    Captures triggered via the in-game key are written by PIX into
    ``Documents\\PIX\\Captures``. Use ``pix_find_recent_captures`` to discover them.
    """
    chain: list[PixCommand] = []
    chain.append(cmd("set-gpu-capture-parameters", frames=frames, **{"capture-key": capture_key}))
    launch_opts: dict[str, Any] = {}
    if args:
        launch_opts["command-line"] = args
    if working_directory:
        launch_opts["working-directory"] = working_directory
    if remote:
        launch_opts["remote"] = remote
    if capture_from_start:
        launch_opts["captureFromStart"] = True
    if use_d3d11on12:
        launch_opts["force11on12"] = True
    if setenv:
        # pixtool accepts repeated --setenv on the launch command.
        for k, v in setenv.items():
            launch_opts.setdefault("__setenv_extras", []).append(f"{k}={v}")
    extras = launch_opts.pop("__setenv_extras", []) if launch_opts else []
    chain.append(cmd("launch", exe, **launch_opts))
    for extra in extras:
        # Reissue the same launch with the extra env var — pixtool repeats are
        # cumulative on the prior launch.
        chain.append(cmd("launch", exe, setenv=extra))

    handle = f"run_{Path(exe).stem}_{int(time.time()) & 0xFFFFFF:06x}"
    bg = start_background_pixtool(handle, chain)
    sess = get_sessions().register_launch(
        bg,
        exe=exe,
        args=shlex.split(args) if args else [],
        notes=notes,
        handle=handle,
    )
    return sess.summary()


@mcp.tool()
async def pix_launch_status(handle: str) -> dict[str, Any]:
    """Status, recent stdout/stderr (last ~200 lines) for a background launch."""
    sess = get_sessions().get_launch(handle)
    out = sess.summary()
    out["stdout_tail"] = sess.bg.recent_stdout(200)
    out["stderr_tail"] = sess.bg.recent_stderr(200)
    return out


@mcp.tool()
async def pix_list_background_launches() -> dict[str, Any]:
    """List all active background pixtool launches."""
    sessions = get_sessions()
    sessions.reap()
    return {"launches": [s.summary() for s in sessions.list_launches()]}


@mcp.tool()
async def pix_stop_background_launch(handle: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """Terminate a background launch. Returns its exit code."""
    rc = get_sessions().stop_launch(handle, timeout=timeout)
    return {"handle": handle, "returncode": rc}


@mcp.tool()
async def pix_find_recent_captures(
    directory: str | None = None,
    *,
    limit: int = 25,
    extensions: list[str] | None = None,
) -> dict[str, Any]:
    """List recent capture files (.wpix by default) in PIX's captures directory."""
    exts = [e.lower() if e.startswith(".") else "." + e.lower() for e in (extensions or [".wpix"])]
    target = Path(directory).expanduser() if directory else default_pix_captures_dir()
    if not target.is_dir():
        return {"directory": str(target), "exists": False, "captures": []}
    candidates: list[tuple[float, Path]] = []
    for p in target.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                candidates.append((p.stat().st_mtime, p))
            except OSError:
                continue
    candidates.sort(reverse=True)
    out = []
    for mtime, p in candidates[:limit]:
        st = p.stat()
        out.append({
            "path": str(p),
            "name": p.name,
            "size_bytes": st.st_size,
            "mtime": mtime,
            "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime)),
        })
    return {"directory": str(target), "exists": True, "captures": out}


# ---------------------------------------------------------------------------
# capture session lifecycle
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_open_capture(path: str) -> dict[str, Any]:
    """Register a .wpix in the session manager and return a handle.

    This doesn't run pixtool — it just bookkeeps. Each tool that needs the
    capture will pass ``open-capture <path>`` to pixtool itself.
    """
    sess = get_sessions().open_capture(Path(path).expanduser())
    return sess.summary()


@mcp.tool()
async def pix_close_capture(handle: str) -> dict[str, Any]:
    """Drop a registered capture session (does not delete files)."""
    ok = get_sessions().close_capture(handle)
    return {"handle": handle, "closed": ok}


@mcp.tool()
async def pix_list_open_captures() -> dict[str, Any]:
    """List all currently-registered capture sessions."""
    return {"captures": [s.summary() for s in get_sessions().list_captures()]}


# ---------------------------------------------------------------------------
# coarse pixtool wrappers (raw-to-disk)
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_save_event_list(
    capture: str,
    output_csv: str,
    *,
    counters: list[str] | None = None,
    counter_groups: list[str] | None = None,
    queue_name: str | None = None,
    remote: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run ``open-capture … save-event-list``. Saves an event-list CSV to disk.

    ``counters`` and ``counter_groups`` accept glob patterns (e.g. ``"D3D:*"``).
    They map to repeated ``--counters=``/``--counter-groups=`` flags.
    """
    sess, wpix = _resolve_wpix(capture)
    chain = _open_chain(wpix, remote=remote)
    save_opts: dict[str, Any] = {}
    if queue_name:
        save_opts["queue-name"] = queue_name
    save_cmd = cmd("save-event-list", output_csv, **save_opts)
    # We need to express repeated --counters/--counter-groups, which our PixCommand
    # builder supports via tuple appends.
    extras: list[tuple[str, str]] = []
    for c in counters or []:
        extras.append(("--counters", c))
    for g in counter_groups or []:
        extras.append(("--counter-groups", g))
    if extras:
        save_cmd = PixCommand(
            name=save_cmd.name,
            positional=save_cmd.positional,
            options=save_cmd.options + tuple((flag, value) for flag, value in extras),
        )
    chain.append(save_cmd)
    result = await _run_chain(chain, timeout=timeout)
    out = {
        "result": _pixresult_to_dict(result),
        "output_csv": output_csv,
        "csv_exists": Path(output_csv).is_file(),
    }
    if sess and Path(output_csv).is_file():
        sess.csv_path = Path(output_csv).resolve()
    return out


@mcp.tool()
async def pix_save_resource(
    capture: str,
    output_path: str,
    *,
    global_id: int | None = None,
    marker: str | None = None,
    rtv: int | None = None,
    depth: bool = False,
    remote: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Save a single resource to disk at a specific event.

    For visual buffers (RTVs/DSVs/UAVs) the file extension picks the format
    (PNG/DDS). Exactly one of ``global_id`` or ``marker`` should be provided.
    """
    if global_id is None and marker is None:
        raise ValueError("pix_save_resource requires either global_id or marker.")
    _, wpix = _resolve_wpix(capture)
    chain = _open_chain(wpix, remote=remote)
    opts: dict[str, Any] = {}
    if global_id is not None:
        opts["global-id"] = global_id
    if marker is not None:
        opts["marker"] = marker
    if rtv is not None:
        opts["rtv"] = rtv
    if depth:
        opts["depth"] = True
    chain.append(cmd("save-resource", output_path, **opts))
    result = await _run_chain(chain, timeout=timeout)
    return {
        "result": _pixresult_to_dict(result),
        "output_path": output_path,
        "output_exists": Path(output_path).is_file(),
    }


@mcp.tool()
async def pix_save_screenshot(
    capture: str,
    output_png: str,
    *,
    remote: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Save the capture's swapchain screenshot as PNG."""
    _, wpix = _resolve_wpix(capture)
    chain = _open_chain(wpix, remote=remote)
    chain.append(cmd("save-screenshot", output_png))
    result = await _run_chain(chain, timeout=timeout)
    return {
        "result": _pixresult_to_dict(result),
        "output_png": output_png,
        "output_exists": Path(output_png).is_file(),
    }


@mcp.tool()
async def pix_list_counters(
    capture: str | None = None,
    *,
    remote: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """List counters available for an open capture (some counters are HW-dependent)."""
    chain: list[PixCommand] = []
    if capture:
        _, wpix = _resolve_wpix(capture)
        chain.extend(_open_chain(wpix, remote=remote))
    chain.append(cmd("list-counters"))
    result = await _run_chain(chain, timeout=timeout)
    # The counters list is plain text on stdout. We pass it through verbatim and
    # also try to split into a list.
    lines = [
        ln.strip()
        for ln in result.stdout.splitlines()
        if ln.strip() and not ln.strip().startswith(("Available", "Usage", "---"))
    ]
    return {
        "result": _pixresult_to_dict(result),
        "counters_text": result.stdout,
        "counters_lines": lines,
    }


@mcp.tool()
async def pix_save_high_frequency_counters(
    capture: str,
    output_csv: str,
    *,
    counters: list[str] | None = None,
    counter_groups: list[str] | None = None,
    remote: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Collect and save High Frequency Counters CSV."""
    _, wpix = _resolve_wpix(capture)
    chain = _open_chain(wpix, remote=remote)
    save_cmd = cmd("save-high-frequency-counters", output_csv)
    extras: list[tuple[str, str]] = []
    for c in counters or []:
        extras.append(("--counters", c))
    for g in counter_groups or []:
        extras.append(("--counter-groups", g))
    if extras:
        save_cmd = PixCommand(
            name=save_cmd.name,
            positional=save_cmd.positional,
            options=save_cmd.options + tuple(extras),
        )
    chain.append(save_cmd)
    result = await _run_chain(chain, timeout=timeout)
    return {
        "result": _pixresult_to_dict(result),
        "output_csv": output_csv,
        "csv_exists": Path(output_csv).is_file(),
    }


@mcp.tool()
async def pix_export_to_cpp(
    capture: str,
    output_dir: str,
    *,
    force: bool = False,
    use_winpixeventruntime: bool = False,
    use_agility_sdk: bool = False,
    use_replay_time_executeindirect_buffers: bool = False,
    remote: str | None = None,
    timeout: float | None = None,
    parse_after: bool = True,
) -> dict[str, Any]:
    """Export the capture as a C++ replay project. If ``parse_after`` is true,
    the project is parsed immediately and attached to the session so subsequent
    ``pix_state_at_event`` calls are instant.
    """
    sess, wpix = _resolve_wpix(capture)
    chain = _open_chain(wpix, remote=remote)
    opts: dict[str, Any] = {}
    if force:
        opts["force"] = True
    if use_winpixeventruntime:
        opts["use-winpixeventruntime"] = True
    if use_agility_sdk:
        opts["use-agilitySdk"] = True
    if use_replay_time_executeindirect_buffers:
        opts["use-replay-time-executeindirect-buffers"] = True
    chain.append(cmd("export-to-cpp", output_dir, **opts))
    result = await _run_chain(chain, timeout=timeout)
    out: dict[str, Any] = {
        "result": _pixresult_to_dict(result),
        "output_dir": output_dir,
        "output_dir_exists": Path(output_dir).is_dir(),
    }
    if parse_after and Path(output_dir).is_dir():
        export = cpp_export.parse_export(Path(output_dir))
        if sess is None:
            # Auto-register a session so future calls can reference the cache.
            sess = get_sessions().open_capture(wpix)
        sess.cpp_export_dir = Path(output_dir).resolve()
        sess.cpp_export_parsed = export
        sess.cpp_export_fingerprint = cpp_export.fingerprint_dir(Path(output_dir))
        out["parsed"] = {
            "call_count": export.call_count,
            "files": [str(p.relative_to(Path(output_dir))) for p in export.files_parsed],
            "global_ids_seen": export.event_ids_seen[:50],
            "global_ids_count": len(export.event_ids_seen),
        }
        out["session_handle"] = sess.handle
    return out


@mcp.tool()
async def pix_recapture_region(
    capture: str,
    output_wpix: str,
    *,
    start: int,
    end: int,
    remote: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Recapture a region of an existing capture into a new .wpix.

    ``start`` and ``end`` are inclusive Global IDs.
    """
    _, wpix = _resolve_wpix(capture)
    chain = _open_chain(wpix, remote=remote)
    chain.append(cmd("recapture-region", output_wpix, start=start, end=end))
    result = await _run_chain(chain, timeout=timeout)
    return {
        "result": _pixresult_to_dict(result),
        "output_wpix": output_wpix,
        "output_exists": Path(output_wpix).is_file(),
    }


@mcp.tool()
async def pix_save_capture(
    capture: str,
    output_wpix: str,
    *,
    remote: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Re-save the currently-open capture under a new filename (e.g. after upgrading)."""
    _, wpix = _resolve_wpix(capture)
    chain = _open_chain(wpix, remote=remote)
    chain.append(cmd("save-capture", output_wpix))
    result = await _run_chain(chain, timeout=timeout)
    return {
        "result": _pixresult_to_dict(result),
        "output_wpix": output_wpix,
        "output_exists": Path(output_wpix).is_file(),
    }


@mcp.tool()
async def pix_upgrade_gpu_capture(
    source_wpix: str,
    *,
    dest_wpix: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Upgrade an older .wpix to the latest format."""
    args = ["upgrade-gpu-capture", source_wpix]
    if dest_wpix:
        args.append(f"--dest={dest_wpix}")
    return await pix_raw(args=args, timeout=timeout)


# ---------------------------------------------------------------------------
# index build / query
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_index_events(
    capture: str,
    *,
    counters: list[str] | None = None,
    counter_groups: list[str] | None = None,
    queue_name: str | None = None,
    rebuild: bool = False,
    remote: str | None = None,
    csv_path: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Build (or refresh) the SQLite index for a capture.

    Runs ``save-event-list`` if no fresh CSV exists, then ingests it into a
    SQLite DB beside the .wpix. After this, ``pix_find_events`` / ``pix_get_event``
    / ``pix_top_by_counter`` / ``pix_query_sql`` work without re-running pixtool.

    If ``csv_path`` is provided and exists, no pixtool is run — the existing
    CSV is ingested directly.
    """
    sess, wpix = _resolve_wpix(capture)
    if sess is None:
        sess = get_sessions().open_capture(wpix)

    # Choose CSV path.
    if csv_path:
        csv = Path(csv_path).resolve()
    else:
        cache = get_settings().ensure_cache_dir()
        safe = wpix.stem.replace(" ", "_")
        csv = cache / f"{safe}.event_list.csv"

    # Decide if we need to re-run pixtool.
    need_run = True
    if csv.is_file() and not rebuild:
        # If the wpix is older than the CSV and counter selection looks the same,
        # we can reuse it. We don't store the previous counter selection though,
        # so be conservative: reuse only if no counters were requested.
        if wpix.stat().st_mtime <= csv.stat().st_mtime and not counters and not counter_groups:
            need_run = False

    pix_result: PixtoolResult | None = None
    if need_run:
        chain = _open_chain(wpix, remote=remote)
        save_cmd = cmd("save-event-list", str(csv), **({"queue-name": queue_name} if queue_name else {}))
        extras: list[tuple[str, str]] = []
        for c in counters or []:
            extras.append(("--counters", c))
        for g in counter_groups or []:
            extras.append(("--counter-groups", g))
        if extras:
            save_cmd = PixCommand(
                name=save_cmd.name,
                positional=save_cmd.positional,
                options=save_cmd.options + tuple(extras),
            )
        chain.append(save_cmd)
        pix_result = await _run_chain(chain, timeout=timeout)
        if not csv.is_file():
            raise RuntimeError(
                f"save-event-list ran but CSV missing at {csv}\n"
                f"stderr tail: {pix_result.stderr[-1500:]}"
            )

    stats = index.build_index(wpix, csv, rebuild=rebuild)
    sess.csv_path = csv
    get_sessions().mark_indexed(sess, stats)

    return {
        "handle": sess.handle,
        "csv_path": str(csv),
        "pixtool": _pixresult_to_dict(pix_result) if pix_result else {"skipped": True},
        "index": {
            "event_count": stats.event_count,
            "marker_count": stats.marker_count,
            "queue_count": stats.queue_count,
            "counter_columns": stats.counter_columns,
            "db_path": str(stats.db_path),
        },
    }


@mcp.tool()
async def pix_index_status(capture: str) -> dict[str, Any]:
    """Report whether the SQLite index is current for a capture."""
    sess, wpix = _resolve_wpix(capture)
    csv = sess.csv_path if sess and sess.csv_path else None
    if csv is None:
        # Best-effort guess at where the indexer would have placed it.
        cache = get_settings().ensure_cache_dir()
        candidate = cache / f"{wpix.stem.replace(' ', '_')}.event_list.csv"
        if candidate.is_file():
            csv = candidate
    if csv is None:
        return {"indexed": False, "reason": "no CSV recorded for this capture"}
    return index.index_status(wpix, csv)


@mcp.tool()
async def pix_get_event(capture: str, global_id: int, *, include_raw: bool = False) -> dict[str, Any]:
    """Get full event details for a single Global ID."""
    _, wpix = _resolve_wpix(capture)
    ev = index.get_event(wpix, global_id, include_raw=include_raw)
    if ev is None:
        return {"found": False, "global_id": global_id}
    return {"found": True, "event": ev}


@mcp.tool()
async def pix_find_events(
    capture: str,
    *,
    name_like: str | None = None,
    event_type: str | None = None,
    queue_id: int | None = None,
    inside_marker: str | None = None,
    between: list[int] | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    """Filter the indexed event list.

    Pattern matching uses SQL ``LIKE`` (use ``%`` as a wildcard). ``event_type``
    is one of: ``draw``, ``dispatch``, ``copy``, ``barrier``, ``begin_marker``,
    ``end_marker``, ``marker``, ``clear``, ``resolve``, ``present``,
    ``execute_command_lists``, ``execute_indirect``, ``build_as``, ``other``.
    """
    _, wpix = _resolve_wpix(capture)
    between_tuple: tuple[int, int] | None = None
    if between:
        if len(between) != 2:
            raise ValueError("'between' must be a 2-element [start, end] list.")
        between_tuple = (int(between[0]), int(between[1]))
    rows = index.find_events(
        wpix,
        name_like=name_like,
        event_type=event_type,
        queue_id=queue_id,
        marker_name=inside_marker,
        between=between_tuple,
        limit=limit,
        offset=offset,
    )
    return {"count": len(rows), "events": rows}


@mcp.tool()
async def pix_list_markers(
    capture: str,
    *,
    name_like: str | None = None,
    queue_id: int | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """List PIX markers extracted from the event list (range per marker included)."""
    _, wpix = _resolve_wpix(capture)
    rows = index.list_markers(wpix, name_like=name_like, queue_id=queue_id, limit=limit)
    return {"count": len(rows), "markers": rows}


@mcp.tool()
async def pix_list_queues(capture: str) -> dict[str, Any]:
    """List queues seen in this capture, with event counts."""
    _, wpix = _resolve_wpix(capture)
    rows = index.list_queues(wpix)
    return {"queues": rows}


@mcp.tool()
async def pix_event_type_counts(capture: str) -> dict[str, Any]:
    """Histogram of event types across the whole capture (draws vs dispatches vs barriers …)."""
    _, wpix = _resolve_wpix(capture)
    return {"counts": index.event_type_counts(wpix)}


@mcp.tool()
async def pix_top_by_counter(
    capture: str,
    counter: str,
    *,
    n: int = 25,
    event_type: str | None = None,
) -> dict[str, Any]:
    """Sort events by a numeric counter (e.g. ``gpu_duration``). The counter
    must have been included when the index was built (``pix_index_events
    --counters=…``). Use ``pix_index_status`` to see which counter columns exist.
    """
    _, wpix = _resolve_wpix(capture)
    rows = index.top_by_counter(wpix, counter, n=n, event_type=event_type)
    return {"counter": counter, "rows": rows}


@mcp.tool()
async def pix_marker_range(capture: str, marker_name: str) -> dict[str, Any]:
    """Resolve a marker name to its [start, end] Global IDs (the first match)."""
    _, wpix = _resolve_wpix(capture)
    rows = index.list_markers(wpix, name_like=marker_name, limit=5)
    if not rows:
        return {"found": False, "marker_name": marker_name}
    m = rows[0]
    return {
        "found": True,
        "marker": m,
        "start": m["start_global_id"],
        "end": m["end_global_id"],
    }


@mcp.tool()
async def pix_query_sql(
    capture: str,
    sql: str,
    *,
    params: list[Any] | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Run an arbitrary SELECT against the index DB.

    Schema:
      events(global_id, queue_id, queue, name, event_type, parent_marker_id, depth, counters, raw)
      markers(id, name, start_global_id, end_global_id, parent_id, depth, queue_id, queue)
      queues(queue_id, name, event_count)

    ``counters`` and ``raw`` are JSON blobs; use ``json_extract(counters, '$.foo')``.
    """
    _, wpix = _resolve_wpix(capture)
    rows = index.raw_sql(wpix, sql, params, limit=limit)
    return {"count": len(rows), "rows": rows}


# ---------------------------------------------------------------------------
# state-at-event (C++-export-backed)
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_parse_cpp_export(
    capture: str,
    export_dir: str,
) -> dict[str, Any]:
    """Attach an already-exported C++ project to a capture session and parse it
    (so subsequent ``pix_state_at_event`` calls don't re-export)."""
    sess, wpix = _resolve_wpix(capture)
    if sess is None:
        sess = get_sessions().open_capture(wpix)
    export = cpp_export.parse_export(Path(export_dir).expanduser())
    sess.cpp_export_dir = Path(export_dir).resolve()
    sess.cpp_export_parsed = export
    sess.cpp_export_fingerprint = cpp_export.fingerprint_dir(Path(export_dir))
    return {
        "handle": sess.handle,
        "export_dir": str(sess.cpp_export_dir),
        "call_count": export.call_count,
        "global_ids_seen_sample": export.event_ids_seen[:50],
        "global_ids_count": len(export.event_ids_seen),
        "files_parsed": [str(p) for p in export.files_parsed],
    }


def _bindings_to_dict(s: cpp_export.StateAtEvent) -> dict[str, Any]:
    gfx = s.graphics
    comp = s.compute
    return {
        "target_global_id": s.target_global_id,
        "target_call_index": s.target_call_index,
        "applied_calls": s.applied_calls,
        "graphics": {
            "root_signature": gfx.root_signature,
            "pso": gfx.pso,
            "primitive_topology": gfx.primitive_topology,
            "root_params": gfx.root_params,
            "descriptor_heaps": gfx.descriptor_heaps,
            "rtvs": gfx.rtvs,
            "dsv": gfx.dsv,
            "index_buffer": gfx.index_buffer,
            "vertex_buffers": gfx.vertex_buffers,
            "viewports": gfx.viewports,
            "scissors": gfx.scissors,
            "last_event_index_applied": gfx.last_event_index_applied,
            "last_event_global_id": gfx.last_event_global_id,
        },
        "compute": {
            "root_signature": comp.root_signature,
            "pso": comp.pso,
            "root_params": comp.root_params,
            "descriptor_heaps": comp.descriptor_heaps,
            "last_event_index_applied": comp.last_event_index_applied,
            "last_event_global_id": comp.last_event_global_id,
        },
    }


@mcp.tool()
async def pix_state_at_event(
    capture: str,
    *,
    global_id: int | None = None,
    call_index: int | None = None,
    inclusive: bool = True,
) -> dict[str, Any]:
    """Replay state up to a target event and return all bound state.

    Pick one of ``global_id`` (PIX Global ID) or ``call_index`` (index in the
    parsed C++ export). The capture must have a C++ export attached (via
    ``pix_export_to_cpp`` with ``parse_after=True``, or ``pix_parse_cpp_export``).
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError(
            "pix_state_at_event needs an open capture session — call "
            "pix_open_capture and pix_export_to_cpp / pix_parse_cpp_export first."
        )
    export = sess.ensure_cpp_export()
    state = cpp_export.state_at_event(
        export, global_id=global_id, call_index=call_index, inclusive=inclusive
    )
    return _bindings_to_dict(state)


@mcp.tool()
async def pix_get_resource_at_root_param(
    capture: str,
    global_id: int,
    root_param_index: int,
    *,
    pipeline: str = "graphics",
) -> dict[str, Any]:
    """Answer 'what's bound at root parameter N of event G?' by replaying state.

    ``pipeline`` is ``"graphics"`` or ``"compute"`` — both root tables are kept
    in parallel and we return the one you ask for.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    export = sess.ensure_cpp_export()
    state = cpp_export.state_at_event(export, global_id=global_id, inclusive=True)
    table = state.graphics.root_params if pipeline == "graphics" else state.compute.root_params
    binding = table.get(root_param_index)
    return {
        "global_id": global_id,
        "root_param_index": root_param_index,
        "pipeline": pipeline,
        "binding": binding,
        "graphics_pso": state.graphics.pso,
        "compute_pso": state.compute.pso,
        "graphics_root_signature": state.graphics.root_signature,
        "compute_root_signature": state.compute.root_signature,
        "descriptor_heaps": (
            state.graphics.descriptor_heaps if pipeline == "graphics" else state.compute.descriptor_heaps
        ),
    }


@mcp.tool()
async def pix_find_cpp_calls(
    capture: str,
    *,
    method: str | None = None,
    method_like: str | None = None,
    args_substring: str | None = None,
    global_id: int | None = None,
    between: list[int] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Search the parsed C++ export for specific D3D12 calls.

    Useful when you want to inspect a call's *exact* argument string (e.g.
    "find every SetGraphicsRootDescriptorTable inside event range 2000..2400").
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    export = sess.ensure_cpp_export()
    between_tuple: tuple[int, int] | None = None
    if between:
        if len(between) != 2:
            raise ValueError("'between' must be a 2-element list.")
        between_tuple = (int(between[0]), int(between[1]))
    rows = cpp_export.find_calls(
        export,
        method=method,
        method_like=method_like,
        args_substring=args_substring,
        global_id=global_id,
        between=between_tuple,
        limit=limit,
    )
    return {
        "count": len(rows),
        "calls": [
            {
                "call_index": r.call_index,
                "global_id": r.global_id,
                "method": r.method,
                "receiver": r.receiver,
                "args": r.args,
                "source_file": r.source_file,
                "source_line": r.source_line,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# convenience: high-level capture summary
# ---------------------------------------------------------------------------


@mcp.tool()
async def pix_capture_summary(capture: str, *, quick: bool = True) -> dict[str, Any]:
    """High-level summary of a capture.

    If the index already exists, it's used. Otherwise (when ``quick=False``)
    this will build a minimal index first.
    """
    sess, wpix = _resolve_wpix(capture)
    if sess is None:
        sess = get_sessions().open_capture(wpix)
    out: dict[str, Any] = {"wpix_path": str(wpix), "handle": sess.handle}

    csv = sess.csv_path
    if csv is None:
        cache = get_settings().ensure_cache_dir()
        candidate = cache / f"{wpix.stem.replace(' ', '_')}.event_list.csv"
        if candidate.is_file():
            csv = candidate
    status = index.index_status(wpix, csv) if csv else {"exists": False}
    out["index_status"] = status

    if not status.get("up_to_date"):
        if quick:
            out["note"] = "no fresh index — call pix_index_events for full summary"
            return out
        await pix_index_events(sess.handle)  # build with no counters

    out["event_type_counts"] = index.event_type_counts(wpix)
    out["queues"] = index.list_queues(wpix)
    out["marker_count"] = len(index.list_markers(wpix, limit=1_000_000))
    return out


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def serve(transport: str = "stdio") -> None:
    """Run the MCP server on the given transport."""
    mcp.run(transport=transport)
