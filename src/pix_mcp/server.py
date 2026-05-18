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

from . import cpp_export, csv_parser, index, resources_bin, shaders
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
        "found_global_id": s.found_global_id,
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

    For root descriptors backed by a ``GetGpuva(resource_id, offset)`` literal
    (the common PIX 2603.x form), ``binding`` includes structured
    ``resource_id`` and ``offset`` keys — feed those straight into
    ``pix_get_resource_bytes``. Returns ``found_global_id=False`` if the
    requested ``global_id`` doesn't appear in the export (the binding then
    reflects end-of-frame state, not the state at your event).
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
        "found_global_id": state.found_global_id,
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
# raw resource bytes (resources.bin)
# ---------------------------------------------------------------------------


def _hex_preview(data: bytes, *, limit: int = 256) -> str:
    sl = data[:limit]
    return " ".join(f"{b:02x}" for b in sl)


def _decode_floats(data: bytes, count: int) -> list[float] | None:
    import struct
    if count <= 0:
        return None
    need = count * 4
    if len(data) < need:
        return None
    return list(struct.unpack(f"<{count}f", data[:need]))


def _decode_32bit_constants(binding: dict[str, Any]) -> dict[str, Any]:
    """Interpret a root 32BIT_CONSTANTS binding's captured slot values.

    The state replay stored individual ``SetGraphics/ComputeRoot32BitConstant``
    calls as ``binding['values'][slot] = "<raw arg>"`` and a single
    ``SetGraphics/ComputeRoot32BitConstants`` call as ``binding['raw']`` with
    the whole argument string. We decode the per-slot values to int / float
    where possible so callers don't have to parse the string forms themselves.
    """
    import struct
    out: dict[str, Any] = {}
    values = binding.get("values")
    if isinstance(values, dict):
        slots: dict[int, dict[str, Any]] = {}
        for slot, raw in values.items():
            entry: dict[str, Any] = {"raw": raw}
            if isinstance(raw, str):
                token = raw.strip().rstrip("uUlL")
                try:
                    as_u32 = int(token, 0) & 0xFFFFFFFF
                    entry["uint"] = as_u32
                    entry["int"] = struct.unpack("<i", struct.pack("<I", as_u32))[0]
                    entry["float"] = struct.unpack("<f", struct.pack("<I", as_u32))[0]
                except (ValueError, struct.error):
                    pass
            try:
                slots[int(slot)] = entry
            except (TypeError, ValueError):
                slots[slot] = entry  # fall back to raw key
        out["slots"] = slots
    if binding.get("kind") == "32bit_constants_block":
        out["block_raw"] = binding.get("raw")
    return out


@mcp.tool()
async def pix_get_resource_bytes(
    capture: str,
    resource_id: int,
    *,
    offset: int = 0,
    length: int | None = 256,
    chunk_index: int = 0,
    output_file: str | None = None,
    preview_floats: int = 0,
) -> dict[str, Any]:
    """Read raw bytes from a resource's initial data in the C++ export.

    PIX's ``export-to-cpp`` writes initial resource contents to a compressed
    ``resources.bin`` sidecar. This tool reconstructs the read sequence from
    the generated ``CreateAndInitResource_<id>()`` C++ definitions, locates the
    chunk for ``resource_id``, decompresses it (XPRESS via Windows Cabinet
    API), and returns the requested slice as a hex preview — and optionally
    writes the full slice to ``output_file``.

    Use this when ``pix_save_resource`` can't help you (it only emits visual
    PNG/DDS); for raw cbuffer / vertex / index data, this is the path.

    Args:
        resource_id: PIX ApiObjectId of the resource (e.g. ``2362``).
        offset:      byte offset into the decompressed resource.
        length:      bytes to return; ``None`` = "everything from offset".
                     Defaults to 256 to keep the response small.
        chunk_index: 0-based index when a resource initializer does multiple
                     Reads (e.g. a multi-subresource texture).
        output_file: optional path to write the full slice to disk.
        preview_floats: if >0, also decode the first N float32s and include
                     them in the response (useful for cbuffer inspection).
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError(
            "call pix_open_capture and pix_export_to_cpp first."
        )
    rb = sess.ensure_resources_bin()
    chunks = rb.chunks(resource_id)
    if not chunks:
        # Tell the caller exactly what IS known, so they can sanity-check the
        # resource_id (PIX numbering doesn't always match RenderDoc/Nsight).
        sample = rb.list_resources()[:20]
        raise RuntimeError(
            f"resource {resource_id} not found in resources.bin map "
            f"(tracked: {len(rb.chunks_by_resource)} resources). "
            f"Sample tracked IDs: {sample}"
        )
    data = rb.read_resource_bytes(
        resource_id, offset=offset, length=length, chunk_index=chunk_index
    )
    info: dict[str, Any] = {
        "resource_id": resource_id,
        "chunk_index": chunk_index,
        "chunk_count": len(chunks),
        "offset": offset,
        "length": len(data),
        "hex_preview": _hex_preview(data, limit=length or 256),
        "chunks": [
            {
                "file_offset": c.file_offset,
                "compressed_size": c.compressed_size,
                "source_function": c.source_function,
                "source_file": c.source_file,
            }
            for c in chunks
        ],
    }
    if preview_floats > 0:
        floats = _decode_floats(data, preview_floats)
        if floats is not None:
            info["preview_floats"] = floats
    if output_file:
        out = Path(output_file).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        info["output_file"] = str(out)
        info["output_bytes_written"] = len(data)
    return info


@mcp.tool()
async def pix_list_tracked_resources(capture: str) -> dict[str, Any]:
    """List every resource ID that ``pix_get_resource_bytes`` can read,
    along with each chunk's compressed size and file offset.

    Useful as a first sanity-check after ``pix_export_to_cpp`` — confirms
    the resources.bin map built cleanly before you try to dump bytes.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    rb = sess.ensure_resources_bin()
    rows: list[dict[str, Any]] = []
    for rid in rb.list_resources():
        chunks = rb.chunks(rid)
        rows.append({
            "resource_id": rid,
            "chunk_count": len(chunks),
            "total_compressed_bytes": sum(c.compressed_size for c in chunks),
            "first_source_function": chunks[0].source_function if chunks else None,
        })
    return {
        "count": len(rows),
        "resources": rows,
        "resources_bin_path": str(rb.bin_path),
        "resources_bin_size": rb.bin_path.stat().st_size,
        "bytes_walked_in_static_replay": rb.total_reads_walked,
        "unresolved_callees_sample": rb.unresolved_callees[:20],
    }


@mcp.tool()
async def pix_dump_cbuffer_at_root_param(
    capture: str,
    global_id: int,
    root_param_index: int,
    *,
    pipeline: str = "graphics",
    length: int = 256,
    preview_floats: int = 64,
    output_file: str | None = None,
) -> dict[str, Any]:
    """End-to-end: 'what bytes are in the cbuffer at root param N of event G?'

    Combines ``pix_state_at_event``, the parsed ``GetGpuva(resource_id,
    offset)`` extraction, and ``pix_get_resource_bytes`` into a single call
    that replaces the multi-click PIX-UI workflow.

    Fails if the root param isn't a root CBV/SRV/UAV at that event — root
    descriptor tables are indirect (a descriptor heap slot referring to a
    resource), and this tool doesn't yet chase descriptor heap state.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    export = sess.ensure_cpp_export()
    state = cpp_export.state_at_event(export, global_id=global_id, inclusive=True)
    if not state.found_global_id:
        sample = sorted({e.global_id for e in export.events[:5000] if e.global_id is not None})[:10]
        return {
            "global_id": global_id,
            "root_param_index": root_param_index,
            "pipeline": pipeline,
            "found_global_id": False,
            "error": (
                f"global_id {global_id} not found in the parsed C++ export. "
                "PIX assigns Global IDs across all queues including "
                "barriers/markers, but the C++ export only contains "
                "command-list calls. Try pix_get_event to confirm the gid "
                "exists, or pix_find_cpp_calls to locate a nearby call."
            ),
            "sample_known_gids": sample,
        }
    table = (
        state.graphics.root_params
        if pipeline == "graphics"
        else state.compute.root_params
    )
    binding = table.get(root_param_index)
    if binding is None:
        return {
            "global_id": global_id,
            "root_param_index": root_param_index,
            "pipeline": pipeline,
            "found_global_id": True,
            "error": "no binding at that root param at that event",
            "bound_root_params": sorted(table.keys()),
        }
    kind = binding.get("kind")
    # 32-bit root constants: the values were already captured at bind time —
    # decode them in-place instead of erroring.
    if kind in ("32bit_constants", "32bit_constants_block"):
        return {
            "global_id": global_id,
            "root_param_index": root_param_index,
            "pipeline": pipeline,
            "found_global_id": True,
            "binding": binding,
            "constants": _decode_32bit_constants(binding),
            "note": (
                "this is a root 32-bit constants slot — values come from the "
                "Set*Root32BitConstant{,s} calls captured directly, not from "
                "resources.bin."
            ),
        }
    if kind not in ("cbv", "srv", "uav"):
        return {
            "global_id": global_id,
            "root_param_index": root_param_index,
            "pipeline": pipeline,
            "found_global_id": True,
            "binding": binding,
            "error": (
                f"binding kind {kind!r} is not a root descriptor "
                "— descriptor tables resolve via descriptor heap inspection "
                "(not yet supported). Use pix_get_root_signature_layout to "
                "see the table layout."
            ),
        }
    resource_id = binding.get("resource_id")
    offset = binding.get("offset")
    if resource_id is None or offset is None:
        return {
            "global_id": global_id,
            "root_param_index": root_param_index,
            "binding": binding,
            "error": (
                "binding GPU VA isn't a GetGpuva(rid, off) literal — can't "
                "resolve resource_id + offset statically."
            ),
        }
    rb = sess.ensure_resources_bin()
    if resource_id not in rb.chunks_by_resource:
        return {
            "global_id": global_id,
            "root_param_index": root_param_index,
            "binding": binding,
            "error": (
                f"resource {resource_id} not tracked in resources.bin map "
                "(no CreateAndInitResource_<id> found in the export)."
            ),
        }
    data = rb.read_resource_bytes(resource_id, offset=offset, length=length)
    out: dict[str, Any] = {
        "global_id": global_id,
        "root_param_index": root_param_index,
        "pipeline": pipeline,
        "binding": binding,
        "resource_id": resource_id,
        "offset": offset,
        "length": len(data),
        "hex_preview": _hex_preview(data, limit=length),
    }
    if preview_floats > 0:
        floats = _decode_floats(data, preview_floats)
        if floats is not None:
            out["preview_floats"] = floats
    if output_file:
        op = Path(output_file).expanduser()
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_bytes(data)
        out["output_file"] = str(op)
    return out


@mcp.tool()
async def pix_get_root_signature_layout(
    capture: str,
    root_sig_obj_id: int,
) -> dict[str, Any]:
    """Parse the inline ``D3D12_ROOT_PARAMETER1`` definition for a root
    signature and return its per-root-param layout.

    PIX's export-to-cpp doesn't store the serialized root sig blob — it
    re-emits the desc inline as a sequence of ``rootParameters[K].* = ...``
    assignments wrapped in a block ending with ``CreateAndTrackRootSignature
    (<id>, ...)``. We locate that block by ApiObjectId and unpack:

      * ``kind``: ``DESCRIPTOR_TABLE`` / ``CBV`` / ``SRV`` / ``UAV`` / ``32BIT_CONSTANTS``
      * ``visibility``: ``ALL`` / ``VERTEX`` / ``PIXEL`` / ``HULL`` / ``DOMAIN`` / ``GEOMETRY`` / ``AMPLIFICATION`` / ``MESH``
      * For root descriptors: ``shader_register``, ``register_space``, ``flags``
      * For 32-bit constants: ``shader_register``, ``register_space``, ``num_32bit_values``
      * For descriptor tables: ordered ``descriptor_ranges`` list with each
        range's type / num / base register / space / flags / offset

    Pair this with ``pix_get_resource_at_root_param`` (resource bound to a
    root descriptor) or with descriptor-heap inspection (root descriptor
    table) to fully understand what each slot's shader sees.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    export = sess.ensure_cpp_export()
    layout = cpp_export.parse_root_signature_layout(export, root_sig_obj_id)
    if layout is None:
        return {
            "found": False,
            "root_signature_id": root_sig_obj_id,
            "reason": (
                f"no CreateAndTrackRootSignature({root_sig_obj_id}, ...) call "
                "found in the parsed C++ export. Check the ID with "
                "pix_find_cpp_calls(method='CreateAndTrackRootSignature')."
            ),
        }
    return {
        "found": True,
        "root_signature_id": layout.root_signature_id,
        "source_file": layout.source_file,
        "source_line": layout.source_line,
        "flags": layout.flags,
        # Truncated copy of the inline C++ block PIX generated, so callers can
        # verify the structured output against the source text without
        # re-grepping the export.
        "raw_block_excerpt": layout.raw_block_excerpt,
        "params": [
            {
                "index": p.index,
                "kind": p.kind,
                "visibility": p.visibility,
                "shader_register": p.shader_register,
                "register_space": p.register_space,
                "num_32bit_values": p.num_32bit_values,
                "flags": p.flags,
                "descriptor_ranges": [
                    {
                        "range_type": r.range_type,
                        "num_descriptors": r.num_descriptors,
                        "base_shader_register": r.base_shader_register,
                        "register_space": r.register_space,
                        "flags": r.flags,
                        "offset_in_descriptors": r.offset_in_descriptors,
                    }
                    for r in p.descriptor_ranges
                ],
            }
            for p in layout.params
        ],
    }


# ---------------------------------------------------------------------------
# shader bytecode + disassembly
# ---------------------------------------------------------------------------


def _stage_list(layout: cpp_export.PsoStageLayout) -> list[dict[str, Any]]:
    return [
        {
            "stage": s.stage,
            "offset": s.offset,
            "length": s.length,
            "source_line": s.source_line,
        }
        for s in layout.stages
    ]


@mcp.tool()
async def pix_list_psos(
    capture: str,
    *,
    has_stage: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """List every PSO in the export with its shader-stage layout.

    Each PSO has one combined bytecode blob in ``resources.bin`` that the
    generated C++ slices into per-stage views (``pssDesc.VS = { ..., LEN }``
    etc.). We return the per-stage offsets/lengths so callers know what they
    can dump.

    Args:
        has_stage: only return PSOs that contain this stage (e.g. ``"CS"`` to
                   list compute-only PSOs).
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    export = sess.ensure_cpp_export()
    layouts = cpp_export.list_all_psos(export)
    if has_stage:
        want = has_stage.upper()
        layouts = [L for L in layouts if any(s.stage == want for s in L.stages)]
    rows: list[dict[str, Any]] = []
    for L in layouts[:limit]:
        rows.append({
            "pso_id": L.pso_id,
            "root_signature_id": L.root_signature_id,
            "compressed_blob_size": L.compressed_blob_size,
            "stages": _stage_list(L),
            "source_file": L.source_file,
            "source_line": L.source_line,
        })
    return {
        "count": len(rows),
        "total_psos": len(cpp_export.list_all_psos(export)) if has_stage else len(layouts),
        "psos": rows,
    }


@mcp.tool()
async def pix_get_pso_stage_layout(
    capture: str,
    pso_id: int,
) -> dict[str, Any]:
    """Return the per-stage byte offsets/lengths inside a single PSO's blob.

    This is the canonical "what shaders does this PSO have" lookup. Pair the
    output with ``pix_dump_shader_bytecode`` or ``pix_disassemble_shader``
    to extract / inspect individual stages.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    export = sess.ensure_cpp_export()
    layout = cpp_export.parse_pso_shader_stages(export, pso_id)
    if layout is None:
        return {
            "found": False,
            "pso_id": pso_id,
            "reason": (
                f"no CreatePipelineState_{pso_id}() found in the export. Check "
                "with pix_list_psos() to see what PSO ids are available."
            ),
        }
    return {
        "found": True,
        "pso_id": layout.pso_id,
        "root_signature_id": layout.root_signature_id,
        "compressed_blob_size": layout.compressed_blob_size,
        "stages": _stage_list(layout),
        "source_file": layout.source_file,
        "source_line": layout.source_line,
    }


def _load_pso_blob(
    sess: Any, pso_id: int
) -> tuple[cpp_export.PsoStageLayout, bytes]:
    """Decompress a PSO's bytecode blob from resources.bin.

    Returns (layout, decompressed_blob_bytes). Raises if the PSO can't be
    located or its chunk isn't tracked.
    """
    export = sess.ensure_cpp_export()
    layout = cpp_export.parse_pso_shader_stages(export, pso_id)
    if layout is None:
        raise RuntimeError(
            f"no CreatePipelineState_{pso_id}() found in the export."
        )
    rb = sess.ensure_resources_bin()
    pso_chunks = rb.chunks_by_pso.get(pso_id)
    if not pso_chunks:
        sample = sorted(rb.chunks_by_pso.keys())[:10]
        raise RuntimeError(
            f"PSO {pso_id} not tracked in resources.bin walker. "
            f"Tracked PSO IDs sample: {sample}"
        )
    blob = rb.read_chunk(pso_chunks[0])
    return layout, blob


@mcp.tool()
async def pix_dump_shader_bytecode(
    capture: str,
    pso_id: int,
    stage: str,
    *,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Extract one shader stage's bytecode from a PSO and (optionally) save it
    as a ``.cso``.

    The returned bytecode is the raw DXBC/DXIL container suitable for
    ``dxc.exe -dumpbin`` or any DXBC tool. If ``output_file`` is provided the
    full bytecode is written there; otherwise only the header bytes are
    surfaced (full bytecode would blow up the MCP response).

    Args:
        pso_id: PSO ApiObjectId (e.g. ``852``).
        stage:  ``"VS"`` | ``"PS"`` | ``"CS"`` | ``"HS"`` | ``"DS"`` | ``"GS"``
                | ``"AS"`` | ``"MS"``.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    layout, blob = _load_pso_blob(sess, pso_id)
    want = stage.upper()
    match = next((s for s in layout.stages if s.stage == want), None)
    if match is None:
        return {
            "pso_id": pso_id,
            "requested_stage": stage,
            "error": f"PSO {pso_id} has no {want} stage. Available: "
                     f"{[s.stage for s in layout.stages]}",
        }
    bytecode = shaders.extract_shader_bytes(blob, match.offset, match.length)
    fmt = shaders.detect_shader_format(bytecode)
    info: dict[str, Any] = {
        "pso_id": pso_id,
        "stage": want,
        "byte_length": len(bytecode),
        "container_format": fmt,
        "header_hex": _hex_preview(bytecode[:32], limit=32),
        "blob_offset_within_pso": match.offset,
        "source_line": match.source_line,
    }
    if output_file:
        op = Path(output_file).expanduser()
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_bytes(bytecode)
        info["output_file"] = str(op)
        info["bytes_written"] = len(bytecode)
    return info


@mcp.tool()
async def pix_disassemble_shader(
    capture: str,
    pso_id: int,
    stage: str,
    *,
    output_file: str | None = None,
    head_lines: int = 400,
    prefer: str = "auto",
) -> dict[str, Any]:
    """Disassemble one shader stage of a PSO and return the asm/IR text.

    Uses ``dxc.exe -dumpbin`` (default) which handles both DXIL (SM6+) and
    legacy DXBC containers. To force the FXC disassembler instead, pass
    ``prefer="fxc"`` (legacy DXBC only).

    Args:
        head_lines: cap the response at this many lines of disassembly to
                    keep tool output manageable. Pass ``0`` for the full text.
                    Use ``output_file`` if you need the whole disassembly on
                    disk regardless.
        prefer:     ``"auto"`` | ``"dxc"`` | ``"fxc"``.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    layout, blob = _load_pso_blob(sess, pso_id)
    want = stage.upper()
    match = next((s for s in layout.stages if s.stage == want), None)
    if match is None:
        return {
            "pso_id": pso_id,
            "requested_stage": stage,
            "error": f"PSO {pso_id} has no {want} stage. Available: "
                     f"{[s.stage for s in layout.stages]}",
        }
    bytecode = shaders.extract_shader_bytes(blob, match.offset, match.length)
    result = shaders.disassemble_shader(bytecode, prefer=prefer)
    text = result.text or ""
    lines = text.splitlines()
    truncated = False
    if head_lines and head_lines > 0 and len(lines) > head_lines:
        text = "\n".join(lines[:head_lines])
        truncated = True
    info: dict[str, Any] = {
        "pso_id": pso_id,
        "stage": want,
        "container_format": result.container_format,
        "disassembler": result.disassembler,
        "disassembler_tool": result.disassembler_tool,
        "returncode": result.returncode,
        "stderr_tail": result.stderr_tail,
        "byte_length": len(bytecode),
        "total_lines": len(lines),
        "truncated": truncated,
        "disassembly": text,
    }
    if output_file:
        op = Path(output_file).expanduser()
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(result.text or "", encoding="utf-8")
        info["output_file"] = str(op)
        info["bytes_written"] = op.stat().st_size if op.is_file() else 0
    return info


@mcp.tool()
async def pix_analyze_shader_cbuffer_reads(
    capture: str,
    pso_id: int,
    stage: str,
    *,
    prefer: str = "auto",
    cbuffer_slot: int | None = None,
) -> dict[str, Any]:
    """Disassemble a shader stage and surface which cbuffer rows it reads.

    Each row in a D3D constant buffer is 16 bytes (4 floats). The returned
    ``reads`` list answers "which 16-byte slices of cbN does the shader
    actually touch", which is the question that motivated this entire
    shader-extraction path: if compute pass A reads cb0 rows {5, 9, 14} and
    compute pass B reads cb0 rows {5, 9, 14, 22}, then row 22 is where the
    behavior diverges and the cbuffer-bytes inspection should focus there.

    Args:
        cbuffer_slot: optional filter — only return reads from this b<N> slot.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    layout, blob = _load_pso_blob(sess, pso_id)
    want = stage.upper()
    match = next((s for s in layout.stages if s.stage == want), None)
    if match is None:
        return {
            "pso_id": pso_id,
            "requested_stage": stage,
            "error": f"PSO {pso_id} has no {want} stage. Available: "
                     f"{[s.stage for s in layout.stages]}",
        }
    bytecode = shaders.extract_shader_bytes(blob, match.offset, match.length)
    disasm = shaders.disassemble_shader(bytecode, prefer=prefer)
    analysis = shaders.analyze_cbuffer_reads(disasm)
    reads = analysis.reads
    if cbuffer_slot is not None:
        reads = [r for r in reads if r.cbuffer_slot == cbuffer_slot]
    return {
        "pso_id": pso_id,
        "stage": want,
        "container_format": analysis.container_format,
        "disassembler": disasm.disassembler_tool,
        "disassembler_returncode": disasm.returncode,
        "declared_cbuffers": analysis.declared_cbuffers,
        "total_load_sites": analysis.total_load_sites,
        "reads": [
            {
                "cbuffer_slot": r.cbuffer_slot,
                "register_space": r.register_space,
                "row": r.row,
                "byte_offset": r.byte_offset,
                "occurrences": r.occurrences,
                "kind": r.kind,
            }
            for r in reads
        ],
        "warnings": analysis.warnings,
    }


@mcp.tool()
async def pix_analyze_shader_at_event(
    capture: str,
    global_id: int,
    stage: str,
    *,
    pipeline: str = "auto",
    cbuffer_slot: int | None = None,
    prefer: str = "auto",
) -> dict[str, Any]:
    """End-to-end: 'which cbuffer rows does the shader bound at event G read?'

    Replays C++ state up to ``global_id``, extracts the PSO ApiObjectId from
    the active SetPipelineState binding, locates the requested stage's
    bytecode in resources.bin, disassembles it, and returns the cbuffer-read
    analysis. This is the one-call answer to "what does the compute shader
    sample from cb0 in view 1's fog dispatch?"

    Args:
        pipeline: ``"auto"`` (try compute then graphics) | ``"graphics"``
                  | ``"compute"``. Compute is the right choice for fog
                  volume dispatches.
    """
    sess, _wpix = _resolve_wpix(capture)
    if sess is None:
        raise RuntimeError("call pix_open_capture and pix_export_to_cpp first.")
    export = sess.ensure_cpp_export()
    state = cpp_export.state_at_event(export, global_id=global_id, inclusive=True)
    pso_raw: str | None
    if pipeline == "compute":
        pso_raw = state.compute.pso
    elif pipeline == "graphics":
        pso_raw = state.graphics.pso
    else:
        pso_raw = state.compute.pso or state.graphics.pso
    pso_id = shaders.pso_id_from_state_value(pso_raw)
    if pso_id is None:
        return {
            "global_id": global_id,
            "error": (
                f"no PSO bound at event {global_id} (pipeline={pipeline}). "
                f"compute.pso={state.compute.pso!r}, graphics.pso={state.graphics.pso!r}"
            ),
        }
    inner = await pix_analyze_shader_cbuffer_reads(
        capture, pso_id, stage,
        prefer=prefer, cbuffer_slot=cbuffer_slot,
    )
    inner["resolved_pso_id"] = pso_id
    inner["resolved_pso_raw"] = pso_raw
    inner["target_global_id"] = global_id
    inner["applied_calls"] = state.applied_calls
    return inner


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
