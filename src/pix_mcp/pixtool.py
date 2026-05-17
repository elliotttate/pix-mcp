"""Subprocess wrapper for pixtool.exe.

pixtool is a batch tool: a single invocation can chain multiple commands
(e.g. ``pixtool open-capture foo.wpix save-event-list out.csv``). This module
builds those command chains as a list of (command, args) pairs and runs the
process synchronously or in the background.
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import Settings, get_settings


CommandArg = str | int | float | Path


@dataclass(frozen=True)
class PixCommand:
    """A single pixtool sub-command, e.g. ``open-capture foo.wpix --remote=host``."""

    name: str
    positional: tuple[str, ...] = ()
    options: tuple[tuple[str, str | None], ...] = ()  # (--flag, value or None for bare flag)

    def render(self) -> list[str]:
        out: list[str] = [self.name, *self.positional]
        for flag, value in self.options:
            if value is None:
                out.append(flag)
            else:
                out.append(f"{flag}={value}")
        return out


def cmd(
    name: str,
    *positional: CommandArg,
    **options: CommandArg | bool | None,
) -> PixCommand:
    """Build a PixCommand. Bool True → bare flag; None → omitted; everything else stringified."""
    pos = tuple(str(p) for p in positional)
    opts: list[tuple[str, str | None]] = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            opts.append((flag, None))
        else:
            opts.append((flag, str(value)))
    return PixCommand(name=name, positional=pos, options=tuple(opts))


@dataclass
class PixtoolResult:
    returncode: int
    stdout: str
    stderr: str
    cmdline: list[str]
    duration_sec: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> None:
        if not self.ok:
            raise PixtoolError(
                f"pixtool exited with code {self.returncode}\n"
                f"cmdline: {' '.join(shlex.quote(p) for p in self.cmdline)}\n"
                f"stderr: {self.stderr.strip()}\n"
                f"stdout tail: {self.stdout[-2000:]}"
            )


class PixtoolError(RuntimeError):
    pass


def build_cmdline(
    commands: Iterable[PixCommand],
    *,
    output_level: str | None = None,
    log_level: str | None = None,
    log_file: Path | str | None = None,
    settings: Settings | None = None,
) -> list[str]:
    """Assemble the full ``[pixtool.exe, ...args]`` list."""
    s = settings or get_settings()
    pixtool = s.require_pixtool()
    parts: list[str] = [str(pixtool)]
    if output_level:
        parts.append(f"--output={output_level}")
    if log_level:
        parts.append(f"--log={log_level}")
    if log_file:
        parts.append(f"--log-file={log_file}")
    for c in commands:
        parts.extend(c.render())
    return parts


def run_pixtool(
    commands: Iterable[PixCommand],
    *,
    output_level: str | None = "trace",
    log_level: str | None = "verbose",
    log_file: Path | str | None = None,
    timeout: float | None = None,
    cwd: Path | str | None = None,
    settings: Settings | None = None,
    check: bool = False,
) -> PixtoolResult:
    """Run pixtool synchronously to completion."""
    s = settings or get_settings()
    cmdline = build_cmdline(
        commands,
        output_level=output_level,
        log_level=log_level,
        log_file=log_file,
        settings=s,
    )
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmdline,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            timeout=timeout if timeout is not None else s.pixtool_timeout_sec,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PixtoolError(
            f"pixtool timed out after {exc.timeout}s\n"
            f"cmdline: {' '.join(shlex.quote(p) for p in cmdline)}"
        ) from exc
    dt = time.monotonic() - t0
    result = PixtoolResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        cmdline=cmdline,
        duration_sec=dt,
    )
    if check:
        result.raise_for_status()
    return result


async def run_pixtool_async(
    commands: Iterable[PixCommand],
    *,
    output_level: str | None = "trace",
    log_level: str | None = "verbose",
    log_file: Path | str | None = None,
    timeout: float | None = None,
    cwd: Path | str | None = None,
    settings: Settings | None = None,
    check: bool = False,
) -> PixtoolResult:
    """Async variant — never blocks the asyncio loop."""
    return await asyncio.to_thread(
        run_pixtool,
        commands,
        output_level=output_level,
        log_level=log_level,
        log_file=log_file,
        timeout=timeout,
        cwd=cwd,
        settings=settings,
        check=check,
    )


@dataclass
class BackgroundProcess:
    """A long-running pixtool subprocess (e.g. for ``launch <exe>``).

    Captures stdout/stderr into in-memory ring buffers so the MCP can surface
    recent output back to the caller without blocking.
    """

    handle: str
    proc: subprocess.Popen
    cmdline: list[str]
    started_at: float
    stdout_buf: list[str] = field(default_factory=list)
    stderr_buf: list[str] = field(default_factory=list)
    _stdout_thread: threading.Thread | None = None
    _stderr_thread: threading.Thread | None = None
    max_buffer_lines: int = 2000

    def is_running(self) -> bool:
        return self.proc.poll() is None

    def returncode(self) -> int | None:
        return self.proc.poll()

    def recent_stdout(self, n: int = 200) -> str:
        return "".join(self.stdout_buf[-n:])

    def recent_stderr(self, n: int = 200) -> str:
        return "".join(self.stderr_buf[-n:])

    def terminate(self, timeout: float = 5.0) -> int:
        if self.is_running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=timeout)
        return self.proc.returncode or 0


def start_background_pixtool(
    handle: str,
    commands: Iterable[PixCommand],
    *,
    output_level: str | None = "trace",
    log_level: str | None = "verbose",
    log_file: Path | str | None = None,
    cwd: Path | str | None = None,
    settings: Settings | None = None,
) -> BackgroundProcess:
    """Spawn pixtool as a background process and stream its output into buffers."""
    cmdline = build_cmdline(
        commands,
        output_level=output_level,
        log_level=log_level,
        log_file=log_file,
        settings=settings,
    )
    proc = subprocess.Popen(
        cmdline,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        bufsize=1,
    )
    bg = BackgroundProcess(handle=handle, proc=proc, cmdline=cmdline, started_at=time.monotonic())

    def _pump(stream, buf: list[str]) -> None:
        try:
            for line in stream:
                buf.append(line)
                if len(buf) > bg.max_buffer_lines:
                    del buf[: len(buf) - bg.max_buffer_lines]
        except Exception:
            pass

    bg._stdout_thread = threading.Thread(
        target=_pump, args=(proc.stdout, bg.stdout_buf), daemon=True
    )
    bg._stderr_thread = threading.Thread(
        target=_pump, args=(proc.stderr, bg.stderr_buf), daemon=True
    )
    bg._stdout_thread.start()
    bg._stderr_thread.start()
    return bg
