"""Shader bytecode extraction + disassembly for pix-mcp.

PIX's ``export-to-cpp`` writes each PSO's combined shader-bytecode blob into
``resources.bin``. The generated ``CreatePipelineState_<id>()`` C++ function
calls ``g_resourceReader->Read(data, N)`` once, then slices the decompressed
result into per-stage views via ``pssDesc.<STAGE> = { &data[offset], LEN }``.

This module:

  1. Pulls one stage's bytecode out of a PSO blob (DXBC / DXIL).
  2. Disassembles it with ``dxc.exe -dumpbin`` (handles both legacy DXBC and
     modern DXIL containers — UE5 ships SM6+ which is DXIL).
  3. Parses the disassembly to surface which cbuffer offsets each shader
     actually reads. That's the question that motivated this whole module:
     "which fields of the per-view cbuffer does the fog compute sample?"

Disassembler selection: dxc.exe is preferred (it transparently handles DXBC
and DXIL). If unavailable, falls back to fxc.exe which only handles legacy
DXBC. Both ship with the Windows 10 SDK; the search heuristic picks the
newest Windows Kits build that has the requested binary.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# --- bytecode container detection ------------------------------------------


def detect_shader_format(data: bytes) -> str:
    """Return ``"DXBC"``, ``"DXIL"``, or ``"unknown"``.

    Both DXBC (FXC, SM5-) and DXIL (DXC, SM6+) shaders are wrapped in the
    DXBC container format — the first four bytes are always ``"DXBC"``. The
    difference is which chunk type the container holds; DXIL containers carry
    a ``"DXIL"`` chunk (or ``"DXBC"`` chunk with embedded bitcode) alongside
    metadata, while pure DXBC carries ``"SHEX"`` / ``"SHDR"`` chunks.

    This is a heuristic — we look for ``b"DXIL"`` anywhere in the first 4 KB
    of the container. dxc.exe handles both formats either way; this is mainly
    informational.
    """
    if len(data) < 4 or data[:4] != b"DXBC":
        return "unknown"
    head = data[: min(len(data), 4096)]
    if b"DXIL" in head:
        return "DXIL"
    return "DXBC"


# --- disassembler discovery ------------------------------------------------


def _windows_kits_dir() -> Path | None:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Windows Kits" / "10" / "bin",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def find_disassembler(name: str = "dxc.exe") -> Path | None:
    """Locate ``dxc.exe`` or ``fxc.exe``.

    Order:
      1. ``PATH`` (so users can override with newer DXC builds).
      2. Newest Windows 10 SDK ``bin\\<version>\\x64`` directory.
    """
    p = shutil.which(name)
    if p:
        return Path(p)
    root = _windows_kits_dir()
    if root is None:
        return None
    versions = sorted(
        (d for d in root.iterdir() if d.is_dir() and d.name[0].isdigit()),
        key=lambda d: d.name,
        reverse=True,
    )
    for v in versions:
        cand = v / "x64" / name
        if cand.is_file():
            return cand
    return None


# --- bytecode extraction ---------------------------------------------------


@dataclass
class ExtractedShader:
    """A single stage's bytecode pulled out of a PSO's decompressed blob."""

    pso_id: int
    stage: str             # "VS" | "PS" | "CS" | ...
    bytes_: bytes
    container_format: str  # "DXBC" | "DXIL" | "unknown"
    blob_offset: int       # offset within the PSO's decompressed blob
    output_file: Path | None = None  # set if caller asked us to write to disk


def extract_shader_bytes(
    decompressed_blob: bytes,
    stage_offset: int,
    stage_length: int,
) -> bytes:
    """Slice one stage's bytecode out of the PSO's decompressed blob.

    Trivial helper, but isolated so the offset/length math has exactly one
    home (and so callers don't accidentally read past the blob end).
    """
    if stage_offset < 0 or stage_offset + stage_length > len(decompressed_blob):
        raise ValueError(
            f"stage slice [{stage_offset}, {stage_offset + stage_length}) "
            f"out of range for blob of {len(decompressed_blob)} bytes"
        )
    return decompressed_blob[stage_offset : stage_offset + stage_length]


# --- disassembly -----------------------------------------------------------


@dataclass
class DisassemblyResult:
    disassembler: str           # path to the tool that produced this
    disassembler_tool: str      # "dxc" | "fxc"
    text: str                   # disassembly text (DXIL IR or DXBC asm)
    stderr_tail: str = ""
    returncode: int = 0
    container_format: str = "unknown"


def disassemble_shader(
    bytecode: bytes,
    *,
    prefer: str = "auto",
    timeout: float = 30.0,
) -> DisassemblyResult:
    """Run a disassembler on a shader bytecode buffer and return its text.

    ``prefer``:
      * ``"auto"`` (default) — dxc.exe (handles DXBC + DXIL).
      * ``"dxc"`` — force dxc.exe.
      * ``"fxc"`` — force fxc.exe (only works on legacy DXBC).

    The bytecode is written to a temp ``.cso`` file because both tools want a
    real on-disk file rather than stdin.
    """
    fmt = detect_shader_format(bytecode)

    tool_name = "dxc.exe"
    if prefer == "fxc":
        tool_name = "fxc.exe"
    elif prefer == "dxc":
        tool_name = "dxc.exe"
    # "auto": prefer dxc since UE5 ships SM6+.
    tool = find_disassembler(tool_name)
    if tool is None and prefer == "auto":
        tool = find_disassembler("fxc.exe")
        tool_name = "fxc.exe"
    if tool is None:
        raise RuntimeError(
            f"could not locate {tool_name} (or any disassembler). Install the "
            "Windows 10 SDK or put dxc.exe / fxc.exe on PATH."
        )

    with tempfile.NamedTemporaryFile(suffix=".cso", delete=False) as f:
        in_path = Path(f.name)
        f.write(bytecode)
    try:
        if tool_name.lower().startswith("dxc"):
            # `dxc.exe -dumpbin -Fc <output>` writes disasm to a file; without
            # -Fc it goes to stdout. Stdout keeps it simple.
            cmdline = [str(tool), "-dumpbin", str(in_path)]
        else:
            # fxc.exe: `fxc /dumpbin <input> /Fc <output>` is the standard
            # invocation. We use /nologo /dumpbin /Fc and capture the file.
            out_path = in_path.with_suffix(".asm")
            cmdline = [
                str(tool),
                "/nologo",
                "/dumpbin",
                str(in_path),
                "/Fc",
                str(out_path),
            ]
        proc = subprocess.run(
            cmdline,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if tool_name.lower().startswith("fxc"):
            try:
                disasm = Path(cmdline[-1]).read_text(encoding="utf-8", errors="replace")
            except OSError:
                disasm = proc.stdout
            finally:
                try:
                    Path(cmdline[-1]).unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            disasm = proc.stdout
        return DisassemblyResult(
            disassembler=str(tool),
            disassembler_tool="dxc" if tool_name.lower().startswith("dxc") else "fxc",
            text=disasm,
            stderr_tail=(proc.stderr or "")[-4000:],
            returncode=proc.returncode,
            container_format=fmt,
        )
    finally:
        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass


# --- cbuffer-read analysis -------------------------------------------------


# DXIL cbuffer access uses one of two handle-creation styles depending on
# the shader model the bytecode was built for:
#
#   * SM6.0–6.5: `dx.op.createHandle(opcode, i8 RESCLASS, i32 RANGE_ID, i32 BIND, i1 NU)`
#     RESCLASS 2 = CBV. We pull RANGE_ID + BIND to identify the slot.
#
#   * SM6.6+:    `dx.op.createHandleFromBinding(opcode,
#                    %dx.types.ResBind { i32 LO, i32 HI, i32 SPACE, i8 CLASS },
#                    i32 INDEX, i1 NU)`
#     CLASS 2 = CBV. We pull LO (== shader register) and SPACE.
#
# UE5 ships SM6.6+ output, so we MUST handle the Binding-style. Each handle
# is bound to an SSA name like `%4 = call ... createHandleFromBinding(...)`.
# `annotateHandle` may wrap the raw handle in a typed one (`%5 = annotateHandle(%4)`).
# `cbufferLoadLegacy.<type>(opcode, handle, row)` then references the
# eventually-annotated handle.
#
# Parsing strategy: walk lines top-to-bottom, build an SSA → binding map
# (propagating through annotateHandle), and resolve each cbufferLoad's
# handle argument against the map.

# Match `%NAME = ` SSA assignments to capture the result name.
_DXIL_SSA_ASSIGN_RE = re.compile(r"^\s*(%[A-Za-z0-9_.]+)\s*=\s*")

# createHandle (SM6.0–6.5)
_DXIL_CREATE_HANDLE_RE = re.compile(
    r"@dx\.op\.createHandle\s*\(\s*"
    r"i32\s+\d+\s*,\s*"
    r"i8\s+(?P<resclass>\d+)\s*,\s*"
    r"i32\s+(?P<range_id>\d+)\s*,\s*"
    r"i32\s+(?P<bind>\d+)"
)
# createHandleFromBinding (SM6.6+) — note the ResBind struct can be either
# a struct literal `{ i32 LO, i32 HI, i32 SPACE, i8 CLASS }` or shorthand
# `zeroinitializer` (which is all zeros: LO=0,HI=0,SPACE=0,CLASS=0).
_DXIL_CREATE_HANDLE_BIND_RE = re.compile(
    r"@dx\.op\.createHandleFromBinding\s*\(\s*"
    r"i32\s+\d+\s*,\s*"
    r"%[A-Za-z0-9_.]+\s+"
    r"(?:"
    r"\{\s*"
    r"i32\s+(?P<lo>\d+)\s*,\s*"
    r"i32\s+(?P<hi>\d+)\s*,\s*"
    r"i32\s+(?P<space>\d+)\s*,\s*"
    r"i8\s+(?P<resclass>\d+)\s*\}"
    r"|"
    r"(?P<zero>zeroinitializer)"
    r")"
)
# annotateHandle wraps a raw handle in a typed one. We just need to propagate
# the binding through it.
#   %5 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %4, ...)
_DXIL_ANNOTATE_HANDLE_RE = re.compile(
    r"@dx\.op\.annotateHandle\s*\(\s*"
    r"i32\s+\d+\s*,\s*"
    r"%[A-Za-z0-9_.]+\s+(?P<src>%[A-Za-z0-9_]+)"
)
# cbufferLoadLegacy.<type>(opcode, handle, row) — capture handle SSA + row.
_DXIL_CBUF_LOAD_RE = re.compile(
    r"call\s+%[A-Za-z0-9_.]*CBufRet\.[A-Za-z0-9_.]+\s*"
    r"@dx\.op\.cbufferLoad(?:Legacy)?\.[A-Za-z0-9_]+\s*\(\s*"
    r"i32\s+\d+\s*,\s*"
    r"%[A-Za-z0-9_.]+\s+(?P<handle>%[A-Za-z0-9_]+)\s*,\s*"
    r"i32\s+(?P<row>\d+)"
)
# DXBC asm cbuffer access: `mov r0.x, cb0[3].x` / `dp4 r0.x, cb0[5].xyzw, v0.xyzw`.
# We extract `cb<N>[<ROW>]` references — same row-as-16-byte semantics.
_DXBC_CB_REF_RE = re.compile(
    r"\bcb(?P<slot>\d+)\s*\[\s*(?P<row>\d+)\s*\]"
)
# `dcl_constantbuffer cb0[12], immediateIndexed` declares slot 0 with 12 rows.
_DXBC_DCL_CB_RE = re.compile(
    r"dcl_constantbuffer\s+cb(?P<slot>\d+)\s*\[\s*(?P<size>\d+)\s*\]"
)


@dataclass
class CbufferReadSite:
    cbuffer_slot: int | None       # b<slot> binding, if recoverable
    register_space: int | None     # only meaningful for DXIL; DXBC has no spaces
    row: int                       # 16-byte index into the cbuffer (multiply by 16 for byte offset)
    byte_offset: int               # row * 16
    occurrences: int               # how many times this row was read
    kind: str                      # "DXIL" | "DXBC"


@dataclass
class CbufferAnalysis:
    container_format: str
    declared_cbuffers: list[dict[str, int]] = field(default_factory=list)
    reads: list[CbufferReadSite] = field(default_factory=list)
    total_load_sites: int = 0
    # Free-form notes: what we couldn't recover (e.g. dynamic indexing,
    # handle SSA we didn't resolve).
    warnings: list[str] = field(default_factory=list)


def analyze_cbuffer_reads(disasm: DisassemblyResult) -> CbufferAnalysis:
    """Extract every cbuffer row that the shader actually reads.

    For DXIL we look for ``dx.op.cbufferLoad{,Legacy}.<type>`` calls and read
    the trailing ``i32 ROW`` argument. We try to attribute each call back to a
    binding by walking ``dx.op.createHandle`` calls in textual order — the IR
    pretty much always emits the handle right before its uses, but this is a
    heuristic, not a real SSA chase.

    For DXBC we look for ``cb<N>[<ROW>]`` operand references and group by N.

    The result is a list of ``(cbuffer_slot, row, byte_offset, occurrences)``
    tuples — enough to answer "which fields of cb0 does this shader read?"
    """
    text = disasm.text or ""
    fmt = disasm.container_format
    if fmt == "unknown" and disasm.disassembler_tool == "dxc":
        # dxc happily dumps both; treat as DXIL if we see IR-style syntax.
        fmt = "DXIL" if "dx.op" in text else "DXBC"

    out = CbufferAnalysis(container_format=fmt)

    if fmt == "DXIL":
        # SSA handle map: SSA-name (e.g. "%4") -> (slot, space) if it is a CBV.
        # createHandle / createHandleFromBinding define handles; annotateHandle
        # propagates them; cbufferLoad consumes them. We walk lines in order
        # and resolve each load's handle argument by looking up the SSA name.
        handle_binding: dict[str, tuple[int, int]] = {}  # cbv only
        # (slot, space, row) -> count
        sites: dict[tuple[int | None, int | None, int], int] = {}
        for line in text.splitlines():
            ssa = _DXIL_SSA_ASSIGN_RE.match(line)
            ssa_name = ssa.group(1) if ssa else None

            ch = _DXIL_CREATE_HANDLE_RE.search(line)
            if ch and ssa_name:
                try:
                    if int(ch.group("resclass")) == 2:
                        handle_binding[ssa_name] = (
                            int(ch.group("bind")),
                            int(ch.group("range_id")),
                        )
                except ValueError:
                    pass

            chb = _DXIL_CREATE_HANDLE_BIND_RE.search(line)
            if chb and ssa_name and chb.group("zero") is None:
                try:
                    if int(chb.group("resclass")) == 2:
                        handle_binding[ssa_name] = (
                            int(chb.group("lo")),
                            int(chb.group("space")),
                        )
                except ValueError:
                    pass

            ah = _DXIL_ANNOTATE_HANDLE_RE.search(line)
            if ah and ssa_name:
                src = ah.group("src")
                if src in handle_binding:
                    handle_binding[ssa_name] = handle_binding[src]

            ld = _DXIL_CBUF_LOAD_RE.search(line)
            if ld:
                try:
                    row = int(ld.group("row"))
                except ValueError:
                    continue
                handle = ld.group("handle")
                binding = handle_binding.get(handle)
                slot = binding[0] if binding else None
                space = binding[1] if binding else None
                key = (slot, space, row)
                sites[key] = sites.get(key, 0) + 1
                out.total_load_sites += 1
        out.reads = [
            CbufferReadSite(
                cbuffer_slot=slot,
                register_space=space,
                row=row,
                byte_offset=row * 16,
                occurrences=n,
                kind="DXIL",
            )
            for (slot, space, row), n in sorted(
                sites.items(),
                key=lambda kv: (kv[0][0] is None, kv[0]),
            )
        ]
        unattributed = sum(n for (slot, _, _), n in sites.items() if slot is None)
        if unattributed > 0:
            out.warnings.append(
                f"{unattributed} cbufferLoad site(s) could not be attributed to a "
                "binding — the load's handle SSA wasn't preceded by a tracked "
                "createHandle / createHandleFromBinding chain."
            )

    elif fmt == "DXBC":
        for dm in _DXBC_DCL_CB_RE.finditer(text):
            try:
                out.declared_cbuffers.append({
                    "slot": int(dm.group("slot")),
                    "size_rows": int(dm.group("size")),
                })
            except ValueError:
                pass
        sites: dict[tuple[int, int], int] = {}
        for rm in _DXBC_CB_REF_RE.finditer(text):
            try:
                slot = int(rm.group("slot"))
                row = int(rm.group("row"))
            except ValueError:
                continue
            sites[(slot, row)] = sites.get((slot, row), 0) + 1
            out.total_load_sites += 1
        out.reads = [
            CbufferReadSite(
                cbuffer_slot=slot,
                register_space=None,
                row=row,
                byte_offset=row * 16,
                occurrences=n,
                kind="DXBC",
            )
            for (slot, row), n in sorted(sites.items())
        ]
    else:
        out.warnings.append(
            f"unknown container format ({disasm.container_format}); cbuffer-read "
            "analysis skipped. Inspect disassembly text manually."
        )

    return out


# --- bound-PSO resolution at an event --------------------------------------


_GET_PSO_RE = re.compile(r"GetPipelineState\s*\(\s*(\d+)\s*\)")


def pso_id_from_state_value(raw: str | None) -> int | None:
    """Pull the integer PSO ApiObjectId out of a SetPipelineState argument
    string like ``"GetPipelineState(852)"``. Returns None if not parseable.
    """
    if not raw:
        return None
    m = _GET_PSO_RE.search(raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
