"""Parser for the C++ project produced by ``pixtool export-to-cpp``.

PIX's C++ export is the only practical way to recover *the exact bound state
at a specific event* without an interactive UI session — it serializes every
command list call as a sequence of D3D12 API invocations, so we can walk that
sequence and replay state up to a target Global ID.

We don't try to compile the C++ — we just parse the call sequence with regex.
The parser is intentionally permissive: PIX's export formatting has shifted
between versions, so we capture the call name and raw argument string, then
extract specific arguments only for the call kinds we care about (root-signature
binding, descriptor table, PSO, root descriptor, IB/VB views, RTV/DSV sets).

Output: a list of `BindingEvent` records ordered by event index, plus helpers
to compute the bound state at any event.

Caveats:
  - "Global ID" in the event list is not always identical to the call index in
    the C++ export. PIX assigns Global IDs across queues including markers /
    barriers, while the C++ export typically lists only command-list calls.
    The parser records the original PIX-style ``// Event <id>`` comment when
    present so callers can correlate. If no event-id comment is present, the
    parser falls back to a monotonic counter.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Any


# Match call lines like:
#   pCommandList->SetGraphicsRootDescriptorTable(0, BaseGpuDescriptor + 12);
#   commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
# Be generous about whitespace, the receiver name, and arrow vs dot.
_CALL_RE = re.compile(
    r"""
    (?P<receiver>[A-Za-z_][A-Za-z_0-9]*)         # e.g. pCommandList
    \s*(?:->|\.)\s*
    (?P<method>[A-Za-z_][A-Za-z_0-9]*)           # method name
    \s*\(
    (?P<args>.*?)                                # raw args (non-greedy)
    \)\s*;
    """,
    re.VERBOSE | re.DOTALL,
)

# `// Event 2372` markers PIX often emits.
_EVENT_COMMENT_RE = re.compile(r"//\s*Event\s+(?:#\s*)?(\d+)\b", re.I)
# Some versions emit `// GlobalID 2372` or `// PIX Event 2372`.
_GLOBAL_ID_COMMENT_RE = re.compile(r"//\s*(?:Global\s*ID|PIX\s*Event)\s*[:=]?\s*(\d+)\b", re.I)


# Methods we explicitly understand. Anything else is still captured as raw.
INTERESTING_METHODS = {
    "SetGraphicsRootSignature",
    "SetComputeRootSignature",
    "SetPipelineState",
    "SetPipelineState1",
    "SetGraphicsRootDescriptorTable",
    "SetComputeRootDescriptorTable",
    "SetGraphicsRootConstantBufferView",
    "SetComputeRootConstantBufferView",
    "SetGraphicsRootShaderResourceView",
    "SetComputeRootShaderResourceView",
    "SetGraphicsRootUnorderedAccessView",
    "SetComputeRootUnorderedAccessView",
    "SetGraphicsRoot32BitConstant",
    "SetComputeRoot32BitConstant",
    "SetGraphicsRoot32BitConstants",
    "SetComputeRoot32BitConstants",
    "SetDescriptorHeaps",
    "OMSetRenderTargets",
    "OMSetStencilRef",
    "OMSetBlendFactor",
    "IASetIndexBuffer",
    "IASetVertexBuffers",
    "IASetPrimitiveTopology",
    "RSSetViewports",
    "RSSetScissorRects",
    "DrawInstanced",
    "DrawIndexedInstanced",
    "Dispatch",
    "DispatchMesh",
    "DispatchRays",
    "ExecuteIndirect",
    "ResourceBarrier",
    "CopyResource",
    "CopyTextureRegion",
    "CopyBufferRegion",
    "ClearRenderTargetView",
    "ClearDepthStencilView",
    "ClearUnorderedAccessViewUint",
    "ClearUnorderedAccessViewFloat",
    "BeginEvent",
    "EndEvent",
    "SetMarker",
    "Close",
    "Reset",
}


@dataclass
class BindingEvent:
    """One parsed C++ call site."""

    call_index: int                # monotonic index across the file (0-based)
    global_id: int | None          # PIX Global ID if a comment recovered it
    receiver: str                  # e.g. 'pCommandList'
    method: str
    args: str                      # raw args inside the parens, whitespace-normalized
    source_file: str
    source_line: int

    def is_draw(self) -> bool:
        return self.method in ("DrawInstanced", "DrawIndexedInstanced", "ExecuteIndirect")

    def is_dispatch(self) -> bool:
        return self.method in ("Dispatch", "DispatchMesh", "DispatchRays")


def _split_args(arg_str: str) -> list[str]:
    """Split a comma-separated argument string while respecting nesting / strings."""
    out: list[str] = []
    depth = 0
    in_str = False
    str_quote = ""
    current: list[str] = []
    i = 0
    while i < len(arg_str):
        ch = arg_str[i]
        if in_str:
            current.append(ch)
            if ch == "\\" and i + 1 < len(arg_str):
                current.append(arg_str[i + 1])
                i += 2
                continue
            if ch == str_quote:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                str_quote = ch
                current.append(ch)
            elif ch in "({[<":
                depth += 1
                current.append(ch)
            elif ch in ")}]>":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                out.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        out.append(tail)
    return out


def parse_file(path: Path) -> Iterator[BindingEvent]:
    """Yield every interesting D3D12 call from a single C++ file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    # Track last-seen event-id comments so we can attach them to the next call.
    pending_event_id: int | None = None
    # We walk the text line-by-line to maintain source_line, but call matches
    # might span lines. We fall back to a line-aware buffer.
    buf: list[str] = []
    buf_start_line = 1
    line_no = 0
    call_idx = 0
    src_name = path.name

    for raw_line in text.splitlines(keepends=False):
        line_no += 1
        ec = _EVENT_COMMENT_RE.search(raw_line) or _GLOBAL_ID_COMMENT_RE.search(raw_line)
        if ec:
            try:
                pending_event_id = int(ec.group(1))
            except ValueError:
                pass
        if not buf:
            buf_start_line = line_no
        buf.append(raw_line)
        # Try to consume any complete `…;` statements in the accumulated buffer.
        joined = "\n".join(buf)
        last_semi = joined.rfind(";")
        if last_semi == -1:
            # No statement terminator yet — keep buffering but cap growth.
            if len(buf) > 200:
                buf = buf[-50:]
                buf_start_line = max(1, line_no - 50)
            continue
        chunk = joined[: last_semi + 1]
        remainder = joined[last_semi + 1 :]
        # Process chunk for call matches.
        for m in _CALL_RE.finditer(chunk):
            method = m.group("method")
            args = re.sub(r"\s+", " ", m.group("args")).strip()
            evt = BindingEvent(
                call_index=call_idx,
                global_id=pending_event_id,
                receiver=m.group("receiver"),
                method=method,
                args=args,
                source_file=src_name,
                source_line=buf_start_line,  # approximate
            )
            call_idx += 1
            yield evt
            # Each consumed call invalidates the most-recent event-id comment.
            pending_event_id = None
        buf = [remainder] if remainder.strip() else []
        if buf:
            buf_start_line = line_no


@dataclass
class CppExport:
    """In-memory view of an export-to-cpp directory."""

    root: Path
    events: list[BindingEvent] = field(default_factory=list)
    files_parsed: list[Path] = field(default_factory=list)

    @property
    def call_count(self) -> int:
        return len(self.events)

    @property
    def event_ids_seen(self) -> list[int]:
        return sorted({e.global_id for e in self.events if e.global_id is not None})


def fingerprint_dir(root: Path) -> str:
    """A stable hash of every .cpp/.h under ``root`` (path + size + mtime)."""
    h = hashlib.sha256()
    items: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".cpp", ".h", ".hpp", ".inl"):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        items.append(f"{p.relative_to(root)}|{st.st_size}|{int(st.st_mtime)}")
    for item in items:
        h.update(item.encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def parse_export(root: Path) -> CppExport:
    """Parse all C++ source files under ``root`` into a unified call list.

    Files are processed in name order so call_index is reproducible.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"export-to-cpp directory not found: {root}")
    files: list[Path] = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".cpp"
    )
    export = CppExport(root=root, files_parsed=files)
    base_idx = 0
    for f in files:
        for ev in parse_file(f):
            # Re-base call_index across multiple files.
            ev = BindingEvent(
                call_index=base_idx + ev.call_index,
                global_id=ev.global_id,
                receiver=ev.receiver,
                method=ev.method,
                args=ev.args,
                source_file=str(f.relative_to(root)),
                source_line=ev.source_line,
            )
            export.events.append(ev)
        # advance base by last event's call_index + 1 if any new events were added
        if export.events:
            base_idx = export.events[-1].call_index + 1
    return export


# ---- state replay -----------------------------------------------------------


@dataclass
class GraphicsBindings:
    root_signature: str | None = None
    pso: str | None = None
    primitive_topology: str | None = None
    root_params: dict[int, dict[str, Any]] = field(default_factory=dict)
    descriptor_heaps: list[str] = field(default_factory=list)
    rtvs: list[str] = field(default_factory=list)
    dsv: str | None = None
    index_buffer: dict[str, Any] | None = None
    vertex_buffers: dict[int, dict[str, Any]] = field(default_factory=dict)
    viewports: list[str] = field(default_factory=list)
    scissors: list[str] = field(default_factory=list)
    last_event_index_applied: int | None = None
    last_event_global_id: int | None = None


@dataclass
class ComputeBindings:
    root_signature: str | None = None
    pso: str | None = None
    root_params: dict[int, dict[str, Any]] = field(default_factory=dict)
    descriptor_heaps: list[str] = field(default_factory=list)
    last_event_index_applied: int | None = None
    last_event_global_id: int | None = None


@dataclass
class StateAtEvent:
    graphics: GraphicsBindings
    compute: ComputeBindings
    target_global_id: int | None
    target_call_index: int
    applied_calls: int


def _coerce_root_param_index(arg_list: list[str]) -> int | None:
    if not arg_list:
        return None
    raw = arg_list[0].strip()
    try:
        return int(raw, 0)
    except ValueError:
        m = re.match(r"^[-+]?\d+", raw)
        if m:
            try:
                return int(m.group(0))
            except ValueError:
                return None
        return None


def _apply_call(ev: BindingEvent, gfx: GraphicsBindings, comp: ComputeBindings) -> None:
    args = _split_args(ev.args)
    m = ev.method

    if m == "SetGraphicsRootSignature":
        gfx.root_signature = args[0] if args else None
    elif m == "SetComputeRootSignature":
        comp.root_signature = args[0] if args else None
    elif m in ("SetPipelineState", "SetPipelineState1"):
        gfx.pso = args[0] if args else None
        comp.pso = args[0] if args else None  # PSO is shared; assignment per-engine is set on bind
    elif m == "SetDescriptorHeaps":
        heaps = args[1:] if len(args) >= 2 else []
        gfx.descriptor_heaps = heaps
        comp.descriptor_heaps = heaps
    elif m in ("SetGraphicsRootDescriptorTable", "SetComputeRootDescriptorTable"):
        idx = _coerce_root_param_index(args)
        if idx is not None:
            entry = {"kind": "descriptor_table", "handle": args[1] if len(args) > 1 else None, "raw": ev.args}
            (gfx if m.startswith("SetGraphics") else comp).root_params[idx] = entry
    elif m in (
        "SetGraphicsRootConstantBufferView",
        "SetComputeRootConstantBufferView",
        "SetGraphicsRootShaderResourceView",
        "SetComputeRootShaderResourceView",
        "SetGraphicsRootUnorderedAccessView",
        "SetComputeRootUnorderedAccessView",
    ):
        idx = _coerce_root_param_index(args)
        if idx is not None:
            kind_map = {
                "SetGraphicsRootConstantBufferView": "cbv",
                "SetComputeRootConstantBufferView": "cbv",
                "SetGraphicsRootShaderResourceView": "srv",
                "SetComputeRootShaderResourceView": "srv",
                "SetGraphicsRootUnorderedAccessView": "uav",
                "SetComputeRootUnorderedAccessView": "uav",
            }
            entry = {
                "kind": kind_map[m],
                "gpu_va": args[1] if len(args) > 1 else None,
                "raw": ev.args,
            }
            (gfx if m.startswith("SetGraphics") else comp).root_params[idx] = entry
    elif m in ("SetGraphicsRoot32BitConstant", "SetComputeRoot32BitConstant"):
        idx = _coerce_root_param_index(args)
        if idx is not None:
            target = gfx if m.startswith("SetGraphics") else comp
            entry = target.root_params.setdefault(idx, {"kind": "32bit_constants", "values": {}})
            if entry.get("kind") != "32bit_constants":
                entry = {"kind": "32bit_constants", "values": {}}
                target.root_params[idx] = entry
            try:
                slot = int(args[2], 0) if len(args) > 2 else 0
            except ValueError:
                slot = 0
            entry["values"][slot] = args[1] if len(args) > 1 else None
    elif m in ("SetGraphicsRoot32BitConstants", "SetComputeRoot32BitConstants"):
        idx = _coerce_root_param_index(args)
        if idx is not None:
            entry = {"kind": "32bit_constants_block", "raw": ev.args}
            (gfx if m.startswith("SetGraphics") else comp).root_params[idx] = entry
    elif m == "IASetPrimitiveTopology":
        gfx.primitive_topology = args[0] if args else None
    elif m == "IASetIndexBuffer":
        gfx.index_buffer = {"raw": ev.args}
    elif m == "IASetVertexBuffers":
        # (StartSlot, NumViews, pViews) — keep raw.
        try:
            start = int(args[0], 0)
            num = int(args[1], 0)
        except (ValueError, IndexError):
            start = 0
            num = 0
        gfx.vertex_buffers[start] = {"num_views": num, "raw": ev.args}
    elif m == "RSSetViewports":
        gfx.viewports = [args[1]] if len(args) > 1 else [ev.args]
    elif m == "RSSetScissorRects":
        gfx.scissors = [args[1]] if len(args) > 1 else [ev.args]
    elif m == "OMSetRenderTargets":
        try:
            num_rt = int(args[0], 0)
        except (ValueError, IndexError):
            num_rt = 0
        gfx.rtvs = [args[1]] if len(args) > 1 else []
        gfx.rtvs_count = num_rt  # informational
        gfx.dsv = args[3] if len(args) > 3 else None


def state_at_event(
    export: CppExport,
    *,
    global_id: int | None = None,
    call_index: int | None = None,
    inclusive: bool = True,
) -> StateAtEvent:
    """Replay calls up to (and optionally including) the target event.

    Pick ONE of ``global_id`` or ``call_index``. If ``global_id`` is given, the
    parser searches forward for the first event whose ``global_id`` attribute
    matches, then optionally includes it. If neither is given, replays the
    whole export.
    """
    gfx = GraphicsBindings()
    comp = ComputeBindings()
    target_call_index = len(export.events)
    target_gid = global_id

    if global_id is not None:
        target_call_index = None
        for ev in export.events:
            if ev.global_id == global_id:
                target_call_index = ev.call_index
                break
        if target_call_index is None:
            # Couldn't locate that event id; fall back to "everything".
            target_call_index = len(export.events)
    elif call_index is not None:
        target_call_index = call_index

    applied = 0
    for ev in export.events:
        if ev.call_index > target_call_index:
            break
        if ev.call_index == target_call_index and not inclusive:
            break
        _apply_call(ev, gfx, comp)
        gfx.last_event_index_applied = ev.call_index
        comp.last_event_index_applied = ev.call_index
        if ev.global_id is not None:
            gfx.last_event_global_id = ev.global_id
            comp.last_event_global_id = ev.global_id
        applied += 1

    return StateAtEvent(
        graphics=gfx,
        compute=comp,
        target_global_id=target_gid,
        target_call_index=target_call_index,
        applied_calls=applied,
    )


def find_calls(
    export: CppExport,
    *,
    method: str | None = None,
    method_like: str | None = None,
    args_substring: str | None = None,
    global_id: int | None = None,
    between: tuple[int, int] | None = None,
    limit: int = 200,
) -> list[BindingEvent]:
    out: list[BindingEvent] = []
    for ev in export.events:
        if method and ev.method != method:
            continue
        if method_like and method_like.lower() not in ev.method.lower():
            continue
        if args_substring and args_substring.lower() not in ev.args.lower():
            continue
        if global_id is not None and ev.global_id != global_id:
            continue
        if between is not None:
            lo, hi = between
            if ev.global_id is None or ev.global_id < lo or ev.global_id > hi:
                continue
        out.append(ev)
        if len(out) >= limit:
            break
    return out
