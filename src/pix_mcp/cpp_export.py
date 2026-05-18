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


# Find the method name + opening paren after an arrow or dot operator. Works
# for all of these receiver forms:
#   pCommandList->SetGraphicsRootDescriptorTable(0, BaseGpuDescriptor + 12);
#   commandList.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
#   GetCommandList(3)->SetGraphicsRootConstantBufferView(5, GetGpuva(2362, 10240));
#   GetCommandList(3).Get()->ResourceBarrier(1, &barrier);
# We don't capture the full receiver — replay only cares about method/args —
# but we record the immediately-preceding identifier-or-paren-group as a hint.
_METHOD_RE = re.compile(
    r"""
    (?:->|\.)\s*
    (?P<method>[A-Za-z_][A-Za-z_0-9]*)
    \s*\(
    """,
    re.VERBOSE,
)

# `// GlobalId        = 1` (PIX 2603.25), `// Event 2372` (older PIX),
# `// GlobalID: 2372` / `// PIX Event 2372` (variants).
_EVENT_COMMENT_RE = re.compile(r"//\s*Event\s+(?:#\s*)?(\d+)\b", re.I)
_GLOBAL_ID_COMMENT_RE = re.compile(
    r"//\s*(?:Global\s*ID|PIX\s*Event)\s*[:=]?\s*(\d+)\b", re.I
)
# PIX-export helper for "this resource's GPU virtual address + offset":
#   GetGpuva(resource_id, offset)
# Older variants: GpuVa(rid, off), GetGPUVA(rid, off).
_GPUVA_RE = re.compile(
    r"\b(?:GetGpuva|GpuVa|GetGPUVA)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
    re.I,
)


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


def _balanced_args(text: str, open_paren_idx: int) -> tuple[str, int] | None:
    """Given text and the index of an opening '(', return (inside, end_idx)
    where end_idx points at the matching ')'. Respects nested parens, char/string
    literals, and line/block comments. Returns None if unbalanced.
    """
    assert text[open_paren_idx] == "("
    i = open_paren_idx + 1
    depth = 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"' or ch == "'":
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n:
            if text[i + 1] == "/":
                # line comment — skip to end of line
                nl = text.find("\n", i + 2)
                i = n if nl == -1 else nl + 1
                continue
            if text[i + 1] == "*":
                end = text.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx + 1 : i], i
        i += 1
    return None


def _extract_receiver(text: str, op_idx: int) -> str:
    """Walk backwards from the ``->`` / ``.`` operator at ``op_idx`` and
    return the receiver expression (whitespace-normalized).

    Handles all the receiver forms PIX emits:
      * ``pCommandList`` (plain identifier)
      * ``GetCommandList(3)`` (call expression)
      * ``GetCommandList(3).Get()`` (chained)
      * ``g_resourceReader`` (smart pointer / global)
    Stops at the previous statement terminator (``;``), block boundary
    (``{``/``}``), or top-level comma in an expression.
    """
    i = op_idx - 1
    # Skip whitespace immediately before the operator.
    while i >= 0 and text[i] in " \t\r\n":
        i -= 1
    end = i + 1  # exclusive
    depth = 0
    while i >= 0:
        ch = text[i]
        if ch in ")]}":
            depth += 1
            i -= 1
            continue
        if ch in "([{":
            if depth == 0:
                break  # we've stepped outside the receiver expression
            depth -= 1
            i -= 1
            continue
        if depth == 0 and ch in ";,":
            break
        # Skip backwards through string/char literals so we don't trip on
        # quotes (uncommon as a receiver but defensible).
        if ch in ('"', "'"):
            quote = ch
            j = i - 1
            while j >= 0:
                if text[j] == quote and (j == 0 or text[j - 1] != "\\"):
                    i = j - 1
                    break
                j -= 1
            else:
                break
            continue
        i -= 1
    start = i + 1
    raw = text[start:end].strip()
    # Normalize whitespace + line continuations.
    return re.sub(r"\s+", " ", raw)


def parse_file(path: Path) -> Iterator[BindingEvent]:
    """Yield every D3D12 call from a single C++ file.

    Strategy: walk the file once, tracking ``{`` / ``}`` block depth and the
    ``// GlobalId = N`` comment that opens each block. PIX 2603.25 groups
    *several* D3D12 calls under one GlobalId, so we attach the pending id to
    every call until the enclosing block closes — older parsers that consumed
    the id on first match would leave most calls without an id.

    We find calls by looking for ``->Method(`` or ``.Method(`` and balancing
    parens by hand (the previous regex-only approach choked on
    ``GetCommandList(3)->Method(...)`` because the receiver had parens of
    its own).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    src_name = path.name

    # Precompute line offsets for source_line lookup.
    line_starts: list[int] = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def line_of(idx: int) -> int:
        # Binary search line_starts for idx.
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    # Walk the file, tracking brace depth and per-depth pending GlobalId.
    pending_id_stack: list[int | None] = [None]  # depth 0
    call_idx = 0
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        # Skip strings / chars
        if ch == '"' or ch == "'":
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        # Skip block comments
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        # Line comment — also check for GlobalId / Event markers here
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            nl = text.find("\n", i + 2)
            line = text[i : (n if nl == -1 else nl)]
            ec = (
                _GLOBAL_ID_COMMENT_RE.search(line)
                or _EVENT_COMMENT_RE.search(line)
            )
            if ec:
                try:
                    pending_id_stack[-1] = int(ec.group(1))
                except ValueError:
                    pass
            i = n if nl == -1 else nl + 1
            continue
        if ch == "{":
            # Inherit pending id into the new block (PIX wraps GlobalId blocks
            # in `// GlobalId = N\n{ ... }`).
            pending_id_stack.append(pending_id_stack[-1])
            i += 1
            continue
        if ch == "}":
            if len(pending_id_stack) > 1:
                pending_id_stack.pop()
            i += 1
            continue
        # Look for `->Ident(` or `.Ident(`
        m = _METHOD_RE.match(text, i)
        if not m:
            i += 1
            continue
        open_paren_idx = m.end() - 1  # the '(' just consumed
        balanced = _balanced_args(text, open_paren_idx)
        if balanced is None:
            i = m.end()
            continue
        inside, close_idx = balanced
        # Need a trailing ';' (possibly preceded by whitespace) — otherwise this
        # is a method call inside an expression (e.g. inside another call), not
        # a top-level statement.
        j = close_idx + 1
        while j < n and text[j] in " \t":
            j += 1
        if j >= n or text[j] != ";":
            # not a statement — skip past the close paren and continue
            i = close_idx + 1
            continue
        method = m.group("method")
        args = re.sub(r"\s+", " ", inside).strip()
        receiver = _extract_receiver(text, m.start())
        yield BindingEvent(
            call_index=call_idx,
            global_id=pending_id_stack[-1],
            receiver=receiver,
            method=method,
            args=args,
            source_file=src_name,
            source_line=line_of(m.start()),
        )
        call_idx += 1
        i = j + 1


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
    # True iff the requested global_id was located in the export. When False
    # the replay still produced a state, but it's "state after applying every
    # call we have" — usually wrong; callers should treat the result as
    # unreliable.
    found_global_id: bool = True


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
            gpu_va = args[1] if len(args) > 1 else None
            entry: dict[str, Any] = {
                "kind": kind_map[m],
                "gpu_va": gpu_va,
                "raw": ev.args,
            }
            # PIX export-to-cpp emits GPU virtual addresses as
            # `GetGpuva(resource_id, offset)` — extract structured ids so
            # downstream tools can fetch the underlying bytes without
            # re-parsing the call string.
            if gpu_va:
                gm = _GPUVA_RE.search(gpu_va)
                if gm:
                    try:
                        entry["resource_id"] = int(gm.group(1))
                        entry["offset"] = int(gm.group(2))
                    except ValueError:
                        pass
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

    found_gid = True
    if global_id is not None:
        # PIX groups several D3D12 calls (SetRootSig, SetPSO, SetCBV*, Draw)
        # under a single GlobalId block, so the natural meaning of
        # "state at global_id N" is "every call from that block has been
        # applied". Find the LAST event with this global_id, not the first.
        target_call_index = None
        for ev in export.events:
            if ev.global_id == global_id:
                target_call_index = ev.call_index
        if target_call_index is None:
            # Couldn't locate that event id. Fall back to "everything", but
            # flag the result so callers don't act on a bogus end-of-frame
            # state thinking it's the state at their event.
            target_call_index = len(export.events)
            found_gid = False
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
        found_global_id=found_gid,
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


# ---- root signature layout extraction --------------------------------------

# PIX emits root signatures as inline C++ array initialization. Example:
#   // ApiObjectId     = 2361
#   {
#       static D3D12_ROOT_PARAMETER1 rootParameters[8];
#       rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
#       rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
#       {
#           static D3D12_DESCRIPTOR_RANGE1 descriptorRanges[1];
#           descriptorRanges[0] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 64, 0, 0, ..., 4294967295 };
#           rootParameters[0].DescriptorTable = { 1, descriptorRanges };
#       }
#       ...
#       rootParameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
#       rootParameters[3].Descriptor = { 0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC };
#       D3D12_STATIC_SAMPLER_DESC samplers[6];
#       samplers[0] = { ... };
#       D3D12_ROOT_SIGNATURE_DESC1 rootSignatureDesc = { 8, rootParameters, 6, samplers, FLAGS };
#       ...
#       CreateAndTrackRootSignature(2361, ...);
#   }
#
# We locate the block by searching for CreateAndTrackRootSignature(<id>, then
# walking back to the matching '{'.

_TRACK_ROOTSIG_RE = re.compile(
    r"CreateAndTrackRootSignature\s*\(\s*(\d+)\s*,",
)
_ROOT_PARAM_TYPE_RE = re.compile(
    r"rootParameters\[(\d+)\]\.ParameterType\s*=\s*D3D12_ROOT_PARAMETER_TYPE_([A-Z0-9_]+)\s*;"
)
_ROOT_PARAM_VIS_RE = re.compile(
    r"rootParameters\[(\d+)\]\.ShaderVisibility\s*=\s*D3D12_SHADER_VISIBILITY_([A-Z0-9_]+)\s*;"
)
# rootParameters[3].Descriptor = { 0, 0, FLAGS };   -- shader_register, register_space, flags
_ROOT_PARAM_DESC_RE = re.compile(
    r"rootParameters\[(\d+)\]\.Descriptor\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*([^}]+?)\s*\}\s*;"
)
# rootParameters[K].Constants = { shader_register, register_space, num_32bit_values };
_ROOT_PARAM_CONST_RE = re.compile(
    r"rootParameters\[(\d+)\]\.Constants\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}\s*;"
)
# rootParameters[K].DescriptorTable = { num_ranges, ranges_identifier };
_ROOT_PARAM_TABLE_RE = re.compile(
    r"rootParameters\[(\d+)\]\.DescriptorTable\s*=\s*\{\s*(\d+)\s*,\s*([A-Za-z_][A-Za-z_0-9]*)\s*\}\s*;"
)
# descriptorRanges[K] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, num, baseRegister, regSpace, FLAGS, offset };
_DESC_RANGE_RE = re.compile(
    r"descriptorRanges\[(\d+)\]\s*=\s*\{\s*"
    r"D3D12_DESCRIPTOR_RANGE_TYPE_([A-Z0-9_]+)\s*,\s*"
    r"(\d+)\s*,\s*"     # num descriptors (-1 / 0xFFFFFFFF for unbounded)
    r"(\d+)\s*,\s*"     # base shader register
    r"(\d+)\s*,\s*"     # register space
    r"([^,}]+?)\s*,\s*" # flags
    r"(\d+)\s*\}\s*;"   # offset in descriptors from table start
)
_ROOTSIG_DESC_RE = re.compile(
    r"D3D12_ROOT_SIGNATURE_DESC1?\s+\w+\s*=\s*\{[^}]*?\}\s*;",
    re.DOTALL,
)
_FLAGS_RE = re.compile(
    r"D3D12_ROOT_SIGNATURE_FLAG_[A-Z0-9_]+"
)


def _find_enclosing_block(text: str, idx: int) -> tuple[int, int] | None:
    """Given an index inside text, find the smallest enclosing { ... } block.

    Returns (open_idx, close_idx) inclusive of braces, or None if unbalanced.
    """
    # Walk backwards counting braces to find the opening '{'.
    depth = 0
    i = idx
    open_idx: int | None = None
    while i >= 0:
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth == 0:
                open_idx = i
                break
            depth -= 1
        i -= 1
    if open_idx is None:
        return None
    # Walk forward from open_idx to find matching '}'.
    j = open_idx + 1
    depth = 1
    n = len(text)
    while j < n and depth > 0:
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return open_idx, j - 1


@dataclass
class DescriptorRange:
    range_type: str           # "SRV" / "UAV" / "CBV" / "SAMPLER"
    num_descriptors: int      # 4294967295 means unbounded (-1)
    base_shader_register: int
    register_space: int
    flags: str                # raw text — typically a `|`-joined list of D3D12_DESCRIPTOR_RANGE_FLAG_*
    offset_in_descriptors: int  # 4294967295 means D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND


@dataclass
class RootParameter:
    index: int
    kind: str                 # "DESCRIPTOR_TABLE" / "CBV" / "SRV" / "UAV" / "32BIT_CONSTANTS"
    visibility: str           # "ALL" / "PIXEL" / "VERTEX" / "HULL" / "DOMAIN" / "GEOMETRY" / "AMPLIFICATION" / "MESH"
    shader_register: int | None = None
    register_space: int | None = None
    num_32bit_values: int | None = None
    flags: str | None = None  # D3D12_ROOT_DESCRIPTOR_FLAG_*
    descriptor_ranges: list[DescriptorRange] = field(default_factory=list)


@dataclass
class RootSignatureLayout:
    root_signature_id: int
    params: list[RootParameter]
    flags: list[str]
    source_file: str
    source_line: int
    raw_block_excerpt: str   # first ~1500 chars of the block, for verification


# ---- PSO shader-stage extraction -------------------------------------------

# Match a CreatePipelineState_<id>() function definition opener.
_PSO_FUNC_RE = re.compile(
    r"^void\s+CreatePipelineState_(\d+)\s*\(\s*\)\s*$",
    re.MULTILINE,
)
# `g_resourceReader->Read(data, 11314);` — the compressed shader-blob read.
_PSO_READ_RE = re.compile(
    r"g_resourceReader\s*->\s*Read\s*\(\s*[A-Za-z_][A-Za-z_0-9]*\s*,\s*(\d+)\s*\)\s*;"
)
# `pssDesc.VS = { reinterpret_cast<BYTE*>(&data[offset]), 7776 };`
# We only need the stage name and the length literal — the offset is always
# a running `offset` variable in PIX's generated code, so we compute it
# ourselves from the source order.
_PSO_STAGE_RE = re.compile(
    r"pssDesc\s*\.\s*(?P<stage>VS|PS|HS|DS|GS|CS|AS|MS)\s*=\s*\{"
    r"[^{}]*?(?P<length>\d+)\s*\}\s*;"
)
# `pssDesc.pRootSignature = GetRootSignature(2361);`
_PSO_ROOTSIG_RE = re.compile(
    r"pssDesc\s*\.\s*pRootSignature\s*=\s*GetRootSignature\s*\(\s*(\d+)\s*\)"
)


@dataclass
class PsoShaderStage:
    """One shader stage within a PSO's bytecode blob.

    The ``offset`` and ``length`` are in *decompressed* bytes — they slice
    into the decompressed contents of the PSO's resources.bin chunk.
    """

    stage: str         # "VS" | "PS" | "HS" | "DS" | "GS" | "CS" | "AS" | "MS"
    offset: int        # byte offset into the decompressed blob
    length: int        # bytecode length in bytes
    source_line: int   # line of the pssDesc.<STAGE> = {...} assignment


@dataclass
class PsoStageLayout:
    pso_id: int
    root_signature_id: int | None
    compressed_blob_size: int      # bytes the Read() call requests
    stages: list[PsoShaderStage]   # in source order (also the slicing order)
    source_file: str
    source_line: int


def parse_pso_shader_stages(export: CppExport, pso_id: int) -> PsoStageLayout | None:
    """Locate ``CreatePipelineState_<pso_id>()`` and return its per-stage layout.

    PIX's export-to-cpp emits each PSO as a ``CreatePipelineState_<id>()``
    function that:

      1. Calls ``g_resourceReader->Read(data, <COMPRESSED_SIZE>)`` to pull the
         PSO's combined shader-bytecode blob out of ``resources.bin``.
      2. Slices that decompressed blob into per-stage views with
         ``pssDesc.<STAGE> = { reinterpret_cast<BYTE*>(&data[offset]), <LEN> };``
         where the C++ variable ``offset`` is bumped by each stage's length.

    We parse the function body in source order, recover each stage's length
    literal, and recompute the running offsets (since the source uses a
    runtime variable). The result is everything the disassembler needs to
    locate each stage inside the decompressed blob.
    """
    for path in export.files_parsed:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _PSO_FUNC_RE.finditer(text):
            if int(m.group(1)) != pso_id:
                continue
            # Find the matching {...} body.
            i = m.end()
            while i < len(text) and text[i] in " \t\r\n":
                i += 1
            if i >= len(text) or text[i] != "{":
                continue
            depth = 1
            j = i + 1
            n = len(text)
            in_str = False
            quote = ""
            while j < n and depth > 0:
                ch = text[j]
                if in_str:
                    if ch == "\\" and j + 1 < n:
                        j += 2
                        continue
                    if ch == quote:
                        in_str = False
                    j += 1
                    continue
                if ch == '"' or ch == "'":
                    in_str = True
                    quote = ch
                    j += 1
                    continue
                if ch == "/" and j + 1 < n:
                    if text[j + 1] == "/":
                        nl = text.find("\n", j + 2)
                        j = n if nl == -1 else nl + 1
                        continue
                    if text[j + 1] == "*":
                        end = text.find("*/", j + 2)
                        j = n if end == -1 else end + 2
                        continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                j += 1
            if depth != 0:
                continue
            body = text[i + 1 : j - 1]
            body_start_in_file = i + 1

            # Helper: line number in file for a body-relative index.
            def line_of(body_idx: int) -> int:
                abs_idx = body_start_in_file + body_idx
                return text.count("\n", 0, abs_idx) + 1

            read_match = _PSO_READ_RE.search(body)
            compressed_size = int(read_match.group(1)) if read_match else 0

            rootsig_match = _PSO_ROOTSIG_RE.search(body)
            root_sig_id = int(rootsig_match.group(1)) if rootsig_match else None

            stages: list[PsoShaderStage] = []
            running = 0
            for sm in _PSO_STAGE_RE.finditer(body):
                stage = sm.group("stage")
                length = int(sm.group("length"))
                stages.append(PsoShaderStage(
                    stage=stage,
                    offset=running,
                    length=length,
                    source_line=line_of(sm.start()),
                ))
                running += length

            func_line = text.count("\n", 0, m.start()) + 1
            return PsoStageLayout(
                pso_id=pso_id,
                root_signature_id=root_sig_id,
                compressed_blob_size=compressed_size,
                stages=stages,
                source_file=str(path.relative_to(export.root)),
                source_line=func_line,
            )
    return None


def list_all_psos(export: CppExport) -> list[PsoStageLayout]:
    """Parse every ``CreatePipelineState_<id>()`` in the export.

    Useful as an overview of what shaders are available to dump.
    """
    out: list[PsoStageLayout] = []
    seen: set[int] = set()
    for path in export.files_parsed:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _PSO_FUNC_RE.finditer(text):
            pid = int(m.group(1))
            if pid in seen:
                continue
            seen.add(pid)
            layout = parse_pso_shader_stages(export, pid)
            if layout is not None:
                out.append(layout)
    return out


def parse_root_signature_layout(export: CppExport, root_sig_id: int) -> RootSignatureLayout | None:
    """Locate the inline definition of root signature ``root_sig_id`` and
    return its per-root-parameter layout.

    PIX's export-to-cpp inlines root sigs as ``D3D12_ROOT_PARAMETER1`` array
    initialization. We find the ``CreateAndTrackRootSignature(<id>, ...)``
    call, walk back to the enclosing ``{`` block, and parse the array
    assignments inside.
    """
    needle = f"CreateAndTrackRootSignature({root_sig_id},"
    for path in export.files_parsed:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Tolerate whitespace variations like `( 2361,` and `(  2361 ,`.
        for m in _TRACK_ROOTSIG_RE.finditer(text):
            try:
                if int(m.group(1)) != root_sig_id:
                    continue
            except ValueError:
                continue
            block = _find_enclosing_block(text, m.start())
            if not block:
                continue
            open_idx, close_idx = block
            body = text[open_idx : close_idx + 1]
            # Source line for the open brace.
            line_no = text.count("\n", 0, open_idx) + 1

            # Build per-index records.
            params_map: dict[int, RootParameter] = {}

            def get_param(i: int) -> RootParameter:
                if i not in params_map:
                    params_map[i] = RootParameter(index=i, kind="UNKNOWN", visibility="ALL")
                return params_map[i]

            for tm in _ROOT_PARAM_TYPE_RE.finditer(body):
                p = get_param(int(tm.group(1)))
                p.kind = tm.group(2)
            for vm in _ROOT_PARAM_VIS_RE.finditer(body):
                p = get_param(int(vm.group(1)))
                p.visibility = vm.group(2)
            for dm in _ROOT_PARAM_DESC_RE.finditer(body):
                p = get_param(int(dm.group(1)))
                p.shader_register = int(dm.group(2))
                p.register_space = int(dm.group(3))
                p.flags = dm.group(4).strip()
            for cm in _ROOT_PARAM_CONST_RE.finditer(body):
                p = get_param(int(cm.group(1)))
                p.shader_register = int(cm.group(2))
                p.register_space = int(cm.group(3))
                p.num_32bit_values = int(cm.group(4))

            # Descriptor tables — each rootParameters[K].DescriptorTable = {N, descriptorRanges}
            # references the `descriptorRanges` declared in the immediately
            # preceding inner `{...}` block. We scan all descriptorRanges
            # assignments inside body and group them by the *enclosing* inner
            # block, then attach to whichever rootParameters[K] the SAME block
            # also assigned via .DescriptorTable.
            # Simpler heuristic that matches PIX's pattern: for each
            # rootParameters[K].DescriptorTable assignment, scan backwards in
            # body for descriptorRanges[J] = {...} assignments inside the
            # same inner block, stopping at the previous rootParameters
            # assignment (which marks a different param's block).
            for tabm in _ROOT_PARAM_TABLE_RE.finditer(body):
                k = int(tabm.group(1))
                num_ranges = int(tabm.group(2))
                p = get_param(k)
                # Walk backwards from this match position to collect
                # descriptorRanges assignments up to the previous
                # `rootParameters[` token (or start of body).
                start = 0
                cutoff_match = None
                for prev in re.finditer(
                    r"rootParameters\[\d+\]", body[: tabm.start()]
                ):
                    cutoff_match = prev
                if cutoff_match is not None:
                    start = cutoff_match.end()
                window = body[start : tabm.start()]
                ranges: list[DescriptorRange] = []
                for rm in _DESC_RANGE_RE.finditer(window):
                    ranges.append(
                        DescriptorRange(
                            range_type=rm.group(2),
                            num_descriptors=int(rm.group(3)),
                            base_shader_register=int(rm.group(4)),
                            register_space=int(rm.group(5)),
                            flags=rm.group(6).strip(),
                            offset_in_descriptors=int(rm.group(7)),
                        )
                    )
                # Trim/extend to declared range count.
                if len(ranges) > num_ranges:
                    ranges = ranges[-num_ranges:]
                p.descriptor_ranges = ranges

            params = [params_map[i] for i in sorted(params_map.keys())]

            # Root signature flags — look for the D3D12_ROOT_SIGNATURE_DESC{,1}
            # struct literal and capture every flag mention inside it.
            sig_flags: list[str] = []
            for dm in _ROOTSIG_DESC_RE.finditer(body):
                # The flags portion is the last field before the closing brace.
                segment = dm.group(0)
                sig_flags = sorted(set(_FLAGS_RE.findall(segment)))
                if sig_flags:
                    break

            excerpt = body[:1500]
            return RootSignatureLayout(
                root_signature_id=root_sig_id,
                params=params,
                flags=sig_flags,
                source_file=str(path.relative_to(export.root)),
                source_line=line_no,
                raw_block_excerpt=excerpt,
            )
    return None
