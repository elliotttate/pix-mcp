"""Extractor for the binary resource data in a ``pixtool export-to-cpp`` directory.

PIX's C++ export bundles all captured resource bytes into a single
``resources.bin`` file. The file is **XPRESS-compressed** in sequential chunks
— the generated C++ replay engine reads chunks in a strict order via
``g_resourceReader->Read(buf, compressed_size)``. To pull bytes out of a
specific resource statically (i.e. without running the replay), we have to
reconstruct that read sequence ourselves: walk the call graph rooted at
``CreateAppResources_000()``, accumulate the per-call compressed sizes, and
that tells us the file offset of each chunk.

What we extract (v1):
  * Initial bytes for any resource created via ``CreateAndInitResource_<id>()``
    — this is where cbuffers, vertex/index buffers, and texture-upload data
    live at frame start.

Out of scope (yet):
  * Per-frame modifications to upload heaps inside ``PopulateCommandList_*``
    (those also consume reads, but discovering which read updates which
    resource needs richer call-graph tracking).
  * Shader bytecode in ``CreatePipelineState_*`` is *not* exposed as a
    resource, but we *do* still walk those reads to keep file offsets in
    sync.

Decompression uses Windows' Cabinet ``Decompress`` API via ctypes (the same
API that ``ResourceReader.cpp`` in the generated project calls).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import re
from dataclasses import dataclass, field
from pathlib import Path


# --- Windows XPRESS decompression via cabinet.dll ---------------------------

# COMPRESS_ALGORITHM_XPRESS from compressapi.h
_COMPRESS_ALGORITHM_XPRESS = 3
_ERROR_INSUFFICIENT_BUFFER = 122

_cabinet = ctypes.WinDLL("cabinet.dll")

_CreateDecompressor = _cabinet.CreateDecompressor
_CreateDecompressor.argtypes = [wt.DWORD, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_CreateDecompressor.restype = wt.BOOL

_CloseDecompressor = _cabinet.CloseDecompressor
_CloseDecompressor.argtypes = [ctypes.c_void_p]
_CloseDecompressor.restype = wt.BOOL

_Decompress = _cabinet.Decompress
_Decompress.argtypes = [
    ctypes.c_void_p,                 # decompressor handle
    ctypes.c_void_p, ctypes.c_size_t,  # compressed src + size
    ctypes.c_void_p, ctypes.c_size_t,  # uncompressed dst + size
    ctypes.POINTER(ctypes.c_size_t),  # *uncompressed_data_size
]
_Decompress.restype = wt.BOOL


def xpress_decompress(compressed: bytes) -> bytes:
    """Decompress an XPRESS chunk using the Windows Compression API."""
    handle = ctypes.c_void_p()
    if not _CreateDecompressor(_COMPRESS_ALGORITHM_XPRESS, None, ctypes.byref(handle)):
        raise OSError(ctypes.get_last_error(), "CreateDecompressor failed")
    try:
        src = (ctypes.c_ubyte * len(compressed)).from_buffer_copy(compressed)
        # First call: probe required uncompressed size.
        out_size = ctypes.c_size_t(0)
        ok = _Decompress(handle, src, len(compressed), None, 0, ctypes.byref(out_size))
        if ok:
            # Decompressed to zero bytes — unusual but valid.
            return b""
        err = ctypes.get_last_error()
        if err != _ERROR_INSUFFICIENT_BUFFER:
            # Some Windows builds don't populate GetLastError reliably through ctypes
            # if use_last_error wasn't requested on the DLL. Fall back to trusting
            # out_size, which Decompress fills even on the probe.
            if out_size.value == 0:
                raise OSError(err, f"Decompress probe failed (GetLastError={err})")
        dst_buf = (ctypes.c_ubyte * out_size.value)()
        actual = ctypes.c_size_t(0)
        ok = _Decompress(
            handle, src, len(compressed), dst_buf, out_size.value, ctypes.byref(actual)
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "Decompress (full) failed")
        return bytes(dst_buf[: actual.value])
    finally:
        _CloseDecompressor(handle)


# --- C++ parsing -----------------------------------------------------------

# Match a top-level `void Name()` function definition opening brace on the
# same or next non-blank line.
_FUNC_DEF_RE = re.compile(
    r"^void\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(\s*\)\s*$", re.MULTILINE
)
# A `g_resourceReader->Read(buf, NNN);` call. We don't need the buffer name.
_READ_CALL_RE = re.compile(
    r"g_resourceReader\s*->\s*Read\s*\(\s*[A-Za-z_][A-Za-z_0-9]*\s*,\s*(\d+)\s*\)\s*;"
)
# Plain top-level `Func();` invocation inside a body.
_INVOKE_RE = re.compile(
    r"^[ \t]*([A-Za-z_][A-Za-z_0-9]*)\s*\(\s*\)\s*;", re.MULTILINE
)


@dataclass
class _Item:
    """One source-ordered statement inside a function body that affects the
    runtime read sequence — either a g_resourceReader->Read call (with its
    compressed size literal), or a plain ``Foo();`` invocation that we'll
    recurse into."""

    kind: str        # "read" | "call"
    size: int | None  # read size (if kind == "read")
    callee: str | None  # function name (if kind == "call")


@dataclass
class _FuncBody:
    """Static view of one ``void Name()`` function — body text plus the
    lazily-computed item list."""

    name: str
    text: str
    items: list[_Item] = field(default_factory=list)


# Single pass per file uses this regex to enumerate every delimiter that
# affects brace depth or starts a string/comment region. Way faster than
# char-by-char walking in Python, and avoids the O(n²) rescan that
# per-delimiter ``str.find`` from a moving cursor would do.
_TOKEN_RE = re.compile(r'//|/\*|"|\'|\{|\}')


def _index_file_functions(src: str) -> dict[str, tuple[int, int]]:
    """Single-pass brace-balancing scan of one .cpp file.

    Returns ``name -> (body_start, body_end)`` where the indices bound the
    function body *excluding* the outer braces, ready for `src[start:end]`.

    Why this exists: PIX's 112-MB-of-source export contains ~16k function
    definitions. Doing one balanced-paren scan per function (or worse, one
    ``str.find`` per delimiter from a moving cursor) is O(n·f). This single
    pass uses ``re.finditer`` over all delimiters once and is O(n).
    """
    # First locate every "void Name()" so we know where to expect bodies.
    func_starts: list[tuple[str, int]] = []
    for m in _FUNC_DEF_RE.finditer(src):
        func_starts.append((m.group(1), m.end()))
    if not func_starts:
        return {}
    n = len(src)
    # Walk the file once, tracking brace depth and current "in string" /
    # "in comment" state. Whenever depth drops to 0 just after we entered
    # depth 1 inside a known function, record (name, body_start, here).
    bodies: dict[str, tuple[int, int]] = {}
    # Active stack of function names whose '{' we've passed but '}' we haven't.
    # In practice this is at most a few deep (PIX doesn't nest function defs).
    active: list[tuple[str, int]] = []  # (name, body_start)
    pending_funcs = iter(func_starts)
    next_func: tuple[str, int] | None = next(pending_funcs, None)
    depth = 0
    pos = 0
    while pos < n:
        m = _TOKEN_RE.search(src, pos)
        if not m:
            break
        idx = m.start()
        tok = m.group(0)
        # Before processing the token: if a known function definition starts
        # before this delimiter and at depth 0, we're about to enter it.
        while next_func is not None and depth == 0 and next_func[1] <= idx:
            # Verify that the next non-whitespace char at next_func[1] is '{'
            j = next_func[1]
            while j < n and src[j] in " \t\r\n":
                j += 1
            if j < n and src[j] == "{":
                # Push pending function — its '{' is at position j.
                # We'll register it when depth transitions from 0 to 1 below.
                active.append((next_func[0], j + 1))
            next_func = next(pending_funcs, None)
        if tok == "//":
            nl = src.find("\n", idx + 2)
            pos = n if nl == -1 else nl + 1
            continue
        if tok == "/*":
            end = src.find("*/", idx + 2)
            pos = n if end == -1 else end + 2
            continue
        if tok == '"' or tok == "'":
            j = idx + 1
            while j < n:
                ch = src[j]
                if ch == "\\" and j + 1 < n:
                    j += 2
                    continue
                if ch == tok:
                    j += 1
                    break
                j += 1
            pos = j
            continue
        if tok == "{":
            depth += 1
            pos = idx + 1
            continue
        # tok == "}"
        if depth > 0:
            depth -= 1
            if depth == 0 and active:
                name, body_start = active.pop()
                if body_start - 1 < idx:
                    bodies[name] = (body_start, idx)
        pos = idx + 1
    return bodies


def _build_function_index(
    cpp_files: list[Path],
) -> tuple[
    dict[str, tuple[Path, int, int]],  # name -> (file, body_start, body_end)
    dict[Path, str],                    # file -> source text (kept for slicing)
]:
    """One scan per file: get every function's body bounds + cache the source.

    Returns a name -> (path, start, end) map plus the file_text cache so
    later body slicing is just ``text[start:end]``.
    """
    out: dict[str, tuple[Path, int, int]] = {}
    cache: dict[Path, str] = {}
    for path in cpp_files:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cache[path] = src
        for name, (s, e) in _index_file_functions(src).items():
            out[name] = (path, s, e)
    return out, cache


def _compute_items(body: _FuncBody) -> list[_Item]:
    """Extract the source-ordered sequence of reads and call-statements."""
    items: list[_Item] = []
    events: list[tuple[int, str, str]] = []
    for m in _READ_CALL_RE.finditer(body.text):
        events.append((m.start(), "read", m.group(1)))
    for m in _INVOKE_RE.finditer(body.text):
        events.append((m.start(), "call", m.group(1)))
    events.sort(key=lambda t: t[0])
    for _, kind, val in events:
        if kind == "read":
            items.append(_Item(kind="read", size=int(val), callee=None))
        else:
            items.append(_Item(kind="call", size=None, callee=val))
    return items


# --- The map builder + bytes extractor --------------------------------------

# Function-name prefixes that wrap an ApiObjectId we want to track.
# value -> (regex_prefix, key_in_map)
_TRACKED_PREFIXES = {
    "CreateAndInitResource_": "resource",
    "CreatePipelineState_": "pso",
}


@dataclass
class ResourceChunk:
    """Pointer to one resource's compressed bytes in resources.bin."""

    resource_id: int
    file_offset: int            # byte offset into resources.bin
    compressed_size: int        # bytes to read
    source_function: str        # e.g. "CreateAndInitResource_2362"
    source_file: str            # which .cpp the function lives in (best-effort)
    # Index of the read within source_function (0 = first read in the body).
    read_index_in_func: int = 0


@dataclass
class ResourceBin:
    """All-in-one view of the resource-bytes side of a C++ export."""

    export_dir: Path
    bin_path: Path
    # resource_id -> list of chunks (sometimes a Create function does several
    # Reads in one body — e.g. a texture with multiple subresources).
    chunks_by_resource: dict[int, list[ResourceChunk]] = field(default_factory=dict)
    # pso_id -> list of chunks (the shader bytecode blob for a PSO, which the
    # generated C++ slices into VS/PS/CS sub-ranges via pssDesc.<stage>={ &data[off], len }).
    chunks_by_pso: dict[int, list[ResourceChunk]] = field(default_factory=dict)
    # Total bytes the static walk accounted for (helps debug mismatches with
    # the actual file size).
    total_reads_walked: int = 0
    # Functions we couldn't resolve (called from CreateAppResources_000 but
    # whose body wasn't in any scanned .cpp). These are usually
    # ``ResetCommandAllocators``, std-lib things, etc.
    unresolved_callees: list[str] = field(default_factory=list)

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_export(
        cls,
        export_dir: Path,
        *,
        root_function: str = "CreateAppResources_000",
    ) -> "ResourceBin":
        export_dir = Path(export_dir).resolve()
        bin_path = export_dir / "resources.bin"
        if not bin_path.is_file():
            raise FileNotFoundError(
                f"resources.bin not found in {export_dir}; the export may have "
                "been generated without binary data, or path is wrong."
            )
        # Single O(n) pass per file: get function-body bounds + cached text.
        cpp_files = sorted(export_dir.glob("*.cpp"))
        index, file_text_cache = _build_function_index(cpp_files)

        if root_function not in index:
            raise RuntimeError(
                f"root function {root_function!r} not found in export. "
                "Pass the correct entry-point name via root_function="
            )

        # Body items computed lazily per function — most exports won't visit
        # every indexed function, so we only do _compute_items for the
        # transitive callees of root_function.
        body_cache: dict[str, _FuncBody | None] = {}

        def load_body(name: str) -> _FuncBody | None:
            if name in body_cache:
                return body_cache[name]
            loc = index.get(name)
            if loc is None:
                body_cache[name] = None
                return None
            path, start, end = loc
            text = file_text_cache[path]
            body_str = text[start:end]
            body = _FuncBody(name=name, text=body_str)
            body.items = _compute_items(body)
            body.text = ""  # release once items are computed
            body_cache[name] = body
            return body

        # Walk in execution order, accumulating byte offsets.
        cursor = 0
        chunks: dict[int, list[ResourceChunk]] = {}
        pso_chunks: dict[int, list[ResourceChunk]] = {}
        unresolved: list[str] = []
        visited: set[str] = set()

        def walk(func_name: str) -> None:
            nonlocal cursor
            if func_name in visited:
                # PIX's init graph is a tree in practice (each per-object init
                # is called once); recursion here would be a bug, but guarding
                # is cheap.
                return
            visited.add(func_name)
            body = load_body(func_name)
            if body is None:
                unresolved.append(func_name)
                return
            for item in body.items:
                if item.kind == "read":
                    size = item.size or 0
                    # Tag the read against the current function name if it
                    # matches a tracked prefix. The destination map depends on
                    # which prefix matched: CreateAndInitResource_* → resources,
                    # CreatePipelineState_* → PSOs (their reads are shader blobs).
                    tracked_id: int | None = None
                    tracked_kind: str | None = None
                    for prefix, kind in _TRACKED_PREFIXES.items():
                        if func_name.startswith(prefix):
                            try:
                                tracked_id = int(func_name[len(prefix) :])
                                tracked_kind = kind
                            except ValueError:
                                tracked_id = None
                            break
                    if tracked_id is not None and tracked_kind is not None:
                        dest = pso_chunks if tracked_kind == "pso" else chunks
                        loc = index.get(func_name)
                        src_file = loc[0].name if loc is not None else ""
                        chunk = ResourceChunk(
                            resource_id=tracked_id,
                            file_offset=cursor,
                            compressed_size=size,
                            source_function=func_name,
                            source_file=src_file,
                            read_index_in_func=sum(
                                1 for c in dest.get(tracked_id, [])
                                if c.source_function == func_name
                            ),
                        )
                        dest.setdefault(tracked_id, []).append(chunk)
                    cursor += size
                else:
                    callee = item.callee or ""
                    if callee:
                        walk(callee)

        walk(root_function)

        return cls(
            export_dir=export_dir,
            bin_path=bin_path,
            chunks_by_resource=chunks,
            chunks_by_pso=pso_chunks,
            total_reads_walked=cursor,
            unresolved_callees=sorted(set(unresolved)),
        )

    # ---- public API -----------------------------------------------------

    def list_resources(self) -> list[int]:
        """Resource IDs we have initial-bytes for, sorted."""
        return sorted(self.chunks_by_resource.keys())

    def chunks(self, resource_id: int) -> list[ResourceChunk]:
        return list(self.chunks_by_resource.get(resource_id, []))

    def read_chunk(self, chunk: ResourceChunk) -> bytes:
        """Decompress one chunk into its full uncompressed bytes."""
        with self.bin_path.open("rb") as f:
            f.seek(chunk.file_offset)
            compressed = f.read(chunk.compressed_size)
        if len(compressed) != chunk.compressed_size:
            raise IOError(
                f"short read at offset {chunk.file_offset}: "
                f"asked {chunk.compressed_size} got {len(compressed)}"
            )
        return xpress_decompress(compressed)

    def read_resource_bytes(
        self,
        resource_id: int,
        *,
        offset: int = 0,
        length: int | None = None,
        chunk_index: int = 0,
    ) -> bytes:
        """Get the (offset, length) slice of resource's uncompressed bytes.

        If the resource has multiple Read calls in its initializer (e.g. a
        texture with several subresources), ``chunk_index`` picks which one.
        """
        chunk_list = self.chunks_by_resource.get(resource_id)
        if not chunk_list:
            raise KeyError(
                f"resource {resource_id} not tracked (no "
                f"CreateAndInitResource_{resource_id} found in export). "
                f"Resources we know about: {len(self.chunks_by_resource)} total"
            )
        if chunk_index < 0 or chunk_index >= len(chunk_list):
            raise IndexError(
                f"chunk_index {chunk_index} out of range for resource "
                f"{resource_id} (has {len(chunk_list)} chunk(s))"
            )
        decompressed = self.read_chunk(chunk_list[chunk_index])
        if offset < 0 or offset > len(decompressed):
            raise ValueError(
                f"offset {offset} outside decompressed size {len(decompressed)}"
            )
        if length is None:
            return decompressed[offset:]
        return decompressed[offset : offset + length]
