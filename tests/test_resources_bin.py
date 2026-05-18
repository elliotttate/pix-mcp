"""Tests for the resources.bin extractor.

Builds a minimal synthetic export-to-cpp directory with a real (XPRESS-
compressed) ``resources.bin`` and exercises the static call-graph walk that
maps ``CreateAndInitResource_<id>`` functions onto file offsets, then
decompresses + slices the bytes.

XPRESS decompression goes through the Windows Cabinet API via ctypes, so
these tests only run on Windows. Skipped elsewhere.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import sys
import textwrap
from pathlib import Path

import pytest

if sys.platform != "win32":
    pytest.skip("resources.bin requires Windows Cabinet API", allow_module_level=True)

from pix_mcp import resources_bin


# --- helper: compress with the same Windows API our reader decompresses with -

_COMPRESS_ALGORITHM_XPRESS = 3
_ERROR_INSUFFICIENT_BUFFER = 122


def _xpress_compress(data: bytes) -> bytes:
    cab = ctypes.WinDLL("cabinet.dll")
    CreateCompressor = cab.CreateCompressor
    CreateCompressor.argtypes = [wt.DWORD, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    CreateCompressor.restype = wt.BOOL
    Compress = cab.Compress
    Compress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    Compress.restype = wt.BOOL
    CloseCompressor = cab.CloseCompressor
    CloseCompressor.argtypes = [ctypes.c_void_p]
    CloseCompressor.restype = wt.BOOL

    h = ctypes.c_void_p()
    if not CreateCompressor(_COMPRESS_ALGORITHM_XPRESS, None, ctypes.byref(h)):
        raise OSError(ctypes.get_last_error(), "CreateCompressor failed")
    try:
        src = (ctypes.c_ubyte * len(data)).from_buffer_copy(data) if data else None
        # Probe required size.
        out_size = ctypes.c_size_t(0)
        Compress(h, src, len(data), None, 0, ctypes.byref(out_size))
        # On insufficient-buffer the API populates out_size; ignore returned BOOL.
        if out_size.value == 0:
            # Empty input edge case.
            return b""
        buf = (ctypes.c_ubyte * out_size.value)()
        actual = ctypes.c_size_t(0)
        ok = Compress(h, src, len(data), buf, out_size.value, ctypes.byref(actual))
        if not ok:
            raise OSError(ctypes.get_last_error(), "Compress failed")
        return bytes(buf[: actual.value])
    finally:
        CloseCompressor(h)


def _build_export(tmp_path: Path) -> tuple[Path, dict[int, bytes]]:
    """Create a synthetic export dir mirroring PIX's structure."""
    resources = {
        5: b"\x00" * 64,                                    # zeros
        7: bytes(range(256)) * 4,                            # 1024 bytes pattern
        2362: b"HELLO_CBUFFER_DATA_" + (b"X" * 1024) + b"_END",  # multi-block
    }
    # Build resources.bin by compressing each blob in invocation order.
    bin_chunks: list[tuple[int, bytes]] = []  # (resource_id, compressed)
    for rid in (5, 7, 2362):
        bin_chunks.append((rid, _xpress_compress(resources[rid])))
    (tmp_path / "resources.bin").write_bytes(b"".join(c for _, c in bin_chunks))

    # CreateAppResources_000 dispatcher
    init_cpp = textwrap.dedent(
        f"""\
        #include "pch.h"

        void CreateAndInitResource_5()
        {{
            std::vector<BYTE> uncompressedData;
            g_resourceReader->Read(uncompressedData, {len(bin_chunks[0][1])});
        }}

        void CreateAndInitResource_7()
        {{
            std::vector<BYTE> uncompressedData;
            g_resourceReader->Read(uncompressedData, {len(bin_chunks[1][1])});
        }}

        void CreateAndInitResource_2362()
        {{
            std::vector<BYTE> uncompressedData;
            g_resourceReader->Read(uncompressedData, {len(bin_chunks[2][1])});
        }}

        void CreateAppResources_000()
        {{
            CreateAndInitResource_5();
            CreateAndInitResource_7();
            CreateAndInitResource_2362();
        }}
        """
    )
    (tmp_path / "FrameResources_000.cpp").write_text(init_cpp, encoding="utf-8")
    return tmp_path, resources


def test_round_trip_three_resources(tmp_path: Path) -> None:
    export_dir, originals = _build_export(tmp_path)
    rb = resources_bin.ResourceBin.from_export(export_dir)
    assert sorted(rb.list_resources()) == [5, 7, 2362]
    # File offsets must be cumulative compressed sizes.
    chunks_5 = rb.chunks(5)
    chunks_7 = rb.chunks(7)
    chunks_2362 = rb.chunks(2362)
    assert chunks_5[0].file_offset == 0
    assert chunks_7[0].file_offset == chunks_5[0].compressed_size
    assert chunks_2362[0].file_offset == (
        chunks_5[0].compressed_size + chunks_7[0].compressed_size
    )
    # Round-trip every resource.
    for rid, original in originals.items():
        got = rb.read_resource_bytes(rid, offset=0, length=None)
        assert got == original, f"mismatch on resource {rid}"


def test_offset_and_length_slicing(tmp_path: Path) -> None:
    export_dir, originals = _build_export(tmp_path)
    rb = resources_bin.ResourceBin.from_export(export_dir)
    # Resource 2362 starts with "HELLO_CBUFFER_DATA_"
    sl = rb.read_resource_bytes(2362, offset=0, length=19)
    assert sl == b"HELLO_CBUFFER_DATA_"
    end = rb.read_resource_bytes(2362, offset=len(originals[2362]) - 4, length=4)
    assert end == b"_END"


def test_unknown_resource_raises(tmp_path: Path) -> None:
    export_dir, _ = _build_export(tmp_path)
    rb = resources_bin.ResourceBin.from_export(export_dir)
    with pytest.raises(KeyError):
        rb.read_resource_bytes(42, offset=0, length=4)
