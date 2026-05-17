"""Example: launch a D3D12 app, take a capture, index it, print a summary.

Usage:
    python examples/01_capture_and_summarize.py "C:/Path/To/MyGame.exe" out.wpix

This drives pix_mcp's tools directly (without going through the MCP transport)
to demonstrate the end-to-end flow. The same calls are exposed as MCP tools
to any LLM agent connected to the server.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


async def main() -> int:
    if len(sys.argv) < 3:
        print("usage: 01_capture_and_summarize.py <exe> <output.wpix>")
        return 2
    exe = sys.argv[1]
    out_wpix = sys.argv[2]

    from pix_mcp.server import (
        pix_capture_launched,
        pix_capture_summary,
        pix_index_events,
        pix_open_capture,
    )

    print(f"Launching {exe} under PIX and capturing 1 frame…")
    r = await pix_capture_launched(exe=exe, output_wpix=out_wpix, frames=1)
    if not r["result"]["ok"]:
        print("Capture failed:")
        print(r["result"]["stderr_tail"])
        return 1

    print(f"Opening {out_wpix}…")
    sess = await pix_open_capture(out_wpix)
    handle = sess["handle"]

    print("Indexing event list…")
    idx = await pix_index_events(capture=handle)
    print(
        f"  events={idx['index']['event_count']} "
        f"markers={idx['index']['marker_count']} "
        f"queues={idx['index']['queue_count']}"
    )

    print("Summary:")
    s = await pix_capture_summary(capture=handle, quick=False)
    print(f"  event types: {s['event_type_counts']}")
    print(f"  queues: {[q['name'] for q in s['queues']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
