"""Example: list the most expensive draws in a capture by GPU Duration."""

from __future__ import annotations

import asyncio
import json
import sys


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: 03_top_expensive_draws.py <wpix>")
        return 2
    wpix = sys.argv[1]

    from pix_mcp.server import pix_index_events, pix_open_capture, pix_top_by_counter

    sess = await pix_open_capture(wpix)
    handle = sess["handle"]

    print("Indexing with GPU Duration counter…")
    idx = await pix_index_events(capture=handle, counters=["GPU Duration"])
    cols = idx["index"]["counter_columns"]
    print(f"  counter columns indexed: {cols}")

    # The column name will be the snake_case of "GPU Duration" → "gpu_duration".
    print("Top 20 most expensive draws:")
    top = await pix_top_by_counter(capture=handle, counter="gpu_duration", n=20, event_type="draw")
    for row in top["rows"]:
        print(f"  gid={row['global_id']:>6}  {row['counter_value']:.4f}  {row['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
