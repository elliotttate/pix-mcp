"""Example: answer 'what's bound at root param N of event G?' for an existing
capture, by exporting to C++ and replaying state.

Usage:
    python examples/02_resource_at_root_param.py path/to/capture.wpix 2372 0
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


async def main() -> int:
    if len(sys.argv) < 4:
        print("usage: 02_resource_at_root_param.py <wpix> <global_id> <root_param_index>")
        return 2
    wpix = sys.argv[1]
    gid = int(sys.argv[2])
    rp = int(sys.argv[3])

    from pix_mcp.server import (
        pix_export_to_cpp,
        pix_get_resource_at_root_param,
        pix_open_capture,
    )

    sess = await pix_open_capture(wpix)
    handle = sess["handle"]

    cpp_dir = Path(wpix).with_suffix(".wpix.cpp_export")
    print(f"Exporting to C++ in {cpp_dir}…")
    await pix_export_to_cpp(
        capture=handle,
        output_dir=str(cpp_dir),
        force=True,
        use_winpixeventruntime=True,
        use_agility_sdk=True,
        parse_after=True,
    )

    print(f"Querying root param {rp} at event {gid}…")
    binding = await pix_get_resource_at_root_param(
        capture=handle, global_id=gid, root_param_index=rp, pipeline="graphics"
    )
    print(json.dumps(binding, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
