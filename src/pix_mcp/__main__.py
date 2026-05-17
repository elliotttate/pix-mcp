"""Console entrypoint: ``pix-mcp`` (or ``python -m pix_mcp``).

By default runs the MCP server on stdio. Pass ``--http``/``--sse`` for
alternate transports (delegated to the MCP SDK).
"""

from __future__ import annotations

import argparse
import sys

from .server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pix-mcp", description="MCP server for Microsoft PIX (pixtool.exe).")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport (default: stdio, for use with Claude Desktop / Claude Code).",
    )
    args = parser.parse_args(argv)
    try:
        serve(transport=args.transport)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
