"""CLI / Docker entry: re-export `mcp` for `fastmcp run ...:mcp` and run.

`fastmcp run cs_ai_bridge_mcp/server.py:mcp` loads this file without package
context, so relative imports fail. Ensure the project root is on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_rp = str(_root)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

from cs_ai_bridge_mcp.app import mcp


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    mcp.run()


if __name__ == "__main__":
    main()
