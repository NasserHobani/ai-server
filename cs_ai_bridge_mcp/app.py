"""FastMCP application instance; importing tools registers MCP handlers."""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("cs-ai-bridge")

from . import tools  # noqa: E402,F401 — side effect: register tools on mcp
