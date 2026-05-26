"""MCP client helpers used by the AI API."""

from __future__ import annotations

import logging
import os
from typing import Any

from cs_ai_bridge_api.mcp_format import unwrap_tool_payload
from fastmcp import Client


logger = logging.getLogger(__name__)


def mcp_url() -> str:
    return os.getenv("CS_AI_BRIDGE_MCP_URL", "http://cs-ai-bridge-mcp:8000/mcp").strip()


def mcp_timeout_seconds() -> float:
    return float(os.getenv("CS_AI_BRIDGE_MCP_TIMEOUT", "30"))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if type(value).__name__ == "CallToolResult":
        raw = {
            "is_error": bool(getattr(value, "is_error", False)),
            "structured_content": _jsonable(getattr(value, "structured_content", None)),
            "data": _jsonable(getattr(value, "data", None)),
            "content": _jsonable_content(getattr(value, "content", None)),
            "meta": _jsonable(getattr(value, "meta", None)),
        }
        return unwrap_tool_payload(raw)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "text") and isinstance(getattr(value, "text"), str):
        return getattr(value, "text")
    return str(value)


def _jsonable_content(content: Any) -> Any:
    if isinstance(content, list):
        blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(block)
            elif hasattr(block, "text"):
                blocks.append(
                    {
                        "type": getattr(block, "type", "text"),
                        "text": getattr(block, "text", ""),
                    }
                )
            else:
                blocks.append(_jsonable(block))
        return blocks
    return _jsonable(content)


async def call_mcp_tools(
    calls: list[dict[str, Any]],
    request_id: str,
) -> list[dict[str, Any]]:
    url = mcp_url()
    timeout = mcp_timeout_seconds()
    logger.info(
        "mcp_tool_calls_start request_id=%s url=%s count=%s",
        request_id,
        url,
        len(calls),
    )

    results: list[dict[str, Any]] = []
    async with Client(url, timeout=timeout) as client:
        for call in calls:
            name = str(call.get("name", "")).strip()
            if not name:
                raise ValueError("Each MCP tool call must include a non-empty 'name'.")
            arguments = call.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError(f"MCP tool '{name}' arguments must be a JSON object.")

            logger.info(
                "mcp_tool_call request_id=%s tool=%s argument_keys=%s",
                request_id,
                name,
                ",".join(sorted(arguments.keys())),
            )
            result = await client.call_tool(name, arguments)
            results.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": _jsonable(result),
                }
            )

    logger.info("mcp_tool_calls_done request_id=%s count=%s", request_id, len(results))
    return results


async def list_mcp_tool_definitions(request_id: str) -> list[dict[str, Any]]:
    """List raw MCP tool definitions from the MCP server."""
    url = mcp_url()
    timeout = mcp_timeout_seconds()
    logger.info("mcp_list_tools_start request_id=%s url=%s", request_id, url)

    async with Client(url, timeout=timeout) as client:
        tools = await client.list_tools()

    out: list[dict[str, Any]] = []
    for tool in tools:
        name = str(getattr(tool, "name", "")).strip()
        if not name:
            continue
        input_schema = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}}
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        out.append(
            {
                "name": name,
                "description": str(getattr(tool, "description", "") or ""),
                "inputSchema": input_schema,
            }
        )

    logger.info("mcp_list_tools_done request_id=%s count=%s", request_id, len(out))
    return out


async def call_mcp_tool(name: str, arguments: dict[str, Any], request_id: str) -> Any:
    """Execute a single MCP tool and return the normalized result payload."""
    results = await call_mcp_tools([{"name": name, "arguments": arguments}], request_id)
    if not results:
        return None
    return results[0].get("result")
