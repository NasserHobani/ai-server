"""MCP client helpers used by the AI API."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastmcp import Client


logger = logging.getLogger(__name__)


def mcp_url() -> str:
    return os.getenv("CS_AI_BRIDGE_MCP_URL", "http://host.docker.internal:8071/mcp").strip()


def mcp_timeout_seconds() -> float:
    return float(os.getenv("CS_AI_BRIDGE_MCP_TIMEOUT", "30"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


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
