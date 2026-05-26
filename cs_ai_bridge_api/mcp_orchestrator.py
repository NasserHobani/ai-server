"""Internal MCP orchestrator for OpenAI Responses API (gateway-managed tools)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import HTTPException

from cs_ai_bridge_api.mcp_client import call_mcp_tool
from cs_ai_bridge_api.mcp_tools import build_openai_function_tools, schema_summary_for_instructions
from cs_ai_bridge_api.openai_responses import (
    extract_function_calls,
    messages_to_responses_input,
    to_chat_completion_shape,
)
from cs_ai_bridge_api.schema_redis import normalize_tenant, read_schema_metadata
from cs_ai_bridge_api.schema_validate import validate_tool_call


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorRun:
    response: dict[str, Any]
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0


def orchestrator_enabled() -> bool:
    raw = os.getenv("CS_AI_BRIDGE_MCP_ORCHESTRATOR", "true").strip().lower()
    return raw not in {"0", "false", "no"}


def max_tool_rounds() -> int:
    try:
        value = int(os.getenv("CS_AI_BRIDGE_MCP_ORCHESTRATOR_MAX_ROUNDS", "8"))
    except ValueError:
        value = 8
    return max(1, min(value, 32))


def load_tenant_schema(tenant: str | None) -> dict[str, Any]:
    normalized = normalize_tenant(tenant)
    if not normalized:
        raise ValueError("Tenant is required for MCP orchestration.")
    return read_schema_metadata(normalized)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Tool arguments must be a JSON object.")
    return {}


def _inject_tenant(arguments: dict[str, Any], tenant: str | None) -> dict[str, Any]:
    out = dict(arguments)
    if tenant and not out.get("tenant"):
        out["tenant"] = tenant
    return out


async def execute_validated_mcp_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
    tenant: str | None,
    request_id: str,
) -> dict[str, Any]:
    normalized_tenant = normalize_tenant(tenant)
    args = _inject_tenant(arguments, normalized_tenant)
    validate_tool_call(tool_name, args, schema, normalized_tenant)

    logger.info(
        "mcp_orchestrator_execute request_id=%s tool=%s tenant=%s keys=%s",
        request_id,
        tool_name,
        normalized_tenant,
        ",".join(sorted(args.keys())),
    )
    result = await call_mcp_tool(tool_name, args, request_id)
    return {"name": tool_name, "arguments": args, "result": result}


def _tool_output_payload(execution: dict[str, Any]) -> str:
    if "error" in execution:
        return json.dumps({"error": execution["error"]}, ensure_ascii=False, default=str)
    return json.dumps(execution.get("result"), ensure_ascii=False, default=str)


def _function_call_output_items(pairs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": _tool_output_payload(execution),
        }
        for call_id, execution in pairs
        if call_id
    ]


async def _post_responses(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(url, json=payload, headers=headers)
    if "application/json" not in response.headers.get("content-type", ""):
        raise HTTPException(
            status_code=502,
            detail={
                "provider": "openai",
                "message": f"Upstream returned non-JSON (status {response.status_code}).",
                "body": response.text[:4000],
            },
        )
    data = response.json()
    if response.is_error:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "provider": "openai",
                "message": f"Upstream OpenAI returned status {response.status_code}.",
                "body": data,
            },
        )
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="OpenAI returned unexpected JSON root.")
    return data


async def run_openai_responses_with_mcp(
    *,
    url: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    tenant: str | None,
    request_id: str,
    timeout: float,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> OrchestratorRun:
    schema = load_tenant_schema(tenant)
    tools = await build_openai_function_tools(request_id, schema)

    instructions, input_items = messages_to_responses_input(messages)
    if not input_items:
        raise HTTPException(
            status_code=400,
            detail="At least one non-system user or assistant message is required.",
        )

    merged_instructions = schema_summary_for_instructions(schema)
    if instructions:
        merged_instructions = f"{merged_instructions}\n\n{instructions}"

    base_payload: dict[str, Any] = {
        "model": model,
        "store": False,
        "tools": tools,
        "instructions": merged_instructions,
    }
    if temperature is not None:
        base_payload["temperature"] = temperature
    if max_output_tokens is not None:
        base_payload["max_output_tokens"] = max_output_tokens

    all_executions: list[dict[str, Any]] = []
    previous_response_id: str | None = None
    current_input: Any = input_items
    last_response: dict[str, Any] = {}
    rounds_completed = 0

    async with httpx.AsyncClient(timeout=timeout) as client:
        for round_index in range(max_tool_rounds()):
            rounds_completed = round_index + 1
            payload = {**base_payload, "input": current_input}
            if previous_response_id:
                payload["previous_response_id"] = previous_response_id

            logger.info(
                "mcp_orchestrator_round request_id=%s round=%s",
                request_id,
                rounds_completed,
            )

            last_response = await _post_responses(client, url=url, headers=headers, payload=payload)
            previous_response_id = str(last_response.get("id", "")) or None

            function_calls = extract_function_calls(last_response)
            if not function_calls:
                break

            output_pairs: list[tuple[str, dict[str, Any]]] = []
            for call in function_calls:
                call_id = str(call.get("call_id", "")).strip()
                tool_name = str(call.get("name", "")).strip()
                if not tool_name:
                    continue
                try:
                    args = _parse_tool_arguments(call.get("arguments"))
                except (json.JSONDecodeError, ValueError) as exc:
                    execution = {
                        "name": tool_name,
                        "arguments": call.get("arguments"),
                        "error": str(exc),
                    }
                else:
                    try:
                        execution = await execute_validated_mcp_tool(
                            tool_name=tool_name,
                            arguments=args,
                            schema=schema,
                            tenant=tenant,
                            request_id=request_id,
                        )
                    except (ValueError, HTTPException) as exc:
                        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                        execution = {
                            "name": tool_name,
                            "arguments": args,
                            "error": detail,
                        }
                all_executions.append(execution)
                if call_id:
                    output_pairs.append((call_id, execution))

            output_items = _function_call_output_items(output_pairs)
            if not output_items:
                break
            current_input = output_items

    shaped = to_chat_completion_shape(last_response, model)
    shaped["mcp_tool_executions"] = all_executions
    shaped["mcp_orchestrator_rounds"] = rounds_completed
    shaped["answer_source"] = "openai_responses_orchestrator"
    if rounds_completed >= max_tool_rounds() and extract_function_calls(last_response):
        shaped["orchestrator_warning"] = "max_tool_rounds_reached"

    return OrchestratorRun(
        response=shaped,
        tool_executions=all_executions,
        rounds=rounds_completed,
    )
