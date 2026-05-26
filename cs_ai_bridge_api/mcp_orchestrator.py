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
    """Validate against schema, then execute MCP tool locally."""
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
    return {
        "name": tool_name,
        "arguments": args,
        "result": result,
    }


def extract_function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    output = response.get("output")
    if not isinstance(output, list):
        return calls
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(item)
    return calls


def extract_response_text(response: dict[str, Any]) -> str:
    top = response.get("output_text")
    if isinstance(top, str) and top:
        return top

    texts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                texts.append(block["refusal"])
    return "".join(texts)


def _tool_output_payload(execution: dict[str, Any]) -> str:
    if "error" in execution:
        try:
            return json.dumps({"error": execution["error"]}, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(execution["error"])
    payload = execution.get("result")
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(payload)


def _function_call_output_items(pairs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call_id, execution in pairs:
        if not call_id:
            continue
        items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _tool_output_payload(execution),
            }
        )
    return items


async def _post_responses(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(url, json=payload, headers=headers)
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
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
    extra_instructions: str | None = None,
) -> OrchestratorRun:
    """
    OpenAI Responses tool loop:
    load MCP tools → OpenAI function tools → model calls → gateway executes MCP → final answer.
    """
    from cs_ai_bridge_api.llm_providers import (
        _messages_to_responses_input,
        _responses_to_chat_completion_shape,
    )

    schema = load_tenant_schema(tenant)
    tools = await build_openai_function_tools(request_id, schema)

    instructions, input_items = _messages_to_responses_input(messages)
    if not input_items:
        raise HTTPException(
            status_code=400,
            detail="At least one non-system user or assistant message is required.",
        )

    schema_instructions = schema_summary_for_instructions(schema)
    instruction_parts = [schema_instructions]
    if instructions:
        instruction_parts.append(instructions)
    if extra_instructions:
        instruction_parts.append(extra_instructions)
    merged_instructions = "\n\n".join(part for part in instruction_parts if part.strip())

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
            payload = dict(base_payload)
            payload["input"] = current_input
            if previous_response_id:
                payload["previous_response_id"] = previous_response_id

            logger.info(
                "mcp_orchestrator_round request_id=%s round=%s previous_response_id=%s",
                request_id,
                round_index + 1,
                previous_response_id or "<none>",
            )

            last_response = await _post_responses(
                client,
                url=url,
                headers=headers,
                payload=payload,
            )
            previous_response_id = str(last_response.get("id", "")) or None

            function_calls = extract_function_calls(last_response)
            if not function_calls:
                break

            round_executions: list[dict[str, Any]] = []
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
                    round_executions.append(execution)
                    if call_id:
                        output_pairs.append((call_id, execution))
                    continue

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
                round_executions.append(execution)
                if call_id:
                    output_pairs.append((call_id, execution))

            all_executions.extend(round_executions)
            output_items = _function_call_output_items(output_pairs)
            if not output_items:
                logger.warning(
                    "mcp_orchestrator_no_outputs request_id=%s round=%s",
                    request_id,
                    round_index + 1,
                )
                break

            current_input = output_items

    shaped = _responses_to_chat_completion_shape(last_response, model)
    shaped["mcp_tool_executions"] = all_executions
    shaped["mcp_orchestrator_rounds"] = rounds_completed
    shaped["answer_source"] = "openai_responses_orchestrator"

    if rounds_completed >= max_tool_rounds() and extract_function_calls(last_response):
        logger.warning(
            "mcp_orchestrator_max_rounds request_id=%s rounds=%s",
            request_id,
            rounds_completed,
        )
        shaped["orchestrator_warning"] = "max_tool_rounds_reached"

    return OrchestratorRun(
        response=shaped,
        tool_executions=all_executions,
        rounds=rounds_completed,
    )
