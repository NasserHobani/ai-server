"""Internal MCP orchestrator for OpenAI Responses API (gateway-managed tools)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
from cs_ai_bridge_api.schema_redis import normalize_tenant
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


def max_upstream_retries() -> int:
    try:
        value = int(os.getenv("CS_AI_BRIDGE_OPENAI_RETRY_MAX_ATTEMPTS", "2"))
    except ValueError:
        value = 2
    return max(0, min(value, 5))


def upstream_retry_backoff_seconds() -> float:
    try:
        value = float(os.getenv("CS_AI_BRIDGE_OPENAI_RETRY_BACKOFF_SECONDS", "1.5"))
    except ValueError:
        value = 1.5
    return max(0.0, min(value, 30.0))


def _coerce_schema_payload(value: Any, tenant: str) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("models"), dict):
        return value
    raise ValueError(
        f"MCP get_schema_metadata returned invalid schema payload for tenant '{tenant}'."
    )


async def load_tenant_schema(tenant: str | None, request_id: str) -> dict[str, Any]:
    normalized = normalize_tenant(tenant)
    if not normalized:
        raise ValueError("Tenant is required for MCP orchestration.")
    result = await call_mcp_tool(
        "get_schema_metadata",
        {"tenant": normalized},
        request_id,
    )
    return _coerce_schema_payload(result, normalized)


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


def _call_id_from_function_call(call: dict[str, Any]) -> str:
    for key in ("call_id", "id"):
        value = call.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sanitize_function_call_item(call: dict[str, Any]) -> dict[str, Any]:
    """Minimal function_call item for Responses ``input`` (avoids unknown/null fields)."""
    arguments = call.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)

    item: dict[str, Any] = {
        "type": "function_call",
        "name": str(call.get("name", "")),
        "arguments": arguments,
    }
    call_id = _call_id_from_function_call(call)
    if call_id:
        item["call_id"] = call_id
    return item


def _append_pre_tool_output_items(
    conversation_input: list[dict[str, Any]],
    response: dict[str, Any],
) -> None:
    """Reasoning models may require reasoning output items before tool continuations."""
    output = response.get("output")
    if not isinstance(output, list):
        return
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            break
        if item_type == "reasoning":
            conversation_input.append(item)


def _append_tool_results_to_conversation(
    conversation_input: list[dict[str, Any]],
    call_execution_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """Append function_call + function_call_output pairs (required by Responses API)."""
    for call, execution in call_execution_pairs:
        call_id = _call_id_from_function_call(call)
        if not call_id:
            logger.warning(
                "function_call missing call_id tool=%s keys=%s",
                call.get("name"),
                sorted(call.keys()),
            )
            continue
        conversation_input.append(_sanitize_function_call_item(call))
        conversation_input.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": _tool_output_payload(execution),
            }
        )


def _maybe_extract_openai_request_id(response: httpx.Response, data: Any) -> str | None:
    header_id = response.headers.get("x-request-id")
    if isinstance(header_id, str) and header_id.strip():
        return header_id.strip()
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    message = error.get("message")
    if not isinstance(message, str):
        return None
    match = re.search(r"(req_[a-zA-Z0-9]+)", message)
    if not match:
        return None
    return match.group(1)


def _is_retryable_openai_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def openai_request_log_max_chars() -> int:
    try:
        value = int(os.getenv("CS_AI_BRIDGE_OPENAI_REQUEST_LOG_MAX_CHARS", "20000"))
    except ValueError:
        value = 20000
    return max(1000, min(value, 200000))


def _serialize_payload_for_log(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    max_chars = openai_request_log_max_chars()
    if len(serialized) <= max_chars:
        return serialized
    return serialized[:max_chars] + "...<truncated>"


async def _post_responses(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    retry_max = max_upstream_retries()
    backoff = upstream_retry_backoff_seconds()

    for attempt in range(retry_max + 1):
        logger.info(
            "openai_responses_request request_id=%s attempt=%s/%s payload=%s",
            request_id,
            attempt + 1,
            retry_max + 1,
            _serialize_payload_for_log(payload),
        )
        response = await client.post(url, json=payload, headers=headers)
        is_last_attempt = attempt >= retry_max
        content_type = response.headers.get("content-type", "")
        data: Any = None

        if "application/json" in content_type:
            data = response.json()
        else:
            if _is_retryable_openai_status(response.status_code) and not is_last_attempt:
                logger.warning(
                    "openai_responses_retry_non_json status=%s attempt=%s/%s",
                    response.status_code,
                    attempt + 1,
                    retry_max + 1,
                )
                if backoff > 0:
                    await asyncio.sleep(backoff * (attempt + 1))
                continue
            raise HTTPException(
                status_code=502,
                detail={
                    "provider": "openai",
                    "message": f"Upstream returned non-JSON (status {response.status_code}).",
                    "body": response.text[:4000],
                },
            )

        if response.is_error:
            openai_request_id = _maybe_extract_openai_request_id(response, data)
            log_suffix = f" openai_request_id={openai_request_id}" if openai_request_id else ""
            logger.warning(
                "openai_responses_error status=%s attempt=%s/%s%s body=%s",
                response.status_code,
                attempt + 1,
                retry_max + 1,
                log_suffix,
                json.dumps(data, ensure_ascii=False)[:2000],
            )
            if _is_retryable_openai_status(response.status_code) and not is_last_attempt:
                if backoff > 0:
                    await asyncio.sleep(backoff * (attempt + 1))
                continue
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

    raise HTTPException(status_code=502, detail="OpenAI request retries exhausted unexpectedly.")


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
    schema = await load_tenant_schema(tenant, request_id)
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
    # Cumulative input (works with store=false). Do not use previous_response_id alone on
    # round 2+ — it often 400s when store is disabled.
    conversation_input: list[dict[str, Any]] = list(input_items)
    last_response: dict[str, Any] = {}
    rounds_completed = 0

    async with httpx.AsyncClient(timeout=timeout) as client:
        for round_index in range(max_tool_rounds()):
            rounds_completed = round_index + 1
            payload = {**base_payload, "input": conversation_input}

            logger.info(
                "mcp_orchestrator_round request_id=%s round=%s input_items=%s",
                request_id,
                rounds_completed,
                len(conversation_input),
            )

            last_response = await _post_responses(
                client,
                url=url,
                headers=headers,
                payload=payload,
                request_id=request_id,
            )

            function_calls = extract_function_calls(last_response)
            if not function_calls:
                break

            call_execution_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for call in function_calls:
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
                call_execution_pairs.append((call, execution))

            all_executions.extend([exec_ for _, exec_ in call_execution_pairs])
            _append_pre_tool_output_items(conversation_input, last_response)
            _append_tool_results_to_conversation(conversation_input, call_execution_pairs)

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
