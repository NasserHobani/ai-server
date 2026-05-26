"""FastAPI service: external chat requests with model config loaded from Redis."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import Any, AsyncIterator
import uuid

from cs_ai_bridge_api.llm_config_redis import llm_api_key, read_ai_runtime_config, redis_url
from cs_ai_bridge_api.llm_providers import merge_request_body, upstream_chat_completion
from cs_ai_bridge_api.mcp_client import call_mcp_tools
from cs_ai_bridge_api.mcp_format import (
    ensure_assistant_text,
    format_mcp_results_for_context,
    format_mcp_results_plain,
    normalize_chat_messages,
    normalize_mcp_results_list,
)
from cs_ai_bridge_api.mcp_intent import (
    has_query_tool_call,
    infer_mcp_queries,
    last_user_message_text,
    schema_from_mcp_results,
)
from cs_ai_bridge_api.mcp_orchestrator import orchestrator_enabled
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator


logging.basicConfig(
    level=os.getenv("CS_AI_BRIDGE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_detail(detail: Any) -> str:
    return str(detail).replace("\n", " ")[:1000]


def _schema_key_prefix() -> str:
    return os.getenv("CS_AI_BRIDGE_SCHEMA_KEY_PREFIX", "cs_ai_bridge:schema").rstrip(":")


def _normalize_schema_tenant(value: str | None) -> str | None:
    """Accept tenant id or full Redis key like cs_ai_bridge:schema:jhzly."""
    token = (value or "").strip()
    if not token:
        return None
    prefix = f"{_schema_key_prefix()}:"
    if token.startswith(prefix):
        token = token[len(prefix) :].strip()
    return token or None


def _build_effective_mcp_calls(
    req: "ChatCompletionRequest",
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls = list(req.mcp_tool_calls or [])
    schema_tenant = _normalize_schema_tenant(req.tenant) or _normalize_schema_tenant(
        req.schema_key
    )
    if not schema_tenant:
        return calls

    if not any(str(call.get("name", "")).strip() == "get_schema_metadata" for call in calls):
        calls.insert(
            0,
            {
                "name": "get_schema_metadata",
                "arguments": {"tenant": schema_tenant},
            },
        )
    return calls


async def _resolve_mcp_calls_with_data(
    req: "ChatCompletionRequest",
    messages: list[dict[str, Any]],
    request_id: str,
) -> list[dict[str, Any]]:
    """Fetch schema, infer ``mcp_query`` from user text, then run all MCP tools."""
    calls = _build_effective_mcp_calls(req, messages)
    if not calls:
        return []

    schema_tenant = _normalize_schema_tenant(req.tenant) or _normalize_schema_tenant(
        req.schema_key
    )
    if not schema_tenant or has_query_tool_call(calls):
        return normalize_mcp_results_list(await call_mcp_tools(calls, request_id))

    # Phase 1: schema only (needed to pick model/fields for auto-query).
    schema_calls = [c for c in calls if str(c.get("name", "")).strip() == "get_schema_metadata"]
    other_calls = [c for c in calls if str(c.get("name", "")).strip() != "get_schema_metadata"]
    schema_results = normalize_mcp_results_list(await call_mcp_tools(schema_calls, request_id))

    auto_query = os.getenv("CS_AI_BRIDGE_AUTO_MCP_QUERY", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    schema = schema_from_mcp_results(schema_results)
    user_text = last_user_message_text(messages)
    inferred = infer_mcp_queries(user_text, schema or {}, schema_tenant) if auto_query else []
    if not inferred:
        return schema_results

    query_results = normalize_mcp_results_list(
        await call_mcp_tools(inferred, request_id)
    )
    return schema_results + query_results


class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat body; ``tenant`` selects Redis config key when set.

    Any additional fields are forwarded to the upstream API (OpenAI) or mapped
    for Gemini (``temperature``, ``max_tokens``, ``messages``).
    """

    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]]
    tenant: str | None = None
    provider: str | None = None
    assistant_key: str | None = None
    schema_key: str | None = None
    mcp_tool_calls: list[dict[str, Any]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _require_messages(cls, data: Any) -> Any:
        if isinstance(data, dict) and "messages" not in data:
            raise ValueError("Field 'messages' is required.")
        return data


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


app = FastAPI(
    title="CS AI Bridge LLM API",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "build": os.getenv("CS_AI_BRIDGE_API_BUILD_ID", "mcp-orchestrator-v1"),
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    if not redis_url():
        raise HTTPException(status_code=503, detail="Redis URL not configured.")
    return {"status": "ready"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    schema_tenant = _normalize_schema_tenant(req.tenant) or _normalize_schema_tenant(
        req.schema_key
    )
    logger.info(
        "chat_completion_request request_id=%s provider=%s tenant=%s messages=%s",
        request_id,
        req.provider or "<auto>",
        schema_tenant or "<none>",
        len(req.messages),
    )

    try:
        redis_cfg = read_ai_runtime_config(req.tenant, req.provider)
    except ValueError as exc:
        logger.warning(
            "chat_completion_config_error request_id=%s detail=%s",
            request_id,
            _safe_detail(str(exc)),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = req.model_dump(
        exclude_none=True,
        exclude={
            "mcp_tool_calls",
            "tenant",
            "provider",
            "assistant_key",
            "schema_key",
        },
    )
    if isinstance(body.get("messages"), list):
        body["messages"] = normalize_chat_messages(body["messages"])

    use_orchestrator = (
        orchestrator_enabled()
        and bool(schema_tenant)
        and not req.mcp_tool_calls
    )

    mcp_results: list[dict[str, Any]] | None = None
    effective_mcp_calls = _build_effective_mcp_calls(req, body.get("messages") or [])
    if effective_mcp_calls and not use_orchestrator:
        try:
            mcp_results = await _resolve_mcp_calls_with_data(
                req,
                body.get("messages") or [],
                request_id,
            )
        except ValueError as exc:
            logger.warning(
                "mcp_tool_calls_invalid request_id=%s detail=%s",
                request_id,
                _safe_detail(str(exc)),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning(
                "mcp_tool_calls_failed request_id=%s error=%s detail=%s",
                request_id,
                type(exc).__name__,
                _safe_detail(str(exc)),
            )
            raise HTTPException(status_code=502, detail=f"MCP tool call failed: {exc}") from exc

        mcp_context = format_mcp_results_for_context(mcp_results)
        if mcp_context.strip() and "Record 1:" not in mcp_context:
            logger.warning(
                "mcp_context_no_records request_id=%s hint=check_schema_or_odoo_query",
                request_id,
            )
        if mcp_context.strip():
            body["messages"] = [
                *body["messages"],
                {
                    "role": "system",
                    "content": mcp_context,
                },
            ]
        body["_mcp_prefetched"] = True

    merged = merge_request_body(body, redis_cfg)
    logger.info(
        "chat_completion_config_loaded request_id=%s source=%s provider=%s model=%s",
        request_id,
        redis_cfg.get("_config_source", "<unknown>"),
        redis_cfg.get("provider", "openai"),
        merged.get("model"),
    )
    try:
        result = await upstream_chat_completion(
            redis_cfg,
            merged,
            llm_api_key(),
            request_id=request_id,
            tenant=schema_tenant,
            use_mcp_orchestrator=use_orchestrator,
        )
    except HTTPException as exc:
        logger.warning(
            "chat_completion_failed request_id=%s status_code=%s detail=%s",
            request_id,
            exc.status_code,
            _safe_detail(exc.detail),
        )
        raise
    except Exception as exc:
        logger.exception(
            "chat_completion_unhandled request_id=%s error=%s detail=%s",
            request_id,
            type(exc).__name__,
            _safe_detail(str(exc)),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Unhandled server error.",
                "error": str(exc),
            },
        ) from exc

    logger.info(
        "chat_completion_success request_id=%s provider=%s model=%s",
        request_id,
        result.get("provider", redis_cfg.get("provider", "openai")),
        result.get("model", merged.get("model")),
    )

    choices = result.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else {}
    if not isinstance(message, dict):
        message = {}
    content = ensure_assistant_text(message.get("content"))
    if mcp_results is not None:
        mcp_text = format_mcp_results_plain(mcp_results)
        result["mcp_results_text"] = mcp_text
        if not content:
            content = mcp_text
            result["answer_source"] = "mcp_fallback"
        elif "[object Object]" in content:
            content = mcp_text
            result["answer_source"] = "mcp_fallback"
    elif not content and "[object Object]" in ensure_assistant_text(message.get("content")):
        content = ""

    if isinstance(choices, list) and choices:
        choices[0]["message"] = {"role": "assistant", "content": content}
    else:
        result["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    result["content"] = content

    if mcp_results is not None:
        result["mcp_results"] = mcp_results
    if use_orchestrator and result.get("mcp_tool_executions") is not None:
        result["mcp_results"] = result.get("mcp_tool_executions")
    return result
