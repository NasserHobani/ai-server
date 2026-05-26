"""FastAPI AI Gateway: OpenAI/Gemini proxy with gateway-managed MCP orchestration."""

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
    finalize_chat_response,
    format_mcp_results_text,
    normalize_chat_messages,
    normalize_mcp_results_list,
)
from cs_ai_bridge_api.mcp_orchestrator import orchestrator_enabled
from cs_ai_bridge_api.schema_redis import normalize_tenant
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator


logging.basicConfig(
    level=os.getenv("CS_AI_BRIDGE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_detail(detail: Any) -> str:
    return str(detail).replace("\n", " ")[:1000]


class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat body. ``tenant`` enables MCP orchestration + schema validation."""

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
    version="0.2.0",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "build": os.getenv("CS_AI_BRIDGE_API_BUILD_ID", "refactor-orchestrator-v2"),
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    if not redis_url():
        raise HTTPException(status_code=503, detail="Redis URL not configured.")
    return {"status": "ready"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    tenant = normalize_tenant(req.tenant) or normalize_tenant(req.schema_key)
    use_orchestrator = orchestrator_enabled() and bool(tenant) and not req.mcp_tool_calls

    logger.info(
        "chat_completion_request request_id=%s provider=%s tenant=%s orchestrator=%s",
        request_id,
        req.provider or "<auto>",
        tenant or "<none>",
        use_orchestrator,
    )

    try:
        redis_cfg = read_ai_runtime_config(req.tenant, req.provider)
    except ValueError as exc:
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

    mcp_results: list[dict[str, Any]] | None = None
    if req.mcp_tool_calls and not use_orchestrator:
        try:
            mcp_results = normalize_mcp_results_list(
                await call_mcp_tools(req.mcp_tool_calls, request_id)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"MCP tool call failed: {exc}") from exc

        context = format_mcp_results_text(mcp_results, include_instructions=True)
        if context.strip():
            body["messages"] = [
                *body["messages"],
                {"role": "system", "content": context},
            ]

    merged = merge_request_body(body, redis_cfg)
    try:
        result = await upstream_chat_completion(
            redis_cfg,
            merged,
            llm_api_key(),
            request_id=request_id,
            tenant=tenant,
            use_mcp_orchestrator=use_orchestrator,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat_completion_unhandled request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Unhandled server error.", "error": str(exc)},
        ) from exc

    logger.info(
        "chat_completion_success request_id=%s provider=%s model=%s",
        request_id,
        result.get("provider"),
        result.get("model"),
    )
    return finalize_chat_response(result, mcp_results)
