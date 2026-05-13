"""FastAPI service: external chat requests with model config loaded from Redis."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import Any, AsyncIterator
import uuid

from cs_ai_bridge_api.llm_config_redis import llm_api_key, read_ai_runtime_config, redis_url
from cs_ai_bridge_api.llm_providers import merge_request_body, upstream_chat_completion
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
    """OpenAI-style chat body; ``tenant`` selects Redis config key when set.

    Any additional fields are forwarded to the upstream API (OpenAI) or mapped
    for Gemini (``temperature``, ``max_tokens``, ``messages``).
    """

    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]]
    tenant: str | None = None
    provider: str | None = None

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
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if not redis_url():
        raise HTTPException(status_code=503, detail="Redis URL not configured.")
    return {"status": "ready"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    logger.info(
        "chat_completion_request request_id=%s provider=%s tenant_set=%s messages=%s",
        request_id,
        req.provider or "<auto>",
        bool(req.tenant),
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

    body = req.model_dump(exclude_none=True)
    merged = merge_request_body(body, redis_cfg)
    logger.info(
        "chat_completion_config_loaded request_id=%s source=%s provider=%s model=%s",
        request_id,
        redis_cfg.get("_config_source", "<unknown>"),
        redis_cfg.get("provider", "openai"),
        merged.get("model"),
    )
    try:
        result = await upstream_chat_completion(redis_cfg, merged, llm_api_key())
    except HTTPException as exc:
        logger.warning(
            "chat_completion_failed request_id=%s status_code=%s detail=%s",
            request_id,
            exc.status_code,
            _safe_detail(exc.detail),
        )
        raise

    logger.info(
        "chat_completion_success request_id=%s provider=%s model=%s",
        request_id,
        result.get("provider", redis_cfg.get("provider", "openai")),
        result.get("model", merged.get("model")),
    )
    return result
