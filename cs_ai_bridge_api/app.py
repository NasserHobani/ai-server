"""FastAPI service: external chat requests with model config loaded from Redis."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from cs_ai_bridge_api.llm_config_redis import llm_api_key, read_ai_runtime_config, redis_url
from cs_ai_bridge_api.llm_providers import merge_request_body, upstream_chat_completion
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator


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
    try:
        redis_cfg = read_ai_runtime_config(req.tenant, req.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    body = req.model_dump(exclude_none=True)
    merged = merge_request_body(body, redis_cfg)
    return await upstream_chat_completion(redis_cfg, merged, llm_api_key())
