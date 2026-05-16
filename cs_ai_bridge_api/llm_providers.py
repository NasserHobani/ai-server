"""OpenAI and Gemini upstream calls; routing uses Redis ``provider`` field."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from urllib.parse import quote

import httpx
from fastapi import HTTPException


logger = logging.getLogger(__name__)


def _safe_detail(detail: Any) -> str:
    return str(detail).replace("\n", " ")[:1000]


def normalize_provider(redis_cfg: dict[str, Any]) -> str:
    raw = redis_cfg.get("provider", "openai")
    if not isinstance(raw, str):
        return "openai"
    p = raw.strip().lower()
    if p in {"openai", "oai"}:
        return "openai"
    if p in {"gemini", "google", "google-ai"}:
        return "gemini"
    raise HTTPException(
        status_code=500,
        detail="Redis AI config 'provider' must be 'openai' or 'gemini'.",
    )


def resolve_openai_api_key(redis_cfg: dict[str, Any], env_openai: str | None) -> str | None:
    k = redis_cfg.get("api_key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    return env_openai


def resolve_gemini_api_key(redis_cfg: dict[str, Any]) -> str | None:
    k = redis_cfg.get("api_key")
    if isinstance(k, str) and k.strip():
        return k.strip()
    for name in (
        "CS_AI_BRIDGE_GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        v = os.getenv(name, "").strip()
        if v:
            return v
    return None


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _coerce_openai_metadata_value(value: Any) -> str:
    """OpenAI requires every metadata value to be a string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_openai_metadata(out: dict[str, Any]) -> None:
    # Flat keys such as metadata.user_id (common from some HTTP clients)
    for key in list(out.keys()):
        if not key.startswith("metadata."):
            continue
        subkey = key[len("metadata.") :]
        if not subkey:
            continue
        meta = out.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta[subkey] = out.pop(key)
        out["metadata"] = meta

    metadata = out.get("metadata")
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        out["metadata"] = {"value": _coerce_openai_metadata_value(metadata)}
        return
    out["metadata"] = {
        str(k): _coerce_openai_metadata_value(v) for k, v in metadata.items()
    }


def prepare_openai_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy safe to POST to OpenAI chat/completions."""
    payload = dict(body)
    _normalize_openai_metadata(payload)
    return payload


def _merge_openai_payload(
    body: dict[str, Any],
    redis_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = {k: v for k, v in body.items() if k not in {"tenant", "provider"}}
    _normalize_openai_metadata(out)
    model = out.get("model")
    if not model:
        m = redis_cfg.get("model")
        if not m or not isinstance(m, str):
            raise HTTPException(
                status_code=500,
                detail="Redis AI config must include string field 'model'.",
            )
        out["model"] = m
    if out.get("temperature") is None and redis_cfg.get("default_temperature") is not None:
        out["temperature"] = redis_cfg["default_temperature"]
    if out.get("max_tokens") is None and redis_cfg.get("default_max_tokens") is not None:
        out["max_tokens"] = redis_cfg["default_max_tokens"]
    return out


def _openai_chat_url(redis_cfg: dict[str, Any]) -> str:
    raw = redis_cfg.get("base_url") or os.getenv(
        "CS_AI_BRIDGE_LLM_BASE_URL", "https://api.openai.com/v1"
    )
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=500, detail="Invalid base_url in Redis AI config.")
    return f"{_normalize_base_url(raw.strip())}/chat/completions"


def _timeout_seconds(redis_cfg: dict[str, Any]) -> float:
    v = redis_cfg.get("timeout_seconds")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return float(os.getenv("CS_AI_BRIDGE_LLM_TIMEOUT", "120"))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    if content is None:
        return ""
    return str(content)


def _messages_to_gemini(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_chunks: list[str] = []
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip().lower()
        text = _message_text(msg.get("content"))
        if role == "system":
            if text:
                system_chunks.append(text)
            continue
        if role == "assistant":
            gemini_role = "model"
        elif role in {"user", "model"}:
            gemini_role = "user" if role == "user" else "model"
        else:
            gemini_role = "user"
            if text:
                text = f"[{role}]\n{text}"
        if not text and gemini_role == "user":
            continue
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    system_instruction = "\n\n".join(system_chunks) if system_chunks else None
    return system_instruction, contents


def _gemini_generate_url(redis_cfg: dict[str, Any], model: str, api_key: str) -> str:
    base = redis_cfg.get("base_url") or os.getenv(
        "CS_AI_BRIDGE_GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    if not isinstance(base, str) or not base.strip():
        raise HTTPException(status_code=500, detail="Invalid Gemini base_url in Redis AI config.")
    root = _normalize_base_url(base.strip())
    m = model.strip()
    if not m:
        raise HTTPException(status_code=500, detail="Gemini model name is empty.")
    return f"{root}/models/{quote(m, safe='')}:generateContent?key={quote(api_key, safe='')}"


def _gemini_body(
    redis_cfg: dict[str, Any],
    merged_openai_like: dict[str, Any],
    system_instruction: str | None,
    contents: list[dict[str, Any]],
) -> dict[str, Any]:
    gen: dict[str, Any] = {}
    t = merged_openai_like.get("temperature")
    if t is not None:
        gen["temperature"] = float(t)
    mt = merged_openai_like.get("max_tokens")
    if mt is not None:
        gen["maxOutputTokens"] = int(mt)
    if redis_cfg.get("default_temperature") is not None and "temperature" not in gen:
        gen["temperature"] = float(redis_cfg["default_temperature"])
    if redis_cfg.get("default_max_tokens") is not None and "maxOutputTokens" not in gen:
        gen["maxOutputTokens"] = int(redis_cfg["default_max_tokens"])

    body: dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if gen:
        body["generationConfig"] = gen
    return body


def _gemini_to_openai_shape(
    raw: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    text = ""
    cands = raw.get("candidates")
    if isinstance(cands, list) and cands:
        first = cands[0]
        if isinstance(first, dict):
            content = first.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict) and isinstance(p.get("text"), str):
                            text += p["text"]
    finish = "stop"
    c0 = cands[0] if isinstance(cands, list) and cands and isinstance(cands[0], dict) else {}
    fr = c0.get("finishReason") if isinstance(c0, dict) else None
    if isinstance(fr, str) and fr:
        finish = fr.lower().replace("_", "")

    usage: dict[str, int] = {}
    um = raw.get("usageMetadata")
    if isinstance(um, dict):
        if isinstance(um.get("promptTokenCount"), int):
            usage["prompt_tokens"] = um["promptTokenCount"]
        if isinstance(um.get("candidatesTokenCount"), int):
            usage["completion_tokens"] = um["candidatesTokenCount"]
        if isinstance(um.get("totalTokenCount"), int):
            usage["total_tokens"] = um["totalTokenCount"]

    return {
        "id": raw.get("responseId") or f"gemini-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "provider": "gemini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        "usage": usage or None,
    }


async def upstream_chat_completion(
    redis_cfg: dict[str, Any],
    merged_body: dict[str, Any],
    env_openai_key: str | None,
) -> dict[str, Any]:
    """``merged_body`` is OpenAI-shaped (after merge, before provider branch)."""
    provider = normalize_provider(redis_cfg)
    timeout = _timeout_seconds(redis_cfg)

    if merged_body.get("stream"):
        raise HTTPException(status_code=400, detail="Streaming is not supported; set stream=false.")

    if provider == "openai":
        api_key = resolve_openai_api_key(redis_cfg, env_openai_key)
        logger.info(
            "upstream_openai_prepare model=%s timeout=%s api_key_set=%s api_key_source=%s config_source=%s",
            merged_body.get("model"),
            timeout,
            bool(api_key),
            "redis" if isinstance(redis_cfg.get("api_key"), str) and redis_cfg.get("api_key", "").strip() else "env",
            redis_cfg.get("_config_source", "<unknown>"),
        )
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="OpenAI: set CS_AI_BRIDGE_LLM_API_KEY or OPENAI_API_KEY, or Redis api_key.",
            )
        url = _openai_chat_url(redis_cfg)
        openai_payload = prepare_openai_request_body(merged_body)
        logger.info("upstream_openai_request url=%s model=%s", url, openai_payload.get("model"))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        extra_headers = redis_cfg.get("extra_headers")
        if isinstance(extra_headers, dict):
            for hk, hv in extra_headers.items():
                headers[str(hk)] = str(hv)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                r = await client.post(url, json=openai_payload, headers=headers)
            except httpx.RequestError as exc:
                logger.warning("upstream_openai_request_error error=%s detail=%s", type(exc).__name__, exc)
                raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

        ct = r.headers.get("content-type", "")
        logger.info("upstream_openai_response status_code=%s content_type=%s", r.status_code, ct)
        if "application/json" not in ct:
            logger.warning(
                "upstream_openai_non_json status_code=%s body=%s",
                r.status_code,
                r.text[:500],
            )
            raise HTTPException(
                status_code=502,
                detail=f"Upstream returned non-JSON ({r.status_code}): {r.text[:500]}",
            )
        try:
            data = r.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Upstream response was not valid JSON.") from exc
        if r.is_error:
            detail = data if isinstance(data, dict) else {"error": str(data)}
            logger.warning(
                "upstream_openai_error status_code=%s detail=%s",
                r.status_code,
                _safe_detail(detail),
            )
            raise HTTPException(status_code=r.status_code, detail=detail)
        if isinstance(data, dict):
            data.setdefault("provider", "openai")
        return data

    # Gemini
    api_key = resolve_gemini_api_key(redis_cfg)
    logger.info(
        "upstream_gemini_prepare model=%s timeout=%s api_key_set=%s api_key_source=%s config_source=%s",
        merged_body.get("model"),
        timeout,
        bool(api_key),
        "redis" if isinstance(redis_cfg.get("api_key"), str) and redis_cfg.get("api_key", "").strip() else "env",
        redis_cfg.get("_config_source", "<unknown>"),
    )
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini: set CS_AI_BRIDGE_GEMINI_API_KEY (or GOOGLE_API_KEY), or Redis api_key.",
        )
    model = merged_body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(status_code=500, detail="Gemini requires Redis or request field 'model'.")
    messages = merged_body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list.")

    sys_inst, contents = _messages_to_gemini(messages)
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="After conversion, no Gemini contents remained (check roles/content).",
        )

    gemini_payload = _gemini_body(redis_cfg, merged_body, sys_inst, contents)
    url = _gemini_generate_url(redis_cfg, model, api_key)
    safe_url = url.split("?key=", 1)[0] + "?key=<redacted>"
    logger.info("upstream_gemini_request url=%s model=%s", safe_url, model)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(
                url,
                json=gemini_payload,
                headers={"Content-Type": "application/json"},
            )
        except httpx.RequestError as exc:
            logger.warning("upstream_gemini_request_error error=%s detail=%s", type(exc).__name__, exc)
            raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    ct = r.headers.get("content-type", "")
    logger.info("upstream_gemini_response status_code=%s content_type=%s", r.status_code, ct)
    if "application/json" not in ct:
        logger.warning(
            "upstream_gemini_non_json status_code=%s body=%s",
            r.status_code,
            r.text[:500],
        )
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned non-JSON ({r.status_code}): {r.text[:500]}",
        )
    try:
        data = r.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Gemini response was not valid JSON.") from exc

    if r.is_error:
        detail = data if isinstance(data, dict) else {"error": str(data)}
        logger.warning(
            "upstream_gemini_error status_code=%s detail=%s",
            r.status_code,
            _safe_detail(detail),
        )
        raise HTTPException(status_code=r.status_code, detail=detail)

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Gemini returned unexpected JSON root.")

    return _gemini_to_openai_shape(data, model)


def merge_request_body(body: dict[str, Any], redis_cfg: dict[str, Any]) -> dict[str, Any]:
    """Shared OpenAI-shaped merge (model / temperature / max_tokens defaults)."""
    return _merge_openai_payload(body, redis_cfg)
