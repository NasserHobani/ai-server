"""OpenAI and Gemini upstream calls; routing uses Redis ``provider`` field."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from urllib.parse import quote

import httpx
from cs_ai_bridge_api.mcp_client import list_openai_function_tools
from fastapi import HTTPException


logger = logging.getLogger(__name__)

# CS AI Bridge selectors — must not be forwarded to OpenAI / Gemini HTTP APIs.
_BRIDGE_ONLY_REQUEST_FIELDS = frozenset(
    {
        "tenant",
        "provider",
        "mcp_tool_calls",
        "assistant_key",
        "schema_key",
        "assistantKey",
        "schemaKey",
    }
)

# OpenAI Chat Completions body fields (unknown keys cause 400 invalid_request_error).
_OPENAI_CHAT_COMPLETION_ALLOWED = frozenset(
    {
        "messages",
        "model",
        "audio",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "web_search_options",
        # Deprecated but still accepted by some gateways
        "function_call",
        "functions",
    }
)


def _strip_bridge_only_fields(out: dict[str, Any]) -> None:
    for key in list(out.keys()):
        if key in _BRIDGE_ONLY_REQUEST_FIELDS or key.startswith("_"):
            out.pop(key, None)


def _filter_openai_allowed_fields(out: dict[str, Any]) -> list[str]:
    removed: list[str] = []
    for key in list(out.keys()):
        if key not in _OPENAI_CHAT_COMPLETION_ALLOWED:
            out.pop(key, None)
            removed.append(key)
    return removed


def _safe_detail(detail: Any) -> str:
    return str(detail).replace("\n", " ")[:1000]


def _truncate_body(value: Any, limit: int = 4000) -> Any:
    if isinstance(value, str):
        return value[:limit]
    return value


def _log_json(label: str, payload: Any, *, max_len: int = 12000) -> None:
    """Log a JSON-serializable payload (truncated) for debugging upstream LLM I/O."""
    try:
        raw = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = str(payload)
    if len(raw) > max_len:
        raw = f"{raw[:max_len]}...[truncated {len(raw) - max_len} chars]"
    logger.info("%s %s", label, raw)


def _extract_responses_output_text(data: dict[str, Any]) -> str:
    """Aggregate assistant text from an OpenAI Responses API JSON body."""
    top = data.get("output_text")
    if isinstance(top, str) and top:
        return top

    texts: list[str] = []
    output = data.get("output")
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
            block_type = block.get("type")
            if block_type == "output_text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif block_type == "refusal" and isinstance(block.get("refusal"), str):
                texts.append(block["refusal"])
    return "".join(texts)


def _response_body_or_text(response: httpx.Response) -> Any:
    ct = response.headers.get("content-type", "")
    if "application/json" in ct:
        try:
            return response.json()
        except ValueError:
            return _truncate_body(response.text)
    return _truncate_body(response.text)


def _raise_upstream_error(
    *,
    provider: str,
    status_code: int,
    message: str,
    body: Any = None,
    error: str | None = None,
) -> None:
    detail: dict[str, Any] = {
        "provider": provider,
        "message": message,
    }
    if error:
        detail["error"] = error
    if body is not None:
        detail["body"] = _truncate_body(body)
    raise HTTPException(status_code=status_code, detail=detail)


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


def prepare_openai_request_body(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy safe to send to OpenAI (without bridge-only fields/metadata)."""
    payload = dict(body)
    _strip_bridge_only_fields(payload)
    removed = _filter_openai_allowed_fields(payload)
    if removed:
        logger.info("openai_request_fields_removed keys=%s", ",".join(sorted(removed)))
    # User requested never forwarding metadata to AI.
    payload.pop("metadata", None)
    for key in list(payload.keys()):
        if key.startswith("metadata."):
            payload.pop(key, None)
    return payload


def _merge_openai_payload(
    body: dict[str, Any],
    redis_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(body)
    _strip_bridge_only_fields(out)
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


def _openai_responses_url(redis_cfg: dict[str, Any]) -> str:
    raw = redis_cfg.get("base_url") or os.getenv(
        "CS_AI_BRIDGE_LLM_BASE_URL", "https://api.openai.com/v1"
    )
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=500, detail="Invalid base_url in Redis AI config.")
    return f"{_normalize_base_url(raw.strip())}/responses"


def _messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Map chat messages to OpenAI Responses API ``input`` items.

    Assistant history must use ``output_text`` blocks; user/developer use ``input_text``.
    System prompts are returned separately as ``instructions``.
    """
    instructions_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        text = _message_text(msg.get("content"))

        if role == "system":
            if text:
                instructions_parts.append(text)
            continue

        if role == "assistant":
            out_role = "assistant"
            block_type = "output_text"
        elif role == "tool":
            out_role = "user"
            block_type = "input_text"
            if text:
                text = f"[tool]\n{text}"
        elif role in {"user", "developer"}:
            out_role = role
            block_type = "input_text"
        else:
            out_role = "user"
            block_type = "input_text"
            if text:
                text = f"[{role}]\n{text}"

        if not text and out_role != "assistant":
            continue

        out.append(
            {
                "role": out_role,
                "content": [
                    {
                        "type": block_type,
                        "text": text,
                    }
                ],
            }
        )

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, out


def _responses_to_chat_completion_shape(data: dict[str, Any], model: str) -> dict[str, Any]:
    text = _extract_responses_output_text(data)
    created = int(time.time())
    usage_raw = data.get("usage")
    usage: dict[str, int] = {}
    if isinstance(usage_raw, dict):
        inp = usage_raw.get("input_tokens")
        outp = usage_raw.get("output_tokens")
        total = usage_raw.get("total_tokens")
        if isinstance(inp, int):
            usage["prompt_tokens"] = inp
        if isinstance(outp, int):
            usage["completion_tokens"] = outp
        if isinstance(total, int):
            usage["total_tokens"] = total

    return {
        "id": data.get("id") or f"resp-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "provider": "openai",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or None,
        "response": data,
    }


def _timeout_seconds(redis_cfg: dict[str, Any]) -> float:
    v = redis_cfg.get("timeout_seconds")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return float(os.getenv("CS_AI_BRIDGE_LLM_TIMEOUT", "120"))


def _message_text(content: Any) -> str:
    from cs_ai_bridge_api.mcp_format import normalize_message_content

    return normalize_message_content(content)


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
        url = _openai_responses_url(redis_cfg)
        openai_payload = prepare_openai_request_body(merged_body)
        messages = openai_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list.")
        tools = await list_openai_function_tools(request_id=f"openai-{int(time.time())}")
        instructions, responses_input = _messages_to_responses_input(messages)
        if not responses_input:
            raise HTTPException(
                status_code=400,
                detail="At least one non-system user or assistant message is required.",
            )
        responses_payload: dict[str, Any] = {
            "model": str(openai_payload.get("model", "")),
            "input": responses_input,
            "tools": tools,
        }
        if instructions:
            responses_payload["instructions"] = instructions
        if openai_payload.get("temperature") is not None:
            responses_payload["temperature"] = openai_payload["temperature"]
        if openai_payload.get("max_tokens") is not None:
            responses_payload["max_output_tokens"] = openai_payload["max_tokens"]
        logger.info(
            "upstream_openai_request url=%s model=%s payload_keys=%s",
            url,
            responses_payload.get("model"),
            ",".join(sorted(responses_payload.keys())),
        )
        _log_json("upstream_openai_request_body", responses_payload)
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
                r = await client.post(url, json=responses_payload, headers=headers)
            except httpx.RequestError as exc:
                logger.warning("upstream_openai_request_error error=%s detail=%s", type(exc).__name__, exc)
                _raise_upstream_error(
                    provider="openai",
                    status_code=502,
                    message="Upstream request failed.",
                    error=str(exc),
                )

        ct = r.headers.get("content-type", "")
        logger.info("upstream_openai_response status_code=%s content_type=%s", r.status_code, ct)
        if "application/json" not in ct:
            logger.warning(
                "upstream_openai_non_json status_code=%s body=%s",
                r.status_code,
                r.text[:500],
            )
            _raise_upstream_error(
                provider="openai",
                status_code=502,
                message=f"Upstream returned non-JSON (status {r.status_code}).",
                body=_response_body_or_text(r),
            )
        try:
            data = r.json()
        except ValueError as exc:
            _raise_upstream_error(
                provider="openai",
                status_code=502,
                message="Upstream response was not valid JSON.",
                body=_truncate_body(r.text),
                error=str(exc),
            )
        if r.is_error:
            detail = data if isinstance(data, dict) else data
            logger.warning(
                "upstream_openai_error status_code=%s detail=%s",
                r.status_code,
                _safe_detail(detail),
            )
            _raise_upstream_error(
                provider="openai",
                status_code=r.status_code,
                message=f"Upstream OpenAI returned status {r.status_code}.",
                body=detail,
            )
        if not isinstance(data, dict):
            _raise_upstream_error(
                provider="openai",
                status_code=502,
                message="OpenAI returned unexpected JSON root.",
                body=data,
            )
        _log_json("upstream_openai_response_body", data)
        assistant_text = _extract_responses_output_text(data)
        logger.info(
            "upstream_openai_assistant_text len=%s preview=%s",
            len(assistant_text),
            _safe_detail(assistant_text[:500] if assistant_text else "<empty>"),
        )
        shaped = _responses_to_chat_completion_shape(data, str(responses_payload["model"]))
        _log_json("upstream_openai_chat_completion", shaped)
        return shaped

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
            _raise_upstream_error(
                provider="gemini",
                status_code=502,
                message="Upstream request failed.",
                error=str(exc),
            )

    ct = r.headers.get("content-type", "")
    logger.info("upstream_gemini_response status_code=%s content_type=%s", r.status_code, ct)
    if "application/json" not in ct:
        logger.warning(
            "upstream_gemini_non_json status_code=%s body=%s",
            r.status_code,
            r.text[:500],
        )
        _raise_upstream_error(
            provider="gemini",
            status_code=502,
            message=f"Upstream returned non-JSON (status {r.status_code}).",
            body=_response_body_or_text(r),
        )
    try:
        data = r.json()
    except ValueError as exc:
        _raise_upstream_error(
            provider="gemini",
            status_code=502,
            message="Upstream response was not valid JSON.",
            body=_truncate_body(r.text),
            error=str(exc),
        )

    if r.is_error:
        detail = data if isinstance(data, dict) else data
        logger.warning(
            "upstream_gemini_error status_code=%s detail=%s",
            r.status_code,
            _safe_detail(detail),
        )
        _raise_upstream_error(
            provider="gemini",
            status_code=r.status_code,
            message=f"Upstream Gemini returned status {r.status_code}.",
            body=detail,
        )

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Gemini returned unexpected JSON root.")

    return _gemini_to_openai_shape(data, model)


def merge_request_body(body: dict[str, Any], redis_cfg: dict[str, Any]) -> dict[str, Any]:
    """Shared OpenAI-shaped merge (model / temperature / max_tokens defaults)."""
    return _merge_openai_payload(body, redis_cfg)
