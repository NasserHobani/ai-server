"""LLM model config from Redis and env (lives in API package; MCP package unchanged)."""

from __future__ import annotations

import json
import os
from typing import Any

import redis
from redis.exceptions import RedisError


def redis_url() -> str | None:
    url = os.getenv("CS_AI_BRIDGE_REDIS_URL", "").strip()
    return url or None


def llm_api_key() -> str | None:
    key = os.getenv("CS_AI_BRIDGE_LLM_API_KEY", "").strip()
    if key:
        return key
    alt = os.getenv("OPENAI_API_KEY", "").strip()
    return alt or None


def ai_config_key_prefix() -> str:
    return os.getenv("CS_AI_BRIDGE_AI_CONFIG_KEY_PREFIX", "cs_ai_bridge:ai:config").strip()


def ai_config_redis_key(tenant: str | None) -> str:
    prefix = ai_config_key_prefix().rstrip(":")
    t = (tenant or "").strip()
    if t:
        return f"{prefix}:{t}"
    return prefix


def _normalize_provider(provider: str | None) -> str | None:
    p = (provider or "").strip().lower()
    if p in {"openai", "oai"}:
        return "openai"
    if p in {"gemini", "google", "google-ai"}:
        return "gemini"
    return None


def _env_runtime_config(provider: str | None) -> dict[str, Any] | None:
    p = _normalize_provider(provider) or _normalize_provider(
        os.getenv("CS_AI_BRIDGE_LLM_PROVIDER", "openai")
    )
    if p == "gemini":
        return {
            "provider": "gemini",
            "model": os.getenv(
                "CS_AI_BRIDGE_GEMINI_MODEL",
                os.getenv("CS_AI_BRIDGE_LLM_MODEL", "gemini-2.0-flash"),
            ),
            "base_url": os.getenv(
                "CS_AI_BRIDGE_GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            ),
        }
    if p == "openai":
        return {
            "provider": "openai",
            "model": os.getenv(
                "CS_AI_BRIDGE_OPENAI_MODEL",
                os.getenv("CS_AI_BRIDGE_LLM_MODEL", "gpt-4o-mini"),
            ),
            "base_url": os.getenv("CS_AI_BRIDGE_LLM_BASE_URL", "https://api.openai.com/v1"),
        }
    return None


def get_redis_client_optional() -> redis.Redis | None:
    url = redis_url()
    if not url:
        return None
    return redis.from_url(url, decode_responses=True)


def read_ai_runtime_config(tenant: str | None, provider: str | None = None) -> dict[str, Any]:
    """Return JSON object from Redis for the configured AI config key.

    Common fields:

    - ``provider`` (str): ``openai`` (default) or ``gemini``.
    - ``model`` (str, required): e.g. ``gpt-4o-mini`` or ``gemini-2.0-flash``.
    - ``api_key`` (str, optional): overrides environment keys for this tenant.
    - ``base_url`` (str, optional): OpenAI-compatible root or Gemini API root
      (default OpenAI ``https://api.openai.com/v1``, Gemini
      ``https://generativelanguage.googleapis.com/v1beta``).
    - ``default_temperature``, ``default_max_tokens``, ``timeout_seconds``:
      optional defaults.
    - ``extra_headers`` (object, optional): merged into OpenAI requests only.
    """
    provider_key = _normalize_provider(provider)
    key_hint = tenant or provider_key
    fallback = _env_runtime_config(provider_key or tenant)

    client = get_redis_client_optional()
    if client is None:
        if fallback is not None:
            return fallback
        raise ValueError("CS_AI_BRIDGE_REDIS_URL is not set; cannot load AI model configuration.")

    key = ai_config_redis_key(key_hint)
    try:
        raw = client.get(key)
    except RedisError as exc:
        if fallback is not None:
            return fallback
        raise ValueError(f"Could not read AI configuration from Redis key '{key}'.") from exc
    if raw is None:
        if fallback is not None:
            return fallback
        hint = f"tenant '{tenant}'" if tenant else "global key"
        raise ValueError(f"AI configuration not found in Redis for {hint} (key '{key}').")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in Redis AI config key '{key}'.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"AI configuration in Redis key '{key}' must be a JSON object.")
    return parsed
