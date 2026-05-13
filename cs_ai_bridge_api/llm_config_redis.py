"""LLM model config from Redis and env (lives in API package; MCP package unchanged)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)


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


def assistants_config_key_prefix() -> str:
    return os.getenv(
        "CS_AI_BRIDGE_ASSISTANTS_CONFIG_KEY_PREFIX",
        "cs_ai_bridge:config:assistants",
    ).strip()


def ai_config_redis_key(tenant: str | None) -> str:
    prefix = ai_config_key_prefix().rstrip(":")
    t = (tenant or "").strip()
    if t:
        return f"{prefix}:{t}"
    return prefix


def assistants_config_redis_key(tenant: str) -> str:
    prefix = assistants_config_key_prefix().rstrip(":")
    return f"{prefix}:{tenant.strip()}"


def _normalize_provider(provider: str | None) -> str | None:
    p = (provider or "").strip().lower()
    if p in {"openai", "oai"}:
        return "openai"
    if p in {"gemini", "google", "google-ai"}:
        return "gemini"
    return None


def _tenant_hint(tenant: str | None) -> str | None:
    hint = (tenant or "").strip() or os.getenv("CS_AI_BRIDGE_TENANT", "").strip()
    return hint or None


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
            "_config_source": "env",
        }
    if p == "openai":
        return {
            "provider": "openai",
            "model": os.getenv(
                "CS_AI_BRIDGE_OPENAI_MODEL",
                os.getenv("CS_AI_BRIDGE_LLM_MODEL", "gpt-4o-mini"),
            ),
            "base_url": os.getenv("CS_AI_BRIDGE_LLM_BASE_URL", "https://api.openai.com/v1"),
            "_config_source": "env",
        }
    return None


def _sort_assistant_key(item: dict[str, Any]) -> tuple[int, int]:
    sequence = item.get("sequence")
    assistant_id = item.get("id")
    return (
        sequence if isinstance(sequence, int) else 999_999,
        assistant_id if isinstance(assistant_id, int) else 999_999,
    )


def _runtime_config_from_assistants(
    parsed: dict[str, Any],
    provider: str | None,
    source_key: str,
) -> dict[str, Any] | None:
    assistants = parsed.get("assistants")
    if not isinstance(assistants, list):
        return None

    provider_key = _normalize_provider(provider)
    active: list[dict[str, Any]] = []
    for item in assistants:
        if not isinstance(item, dict):
            continue
        if item.get("active") is False:
            continue
        item_provider = _normalize_provider(str(item.get("provider", "")))
        if provider_key and item_provider != provider_key:
            continue
        if item_provider is None:
            continue
        active.append(item)

    if not active:
        return None

    general = [
        item
        for item in active
        if str(item.get("purpose", "")).strip().lower() in {"", "general"}
    ]
    selected = sorted(general or active, key=_sort_assistant_key)[0]
    selected_provider = _normalize_provider(str(selected.get("provider", ""))) or "openai"
    model = selected.get("model_name") or selected.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Assistant config in Redis key '{source_key}' must include model_name.")

    cfg: dict[str, Any] = {
        "provider": selected_provider,
        "model": model.strip(),
        "_config_source": f"redis:{source_key}:assistant:{selected.get('id', selected.get('name', 'unknown'))}",
    }

    api_key = selected.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        cfg["api_key"] = api_key.strip()

    endpoint = selected.get("endpoint_url") or selected.get("base_url")
    if isinstance(endpoint, str) and endpoint.strip():
        cfg["base_url"] = endpoint.strip()

    temperature = selected.get("temperature")
    if isinstance(temperature, (int, float)):
        cfg["default_temperature"] = float(temperature)

    max_tokens = selected.get("max_tokens")
    if isinstance(max_tokens, int):
        cfg["default_max_tokens"] = max_tokens

    timeout_seconds = selected.get("timeout_seconds")
    if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0:
        cfg["timeout_seconds"] = float(timeout_seconds)

    logger.info(
        "ai_config_using_assistant key=%s assistant_id=%s provider=%s model=%s api_key_in_redis=%s",
        source_key,
        selected.get("id"),
        cfg.get("provider"),
        cfg.get("model"),
        "api_key" in cfg,
    )
    return cfg


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
    tenant_key = _tenant_hint(tenant)
    key_hint = tenant_key or provider_key
    fallback = _env_runtime_config(provider_key or tenant_key)

    client = get_redis_client_optional()
    if client is None:
        if fallback is not None:
            logger.info(
                "ai_config_using_env reason=redis_url_missing provider=%s model=%s",
                fallback.get("provider"),
                fallback.get("model"),
            )
            return fallback
        raise ValueError("CS_AI_BRIDGE_REDIS_URL is not set; cannot load AI model configuration.")

    keys: list[str] = []
    if key_hint:
        keys.append(ai_config_redis_key(key_hint))
    if tenant_key:
        keys.append(assistants_config_redis_key(tenant_key))
    if not keys:
        keys.append(ai_config_redis_key(None))

    try:
        raw_by_key = {key: client.get(key) for key in dict.fromkeys(keys)}
    except RedisError as exc:
        if fallback is not None:
            logger.warning(
                "ai_config_using_env reason=redis_error keys=%s provider=%s model=%s error=%s",
                ",".join(dict.fromkeys(keys)),
                fallback.get("provider"),
                fallback.get("model"),
                type(exc).__name__,
            )
            return fallback
        raise ValueError("Could not read AI configuration from Redis.") from exc

    missing: list[str] = []
    for key, raw in raw_by_key.items():
        if raw is None:
            missing.append(key)
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in Redis AI config key '{key}'.") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"AI configuration in Redis key '{key}' must be a JSON object.")

        assistant_cfg = _runtime_config_from_assistants(parsed, provider_key, key)
        if assistant_cfg is not None:
            return assistant_cfg

        parsed["_config_source"] = f"redis:{key}"
        logger.info(
            "ai_config_using_redis key=%s provider=%s model=%s api_key_in_redis=%s",
            key,
            parsed.get("provider", "openai"),
            parsed.get("model"),
            bool(str(parsed.get("api_key", "")).strip()),
        )
        return parsed

    if fallback is not None:
        logger.info(
            "ai_config_using_env reason=redis_keys_missing keys=%s provider=%s model=%s",
            ",".join(missing or keys),
            fallback.get("provider"),
            fallback.get("model"),
        )
        return fallback
    hint = f"tenant '{tenant_key}'" if tenant_key else "global key"
    raise ValueError(
        f"AI configuration not found in Redis for {hint} "
        f"(keys: {', '.join(missing or keys)})."
    )
