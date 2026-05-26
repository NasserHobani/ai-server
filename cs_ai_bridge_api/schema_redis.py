"""Load tenant schema metadata from Redis (gateway source of truth)."""

from __future__ import annotations

import json
import os
from typing import Any

import redis
from redis.exceptions import RedisError

from cs_ai_bridge_api.llm_config_redis import redis_url


def schema_key_prefix() -> str:
    return os.getenv("CS_AI_BRIDGE_SCHEMA_KEY_PREFIX", "cs_ai_bridge:schema").rstrip(":")


def normalize_tenant(value: str | None) -> str | None:
    token = (value or "").strip()
    if not token:
        return None
    prefix = f"{schema_key_prefix()}:"
    if token.startswith(prefix):
        token = token[len(prefix) :].strip()
    return token or None


def schema_redis_key(tenant: str) -> str:
    return f"{schema_key_prefix()}:{tenant.strip()}"


def read_schema_metadata(tenant: str) -> dict[str, Any]:
    url = redis_url()
    if not url:
        raise ValueError("CS_AI_BRIDGE_REDIS_URL is not set; cannot load schema metadata.")

    normalized = normalize_tenant(tenant)
    if not normalized:
        raise ValueError("Tenant is required for schema metadata.")

    key = schema_redis_key(normalized)
    try:
        client = redis.from_url(url, decode_responses=True)
        raw = client.get(key)
    except RedisError as exc:
        raise ValueError(f"Redis error loading schema key '{key}': {exc}") from exc

    if raw is None:
        raise ValueError(
            f"Schema metadata not found in Redis for tenant '{normalized}' (key '{key}')."
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON schema metadata in Redis key '{key}'.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Schema metadata in Redis key '{key}' must be a JSON object.")
    return parsed
