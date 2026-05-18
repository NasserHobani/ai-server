"""Redis schema metadata: read, in-process TTL cache, optional pub/sub invalidation."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import redis

from . import config

_schema_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_schema_lock = threading.Lock()
_pubsub_thread: threading.Thread | None = None


def get_redis_client_optional() -> redis.Redis | None:
    url = config.redis_url()
    if not url:
        return None
    return redis.from_url(url, decode_responses=True)


def invalidate_schema_cache_tenant(tenant: str | None) -> None:
    if not tenant:
        return
    with _schema_lock:
        _schema_cache.pop(tenant, None)


def get_cached_schema(tenant: str) -> dict[str, Any] | None:
    ttl = config.schema_cache_ttl_seconds()
    if ttl <= 0:
        return None
    now = time.monotonic()
    with _schema_lock:
        entry = _schema_cache.get(tenant)
        if entry and entry[0] >= now:
            return entry[1]
    return None


def put_cached_schema(tenant: str, payload: dict[str, Any]) -> None:
    ttl = config.schema_cache_ttl_seconds()
    if ttl <= 0:
        return
    expiry = time.monotonic() + ttl
    with _schema_lock:
        _schema_cache[tenant] = (expiry, payload)


def read_schema_metadata(tenant: str) -> dict[str, Any]:
    client = get_redis_client_optional()
    if client is None:
        raise ValueError("CS_AI_BRIDGE_REDIS_URL is not set; cannot load schema metadata.")
    key = config.schema_key_for_tenant(tenant)
    raw = client.get(key)
    if raw is None:
        raise ValueError(f"Schema metadata not found in Redis for tenant '{tenant}'.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON schema metadata in Redis key '{key}'.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Schema metadata in Redis key '{key}' must be a JSON object.")
    return parsed


def _maybe_start_schema_pubsub() -> None:
    global _pubsub_thread

    if not config.schema_pubsub_enabled() or _pubsub_thread is not None:
        return

    url = config.redis_url()
    if not url:
        return

    channel = config.schema_pubsub_channel()

    def _worker() -> None:
        try:
            r = redis.from_url(url, decode_responses=True)
            pubsub = r.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(channel)
            for message in pubsub.listen():
                if not isinstance(message, dict) or message.get("type") != "message":
                    continue
                raw = message.get("data")
                if not isinstance(raw, str):
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and parsed.get("tenant"):
                    invalidate_schema_cache_tenant(str(parsed["tenant"]))
        except Exception:
            pass

    t = threading.Thread(target=_worker, name="schema-pubsub", daemon=True)
    _pubsub_thread = t
    t.start()


def maybe_start_schema_helpers() -> None:
    _maybe_start_schema_pubsub()


def tenant_for_schema_hints(tenant: str | None) -> str | None:
    return config.resolve_tenant(tenant)
