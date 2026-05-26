"""Infer MCP tool calls from user text and Redis schema metadata (no hardcoded models)."""

from __future__ import annotations

import os
import re
from typing import Any


def last_user_message_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).strip().lower() != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return " ".join(parts).strip()
    return ""


def has_query_tool_call(calls: list[dict[str, Any]]) -> bool:
    for call in calls:
        name = str(call.get("name", "")).strip()
        if name in {"mcp_query", "ai_query"}:
            return True
    return False


def schema_from_mcp_results(mcp_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in mcp_results:
        if str(entry.get("name", "")).strip() != "get_schema_metadata":
            continue
        result = entry.get("result")
        if isinstance(result, dict):
            return result
    return None


def _readable_models(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = schema.get("models")
    return models if isinstance(models, dict) else {}


def _model_allows_read(models: dict[str, dict[str, Any]], model: str) -> bool:
    meta = models.get(model)
    if not isinstance(meta, dict):
        return False
    ops = meta.get("operations")
    return isinstance(ops, dict) and bool(ops.get("read"))


def _tokenize_user_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    out: list[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        out.append(token)
        if token.endswith("s") and len(token) > 3:
            out.append(token[:-1])
    return out


def _add_terms(bucket: set[str], value: Any) -> None:
    if isinstance(value, str):
        for part in re.findall(r"[a-z0-9_]+", value.lower()):
            if len(part) >= 2:
                bucket.add(part)
    elif isinstance(value, list):
        for item in value:
            _add_terms(bucket, item)


def _model_search_terms(model_name: str, meta: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    _add_terms(terms, model_name)
    for part in model_name.replace("-", "_").split("."):
        _add_terms(terms, part)

    for key in (
        "description",
        "label",
        "title",
        "display_name",
        "name",
        "aliases",
        "keywords",
        "synonyms",
        "tags",
    ):
        _add_terms(terms, meta.get(key))

    fields_meta = meta.get("fields")
    if isinstance(fields_meta, dict):
        for field_name, field_info in fields_meta.items():
            _add_terms(terms, field_name)
            if isinstance(field_info, dict):
                for key in ("label", "description", "string", "help"):
                    _add_terms(terms, field_info.get(key))

    routing = meta.get("routing")
    if isinstance(routing, dict):
        for key in ("keywords", "aliases", "intents", "queries"):
            _add_terms(terms, routing.get(key))

    return terms


def _score_model(tokens: list[str], model_name: str, meta: dict[str, Any]) -> int:
    terms = _model_search_terms(model_name, meta)
    if not terms:
        return 0
    score = 0
    for token in tokens:
        if token in terms:
            score += 2
            continue
        for term in terms:
            if len(token) >= 4 and (token in term or term in token):
                score += 1
                break
    return score


def match_models_from_schema(
    user_text: str,
    schema: dict[str, Any],
    *,
    max_models: int = 3,
) -> list[str]:
    """Pick readable models whose schema metadata best matches the user message."""
    models = _readable_models(schema)
    if not models:
        return []

    tokens = _tokenize_user_text(user_text)
    if not tokens:
        return []

    ranked: list[tuple[str, int]] = []
    for model_name, meta in models.items():
        if not isinstance(meta, dict) or not _model_allows_read(models, model_name):
            continue
        score = _score_model(tokens, model_name, meta)
        if score > 0:
            ranked.append((model_name, score))

    if not ranked:
        return []

    ranked.sort(key=lambda item: (-item[1], item[0]))
    best = ranked[0][1]
    threshold = max(1, int(best * 0.6))
    selected = [name for name, score in ranked if score >= threshold]
    return selected[:max_models]


def _auto_query_mode() -> str:
    """``ai_query`` | ``schema`` | ``schema_then_ai`` (default)."""
    mode = os.getenv("CS_AI_BRIDGE_AUTO_QUERY_MODE", "schema_then_ai").strip().lower()
    if mode in {"ai_query", "ai", "router"}:
        return "ai_query"
    if mode in {"schema", "mcp_query", "match"}:
        return "schema"
    return "schema_then_ai"


def infer_mcp_queries(
    user_text: str,
    schema: dict[str, Any],
    tenant: str,
    *,
    default_limit: int = 80,
) -> list[dict[str, Any]]:
    """Build MCP tool calls from natural language using schema metadata only."""
    text = (user_text or "").strip()
    if not text:
        return []

    mode = _auto_query_mode()

    if mode == "ai_query":
        return [
            {
                "name": "ai_query",
                "arguments": {
                    "tenant": tenant,
                    "query": text,
                    "route": "auto",
                },
            }
        ]

    matched = match_models_from_schema(text, schema)
    if matched:
        return [
            {
                "name": "mcp_query",
                "arguments": {
                    "tenant": tenant,
                    "model": model_name,
                    "limit": default_limit,
                },
            }
            for model_name in matched
        ]

    if mode == "schema":
        return []

    return [
        {
            "name": "ai_query",
            "arguments": {
                "tenant": tenant,
                "query": text,
                "route": "auto",
            },
        }
    ]


def format_schema_summary(schema: dict[str, Any]) -> str:
    """Compact schema summary for the LLM (no raw nested JSON)."""
    models = _readable_models(schema)
    if not models:
        return ""

    lines = [
        "Odoo schema (whitelisted models you may read via attached record data):",
    ]
    for model_name in sorted(models.keys()):
        meta = models[model_name]
        if not isinstance(meta, dict) or not _model_allows_read(models, model_name):
            continue
        fields_meta = meta.get("fields")
        field_names: list[str] = []
        if isinstance(fields_meta, dict):
            field_names = sorted(
                name for name in fields_meta.keys() if not str(name).startswith("_")
            )
        label = meta.get("label") or meta.get("description") or meta.get("title")
        suffix = f" — {label}" if isinstance(label, str) and label.strip() else ""
        lines.append(
            f"- {model_name}{suffix}: "
            f"{', '.join(field_names) if field_names else '(no fields listed)'}"
        )
    return "\n".join(lines)
