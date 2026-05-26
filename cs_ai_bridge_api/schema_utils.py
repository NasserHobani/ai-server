"""Shared helpers for Redis schema metadata (models, operations, fields)."""

from __future__ import annotations

from typing import Any


def readable_models(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = schema.get("models")
    return models if isinstance(models, dict) else {}


def model_allows_operation(
    schema: dict[str, Any],
    model: str,
    operation: str,
) -> bool:
    meta = readable_models(schema).get(model)
    if not isinstance(meta, dict):
        return False
    ops = meta.get("operations")
    return isinstance(ops, dict) and bool(ops.get(operation))


def models_for_operation(schema: dict[str, Any], operation: str) -> list[str]:
    out: list[str] = []
    for model_name in readable_models(schema):
        if model_allows_operation(schema, model_name, operation):
            out.append(model_name)
    return sorted(out)


def field_names_for_model(schema: dict[str, Any], model: str) -> list[str]:
    meta = readable_models(schema).get(model)
    if not isinstance(meta, dict):
        return []
    fields = meta.get("fields")
    if not isinstance(fields, dict):
        return []
    return sorted(name for name in fields if not str(name).startswith("_"))


def model_label(schema: dict[str, Any], model: str) -> str:
    meta = readable_models(schema).get(model)
    if not isinstance(meta, dict):
        return model
    for key in ("label", "description", "title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return model
