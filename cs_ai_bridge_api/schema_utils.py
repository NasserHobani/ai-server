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


def _required_create_fields(meta: dict[str, Any]) -> list[str]:
    required: list[str] = []

    mutation = meta.get("mutation")
    if isinstance(mutation, dict):
        create_rules = mutation.get("create")
        if isinstance(create_rules, dict):
            raw = create_rules.get("required_fields") or create_rules.get("required")
            if isinstance(raw, list):
                required.extend(str(name) for name in raw if str(name).strip())

    top_level = meta.get("required_fields")
    if isinstance(top_level, list):
        required.extend(str(name) for name in top_level if str(name).strip())

    fields = meta.get("fields")
    if isinstance(fields, dict):
        for name, info in fields.items():
            if isinstance(info, dict) and info.get("required"):
                required.append(str(name))

    seen: set[str] = set()
    out: list[str] = []
    for name in required:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def writable_field_names_for_model(
    schema: dict[str, Any],
    model: str,
    *,
    operation: str | None = None,
) -> list[str]:
    """Return schema-whitelisted writable fields for create/write prompts."""
    meta = readable_models(schema).get(model)
    if not isinstance(meta, dict):
        return []
    if operation and not model_allows_operation(schema, model, operation):
        return []

    fields = meta.get("fields")
    if not isinstance(fields, dict):
        return []

    out: list[str] = []
    for name, info in fields.items():
        if str(name).startswith("_"):
            continue
        if isinstance(info, dict):
            if info.get("readonly"):
                continue
            if info.get("can_write") is False:
                continue
        out.append(str(name))
    return sorted(out)


def required_create_fields_for_model(schema: dict[str, Any], model: str) -> list[str]:
    meta = readable_models(schema).get(model)
    if not isinstance(meta, dict):
        return []
    required = _required_create_fields(meta)
    writable = set(writable_field_names_for_model(schema, model, operation="create"))
    if writable:
        return [name for name in required if name in writable]
    return required


def model_label(schema: dict[str, Any], model: str) -> str:
    meta = readable_models(schema).get(model)
    if not isinstance(meta, dict):
        return model
    for key in ("label", "description", "title"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return model
