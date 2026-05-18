"""Validate MCP hints against Redis schema metadata (mirror Odoo rules)."""

from __future__ import annotations

from typing import Any

from . import config


def extract_domain_field_tokens(
    domain: list[Any] | tuple[Any, ...] | None,
) -> set[str]:
    names: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if not node:
                return
            token = node[0]
            if isinstance(token, str) and token in {"|", "&", "!"}:
                for child in node[1:]:
                    walk(child)
            elif isinstance(token, str):
                names.add(token)
            else:
                for item in node:
                    walk(item)

    walk(domain or [])
    return names


def validate_hints_with_schema(
    schema: dict[str, Any],
    tenant: str | None,
    model: str | None,
    fields: list[str] | None,
    domain: list[Any] | None,
) -> None:
    if tenant:
        cached_tenant = schema.get("tenant")
        if isinstance(cached_tenant, str) and cached_tenant != tenant:
            raise ValueError(
                f"Schema tenant mismatch key vs payload: "
                f"expected '{tenant}', got '{cached_tenant}'."
            )

    models = schema.get("models")
    if not isinstance(models, dict):
        raise ValueError("Schema metadata invalid: missing 'models' mapping.")

    if model:
        meta = models.get(model)
        if not meta:
            raise ValueError(f"Model '{model}' not allowed by schema whitelist.")
        if not isinstance(meta, dict):
            raise ValueError(f"Schema metadata for model '{model}' is invalid.")
        ops = meta.get("operations", {})
        if not isinstance(ops, dict) or not ops.get("read"):
            raise ValueError(
                f"Model '{model}' does not permit read operations per schema metadata."
            )

        fld = meta.get("fields", {})
        if isinstance(fld, dict):
            invalid = []
            for name in fields or []:
                if name not in fld:
                    invalid.append(name)
            if invalid:
                raise ValueError(
                    "Fields not whitelisted or unknown in schema "
                    f"for model '{model}': {sorted(invalid)}"
                )
            forbidden_filter = []
            for name in extract_domain_field_tokens(domain):
                finfo = fld.get(name)
                if not isinstance(finfo, dict) or not finfo.get("allow_filter"):
                    forbidden_filter.append(name)
            if forbidden_filter:
                raise ValueError(
                    "Domain references fields not permitted for filtering per schema "
                    f"on model '{model}': {sorted(forbidden_filter)}"
                )


def validate_create_vals_with_schema(
    schema: dict[str, Any],
    tenant: str | None,
    model: str | None,
    vals: dict[str, Any],
) -> None:
    if not model:
        raise ValueError("Model is required for MCP create.")

    if tenant:
        cached_tenant = schema.get("tenant")
        if isinstance(cached_tenant, str) and cached_tenant != tenant:
            raise ValueError(
                f"Schema tenant mismatch key vs payload: "
                f"expected '{tenant}', got '{cached_tenant}'."
            )

    models = schema.get("models")
    if not isinstance(models, dict):
        raise ValueError("Schema metadata invalid: missing 'models' mapping.")

    meta = models.get(model)
    if not meta:
        raise ValueError(f"Model '{model}' not allowed by schema whitelist.")
    if not isinstance(meta, dict):
        raise ValueError(f"Schema metadata for model '{model}' is invalid.")
    ops = meta.get("operations", {})
    if not isinstance(ops, dict) or not ops.get("create"):
        raise ValueError(
            f"Model '{model}' does not permit create operations in MCP schema whitelist."
        )

    fld = meta.get("fields", {})
    if not isinstance(fld, dict):
        raise ValueError(f"Schema metadata for model '{model}' lacks field entries.")

    for field_name in vals:
        info = fld.get(field_name)
        if not info:
            raise ValueError(
                f"Field '{field_name}' not allowed on model '{model}' for create."
            )
        if not isinstance(info, dict):
            raise ValueError(f"Invalid schema field definition for '{field_name}'.")
        if info.get("readonly"):
            raise ValueError(f"Cannot set readonly field '{field_name}' via MCP create.")
        # Mirror Odoo `_field_schema_entry` ``can_write`` (older cached schemas omit key).
        if info.get("can_write") is False:
            raise ValueError(f"Field '{field_name}' is not writable via MCP create (schema mutation).")


def validate_write_vals_with_schema(
    schema: dict[str, Any],
    tenant: str | None,
    model: str | None,
    vals: dict[str, Any],
) -> None:
    if not model:
        raise ValueError("Model is required for MCP write.")

    if tenant:
        cached_tenant = schema.get("tenant")
        if isinstance(cached_tenant, str) and cached_tenant != tenant:
            raise ValueError(
                f"Schema tenant mismatch key vs payload: "
                f"expected '{tenant}', got '{cached_tenant}'."
            )

    models = schema.get("models")
    if not isinstance(models, dict):
        raise ValueError("Schema metadata invalid: missing 'models' mapping.")

    meta = models.get(model)
    if not meta:
        raise ValueError(f"Model '{model}' not allowed by schema whitelist.")
    if not isinstance(meta, dict):
        raise ValueError(f"Schema metadata for model '{model}' is invalid.")
    ops = meta.get("operations", {})
    if not isinstance(ops, dict) or not ops.get("write"):
        raise ValueError(
            f"Model '{model}' does not permit write operations in MCP schema whitelist."
        )

    fld = meta.get("fields", {})
    if not isinstance(fld, dict):
        raise ValueError(f"Schema metadata for model '{model}' lacks field entries.")

    for field_name in vals:
        info = fld.get(field_name)
        if not info:
            raise ValueError(
                f"Field '{field_name}' not allowed on model '{model}' for write."
            )
        if not isinstance(info, dict):
            raise ValueError(f"Invalid schema field definition for '{field_name}'.")
        if info.get("readonly"):
            raise ValueError(f"Cannot set readonly field '{field_name}' via MCP write.")
        if info.get("can_write") is False:
            raise ValueError(f"Field '{field_name}' is not writable via MCP write (schema mutation).")


def ai_hints_validation_enabled(flag: bool | None) -> bool:
    if flag is not None:
        return flag
    return config.ai_validate_schema_default()


def normalize_ai_reuse_cached(reuse_cached_schema: bool) -> bool:
    if reuse_cached_schema:
        return True
    return config.ai_reuse_schema_cache_from_env()
