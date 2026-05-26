"""Validate MCP tool arguments against Redis schema before execution."""

from __future__ import annotations

import os
from typing import Any


def validation_enabled() -> bool:
    raw = os.getenv("CS_AI_BRIDGE_VALIDATE_MCP_TOOLS", "true").strip().lower()
    return raw not in {"0", "false", "no"}


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


def _assert_tenant(schema: dict[str, Any], tenant: str | None) -> None:
    if not tenant:
        raise ValueError("Tenant is required for MCP tool execution.")
    cached = schema.get("tenant")
    if isinstance(cached, str) and cached.strip() and cached != tenant:
        raise ValueError(
            f"Schema tenant mismatch: expected '{tenant}', schema has '{cached}'."
        )


def _assert_argument_tenant(arguments: dict[str, Any], tenant: str) -> None:
    arg_tenant = arguments.get("tenant")
    if arg_tenant is None:
        return
    if not isinstance(arg_tenant, str) or arg_tenant.strip() != tenant:
        raise ValueError(
            f"Argument tenant '{arg_tenant}' does not match request tenant '{tenant}'."
        )


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


def _writable_fields(meta: dict[str, Any]) -> list[str]:
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


def _required_writable_create_fields(meta: dict[str, Any]) -> list[str]:
    required = _required_create_fields(meta)
    writable = set(_writable_fields(meta))
    if not writable:
        return required
    return [name for name in required if name in writable]


def _writable_fields_hint(meta: dict[str, Any], *, max_fields: int = 20) -> str:
    writable = _writable_fields(meta)
    if not writable:
        return ""
    preview = ", ".join(writable[:max_fields])
    if len(writable) > max_fields:
        preview += ", ..."
    return f" Allowed writable fields: {preview}"


def validate_mcp_query(
    schema: dict[str, Any],
    *,
    tenant: str | None,
    model: str | None,
    fields: list[str] | None,
    domain: list[Any] | None,
) -> None:
    _assert_tenant(schema, tenant)
    models = schema.get("models")
    if not isinstance(models, dict):
        raise ValueError("Schema metadata invalid: missing 'models' mapping.")
    if not model:
        raise ValueError("Parameter 'model' is required for mcp_query.")

    meta = models.get(model)
    if not meta:
        raise ValueError(f"Model '{model}' is not allowed by schema whitelist.")
    if not isinstance(meta, dict):
        raise ValueError(f"Schema metadata for model '{model}' is invalid.")
    ops = meta.get("operations", {})
    if not isinstance(ops, dict) or not ops.get("read"):
        raise ValueError(f"Model '{model}' does not permit read operations.")

    fld = meta.get("fields", {})
    if not isinstance(fld, dict):
        return

    if fields:
        invalid = [name for name in fields if name not in fld]
        if invalid:
            raise ValueError(
                f"Fields not whitelisted for model '{model}': {sorted(invalid)}"
            )

    forbidden_filter = []
    for name in extract_domain_field_tokens(domain):
        finfo = fld.get(name)
        if not isinstance(finfo, dict) or not finfo.get("allow_filter"):
            forbidden_filter.append(name)
    if forbidden_filter:
        raise ValueError(
            f"Domain uses non-filterable fields on '{model}': {sorted(forbidden_filter)}"
        )


def validate_mcp_create(
    schema: dict[str, Any],
    *,
    tenant: str | None,
    model: str | None,
    vals: dict[str, Any],
) -> None:
    _assert_tenant(schema, tenant)
    if not model:
        raise ValueError("Parameter 'model' is required for mcp_create.")
    if not isinstance(vals, dict) or not vals:
        raise ValueError("Parameter 'vals' must be a non-empty object for mcp_create.")

    models = schema.get("models")
    if not isinstance(models, dict):
        raise ValueError("Schema metadata invalid: missing 'models' mapping.")

    meta = models.get(model)
    if not meta:
        raise ValueError(f"Model '{model}' is not allowed by schema whitelist.")
    ops = meta.get("operations", {})
    if not isinstance(ops, dict) or not ops.get("create"):
        raise ValueError(f"Model '{model}' does not permit create operations.")

    fld = meta.get("fields", {})
    if not isinstance(fld, dict):
        raise ValueError(f"Schema metadata for model '{model}' lacks field entries.")

    for field_name in vals:
        info = fld.get(field_name)
        if not info:
            raise ValueError(
                f"Field '{field_name}' is not allowed on '{model}' for create."
                + _writable_fields_hint(meta)
            )
        if not isinstance(info, dict):
            continue
        if info.get("readonly"):
            raise ValueError(
                f"Cannot set readonly field '{field_name}' on create."
                + _writable_fields_hint(meta)
            )
        if info.get("can_write") is False:
            raise ValueError(
                f"Field '{field_name}' is not writable on create."
                + _writable_fields_hint(meta)
            )

    missing = [name for name in _required_writable_create_fields(meta) if name not in vals]
    if missing:
        raise ValueError(
            f"Missing required fields for create on '{model}': {sorted(missing)}"
        )


def validate_mcp_write(
    schema: dict[str, Any],
    *,
    tenant: str | None,
    model: str | None,
    vals: dict[str, Any],
    record_id: int | None,
) -> None:
    _assert_tenant(schema, tenant)
    if not model:
        raise ValueError("Parameter 'model' is required for mcp_write.")
    if record_id is None:
        raise ValueError("Parameter 'record_id' is required for mcp_write.")
    if not isinstance(vals, dict) or not vals:
        raise ValueError("Parameter 'vals' must be a non-empty object for mcp_write.")

    models = schema.get("models")
    if not isinstance(models, dict):
        raise ValueError("Schema metadata invalid: missing 'models' mapping.")

    meta = models.get(model)
    if not meta:
        raise ValueError(f"Model '{model}' is not allowed by schema whitelist.")
    ops = meta.get("operations", {})
    if not isinstance(ops, dict) or not ops.get("write"):
        raise ValueError(f"Model '{model}' does not permit write operations.")

    fld = meta.get("fields", {})
    if not isinstance(fld, dict):
        raise ValueError(f"Schema metadata for model '{model}' lacks field entries.")

    for field_name in vals:
        info = fld.get(field_name)
        if not info:
            raise ValueError(
                f"Field '{field_name}' is not allowed on '{model}' for write."
                + _writable_fields_hint(meta)
            )
        if not isinstance(info, dict):
            continue
        if info.get("readonly"):
            raise ValueError(
                f"Cannot set readonly field '{field_name}' on write."
                + _writable_fields_hint(meta)
            )
        if info.get("can_write") is False:
            raise ValueError(
                f"Field '{field_name}' is not writable on write."
                + _writable_fields_hint(meta)
            )


def validate_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
    tenant: str | None,
) -> None:
    if not validation_enabled():
        return

    name = tool_name.strip()
    args = dict(arguments)
    normalized_tenant = (tenant or "").strip() or None
    _assert_tenant(schema, normalized_tenant)
    if normalized_tenant:
        _assert_argument_tenant(args, normalized_tenant)

    if name == "mcp_query":
        validate_mcp_query(
            schema,
            tenant=normalized_tenant,
            model=args.get("model") if isinstance(args.get("model"), str) else None,
            fields=args.get("fields") if isinstance(args.get("fields"), list) else None,
            domain=args.get("domain") if isinstance(args.get("domain"), list) else None,
        )
    elif name == "mcp_create":
        validate_mcp_create(
            schema,
            tenant=normalized_tenant,
            model=args.get("model") if isinstance(args.get("model"), str) else None,
            vals=args.get("vals") if isinstance(args.get("vals"), dict) else {},
        )
    elif name == "mcp_write":
        rid = args.get("record_id")
        validate_mcp_write(
            schema,
            tenant=normalized_tenant,
            model=args.get("model") if isinstance(args.get("model"), str) else None,
            vals=args.get("vals") if isinstance(args.get("vals"), dict) else {},
            record_id=rid if isinstance(rid, int) else None,
        )
    elif name == "ai_query":
        if args.get("model") and isinstance(args.get("model"), str):
            validate_mcp_query(
                schema,
                tenant=normalized_tenant,
                model=args["model"],
                fields=args.get("fields") if isinstance(args.get("fields"), list) else None,
                domain=args.get("domain") if isinstance(args.get("domain"), list) else None,
            )
