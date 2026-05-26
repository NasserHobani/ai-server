"""Convert MCP tool definitions into OpenAI Responses API function tools."""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

from cs_ai_bridge_api.mcp_client import list_mcp_tool_definitions
from cs_ai_bridge_api.schema_utils import (
    field_names_for_model,
    model_label,
    models_for_operation,
)


logger = logging.getLogger(__name__)

_INTERNAL_ONLY_TOOLS = frozenset({"get_schema_metadata"})
DEFAULT_EXPOSED_TOOLS = frozenset({"mcp_query", "mcp_create", "mcp_write", "ai_query"})


def exposed_tool_names() -> frozenset[str]:
    raw = os.getenv("CS_AI_BRIDGE_OPENAI_MCP_TOOLS", "").strip()
    if not raw:
        return DEFAULT_EXPOSED_TOOLS
    names = {part.strip() for part in raw.split(",") if part.strip()}
    return frozenset(names) if names else DEFAULT_EXPOSED_TOOLS


def _enrich_description(base: str, schema: dict[str, Any] | None, tool_name: str) -> str:
    if not schema:
        return base.strip()

    parts = [base.strip()] if base.strip() else []
    tenant = schema.get("tenant")
    if isinstance(tenant, str) and tenant.strip():
        parts.append(f"Tenant context: {tenant.strip()}.")

    if tool_name == "mcp_query":
        read_models = models_for_operation(schema, "read")
        if read_models:
            lines = ["Readable models (schema whitelist):"]
            for model in read_models[:40]:
                fields = field_names_for_model(schema, model)
                preview = ", ".join(fields[:12])
                if len(fields) > 12:
                    preview += ", ..."
                lines.append(f"- {model_label(schema, model)} (`{model}`): {preview or 'see schema'}")
            parts.append("\n".join(lines))
    elif tool_name == "mcp_create":
        create_models = models_for_operation(schema, "create")
        if create_models:
            parts.append(
                "Creatable models: "
                + ", ".join(f"{model_label(schema, m)} (`{m}`)" for m in create_models[:30])
            )
    elif tool_name == "mcp_write":
        write_models = models_for_operation(schema, "write")
        if write_models:
            parts.append(
                "Writable models: "
                + ", ".join(f"{model_label(schema, m)} (`{m}`)" for m in write_models[:30])
            )

    parts.append("Omit `fields` to use schema defaults. Pass `tenant` when required.")
    return "\n\n".join(parts)


def _apply_schema_to_parameters(
    tool_name: str,
    parameters: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    if not schema:
        return parameters

    params = copy.deepcopy(parameters)
    if params.get("type") != "object":
        params = {"type": "object", "properties": {}, "required": []}

    properties = params.setdefault("properties", {})
    if not isinstance(properties, dict):
        properties = {}
        params["properties"] = properties

    if tool_name == "mcp_query":
        read_models = models_for_operation(schema, "read")
        if read_models:
            properties["model"] = {
                "type": "string",
                "description": "Odoo model technical name (schema whitelist).",
                "enum": read_models,
            }
    elif tool_name in {"mcp_create", "mcp_write"}:
        op = "create" if tool_name == "mcp_create" else "write"
        allowed = models_for_operation(schema, op)
        if allowed:
            properties["model"] = {
                "type": "string",
                "description": f"Odoo model for {op} (schema whitelist).",
                "enum": allowed,
            }

    if "tenant" not in properties:
        properties["tenant"] = {
            "type": "string",
            "description": "Tenant id for Odoo bridge routing and isolation.",
        }
    return params


def _assert_gateway_function_tool(tool: dict[str, Any]) -> None:
    tool_type = str(tool.get("type", "")).strip().lower()
    if tool_type == "mcp" or tool.get("server_url"):
        raise ValueError(
            "Remote MCP tools are disabled; gateway executes function tools locally only."
        )
    if tool_type != "function":
        raise ValueError(f"Unsupported OpenAI tool type '{tool_type}' (expected 'function').")


def mcp_tool_to_openai_function(
    tool_def: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    name = str(tool_def.get("name", "")).strip()
    if not name or name in _INTERNAL_ONLY_TOOLS:
        return None
    if name not in exposed_tool_names():
        return None

    input_schema = tool_def.get("inputSchema")
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}

    tool = {
        "type": "function",
        "name": name,
        "description": _enrich_description(
            str(tool_def.get("description", "") or ""),
            schema,
            name,
        ),
        "parameters": _apply_schema_to_parameters(name, input_schema, schema),
    }
    _assert_gateway_function_tool(tool)
    return tool


async def build_openai_function_tools(
    request_id: str,
    schema: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    definitions = await list_mcp_tool_definitions(request_id)
    tools = [
        converted
        for tool_def in definitions
        if (converted := mcp_tool_to_openai_function(tool_def, schema))
    ]
    logger.info(
        "openai_function_tools_built request_id=%s count=%s names=%s",
        request_id,
        len(tools),
        ",".join(t["name"] for t in tools),
    )
    return tools


def schema_summary_for_instructions(schema: dict[str, Any]) -> str:
    lines = [
        "You are an Odoo ERP assistant. Use function tools to read or mutate live business data.",
        "The gateway validates every tool call against the tenant schema (models, operations, fields).",
        "Do not refuse data listing requests when tools can fetch records.",
        "Never output [object Object]; format record fields as plain text or tables.",
    ]
    tenant = schema.get("tenant")
    if isinstance(tenant, str) and tenant.strip():
        lines.append(f"Active tenant: {tenant.strip()}.")

    read_models = models_for_operation(schema, "read")
    if read_models:
        lines.append("Readable models: " + ", ".join(f"`{m}`" for m in read_models[:50]))
    return "\n".join(lines)
