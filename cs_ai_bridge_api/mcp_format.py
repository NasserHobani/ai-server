"""Normalize MCP / Odoo payloads and format them for LLM and API clients."""

from __future__ import annotations

import json
from typing import Any

_SCHEMA_TOOL = "get_schema_metadata"


def normalize_message_content(content: Any) -> str:
    """Turn message ``content`` into plain text (never leave opaque objects)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(field_to_display(block))
        return "\n".join(parts) if parts else ""
    if isinstance(content, dict):
        if "content" in content and "role" in content:
            return normalize_message_content(content.get("content"))
        return json.dumps(content, ensure_ascii=False, indent=2)
    return field_to_display(content)


def field_to_display(value: Any) -> str:
    """Human-readable scalar for Odoo fields (many2one, x2many, nested dicts)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], str):
            return f"{value[1]} (id {value[0]})"
        if not value:
            return ""
        return ", ".join(field_to_display(item) for item in value)
    if isinstance(value, dict):
        name = value.get("display_name") or value.get("name")
        record_id = value.get("id")
        if name is not None and record_id is not None:
            return f"{name} (id {record_id})"
        if name is not None:
            return str(name)
        if record_id is not None and len(value) <= 2:
            return f"id {record_id}"
        flat = {str(k): field_to_display(v) for k, v in value.items()}
        return json.dumps(flat, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, default=str)


def normalize_record(record: dict[str, Any]) -> dict[str, str]:
    return {str(key): field_to_display(val) for key, val in record.items()}


def normalize_business_payload(value: Any) -> Any:
    """Unwrap JSON-RPC and normalize nested record lists."""
    if isinstance(value, dict):
        if value.get("jsonrpc") and "result" in value:
            return normalize_business_payload(value["result"])
        if "error" in value and "result" not in value:
            return value

        records = _records_from_payload(value)
        if records is not None:
            return {"records": records}

        out: dict[str, Any] = {}
        for key, val in value.items():
            if key == "records" and isinstance(val, list):
                out[key] = [
                    normalize_record(row) if isinstance(row, dict) else field_to_display(row)
                    for row in val
                ]
            else:
                out[key] = normalize_field_value(val)
        return out

    if isinstance(value, list):
        if value and all(isinstance(row, dict) for row in value):
            return {"records": [normalize_record(row) for row in value]}
        return [normalize_field_value(item) for item in value]
    return normalize_field_value(value)


def normalize_field_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "display_name" in value or ("id" in value and "name" in value):
            return field_to_display(value)
        return {str(k): normalize_field_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], str):
            return field_to_display(value)
        return [normalize_field_value(v) for v in value]
    return value


def unwrap_tool_payload(value: Any) -> Any:
    """Extract business data from a CallToolResult-shaped dict."""
    if not isinstance(value, dict):
        return normalize_business_payload(value)

    if "structured_content" in value or "content" in value or "data" in value:
        structured = value.get("structured_content")
        if structured is not None:
            return normalize_business_payload(structured)

        data = value.get("data")
        if data is not None:
            return normalize_business_payload(data)

        parsed = _parse_content_blocks(value.get("content"))
        if parsed is not None:
            return normalize_business_payload(parsed)

    return normalize_business_payload(value)


def _parse_content_blocks(content: Any) -> Any | None:
    if content is None:
        return None
    if isinstance(content, str):
        return _maybe_parse_json(content)
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
            elif hasattr(block, "text") and isinstance(getattr(block, "text"), str):
                texts.append(getattr(block, "text"))
        if not texts:
            return None
        combined = "\n".join(texts)
        return _maybe_parse_json(combined) or combined
    return None


def _maybe_parse_json(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _looks_like_record(row: Any) -> bool:
    return isinstance(row, dict) and bool(row) and "id" in row


def _records_from_payload(payload: Any) -> list[dict[str, str]] | None:
    if isinstance(payload, list):
        if payload and all(_looks_like_record(row) for row in payload):
            return [normalize_record(row) for row in payload]
        return None
    if not isinstance(payload, dict):
        return None

    for key in ("records", "data", "result", "rows", "items"):
        candidate = payload.get(key)
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return [normalize_record(row) for row in candidate]
        if isinstance(candidate, dict):
            nested = _records_from_payload(candidate)
            if nested:
                return nested
    return None


def format_payload_content(payload: Any) -> str:
    """Format business payload as plain text (no tool names or JSON wrappers)."""
    records = _records_from_payload(payload) if payload is not None else None
    if records:
        blocks: list[str] = []
        for index, row in enumerate(records, start=1):
            lines = [f"Record {index}:"]
            for key in sorted(row.keys()):
                value = row.get(key, "")
                if value:
                    lines.append(f"- {key}: {value}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    if isinstance(payload, dict):
        lines = [f"- {key}: {field_to_display(val)}" for key, val in sorted(payload.items())]
        return "\n".join(lines)
    if payload is None:
        return ""
    return field_to_display(payload)


def format_mcp_results_for_context(
    mcp_results: list[dict[str, Any]],
    *,
    schema_summary: str = "",
) -> str:
    """Plain business data for the LLM (records + optional schema summary)."""
    from cs_ai_bridge_api.mcp_intent import format_schema_summary, schema_from_mcp_results

    sections: list[str] = [
        "You have live Odoo ERP data below. Answer the user using these records. "
        "Do not refuse listing customers, contacts, invoices, or other business data "
        "when records are attached.",
    ]
    if schema_summary.strip():
        sections.append(schema_summary.strip())
    else:
        schema = schema_from_mcp_results(mcp_results)
        if schema:
            summary = format_schema_summary(schema)
            if summary.strip():
                sections.append(summary)

    for entry in mcp_results:
        if str(entry.get("name", "")).strip() == _SCHEMA_TOOL:
            continue
        payload = entry.get("result")
        formatted = format_payload_content(payload)
        if formatted.strip():
            sections.append(formatted)
    return "\n\n".join(sections)


def format_mcp_results_plain(mcp_results: list[dict[str, Any]]) -> str:
    """Human-readable fallback answer built from MCP query results."""
    sections: list[str] = []
    for entry in mcp_results:
        if str(entry.get("name", "")).strip() == _SCHEMA_TOOL:
            continue
        payload = entry.get("result")
        formatted = format_payload_content(payload)
        if formatted.strip():
            sections.append(formatted)
    if not sections:
        return "No matching records were found."
    return "\n\n".join(sections)


def normalize_mcp_results_list(mcp_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in mcp_results:
        raw = entry.get("result")
        payload = unwrap_tool_payload(raw)
        out.append(
            {
                "name": entry.get("name"),
                "arguments": entry.get("arguments"),
                "result": payload,
            }
        )
    return out


def normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        copy = dict(msg)
        if "content" in copy:
            copy["content"] = normalize_message_content(copy.get("content"))
        normalized.append(copy)
    return normalized


def ensure_assistant_text(content: Any) -> str:
    text = normalize_message_content(content)
    if text.strip() in {"", "[object Object]"}:
        return ""
    if "[object Object]" in text:
        return text.replace("[object Object]", "").strip()
    return text
