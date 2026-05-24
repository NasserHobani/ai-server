"""Normalize MCP / Odoo payloads and format them for LLM and API clients."""

from __future__ import annotations

import json
from typing import Any


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
        return json.dumps(value, ensure_ascii=False)
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
        return [normalize_field_value(item) for item in value]
    return normalize_field_value(value)


def normalize_field_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "display_name" in value or ("id" in value and "name" in value):
            return {
                "id": value.get("id"),
                "name": value.get("display_name") or value.get("name"),
            }
        return {str(k): normalize_field_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], str):
            return {"id": value[0], "name": value[1]}
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


def _records_from_payload(payload: Any) -> list[dict[str, str]] | None:
    if not isinstance(payload, dict):
        return None
    for key in ("records", "data", "result"):
        candidate = payload.get(key)
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return [normalize_record(row) for row in candidate]
    return None


def format_payload_markdown(tool_name: str, payload: Any) -> str:
    lines = [f"### MCP tool `{tool_name}`"]
    records = _records_from_payload(payload) if isinstance(payload, dict) else None
    if records:
        headers = sorted({key for row in records for key in row.keys()})
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in records:
            lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
        return "\n".join(lines)

    if isinstance(payload, dict):
        lines.append("```json")
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append(str(payload))
    return "\n".join(lines)


def format_mcp_results_for_context(mcp_results: list[dict[str, Any]]) -> str:
    sections: list[str] = [
        "Live Odoo/MCP data for this request. Present customers and related records "
        "using the field labels below as a markdown table or bullet list. "
        "Never output the literal text [object Object]; always use id/name strings "
        "from the data.",
    ]
    for entry in mcp_results:
        name = str(entry.get("name", "tool"))
        payload = entry.get("result")
        sections.append(format_payload_markdown(name, payload))
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
