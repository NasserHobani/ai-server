"""Normalize MCP / Odoo payloads and format them for clients."""

from __future__ import annotations

import json
from typing import Any

_SCHEMA_TOOL = "get_schema_metadata"


def normalize_message_content(content: Any) -> str:
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
                if isinstance(block.get("text"), str):
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
        return json.dumps(
            {str(k): field_to_display(v) for k, v in value.items()},
            ensure_ascii=False,
        )
    return json.dumps(value, ensure_ascii=False, default=str)


def normalize_record(record: dict[str, Any]) -> dict[str, str]:
    return {str(key): field_to_display(val) for key, val in record.items()}


def normalize_business_payload(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("jsonrpc") and "result" in value:
            return normalize_business_payload(value["result"])

        records = _records_from_payload(value)
        if records is not None:
            return {"records": records}

        return {
            key: (
                [normalize_record(row) if isinstance(row, dict) else field_to_display(row) for row in val]
                if key == "records" and isinstance(val, list)
                else _normalize_field_value(val)
            )
            for key, val in value.items()
        }

    if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
        return {"records": [normalize_record(row) for row in value]}
    return _normalize_field_value(value)


def _normalize_field_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "display_name" in value or ("id" in value and "name" in value):
            return field_to_display(value)
        return {str(k): _normalize_field_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], str):
            return field_to_display(value)
        return [_normalize_field_value(v) for v in value]
    return value


def unwrap_tool_payload(value: Any) -> Any:
    if not isinstance(value, dict):
        return normalize_business_payload(value)

    if any(key in value for key in ("structured_content", "content", "data")):
        for key in ("structured_content", "data"):
            if value.get(key) is not None:
                return normalize_business_payload(value[key])
        parsed = _parse_content_blocks(value.get("content"))
        if parsed is not None:
            return normalize_business_payload(parsed)

    return normalize_business_payload(value)


def _parse_content_blocks(content: Any) -> Any | None:
    if isinstance(content, str):
        return _maybe_parse_json(content)
    if isinstance(content, list):
        texts = [
            block.get("text")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
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
    if isinstance(payload, list) and payload and all(isinstance(row, dict) and row.get("id") for row in payload):
        return [normalize_record(row) for row in payload]
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
    records = _records_from_payload(payload) if payload is not None else None
    if records:
        blocks: list[str] = []
        for index, row in enumerate(records, start=1):
            lines = [f"Record {index}:"] + [
                f"- {key}: {value}" for key, value in sorted(row.items()) if value
            ]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    if isinstance(payload, dict):
        return "\n".join(
            f"- {key}: {field_to_display(val)}" for key, val in sorted(payload.items())
        )
    return field_to_display(payload) if payload is not None else ""


def format_mcp_results_text(
    mcp_results: list[dict[str, Any]],
    *,
    include_instructions: bool = False,
) -> str:
    """Plain-text MCP results for system context or fallback answers."""
    sections: list[str] = []
    if include_instructions:
        sections.append(
            "Live Odoo data below. Answer using these records; do not refuse listing business data."
        )

    for entry in mcp_results:
        if str(entry.get("name", "")).strip() == _SCHEMA_TOOL:
            continue
        formatted = format_payload_content(entry.get("result"))
        if formatted.strip():
            sections.append(formatted)

    if not sections:
        return "No matching records were found." if not include_instructions else ""
    return "\n\n".join(sections)


def normalize_mcp_results_list(mcp_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": entry.get("name"),
            "arguments": entry.get("arguments"),
            "result": unwrap_tool_payload(entry.get("result")),
        }
        for entry in mcp_results
    ]


def normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        copy = dict(msg)
        if "content" in copy:
            copy["content"] = normalize_message_content(copy.get("content"))
        out.append(copy)
    return out


def ensure_assistant_text(content: Any) -> str:
    text = normalize_message_content(content)
    if text.strip() in {"", "[object Object]"}:
        return ""
    return text.replace("[object Object]", "").strip() if "[object Object]" in text else text


def finalize_chat_response(
    result: dict[str, Any],
    mcp_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ensure ``content`` is a string and apply MCP fallback when needed."""
    choices = result.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices else {}
    content = ensure_assistant_text(message.get("content") if isinstance(message, dict) else "")

    if mcp_results is not None:
        mcp_text = format_mcp_results_text(mcp_results)
        result["mcp_results_text"] = mcp_text
        if not content or "[object Object]" in content:
            content = mcp_text
            result["answer_source"] = "mcp_fallback"

    if isinstance(choices, list) and choices:
        choices[0]["message"] = {"role": "assistant", "content": content}
    else:
        result["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    result["content"] = content

    if mcp_results is not None:
        result["mcp_results"] = mcp_results
    elif result.get("mcp_tool_executions") is not None:
        result["mcp_results"] = result["mcp_tool_executions"]
    return result
