"""OpenAI Responses API message mapping and response shaping."""

from __future__ import annotations

import time
from typing import Any

from cs_ai_bridge_api.mcp_format import ensure_assistant_text, normalize_message_content


def messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Map chat messages to Responses API ``input`` + ``instructions``.

    Assistant history uses ``output_text``; user/developer use ``input_text``.
    """
    instructions_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        text = normalize_message_content(msg.get("content"))

        if role == "system":
            if text:
                instructions_parts.append(text)
            continue

        if role == "assistant":
            out_role = "assistant"
            block_type = "output_text"
        elif role == "tool":
            out_role = "user"
            block_type = "input_text"
            if text:
                text = f"[tool]\n{text}"
        elif role in {"user", "developer"}:
            out_role = role
            block_type = "input_text"
        else:
            out_role = "user"
            block_type = "input_text"
            if text:
                text = f"[{role}]\n{text}"

        if not text and out_role != "assistant":
            continue

        out.append(
            {
                "role": out_role,
                "content": [{"type": block_type, "text": text}],
            }
        )

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, out


def extract_output_text(response: dict[str, Any]) -> str:
    """Aggregate assistant text from a Responses API JSON body."""
    top = response.get("output_text")
    if isinstance(top, str) and top:
        return top

    texts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "output_text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif block_type == "refusal" and isinstance(block.get("refusal"), str):
                texts.append(block["refusal"])
    return "".join(texts)


def extract_function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def to_chat_completion_shape(data: dict[str, Any], model: str) -> dict[str, Any]:
    """Normalize Responses API JSON to OpenAI-style chat.completion."""
    text = ensure_assistant_text(extract_output_text(data))
    created = int(time.time())
    usage_raw = data.get("usage")
    usage: dict[str, int] = {}
    if isinstance(usage_raw, dict):
        inp = usage_raw.get("input_tokens")
        outp = usage_raw.get("output_tokens")
        total = usage_raw.get("total_tokens")
        if isinstance(inp, int):
            usage["prompt_tokens"] = inp
        if isinstance(outp, int):
            usage["completion_tokens"] = outp
        if isinstance(total, int):
            usage["total_tokens"] = total

    return {
        "id": data.get("id") or f"resp-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "provider": "openai",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or None,
        "response": data,
    }
