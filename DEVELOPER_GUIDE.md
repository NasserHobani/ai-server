# CS AI Bridge API — Developer Guide

FastAPI **AI Gateway** between Odoo clients, OpenAI Responses API (or Gemini), and an internal **MCP server**. The gateway is the source of truth for schema, validation, and MCP execution. OpenAI never connects to MCP directly (`type: mcp` / `server_url` are forbidden).

## Architecture

```text
Odoo Client
  → POST /v1/chat/completions
  → app.py (FastAPI)
  → llm_providers.py
      ├─ [tenant + orchestrator] mcp_orchestrator.py
      │     ├─ get_schema_metadata via MCP (schema source)
       │     ├─ mcp_tools.py (MCP → OpenAI function tools)
       │     ├─ openai_responses.py (message mapping)
       │     ├─ OpenAI /v1/responses (store: false)
       │     ├─ schema_validate.py (pre-flight checks)
       │     └─ mcp_client.py → MCP Server → Odoo
       └─ [else] plain Responses or Gemini
  → JSON chat.completion response
```

### Default path (orchestrator)

1. Client sends `tenant` + `messages`.
2. Gateway loads schema from Redis.
3. Gateway lists MCP tools and converts them to OpenAI **`type: function`** tools.
4. OpenAI Responses API runs with a **tool loop** (function_call → local MCP → function_call_output).
5. Final assistant text is returned with `mcp_tool_executions` audit log.

### Explicit MCP path

If the client sends `mcp_tool_calls`, the gateway runs those tools first, injects results as a system message, then calls the LLM **without** the orchestrator loop.

---

## Package layout

| File | Responsibility |
|------|----------------|
| `entry.py` | Uvicorn entrypoint |
| `app.py` | HTTP routes, request model, orchestration mode selection |
| `llm_config_redis.py` | AI provider config from Redis / env |
| `llm_providers.py` | Provider routing (OpenAI Responses, Gemini) |
| `openai_responses.py` | Responses API message mapping and response parsing |
| `mcp_orchestrator.py` | Tool loop: OpenAI ↔ validated MCP execution |
| `mcp_tools.py` | MCP tool → OpenAI function tool converter |
| `mcp_client.py` | FastMCP client (list tools, call tools) |
| `mcp_format.py` | Normalize Odoo payloads; format records for text |
| `schema_redis.py` | Read tenant schema from Redis |
| `schema_validate.py` | Permission checks before MCP execution |
| `schema_utils.py` | Shared schema helpers (models, operations) |

---

## File reference

### `app.py`

| Function / class | Description |
|------------------|-------------|
| `ChatCompletionRequest` | Pydantic body: `messages`, `tenant`, `provider`, optional `mcp_tool_calls` |
| `health()` | Liveness probe |
| `ready()` | Readiness (requires Redis URL) |
| `chat_completions()` | Main endpoint: config load → MCP (optional) → upstream LLM → `finalize_chat_response` |

### `llm_config_redis.py`

| Function | Description |
|----------|-------------|
| `redis_url()` | `CS_AI_BRIDGE_REDIS_URL` |
| `llm_api_key()` | OpenAI key from env |
| `read_ai_runtime_config(tenant, provider)` | Merged AI config (Redis key or env fallback) |
| `ai_config_redis_key(tenant)` | `cs_ai_bridge:ai:config:<tenant>` |

### `llm_providers.py`

| Function | Description |
|----------|-------------|
| `prepare_openai_request_body(body)` | Strips bridge-only fields, metadata, `store` |
| `merge_request_body(body, redis_cfg)` | Applies model / temperature / max_tokens defaults |
| `upstream_chat_completion(...)` | Routes to orchestrator or plain OpenAI / Gemini |
| `normalize_provider(redis_cfg)` | `openai` or `gemini` |

### `openai_responses.py`

| Function | Description |
|----------|-------------|
| `messages_to_responses_input(messages)` | Chat messages → Responses `input` + `instructions` |
| `extract_output_text(response)` | Read assistant text from Responses JSON |
| `extract_function_calls(response)` | List `type: function_call` output items |
| `to_chat_completion_shape(data, model)` | Responses JSON → `chat.completion` shape |

### `mcp_orchestrator.py`

| Function | Description |
|----------|-------------|
| `orchestrator_enabled()` | Env `CS_AI_BRIDGE_MCP_ORCHESTRATOR` (default true) |
| `max_tool_rounds()` | Max tool loop iterations (default 8) |
| `load_tenant_schema(tenant, request_id)` | Schema load via MCP `get_schema_metadata` |
| `execute_validated_mcp_tool(...)` | Validate + `call_mcp_tool` |
| `run_openai_responses_with_mcp(...)` | Full async tool loop |

### `mcp_tools.py`

| Function | Description |
|----------|-------------|
| `exposed_tool_names()` | Which MCP tools OpenAI may call (env list) |
| `mcp_tool_to_openai_function(tool_def, schema)` | One MCP tool → OpenAI function schema |
| `build_openai_function_tools(request_id, schema)` | List all converted tools |
| `schema_summary_for_instructions(schema)` | System instructions for the model |

Internal-only: `get_schema_metadata` is never exposed to OpenAI.

### `mcp_client.py`

| Function | Description |
|----------|-------------|
| `mcp_url()` | MCP server URL (`CS_AI_BRIDGE_MCP_URL`) |
| `list_mcp_tool_definitions(request_id)` | `list_tools` from MCP |
| `call_mcp_tools(calls, request_id)` | Run one or more tools |
| `call_mcp_tool(name, arguments, request_id)` | Single tool helper |
| `_jsonable(value)` | Serialize `CallToolResult` to JSON-safe data |

### `mcp_format.py`

| Function | Description |
|----------|-------------|
| `normalize_message_content(content)` | Objects → plain text (fixes `[object Object]`) |
| `field_to_display(value)` | Odoo many2one / nested values → readable string |
| `unwrap_tool_payload(value)` | Unwrap MCP `CallToolResult` / JSON-RPC |
| `format_payload_content(payload)` | Records → plain-text blocks |
| `format_mcp_results_text(results)` | All tool results as one string |
| `normalize_mcp_results_list(results)` | Normalize API `mcp_results` array |
| `normalize_chat_messages(messages)` | Sanitize incoming messages |
| `ensure_assistant_text(content)` | Safe string for assistant content |
| `finalize_chat_response(result, mcp_results)` | Set `content`, fallback, `mcp_results` |

### `schema_redis.py`

| Function | Description |
|----------|-------------|
| `normalize_tenant(value)` | Strip `cs_ai_bridge:schema:` prefix if present |
| `schema_redis_key(tenant)` | Full Redis key |
| `read_schema_metadata(tenant)` | Load and parse schema JSON |

### `schema_validate.py`

| Function | Description |
|----------|-------------|
| `validation_enabled()` | `CS_AI_BRIDGE_VALIDATE_MCP_TOOLS` |
| `validate_mcp_query(...)` | Model read, fields, domain filters |
| `validate_mcp_create(...)` | Create permission, writable fields, **required fields** |
| `validate_mcp_write(...)` | Write permission, writable fields |
| `validate_tool_call(name, args, schema, tenant)` | Dispatch validator per tool |

### `schema_utils.py`

| Function | Description |
|----------|-------------|
| `readable_models(schema)` | `schema["models"]` dict |
| `model_allows_operation(schema, model, op)` | read / create / write |
| `models_for_operation(schema, op)` | Sorted model names |
| `field_names_for_model(schema, model)` | Whitelisted field names |
| `model_label(schema, model)` | Human label from schema |

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CS_AI_BRIDGE_REDIS_URL` | Redis for AI config + schema |
| `CS_AI_BRIDGE_MCP_URL` | MCP server endpoint |
| `CS_AI_BRIDGE_MCP_ORCHESTRATOR` | `true` = default tool loop when `tenant` set |
| `CS_AI_BRIDGE_MCP_ORCHESTRATOR_MAX_ROUNDS` | Tool loop limit |
| `CS_AI_BRIDGE_VALIDATE_MCP_TOOLS` | Schema validation before MCP |
| `CS_AI_BRIDGE_SCHEMA_KEY_PREFIX` | Default `cs_ai_bridge:schema` |
| `CS_AI_BRIDGE_OPENAI_MCP_TOOLS` | Exposed tools list |
| `CS_AI_BRIDGE_LLM_API_KEY` / `OPENAI_API_KEY` | OpenAI auth |
| `CS_AI_BRIDGE_LLM_BASE_URL` | Default `https://api.openai.com/v1` |

---

## Example request

```json
POST /v1/chat/completions
{
  "tenant": "jhzly",
  "messages": [
    { "role": "user", "content": "Show me the latest invoices" }
  ]
}
```

OpenAI may call `mcp_query` with `model: account.move`; the gateway validates, executes MCP, returns data to OpenAI, then the final natural-language answer.

---

## Response fields

| Field | Description |
|-------|-------------|
| `content` | Final assistant message (string) |
| `choices[0].message.content` | Same text |
| `mcp_tool_executions` | Orchestrator audit log |
| `mcp_orchestrator_rounds` | Number of Responses API rounds |
| `mcp_results` | Copy of tool executions (or explicit prefetch results) |

---

## Removed in refactor

- **`mcp_intent.py`** — legacy auto-query / NL intent matching (replaced by OpenAI tool choice in orchestrator).
- Duplicate Responses helpers in `llm_providers.py` (moved to `openai_responses.py`).
- Duplicate schema model helpers (consolidated in `schema_utils.py`).
- `list_openai_function_tools` wrapper in `mcp_client.py` (use `mcp_tools.build_openai_function_tools`).
