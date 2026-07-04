# CLAUDE.md — llm-api-converter

LLM API protocol converter — GOST Rewriter plugin that converts bidirectionally between OpenAI Chat Completions, Anthropic Messages, and Google Gemini generateContent API formats. Runs as an HTTP plugin on `POST /rewrite`. Used with Claude Code, Codex CLI, OpenCode, and other LLM clients.

## Build & Run

```bash
# Build the binary
cd llm-api-converter && go build -o llm-api-converter .

# Run standalone
cd llm-api-converter && ./llm-api-converter \
  --addr :8000 \
  --model deepseek-chat \
  --max-tokens 8192 \
  --model-map "claude-opus=deepseek-v4-pro,claude-sonnet=deepseek-v4-flash"

# With protocol override (skip conversion when format matches downstream,
# but still rewrite the model name)
cd llm-api-converter && ./llm-api-converter \
  --addr :8000 \
  --model deepseek-chat \
  --model-map "claude-opus=deepseek-v4-pro:openai,*=deepseek-chat:openai"

# With debug logging
cd llm-api-converter && ./llm-api-converter --log.level debug --log.format text
```

### Running tests

```bash
# Unit tests
cd llm-api-converter && go test ./... -v -count=1

# With race detector
cd llm-api-converter && CGO_ENABLED=1 go test -race ./...

# Run specific test
cd llm-api-converter && go test ./convert/... -v -run TestConvert_SimpleUserMessage

# E2E tests (builds GOST + plugin binaries, runs real subprocess chain)
cd llm-api-converter && go test ./tests/e2e/ -v -timeout 5m

# E2E with pre-built binaries (skips compilation)
cd llm-api-converter && go test ./tests/e2e/ -v -timeout 5m \
  -gost-bin /path/to/gost \
  -plugin-bin /path/to/llm-api-converter
```

## Architecture

The converter detects the input format and converts in all directions:

```
Anthropic Request → OpenAI Request
Anthropic Request → Gemini Request
OpenAI Request → Anthropic Request
OpenAI Request → Gemini Request

Anthropic Response → OpenAI Response
Anthropic Response → Gemini Response
OpenAI Response → Anthropic Response
OpenAI Response → Gemini Response
Gemini Response → Anthropic Response
Gemini Response → OpenAI Response

Gemini SSE → OpenAI SSE delta chunks
Gemini SSE → Anthropic SSE event sequence
OpenAI SSE → Anthropic SSE event sequence
Anthropic SSE → OpenAI SSE delta chunks
```

### Package layout

| Package | Purpose |
|---------|---------|
| `main.go` | Entry point, delegates to `cmd.Execute()` |
| `cmd/root.go` | Cobra CLI: `--addr`, `--model`, `--max-tokens`, `--model-map`, `--log.level`, `--log.format` |
| `convert/` | Core conversion logic |
| `convert/types.go` | All data types: OpenAI req/resp/streaming, Anthropic req/resp, Gemini req/resp, SSE events, `ConvertOptions` |
| `convert/convert.go` | `Convert()` + `ConvertSSE()` entry points, SSE parsing, message sequence sanitization |
| `convert/detect.go` | Body-primary protocol detection (positive structural markers) |
| `convert/protocol.go` | `Protocol` type (OpenAI/Anthropic/Gemini), `detectByURI` fallback, `resolveModel` |
| `convert/registry.go` | `ConversionKey` → converter function dispatch map |
| `convert/session.go` | `SessionStore` — per-session client-protocol tracking with FIFO eviction |
| `convert/anthropic_to_openai.go` | Anthropic → OpenAI Chat (request + response) |
| `convert/openai_to_anthropic.go` | OpenAI Chat → Anthropic (request + response) |
| `convert/responses.go` | Responses API ↔ Chat Completions |
| `convert/stream.go` | `StreamConverter` — OpenAI streaming delta chunks → Anthropic SSE event sequence |
| `convert/stream_anthropic.go` | Anthropic SSE state machine |
| `convert/stream_responses.go` | Responses API SSE state machine |
| `convert/reasoning_cache.go` | 3-tier reasoning cache for DeepSeek V4 `reasoning_content` replay (tool call ID / assistant text / tool context) with file persistence |
| `convert/gemini_types.go` | Gemini `generateContent` protocol types (request, response, config, tools, safety settings) |
| `convert/openai_to_gemini.go` | OpenAI Chat ↔ Gemini (request + response) |
| `convert/anthropic_to_gemini.go` | Anthropic Messages ↔ Gemini (request + response) |
| `convert/stream_gemini.go` | Gemini SSE → OpenAI delta chunks / Anthropic event sequence streaming converters |
| `convert/gemini_test.go` | All Gemini conversion tests (~42 tests: req/resp, detection, streaming, blocked responses) |
| `rewriter/server.go` | HTTP server at `/rewrite`, dispatches to `convert.Convert()`, SSE lifecycle routing |
| `tests/e2e/` | Integration tests: GOST → plugin → mock OpenAI server |

### Key interfaces

- `convert.Convert(body []byte, opts *ConvertOptions) ([]byte, error)` — the main entry point; auto-detects format
- `convert.ConvertSSE(body []byte, opts *ConvertOptions) ([]byte, error)` — SSE-aware conversion
- `convert.HandleSSEEvent(sid, phase string, eventIndex int, data []byte, opts *ConvertOptions) ([]byte, error)` — stream lifecycle handler (start/event/end)
- `convert.NewStreamConverter(model string, reasoningCache *ReasoningCache, declaredTools []string)` — streaming state machine
- `convert.NewGeminiToOpenAIStreamConverter(model string)` — Gemini SSE → OpenAI delta chunks
- `convert.NewGeminiToAnthropicStreamConverter(model string)` — Gemini SSE → Anthropic event sequence
- `convert.NewReasoningCache(maxSize int)` — reasoning cache (DeepSeek V4)

### Auto-detection priority

`Convert()` detects the input format via `detectSource()` in `convert/detect.go` — **positive structural markers only**, no negative exclusions, no hardcoded model prefixes:
1. SSE framing (`data:`, `event:`, `id:` prefix) → `ConvertSSE`
2. Response markers: `candidates` + `content.parts` → Gemini; `choices` → OpenAI Chat; `type:"message"`+`stop_reason`/`usage` → Anthropic; `object:"response"`/`output[]` → Responses
3. Request markers: `contents` + `parts` → Gemini; `input`/`instructions` → Responses; `tools[].input_schema` → Anthropic; `tools[].function` → OpenAI Chat; `system`, `stop_sequences`, content-block types (`thinking`/`tool_use`/`tool_result`/`image`) → Anthropic; roles `system`/`tool`/`function`, `tool_calls`, `reasoning_content`, `image_url`, standard OpenAI fields (`frequency_penalty`, `stop`, `stream_options`, …) → OpenAI Chat
4. URI fallback (`detectByURI`) when the body is minimal / has no distinguishing features
5. Unknown → pass through unchanged

### Model map & protocol override

Each `--model-map` entry is `prefix=target[:protocol]`. `prefix` is matched case-insensitively against the request's `model` field (longest-prefix wins; `*` is catch-all). `target` is the rewritten model name sent downstream (required — empty targets are rejected at parse time). `:protocol` is optional:

- **unset** (no `:protocol`) — **passthrough**: body passes through with only the model name rewritten, no format conversion.
- **`:openai` / `:anthropic` / `:gemini`** — declares the downstream protocol. Conversion runs when the incoming protocol differs from it; when they match, only the model name is rewritten.

Resolution helpers (in `convert/protocol.go` + `convert/types.go`):
- `ModelMap.Apply(sourceModel) (target, protocol, ok)` — forward lookup by source prefix
- `ModelMap.lookupTargetProtocol(targetModel) string` — case-insensitive reverse lookup of a target model's protocol; used for responses whose `model` is the downstream target (not a source prefix), so the protocol override still applies on the return path
- `resolveModel(inputModel, inputProtocol, mm) (targetModel, downstreamProtocol)` — resolves target model + downstream protocol from the model map; when the entry has no `:protocol`, downstream equals input (passthrough)

The passthrough-with-model-rewrite runs in three sites:
- `Convert()` — non-streaming request/response bodies
- `ConvertSSE()` — standalone SSE events (with `lookupTargetProtocol` reverse lookup)
- `HandleSSEEvent()` — OpenAI stream chunks (model rewritten in the chunk payload)

### Streaming architecture

**OpenAI → Anthropic** (`StreamConverter`): state machine that converts OpenAI streaming deltas into the proper Anthropic SSE event sequence:

```
message_start → ping → content_block_start → content_block_delta* → content_block_stop → message_delta → message_stop
```

Tracks content block type transitions (text ↔ thinking ↔ tool_use), emits `signature_delta` for thinking blocks, accumulates tool call arguments across deltas, supports tool restriction.

**Gemini SSE → OpenAI** (`GeminiToOpenAIStreamConverter`): Gemini delivers complete `GeminiChatResponse` JSON per SSE chunk with incremental `parts[].text`. Each chunk maps independently to an OpenAI delta chunk (`data: {"choices":[{"delta":{"content":"..."}}]}`). `HandleStreamEnd()` emits the final chunk with `finish_reason` + usage + `data: [DONE]`.

**Gemini SSE → Anthropic** (`GeminiToAnthropicStreamConverter`): Maps each Gemini SSE chunk to the Anthropic event sequence. Tracks content block type transitions (text ↔ tool_use), emits `content_block_start`/`content_block_delta`/`content_block_stop` as needed. `HandleStreamEnd()` emits `message_delta` + `message_stop`.

### SSE lifecycle (GOST integration)

The GOST sniffer sends stream data via a 3-phase lifecycle:
- **start** — initiates stream converter (OpenAI → Anthropic `StreamConverter`, or Gemini → OpenAI/Anthropic converter), attaches first SSE event data
- **event** — processes individual streaming deltas, auto-creates converter if missing (resilient to out-of-order delivery)
- **end** — finalizes stream, emits remaining end-of-stream events
- **error** — emits error event in the target protocol format

For Gemini, the `HandleSSEEvent` routing creates either `GeminiToOpenAIStreamConverter` or `GeminiToAnthropicStreamConverter` based on `target` protocol. The Gemini converters emit `[DONE]` marker (OpenAI) or `message_stop` (Anthropic) on stream end.

### Reasoning cache (DeepSeek V4)

The 3-tier reasoning cache stores `reasoning_content` for replay when Claude Code compresses conversations (losing thinking blocks):

| Tier | Key | Lookup |
|------|-----|--------|
| Tool call ID | SHA256(sorted tool call IDs) | Exact tool call replay |
| Tool context | SHA256(tool context signatures + assistant text) | Same tool pattern, different IDs |
| Assistant text | SHA256(assistant text) | Text-based fallback |

`GetBest()` queries in priority order: tool call ID → tool context → assistant text. Falls back to a placeholder string when empty.

File persistence via `SetPersistence(path)` with atomic write (tmp + rename), 30-day TTL, and FIFO eviction.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--addr` | `:8000` | Listening address |
| `--model` | `deepseek-chat` | Fallback model ID for both directions |
| `--max-tokens` | `8192` | Default `max_tokens` |
| `--model-map` | `` | Model mapping table: `prefix=target[:protocol],...` (`*` for catch-all, protocol: openai\|anthropic\|gemini) |
| `--log.level` | `info` | Log level: debug, info, warn, error |
| `--log.format` | `json` | Log format: text or json |

## Message sequence sanitization

`sanitizeOpenAiToolMessageSequence()` ensures each assistant `tool_calls` message is properly paired with its tool results:
- Unfulfilled tool calls (no matching tool result) are dropped
- Orphan tool results → user text messages with context
- Duplicate tool results for the same ID → orphaned (second+ instances become user messages)
- In-progress tool calls at end of history → emitted as-is

`coalesceAdjacentAssistantToolCalls()` merges consecutive assistant messages with `tool_calls` (handles Claude Code's split tool call emission after conversation compression).

## Verification

```bash
# Unit tests + race
cd llm-api-converter && go test ./... -v -count=1 -race

# Build check
cd llm-api-converter && go build ./...
cd llm-api-converter && go vet ./...

# E2E (slow, requires workspace context)
cd llm-api-converter && go test ./tests/e2e/ -v -timeout 5m
```

## Responses API conversion (stateful stream)

`ResponsesStreamConverter` lifecycle: `HandleStreamStart` → `HandleChunk` (delta merge) → `HandleStreamEnd` (final events). Unlike Chat streaming, deltas accumulate into complete items and events carry item references.

Key gotchas:
- **Codex silently drops items with missing required fields**: codex-rs SSE parser returns `Ok(None)` on deserialization failure — no error, no log. `ResponseItem::FunctionCall.arguments` has no `#[serde(default)]`, so `output_item.done` must carry the complete item with `arguments` populated (codex ignores `function_call_arguments.delta/.done` events and reads directly from `output_item.done`).
- **`call_id` not `id`**: In Responses `function_call` items, use `call_id` as the OpenAI `tool_call.id` — codex omits the `id` field and only sets `call_id`.
- **Parallel tool call coalescing**: Multiple consecutive `function_call` items in Responses must coalesce into a single assistant message with multiple `tool_calls` for Chat Completions format (same for `custom_tool_call` items).
- **Anthropic billing header strip**: `x-anthropic-*` response headers must be stripped before returning to downstream, otherwise they cause 100% prefix cache misses on DeepSeek.

## Known behaviors

- **Protocol override passthrough**: A model-map `:protocol` suffix (`:openai`/`:anthropic`/`:gemini`) declares the downstream protocol — conversion runs when the incoming protocol differs, and is skipped (model still rewritten to `target`) when it matches. Unset protocol means passthrough (model rewrite only, no conversion). Empty targets are rejected at parse time.
- **SSE passthrough**: Non-JSON/unknown SSE data passes through unchanged (`[DONE]` markers → `message_stop`)
- **Empty messages**: After filtering (e.g., only system message), a minimal user message `"..."` is injected (Anthropic requires at least one message)
- **Tool choice**: DeepSeek models reject forced function `tool_choice`; it's softened to a system instruction instead
- **Tool restriction**: When converting upstream responses, `tool_use` blocks not in the original Anthropic request's `tools` list are filtered to prevent tool hallucination
- **ID normalization**: OpenAI `call_xxx` IDs → Anthropic `toolu_xxx` format (required by Anthropic SDK)
- **Image handling**: Only `data:` URIs with `base64` encoding are supported for image conversion
- **Gemini safety**: All safety categories set to `BLOCK_NONE` (least restrictive) — overridable in the generated Gemini request body
- **Gemini role mapping**: Anthropic/OpenAI `assistant` → Gemini `model` role; `tool`/`function` results → Gemini `functionResponse` parts in user role; `system`/`developer` → Gemini `systemInstruction` (top-level, not in contents)
- **Gemini finish reason mapping**: `STOP` → `stop`/`end_turn`, `MAX_TOKENS` → `length`/`max_tokens`, `SAFETY`/`RECITATION` etc. → `content_filter`/error
- **Gemini blocked responses**: Empty `candidates` with `promptFeedback.blockReason` → error response (`content_filter` for OpenAI, `permission_error` for Anthropic)
- **maxChunkSize requirement**: For **non-streaming** LLM responses, the GOST node's `rewriteResponseBody` rule must set `maxChunkSize` (e.g., `maxChunkSize: 2097152` for 2MB), otherwise chunked responses bypass the Rewriter plugin. Streaming (`text/event-stream`) bodies are unaffected.
