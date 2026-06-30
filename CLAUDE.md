# CLAUDE.md — llm-api-converter

LLM API protocol converter — GOST Rewriter plugin that converts bidirectionally between OpenAI Chat Completions and Anthropic Messages API formats. Runs as an HTTP plugin on `POST /rewrite`. Used with Claude Code, Codex CLI, OpenCode, and other LLM clients.

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

The converter is a stateless HTTP plugin (GOST Rewriter protocol) that auto-detects the input format and converts in both directions:

```
Anthropic Request → OpenAI Request  (for downstream OpenAI-compatible API calls)
OpenAI Response → Anthropic Response  (returning to Anthropic SDK client)
OpenAI Request → Anthropic Request  (reverse direction)
Anthropic Response → OpenAI Response  (reverse direction)
```

### Package layout

| Package | Purpose |
|---------|---------|
| `main.go` | Entry point, delegates to `cmd.Execute()` |
| `cmd/root.go` | Cobra CLI: `--addr`, `--model`, `--max-tokens`, `--model-map`, `--log.level`, `--log.format` |
| `convert/` | Core conversion logic |
| `convert/types.go` | All data types: OpenAI req/resp/streaming, Anthropic req/resp, SSE events, `ConvertOptions` |
| `convert/convert.go` | `Convert()` auto-detect + all 4 conversion directions, SSE handling, message sequence sanitization |
| `convert/stream.go` | `StreamConverter` — OpenAI streaming delta chunks → Anthropic SSE event sequence |
| `convert/reasoning_cache.go` | 3-tier reasoning cache for DeepSeek V4 `reasoning_content` replay (tool call ID / assistant text / tool context) with file persistence |
| `rewriter/server.go` | HTTP server at `/rewrite`, dispatches to `convert.Convert()`, SSE lifecycle routing |
| `tests/e2e/` | Integration tests: GOST → plugin → mock OpenAI server |

### Key interfaces

- `convert.Convert(body []byte, opts *ConvertOptions) ([]byte, error)` — the main entry point; auto-detects format
- `convert.ConvertSSE(body []byte, opts *ConvertOptions) ([]byte, error)` — SSE-aware conversion
- `convert.HandleSSEEvent(sid, phase string, eventIndex int, data []byte, opts *ConvertOptions) ([]byte, error)` — stream lifecycle handler (start/event/end)
- `convert.NewStreamConverter(model string, reasoningCache *ReasoningCache, declaredTools []string)` — streaming state machine
- `convert.NewReasoningCache(maxSize int)` — reasoning cache (DeepSeek V4)

### Auto-detection priority

`Convert()` detects the input format in this order:
1. SSE framing (`data:`, `event:`, `id:` prefix) → `ConvertSSE`
2. Protocol-override passthrough (see [Model map & protocol override](#model-map--protocol-override))
3. Anthropic Request (`messages` with `max_tokens` + no OpenAI-specific fields) → Anthropic→OpenAI
4. OpenAI Response (`choices` array) → OpenAI→Anthropic
5. Anthropic Response (`type:"message"` + `stop_reason`/`usage`) → Anthropic→OpenAI
6. OpenAI Request (`model` or `messages` field) → OpenAI→Anthropic
7. Unknown format → pass through unchanged

### Model map & protocol override

Each `--model-map` entry is `prefix=target[:protocol]`. `prefix` is matched case-insensitively against the request's `model` field (longest-prefix wins; `*` is catch-all). `target` is the rewritten model name sent downstream. `:protocol` is optional:

- **unset** (no `:protocol`) — current auto-detect bidirectional conversion behavior (the default).
- **`:openai` / `:anthropic`** — declares the downstream protocol. When the incoming request/response already matches that protocol, **conversion is skipped but the model name is still rewritten** to `target`. Used when routing to a backend that speaks the same protocol as the client and you only want model remapping.
- **empty target** (`prefix=:openai`) — rewrite nothing, only skip conversion for same-protocol traffic.

Resolution helpers (in `convert/convert.go` + `convert/types.go`):
- `ModelMap.Apply(sourceModel) (target, protocol, ok)` — forward lookup by source prefix
- `ModelMap.LookupTarget(targetModel) string` — case-insensitive reverse lookup of a target model's protocol; used for responses whose `model` is the downstream target (not a source prefix), so the protocol override still applies on the return path
- `resolveModel(inputModel, fallback, mapping) (target, protocol)` — wraps `Apply` with passthrough/fallback

The passthrough-with-model-rewrite runs in three sites:
- `Convert()` — non-streaming request/response bodies
- `ConvertSSE()` — standalone SSE events (with `LookupTarget` reverse lookup)
- `HandleSSEEvent()` — OpenAI stream chunks (model rewritten in the chunk payload)

### Streaming architecture (OpenAI → Anthropic)

The `StreamConverter` is a stateful state machine that converts OpenAI streaming deltas into the proper Anthropic SSE event sequence:

```
message_start → ping → content_block_start → content_block_delta* → content_block_stop → message_delta → message_stop
```

It tracks content block type transitions (text ↔ thinking ↔ tool_use), emits `signature_delta` for thinking blocks, accumulates tool call arguments across deltas, and supports tool restriction (only emit `tool_use` for declared tools).

### SSE lifecycle (GOST integration)

The GOST sniffer sends stream data via a 3-phase lifecycle:
- **start** — initiates `StreamConverter`, attaches first SSE event data
- **event** — processes individual streaming deltas, auto-creates converter if missing (resilient to out-of-order delivery)
- **end** — finalizes stream, emits remaining Anthropic events (`message_delta` + `message_stop`)

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
| `--model-map` | `` | Model mapping table: `prefix=target[:protocol],...` (`*` for catch-all, protocol: openai\|anthropic) |
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

## Known behaviors

- **Protocol override passthrough**: A model-map `:protocol` suffix (`prefix=target:openai`/`:anthropic`) skips conversion for traffic already in that format, but the model name is still rewritten to `target`. Empty target (`prefix=:openai`) skips conversion and leaves the model unchanged. Unset protocol preserves the auto-detect conversion behavior.
- **SSE passthrough**: Non-JSON/unknown SSE data passes through unchanged (`[DONE]` markers → `message_stop`)
- **Empty messages**: After filtering (e.g., only system message), a minimal user message `"..."` is injected (Anthropic requires at least one message)
- **Tool choice**: DeepSeek models reject forced function `tool_choice`; it's softened to a system instruction instead
- **Tool restriction**: When converting upstream responses, `tool_use` blocks not in the original Anthropic request's `tools` list are filtered to prevent tool hallucination
- **ID normalization**: OpenAI `call_xxx` IDs → Anthropic `toolu_xxx` format (required by Anthropic SDK)
- **Image handling**: Only `data:` URIs with `base64` encoding are supported for image conversion
- **maxChunkSize requirement**: For **non-streaming** LLM responses, the GOST node's `rewriteResponseBody` rule must set `maxChunkSize` (e.g., `maxChunkSize: 2097152` for 2MB), otherwise chunked responses bypass the Rewriter plugin. Streaming (`text/event-stream`) bodies are unaffected.
