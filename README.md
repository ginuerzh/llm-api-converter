# llm-api-converter

English | [简体中文](README.zh-CN.md)

A GOST Rewriter HTTP plugin that converts bidirectionally between **OpenAI Chat Completions** and **Anthropic Messages API** formats. Designed for use with tools like Claude Code, Codex CLI, OpenCode, and other LLM clients that speak either protocol.

## Table of Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
  - [With Docker Compose](#with-docker-compose)
  - [Claude Code → DeepSeek (via opencode-go)](#claude-code--deepseek-via-opencode-go)
  - [Codex CLI → DeepSeek (via opencode-go)](#codex-cli--deepseek-via-opencode-go)
- [Capabilities](#capabilities)
  - [Protocol conversion](#protocol-conversion)
  - [Streaming](#streaming)
  - [Multi-tier reasoning cache (DeepSeek V4)](#multi-tier-reasoning-cache-deepseek-v4)
  - [Message sequence sanitization](#message-sequence-sanitization)
  - [Content support](#content-support)
- [CLI flags](#cli-flags)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Related projects](#related-projects)
- [License](#license)

## How it works

Deployed as a GOST rewriter plugin, it intercepts HTTP request/response bodies in a forward proxy and transparently converts between the two wire formats:

```
Anthropic SDK client → GOST → llm-api-converter → OpenAI-compatible API
                    ↕ (protocol conversion)       ↕
                Anthropic format               OpenAI format
```

The converter auto-detects the input format using **positive structural markers** only — no negative exclusions, no hardcoded model prefixes. A `SessionStore` tracks client protocol across request/response pairs for correct bidirectional routing. A URI-based fallback handles minimal requests that lack distinguishing features.

## Quick start

```bash
# Build
go build -o llm-api-converter .

# Run standalone
./llm-api-converter --addr :8000 --model-map "claude-opus=deepseek-v4-pro:openai,claude-sonnet=deepseek-v4-flash,*=deepseek-v4-flash:openai"

# Run with GOST
gost -C gost.yaml
```

### With Docker Compose

Run the converter as a container alongside GOST. The published image is `ginuerzh/llm-api-converter` (multi-arch: amd64/arm64/arm v6/v7). Since the image `ENTRYPOINT` is the binary, `command:` supplies the CLI flags.

```yaml
# docker-compose.yml
services:
  llm-converter:
    image: ginuerzh/llm-api-converter:latest
    command:
      - --addr
      - :8000
      - --model
      - deepseek-v4-flash
      - --model-map
      - claude-opus=deepseek-v4-pro:openai,claude-sonnet=deepseek-v4-flash,*=deepseek-v4-flash:openai
    ports:
      - "8000:8000"
    restart: unless-stopped

  gost:
    image: gogost/gost:latest
    command: -C /etc/gost/gost.yaml
    volumes:
      - ./gost.yaml:/etc/gost/gost.yaml:ro
    ports:
      - "8787:8787"
    depends_on:
      - llm-converter
    restart: unless-stopped
```

Point the GOST rewriter plugin at the converter's container address:

```yaml
# in gost.yaml
rewriters:
- name: llm-converter
  plugin:
    type: http
    addr: http://llm-converter:8000/rewrite
```

```bash
docker compose up -d
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude
```

Build the image locally instead of pulling:

```bash
docker build -t ginuerzh/llm-api-converter .
# or, with the multi-arch buildx workflow from .github/workflows/buildx.yml
```

### Claude Code → DeepSeek (via opencode-go)

This setup lets Claude Code (Anthropic protocol) call DeepSeek models (OpenAI protocol) through the converter:

```
Claude Code → GOST (proxy) → llm-api-converter → opencode-go API → DeepSeek
```

**1. Start the converter:**

```bash
./llm-api-converter \
  --addr :8000 \
  --model deepseek-v4-flash \
  --model-map "claude-opus=deepseek-v4-pro:openai,*=deepseek-v4-flash:openai"
```

**2. Configure GOST to intercept Anthropic API calls and forward them through the converter:**

```yaml
# gost.yaml
services:
- name: claude-code-proxy
  addr: :8787
  handler:
    type: tcp
    metadata:
      sniffing: true
  listener:
    type: tcp
  forwarder:
    nodes:
    - name: opencode-go
      addr: opencode.ai:443
      tls:
        secure: true
        serverName: opencode.ai
      http:
        host: opencode.ai
        rewriteURL:
        # Anthropic /v1/messages → OpenAI /v1/chat/completions
        - match: /v1/messages
          replacement: /zen/go/v1/chat/completions
        requestHeader:
          Authorization: "Bearer your-oc-apikey"
          x-api-key: "your-oc-apikey"
        rewriteRequestBody:
        - rewriter: llm-converter
          type: application/json
        rewriteResponseBody:
        - rewriter: llm-converter
          type: "*"

rewriters:
- name: llm-converter
  plugin:
    type: http
    addr: http://127.0.0.1:8000/rewrite
```

**3. Point Claude Code at the proxy:**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude
```

All Anthropic traffic from Claude Code is intercepted by GOST, converted to OpenAI Chat Completions format by the plugin, and forwarded to the opencode-go API for DeepSeek inference. Responses and SSE streams are converted back to Anthropic format transparently.

**Model mapping notes:**

- `claude-opus=deepseek-v4-pro:openai`: Routes requests with model name starting with `claude-opus` to DeepSeek V4 Pro, converting Anthropic→OpenAI
- `*=deepseek-v4-flash:openai`: Catch-all fallback for any unmatched model prefix, also converting to OpenAI
- **Downstream protocol override**: Append `:openai` or `:anthropic` after the target to declare what format the backend speaks (`prefix=target:protocol`). **Without a protocol suffix, the default is passthrough** — the body passes through with only the model name rewritten, no format conversion. With `:openai`/`:anthropic`, conversion runs when the incoming protocol differs from the declared one; when they match, only the model is rewritten. The override applies on both request and response paths via per-session client protocol tracking. Example: `claude-opus=deepseek-v4-pro:openai` — incoming Anthropic differs from `:openai`, so Anthropic→OpenAI conversion runs; `claude-opus=deepseek-v4-pro:anthropic` — incoming Anthropic matches, so only the model is rewritten.
- Note: `:responses` is not a valid override (only `openai`/`anthropic`); Responses API traffic is detected and routed via body markers and the session store, not the model map. Empty targets (e.g. `claude-opus=:openai`) are rejected at parse time.

Update the `--model-map` to match your opencode-go deployment's available models.

### Codex CLI → DeepSeek (via opencode-go)

This setup lets Codex CLI (OpenAI Responses API protocol) call DeepSeek models (OpenAI Chat Completions protocol) through the converter:

```
Codex CLI → GOST (proxy) → llm-api-converter → opencode-go API → DeepSeek
```

Codex CLI sends Responses API format (`POST /v1/responses` with `{model, input, ...}`); the converter translates to Chat Completions format (`POST /v1/chat/completions` with `{model, messages, ...}`) for opencode-go, and reverses the response on the way back.

**1. Start the converter:**

```bash
./llm-api-converter \
  --addr :8000 \
  --model deepseek-v4-flash \
  --model-map "gpt-4=deepseek-v4-pro:openai,*=deepseek-v4-flash:openai"
```

The `:openai` protocol override declares the downstream speaks OpenAI Chat Completions. Responses API detection runs before the passthrough check, so the request still gets full conversion (Responses → Chat); on the response path it prevents the Chat response from being wrongly converted to Anthropic format on the way back.

**2. Configure GOST to intercept Codex CLI's API calls and forward them through the converter:**

```yaml
# gost.yaml
services:
- name: codex-cli-proxy
  addr: :8787
  handler:
    type: tcp
    metadata:
      sniffing: true
  listener:
    type: tcp
  forwarder:
    nodes:
    - name: opencode-go
      addr: opencode.ai:443
      tls:
        secure: true
        serverName: opencode.ai
      http:
        host: opencode.ai
        rewriteURL:
        # Responses API /v1/responses → Chat Completions /v1/chat/completions
        - match: /v1/responses
          replacement: /zen/go/v1/chat/completions
        requestHeader:
          Authorization: "Bearer your-oc-apikey"
          x-api-key: "your-oc-apikey"
        rewriteRequestBody:
        - rewriter: llm-converter
          type: application/json
        rewriteResponseBody:
        - rewriter: llm-converter
          type: "*"

rewriters:
- name: llm-converter
  plugin:
    type: http
    addr: http://127.0.0.1:8000/rewrite
```

**3. Point Codex CLI at the proxy:**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
codex
```

Codex CLI sends Responses API requests to `/v1/responses`; GOST intercepts them, the converter rewrites the body to Chat Completions format (with model name remapping), and the request is forwarded to opencode-go with the URL rewritten to `/zen/go/v1/chat/completions`. Upstream Chat Completions responses are converted back to Responses API format transparently.

## Capabilities

### Protocol conversion

| Direction | Description |
|-----------|-------------|
| OpenAI Request → Anthropic Request | For forwarding to Anthropic API |
| OpenAI Response → Anthropic Response | For returning Anthropic-format responses to clients |
| Anthropic Request → OpenAI Request | For forwarding to OpenAI-compatible downstreams (DeepSeek, etc.) |
| Anthropic Response → OpenAI Response | For returning OpenAI-format responses to clients |
| Responses API Request → Chat Completions Request | For forwarding Codex CLI (Responses API) to OpenAI Chat Completions backends |
| Responses API Request → Anthropic Request | When the model-map routes a Responses request to an Anthropic downstream |
| Chat Completions Response → Responses API Response | Converting upstream Chat response back to Responses API format |
| Anthropic Response → Responses API Response | Converting upstream Anthropic response back to Responses API format |

### Streaming

**Anthropic SSE** — Converts OpenAI streaming delta chunks into the proper Anthropic SSE event sequence:

```
message_start → ping → content_block_start → content_block_delta* → content_block_stop → message_delta → message_stop
```

Supports text, thinking (reasoning), and tool call deltas with proper content block transitions, signature_delta for thinking blocks, and tool name restriction to prevent tool hallucination.

**Responses API SSE** — Converts Chat Completions streaming deltas into the Responses API SSE event sequence:

```
response.created → response.in_progress → output_item.added → content_part.added → response.output_text.delta* → response.output_text.done → output_item.finished → response.completed
```

Handles streaming reasoning content (`thinking` is not a first-class Responses API concept; reasoning is merged as `response.output_text.delta` with a `type: "reasoning"` annotation), text deltas, tool call accumulation across chunks, and error propagation.

### Multi-tier reasoning cache (DeepSeek V4)

Handles DeepSeek V4's requirement that `reasoning_content` must be preserved when tool calls are present. The cache stores reasoning across three tiers:

1. **Tool call ID** — exact tool call replay
2. **Tool context** — same tool pattern across different IDs
3. **Assistant text** — text-based fallback

With optional file persistence, 30-day TTL, and FIFO eviction.

The cache backend is pluggable via the `ReasoningStore` interface (`Get`, `Set`, `Delete`, `Len`), allowing custom storage implementations beyond the default in-memory map.

### Message sequence sanitization

- Pairs tool calls with their results, drops unfulfilled calls
- Converts orphan tool results to user text messages
- Merges consecutive assistant tool call messages (Claude Code conversation compression)
- Injects placeholder reasoning when DeepSeek V4 requires it but cache is empty

### Content support

- Text and multi-part content blocks
- Image data URIs (`data:image/...;base64,...`)
- Tool use / tool result blocks
- Extended thinking / reasoning content
- System messages

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--addr` | `:8000` | Listening address |
| `--model` | `deepseek-chat` | Default fallback model ID |
| `--max-tokens` | `8192` | Default max_tokens |
| `--model-map` | `` | Model mapping: `prefix=target[:protocol],...` (* for catch-all, protocol: openai\|anthropic) |
| `--cache` | `memory` | Reasoning cache backend: `memory` or `file:<path>` |
| `--log.level` | `info` | Log level |
| `--log.format` | `json` | Log format (text or json) |

## Project structure

```
llm-api-converter/
├── main.go              # Entry point
├── cmd/root.go          # Cobra CLI
├── convert/             # Core conversion logic
│   ├── types.go                              # Data types for OpenAI, Anthropic, Responses API, SSE
│   ├── convert.go                            # Entry point: Convert + ConvertSSE dispatch
│   ├── detect.go                             # Body-primary protocol detection (positive markers)
│   ├── protocol.go                           # Protocol type + URI fallback + resolveModel
│   ├── registry.go                           # ConversionKey → converter function map
│   ├── session.go                            # SessionStore (per-session state, FIFO eviction)
│   ├── anthropic_to_openai.go                # Anthropic → OpenAI Chat Completions
│   ├── openai_to_anthropic.go                # OpenAI Chat Completions → Anthropic
│   ├── responses.go                          # Responses API ↔ Chat Completions
│   ├── stream.go                             # SSE stream utilities
│   ├── stream_anthropic.go                   # Anthropic SSE state machine (OpenAI → Anthropic streaming)
│   ├── stream_responses.go                   # Responses API SSE state machine
│   ├── reasoning_cache.go                    # 3-tier reasoning cache + ReasoningStore interface
│   └── *_test.go                             # Tests
├── rewriter/
│   ├── server.go                             # HTTP plugin server
│   └── server_test.go
├── tests/e2e/                                # Integration tests
└── docs/plans/                               # Historical design documents
```

## Tests

```bash
go test ./... -v -count=1
go test ./... -race
go test ./tests/e2e/ -v -timeout 5m
```

## Related projects

- [deepseek-v4-opencode-claude-code-bridge](https://github.com/superheroYu/deepseek-v4-opencode-claude-code-bridge) — DeepSeek V4 adapter for OpenCode and Claude Code
- [opencode-cc](https://github.com/Kiowx/opencode-cc) — OpenCode Claude Code bridge
- [cc-switch](https://github.com/farion1231/cc-switch) — Claude Code provider/config switcher

## License

Part of the [GOST](https://github.com/go-gost/gost) project.
