# llm-api-converter

A GOST Rewriter HTTP plugin that converts bidirectionally between **OpenAI Chat Completions** and **Anthropic Messages API** formats. Designed for use with tools like Claude Code, OpenCode, and other LLM clients that speak either protocol.

## How it works

Deployed as a GOST rewriter plugin, it intercepts HTTP request/response bodies in a forward proxy and transparently converts between the two wire formats:

```
Anthropic SDK client → GOST → llm-api-converter → OpenAI-compatible API
                    ↕ (protocol conversion)       ↕
                Anthropic format               OpenAI format
```

The converter auto-detects the input format — no manual routing needed.

## Quick start

```bash
# Build
go build -o llm-api-converter .

# Run standalone
./llm-api-converter --addr :8000 --model claude-sonnet-4-20250514 --downstream deepseek-chat

# Run with GOST
gost -C gost.yml -D
```

### GOST config

```yaml
services:
- name: anthropic-proxy
  addr: :8080
  handler:
    type: forward
    metadata:
      sniffing: true
  listener:
    type: tcp
  forwarder:
    nodes:
    - name: upstream
      addr: <your-api-host>:443
      http:
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

## Capabilities

### Protocol conversion

| Direction | Description |
|-----------|-------------|
| OpenAI Request → Anthropic Request | For forwarding to Anthropic API |
| OpenAI Response → Anthropic Response | For returning Anthropic-format responses to clients |
| Anthropic Request → OpenAI Request | For forwarding to OpenAI-compatible downstreams (DeepSeek, etc.) |
| Anthropic Response → OpenAI Response | For returning OpenAI-format responses to clients |

### Streaming (OpenAI SSE → Anthropic SSE)

Converts OpenAI streaming delta chunks into the proper Anthropic SSE event sequence:

```
message_start → ping → content_block_start → content_block_delta* → content_block_stop → message_delta → message_stop
```

Supports text, thinking (reasoning), and tool call deltas with proper content block transitions, signature_delta for thinking blocks, and tool name restriction to prevent tool hallucination.

### Multi-tier reasoning cache (DeepSeek V4)

Handles DeepSeek V4's requirement that `reasoning_content` must be preserved when tool calls are present. The cache stores reasoning across three tiers:

1. **Tool call ID** — exact tool call replay
2. **Tool context** — same tool pattern across different IDs
3. **Assistant text** — text-based fallback

With optional file persistence, 30-day TTL, and FIFO eviction.

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
| `--model` | `claude-sonnet-4-20250514` | Target Anthropic model |
| `--max-tokens` | `8192` | Default max_tokens |
| `--downstream` | `deepseek-chat` | Downstream OpenAI model |
| `--log.level` | `info` | Log level |
| `--log.format` | `json` | Log format (text or json) |

## Project structure

```
llm-api-converter/
├── main.go              # Entry point
├── cmd/root.go          # Cobra CLI
├── convert/             # Core conversion logic
│   ├── types.go         # Data types
│   ├── convert.go       # Conversion + auto-detection
│   ├── stream.go        # SSE stream state machine
│   ├── reasoning_cache.go # 3-tier reasoning cache
│   └── *_test.go        # Tests
├── rewriter/
│   ├── server.go        # HTTP plugin server
│   └── server_test.go   # Rewriter tests
└── tests/e2e/           # Integration tests
```

## Tests

```bash
go test ./... -v -count=1
go test ./... -race
go test ./tests/e2e/ -v -timeout 5m
```

## License

Part of the [GOST](https://github.com/go-gost/gost) project.
