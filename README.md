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
    image: ginuerzh/gost:latest
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
  --model-map "claude-opus=deepseek-v4-pro,*=deepseek-v4-flash"
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

- `claude-opus` → `deepseek-v4-pro`: Routes requests with model name starting with `claude-opus` to DeepSeek V4 Pro 
- `*` → `deepseek-v4-flash`: Catch-all fallback for any unmatched model prefix
- **Downstream protocol override**: Append `:openai` or `:anthropic` after the target to declare what format the backend speaks (`prefix=target:protocol`). When the incoming request/response already matches that protocol, conversion is skipped — **but the model name is still rewritten** to the target. Use this when routing to a backend that speaks the same protocol as the client and you only want model-name remapping. Example: `claude-opus=deepseek-v4-pro:openai` keeps OpenAI→OpenAI traffic in OpenAI form while renaming the model.
- **Optional target**: Omit the target to rewrite nothing and only skip conversion, e.g. `claude-opus=:openai` passes same-protocol traffic through with the original model name unchanged.

Update the `--model-map` to match your opencode-go deployment's available models.

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
| `--model` | `deepseek-chat | Default fallback model ID |
| `--max-tokens` | `8192` | Default max_tokens |
| `--model-map` | `` | Model mapping: `prefix=target[:protocol],...` (* for catch-all, protocol: openai\|anthropic) |
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

## Related projects

- [deepseek-v4-opencode-claude-code-bridge](https://github.com/superheroYu/deepseek-v4-opencode-claude-code-bridge) — DeepSeek V4 adapter for OpenCode and Claude Code
- [opencode-cc](https://github.com/Kiowx/opencode-cc) — OpenCode Claude Code bridge

## License

Part of the [GOST](https://github.com/go-gost/gost) project.
