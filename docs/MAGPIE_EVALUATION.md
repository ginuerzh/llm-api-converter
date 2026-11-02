# magpie evaluation for llm-api-converter

Date: 2026-11-02
Reviewed revision: `ea51526` (`github.com/yetone/magpie`, mirrored at
`http://git.home.pi/mirror/magpie`)
License: MIT (Copyright (c) 2026 yetone)

## Summary

`magpie` is a local LLM gateway and multi-agent model router. Its gateway
speaks the same four wire APIs this project converts between (OpenAI Chat
Completions, OpenAI Responses, Anthropic Messages, Gemini generateContent),
including streaming, tool calls and reasoning. That makes it the closest
public implementation to `llm-api-converter` that we have found.

The conclusion is:

- **Do not add magpie as a dependency.** It is an application, not a
  library: every conversion package is under `internal/`, and the code is
  coupled to its provider registry, account routing, plugins, masking and
  billing.
- **Do not port the gateway wholesale.** The conversion core alone is on
  the order of 10-13k lines, and most of it exists to serve magpie's
  product features, not protocol conversion.
- **Do port selected behaviours.** Several small, self-contained pieces
  solve real gaps in this project and are cheap to reimplement with tests.
- **`count_tokens` cannot be fixed by a rewriter alone.** A GOST rewriter
  plugin can replace body bytes, but not the request URI or the response
  status. Serving `POST /v1/messages/count_tokens` needs a local endpoint
  or a change in the GOST routing setup (details below).

## What magpie is

| Item | Value |
| --- | --- |
| Purpose | Menu-bar/desktop app, TUI, web UI, CLI and Docker gateway that wires agents to providers and accounts |
| Gateway address | `127.0.0.1:3425` |
| Protocols | OpenAI Chat, OpenAI Responses, Anthropic Messages, Gemini |
| Conversion model | Canonical intermediate representation (IR) plus per-API parse/build/encode |
| Repository size | ~228 MB, ~4,800 Go files |
| `internal/gateway` | ~101k non-test lines, 990 test files |
| Activity | Very high (tag `v0.1.2717`; 50 commits since 2026-10-01) |
| Upstream policy | README: the repository does not accept external pull requests |

The conversion entry points are:

- `internal/gateway/ir.go` — `Request`, `Message`, `Part`, `Tool`, `Event`,
  `Usage`, `Result` and the stream `collector`.
- `internal/gateway/chat.go`, `anthropic.go`, `gemini.go`,
  `responses.go` — parse/build plus the SSE encoders.
- `internal/gateway/sse.go` — SSE reader/writer and keepalives.

Magpie normally passes a same-protocol body through with targeted
rewrites and only uses the IR when the upstream speaks a different API.
That is an important difference from this project, which converts every
body through typed structs.

## Reuse assessment

| Level | Feasible | Notes |
| --- | --- | --- |
| Import as a Go module | No | `internal/` blocks external imports; no public conversion package |
| Vendor the converter files | No | They import `internal/provider`, `internal/catalog`, `internal/settings` and many gateway helpers |
| Copy a single algorithm | Yes, with rewrite | MIT allows it; most useful functions are small but reference local helpers |
| Reuse behaviour and test fixtures | Yes | The highest-value option; see the candidate list |

## Architecture comparison

`llm-api-converter` uses a pairwise registry:

- `convert/registry.go` maps `{From, To}` to one conversion function.
- Four protocols produce up to 12 directed pairs (plus the Responses
  special cases).
- Each pair has its own request/response converter and, for streaming, its
  own state machine.

Magpie uses hub-and-spoke:

- Parse any request into one `Request`.
- Build the upstream request in its own API shape.
- Decode replies into `Event`s.
- Encode `Event`s back into the client's API.

Trade-offs:

| | Pairwise (this project) | IR (magpie) |
| --- | --- | --- |
| Current size | Small, explicit | Large, shared |
| Adding a 5th protocol | N new pairs (4 → 8 directions) | One parser + one builder + one encoder |
| Field fidelity | Typed structs can drop unknown fields | Passthrough path preserves unknown fields |
| Debuggability | Easy to trace one direction | Requires understanding the IR and encoder state |
| Migration risk | None | High for 16k lines of working code |

Recommendation: keep the pairwise design. Revisit the IR only when a fifth
protocol is actually planned. At three or four protocols the pairwise
matrix is still smaller than the shared machinery.

## Candidate work, by priority

### P0 — small, self-contained, clear user benefit

#### 1. Cache-token accounting

Problem: this project drops cache usage when converting responses.

- `OpenAIUsage` only has `prompt_tokens`, `completion_tokens`,
  `total_tokens` (`convert/types.go:99`).
- OpenAI → Anthropic keeps `prompt_tokens` as `input_tokens`
  (`convert/openai_to_anthropic.go:523`), so a cached prompt looks like a
  full-price prompt to Claude Code.
- The OpenAI → Anthropic stream does the same in `message_delta`
  (`convert/stream.go:233`).
- Anthropic → OpenAI drops `cache_read_input_tokens` and
  `cache_creation_input_tokens` (`convert/anthropic_to_openai.go:514`).

Magpie's behaviour (worth copying):

- OpenAI's `prompt_tokens` includes tokens read from and written to cache.
  Anthropic's `input_tokens` excludes both (`internal/gateway/ir.go`,
  `Usage.prompt`, and `internal/gateway/chat.go`, `cUsage.usage`).
- It reads `prompt_tokens_details.cached_tokens`,
  `prompt_tokens_details.cache_write_tokens`, DeepSeek's
  `prompt_cache_hit_tokens`, Moonshot's top-level `cached_tokens`, and
  Anthropic's `cache_read_input_tokens` /
  `cache_creation_input_tokens` when a relay emits them in an
  OpenAI-shaped body.
- On the way back it emits `prompt_tokens_details.cached_tokens` and
  `cache_write_tokens`, and includes both in `prompt_tokens`.
- Gemini's `promptTokenCount` includes cached tokens;
  `cachedContentTokenCount` carries the cache read count
  (`internal/gateway/gemini.go`, `Usage.gemini`).

Proposed conversion rules:

| Direction | Rule |
| --- | --- |
| OpenAI → Anthropic | `cache_read = cached_tokens (or vendor spelling)`, `cache_write = cache_write_tokens`, `input_tokens = prompt_tokens - cache_read - cache_write` (clamped at 0) |
| Anthropic → OpenAI | `prompt_tokens = input_tokens + cache_read + cache_creation`, `cached_tokens = cache_read`, `cache_write_tokens = cache_creation` |
| Gemini → Anthropic | `cache_read = cachedContentTokenCount`, `input_tokens = promptTokenCount - cache_read` |
| Gemini → OpenAI | `cached_tokens = cachedContentTokenCount`, `prompt_tokens = promptTokenCount` |
| OpenAI → Gemini | `cachedContentTokenCount = cache_read`, `promptTokenCount = prompt_tokens` |
| Anthropic → Gemini | `promptTokenCount = input + cache_read + cache_creation`, `cachedContentTokenCount = cache_read` |

Apply it to non-streaming responses and all streaming encoders
(`convert/stream.go`, `convert/stream_gemini.go`,
`convert/stream_anthropic.go`). Add `OpenAIPromptTokensDetails` and the
vendor aliases to `OpenAIUsage`; they are ignored on marshal when zero.

Tests to add: cached hit, DeepSeek spelling, cached greater than prompt,
cache creation only, and a streaming `message_delta` case.

#### 2. Gemini thought signatures keyed by tool-call ID

Problem: this project stores Gemini signatures per session and per
function name:

- `convert/session.go:20` — `ThoughtSignatures: sid -> funcName -> sig`.
- Written by `convert/anthropic_to_gemini.go:262` and read by
  `convert/openai_to_gemini.go:221`.

Consequences:

- Two calls to the same tool in one turn overwrite each other.
- Parallel calls with different signatures collide.
- A new session ID or an evicted session loses the signature, and Gemini
  rejects the next request with "missing a thought_signature".

Magpie's approach (`internal/gateway/gemini_signature.go`,
`internal/gateway/gemini_sigstore.go`):

- Encode the signature into the tool-call ID that the client receives:
  `<id>__ts<n>__<base64url(sig)>`.
- If the signature is long, store it by hash and put a short key in the
  ID: `<id>__tsk16__<key>`.
- The length marker distinguishes a whole signature from one truncated by
  a client (some clients cap IDs at 40 characters).
- A missing or shortened signature degrades to
  `skip_thought_signature_validator`.

A minimal port for this project: keep an in-memory `signature -> key`
store (the existing session store can own it), encode by call ID, and
strip the marker on the request path. Do not key by function name.

#### 3. SSE hygiene for relayed streams

`PassthroughStreamHandler` (`convert/session.go:113`) relays upstream
events as-is. Real vendors emit shapes that strict clients reject.
Magpie documents and repairs these in `internal/gateway/chattidy.go`:

- `reasoning_content: ""` on every delta after thinking ends — some
  clients treat it as thinking restarting.
- `tool_calls: []` on text deltas — some clients close the text block.
- `finish_reason: ""` instead of `null` — the AI SDK rejects the chunk.
- Repeated empty `function.name` in tool-call fragments.
- Missing `tool_calls[].index` (Gemini's OpenAI-compatible endpoint).
- `finish_reason: "stop"` when the chunk contains tool calls; clients
  then end the turn without running them.
- Usage cache counts spelled in another vendor's field names.

Not all of these apply to every relayed protocol, but they are a good
checklist and each fix is small.

### P1 — medium effort, broader protocol coverage

#### 4. Structured-output bridging

Magpie normalises `response_format` (Chat), `text.format` (Responses),
`output_config.format` (Anthropic) and Gemini's
`responseMimeType`/`responseSchema` into a single `Format`
(`internal/gateway/format.go`). When an upstream refuses the field, it
retries with the schema stated in the system prompt and remembers the
refusal for that model (`format_fence.go` extracts a fenced JSON answer).

This project only handles structured output in the Gemini direction
(`convert/openai_to_gemini.go:65`). Worth porting if users route
structured-output requests through other pairs.

#### 5. Reasoning and thinking signatures

This project already has a three-tier DeepSeek reasoning cache
(`convert/reasoning_cache.go`), which magpie does not have in the same
form. Magpie has complementary pieces:

- `internal/gateway/reasoning_carry.go` — carries sealed Responses
  reasoning items in tool-call IDs across Chat/Anthropic/Gemini clients.
- `internal/gateway/thinking_signature.go` — always emits a `signature`
  field on Anthropic thinking blocks (empty when unknown); some clients
  reject a block without the field.
- `internal/gateway/thinking_bound.go` — detects Claude's "signature
  bound to a different conversation" refusal, drops the offending block
  and retries.

Compare with the existing cache before changing anything; the call-ID
carrying technique is the part most likely to be useful.

#### 6. Gemini/Vertex schema typing

This project strips unsupported keywords in `sanitizeGeminiSchema`
(`convert/openai_to_gemini.go:799`). Magpie additionally infers and
inserts `type` on untyped schema nodes (`internal/gateway/gemini_schema.go`)
because Vertex AI rejects a function declaration whose parameter schema
lacks a type. Relevant for MCP tools sent through an OpenAI → Gemini
translation.

### P2 — investigate only when the need appears

#### 7. `count_tokens`

Claude Code sends `POST /v1/messages/count_tokens`. This project has no
handling for it (`grep count_tokens` returns nothing), and the request is
forwarded to the OpenAI-compatible upstream, which normally has no such
endpoint.

Why a rewriter cannot fix it alone:

- `core/rewriter/rewriter.go` exposes only `Rewrite(ctx, body) ([]byte, error)`.
- The HTTP plugin protocol (`x/rewriter/plugin/http.go`) carries only
  `{ok, data}`; there is no status or URI override.
- `rewriteURL` runs in the forwarder before the body rewriter, and it
  maps a path, not a status. A failed upstream response keeps its status.

Viable options, in increasing cost:

1. Add a local `POST /v1/messages/count_tokens` endpoint to this server
   (a character-based estimate is enough; magpie uses `len(text)/4` in
   `internal/gateway/gateway.go`, `estimate`) and document a dedicated
   GOST node/route that sends that path here.
2. Extend GOST's rewriter contract to allow a short-circuit response.
   This is a cross-repository change and needs a design decision.
3. Do nothing, if Claude Code treats a failed count as non-fatal in the
   targeted versions. This is unverified; do not assume it.

A local estimate is cheap and deterministic. Do not forward the request
to an upstream that cannot answer it.

#### 8. Canonical IR refactor

Only worth it when adding a fifth protocol. Do not rewrite a working
16k-line pairwise converter for symmetry alone.

#### 9. Fixtures and vendor-quirk checklist

Magpie keeps real vendor payloads under
`internal/gateway/testdata/` (~624 KB, including SSE streams and
`upstream_bodies`). These can seed conformance tests here. The comment
style in magpie records the client, vendor and issue behind each fix;
that list is a useful compatibility checklist even when the code is not
ported.

## Risks and constraints

- **License**: MIT permits copying with the copyright notice. Add the
  notice to `THIRD_PARTY_NOTICES` if code is copied rather than
  reimplemented.
- **Governance**: upstream does not take pull requests. Ported code
  cannot be maintained by upstream, and bug reports go through Discord.
- **Churn**: magpie releases very frequently. Pin a commit (`ea51526`)
  when comparing behaviour; do not track `main`.
- **Coupling**: its conversion functions assume magpie's `provider`,
  `catalog`, `settings` and account model. Copying them directly will
  bring unrelated dependencies into this project.
- **Scope**: magpie's prompt caching, masking, routing groups, quota
  accounting and subscription bridges are product features, not protocol
  conversion. They are out of scope here.

## Implementation status

2026-11-02: the P0 items above and a follow-up audit are implemented in the
working tree.

| Item | Files | Tests |
| --- | --- | --- |
| Cache-token accounting across all directions and streams | `convert/usage.go`, `convert/types.go`, `convert/gemini_types.go`, the OpenAI/Anthropic/Gemini converters and stream handlers | `convert/usage_test.go` |
| Gemini thought signature kept per tool-call ID | `convert/session.go` (`GetThoughtSigFor`, `SetThoughtSigFor`), Gemini request/response and SSE converters | `convert/gemini_signature_test.go` |
| OpenAI passthrough stream hygiene | `convert/tidy.go`, `convert/session.go` | `convert/tidy_test.go` |
| Anthropic SSE → OpenAI Chat streaming | `convert/stream_anthropic_to_openai.go`, `convert/convert.go` | `convert/audit_test.go` |
| Gemini thought parts and thinkingConfig | `convert/gemini_types.go`, the Gemini converters and stream handlers | `convert/audit_test.go` |
| Gemini lifecycle signatures and cache usage | `convert/stream_gemini.go`, `convert/convert.go` | `convert/audit_test.go` |
| Responses API cache usage (whole and streaming) | `convert/usage.go`, `convert/responses.go`, `convert/stream_anthropic.go`, `convert/stream_responses.go` | `convert/audit_test.go` |
| Tool-call index inference and restriction bypass | `convert/types.go`, `convert/stream.go`, `convert/stream_responses.go` | `convert/audit_test.go` |
| Bounded thought-signature store | `convert/session.go` | `convert/audit_test.go` |

The end-to-end harness also passed `--downstream`, a flag that no longer
exists; it now uses `--model-map "*=deepseek-chat:openai"` instead
(`tests/e2e/rewriter_e2e_test.go`).

Verification run:

```sh
cd llm-api-converter
go test ./... -count=1
CGO_ENABLED=1 go test -race ./convert ./rewriter -count=1
go vet ./...
```

### Known gaps after the audit

- `count_tokens` still has the rewriter-contract limitation described
  above.
- The stateless `ConvertSSE` Gemini → Anthropic fallback still emits text
  deltas without a text `content_block_start`; the lifecycle path emits
  complete blocks. Thinking blocks are emitted on both paths now.
- Gemini clients pointed at an OpenAI Responses upstream have no dedicated
  streaming converter (Anthropic and OpenAI Chat upstreams do).

## Recommendation

1. Keep this project's pairwise architecture and GOST rewriter role.
2. Implement P0 items in this order:
   cache-token accounting, Gemini signature by call ID, SSE hygiene.
3. Add regression tests for each, using real vendor payload shapes from
   magpie's `testdata` where useful.
4. Decide separately whether `count_tokens` deserves a local endpoint and
   routing documentation. Do not attempt it as a pure body rewrite.
5. Re-evaluate the IR refactor only when a new protocol is planned.

## Verification

Commands used for this evaluation:

```sh
git clone http://git.home.pi/mirror/magpie /tmp/opencode/magpie
cd /tmp/opencode/magpie
git log -1 --format='%h %ci %s'
find internal/gateway -name '*.go' | wc -l          # 1247
cat internal/gateway/*.go | wc -l                  # 289090
ls internal/gateway/*_test.go | wc -l              # 990
grep -rn "func (s \*Server) countTokens" internal/gateway/gateway.go
grep -rn "cachedContentTokenCount" internal/gateway
```

Paths beginning with `internal/` in this document are relative to the
magpie repository root. Paths beginning with `convert/` or `rewriter/`
are relative to this repository. Paths beginning with `core/` or `x/`
are relative to the go-gost workspace root.
