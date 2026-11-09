# Anthropic Client → Responses Upstream Design

Date: 2026-11-09
Repository: `llm-api-converter`
Base revision: `8d40736`
Driving incident: cuttle `bailian-dsf` pinned to `node.api: responses` returned
`400 Unknown parameter: 'messages'. Expects 'input' instead of 'messages'`
for Claude Code `/v1/messages` traffic.
Reference implementation: magpie (`github.com/yetone/magpie`), read as a
behaviour reference only.

## Intent

Make an Anthropic Messages client (Claude Code) able to use a provider whose
only usable API is OpenAI Responses. The request, the non-streaming response,
and the streaming SSE response must all cross the protocol boundary; the
upstream must never receive an Anthropic body on the Responses path.

The first concrete target is cuttle → `aliyun-bailian` →
`deepseek-v4.1-flash-1017`. The design stays general enough for any
Responses-compatible upstream.

## Goals

- `POST /v1/messages` with and without `stream: true` routed to an upstream
  Responses API returns a valid Anthropic response.
- Anthropic text, system blocks, images, `tool_use` / `tool_result`, tools,
  `tool_choice`, `disable_parallel_tool_use`, `max_tokens`, `temperature`,
  `top_p`, `thinking`, `output_config.effort`, `output_config.format` and
  `metadata` are mapped where the Responses API has an equivalent.
- Responses text, reasoning, function calls, status, incomplete details and
  usage are mapped back to Anthropic content blocks, `stop_reason` and usage.
- Streaming maps Responses text, reasoning (`reasoning_text` and
  `reasoning_summary_text` variants) and function-call argument deltas to the
  Anthropic event sequence, and ends with `message_delta` + `message_stop`.
- A missing conversion is never silent: the request either converts or fails
  with a diagnostic, instead of forwarding an Anthropic body to a Responses
  endpoint.
- Existing Anthropic↔Chat, Chat↔Gemini and Anthropic↔Gemini conversions are
  unchanged.

## Non-goals

- Chat Completions clients → Responses upstream. That is a separate pair and a
  separate trigger for the IR discussion.
- Fixing Responses-client → Responses-upstream passthrough. It is outside this
  pair; if it is broken, it gets its own change.
- Sealed/encrypted reasoning replay, web search, custom tools, namespaced
  tools, deferred tools, and prompt-cache breakpoint semantics.
- `count_tokens` against the upstream. cuttle answers that locally for this
  target.
- Adopting magpie's IR or importing magpie as a dependency.
- The legacy stateless `ConvertSSE` route. It is already documented as
  incomplete; the supported stream route is `HandleSSEEvent`.

## Anti-goals

- Do not send `messages` to a Responses endpoint as a "best effort".
- Do not invent a successful response when the upstream returned a failure.
- Do not emit a thinking block without the `signature` field; some Anthropic
  clients reject it.
- Do not change existing pair behaviour or registry contracts to make this pair
  fit.
- Do not force `store: true` or depend on upstream conversation state. Every
  request is self-contained.

## Constraints

- Existing pairwise architecture; no new dependency.
- `go test -count=1 ./...`, `go vet ./...`, and the provider contract
  validators must pass.
- Code comments are English.
- Commit after each plan task with its implementation record. Never push.
- cuttle consumes the converter by tag. The converter change is released first;
  cuttle then bumps the tag. Record both repositories.
- `ResponsesRequest` / `ResponsesResponse` / stream event types live in
  `convert/responses.go`; extend them there rather than creating a parallel
  type set.

## Evidence from the target upstream

Read-only probes against
`https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/responses`
with the live `bailian-dsf` model produced:

- Non-streaming text, streaming text, function calls and reasoning all work.
- Streaming events are the standard Responses sequence:
  `response.created`, `response.in_progress`, `response.output_item.added`,
  `response.output_text.delta`, `response.output_text.done`,
  `response.content_part.done`, `response.output_item.done`,
  `response.completed`.
- Function calls arrive as a `function_call` output item with `call_id`, and
  arguments in `response.function_call_arguments.delta/done`. A `call_id` that
  looks like an Anthropic `toolu_...` id is accepted.
- Reasoning arrives as `response.reasoning_text.delta/done`, not OpenAI's
  `response.reasoning_summary_text.*`.
- A conversation containing `function_call` / `function_call_output` without a
  reasoning item is accepted. Reasoning replay is therefore optional for this
  upstream.
- `input_tokens` includes cached tokens; `input_tokens_details.cached_tokens`
  and `cache_write_tokens` are present.
- SSE frames carry extra `id: N` and `:HTTP_STATUS/200` lines; the converter
  receives the `data:` payload through the existing lifecycle and must not rely
  on those extra lines.

## Approaches considered

**A. Pairwise converters plus a stateful stream encoder (chosen).**
Add two request/response functions and one `responsesDecoder → Anthropic
encoder` state machine, all local to the new pair. Register only the two needed
directions. This matches the repository's existing style and ships fastest.

**B. Pair-local IR modelled on magpie.**
Parse both protocols into a private `Request`/`Message`/`Part`/`Tool`, then
build the other protocol. Cleaner for a large feature surface, but it adds an
abstraction layer before the first working version and risks a half-adopted IR.

**C. Port magpie's IR across all protocols.**
Rejected. It replaces working pairwise code, changes every hot path, and is far
larger than the incident requires. The project deliberately stays pairwise;
see the long-term decision below.

## Long-term architecture decision

Keep the pairwise architecture. Do not migrate `llm-api-converter` to magpie's
IR, now or as a planned destination.

Position recorded 2026-11-09: there are only four mainstream wire protocols.
Even full all-pairs handling is at most 4×4 = 16 directed conversions
(12 cross-protocol directions plus 4 same-protocol passthroughs). Each
direction is independent, coupling stays low, and a self-contained body
transformer is exactly the responsibility of a GOST rewriter plugin. An IR is
useful for a gateway that also owns routing, accounts, plugins, masking and
billing — magpie's shape — not for this plugin.

This also fits the existing code:

- Each pair has its own request/response converter and, when needed, its own
  stream state machine. A change to one pair cannot alter another.
- The registry key is the public contract. Adding a direction is additive and
  reviewable on its own.
- magpie's IR is not a separable library. `internal/gateway/ir.go` is ~981
  lines and the protocol readers/builders are ~7k lines, coupled to magpie's
  catalog, provider, account and product features. Its gateway tests are
  roughly 200k lines. Porting would be a project of its own and would replace
  working code.
- The incident was a dispatch/coverage gap, not an IR-shaped bug. A shared IR
  would not have registered the missing pair by itself; tests and explicit
  routing do that.

The IR remains a behavioural reference, not an architecture target. Revisit
only if the project stops being a rewriter plugin and becomes a stateful
gateway, or if the protocol set grows beyond the four mainstream APIs. Even in
that case the pairwise registry stays the public contract and any shared model
would be introduced one pair at a time, against golden fixtures.

For this change, keep the pair self-contained behind two pure functions:
`anthropicRequestToResponses` and `responsesResponseToAnthropic`. They take and
return bytes plus `ConvertOptions` and contain no engine/session state, so the
pair remains easy to test and to change independently.

## Builder-check resolutions

A fresh pass over the design against the existing code raised eight points.
They are settled here so the plan does not have to guess:

1. **Interrupted streams.** Follow the existing lifecycle rule: when the end
   phase carries `opts.ErrorMsg`, `ResponsesToAnthropicStreamConverter` emits an
   Anthropic `error` event instead of a graceful `message_stop`, and it is added
   to the known-stateful-handler switch in `HandleSSEEvent`. If the stream ends
   without a completion and without `ErrorMsg`, close the open block and finish
   gracefully. If an open tool block's accumulated arguments are not valid JSON,
   treat it as a terminal error, matching the P2 truncated-tool rule.
2. **How cuttle selects the pair.** cuttle uses `engine.Options.ResolveModel`
   and returns `engineProtocol(target.api)`, which already yields
   `"responses"`. The CLI `ParseModelMap` only accepts `openai|anthropic|gemini`
   and is not on cuttle's path. Add `responses` to `ParseModelMap` for CLI
   parity, but no engine change is required for cuttle.
3. **Parallel function calls.** Bailian groups one call at a time (probed with
   two tools). The converter still keeps per-item argument buffers and releases
   calls in output order, so an upstream that interleaves argument deltas cannot
   corrupt a block. A newer call's deltas are held until the open call closes.
4. **Thinking signature.** Non-stream responses emit `"signature": ""`.
   Streams emit `"signature": ""` in the `content_block_start` and do **not**
   emit an empty `signature_delta`, because there is no real signature. The
   Anthropic contract validator currently requires a non-empty signature for
   non-DeepSeek models and does not list `signature_delta`; extend it with a
   translated-response allowance rather than weakening the existing rule for
   native Anthropic responses.
5. **Streamed `response.incomplete`.** Use the same rule as the non-stream
   path: explicit `max_output_tokens` → `max_tokens`, `content_filter` →
   `refusal`; otherwise any tool call → `tool_use`, else `end_turn`. An
   explicitly truncated turn stays `max_tokens` so the client can see the cut,
   even when a tool call is present.
6. **Missing session on stream start.** Prefer `Session.To`/`Session.From`.
   If the session is absent, fall back to `detectByURI(opts.URI,
   DirectionRequest)` (the inbound `/v1/messages` URI) before using model-map
   resolution. This prevents `resolveModel(upstreamModel, Responses, ...)` from
   resolving back to `Responses` and taking the passthrough branch. A bare
   `{"type":"error"}` first event is routed by the session/URI, not by
   `detectSource` alone.
7. **One-way registry entries.** The response path resolves the client protocol
   first and routes a Responses client to `convertToResponsesResponse` before
   consulting the registry. The new converters also guard by body shape
   (`messages` for the request direction, `object:"response"` for the response
   direction), so a misrouted body cannot be fed to the wrong function. Unknown
   pairs keep the existing passthrough; the new pair always has a converter.
8. **Fixtures and validators.** There is no Responses contract validator in the
   repository. Add direct shape assertions for the new Responses request and
   response, and check in the captured Bailian streams as
   `convert/testdata/responses/*.sse` (text, tool, reasoning, parallel tool).
   Extending the full contract validator to Responses is optional and not a
   prerequisite for this pair.

## Design

### New files

| File | Responsibility |
| --- | --- |
| `convert/anthropic_responses.go` | Request conversion, non-streaming response conversion, shared field helpers |
| `convert/stream_responses_to_anthropic.go` | Responses SSE → Anthropic SSE state machine |
| `convert/anthropic_responses_test.go` | Request/response unit tests and routing tests |
| `convert/stream_responses_to_anthropic_test.go` | Stream tests using captured Bailian fixtures |

### Type changes

`convert/responses.go`:

- Add `ResponsesReasoning{Effort, Summary string}` and `Text *ResponsesText`
  to `ResponsesRequest`.
- Add `ResponsesText{Format map[string]any}`.
- Add `CacheWriteTokens int` to `ResponsesUsageDetails`.
- Add `Name string` to `ResponsesStreamEvent` (used by
  `response.function_call_arguments.done`).
- Keep `Store *bool`; the converter sets it to `false`.

No change to `AnthropicRequest` / `AnthropicContent` is required for the
request path. For the response path, use a small local output block type so a
thinking block can carry `"signature": ""` without adding that field to every
Anthropic content block in the existing pairs.

`engine.ParseModelMap` (used by the CLI, not by cuttle) currently accepts only
`openai`, `anthropic` and `gemini`. Add `responses` so the documented
`prefix=target:responses` form works outside cuttle too.

### Request mapping (Anthropic → Responses)

| Anthropic | Responses | Notes |
| --- | --- | --- |
| `system[]` text blocks | `instructions` | Join non-empty text with `\n\n`; ignore `cache_control` |
| user `text` | `message.role=user`, `input_text` | |
| user `image` | `input_image` | `data:<media_type>;base64,<data>` or the URL; `detail: "auto"` |
| user `tool_result` | `function_call_output` | Emitted as its own input item; `call_id = tool_use_id`; `output` is a string or a list of text/image parts |
| assistant `text` | `message.role=assistant`, `output_text` | |
| assistant `tool_use` | `function_call` | `call_id = id`, `arguments = json(input)`; `{}` when input is absent |
| assistant `thinking` | dropped | Bailian accepts tool turns without reasoning; replay is deferred |
| `tools[]` | `tools[]` | `{type:"function", name, description, parameters: input_schema}` |
| `tool_choice.auto` | `"auto"` | |
| `tool_choice.none` | `"none"` | |
| `tool_choice.any` | `"required"` | |
| `tool_choice.tool` | `{type:"function", name}` | |
| `disable_parallel_tool_use: true` | `parallel_tool_calls: false` | Otherwise omit |
| `max_tokens` | `max_output_tokens` | Always set |
| `temperature`, `top_p` | same | |
| `stop_sequences` | dropped | Responses has no stop field; debug log |
| `top_k` | dropped | No Responses equivalent |
| `thinking.enabled/adaptive` | `reasoning: {effort, summary:"auto"}` | Effort from `output_config.effort`, else `effortOfBudget` |
| `thinking.disabled` / absent | omit `reasoning` | Bailian does not think by default |
| `output_config.format` | `text.format` | `{type:"json_schema", name:"response", schema}` |
| `metadata` | `metadata` | Pass through when it is an object |
| `stream` | `stream` | |
| — | `store: false` | Self-contained requests; no upstream conversation state |

Effort mapping: `minimal`/`low` → `low`, `medium` → `medium`,
`high` → `high`, `xhigh`/`max` → `high` (Responses' documented set stops at
`high`). Budget fallback: `<=4096` → `low`, `<=12000` → `medium`,
`<=24000` → `high`, else `high`.

Input item ordering preserves the Anthropic block order. Text before a tool
call is one message item; the tool call follows; a tool result starts a new
output item; later text starts a new message item.

### Non-streaming response mapping (Responses → Anthropic)

Build an `AnthropicResponse` with `type:"message"`, `role:"assistant"`,
`model = resolveResponseModel(resp.Model, opts)`, and one block per output item:

| Responses output | Anthropic block |
| --- | --- |
| `message` with `output_text` parts | `text` blocks, in order |
| `reasoning` with `content[].type == "reasoning_text"` | `thinking` block; text from `reasoning_text` |
| `reasoning` with `summary[]` | `thinking` block; join `summary_text` |
| `function_call` | `tool_use` with `id = call_id` (or item id), `name`, `input = parse(arguments)` |
| anything else | skipped with a debug log |

- A thinking block is emitted with `"signature": ""`.
- Tool arguments that are not a JSON object become `{}` and are logged; the
  raw string is not put in `input`.
- `stop_reason`:
  - any tool call → `tool_use`
  - `status == "incomplete"` with `max_output_tokens` → `max_tokens`
  - `status == "incomplete"` with `content_filter` → `refusal`
  - otherwise → `end_turn`
- Usage:
  - `input_tokens = max(input_tokens - cached_tokens - cache_write_tokens, 0)`
  - `cache_read_input_tokens = cached_tokens`
  - `cache_creation_input_tokens = cache_write_tokens`
  - `output_tokens = output_tokens`
- `stop_sequence` is `null`.
- A Responses error envelope is already returned unchanged by `Convert()`; no
  new error conversion is introduced.

### Streaming response mapping (Responses SSE → Anthropic SSE)

`ResponsesToAnthropicStreamConverter` implements `responsesStreamHandler`
(`HandleStreamStart`, `HandleChunk`, `HandleStreamEnd`, `EmitError`).

State:

- `started`, `finished`
- `messageID`, `model`
- next `content_block` index
- at most one open block: `{index, kind, itemID, argsSent}`
- `sawTool`
- final `AnthropicUsage`

Event handling:

| Responses event | Anthropic output |
| --- | --- |
| `response.created` | `message_start` with empty content, `usage.input_tokens` when present |
| `response.in_progress` | nothing |
| `response.output_item.added` type `reasoning` | open a `thinking` block (`thinking:"", signature:""`) |
| `response.output_item.added` type `function_call` | close any open block; open `tool_use` with `id=call_id`, `name`, `input:{}` |
| `response.content_part.added` type `output_text` | open a `text` block (`text:""`) |
| `response.output_text.delta` | ensure a text block; `text_delta` |
| `response.reasoning_text.delta` | ensure a thinking block; `thinking_delta` |
| `response.reasoning_summary_text.delta` | same as above |
| `response.reasoning_summary_part.added` | ensure a thinking block; no delta |
| `response.function_call_arguments.delta` | ensure the call's tool block; `input_json_delta` |
| `response.function_call_arguments.done` | if no delta was sent for this call, emit the whole `arguments` as one `input_json_delta` |
| `response.output_text.done` / `response.reasoning_text.done` | close the matching block |
| `response.content_part.done` | close the matching text block |
| `response.output_item.done` | close the matching block; for a function call, use the item's full `arguments` when no delta was sent |
| `response.completed` | close open block; `message_delta` with stop reason + usage; `message_stop` |
| `response.incomplete` | same as completed, with `max_tokens` |
| `response.failed` / `error` | close open block; Anthropic `error` event; mark finished |

Rules:

- Close a block exactly once; a later event for the same item is ignored.
- Empty deltas are skipped but do not count as "arguments sent".
- If a stream ends without `response.completed` and without a proxy error,
  `HandleStreamEnd` closes the open block and emits a `message_delta` +
  `message_stop` with the best-known stop reason and usage; it does not
  fabricate text.
- If the end phase carries `opts.ErrorMsg`, or an open tool block's
  accumulated arguments are not valid JSON, emit an Anthropic `error` event
  instead of a successful `message_stop`. This is the same terminal-failure
  rule the P2 work applied to truncated tool arguments.
- `message_start.usage` may be zero when the upstream has not reported usage
  yet. The final `message_delta.usage` carries the full converted usage.
- The empty `signature` on the thinking block start is intentional, matching
  the existing P2-1 rule for Anthropic stream block starts. No empty
  `signature_delta` is emitted; there is no signature to send.
- Argument deltas are buffered per output item. If a newer function-call item
  starts while another call is open, its events are held and released after
  the open call closes, in output order.

### Routing changes

`convert/detect.go`:

- `detectSource` treats a top-level `type` beginning with `response.` as
  `ProtocolOpenAIResponses`. This is needed for the first SSE event; response
  bodies already carry `object:"response"`.

`convert/convert.go`:

- Register two one-way pairs instead of the usual bidirectional pair:
  - `{Anthropic, Responses}` → Anthropic request → Responses request
  - `{Responses, Anthropic}` → Responses response → Anthropic response
- The response path currently returns every Responses body unchanged before it
  resolves the client protocol. Restructure it so:
  1. Requests whose source is Responses keep the existing
     `convertResponsesRequest` path.
  2. Responses whose client is Responses keep
     `convertToResponsesResponse` (an upstream Responses body passes through).
  3. Responses whose client is Anthropic dispatch to `{Responses, Anthropic}`.
  4. Unknown clients keep the current passthrough.
- On the request path, store `Session{From: source, To: downstream}` so the
  stream layer can use the client protocol even when body detection is
  ambiguous.

`convert/convert.go` `HandleSSEEvent`:

- On `start`, prefer `Session.From`/`Session.To`. If the session is absent,
  fall back to `detectByURI(opts.URI, DirectionRequest)` before model-map
  resolution; this keeps an inbound `/v1/messages` client identifiable even
  when the first upstream event is a bare `{"type":"error"}`.
- Add a branch before the generic OpenAI converter:
  `from == Responses && target == Anthropic` → create
  `ResponsesToAnthropicStreamConverter`.
- Add the new handler to the known stateful handlers that turn
  `opts.ErrorMsg` into a terminal error in the end phase.
- `event` and `end` phases reuse the stored handler; `error` calls `EmitError`.
- Unsupported combinations (for example Chat client, Responses upstream) keep
  the current fallback and are out of scope.

`convert/convert.go` `ConvertSSE`:

- Leave the legacy stateless path alone, consistent with the P2 design. The
  lifecycle path is the supported route.

### Testing

Unit tests:

- Request: simple text + system; image; tools; all four `tool_choice` forms;
  `disable_parallel_tool_use`; thinking/effort; structured output; metadata;
  `stream`; `max_tokens`; dropped `stop_sequences`/`top_k`; tool_use /
  tool_result ordering.
- Response: text only; text + reasoning; one and multiple tool calls; malformed
  arguments; incomplete `max_output_tokens`; usage with cached and cache-write
  tokens; empty output; stop-reason inference.
- Stream: captured Bailian text, tool, reasoning and parallel-tool fixtures
  under `convert/testdata/responses/`; OpenAI-style `reasoning_summary_text`
  variant; tool arguments only in the done event; empty deltas; failed event;
  stream end without `response.completed`; truncated tool arguments become an
  error; error-phase `opts.ErrorMsg` becomes an error.
- Routing: `Convert()` with a model map ending in `:responses` converts an
  Anthropic request and a Responses response; a Responses client still passes
  through; existing pairs do not regress.
- Detection: `response.created` and `response.output_text.delta` classify as
  Responses; Anthropic `message_start` still classifies as Anthropic; a bare
  error event is routed by session/URI.

Contract tests:

- Add direct shape assertions for the Responses request and response; extend
  `validateAnthropicResponse` / `validateAnthropicSSE` with a
  translated-response allowance for the empty signature instead of weakening
  the native Anthropic rules.
- `go vet ./...`, `go test -count=1 ./...`, and the race run for `convert/`.

Real upstream verification (after release):

- Bump cuttle to the new tag.
- Pin `bailian-dsf` to `api: responses` with a backup of `/data/cuttle.yaml`.
- Send a Claude Code-shaped `/v1/messages` request with a tool call and a
  reasoning round; verify HTTP 200, `stats.protocol=responses`, a complete
  Anthropic stream ending in `message_stop`, and the tool call executing.
- Roll back by removing the pin and sending `SIGHUP` if anything fails.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Anthropic client rejects thinking without a signature | Emit `"signature": ""` on thinking start and non-stream blocks |
| Upstream uses `reasoning_text` rather than `reasoning_summary_text` | Handle both event families; fixtures cover both |
| Tool arguments arrive only in the done event | Track `argsSent`; emit the full arguments on done |
| Upstream emits empty argument deltas | Skip empty deltas; do not mark arguments as sent |
| Parallel calls interleave or start before the previous call ends | Per-item argument buffers; release in output order after the open call closes |
| Error envelope reaches an Anthropic client unshaped | Non-stream errors pass through today; stream errors are normalized to Anthropic `error` |
| Dropped `stop_sequences` / `top_k` change behaviour | Document as unsupported; log at debug |
| Reasoning replay is required by another Responses provider | Deferred; add a replay switch when a provider needs it |
| Routing change regresses Responses clients | Explicit tests for Responses-client passthrough and existing pairs |
| Missing session on stream start forwards raw Responses events | Session/URI fallback before model-map resolution; tests cover the absent-session case |

## Deferred

- Chat Completions clients → Responses upstream.
- Responses client → Responses upstream passthrough audit.
- Sealed/encrypted reasoning, web search, custom/namespaced/deferred tools.
- Prompt-cache breakpoint semantics and cache-write replay.
- `count_tokens` upstream.
- Full IR migration; see the long-term decision and revisit triggers.

## Implementation seam

The new pair is deliberately two pure functions plus one stream state machine.
If the project later adopts an IR, replace the internals of
`anthropicRequestToResponses` / `responsesResponseToAnthropic` and the stream
decoder/encoder without changing the registry or `Convert()` dispatch. That
keeps the long-term option open without paying for it now.
