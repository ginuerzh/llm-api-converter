# Anthropic ↔ OpenAI Chat P2 Design

Date: 2026-11-06
Repository: `llm-api-converter`
Base revision: `60f59e8` (`merge: use canonical module path`)
Review source: `docs/reviews/2026-11-06-anthropic-openai-chat-review.md`
Scope: the 13 actionable P2 findings from that review, plus two stream-terminal
gaps found while validating the released artifact (Responses truncated tool
arguments and Gemini `EmitError`).

## Intent

Close the P2 findings that can be fixed locally without changing the pairwise
converter architecture or adding provider-specific request plumbing. P0 and P1
removed the direct 400s and the largest silent data-loss paths; P2 removes the
remaining strict-schema mismatches, dropped request/response fields, one data
race, and two cases where a truncated upstream stream could look successful.

The work stays inside `llm-api-converter/convert`. It does not adopt magpie's
IR, does not add dependencies, and does not change same-protocol passthrough.

## Goals

- Strict Anthropic and OpenAI clients receive the fields their schemas expect:
  stream block starts carry their empty payload, no-argument tools carry `{}`,
  empty choices produce an array, and stop reasons stay inside the documented
  value sets.
- Tool IDs that OpenAI-compatible providers emit but Anthropic rejects are
  normalized deterministically, and request/response/stream paths agree on the
  same normalized value so tool pairs still match.
- Request fidelity improves for `developer` role, `reasoning` /
  `reasoning_details`, `parallel_tool_calls` /
  `disable_parallel_tool_use`, `top_k` on known supporting providers, and
  OpenAI `refusal` text.
- A stream that ends with truncated tool arguments is terminal failure, not a
  successful tool call, in the Responses converter as well as the existing
  Anthropic/OpenAI paths.
- Gemini stream error events are JSON-safe and terminal in all four stateful
  Gemini converters.
- `ReasoningCache` is safe under concurrent `Get`/`Set`; the race detector
  proves it.
- Every fix has a regression test that fails on the base revision.

## Non-goals

- Files, audio, and document content-part mapping.
- The legacy stateless `ConvertSSE` path; it is documented as incomplete and is
  not the supported stream route.
- `message_start.usage` population. OpenAI only reports usage in the final
  streaming chunk (even with `stream_options.include_usage`), so there is no
  honest value to put in `message_start`.
- OpenAI annotations / Anthropic citations mapping.
- `count_tokens` handling and Gemini schema type inference from
  `docs/MAGPIE_EVALUATION.md`.
- Adopting magpie's IR or adding dependencies.
- Changing same-protocol passthrough behavior.

## Anti-goals

- Do not forward `top_k` to providers that are known to reject it just to gain
  fidelity; a new 400 is worse than a dropped optional field.
- Do not invent content when an upstream returns no choices; only fix the JSON
  shape.
- Do not turn a successful stream into an error unless the output genuinely
  cannot be completed (truncated tool arguments).
- Do not change IDs that are already valid, so existing logs and tool pairs
  remain stable.

## Constraints

- `go test -count=1 ./...`, `CGO_ENABLED=1 go test -race -count=1 ./convert`,
  and `go vet ./...` must pass.
- Code comments are English.
- Commit after each plan task with its implementation record. Never push.
- Do not modify `core/` or other go-gost modules.
- Keep the provider contract validators green.

## Approaches considered

**A. One branch, eight focused tasks, local fixes plus two stream guards
(chosen).** All selected changes are in `convert/`. Tests already cover both
directions and the contract validators give an independent schema check. The
two stream guards belong in the same risk class as the P1 truncated-stream fix,
so they ship together.

**B. Only the five highest-risk fixes.** Faster, but leaves strict-schema
mismatches, dropped request fields, and the cache race in place. Rejected
because the remaining items are local and testable.

**C. Include multimodal files/documents and rewrite the legacy `ConvertSSE`
path.** Rejected for this wave: file/audio/document mapping needs a separate
capability and provider design, and the stateless path cannot implement stream
lifecycle without a session. Both are recorded as deferred.

## Per-finding design

### P2-1: Stream block starts carry their empty payload

`StreamConverter.ensureBlock` currently emits `{"type":"text"}` and
`{"type":"thinking"}` without the fields strict Anthropic decoders require.

- text start: `{"type":"text","text":""}`
- thinking start: `{"type":"thinking","thinking":"","signature":""}`
- tool_use start: unchanged (`id`, `name`, `input:{}`)

The empty signature is the value Anthropic's SDK type requires; the real
signature still arrives later in `signature_delta`. The contract validator is
strengthened to require these fields on start events.

### P2-2: No-argument tools emit `{}`

A tool call whose accumulated arguments stay empty currently closes without any
`input_json_delta`, so clients accumulate an empty string instead of `{}`.

- When a tool_use block closes with empty accumulated arguments, emit one
  `input_json_delta` containing `{}` immediately before `content_block_stop`.
- Apply this both when `ensureBlock` switches away from a tool block and in
  `HandleStreamEnd`.
- Fix the related hold check: a no-argument open call is complete, so a
  following tool call must not be held until stream end. Use
  `streamToolArgsComplete` instead of `toolArgsComplete` in
  `handleToolCallDelta`.

### P2-3: `reasoning` / `reasoning_details` aliases

Add a custom `UnmarshalJSON` for `OpenAIMessage` and extend the existing
`OpenAIDelta.UnmarshalJSON` shadow struct with:

- `reasoning string`
- `reasoning_details []OpenAIReasoningDetail`

`OpenAIReasoningDetail` carries `type`, `text`, and `summary`. Normalization
rule:

1. `reasoning_content` wins when non-empty (DeepSeek/GLM shape).
2. Else `reasoning`.
3. Else join each detail's non-empty `text`, falling back to `summary`, with
   `\n`.

The normalized value is stored in `ReasoningContent`; alias fields are not
added to the public message struct, so re-marshalling cannot emit both shapes.

### P2-4: `developer` role

Treat `developer` like `system` in the OpenAI → Anthropic request path and
collect it into the top-level Anthropic `system` blocks. No other role changes.

### P2-5: Tool ID sanitization

Add `sanitizeAnthropicToolID`:

- empty → existing `ensureToolID` random `toolu_...`
- otherwise replace every byte outside `[A-Za-z0-9_-]` with `_`
- leave already-valid IDs untouched

Apply it wherever a tool ID crosses into an Anthropic shape:

- request assistant `tool_use` IDs
- request `tool_result.tool_use_id`
- non-stream response `tool_use` IDs
- streaming `content_block_start.tool_use.id`

Because the replacement is deterministic, an assistant call and its matching
result normalize to the same ID. The rare case where two distinct invalid IDs
collapse to the same string is accepted; hashing is not worth the extra ID
length.

Reasoning-cache keys use the sanitized ID on both sides. The OpenAI response
and stream paths sanitize tool IDs before `ReasoningCache.Put`, because the
client will send the sanitized IDs back on the next request; the Anthropic →
OpenAI lookup therefore sees the same values. Storing raw upstream IDs would
make tool-only DeepSeek/GLM replay miss the cache and fall back to the
placeholder reasoning.

### P2-6 and P2-7: Stop-reason normalization and `content_filter`

One shared function normalizes OpenAI finish reasons for Anthropic:

| OpenAI | Anthropic |
|---|---|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls`, `function_call` | `tool_use` |
| `content_filter` | `refusal` |
| empty / unknown | `end_turn` |

Use it for both non-stream responses and stream `message_delta`, so
`content_filter` no longer differs by path. A nil non-stream finish reason maps
to `end_turn` so the Anthropic response always has a valid stop reason.

Tool content takes precedence: when `tool_use` blocks are present, `nil`,
`stop`, and `content_filter` all resolve to `tool_use`, because Anthropic
clients only run tools on that stop reason. An explicit `length` remains
`max_tokens` so truncation stays visible. The existing
`TestP1_NonStreamToolStopInference` expectation for `content_filter` + tools
changes from `end_turn` to `tool_use`.

Anthropic → OpenAI normalization:

| Anthropic | OpenAI |
|---|---|
| `end_turn`, `stop_sequence`, `pause_turn` | `stop` |
| `max_tokens` | `length` |
| `tool_use` | `tool_calls` |
| `refusal`, `content_filter` | `content_filter` |
| empty / unknown | `stop` |

`refusal` is a documented Anthropic stop reason; this converter already accepts
it in the contract validator.

### P2-8: OpenAI refusal text

Add `Refusal string` to `OpenAIMessage` and `OpenAIDelta`. When content text is
empty and `refusal` is set, use the refusal text as the Anthropic text block
instead of the whitespace placeholder. If the finish reason is nil/`stop` and
there are no tool calls, set the Anthropic stop reason to `refusal`. In the
stream path, a non-empty `delta.refusal` opens/uses the text block exactly like
`delta.content`, and the same stop-reason rule applies. Annotations remain
deferred.

### P2-9: Empty OpenAI choices

Initialize the Anthropic response content to an empty, non-nil slice and map a
missing finish reason to `end_turn`. An empty choices list therefore serializes
`"content":[]` instead of `"content":null`. No placeholder text is invented.
The P2-9 test asserts the JSON shape directly; the strict non-stream contract
validator keeps rejecting empty content, because a real Anthropic response
normally has at least one block. The anomaly is an upstream no-choices
response, not a valid converted message.

### P2-10: Files, audio, documents

Deferred. Recorded in the review status as unscheduled; mapping needs a
separate content-part capability design.

### P2-11: Parallel tool calls

Add `ParallelToolCalls *bool` to `OpenAIChatRequest` and
`DisableParallelToolUse *bool` to `AnthropicToolChoice`.

- OpenAI → Anthropic: when tools are present and `parallel_tool_calls` is set,
  create an `auto` tool choice if none exists and set
  `disable_parallel_tool_use = !parallel_tool_calls`.
- Anthropic → OpenAI: when `disable_parallel_tool_use` is set, set
  `parallel_tool_calls = !disable_parallel_tool_use`.
- Responses paths: `ConvertResponsesToAnthropic` applies the same OpenAI →
  Anthropic rule to `ResponsesRequest.ParallelToolCalls` after non-function
  tools are filtered, and `ConvertAnthropicToResponses` applies the reverse
  rule to `ResponsesRequest.ParallelToolCalls`.
- Do not attach `disable_parallel_tool_use` to an Anthropic `none` tool choice,
  and ignore the field when no tools are declared, because Anthropic rejects a
  tool choice without tools.

### P2-12: `top_k` Anthropic → OpenAI

`OpenAIChatRequest` already has `TopK`. Forward it only when the resolved target
is not an OpenAI reasoning model and the model carries a provider prefix known
to accept `top_k`: `groq/`, `openrouter/`, `together/`, `fireworks/`, `vllm/`,
`ollama/`, `lmstudio/`. Unknown and strict OpenAI targets keep dropping it to
avoid a new 400.

The checks inspect the resolved target string in different ways on purpose:

- reasoning detection strips **all** leading `provider/` segments, so
  `openrouter/openai/gpt-5` is still recognized as a reasoning model;
- the allowlist matches the **first** segment before stripping, so
  `groq/llama-3.3-70b` and `openrouter/...` are recognized.

If a model map has already removed the provider prefix and only a bare target
remains, `top_k` is not forwarded. That is the safe outcome, not a bug.

### P2-13: `message_start.usage`

Deferred. Streaming providers do not supply input tokens until the final usage
chunk, and the real values already appear in `message_delta`. An estimate would
be wrong and would make the start event disagree with the final event.

### P2-14: Legacy `ConvertSSE`

Deferred. The function converts one event at a time and cannot own block or tool
state. The supported route is the stateful `HandleSSEEvent` / session path.

### P2-15: Interrupted stream duplicate text block

Add a regression test for an interrupted stream that already emitted text and
for one that emitted nothing. Expected output is exactly one text block with
the interruption marker; the existing placeholder logic must not open a second
block. Change code only if the test reproduces the duplicate.

### P2-16: `ReasoningCache` read/set race

`memoryStore.Get` returns the stored `*cacheEntry`, while `Set` mutates that
entry in place. Change both sides:

- `Get` returns a shallow copy under the read lock.
- `Set` replaces the map entry with the new pointer instead of mutating the
  existing entry.

`fileStore` delegates to `memoryStore`, so it inherits the fix. Add a
concurrent `Get`/`Put` test and run it under `-race`.

### Extra gap 1: Responses truncated tool arguments

Both Responses-bound stateful converters have the same gap:
`ResponsesStreamConverter` (OpenAI Chat → Responses) and
`AnthropicStreamConverter` (Anthropic → Responses). Each currently emits
`function_call_arguments.done` and a completed/incomplete terminal event even
when accumulated arguments are incomplete.

Before finalizing either converter, scan all tool calls; if any non-empty
argument string does not parse as a JSON object, return
`EmitError("stream ended with incomplete tool call arguments")`. Empty
arguments are complete (no-argument call). The error event is terminal and no
`response.completed` / `response.incomplete` follows.

For a no-argument function call, emit `"{}"` in
`response.function_call_arguments.done` and in the
`response.output_item.done` arguments field instead of an empty string. Codex
drops function-call items that do not deserialize.

### Extra gap 2: Gemini stream error safety

Four stateful converters:

- `GeminiToAnthropicStreamConverter` (currently `fmt.Sprintf`, no terminal flag)
- `GeminiToOpenAIStreamConverter` (JSON-safe, but no terminal flag and wrong
  separator)
- `AnthropicToGeminiStreamConverter`
- `OpenAIToGeminiStreamConverter`

All four `EmitError` methods must:

1. use `json.Marshal` for the message
2. return nil when the stream is already finished/finalized
3. mark the stream finished/finalized before returning
4. use the correct SSE separator (`\n\n` for data-only Gemini/OpenAI events;
   Anthropic-formatted errors keep the existing event framing)

Hardening the methods alone is not enough if real upstream errors never reach
them. Route the errors these converters already detect through `EmitError`:

- `AnthropicToGeminiStreamConverter`: its inline `case "error"` currently
  builds an error chunk without marking the stream finished, so
  `HandleStreamEnd` later appends a normal finish. Call `EmitError` instead.
- `OpenAIToGeminiStreamConverter`: inspect `chunk.Error` and call `EmitError`.
- `GeminiToOpenAIStreamConverter` / `GeminiToAnthropicStreamConverter`: add an
  `Error` field to `GeminiChatResponse` and call `EmitError` for a top-level
  `{"error":...}` payload. Blocked responses already call or build an error;
  route both through `EmitError` so they become terminal.
- End-phase dispatch: add the four Gemini converters to the `opts.ErrorMsg`
  switch in `handleSSEEvent`, so an interrupted stream emits a terminal error
  in the target protocol instead of a normal close. This intentionally reverses
  `TestP1_GeminiEndPhaseStreamErrorNotDispatched`; that test documented the P1
  scope limit, not desired P2 behavior.

`GeminiStreamHandler.EmitError` is stateless and already JSON-safe; it is out
of scope.

## Builder-check resolutions

The independent builder check raised nine points. They are settled here so the
plan does not have to guess:

1. **Gemini upstream errors:** in scope. Hardening `EmitError` alone would be
   hollow because several detected upstream errors never called it. The task
   includes routing and end-phase dispatch, and updates the P1 non-dispatch
   test.
2. **Reasoning-cache IDs:** cache sanitized IDs on both `Put` and lookup.
3. **Responses truncation:** applies to both `ResponsesStreamConverter` and
   `AnthropicStreamConverter`; no-argument Responses calls emit `"{}"`.
4. **Stop reason with tools:** `tool_use` wins over nil/`stop`/`content_filter`;
   `length` stays `max_tokens`.
5. **Empty `{}` at tool close:** centralize in a helper used by every tool-block
   close site, including the explicit stop in `handleToolCallDelta`.
6. **Parallel tool calls:** cover Chat and Responses request paths; skip the
   field for Anthropic `none`.
7. **Streaming refusal:** include `delta.refusal`, not only the non-stream
   message field.
8. **`top_k` model string:** strip all provider segments for reasoning
   detection; match the allowlist against the first segment.
9. **Empty-choices validator:** assert JSON shape directly; do not weaken the
   contract validator for this anomaly.

## Testing strategy

- New P2 tests live in `convert/anthropic_openai_p2_test.go`, grouped by the
  finding they cover.
- Extend `convert/contract_test.go` only where the contract itself is the
  subject: block-start required fields and stream stop reasons.
- Race coverage: run `CGO_ENABLED=1 go test -race -count=1 ./convert`; the cache
  test must fail under `-race` on the base revision.
- Full verification after every task:
  `go test -count=1 ./...`, `go vet ./...`, `git diff --check`.

## Decisions left to the builder

These are deliberately settled in this document rather than asked again:

- `top_k` provider allowlist (P2-12)
- empty choices serialize `[]`, not a placeholder (P2-9)
- tool IDs use deterministic `_` replacement (P2-5)
- `content_filter` maps to `refusal` on both streaming and non-streaming paths
  (P2-7)
- thinking block start includes an empty signature (P2-1)
- Responses truncation message: `stream ended with incomplete tool call
  arguments` (Extra gap 1)

## Deferred items

Recorded in the review status after implementation:

- files/audio/documents content parts
- legacy `ConvertSSE` lifecycle
- `message_start.usage`
- annotations / citations
- `count_tokens`
- Gemini schema type inference
