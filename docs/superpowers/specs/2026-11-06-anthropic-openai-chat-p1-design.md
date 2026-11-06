# Anthropic ↔ OpenAI Chat P1 Design

Date: 2026-11-06
Repository: `llm-api-converter`
Base revision: `a663b1d` (`merge: Anthropic-OpenAI Chat P0 fixes`)
Review source: `docs/reviews/2026-11-06-anthropic-openai-chat-review.md`
Scope: all 15 P1 findings from that review.

## Intent

Close every P1 finding from the 2026-11-06 protocol review in one branch. The
P0 fixes removed four direct 400/corruption failures; P1 removes the remaining
provider-specific 400s, silent data loss, and malformed stream/error behaviour.

The work keeps the existing pairwise converter architecture. It does not adopt
magpie's intermediate representation or add dependencies.

## Goals

- No known P1 shape produces an Anthropic 400 or a corrupt/empty client result.
- URL images, structured-output requests, typed stream deltas, and upstream
  stream errors survive conversion instead of being silently dropped.
- History tool calls without declared tools and broken tool exchanges become
  valid Anthropic requests.
- Existing DeepSeek/GLM thinking behaviour is preserved; other targets stop
  receiving unsupported reasoning fields.
- Every finding gets a regression test that fails on the base revision.

## Non-goals

- P2 findings in the review.
- Gemini-specific conversion, except where a shared helper is touched.
- Provider/host metadata plumbing through GOST. Capability decisions are made
  from model names (see "Capability gating").
- Adopting magpie's IR or adding dependencies.
- Changing same-protocol passthrough behaviour.

## Anti-goals

- Do not make DeepSeek/GLM regress while fixing strict OpenAI targets.
- Do not turn a conversion bug into an error response for requests that used to
  work; drop or normalize untranslatable fields instead.
- Do not silently accept an upstream error and emit a successful empty response.

## Constraints

- `go test -count=1 ./...`, `CGO_ENABLED=1 go test -race -count=1 ./convert`,
  and `go vet ./...` must pass.
- Code comments are English. Existing test style is table-driven where it fits.
- Commit after each plan task with its implementation record. Never push.
- Do not modify `core/` or other go-gost modules.
- The P1 plan and review doc live in `docs/plans/` and `docs/reviews/`.

## Approaches considered

**A. In-place pairwise fixes with model-based capability gating (chosen).**
All changes stay in `llm-api-converter`. Model prefixes decide thinking replay
and OpenAI reasoning-model parameters. Message coalescing and tool-pair repair
are ported from magpie's logic but adapted to the pairwise types.

**B. Plumb provider/host metadata into the converter.** Closer to magpie's
host-aware gating, but requires changing GOST's rewriter metadata contract and
the root repository. Rejected for this wave: the model name is already the
capability signal used elsewhere in this converter, and cross-repo metadata is
a larger risk for no P1 requirement.

**C. Introduce a magpie-style IR.** Rejected before P0: it rewrites the
conversion layer and adds a dependency for behaviour that can be fixed locally.

## Capability gating

No provider host is available in `ConvertOptions`. The design therefore uses
the resolved target model:

- `isDeepSeekModel`: `deepseek*` (existing).
- `isGLMModel`: `glm-*` (existing).
- `replaysReasoning(model)`: DeepSeek or GLM. Only these targets receive
  `reasoning_content` replay or the DeepSeek compatibility placeholder.
- `isOpenAIReasoningModel(model)`: base model name (after any `provider/`
  prefix) starts with `o1`, `o3`, `o4`, or `gpt-5`. These get
  `max_completion_tokens`, and sampling parameters are dropped.
- Other targets keep the current `max_tokens`/sampling behaviour.

The resolved target model is already computed at the top of the Anthropic →
OpenAI request path (`outputModel`). It is threaded to the assistant-message
converter rather than re-resolved per message.

## Per-finding design

### P1-1: Anthropic → OpenAI stream usage merge

`AnthropicToOpenAIStreamConverter.setUsage` currently overwrites all four
fields. Change it to update each field only when the event supplies a non-zero
value, preserving `message_start` input/cache counts when `message_delta` only
carries `output_tokens`. Add a stream test with input=1000, cache_read=200,
then output=50.

### P1-2: Anthropic message coalescing and tool-pair repair

Add a shared normalizer applied to Anthropic messages produced from OpenAI
Chat and Responses input:

1. Merge consecutive messages with the same role by appending content blocks.
   This turns parallel `tool` messages into one Anthropic user turn.
2. Repair the tool exchange:
   - Track `tool_use` IDs from each assistant message.
   - In the next user message, keep `tool_result` blocks whose ID is open;
     drop duplicate results.
   - Append a synthetic `tool_result` (`is_error: true`, content
     `"Tool result missing."`) for every open call not answered in that turn.
   - Convert an orphan `tool_result` (no open call) into a text block carrying
     its text, or `"Tool result received."` when empty.
   - If an assistant turn with open calls is followed by another assistant
     turn, close the exchange with a synthetic user message first.
   - **User messages put all `tool_result` blocks before text/image blocks.**
     The normalizer keeps two buckets per user message (`results`, `rest`) and
     emits `append(results, rest...)`; a synthetic result discovered while
     processing a user message is prepended to that message's results. This is
     required for the interrupted-turn shape `assistant(tool_use) → user(text)`.
   - **A history that ends with an open call is closed too.** After the loop,
     append a synthetic user message answering any still-open calls. This is
     magpie's final `closeExchange()`, and it is what makes an interrupted
     turn valid.

Apply the same normalizer to the Responses → Anthropic path after its messages
are built. This is magpie's `pairToolBlocks`/same-role merge behaviour, adapted
to `AnthropicMessage`/`AnthropicContent`.

### P1-3: History tool calls without declared tools

When the OpenAI/Responses request declares no tools but the converted history
contains `tool_use`, declare a stub Anthropic tool for each distinct name:

- `description`: `"Tool from earlier conversation. Do not call."`
- `input_schema`: `{"type":"object","properties":{}}`

Then set `tool_choice` to `{"type":"none"}`. If no history tool calls exist,
keep the P0 behaviour: no tools and no `tool_choice`.

### P1-4: Unsigned thinking replayed to Anthropic

Anthropic requires a signature on replayed thinking blocks. In the OpenAI →
Anthropic request path, emit a `thinking` block from `ReasoningContent` only
when the resolved target is DeepSeek-style (`isDeepSeekModel`); otherwise drop
it. This matches magpie's "signed thinking to Anthropic, unsigned only for
DeepSeek-style upstreams" rule without host metadata.

This applies to both callers of the shared assistant-content helper: Chat →
Anthropic and Responses → Anthropic. The Responses path threads its resolved
target model into the same helper instead of preserving the old unconditional
replay.

The response path (OpenAI response → Anthropic response content) is unchanged;
the finding is about request-history replay.

### P1-5: Metadata whitelist

OpenAI Chat → Anthropic:

- Keep only `user_id` from `req.Metadata`.
- If no `user_id` is present and `req.User` is non-empty, use `req.User`.
- Drop every other key; omit `metadata` when the result is empty.

Responses → Anthropic uses the same whitelist helper on `req.Metadata` (the
Responses request has no `user` field). Anthropic → OpenAI already does not
forward metadata.

### P1-6: Reasoning-model parameters

For `isOpenAIReasoningModel(outputModel)`:

- Set `MaxCompletionTokens` instead of `MaxTokens`.
- Omit `temperature` and `top_p`.
- Keep the existing thinking/reasoning-effort mapping.

For all other models keep `max_tokens`, `temperature`, and `top_p` as today.

### P1-7: Gate `reasoning_content` and placeholder

In the Anthropic → OpenAI request path, only set `reasoning_content` when
`replaysReasoning(targetModel)` is true. The cache lookup and the
"Compatibility bridge placeholder reasoning for prior assistant history."
fallback are also gated:

- Replay real cached/thinking reasoning for DeepSeek and GLM.
- Use the placeholder only for DeepSeek (the vendor that requires a non-empty
  value when tool calls are present).
- For all other targets, omit `reasoning_content` entirely.

### P1-8: Effort only when thinking is requested

On the DeepSeek path, forward `reasoning_effort` only when
`req.Thinking != nil && req.Thinking.Type == "enabled"`. Requests that carry
`output_config.effort` without thinking (title/classifier calls) must not
re-enable reasoning upstream.

### P1-9: URL images

- Add `URL string` to `AnthropicImageSource` for `source.type == "url"`.
- OpenAI → Anthropic: a non-data `image_url` becomes an Anthropic image block
  with `source: {"type":"url","url":...}` instead of being dropped.
- Anthropic → OpenAI: `source.type == "url"` becomes an OpenAI
  `image_url` part. Apply this in normal user content and tool-result images.
- Base64 handling is unchanged.

### P1-10: Structured output

Add typed fields:

- `OpenAIChatRequest.ResponseFormat *OpenAIResponseFormat` with
  `Type`, `JSONSchema{Name, Description, Schema, Strict}`.
- `AnthropicOutputConfig.Format *AnthropicOutputFormat` with
  `Type string`, `Schema any`.

OpenAI → Anthropic:

- `response_format.type == "json_schema"` with an object schema →
  `output_config.format = {"type":"json_schema","schema":...}`.
- `json_object` (no schema) has no Anthropic equivalent; append a system
  instruction asking for a JSON object only.
- Do not emit an empty `output_config`.

Anthropic → OpenAI:

- `output_config.format.type == "json_schema"` with a schema →
  `response_format = {"type":"json_schema","json_schema":{"name":"response","schema":...}}`.
- Format without a schema → system instruction fallback (same wording).

Structured output does not change response conversion; the JSON arrives as
text.

### P1-11: Tool-stop inference

Whenever a response contains tool calls, force the tool stop reason, but decide
from the **raw upstream reason**, not the already-mapped one, so an explicit
`content_filter` is never rewritten:

- OpenAI → Anthropic non-stream: if the raw `finish_reason` is nil or `stop`,
  use `tool_use` when the converted content has `tool_use`.
- Anthropic → OpenAI non-stream: if the raw `stop_reason` is nil or `end_turn`,
  use `tool_calls` when the response has `tool_use`.
- OpenAI → Anthropic stream: in `HandleStreamEnd`, if any tool call was
  accumulated (including held calls) and the raw `finish_reason` is empty or
  `stop`, emit `tool_use`.
- Anthropic → OpenAI stream: if any tool call block was emitted and the raw
  `stop_reason` is empty or `end_turn`, emit `tool_calls`.

`length`/`max_tokens`, `content_filter`, and other explicit reasons keep their
existing mapping.

### P1-12: Upstream stream errors

Two paths:

1. **OpenAI SSE error chunk** (`data: {"error":...}`): add an `Error` field to
   `OpenAIStreamChunk`; in `StreamConverter.HandleChunk`, emit an Anthropic
   `event: error` with a JSON-marshalled message, mark the stream final, and
   suppress the normal `message_stop`.
2. **GOST `stream_error` metadata** on the end phase: add `StreamError` to
   `engine.Meta`, parse it in `rewriter.metaFrom`, and in `HandleSSEEvent`'s
   end phase route a non-empty error to `EmitError` **only for converted
   sessions**. Same-protocol passthrough keeps its current behaviour: the
   end phase stays a no-op, because `PassthroughStreamHandler.EmitError` always
   emits an Anthropic-shaped frame and must not be used for OpenAI→OpenAI
   passthrough. The upstream error chunk, when one exists, already passed
   through in its native shape.

Symmetrically, an Anthropic `error` event in the Anthropic → OpenAI stream
must mark the converter finished after emitting the OpenAI error object, so the
end phase does not append a normal finish chunk.

No root-repository change is needed: GOST already sends `stream_error`.

### P1-13: Error JSON escaping

Build Anthropic error payloads with `json.Marshal` in
`StreamConverter.EmitError` and `PassthroughStreamHandler.EmitError` (the
Anthropic → OpenAI converter already does this). Add a test with quotes,
newlines, and backslashes.

### P1-14: Typed `delta.content`

Add `UnmarshalJSON` to `OpenAIDelta` that accepts:

- a string (current shape),
- `null`,
- an array of parts, concatenating `{"type":"text","text":...}` entries.

Other part types are ignored. Marshaling is unchanged. Add a stream test with
an array content delta.

### P1-15: `content_block_start` payload fields

Extend the Anthropic start event with `text`, `thinking`, and `input`:

- `text` → emit an OpenAI text delta immediately.
- `thinking` → emit `reasoning_content` immediately.
- `tool_use.input`: when it is a non-empty object (not `{}`/`null`), emit it as
  the tool call's `arguments` in the start chunk and mark that block as having
  complete input, so later `input_json_delta` events for it are ignored.
- Normal Anthropic `input:{}` followed by deltas keeps the current behaviour.

## Testing strategy

- New regression file `convert/anthropic_openai_p1_test.go`, grouped by
  finding ID, plus focused tests in existing stream/usage test files where the
  helper is already exercised.
- Each test must fail on `a663b1d` for the intended reason. Capture RED/GREEN
  evidence in the task report.
- Engine/rewriter metadata change gets an `engine` test with an end-phase
  `stream_error` and a converter test for the emitted error event.
- Keep round-trip tests, but add exact wire assertions for the fields that
  matter (no `reasoning_content`, `tool_choice:none` with stubs,
  `max_completion_tokens`, `response_format`, image URL source).

## Task outline

1. Usage merge, error JSON escaping, typed delta content (P1-1, P1-13, P1-14).
2. Message coalescing, tool-pair repair, history tool stubs (P1-2, P1-3).
3. Thinking/reasoning gating, effort gating, metadata whitelist (P1-4, P1-7,
   P1-8, P1-5).
4. Reasoning-model parameters and tool-stop inference (P1-6, P1-11).
5. URL images and structured output (P1-9, P1-10).
6. Stream errors and block-start payload fields (P1-12, P1-15).
7. Full verification, docs, and memory update.

Tasks 1–6 are independent enough to review separately but share files; they
run sequentially and each commit leaves the suite green. Task 7 consumes all.

## Risks and mitigations

- **Model-name heuristics are weaker than host detection.** Mitigation: only
  narrow prefixes are treated as reasoning-capable; DeepSeek/GLM behaviour is
  preserved; every gated behaviour has an explicit test.
- **Message normalization can merge turns the client expected separate.**
  Anthropic itself requires alternating/merged user turns; merging same-role
  messages is the documented-compatible shape. Tool-pair repair only changes
  invalid exchanges.
- **Stream error handling can truncate a stream that would otherwise finish.**
  Only explicit error chunks or GOST `stream_error` metadata trigger it; normal
  end phases are unchanged.
- **Structured-output name is required by OpenAI.** Use the fixed name
  `"response"`; do not claim strict mode unless the input was strict.

## Left to the builder

- Exact helper placement and whether the message normalizer is a new file or
  part of `openai_to_anthropic.go`.
- Whether to share the system-instruction fallback helper with the existing
  tool-choice instruction code.
- Test table layout and naming, consistent with `anthropic_openai_p0_test.go`.

## Builder-check resolutions

A fresh reviewer read this design against the code and raised six questions;
these are the rulings, now reflected above:

1. **Tool-result ordering:** user turns always emit `tool_result` blocks before
   text/image blocks, including synthetic results for interrupted turns.
2. **Trailing open call:** a history ending in `assistant(tool_use)` gets a
   synthetic user result so the request is valid; this is a deliberate
   difference from the P0 OpenAI-side sanitizer, which leaves in-progress
   tool calls alone.
3. **Stream-error routing:** only converted sessions turn end-phase
   `stream_error` into an error event. Passthrough stays unchanged.
4. **`content_filter`:** inference reads the raw reason; explicit
   `content_filter` is never rewritten into a tool stop.
5. **Responses paths:** P1-4 (unsigned thinking) and P1-5 (metadata whitelist)
   apply to Responses → Anthropic as well, through the shared helpers.
6. **Reasoning-model parameters:** model detection strips a `provider/` prefix
   before matching `o1`/`o3`/`o4`/`gpt-5`.
