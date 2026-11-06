# Anthropic ↔ OpenAI Chat Conversion Review

Date: 2026-11-06
Reviewed repository: `llm-api-converter`
Reviewed revision: `e4c7c1a` (`feat(engine): expose the resolved logical model`)
Comparison implementation: `magpie` at `e17b6ce`
Status: P0 and P1 fixed and verified; P2 findings remain open.

## Scope

This review covers the pairwise conversion between:

- Anthropic Messages (`/v1/messages`) request/response
- OpenAI Chat Completions (`/v1/chat/completions`) request/response
- their streaming SSE forms, including the shared message/tool/usage/thinking
  helpers used by the Responses path

Primary files:

- `convert/openai_to_anthropic.go`
- `convert/anthropic_to_openai.go`
- `convert/stream.go`
- `convert/stream_anthropic_to_openai.go`
- `convert/types.go`
- `convert/usage.go`
- `convert/session.go`
- `convert/convert.go`
- `convert/responses.go` where it shares the Chat helpers
- `engine/engine.go`, `rewriter/server.go`, and GOST SSE metadata plumbing
  for lifecycle/error behavior

Out of scope: Gemini conversion except where shared helpers are involved;
admin API; provider routing; billing.

## Verification

Commands run at `e4c7c1a`:

```sh
cd llm-api-converter
go test ./...      # pass, including e2e
go vet ./...       # pass
```

Additional reproduction tests were written temporarily and deleted. They
confirmed the findings below, including the exact malformed JSON/SSE shapes.
The repository was left clean after the review.

Existing tests do not protect the protocol contract in two places:

- `convert/convert_test.go:242` expects an assistant tool-call turn to contain
  an empty text block plus `tool_use`.
- `convert/convert_test.go:653` expects `tool_choice: {"type":"none"}` when no
  tools are present.

Both are rejected by Anthropic/strict Anthropic-compatible providers, so a
green test suite is not evidence that these outputs are valid.

## Verdict

Yes, there are real bugs. Four are P0 because they directly produce an
Anthropic 400 or corrupt streamed tool arguments:

1. Empty/missing `text` blocks on tool-only assistant turns.
2. `tool_choice` emitted without `tools`.
3. `tool_use.input` missing when OpenAI arguments are empty or invalid.
4. Interleaved OpenAI tool-call fragments are written to the wrong Anthropic
   content block.

Several P1 findings cause 400s on specific providers or materially wrong
usage/data: unsigned thinking replay, metadata pass-through, `max_tokens` vs
`max_completion_tokens`, unconditional `reasoning_content`, parallel tool
results without coalescing, dropped URL images, swallowed upstream stream
errors, and non-zero-safe usage merging.

## P0 Status

| ID | Status | Commit |
|---|---|---|
| P0-1 empty/missing text block | fixed | `ef2230b` |
| P0-2 `tool_choice` without tools | fixed | `612b033` |
| P0-3 missing `tool_use.input` | fixed | `dbf50b8` |
| P0-4 interleaved stream tool calls | fixed | `4d77eb1` |

The implementation plan is `docs/plans/2026-11-06-anthropic-openai-chat-p0.md`.
P0 was verified on `feat/anthropic-openai-p0` at `4d77eb1` with
`go test -count=1 ./...` (including a fresh e2e run),
`CGO_ENABLED=1 go test -race ./convert ./rewriter`, and `go vet ./...`; all
passed. P2 findings below are unchanged and still open.

## P1 Status

| ID | Status | Commit |
|---|---|---|
| P1-1 Anthropic → OpenAI stream usage | fixed | `e534397` |
| P1-2 message coalescing/tool-pair repair | fixed | `7620613` |
| P1-3 history tools without declarations | fixed | `7620613` |
| P1-4 unsigned thinking replay | fixed | `d2ec345` |
| P1-5 metadata pass-through | fixed | `d2ec345` |
| P1-6 OpenAI reasoning-model parameters | fixed | `8d9e8d3` |
| P1-7 unconditional `reasoning_content` | fixed | `d2ec345` |
| P1-8 effort without thinking | fixed | `d2ec345` |
| P1-9 URL images dropped | fixed | `1ee9c9f` |
| P1-10 structured output lost | fixed | `1ee9c9f` |
| P1-11 no tool-stop inference | fixed | `8d9e8d3` |
| P1-12 upstream stream errors swallowed | fixed | `0f53577`, fix round `2de9c52` |
| P1-13 unescaped error messages | fixed | `e534397` |
| P1-14 typed `delta.content` | fixed | `e534397` |
| P1-15 `content_block_start` payload fields | fixed | `0f53577`, fix round `2de9c52` |

P1 was verified on `feat/anthropic-openai-p1` at `40e70e9` with
`go test -count=1 ./...`,
`CGO_ENABLED=1 go test -race -count=1 ./convert ./engine ./rewriter`, and
`go vet ./...`; all passed. The implementation plan is
`docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md`.

## P0 findings

### P0-1: Empty/missing text block on OpenAI → Anthropic

Location: `convert/openai_to_anthropic.go:383-390`, `:475`, `:491-496`,
`:500-501`

`convertAssistantContent` always appends a text block even when the assistant
content is null/empty. `AnthropicContent.Text` has `omitempty`, so the wire
body is:

```json
{"type":"text"}
```

Anthropic requires a text block to contain non-empty, non-whitespace text. The
common OpenAI tool-call history (`assistant.content = null` plus `tool_calls`)
therefore fails on the next request:

```text
messages: text content blocks must be non-empty
```

The same class of bug exists for empty user content and reasoning-only
assistant turns.

Magpie behavior: `internal/gateway/anthropic.go:922-925` only keeps a text
block when `strings.TrimSpace(p.Text) != ""`.

Required fix: filter empty/whitespace-only text blocks and drop messages left
with no content. Ensure at least one valid user message remains.

Implementation record: see P0 plan Task 1.

### P0-2: `tool_choice` without `tools`

Location: `convert/openai_to_anthropic.go:267-271`, same pattern in
`convert/responses.go:339-340`

When the OpenAI request has no tools, the converter emits:

```json
"tool_choice":{"type":"none"}
```

Anthropic rejects a `tool_choice` that is not accompanied by `tools`; strict
Anthropic-compatible providers use the same validation. LiteLLM, moon-bridge
and APISIX all have dedicated fixes for this error.

Magpie behavior: `internal/gateway/anthropic.go:1088-1124` only sets
`tool_choice` inside the `len(rTools) > 0` branch.

Required fix: omit `tool_choice` entirely when no tools are forwarded.

Implementation record: see P0 plan Task 2.

### P0-3: Missing `tool_use.input`

Location: `convert/openai_to_anthropic.go:476-486`; reverse conversion at
`convert/anthropic_to_openai.go:564-572`, request side `:467-485`

If OpenAI `arguments` is empty or invalid JSON, `input` stays nil. Because
`AnthropicContent.Input` has `omitempty`, the tool block becomes:

```json
{"type":"tool_use","id":"call_1","name":"f"}
```

Anthropic requires `input` to be a JSON object. The reverse direction has the
mirror bug: `json.Marshal(nil)` becomes the string `"null"` in OpenAI
`tool_calls[].function.arguments`, which is not a valid object argument string.

Magpie behavior:

- `argsOf()` always returns at least `{}` (`internal/gateway/ir.go:774-789`)
- `parseArgs()` turns empty into `{}` and wraps non-object/invalid input
  (`:791-805`)

Required fix: normalize both directions. Empty/invalid/nil input becomes `{}`
(or a wrapped `{"input": ...}` for non-object JSON), never omitted and never
`"null"`.

Implementation record: see P0 plan Task 3.

### P0-4: Interleaved streamed tool calls corrupt arguments

Location: `convert/stream.go:411-475`, `:479-489`

OpenAI streams may interleave fragments of parallel calls:

```text
index=0: {"a":
index=1: {"b":
index=0: 1}
index=1: 2}
```

For an already-seen call, `handleToolCallDelta` calls `inputJSONDelta`, which
uses `sc.curBlockIndex` instead of that call's block index. The fragment for
call 0 is written into the currently open block for call 1. Reproduced result:
both calls end with invalid JSON arguments.

Magpie behavior: `internal/gateway/chat.go:1085-1101` holds interleaved calls
by index and releases them in index order; the code comment names the
real-world failure `mysterious_dove`.

Required fix: keep the open call streaming while its arguments are incomplete;
hold any new call that starts during that window, then emit held calls whole
in index order before the stream ends. Preserve the existing sequential
streaming path.

Implementation record: see P0 plan Task 4.

## P1 findings

### P1-1: Anthropic → OpenAI stream usage loses input/cache tokens

Location: `convert/stream_anthropic_to_openai.go:225-234`

`setUsage` overwrites all four usage fields. Anthropic sends input/cache
counts in `message_start` and only `output_tokens` in `message_delta`; the
final OpenAI usage therefore reports `prompt_tokens: 0`.

Reproduction: `message_start input=1000 output=1 cache_read=200`, then
`message_delta output=50` -> final `prompt_tokens=0, total_tokens=50`.

Magpie behavior: `internal/gateway/ir.go:528-545` merges only non-zero fields.

Fix: merge each field only when present/non-zero.

### P1-2: No Anthropic message coalescing or tool-pair repair

Location: `convert/openai_to_anthropic.go:273-287`, `:504-511`

Parallel OpenAI `tool` messages become multiple consecutive `user` messages.
Unfulfilled `tool_use` blocks and orphan `tool_result` blocks are not repaired.
Anthropic's direct API may combine consecutive turns, but Bedrock and strict
Anthropic-compatible endpoints reject them, and all parallel results should
share one user turn.

Magpie behavior: `buildAnthropic` merges same-role turns (`anthropic.go:908-917`)
and `pairToolBlocks` repairs the exchange (`:1142+`).

### P1-3: History tool calls without declared tools

Location: `convert/openai_to_anthropic.go` tools branch

An agent history containing `tool_use` but a request without `tools` is
rejected by Anthropic. Magpie stubs tools seen in history and forbids new calls
(`internal/gateway/anthropic.go:1077-1087`).

### P1-4: Unsigned thinking blocks replayed to Anthropic

Location: `convert/openai_to_anthropic.go:468-474`

OpenAI `reasoning_content` becomes an Anthropic `thinking` block with no
signature. Claude's extended-thinking replay expects the original signed block.
Magpie only sends signed thinking to Anthropic; unsigned reasoning is replayed
only for DeepSeek-style upstreams (`internal/gateway/anthropic.go:951-958`).

### P1-5: OpenAI metadata passed through to Anthropic

Location: `convert/openai_to_anthropic.go:215-217`

Anthropic `metadata` accepts only `user_id`. Any other key produces:

```text
metadata.<key>: Extra inputs are not permitted
```

The OpenAI `user` field is not mapped to `metadata.user_id`. Fix by
whitelisting `user_id` and dropping all other keys.

### P1-6: `max_tokens`/sampling parameters for OpenAI reasoning models

Location: `convert/anthropic_to_openai.go:193-201`

The Anthropic → OpenAI path always emits `max_tokens`. OpenAI o1/o3/GPT-5 Chat
Completions models require `max_completion_tokens`; `temperature`/`top_p` are
also commonly rejected. `OpenAIChatRequest.MaxCompletionTokens` exists
(`types.go:16`) but is never set here.

Magpie chooses by host (`internal/gateway/chat.go:543-549`) and strips sampling
parameters when the model rejects them.

### P1-7: Unconditional `reasoning_content` and fabricated placeholder

Location: `convert/anthropic_to_openai.go:490-492`, `:508-520`

Every OpenAI target receives `reasoning_content`; on a cache miss with
tool calls the converter injects:

```text
Compatibility bridge placeholder reasoning for prior assistant history.
```

This was reproduced for `gpt-4o`, `qwen-max`, and `deepseek-chat`. Strict
OpenAI-compatible endpoints may reject the unknown message field; otherwise it
pollutes context and billing.

Magpie gates replay with `chatReplaysReasoning(host, model)`
(`internal/gateway/chat.go:319-321`) and only supplies the placeholder to those
targets (`:453-460`).

### P1-8: Effort forwarded when thinking is absent/disabled

Location: `convert/anthropic_to_openai.go:216-222`

For DeepSeek targets, `output_config.effort` is always forwarded as
`reasoning_effort`, even when `thinking` is absent or `disabled`. Claude Code
title/classifier requests often carry effort without thinking; this can
re-enable reasoning upstream.

Magpie only uses effort when thinking was requested
(`internal/gateway/anthropic.go:277-291`).

### P1-9: URL images silently dropped

Location: `convert/anthropic_to_openai.go:357-360`,
`convert/openai_to_anthropic.go:419-453`

Only base64 data sources are converted. Anthropic `source.type=url` and remote
OpenAI `image_url` are dropped without an error, so the model never sees the
image.

Magpie preserves URLs (`internal/gateway/chat.go:239-248`, `:282-287`).

### P1-10: Structured output lost

Location: `convert/types.go` (missing fields), both request directions

OpenAI `response_format` and Anthropic `output_config.format` are not modeled
and are silently dropped. This is already listed as P1 in
`docs/MAGPIE_EVALUATION.md`; it remains unfixed. Magpie normalizes the field
(`internal/gateway/format.go`).

### P1-11: No tool-stop inference

Locations:

- Non-stream OpenAI → Anthropic: `convert/openai_to_anthropic.go:553`
- Non-stream Anthropic → OpenAI: `convert/anthropic_to_openai.go:592`
- Stream OpenAI → Anthropic: `convert/stream.go:235-244`
- Stream Anthropic → OpenAI: `convert/stream_anthropic_to_openai.go:260`

If an upstream reports `stop`/`end_turn` but includes a tool call, clients see
a normal stop and do not execute the tool. Some relays return this shape.
Stream OpenAI → Anthropic only overrides when `finishReason == "stop"`; it
still emits `end_turn` when finish_reason is absent.

Magpie centralizes this in `collector.finish()`
(`internal/gateway/ir.go:745-758`): any ToolCall forces `tool` unless already
set.

### P1-12: Upstream stream errors become successful empty responses

Locations:

- `convert/stream.go:103-118`
- `rewriter/server.go:95-112`
- `x/internal/util/forwarder/sniffer_sse.go:300-303`

An OpenAI SSE `data: {"error":...}` unmarshals into a zero-value
`OpenAIStreamChunk`, is treated as a no-op, and the end phase emits a normal
`message_stop`. GOST also passes `stream_error` metadata on interrupted
streams, but the rewriter ignores it and `SetInterrupted()` is only called in
tests.

Magpie recognizes `ch.Error` (`internal/gateway/chat.go:1337-1345`).

### P1-13: Unescaped error messages produce invalid JSON

Location: `convert/stream.go:283-286`, `convert/session.go:185-188`

```go
`data: {"type":"error","error":{"type":"stream_error","message":"%s"}}`
```

An upstream message containing quotes produces invalid JSON. Construct these
payloads with `json.Marshal`. `AnthropicToOpenAIStreamConverter.EmitError`
already does this correctly.

### P1-14: Typed `delta.content` arrays fail the stream

Location: `convert/types.go:507-514`, `convert/stream.go:108-110`

`OpenAIDelta.Content` is a string. OpenAI-compatible providers that send
content parts (`[{"type":"text","text":"..."}]`) cause an unmarshal error,
failing the whole stream. Magpie supports typed parts
(`internal/gateway/chat.go:1304-1350`).

### P1-15: Anthropic `content_block_start` payload fields ignored

Location: `convert/stream_anthropic_to_openai.go:53-70`, `:110-135`

The converter ignores `content_block.text`, `content_block.thinking`, and
`content_block.input`. Some Anthropic-compatible relays put full tool input in
the start event; the OpenAI client then receives empty arguments. Magpie
handles initial text/thinking and `fullInput` (`internal/gateway/anthropic.go:1273-1278`,
`:1327-1343`).

## P2 findings

| Finding | Location | Effect |
|---|---|---|
| Stream block starts lack `text:""` / `input:{}` / thinking `signature` | `convert/stream.go:316-325`, `:340-348` | strict clients may reject or misinitialize blocks |
| No-argument tool does not emit `{}` | `convert/stream.go:411-475` | clients receive empty arguments |
| `reasoning` / `reasoning_details` aliases ignored | `convert/types.go:37-45`, `:507-514` | OpenRouter-style reasoning lost |
| `developer` role becomes user | `convert/openai_to_anthropic.go:280-284` | instruction priority changes |
| Anthropic tool IDs not sanitized | `convert/openai_to_anthropic.go:483` | IDs outside `[A-Za-z0-9_-]` rejected/dropped |
| Unknown `finish_reason`/`stop_reason` passed through | `convert/openai_to_anthropic.go:596-597`, `convert/convert.go:377-388` | invalid Anthropic stop reasons (`function_call`, `content_filter`, ...) |
| `content_filter` mapping inconsistent | non-stream -> `end_turn`, stream -> raw | should map to `refusal` where supported |
| `refusal` / annotations ignored | `OpenAIMessage`/`OpenAIChoice` fields | refusal becomes an empty placeholder |
| Empty OpenAI choices produce `content:null` | `convert/openai_to_anthropic.go:551-555` | Anthropic response content must be an array |
| Files/audio/documents dropped | both directions | multimodal/file inputs silently lost |
| `parallel_tool_calls` / `disable_parallel_tool_use` dropped | tool-choice structs | parallel semantics lost |
| `top_k` dropped Anthropic → OpenAI | request conversion | fidelity loss on providers that support it |
| `message_start.usage` is always 0 | `convert/stream.go:90-97` | clients reading context usage at start see 0 |
| Legacy `ConvertSSE` path incomplete | `convert/convert.go:220-268` | standalone SSE conversion lacks lifecycle events and tool calls |
| Interrupted stream may add two text blocks | `convert/stream.go:218-232` | cosmetic duplicate placeholder |
| `ReasoningCache` read can race with in-place `Set` | `convert/reasoning_cache.go:233-245` | potential data race under concurrent streams |

## Magpie comparison summary

| Behavior | llm-api-converter | magpie | Verdict |
|---|---|---|---|
| Empty text blocks | kept | whitespace-only dropped | P0 bug |
| Consecutive same-role messages | not merged | merged | P1 bug for strict providers |
| tool_use/tool_result pairing | not repaired | `pairToolBlocks` | P1 bug |
| `tool_choice` without tools | always emits `none` | omitted | P0 bug |
| Empty/invalid tool input | omitted or `"null"` | `{}` / wrapped | P0 bug |
| Historical tools without declarations | not stubbed | stubbed | P1 bug |
| Unsigned thinking | always sent | only replay for DeepSeek | P1 bug/risk |
| `reasoning_content` replay | all models + placeholder | host/model gated | P1 bug |
| Interleaved stream tool calls | wrong block index | held by index | P0 bug |
| Usage merge | overwrite all fields | non-zero merge | P1 bug |
| Metadata | passed through | not introduced from Chat | P1 bug |
| OpenAI reasoning-model parameters | `max_tokens` only | host-aware | P1 bug |
| Structured output | dropped | normalized | P1 gap |
| URL images | dropped | preserved | P1 bug |
| Stream error chunks | swallowed | explicit error | P1 bug |

> Historical findings: the P0 and P1 rows above describe the reviewed revision;
> all were fixed by the P0/P1 plans; see the P0 Status and P1 Status tables.

## What is already good

- The pairwise registry/model-map architecture is appropriate; magpie's IR is
  not required to fix these issues.
- Non-streaming usage conversion and the OpenAI → Anthropic streaming usage
  direction are mostly correct.
- The three-tier reasoning cache (tool-ID/context/text) is useful and is not
  something magpie has in the same form.
- SSE lifecycle integration with GOST (`start`/`event`/`end`) is the right
  shape.
- Test coverage is broad, and `go test ./...` plus `go vet ./...` are clean.

## Follow-up

P0 implementation plan:
`docs/plans/2026-11-06-anthropic-openai-chat-p0.md`

P1 implementation plan:
`docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md`

P2 work is intentionally not scheduled in either plan. See
`docs/MAGPIE_EVALUATION.md` for the previously known structured-output and
thinking-signature gaps.
