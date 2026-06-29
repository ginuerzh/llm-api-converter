# Fix llm-api-converter malformed Anthropic responses from DeepSeek reasoning-only output

## Context

**Symptom (reported):** Claude Code's auto-mode safety classifier becomes "temporarily unavailable" and blocks all Bash commands in an endless retry loop (`claude-opus-4-8[1m] is temporarily unavailable...`). This matches [anthropics/claude-code#63873](https://github.com/anthropics/claude-code/issues/63873).

**Root cause — confirmed by reproduction in this repo's `convert` package:**

The deployment routes Claude Code → `llm-api-converter` (GOST Rewriter plugin) → DeepSeek-V4-Pro. Claude Code's classifier sends a tiny Anthropic request (small `max_tokens`, **no `thinking` config**, no tools) and reads a non-streaming Anthropic response.

DeepSeek-V4-Pro is a reasoning model: **even with no thinking config**, it emits `reasoning_content` and spends the entire `max_tokens` budget on it, returning `content: ""` + `finish_reason: "length"`. (Verified empirically: a `max_tokens:10` probe returned `"content":"","reasoning_content":"We are asked: say hi"` with `reasoning_tokens:10`.)

`convert.Convert()` turns that into a malformed Anthropic response (reproduced via a temporary test):
```json
{"content":[{"type":"thinking","thinking":"We are asked: say hi"},{"type":"text"}],
 "stop_reason":"max_tokens",...}
```
Two defects:
1. **Spurious `thinking` block** — emitted whenever `reasoning_content != ""`, regardless of whether the request enabled extended thinking. The classifier request did not, so a `thinking` block is illegal in the response.
2. **Malformed empty text block** — `{"type":"text"}` with no `text` field (the `Text` field is `json:"text,omitempty"`, so empty string omits it). Anthropic requires a text block to carry a `text` field; an empty assistant response should have **no** text block, not an empty one.

Claude Code cannot parse this response → classifier call fails → "temporarily unavailable" retry storm. The non-streaming path is the one the classifier hits (classifier requests are short and non-streamed).

**Intended outcome:** When DeepSeek returns reasoning-only / empty-content output, the converter emits a well-formed Anthropic response with at least one valid, non-empty text block — so the classifier receives a parseable response and stops blocking Bash.

## Scope & non-goals

- **In scope:** Make `convertOpenAIMessageToContent` (non-streaming response direction) and `StreamConverter.HandleStreamEnd` (streaming) always produce at least one schema-valid, non-empty `text` block when the model returned no text and no tool calls — so reasoning-only / empty-content responses from DeepSeek-V4 are parseable by Anthropic clients (notably Claude Code's safety classifier).
- **Out of scope:** Gating the `thinking` block on request intent (statelessness makes this unreliable — see Approach). Changing DeepSeek's behavior. Changing the classifier itself (that's Claude Code, not this repo). The request-direction `convertAssistantContent` is *history replay* of assistant turns and is untouched.

## Key findings (verified, read-only)

- Request conversion does **not** inject thinking: `thinkingToOpenAi(nil)` returns `nil` ([convert.go:893-898](../../convert/convert.go#L893-L898)), and the DeepSeek branch only sets `oai.Thinking` when `profile.isDeepSeek` ([convert.go:1291](../../convert/convert.go#L1291)). So a request with no `thinking` field sends no thinking field downstream — DeepSeek reasons on its own.
- `AnthropicContent` struct: `Text string json:"text,omitempty"` and `Thinking string json:"thinking,omitempty"` ([types.go:264-267](../../convert/types.go#L264-L267)). Empty `Text` serializes to `{"type":"text"}` — malformed.
- Non-streaming emitter: **`convertOpenAIMessageToContent`** ([convert.go:1833-1877](../../convert/convert.go#L1833-L1877)), called once from `convertOpenAIResponseToAnthropic` ([convert.go:1786](../../convert/convert.go#L1786)). Defect sites: thinking block at 1837-1842; empty-text `else` branch at 1872-1874.
- Streaming emitter: **`StreamConverter`** ([stream.go](../../convert/stream.go)). `HandleChunk` emits a thinking block on `reasoning_content` deltas (135-142) with no request-thinking awareness. `HandleStreamEnd` (167-245) emits `message_delta`/`message_stop` even when `accumulatedText == ""` and there were no tool calls.
- No existing test asserts the malformed empty-text block; all tests use non-empty text, and the e2e mock never returns reasoning-only empty content. Adding the fix is safe against current tests.
- The converter has **no access to the original request** at response-conversion time (`convertOpenAIResponseToAnthropic(body, opts)` receives only the response body + opts), and `opts` is reconstructed fresh per `Convert()` call in [rewriter/server.go:79](../../rewriter/server.go#L79). Detecting "was thinking requested?" therefore requires cross-call state that does not exist.

## Approach

**Decision (user-approved): minimal fix — guarantee a valid text block only.** Do not gate the `thinking` block. The converter is stateless per `Convert()` call (request and response are separate invocations with fresh `opts` reconstructed in [rewriter/server.go:79](../../rewriter/server.go#L79)), so it cannot reliably know at response time whether the request asked for extended thinking. The classifier failure is primarily the *malformed empty text block*; the `thinking` block is kept (preserves DeepSeek reasoning replay for real Claude Code chat). Two parts:

### Part A — Non-streaming: fix `convertOpenAIMessageToContent` ([convert.go:1833](../../convert/convert.go#L1833))

1. **Never emit an empty text block.** In the `else` branch (line 1872-1874), if `text == ""`, emit a single non-empty placeholder text block (e.g. `{"type":"text","text":" "}`) so Anthropic clients always receive at least one valid text block with a `text` field. This directly fixes the malformed `{"type":"text"}` and the empty-content case the classifier hits.
   - Reuse the existing placeholder idiom already used elsewhere in the file for empty-message injection: `{"role":"user","content":"..."}` ([convert.go:1341](../../convert/convert.go#L1341)). Use a minimal placeholder (single space or `"..."`) for consistency.
2. **Leave the `thinking` block as-is.** Keep emitting it whenever `reasoning_content != ""` (current behavior at 1837-1842). `reasoning_content` is already cached by `convertOpenAIResponseToAnthropic`'s reasoning-cache logic (1791-1806), so no data loss for later replay.

### Part B — Streaming: guard `StreamConverter` against empty final content ([stream.go](../../convert/stream.go))

`HandleStreamEnd` (167-245) currently emits `message_delta`/`message_stop` even when `accumulatedText == ""` and there were no tool calls — i.e. a reasoning-only stream produces a `message_start` with `content:[]` and no content block at all, plus a spurious `thinking` block. Add: if at finalization `accumulatedText == ""` and `len(toolCallByIndex) == 0`, emit a single `content_block_start`/`text_delta(" ")`/`content_block_stop` so the stream always delivers at least one valid text block. Mirror the non-streaming placeholder. This keeps streaming responses parseable for any client (not just the classifier).

## Files to modify

- [convert/convert.go](../../convert/convert.go) — `convertOpenAIMessageToContent` (1833-1877): add empty-text guard in the `else` branch (1872-1874): when `text == ""`, emit a placeholder `{Type:"text", Text:" "}`.
- [convert/stream.go](../../convert/stream.go) — `HandleStreamEnd` (167-245): when `accumulatedText == ""` and `len(toolCallByIndex) == 0`, emit a `content_block_start`/`text_delta(" ")`/`content_block_stop` so the stream always delivers at least one valid text block.
- [convert/convert_test.go](../../convert/convert_test.go) — add `TestConvert_DeepSeekReasoningOnlyEmptyContent`: feed a DeepSeek response with `content:""`, `reasoning_content:"..."`, `finish_reason:"length"` through `Convert`; assert output has a `text` block with a non-empty `text` field and `stop_reason:"max_tokens"`.
- [convert/stream_test.go](../../convert/stream_test.go) — add a streaming equivalent: reasoning-only deltas + `finish_reason:"length"` → final stream contains at least one valid `text` block.

## Reused utilities

- `extractTextContent` (used throughout the file) for pulling text from `msg.Content`.
- Existing reasoning-cache `Put`/`PutText` calls in `convertOpenAIResponseToAnthropic` (1791-1806) already preserve `reasoning_content` for replay — safe to keep current thinking-block behavior.
- Placeholder idiom `" "` / `"..."` already used at [convert.go:1341](../../convert/convert.go#L1341) for empty-message injection.

## Verification

1. **Unit tests (with race detector, per CLAUDE.md):**
   ```bash
   cd llm-api-converter && CGO_ENABLED=1 go test -race ./... -count=1
   cd llm-api-converter && go vet ./...
   ```
   New tests must pass and existing 171+ tests must stay green.
2. **Reproduction test** (the one used during investigation): feed the DeepSeek reasoning-only response through `Convert` and assert the output is schema-valid (no `{"type":"text"}` without a `text` field).
3. **Live end-to-end (manual, requires the remote `llm.home.pi` deployment):** after deploying the rebuilt plugin, run a Claude Code session on Opus and confirm Bash commands no longer hit the "classifier temporarily unavailable" retry loop. (Deployment is remote — out of scope for this plan; flag to user.)
4. **Build:** `cd llm-api-converter && go build ./...`
