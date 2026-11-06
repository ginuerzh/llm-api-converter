# Anthropic ↔ OpenAI Chat P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all 15 P1 findings from the 2026-11-06 Anthropic ↔ OpenAI Chat review without changing the pairwise converter architecture.

**Architecture:** In-place fixes in `llm-api-converter/convert`, with capability decisions derived from the resolved target model. Message normalization and tool-pair repair are adapted from magpie's logic to the existing `AnthropicMessage`/`AnthropicContent` types.

**Tech Stack:** Go, `encoding/json`, existing `convert`, `engine`, and `rewriter` packages. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-06-anthropic-openai-chat-p1-design.md`

## Global Constraints

- Do not add third-party dependencies.
- Keep the pairwise converter registry, model-map, and same-protocol passthrough behavior intact.
- Code comments are English.
- Every task must leave `go test -count=1 ./...` green.
- Commit only; never push.
- Do not modify `core/` or other go-gost modules.
- Each task fills its **Implementation record** in this plan with date, repository/commit, files, commands/results, review findings, and deferred items, and commits it with the task.
- Follow TDD: write the failing test, capture RED, implement, capture GREEN.

## Review Focus

The spec implies these inputs but no single task's tests cover all of them; each line names the task that owns its test.

1. **Interrupted tool histories** (`assistant(tool_use)` with no result, orphan result, parallel results): must become valid Anthropic turns with results first. Task 2.
2. **Anthropic target receiving unsigned OpenAI reasoning**: must not emit an unsigned `thinking` block for a Claude target. Task 3.
3. **OpenAI reasoning models** (`o1`/`o3`/`o4`/`gpt-5`) with `max_tokens` and sampling params: must map to `max_completion_tokens` and drop unsupported sampling. Task 4.
4. **Upstream SSE error mid-stream and scanner error at end**: must surface an error event, not a successful empty response. Task 6.
5. **Structured-output and URL-image round trips**: must preserve the schema/URL rather than silently dropping. Task 5.

---

### Task 1: Stream usage, error escaping, typed delta content

**Files:**
- Modify: `convert/stream_anthropic_to_openai.go:225-234`
- Modify: `convert/stream.go:292-301`
- Modify: `convert/session.go:184-188`
- Modify: `convert/types.go:507-514`
- Test: `convert/anthropic_openai_p1_test.go` (create)

**Interfaces:**
- Produces: `func (d *OpenAIDelta) UnmarshalJSON(data []byte) error`
- Produces: `setUsage` merges each non-zero field instead of overwriting all four.
- Produces: both `EmitError` implementations build their payload with `json.Marshal`.

- [x] **Step 1: Write the failing tests**

```go
package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestP1_AnthropicStreamUsageMergesPartialUpdates(t *testing.T) {
	c := NewAnthropicToOpenAIStreamConverter("claude", &ConvertOptions{RequestModel: "claude"})
	_, _ = c.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":1000,"output_tokens":1,"cache_read_input_tokens":200}}}`))
	_, _ = c.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}`))

	var usage *OpenAIUsage
	for _, line := range strings.Split(string(c.HandleStreamEnd()), "\n") {
		line = strings.TrimPrefix(line, "data: ")
		var chunk OpenAIStreamChunk
		if json.Unmarshal([]byte(line), &chunk) == nil && chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if usage == nil {
		t.Fatal("no usage chunk")
	}
	if usage.PromptTokens != 1200 || usage.CompletionTokens != 50 || usage.TotalTokens != 1250 {
		t.Fatalf("usage = %+v, want prompt=1200 completion=50 total=1250", usage)
	}
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens != 200 {
		t.Fatalf("cached tokens lost: %+v", usage.PromptTokensDetails)
	}
}

func TestP1_AnthropicEmitErrorEscapesJSON(t *testing.T) {
	want := "bad \"quote\"\nline\\backslash"
	sc := NewStreamConverter("claude", nil, nil)
	out := string(sc.EmitError(want))
	parts := strings.SplitN(out, "\n", 2)
	if len(parts) != 2 {
		t.Fatalf("error event = %q", out)
	}
	data := strings.TrimPrefix(parts[1], "data: ")
	var evt struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(data), &evt); err != nil {
		t.Fatalf("invalid error JSON %q: %v", data, err)
	}
	if evt.Error.Message != want {
		t.Fatalf("message = %q, want %q", evt.Error.Message, want)
	}
}

func TestP1_TypedDeltaContent(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"hel"},{"type":"text","text":"lo"}]},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"text":"hello"`) {
		t.Fatalf("typed content not joined: %s", out)
	}
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_(AnthropicStreamUsageMergesPartialUpdates|AnthropicEmitErrorEscapesJSON|TypedDeltaContent)' -v`

Expected: FAIL — usage prompt is 1000/50 instead of 1200/50; error JSON is malformed; typed content fails to unmarshal.

- [x] **Step 3: Implement the three fixes**

`setUsage`: update each of `InputTokens`, `OutputTokens`, `CacheReadInputTokens`, `CacheCreationInputTokens` only when the argument is greater than zero; set `hasUsage = true` if any argument was greater than zero.

`StreamConverter.EmitError` and `PassthroughStreamHandler.EmitError`: marshal
`map[string]any{"type":"error","error":map[string]any{"type":"stream_error","message":message}}`
and prefix `event: error\ndata: `.

`OpenAIDelta.UnmarshalJSON`: decode into a shadow type plus a
`Content json.RawMessage`; accept a string or `null` directly, accept an array
of `{type,text}` parts and concatenate only `type == "text"`; ignore malformed
content shapes instead of failing the stream.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_' -v`

Expected: PASS.

- [x] **Step 5: Run the full suite and commit**

Run: `cd llm-api-converter && go test -count=1 ./... && go vet ./...`

Expected: PASS and clean.

```bash
git add convert/anthropic_openai_p1_test.go convert/stream_anthropic_to_openai.go convert/stream.go convert/session.go convert/types.go docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md
git commit -m "fix(convert): merge stream usage and accept typed deltas"
```

**Implementation record**

- Status: Complete
- Date: 2026-11-06
- Repository / commit: `llm-api-converter` @ `e534397` (`fix(convert): merge stream usage and accept typed deltas`, branch `feat/anthropic-openai-p1`). The task commit contains the code, the tests, and this record; the resolved SHA was added by the follow-up docs commit that carries this line.
- Files changed:
  - `convert/stream_anthropic_to_openai.go` — `setUsage` merges non-zero fields instead of overwriting all four, so `message_start` input and cache counts survive the `message_delta` output update.
  - `convert/stream.go` — `StreamConverter.EmitError` builds its payload with `json.Marshal`.
  - `convert/session.go` — `PassthroughStreamHandler.EmitError` builds its payload with `json.Marshal`; dropped the now-unused `fmt` import.
  - `convert/types.go` — added `(*OpenAIDelta).UnmarshalJSON` and `joinOpenAIDeltaContent`; content accepts a string, `null`, or typed parts (only `type == "text"` is concatenated), and other shapes decode to empty content instead of failing the stream.
  - `convert/anthropic_openai_p1_test.go` — new: the three plan tests plus coverage for the passthrough `EmitError`, string/null/non-text/malformed delta content, and preservation of the other delta fields.
- Tests / commands:
  - RED: `cd llm-api-converter && go test ./convert -run 'TestP1_(AnthropicStreamUsageMergesPartialUpdates|AnthropicEmitErrorEscapesJSON|TypedDeltaContent)' -v` → 3 failures: `usage = ... PromptTokens:0 ...`; `invalid error JSON ...: invalid character 'q' after object key:value pair`; `unmarshal stream chunk: json: cannot unmarshal array into Go struct field OpenAIStreamChunk.choices.0.delta.content of type string`.
  - GREEN: `cd llm-api-converter && go test ./convert -run 'TestP1_' -v` → PASS (6 tests, 5 subtests).
  - `go test -count=1 ./...` → PASS (`convert`, `engine`, `rewriter`, `tests/e2e`).
  - `CGO_ENABLED=1 go test -race -count=1 ./convert ./engine ./rewriter` → PASS.
  - `go vet ./...` → clean; `git diff --check` → clean.
- Review findings: Self-review found no functional defects. Added three tests beyond the plan snippet to cover the `PassthroughStreamHandler.EmitError` site, the `null`/non-text/malformed content resolutions, and that the custom unmarshaller still decodes `role`, `reasoning_content`, and `tool_calls`; all passed on first run. Marshalling is unchanged because only `UnmarshalJSON` was added, so converter output stays byte-identical. The shadow struct inside `UnmarshalJSON` must stay in sync with `OpenAIDelta`; a comment records that. No dependency, registry, model-map, routing, or passthrough changes.
- Deferred: Per the plan's rule, zero usage fields mean "not reported", so a provider that reports a genuine zero after a non-zero value keeps the earlier value; no such provider is known. `gofmt -l convert/` lists eight pre-existing unformatted files (`anthropic_to_openai_test.go`, `convert_test.go`, `gemini_test.go`, `protocol.go`, `protocol_test.go`, `reasoning_cache.go`, `responses_parity_test.go`, `stream_test.go`), untouched by this task. The resolved SHA is recorded by the follow-up docs commit and Task 7's status table.

---

### Task 2: Message coalescing, tool-pair repair, history tool stubs

**Files:**
- Create: `convert/message_normalize.go`
- Modify: `convert/openai_to_anthropic.go:249-292` (after messages are built)
- Modify: `convert/responses.go:311-345`
- Test: `convert/anthropic_openai_p1_test.go`

**Interfaces:**
- Produces: `func normalizeAnthropicMessages(msgs []AnthropicMessage) []AnthropicMessage`
- Produces: `func stubHistoryTools(req *AnthropicRequest, msgs []AnthropicMessage)`
- `normalizeAnthropicMessages` merges consecutive same-role messages, puts
  `tool_result` blocks before text/image blocks, inserts synthetic results for
  unanswered calls, converts orphan results to text, and closes a trailing open
  exchange with a synthetic user message.
- `stubHistoryTools` adds one `AnthropicTool` per distinct `tool_use` name when
  `len(req.Tools) == 0`, with description
  `"Tool from earlier conversation. Do not call."` and schema
  `{"type":"object","properties":{}}`, then sets
  `req.ToolChoice = &AnthropicToolChoice{Type:"none"}`.

- [x] **Step 1: Write the failing tests**

```go
func TestP1_ParallelToolResultsCoalesce(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"start"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},
			{"id":"b","type":"function","function":{"name":"g","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"a","content":"ra"},
		{"role":"tool","tool_call_id":"b","content":"rb"},
		{"role":"user","content":"next"}]}`
	b, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatal(err)
	}
	if len(anth.Messages) != 3 {
		t.Fatalf("messages = %d, want 3: %s", len(anth.Messages), b)
	}
	last := anth.Messages[2].Content
	if len(last) != 3 || last[0].Type != "tool_result" || last[1].Type != "tool_result" || last[2].Type != "text" {
		t.Fatalf("last user blocks = %+v, want two tool_result then text", last)
	}
}

func TestP1_UnansweredToolUseGetsSyntheticResult(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"user","content":"continue"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	last := anth.Messages[len(anth.Messages)-1].Content
	if len(last) != 2 || last[0].Type != "tool_result" || last[0].ToolUseID != "a" || !last[0].IsError {
		t.Fatalf("blocks = %+v, want synthetic tool_result first", last)
	}
	if last[1].Type != "text" || last[1].Text != "continue" {
		t.Fatalf("text block = %+v", last[1])
	}
}

func TestP1_OrphanToolResultBecomesText(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"tool","tool_call_id":"missing","content":"orphan text"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	got := anth.Messages[0].Content[0]
	if got.Type != "text" || got.Text != "orphan text" {
		t.Fatalf("orphan result = %+v, want text block", got)
	}
}

func TestP1_TrailingToolUseGetsSyntheticUserResult(t *testing.T) {
	msgs := []AnthropicMessage{
		{Role: "user", Content: []AnthropicContent{{Type: "text", Text: "hi"}}},
		{Role: "assistant", Content: []AnthropicContent{{Type: "tool_use", ID: "a", Name: "f", Input: map[string]any{}}}},
	}
	got := normalizeAnthropicMessages(msgs)
	if len(got) != 3 || got[2].Role != "user" {
		t.Fatalf("messages = %+v, want trailing synthetic user", got)
	}
	if got[2].Content[0].Type != "tool_result" || got[2].Content[0].ToolUseID != "a" {
		t.Fatalf("trailing result = %+v", got[2].Content)
	}
}

func TestP1_HistoryToolsGetStubDeclarations(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
		{"role":"user","content":"go on"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	if len(anth.Tools) != 1 || anth.Tools[0].Name != "lookup" {
		t.Fatalf("tools = %+v, want one lookup stub", anth.Tools)
	}
	if anth.ToolChoice == nil || anth.ToolChoice.Type != "none" {
		t.Fatalf("tool_choice = %+v, want none", anth.ToolChoice)
	}
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_(ParallelToolResultsCoalesce|UnansweredToolUseGetsSyntheticResult|OrphanToolResultBecomesText|TrailingToolUseGetsSyntheticUserResult|HistoryToolsGetStubDeclarations)' -v`

Expected: FAIL — parallel results are separate user messages, no synthetic result, orphan result stays a `tool_result`, no stub tools.

- [x] **Step 3: Implement the normalizer and stubs**

Create `convert/message_normalize.go`. Algorithm:

1. Walk messages in order. For each message, split user content into
   `results` (type `tool_result`) and `rest` (everything else).
2. Maintain `pending []string`, `open map[string]bool`,
   `answered map[string]bool`.
3. On assistant: close the previous exchange (append a synthetic user message
   for unanswered pending calls), collect new `tool_use` IDs into `pending`,
   then append the assistant message.
4. On user: keep results whose ID is open and not already answered; convert
   orphan/duplicate results to text via `resultSaid`; append synthetic results
   for pending unanswered calls; emit `append(results, rest...)`; clear the
   exchange.
5. After the loop, close any open exchange.
6. Merge consecutive same-role messages by appending content.

Synthetic result: `AnthropicContent{Type:"tool_result", ToolUseID:id, Content:"Tool result missing.", IsError:true}`.
`resultSaid` returns a text block with the result text, or
`"Tool result received."` when empty.

Call `normalizeAnthropicMessages` in `convertOpenAIRequestToAnthropic` after
the message loop and before the empty-history fallback. Then call
`stubHistoryTools(&anthropic, anthropic.Messages)` when `len(anthropic.Tools) == 0`
(this also covers Responses tools that were all filtered as non-function).
Apply the same two calls in `ConvertResponsesToAnthropic` after its message
loop and tool mapping, using the same `len(anth.Tools) == 0` condition.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_' -v`

Expected: PASS.

- [x] **Step 5: Run the full suite and commit**

Run: `cd llm-api-converter && go test -count=1 ./... && go vet ./...`

Expected: PASS and clean.

```bash
git add convert/message_normalize.go convert/openai_to_anthropic.go convert/responses.go convert/anthropic_openai_p1_test.go docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md
git commit -m "fix(convert): normalize Anthropic tool histories"
```

**Implementation record**

- Status: Complete
- Date: 2026-11-06
- Repository / commit: `llm-api-converter` @ `7620613` (`fix(convert): normalize Anthropic tool histories`, branch `feat/anthropic-openai-p1`). The task commit contains the code, the tests, and this record; the resolved SHA was added by the follow-up docs commit that carries this line, matching Task 1.
- Files changed:
  - `convert/message_normalize.go` — new: `normalizeAnthropicMessages` processes each run of consecutive user messages as one turn (so parallel tool results coalesce and all `tool_result` blocks precede text/image blocks), keeps results whose ID is open and unanswered, converts orphan/duplicate results to text (`"Tool result received."` when empty), appends synthetic `is_error` results (`"Tool result missing."`) for unanswered calls, closes a trailing open exchange, and merges consecutive same-role messages. `stubHistoryTools` adds one deduplicated stub declaration per distinct history `tool_use` name (description `"Tool from earlier conversation. Do not call."`, schema `{"type":"object","properties":{}}`) and sets `tool_choice: {"type":"none"}`; it is a no-op when tools are already declared or no history names exist.
  - `convert/openai_to_anthropic.go` — calls the normalizer after the message loop and before the empty-history fallback; calls `stubHistoryTools` when no tools are declared.
  - `convert/responses.go` — applies the same normalizer/stub calls in `ConvertResponsesToAnthropic` after the message loop and tool mapping, when `len(anth.Tools) == 0`.
  - `convert/anthropic_openai_p1_test.go` — the five plan tests, a sixth Responses-path test, and three edge tests covering the exact synthetic/orphan text, stub schema/dedup/declared-tools guard, and results-first ordering across a user run.
- Tests / commands:
  - RED: `go test ./convert -run 'TestP1_(ParallelToolResultsCoalesce|UnansweredToolUseGetsSyntheticResult|OrphanToolResultBecomesText|TrailingToolUseGetsSyntheticUserResult|HistoryToolsGetStubDeclarations|ResponsesToolHistoryNormalized)' -v` → build failure `undefined: normalizeAnthropicMessages`; with a temporary passthrough stub all six failed for the expected reasons: `messages = 5, want 3`; `blocks = [text continue], want synthetic tool_result first`; orphan stayed `{Type:tool_result ...}`; `messages = 2, want trailing synthetic user`; `tools = [], want one lookup stub`; Responses `messages = 4, want 3`.
  - GREEN: `cd llm-api-converter && go test ./convert -run 'TestP1_' -v` → PASS (9 new Task 2 tests plus Task 1 tests).
  - `go test -count=1 ./...` → PASS (`convert`, `engine`, `rewriter`, `tests/e2e`).
  - `CGO_ENABLED=1 go test -race -count=1 ./convert ./engine ./rewriter` → PASS.
  - `go vet ./...` → clean; `git diff --check` → clean.
- Review findings: Self-review found no functional defects. Tool-use IDs are tracked only when non-empty: a legacy OpenAI `function_call` has no ID, so synthesizing a `tool_result` with an empty `tool_use_id` would be invalid; this also preserves `TestConvert_FunctionCallToToolUse` (its name still gets a stub). User runs are grouped before partitioning so a later result cannot end up after earlier text (`TestP1_UserRunKeepsResultsFirst`). Merging drops empty-content messages (Anthropic rejects empty arrays) and copies blocks instead of appending into the input's backing array. The normalizer/stub calls are guarded so declared tools, the pairwise converter registry/model map, and same-protocol passthrough are unchanged; no dependencies were added and no helper signatures were changed.
- Deferred: Distinct stub tools are emitted in history order rather than sorted; no ordering requirement exists and output is deterministic for a given input. An orphan tool result containing only image blocks falls back to `"Tool result received."`, losing the image; no known client sends image-only tool results. A history `tool_use` with an empty ID remains unpaired because Anthropic has no valid result ID to use. The resolved SHA is recorded by the follow-up docs commit, matching Task 1.

---

### Task 3: Thinking, reasoning_content, effort, and metadata gating

**Files:**
- Modify: `convert/openai_to_anthropic.go:270-300`, `:390-405`, `:480-515`
- Modify: `convert/responses.go:296-320`, `:697-720`
- Modify: `convert/anthropic_to_openai.go:216-222`, `:252`, `:298-310`, `:452-520`
- Test: `convert/anthropic_openai_p1_test.go`

**Interfaces:**
- Produces: `func replaysReasoning(model string) bool`
- Produces: `func anthropicMetadata(metadata any, user string) any`
- Changes:
  - `convertAssistantContent(msg OpenAIMessage, model string) []AnthropicContent`
  - `convertMessageContent(msg OpenAIMessage, model string) []AnthropicContent`
  - `convertChatMessageToAnthropic(msg OpenAIMessage, model string) AnthropicMessage`
  - `convertAnthropicMessage(msg AnthropicMessage, opts *ConvertOptions, model string, ctxParts []string) []OpenAIMessage`
  - `convertAnthropicAssistantMessage(msg AnthropicMessage, opts *ConvertOptions, model string, ctxParts []string) OpenAIMessage`

- [x] **Step 1: Write the failing tests**

```go
func TestP1_UnsignedThinkingDroppedForAnthropic(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"hi","reasoning_content":"chain",
		 "tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024, ResolvedModel: "gpt-4"})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	for _, block := range anth.Messages[0].Content {
		if block.Type == "thinking" {
			t.Fatalf("unsigned thinking sent to Anthropic target: %+v", block)
		}
	}

	b, _ = convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024, ResolvedModel: "deepseek-chat"})
	_ = json.Unmarshal(b, &anth)
	found := false
	for _, block := range anth.Messages[0].Content {
		found = found || block.Type == "thinking"
	}
	if !found {
		t.Fatal("DeepSeek target lost its thinking block")
	}
}

func TestP1_ReasoningContentGatedByTargetModel(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"messages":[
		{"role":"assistant","content":[{"type":"thinking","thinking":"chain"},{"type":"tool_use","id":"a","name":"f","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`

	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "" {
		t.Fatalf("gpt-4o received reasoning_content: %q", oai.Messages[0].ReasoningContent)
	}

	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "chain" {
		t.Fatalf("deepseek reasoning = %q, want chain", oai.Messages[0].ReasoningContent)
	}
}

func TestP1_DeepSeekPlaceholderOnlyForDeepSeek(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`

	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o", ReasoningCache: NewReasoningCache(10)})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "" {
		t.Fatalf("non-DeepSeek placeholder: %q", oai.Messages[0].ReasoningContent)
	}

	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat", ReasoningCache: NewReasoningCache(10)})
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent == "" {
		t.Fatal("DeepSeek lost its compatibility placeholder")
	}
}

func TestP1_EffortRequiresThinking(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"output_config":{"effort":"high"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.ReasoningEffort != nil {
		t.Fatalf("effort forwarded without thinking: %v", oai.ReasoningEffort)
	}

	body = `{"model":"claude","max_tokens":10,"thinking":{"type":"enabled","budget_tokens":1024},
		"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	_ = json.Unmarshal(b, &oai)
	if oai.ReasoningEffort != "high" {
		t.Fatalf("effort = %v, want high", oai.ReasoningEffort)
	}
}

func TestP1_MetadataWhitelist(t *testing.T) {
	body := `{"model":"gpt-4","user":"u2","metadata":{"user_id":"u1","trace":"x"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	m, _ := anth.Metadata.(map[string]any)
	if len(m) != 1 || m["user_id"] != "u1" {
		t.Fatalf("metadata = %+v, want only user_id=u1", anth.Metadata)
	}

	body = `{"model":"gpt-4","user":"u2","metadata":{"trace":"x"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ = convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	_ = json.Unmarshal(b, &anth)
	m, _ = anth.Metadata.(map[string]any)
	if len(m) != 1 || m["user_id"] != "u2" {
		t.Fatalf("metadata = %+v, want fallback user_id=u2", anth.Metadata)
	}
}

func TestP1_ResponsesUnsignedThinkingAndMetadata(t *testing.T) {
	body := `{"model":"gpt-4o","metadata":{"user_id":"u1","trace":"x"},"input":[
		{"type":"reasoning","text":"chain"},
		{"type":"function_call","call_id":"a","name":"f","arguments":"{}"}]}`
	b, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatal(err)
	}
	for _, msg := range anth.Messages {
		for _, block := range msg.Content {
			if block.Type == "thinking" {
				t.Fatalf("Responses unsigned thinking sent to Anthropic: %+v", block)
			}
		}
	}
	m, _ := anth.Metadata.(map[string]any)
	if len(m) != 1 || m["user_id"] != "u1" {
		t.Fatalf("Responses metadata = %+v, want only user_id=u1", anth.Metadata)
	}
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_(UnsignedThinkingDroppedForAnthropic|ReasoningContentGatedByTargetModel|DeepSeekPlaceholderOnlyForDeepSeek|EffortRequiresThinking|MetadataWhitelist)' -v`

Expected: FAIL — unsigned thinking and reasoning_content appear for non-DeepSeek targets; effort is forwarded without thinking; extra metadata keys pass through.

- [x] **Step 3: Implement the gating**

- `replaysReasoning(model)` returns `isDeepSeekModel(model) || isGLMModel(model)`.
- `anthropicMetadata(metadata any, user string) any` accepts
  `map[string]any`/`map[string]string`, keeps only `user_id`, falls back to
  `user`, returns nil when empty.
- Thread `model` through the converter signatures listed above.
- In `convertAssistantContent`, only append a `thinking` block when
  `isDeepSeekModel(model)`.
- In `convertAnthropicAssistantMessage`, only set `ReasoningContent` when
  `replaysReasoning(model)`; gate the cache lookup the same way; use
  `placeholderReasoning` only when `isDeepSeekModel(model)`.
- In the DeepSeek branch of `convertAnthropicRequestToOpenAI`, assign
  `ReasoningEffort` only when `req.Thinking != nil && req.Thinking.Type == "enabled"`.
- Replace the direct `req.Metadata` copies in `convertOpenAIRequestToAnthropic`
  and `ConvertResponsesToAnthropic` with `anthropicMetadata`.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_' -v`

Expected: PASS.

- [x] **Step 5: Run the full suite and commit**

Run: `cd llm-api-converter && go test -count=1 ./... && go vet ./...`

Expected: PASS and clean.

```bash
git add convert/openai_to_anthropic.go convert/responses.go convert/anthropic_to_openai.go convert/anthropic_openai_p1_test.go docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md
git commit -m "fix(convert): gate reasoning replay and whitelist metadata"
```

**Implementation record**

- Status: Complete
- Date: 2026-11-06
- Repository / commit: `llm-api-converter` @ `d2ec345` (`fix(convert): gate reasoning replay and whitelist metadata`, branch `feat/anthropic-openai-p1`). The task commit contains the code, the tests, and this record; the resolved SHA was added by the follow-up docs commit that carries this line, matching Tasks 1-2.
- Files changed:
  - `convert/anthropic_to_openai.go` — added `replaysReasoning` (DeepSeek or GLM). `convertAnthropicMessage`/`convertAnthropicAssistantMessage` now take the resolved target model; prior thinking/cached reasoning is replayed only when `replaysReasoning` is true, the reasoning-cache lookup is gated the same way, and `placeholderReasoning` is emitted only for DeepSeek. On the DeepSeek branch, `reasoning_effort` is forwarded only when `thinking.type == "enabled"`.
  - `convert/openai_to_anthropic.go` — added `anthropicMetadata` (keeps only `user_id`, falls back to `user`, returns nil when empty) and replaced the raw metadata copy. `convertMessageContent`/`convertAssistantContent` take the resolved target model and emit an unsigned `thinking` block from `reasoning_content` only for DeepSeek targets.
  - `convert/responses.go` — `ConvertResponsesToAnthropic` applies `anthropicMetadata` (no `user` fallback) and threads the resolved target model through `convertChatMessageToAnthropic` into the shared assistant-content helper.
  - `convert/anthropic_openai_p1_test.go` — the six plan tests plus a `replaysReasoning` target table, an `anthropicMetadata` table (including `map[string]string` and the empty result), and a GLM replay/no-placeholder test.
  - `convert/convert_test.go` — updated three existing tests whose old expectations contradicted P1-7: the thinking round-trip and both reasoning-cache tests now target `deepseek-chat` through the model map.
  - `convert/anthropic_to_openai_test.go` — updated `TestConvert_DeepSeekAdaptiveThinkingDropped` for P1-8: adaptive thinking no longer forwards `reasoning_effort`.
- Tests / commands:
  - RED: `go test ./convert -run 'TestP1_(UnsignedThinkingDroppedForAnthropic|ReasoningContentGatedByTargetModel|DeepSeekPlaceholderOnlyForDeepSeek|EffortRequiresThinking|MetadataWhitelist|ResponsesUnsignedThinkingAndMetadata)' -v` → six failures: unsigned thinking sent to a gpt-4 target; `gpt-4o received reasoning_content: "chain"`; non-DeepSeek placeholder; `effort forwarded without thinking: high`; metadata kept `trace`; Responses unsigned thinking.
  - GREEN: same command → PASS; `go test ./convert -run 'TestP1_' -v` → PASS (Tasks 1-3 tests).
  - `go test -count=1 ./...` → PASS (`convert`, `engine`, `rewriter`, `tests/e2e`).
  - `go test -race -count=1 ./...` → PASS.
  - `go vet ./...` → clean; `git diff --check` → clean.
- Review findings:
  - P1-5's plan assertion type-asserts `anth.Metadata.(map[string]any)`, so the helper returns `map[string]any` even for `map[string]string` input.
  - Four pre-existing tests asserted the old ungated behavior; they were updated rather than weakening the implementation. They now use a DeepSeek model map (thinking round-trip and reasoning-cache tests) or assert the P1-8 effort rule (adaptive thinking).
  - `convertAnthropicAssistantMessage` still parses thinking blocks for every target but assigns `ReasoningContent` only when replay is allowed; no other content changes.
  - No third-party dependency, registry/model-map, or same-protocol passthrough changes.
- Deferred:
  - Unsigned OpenAI reasoning is still replayed on the OpenAI→Anthropic response path (`convertOpenAIMessageToContent`), which the design explicitly leaves unchanged; only request-history replay is gated.
  - `replaysReasoning`/`isDeepSeekModel` match bare model prefixes; a `provider/deepseek-*` target string is not detected. This matches the existing DeepSeek checks in the same converter.
  - The resolved SHA is recorded by the follow-up docs commit and Task 7's status table.

---

### Task 4: Reasoning-model parameters and tool-stop inference

**Files:**
- Modify: `convert/types.go:219-240`
- Modify: `convert/anthropic_to_openai.go:193-201`, `:570-590`
- Modify: `convert/openai_to_anthropic.go:570-575`
- Modify: `convert/stream.go:248-258`
- Modify: `convert/stream_anthropic_to_openai.go:245-270`
- Test: `convert/anthropic_openai_p1_test.go`

**Interfaces:**
- Produces: `func isOpenAIReasoningModel(model string) bool`
- `isOpenAIReasoningModel` strips a `provider/` prefix and matches `o1`,
  `o3`, `o4`, or `gpt-5` prefixes.

- [ ] **Step 1: Write the failing tests**

```go
func TestP1_ReasoningModelUsesMaxCompletionTokens(t *testing.T) {
	body := `{"model":"claude","max_tokens":100,"temperature":0.5,"top_p":0.9,
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "o3-mini"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.MaxCompletionTokens == nil || *oai.MaxCompletionTokens != 100 {
		t.Fatalf("max_completion_tokens = %v, want 100", oai.MaxCompletionTokens)
	}
	if oai.MaxTokens != nil || oai.Temperature != nil || oai.TopP != nil {
		t.Fatalf("reasoning model kept unsupported params: %s", b)
	}

	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	_ = json.Unmarshal(b, &oai)
	if oai.MaxTokens == nil || *oai.MaxTokens != 100 {
		t.Fatalf("non-reasoning max_tokens = %v, want 100", oai.MaxTokens)
	}
	if oai.Temperature == nil || oai.TopP == nil {
		t.Fatalf("non-reasoning sampling params dropped: %s", b)
	}
}

func TestP1_NonStreamToolStopInference(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop",
		"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	b, _ := convertOpenAIResponseToAnthropic([]byte(body), nil)
	var anth AnthropicResponse
	_ = json.Unmarshal(b, &anth)
	if anth.StopReason == nil || *anth.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %v, want tool_use", anth.StopReason)
	}

	body = `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"content_filter",
		"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	b, _ = convertOpenAIResponseToAnthropic([]byte(body), nil)
	_ = json.Unmarshal(b, &anth)
	if anth.StopReason == nil || *anth.StopReason != "end_turn" {
		t.Fatalf("content_filter stop_reason = %v, want end_turn", anth.StopReason)
	}

	body = `{"id":"msg_1","model":"claude","stop_reason":"end_turn",
		"content":[{"type":"tool_use","id":"a","name":"f","input":{}}]}`
	b, _ = convertAnthropicResponseToOpenAI([]byte(body))
	var oai OpenAIChatResponse
	_ = json.Unmarshal(b, &oai)
	if oai.Choices[0].FinishReason == nil || *oai.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %v, want tool_calls", oai.Choices[0].FinishReason)
	}
}

func TestP1_StreamToolStopInference(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`))
	out := string(sc.HandleStreamEnd())
	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Fatalf("OpenAI→Anthropic stop = %s", out)
	}

	c := NewAnthropicToOpenAIStreamConverter("claude")
	_, _ = c.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_1","model":"claude"}}`))
	_, _ = c.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f"}}`))
	_, _ = c.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`))
	out = string(c.HandleStreamEnd())
	if !strings.Contains(out, `"finish_reason":"tool_calls"`) {
		t.Fatalf("Anthropic→OpenAI finish = %s", out)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_(ReasoningModelUsesMaxCompletionTokens|NonStreamToolStopInference|StreamToolStopInference)' -v`

Expected: FAIL — `max_tokens`/sampling still emitted for `o3`, stop reasons remain `end_turn`/`stop` with tools.

- [ ] **Step 3: Implement the mappings**

- Add `isOpenAIReasoningModel`.
- In `convertAnthropicRequestToOpenAI`, when true, set `MaxCompletionTokens`
  and skip `MaxTokens`, `Temperature`, `TopP`.
- Non-stream OpenAI→Anthropic: after building `anth.Content`, if it contains a
  `tool_use` and the raw `finish_reason` is nil or `"stop"`, set
  `StopReason = ptr("tool_use")`. Leave `content_filter` alone.
- Non-stream Anthropic→OpenAI: if `msg.ToolCalls` is non-empty and the raw
  `stop_reason` is nil or `"end_turn"`, set `FinishReason = ptr("tool_calls")`.
- Stream OpenAI→Anthropic: if tools were accumulated and the raw
  `finishReason` is empty or `"stop"`, set `stopReason = "tool_use"`; otherwise
  map the raw reason as today.
- Stream Anthropic→OpenAI: if `nextTool > 0` and the raw stop is empty or
  `"end_turn"`, set the finish chunk reason to `"tool_calls"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_' -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `cd llm-api-converter && go test -count=1 ./... && go vet ./...`

Expected: PASS and clean.

```bash
git add convert/types.go convert/anthropic_to_openai.go convert/openai_to_anthropic.go convert/stream.go convert/stream_anthropic_to_openai.go convert/anthropic_openai_p1_test.go docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md
git commit -m "fix(convert): map reasoning-model params and tool stops"
```

**Implementation record**

- Status: Complete
- Date: 2026-11-06
- Repository / commit: `llm-api-converter` @ `pending` (`fix(convert): map reasoning-model params and tool stops`, branch `feat/anthropic-openai-p1`). The task commit contains the code, the tests, and this record; the resolved SHA was added by the follow-up docs commit that carries this line, matching Tasks 1-3.
- Files changed:
  - `convert/types.go` — added `isOpenAIReasoningModel`: strips a `provider/` prefix with `StripProviderPrefix`, lowercases, and matches the `o1`, `o3`, `o4`, and `gpt-5` prefixes.
  - `convert/anthropic_to_openai.go` — `convertAnthropicRequestToOpenAI` computes the reasoning-model flag from the resolved output model; reasoning targets emit `max_completion_tokens` and omit `max_tokens`, `temperature`, and `top_p`, while all other targets keep the previous behavior. The Task 3 DeepSeek `thinking`/`reasoning_effort` gating and GLM thinking mapping are untouched. In `convertAnthropicResponseToOpenAI`, the finish reason is inferred from the raw `stop_reason` (nil or `end_turn`) plus emitted `tool_calls` before mapping; explicit reasons still go through `mapFinishReason`.
  - `convert/openai_to_anthropic.go` — after `anth.Content` is built, a nil or `stop` raw `finish_reason` with a converted `tool_use` block sets `stop_reason: tool_use`. Added `hasAnthropicToolUse` for that check. `content_filter`, `length`, and other explicit reasons keep their existing mapping.
  - `convert/stream.go` — `HandleStreamEnd` infers `stop_reason: tool_use` from the raw `finish_reason` being empty or `stop` when any tool call was accumulated (including calls held until stream end); explicit reasons still pass through `mapOpenAIStreamFinish`.
  - `convert/stream_anthropic_to_openai.go` — `finish` infers `finish_reason: tool_calls` when a tool block was emitted and the raw `stop_reason` is empty or `end_turn`; `max_tokens` and other explicit reasons keep their mapping.
  - `convert/anthropic_openai_p1_test.go` — the three plan tests plus a detection table (provider prefix, case, `o4`/`gpt-5`, non-reasoning names), a provider-prefixed reasoning-parameter test, explicit-stop preservation tests for both non-stream directions, and a stream explicit-stop test.
- Tests / commands:
  - RED: `go test ./convert -run 'TestP1_(ReasoningModelUsesMaxCompletionTokens|NonStreamToolStopInference|StreamToolStopInference)' -v` → build failure `undefined: isOpenAIReasoningModel`; with a temporary passthrough stub the three tests failed for the expected reasons: `max_completion_tokens = <nil>, want 100`; `stop_reason = <end_turn>, want tool_use`; OpenAI→Anthropic stream emitted `"stop_reason":"end_turn"` after a tool call. The extra tests separately showed `isOpenAIReasoningModel` returning false for `o1`/`o3`/`o4`/`gpt-5` names, sampling params retained for `openai/gpt-5`, and a nil `finish_reason` left as nil.
  - GREEN: `go test ./convert -run 'TestP1_' -v` → PASS (all Task 1-4 tests).
  - `go test -count=1 ./...` → PASS (`convert`, `engine`, `rewriter`, `tests/e2e`).
  - `CGO_ENABLED=1 go test -race -count=1 ./convert ./engine ./rewriter` → PASS.
  - `go vet ./...` → clean; `gofmt -l` on the six changed files → clean; `git diff --check` → clean.
- Review findings:
  - The inference reads the raw reason before mapping, as required, so `content_filter` and `length`/`max_tokens` are never rewritten; the explicit-stop tests cover both directions.
  - Held tool calls are already present in `toolCallByIndex` (the map is only cleared for `heldTools`), so the stream inference naturally counts them; tool calls skipped by the declared-tools restriction are absent and therefore do not force a tool stop.
  - Reasoning detection uses the resolved output model, not the client model, so a Claude request routed to `o3-mini` gets reasoning parameters.
  - Task 3's effort gating and GLM thinking mapping were preserved; no dependency, registry/model-map, routing, or same-protocol passthrough changes. No pre-existing test contradicted P1-6/P1-11, so none needed updating.
- Deferred:
  - Detection is prefix-based by design, so a hypothetical `gpt-5-chat-*` or `o10` name would be treated as a reasoning model and lose sampling parameters; no such model is in use.
  - A provider that sends a literal empty-string non-stream `finish_reason` is treated as an explicit reason (the plan and design only specify nil/`stop`); stream handling does treat empty as missing because the accumulated reason is a plain string.
  - The Anthropic→OpenAI stream inference counts a tool synthesized from a bare `input_json_delta`, because the converter emits a tool-call delta for it.
  - `gofmt -l convert/` still lists the same eight pre-existing unformatted files noted in Task 1; this task did not touch them.
  - The resolved SHA is recorded by the follow-up docs commit, matching Tasks 1-3.

---

### Task 5: URL images and structured output

**Files:**
- Modify: `convert/types.go:11-35`, `:130-133`, `:310-314`
- Modify: `convert/openai_to_anthropic.go:215-300`, `:419-453`
- Modify: `convert/anthropic_to_openai.go:193-222`, `:340-365`, `:423-450`
- Test: `convert/anthropic_openai_p1_test.go`

**Interfaces:**
- Adds `OpenAIResponseFormat`, `OpenAIJSONSchema`, `AnthropicOutputFormat`,
  `AnthropicImageSource.URL`.
- Adds `OpenAIChatRequest.ResponseFormat *OpenAIResponseFormat` and
  `AnthropicOutputConfig.Format *AnthropicOutputFormat`.

- [ ] **Step 1: Write the failing tests**

```go
func TestP1_OpenAIImageURLPreserved(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":[
		{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	img := anth.Messages[0].Content[1]
	if img.Type != "image" || img.Source == nil || img.Source.Type != "url" || img.Source.URL != "https://example.com/cat.png" {
		t.Fatalf("image = %+v", img)
	}
}

func TestP1_AnthropicURLImagePreserved(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":[
		{"type":"text","text":"see"},{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}]}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	parts, _ := oai.Messages[0].Content.([]any)
	if len(parts) != 2 {
		t.Fatalf("parts = %+v", parts)
	}
	img, _ := parts[1].(map[string]any)
	imageURL, _ := img["image_url"].(map[string]any)
	if imageURL["url"] != "https://example.com/cat.png" {
		t.Fatalf("image part = %+v", img)
	}
}

func TestP1_StructuredOutputJSONSchemaToAnthropic(t *testing.T) {
	body := `{"model":"gpt-4","response_format":{"type":"json_schema","json_schema":{
		"name":"out","schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	if anth.OutputConfig == nil || anth.OutputConfig.Format == nil || anth.OutputConfig.Format.Type != "json_schema" {
		t.Fatalf("output_config = %+v", anth.OutputConfig)
	}
	schema, _ := anth.OutputConfig.Format.Schema.(map[string]any)
	if schema["type"] != "object" {
		t.Fatalf("schema = %+v", schema)
	}
}

func TestP1_AnthropicStructuredOutputToOpenAI(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"output_config":{"format":{
		"type":"json_schema","schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.ResponseFormat == nil || oai.ResponseFormat.Type != "json_schema" || oai.ResponseFormat.JSONSchema == nil {
		t.Fatalf("response_format = %+v", oai.ResponseFormat)
	}
	if oai.ResponseFormat.JSONSchema.Name != "response" {
		t.Fatalf("json_schema.name = %q, want response", oai.ResponseFormat.JSONSchema.Name)
	}
}

func TestP1_JSONObjectFallsBackToInstruction(t *testing.T) {
	body := `{"model":"gpt-4","response_format":{"type":"json_object"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	if anth.OutputConfig != nil && anth.OutputConfig.Format != nil {
		t.Fatalf("json_object emitted a schema: %+v", anth.OutputConfig)
	}
	found := false
	for _, block := range anth.System {
		found = found || strings.Contains(strings.ToLower(block.Text), "json")
	}
	if !found {
		t.Fatalf("no JSON instruction in system: %+v", anth.System)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_(OpenAIImageURLPreserved|AnthropicURLImagePreserved|StructuredOutputJSONSchemaToAnthropic|AnthropicStructuredOutputToOpenAI|JSONObjectFallsBackToInstruction)' -v`

Expected: FAIL — URL images are dropped; `response_format`/`output_config.format` are not modeled.

- [ ] **Step 3: Implement the mappings**

- Add `URL string json:"url,omitempty"` to `AnthropicImageSource`.
- `convertImageURL`: for non-data URLs return
  `AnthropicContent{Type:"image", Source:&AnthropicImageSource{Type:"url", URL:url}}`.
- Anthropic→OpenAI user content: for `block.Source.Type == "url"` append
  `{"type":"image_url","image_url":{"url":...}}`. Reuse for tool-result images.
- Add the response-format types and fields.
- OpenAI→Anthropic: `json_schema` with a non-empty object schema sets
  `OutputConfig.Format`; `json_object` appends
  `"Respond with a JSON object only."` to the system texts.
- Anthropic→OpenAI: `output_config.format.type == "json_schema"` with a schema
  sets `ResponseFormat` with name `"response"`; a format without a schema
  appends the same JSON instruction to the system message.
- Merge the structured-output instruction with the existing tool-choice
  instruction using `"\n\n"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP1_' -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `cd llm-api-converter && go test -count=1 ./... && go vet ./...`

Expected: PASS and clean.

```bash
git add convert/types.go convert/openai_to_anthropic.go convert/anthropic_to_openai.go convert/anthropic_openai_p1_test.go docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md
git commit -m "fix(convert): preserve image URLs and structured output"
```

**Implementation record**

- Status: Pending
- Date:
- Repository / commit:
- Files changed:
- Tests / commands:
- Review findings:
- Deferred:

---

### Task 6: Stream errors and `content_block_start` payloads

**Files:**
- Modify: `convert/types.go:491-498`
- Modify: `convert/stream.go:112-127`, `:185-285`
- Modify: `convert/stream_anthropic_to_openai.go:16-45`, `:66-70`, `:110-170`, `:194-206`
- Modify: `convert/convert.go:1011-1027`, `:1143-1154`
- Modify: `engine/engine.go:29-35`, `:95-110`
- Modify: `rewriter/server.go:97-117`
- Test: `convert/anthropic_openai_p1_test.go`, `engine/engine_test.go`, `rewriter/server_test.go`

**Interfaces:**
- Adds `OpenAIStreamError{Type, Message, Code string}` and
  `OpenAIStreamChunk.Error *OpenAIStreamError`.
- Adds `engine.Meta.StreamError string`.
- `HandleSSEEvent` end phase emits `EmitError` when `opts.ErrorMsg` is set and
  the handler is not a `*PassthroughStreamHandler`.
- Anthropic→OpenAI `content_block_start` consumes `text`, `thinking`, and
  `input`.

- [ ] **Step 1: Write the failing tests**

```go
func TestP1_OpenAIStreamErrorChunk(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"error":{"type":"server_error","message":"boom \"x\""}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `event: error`) || !strings.Contains(string(out), `boom \"x\"`) {
		t.Fatalf("error chunk output = %s", out)
	}
	if end := sc.HandleStreamEnd(); end != nil {
		t.Fatalf("normal stop after error: %s", end)
	}
}

func TestP1_AnthropicStreamErrorChunk(t *testing.T) {
	c := NewAnthropicToOpenAIStreamConverter("claude")
	out, _ := c.HandleChunk([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"boom"}}`))
	if !strings.Contains(string(out), `"error"`) || !strings.Contains(string(out), `boom`) {
		t.Fatalf("error output = %s", out)
	}
	if end := c.HandleStreamEnd(); end != nil {
		t.Fatalf("normal finish after error: %s", end)
	}
}

func TestP1_EndPhaseStreamErrorEmitsError(t *testing.T) {
	opts := &ConvertOptions{
		SessionStore: NewSessionStore(),
		Direction:    "response",
		URI:          "/v1/messages",
		SID:          "s1",
	}
	start := []byte(`data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)
	if _, err := HandleSSEEvent("s1", "start", 0, start, opts); err != nil {
		t.Fatal(err)
	}
	opts.ErrorMsg = "upstream reset"
	out, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	if !strings.Contains(string(out), `event: error`) || !strings.Contains(string(out), `upstream reset`) {
		t.Fatalf("end error = %s", out)
	}
	if strings.Contains(string(out), `message_stop`) {
		t.Fatalf("normal stop after error: %s", out)
	}
}

func TestP1_PassthroughEndPhaseIgnoresError(t *testing.T) {
	store := NewSessionStore()
	store.Set("s2", &Session{StreamHandler: NewPassthroughStreamHandler("claude", ProtocolAnthropic, false)})
	opts := &ConvertOptions{SessionStore: store, ErrorMsg: "upstream reset"}
	out, _ := HandleSSEEvent("s2", "end", 0, nil, opts)
	if out != nil {
		t.Fatalf("passthrough emitted %s", out)
	}
}

func TestP1_AnthropicBlockStartFields(t *testing.T) {
	c := NewAnthropicToOpenAIStreamConverter("claude")
	out, _ := c.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_1","model":"claude"}}`))
	out2, _ := c.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hello"}}`))
	out3, _ := c.HandleChunk([]byte(`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"a","name":"f","input":{"x":1}}}`))
	out4, _ := c.HandleChunk([]byte(`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"y\":2}"}}`))
	all := string(out) + string(out2) + string(out3) + string(out4)
	if !strings.Contains(all, `"content":"hello"`) {
		t.Fatalf("start text missing: %s", all)
	}
	if !strings.Contains(all, `"arguments":"{\"x\":1}"`) {
		t.Fatalf("start tool input missing: %s", all)
	}
	if strings.Contains(all, `{\"y\":2}`) {
		t.Fatalf("delta duplicated start input: %s", all)
	}
}
```

Add to `engine/engine_test.go`:

```go
func TestRewriteEndPhaseStreamError(t *testing.T) {
	e := New(Options{Model: "claude", MaxTokens: 8192})
	start := []byte(`data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)
	if _, err := e.Rewrite(context.Background(), start, Meta{
		Direction: "response", URI: "/v1/messages", SID: "s1", SSEPhase: "start",
	}); err != nil {
		t.Fatal(err)
	}
	res, err := e.Rewrite(context.Background(), nil, Meta{
		Direction: "response", URI: "/v1/messages", SID: "s1", SSEPhase: "end",
		StreamError: "upstream reset",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(res.Data), "event: error") {
		t.Fatalf("end phase = %s", res.Data)
	}
}
```

Add to `rewriter/server_test.go`:

```go
func TestMetaFromStreamError(t *testing.T) {
	m := metaFrom([]byte(`{"stream_error":"upstream reset","sse_phase":"end"}`))
	if m.StreamError != "upstream reset" {
		t.Fatalf("stream error = %q", m.StreamError)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert ./engine ./rewriter -run 'TestP1_(OpenAIStreamErrorChunk|AnthropicStreamErrorChunk|EndPhaseStreamErrorEmitsError|PassthroughEndPhaseIgnoresError|AnthropicBlockStartFields)|TestRewriteEndPhaseStreamError|TestMetaFromStreamError' -v`

Expected: FAIL/compile error — `OpenAIStreamChunk.Error`, `StreamError`, and start payload fields do not exist; error chunks are swallowed.

- [ ] **Step 3: Implement the error and start-field handling**

- Add `OpenAIStreamError` and `OpenAIStreamChunk.Error`.
- In `StreamConverter.HandleChunk`, check `chunk.Error` before the
  `len(chunk.Choices) == 0` branch: emit `sc.EmitError(message)`, set
  `sc.finalized = true`, return the event. Default the message to
  `"upstream stream error"` when empty.
- In `AnthropicToOpenAIStreamConverter.HandleChunk` `case "error"`, set
  `c.finished = true` after appending the error event.
- Add `StreamError` to `engine.Meta`, parse `stream_error` in `metaFrom`, and
  set `ErrorMsg: meta.StreamError` in the `ConvertOptions` built by
  `Engine.Rewrite`.
- In `HandleSSEEvent` end phase and the Responses end phase, if
  `opts.ErrorMsg != ""`:
  - passthrough handler → return nil;
  - otherwise → return `handler.EmitError(opts.ErrorMsg)`.
- Extend the Anthropic start event with `Text`, `Thinking`, `Signature`, and
  `Input json.RawMessage`. On `content_block_start`:
  - `text` → emit a text delta;
  - `thinking` → emit a `reasoning_content` delta;
  - `tool_use` → emit the start chunk; when `Input` is non-empty and not
    `{}`/`null`, set `Function.Arguments` to the raw input and mark that block
    complete. Skip later `input_json_delta` events for that block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert ./engine ./rewriter -run 'TestP1_|TestRewriteEndPhaseStreamError|TestMetaFromStreamError' -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite and commit**

Run: `cd llm-api-converter && go test -count=1 ./... && go vet ./...`

Expected: PASS and clean.

```bash
git add convert/types.go convert/stream.go convert/stream_anthropic_to_openai.go convert/convert.go engine/engine.go rewriter/server.go convert/anthropic_openai_p1_test.go engine/engine_test.go rewriter/server_test.go docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md
git commit -m "fix(convert): surface stream errors and block-start payloads"
```

**Implementation record**

- Status: Pending
- Date:
- Repository / commit:
- Files changed:
- Tests / commands:
- Review findings:
- Deferred:

---

### Task 7: Verification, documentation, and memory

**Files:**
- Modify: `docs/reviews/2026-11-06-anthropic-openai-chat-review.md`
- Modify: `docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md`
- Modify: `/root/code/go-gost/.memory/notes/llm-api-converter-magpie-eval.md`
- Modify: `/root/code/go-gost/.memory/MEMORY.md` (leave uncommitted if unrelated edits remain)

**Interfaces:**
- Consumes all previous task commits.

- [ ] **Step 1: Run the full verification suite**

Run:

```sh
cd llm-api-converter
go test -count=1 ./...
CGO_ENABLED=1 go test -race -count=1 ./convert ./engine ./rewriter
go vet ./...
git diff --check
```

Expected: all PASS; `git diff --check` clean.

- [ ] **Step 2: Update the review document**

Add a `P1 Status` table after the P0 status table with 15 rows:

| ID | Status | Commit |
|---|---|---|
| P1-1 … P1-15 | fixed | the four-to-six task SHAs |

Update the header status to `P0 and P1 fixed and verified; P2 findings remain
open.` Add one line with the verification commands and the final HEAD SHA.

- [ ] **Step 3: Fill all Implementation records**

Ensure Tasks 1–6 have date, repository/commit, files, commands/results,
review findings, and deferred items. Tick completed checkboxes. Add a
`Post-review fix wave` record if the final review produced one.

- [ ] **Step 4: Update the memory note**

In `/root/code/go-gost/.memory/notes/llm-api-converter-magpie-eval.md`, change
the 2026-11-06 observation to say P1 is also fixed and list the task SHAs.
Update the `MEMORY.md` index line path/status. Stage only the note; if
`MEMORY.md` still has unrelated edits, leave it uncommitted and record that.

- [ ] **Step 5: Commit docs and memory**

```bash
cd /root/code/go-gost/llm-api-converter
git add docs/reviews/2026-11-06-anthropic-openai-chat-review.md docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p1.md
git commit -m "docs(convert): record P1 review status"

cd /root/code/go-gost
git add .memory/notes/llm-api-converter-magpie-eval.md
git commit -m "docs(memory): record llm-api-converter P1 review"
```

Record both repository SHAs in the Task 7 Implementation record. If a commit
cannot contain its own SHA, use the same follow-up docs commit pattern as P0.

**Implementation record**

- Status: Pending
- Date:
- Repository / commit:
- Files changed:
- Tests / commands:
- Review findings:
- Deferred:

---

## Self-Review Notes

- The plan covers every P1 finding from the review: P1-1/P1-13/P1-14 in Task 1,
  P1-2/P1-3 in Task 2, P1-4/P1-5/P1-7/P1-8 in Task 3, P1-6/P1-11 in Task 4,
  P1-9/P1-10 in Task 5, P1-12/P1-15 in Task 6.
- The builder-check rulings are reflected: tool results first, trailing open
  calls closed, passthrough error guard, raw-reason stop inference, Responses
  shared helpers, and `provider/`-prefix stripping.
- All new helper names used by later tasks are defined in the `Interfaces`
  blocks.
