# Anthropic ↔ OpenAI Chat P0 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the four P0 protocol violations in the Anthropic ↔ OpenAI Chat conversion: empty text blocks, orphan `tool_choice`, missing `tool_use.input`, and interleaved streaming tool-call corruption.

**Architecture:** Keep the pairwise converter registry. Add small shared normalization helpers in the `convert` package and apply them at both request and response boundaries. Do not introduce a canonical IR or new dependencies. Preserve the existing streaming behavior for sequential tool calls; buffer only the conflicting sibling calls that arrive while the open call's arguments are still incomplete JSON.

**Tech Stack:** Go 1.26.4, standard library, existing `go test` suite, `go vet`, optional `CGO_ENABLED=1 go test -race`.

**Spec:** `docs/reviews/2026-11-06-anthropic-openai-chat-review.md`

## Global Constraints

- Do not add third-party dependencies.
- Keep the pairwise `conversions` registry and model-map behavior intact.
- Do not change passthrough behavior for same-protocol traffic.
- Code comments are English; user-facing documentation may be Chinese.
- Every task must leave `go test ./...` green and be committed separately.
- Never push. Commit only.
- Do not modify `core/` or other go-gost modules for this plan.
- Every task carries an **Implementation record** filled in after completion.

## Review Focus

These are the input classes most likely to bite a user; each has a regression test in the owning task:

1. **Tool-only assistant turn** (`content: null` + `tool_calls`) must not produce an empty/missing text block.
2. **No-tools request** (title/classifier calls) must not produce `tool_choice` without `tools`.
3. **Empty or invalid tool arguments** must become an object (`{}`) in both directions; never omitted and never `"null"`.
4. **Interleaved streamed tool calls** (`index 0, 1, 0, 1`) must keep each call's arguments separate and valid.
5. **All messages filtered empty** must still produce at least one valid user message, not an invalid empty request.

## P0 Schedule

| Order | Task | Estimate | Depends on | Primary files |
|---|---|---|---|---|
| 1 | Filter empty text blocks | 0.5 day | — | `convert/openai_to_anthropic.go`, `convert/responses.go`, tests |
| 2 | Omit `tool_choice` without tools | 0.25 day | — | `convert/openai_to_anthropic.go`, `convert/responses.go`, tests |
| 3 | Normalize tool input/arguments | 0.5 day | — | new `convert/tool_input.go`, both request/response paths |
| 4 | Fix interleaved streaming tool calls | 1 day | — | `convert/stream.go`, `convert/types.go`, `convert/stream_test.go` |
| 5 | Verification and documentation | 0.25 day | 1–4 | review doc, this plan, memory note |

Suggested execution: Tasks 1 and 2 first (smallest, highest-frequency 400s), then
Task 3, then Task 4. Tasks 1–3 touch shared converters and should be done
sequentially to avoid conflicts. Total estimate: ~2.5 dev days.

---

### Task 1: Filter empty text blocks

**Files:**
- Modify: `convert/openai_to_anthropic.go`
- Modify: `convert/responses.go` (shared Chat→Anthropic message loop)
- Test: new `convert/anthropic_openai_p0_test.go`
- Update: `convert/convert_test.go:230-250`

**Interfaces:**
- Consumes: `AnthropicContent`, `AnthropicMessage`, `convertUserContent`, `convertAssistantContent`, `convertMessageContent`.
- Produces:
  - `func appendTextBlock(blocks []AnthropicContent, text string) []AnthropicContent`
  - `convertMessageContent` may now return an empty slice; callers must skip empty messages.
  - `convertUserContent` and `convertAssistantContent` return no empty/whitespace-only text blocks.

- [ ] **Step 1: Write the failing tests**

Add to `convert/anthropic_openai_p0_test.go`:

```go
package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestP0_OpenAIToAnthropicToolOnlyAssistantHasNoEmptyText(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"weather?"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}
		]},
		{"role":"tool","tool_call_id":"call_1","content":"sunny"}
	]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte(`{"type":"text"}`)) {
		t.Fatalf("empty text block leaked: %s", b)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	for _, m := range a.Messages {
		for _, c := range m.Content {
			if c.Type == "text" && strings.TrimSpace(c.Text) == "" {
				t.Fatalf("empty text block: %+v\nbody: %s", m, b)
			}
		}
	}
}

func TestP0_OpenAIToAnthropicEmptyUserMessageDropped(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":""},
		{"role":"user","content":"hi"}
	]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Messages) != 1 || a.Messages[0].Content[0].Text != "hi" {
		t.Fatalf("empty user message not dropped: %s", b)
	}
}

func TestP0_OpenAIToAnthropicAllEmptyMessagesStillHaveUser(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":""}]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Messages) != 1 || a.Messages[0].Role != "user" || len(a.Messages[0].Content) == 0 {
		t.Fatalf("empty request not repaired: %s", b)
	}
}

func TestP0_ResponsesToAnthropicToolOnlyAssistantHasNoEmptyText(t *testing.T) {
	body := `{"model":"gpt-5","instructions":"be brief","input":[
		{"type":"message","role":"user","content":"weather?"},
		{"type":"function_call","id":"call_1","name":"get_weather","arguments":"{\"loc\":\"NYC\"}"},
		{"type":"function_call_output","call_id":"call_1","output":"sunny"}
	]}`
	o := &ConvertOptions{Model: "claude", MaxTokens: 8192, URI: "/v1/responses",
		ModelMap: ModelMap{{SourcePrefix: "gpt-5", TargetModel: "claude", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), o)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte(`{"type":"text"}`)) {
		t.Fatalf("empty text block leaked from Responses path: %s", b)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert -run 'TestP0_(OpenAI|Responses)ToAnthropic' -v`

Expected: FAIL. The first test prints an empty text block; the empty-user test
sees two messages.

- [ ] **Step 3: Implement `appendTextBlock` and use it**

In `convert/openai_to_anthropic.go`, add near `extractTextContent`:

```go
// appendTextBlock appends a text block only when text has non-whitespace
// content. Anthropic rejects empty and whitespace-only text blocks.
func appendTextBlock(blocks []AnthropicContent, text string) []AnthropicContent {
	if strings.TrimSpace(text) == "" {
		return blocks
	}
	return append(blocks, AnthropicContent{Type: "text", Text: text})
}
```

Replace the direct text-block appends in `convertUserContent` and
`convertAssistantContent`:

- nil/empty user content: return `nil`
- empty string user content: return `nil`
- text parts: `blocks = appendTextBlock(blocks, txt)`
- tool-call path: `blocks = appendTextBlock(blocks, extractTextContent(msg.Content))`
- reasoning-without-tools path: build the reasoning block first, then append
  text only via `appendTextBlock`
- plain text path: use `appendTextBlock`; if no blocks result, return `nil`

In `convertOpenAIRequestToAnthropic` (around `:273-287`), skip messages whose
`Content` is empty:

```go
am.Content = convertMessageContent(m)
if len(am.Content) == 0 {
	continue
}
anthropic.Messages = append(anthropic.Messages, am)
```

Apply the same `if len(am.Content) == 0 { continue }` rule in
`convert/responses.go` around `:315-318`.

- [ ] **Step 4: Update the old assertion**

`convert/convert_test.go:242` currently expects `len(...) == 2` for
“empty text + tool_use”. Change it to expect one block (the `tool_use`) and
add an assertion that no text block is empty.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP0_|TestConvert_AssistantToolCalls|TestConvert_NoTools' -v`

Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `cd llm-api-converter && go test ./...`

Expected: PASS.

- [ ] **Step 7: Fill the implementation record and commit**

Record status, date, commit SHA, files changed, commands/results, review
findings, and deferred items in the block below.

```bash
git add convert/openai_to_anthropic.go convert/responses.go convert/anthropic_openai_p0_test.go convert/convert_test.go docs/plans/2026-11-06-anthropic-openai-chat-p0.md
git commit -m "fix(convert): drop empty Anthropic text blocks"
```

**Implementation record**

- Status: Done
- Date: 2026-11-06
- Repository / commit: `llm-api-converter` @ `ef2230b` (`fix(convert): drop empty Anthropic text blocks`, branch `feat/anthropic-openai-p0`); the record itself lands in the follow-up docs commit `docs(convert): record task 1 implementation`.
- Files changed:
  - `convert/openai_to_anthropic.go` — added `appendTextBlock`; `convertUserContent`/`convertAssistantContent` no longer emit empty or whitespace-only text blocks and return `nil` when nothing valid remains; `convertOpenAIRequestToAnthropic` skips messages whose content is empty.
  - `convert/responses.go` — `ConvertResponsesToAnthropic` applies the same skip rule in the shared Chat→Anthropic message loop.
  - `convert/anthropic_openai_p0_test.go` — new; P0-1 regression tests for the Chat and Responses request paths, empty user messages, and the all-empty fallback.
  - `convert/convert_test.go` — tool-only assistant expectation changed from two blocks (empty text + `tool_use`) to one `tool_use` block, plus an assertion that no text block is empty.
- Tests / commands:
  - RED: `go test ./convert -run 'TestP0_(OpenAI|Responses)ToAnthropic' -v` → FAIL: Chat path emitted `{"type":"text"}` for `content:null` + `tool_calls`, Responses path did the same, and an empty user message was kept.
  - GREEN: `go test ./convert -run 'TestP0_|TestConvert_AssistantToolCalls|TestConvert_NoTools' -v` → PASS.
  - `go test ./...` → PASS (convert, engine, rewriter, tests/e2e).
  - `CGO_ENABLED=1 go test -race -count=1 ./convert ./engine ./rewriter` → PASS.
  - `go vet ./...` → clean.
- Review findings / fixes: self-review confirmed the existing `"..."` user fallback still repairs an all-empty request, reasoning-only assistant turns keep their `thinking` block, and tool-only assistant turns emit only `tool_use`. The editor auto-formatted a pre-existing unformatted struct in `convert_test.go`; that hunk was reverted so the diff stays scoped to P0-1. Pairwise registry, model-map behavior, and same-protocol passthrough are untouched.
- Deferred: the response-side `convertOpenAIMessageToContent` still substitutes a single-space text block when an assistant response has neither text nor tool calls (deliberate existing workaround, outside P0-1); empty `tool_result` content is not normalized (P0-3 scope).

---

### Task 2: Omit `tool_choice` when no tools are provided

**Files:**
- Modify: `convert/openai_to_anthropic.go:267-271`
- Modify: `convert/responses.go:339-340`
- Test: `convert/anthropic_openai_p0_test.go`
- Update: `convert/convert_test.go:640-656`
- Update: `convert/responses_test.go:248-266`

**Interfaces:**
- Consumes: `AnthropicRequest.ToolChoice`, `AnthropicTool`.
- Produces: no behavioral signature change; `ToolChoice` remains `nil` when
  `len(req.Tools) == 0`.

- [ ] **Step 1: Write the failing test**

Append to `convert/anthropic_openai_p0_test.go`:

```go
func TestP0_OpenAIToAnthropicNoToolsNoToolChoice(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte(`"tool_choice"`)) {
		t.Fatalf("tool_choice emitted without tools: %s", b)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice != nil {
		t.Fatalf("tool_choice = %+v, want nil", a.ToolChoice)
	}
}

func TestP0_ResponsesToAnthropicNoToolsNoToolChoice(t *testing.T) {
	body := `{"model":"gpt-4o","input":"hello"}`
	b, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 8192})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte(`"tool_choice"`)) {
		t.Fatalf("tool_choice emitted without tools from Responses: %s", b)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice != nil {
		t.Fatalf("tool_choice = %+v, want nil", a.ToolChoice)
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd llm-api-converter && go test ./convert -run 'TestP0_(OpenAIToAnthropicNoToolsNoToolChoice|ResponsesToAnthropicNoToolsNoToolChoice)' -v`

Expected: FAIL because the output contains `"tool_choice":{"type":"none"}`.

- [ ] **Step 3: Remove the orphan `else` branch**

In `convert/openai_to_anthropic.go`, delete the `else` that sets
`AnthropicToolChoice{Type: "none"}`. Keep `ToolChoice` at its zero value when
there are no tools.

In `convert/responses.go`, remove the corresponding `else` at `:339-340`.
`ToolChoice` remains nil when `req.Tools` is empty.

- [ ] **Step 4: Update the old assertions**

`convert/convert_test.go:653` currently expects `ToolChoice` to be non-nil
with type `none`. Rename `TestConvert_NoToolsSetsToolChoiceNone` to
`TestConvert_NoToolsNoToolChoice` and assert `a.ToolChoice == nil`.

`convert/responses_test.go:248-266` has the same expectation for the Responses
path. Rename `TestConvertResponsesToAnthropic_NoToolsSetsToolChoiceNone` to
`TestConvertResponsesToAnthropic_NoToolsNoToolChoice` and assert
`anth.ToolChoice == nil`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP0_OpenAIToAnthropicNoToolsNoToolChoice|TestConvert_NoTools' -v`

Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `cd llm-api-converter && go test ./...`

Expected: PASS.

- [ ] **Step 7: Fill the implementation record and commit**

```bash
git add convert/openai_to_anthropic.go convert/responses.go convert/anthropic_openai_p0_test.go convert/convert_test.go docs/plans/2026-11-06-anthropic-openai-chat-p0.md
git commit -m "fix(convert): omit tool choice when no tools"
```

**Implementation record**

- Status: Done
- Date: 2026-11-06
- Repository / commit: `llm-api-converter`, branch `feat/anthropic-openai-p0`, commit `fix(convert): omit tool choice when no tools` (2026-11-06), which contains the fix, the tests, and this record. A commit cannot embed its own hash, so the resolved SHA is recorded in the Task 2 report (`.superpowers/sdd/2026-11-06-anthropic-openai-chat-p0/task-2-report.md`) and can be obtained with `git log -1 --format=%H -- convert/openai_to_anthropic.go`.
- Files changed:
  - `convert/openai_to_anthropic.go` — removed the no-tools `else` branch in `convertOpenAIRequestToAnthropic`; `ToolChoice` stays nil when `len(req.Tools) == 0`, so `omitempty` removes the field.
  - `convert/responses.go` — removed the equivalent no-tools `else` in `ConvertResponsesToAnthropic`.
  - `convert/anthropic_openai_p0_test.go` — added `TestP0_OpenAIToAnthropicNoToolsNoToolChoice` and `TestP0_ResponsesToAnthropicNoToolsNoToolChoice`; both assert the serialized body has no `tool_choice` key and that the decoded `ToolChoice` is nil.
  - `convert/convert_test.go` — renamed `TestConvert_NoToolsSetsToolChoiceNone` to `TestConvert_NoToolsNoToolChoice` and now asserts `a.ToolChoice == nil`.
  - `convert/responses_test.go` — renamed `TestConvertResponsesToAnthropic_NoToolsSetsToolChoiceNone` to `TestConvertResponsesToAnthropic_NoToolsNoToolChoice` and now asserts `anth.ToolChoice == nil`.
- Tests / commands:
  - RED: `go test ./convert -run 'TestP0_(OpenAIToAnthropicNoToolsNoToolChoice|ResponsesToAnthropicNoToolsNoToolChoice)|TestConvert_NoTools|TestConvertResponsesToAnthropic_NoTools' -v` → all four FAIL; the Chat body contained `"tool_choice":{"type":"none"}` and the Responses body did too, while the updated legacy tests saw `&{Type:none Name:}`.
  - GREEN: same command after the fix → all four PASS.
  - `go test ./...` → PASS (`convert`, `engine`, `rewriter`, `tests/e2e`).
  - `go vet ./...` → clean.
- Review findings / fixes: self-review confirmed the fix is limited to the two no-tools branches; explicit `tool_choice` mapping when tools are present is untouched (`TestConvert_ToolChoice{Auto,None,Required,SpecificFunction}` still pass, and `TestConvertResponsesToAnthropic_WithTools` still expects nil for an omitted choice). The pairwise registry, model map, and same-protocol passthrough are unaffected, and no dependency was added. The editor auto-formatted two pre-existing unformatted spots in `convert_test.go` and `responses_test.go`; both hunks were reverted so the diff stays scoped to this task.
- Deferred: `convert/anthropic_to_gemini.go` still emits `tool_choice: none` on the Gemini→Anthropic no-tools path (different conversion direction, outside P0-2). Updating the review-document status table and ticking the plan checkboxes belongs to Task 5.

---

### Task 3: Normalize tool input and arguments

**Files:**
- Create: `convert/tool_input.go`
- Modify: `convert/openai_to_anthropic.go:456-512` (request + response conversion)
- Modify: `convert/anthropic_to_openai.go:452-485`, `:560-572`
- Test: `convert/anthropic_openai_p0_test.go`

**Interfaces:**
- Consumes: `OpenAIFunctionCall.Arguments`, `AnthropicContent.Input`.
- Produces:
  - `func anthropicToolInput(arguments string) any`
  - `func openAIToolArguments(input any) string`
  - Both guarantee a JSON object result (`map[string]any` or object JSON
    string), never nil/empty/`"null"`.

- [ ] **Step 1: Write the failing tests**

Append to `convert/anthropic_openai_p0_test.go`:

```go
func TestP0_OpenAIToAnthropicEmptyArgumentsBecomeInputObject(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"f","arguments":""}}
		]},
		{"role":"tool","tool_call_id":"call_1","content":"ok"}
	]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(`"input":{}`)) {
		t.Fatalf("tool_use.input missing: %s", b)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	var input any
	for _, c := range a.Messages[1].Content {
		if c.Type == "tool_use" {
			input = c.Input
		}
	}
	if _, ok := input.(map[string]any); !ok {
		t.Fatalf("input type = %T, want object", input)
	}
}

func TestP0_AnthropicToOpenAINullInputBecomesArgumentsObject(t *testing.T) {
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"claude",
		"content":[{"type":"tool_use","id":"toolu_1","name":"f","input":null}],
		"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}}`
	b, err := convertAnthropicResponseToOpenAI([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	got := o.Choices[0].Message.ToolCalls[0].Function.Arguments
	if got != "{}" {
		t.Fatalf("arguments = %q, want {}", got)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd llm-api-converter && go test ./convert -run 'TestP0_(OpenAIToAnthropicEmptyArguments|AnthropicToOpenAINullInput)' -v`

Expected: FAIL. The first output has no `input`; the second has
`"arguments":"null"`.

- [ ] **Step 3: Implement the shared helpers**

Create `convert/tool_input.go`:

```go
package convert

import (
	"encoding/json"
	"strings"
)

// anthropicToolInput turns OpenAI tool arguments into an Anthropic tool_use
// input object. Empty, null, invalid, or non-object JSON is preserved as an
// object so the tool_use block stays schema-valid.
func anthropicToolInput(arguments string) any {
	s := strings.TrimSpace(arguments)
	if s == "" {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return map[string]any{"input": arguments}
	}
	if v == nil {
		return map[string]any{}
	}
	if _, ok := v.(map[string]any); ok {
		return v
	}
	return map[string]any{"input": v}
}

// openAIToolArguments turns an Anthropic tool_use input into the JSON string
// OpenAI expects. Nil or null becomes {}; non-object input is wrapped so the
// result is always a JSON object.
func openAIToolArguments(input any) string {
	if input == nil {
		return "{}"
	}
	if _, ok := input.(map[string]any); !ok {
		input = map[string]any{"input": input}
	}
	b, err := json.Marshal(input)
	if err != nil || len(b) == 0 || string(b) == "null" {
		return "{}"
	}
	return string(b)
}
```

Use them:

- `convert/openai_to_anthropic.go`, `convertAssistantContent`:
  replace the inline `json.Unmarshal` with
  `Input: anthropicToolInput(tc.Function.Arguments)`.
- `convert/openai_to_anthropic.go`, `convertOpenAIMessageToContent`:
  replace the inline unmarshal with
  `Input: anthropicToolInput(tc.Function.Arguments)`.
- `convert/anthropic_to_openai.go`, `convertAnthropicAssistantMessage`:
  replace the inline marshal with
  `Arguments: openAIToolArguments(block.Input)`.
- `convert/anthropic_to_openai.go`, `convertAnthropicResponseToOpenAI`:
  replace the inline marshal with
  `Arguments: openAIToolArguments(block.Input)`.

Do not change valid object/string behavior.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd llm-api-converter && go test ./convert -run 'TestP0_(OpenAIToAnthropicEmptyArguments|AnthropicToOpenAINullInput)' -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `cd llm-api-converter && go test ./...`

Expected: PASS.

- [ ] **Step 6: Fill the implementation record and commit**

```bash
git add convert/tool_input.go convert/openai_to_anthropic.go convert/anthropic_to_openai.go convert/anthropic_openai_p0_test.go docs/plans/2026-11-06-anthropic-openai-chat-p0.md
git commit -m "fix(convert): normalize tool use input objects"
```

**Implementation record**

- Status: Done
- Date: 2026-11-06
- Repository / commit: `llm-api-converter`, branch `feat/anthropic-openai-p0`, commit `fix(convert): normalize tool use input objects` (2026-11-06), which contains the helpers, the four call-site changes, the regression tests, and this record. A commit cannot embed its own hash, so the resolved SHA is recorded in `.superpowers/sdd/2026-11-06-anthropic-openai-chat-p0/task-3-report.md` and can be obtained with `git log -1 --format=%H -- convert/tool_input.go`.
- Files changed:
  - `convert/tool_input.go` — new; `anthropicToolInput` turns OpenAI arguments into an Anthropic input object and `openAIToolArguments` turns an Anthropic input into an OpenAI arguments string. Both guarantee an object: empty/null/invalid → `{}`, non-object JSON wrapped as `{"input": ...}`, valid objects preserved.
  - `convert/openai_to_anthropic.go` — `convertAssistantContent` (request path) and `convertOpenAIMessageToContent` (response path) use `anthropicToolInput` instead of an inline best-effort unmarshal, so empty arguments no longer drop the `input` field.
  - `convert/anthropic_to_openai.go` — `convertAnthropicAssistantMessage` (request path) and `convertAnthropicResponseToOpenAI` (response path) use `openAIToolArguments`, so nil/null input serializes as `{}` instead of `""` or `"null"`.
  - `convert/anthropic_openai_p0_test.go` — added `TestP0_OpenAIToAnthropicEmptyArgumentsBecomeInputObject` and `TestP0_AnthropicToOpenAINullInputBecomesArgumentsObject`.
- Tests / commands:
  - RED: `go test ./convert -run 'TestP0_(OpenAIToAnthropicEmptyArguments|AnthropicToOpenAINullInput)' -v` → FAIL. The Chat request emitted `{"type":"tool_use","id":"call_1","name":"f"}` with no `input`; the Anthropic response produced `"arguments":"null"`.
  - GREEN: same command after the fix → both PASS.
  - `go test ./...` → PASS (`convert`, `engine`, `rewriter`, `tests/e2e`).
  - `go vet ./...` → clean.
- Review findings / fixes: self-review confirmed all four brief-named call sites use the shared helpers; valid object behavior is unchanged (existing tool-use conversion tests still pass), and no dependency, registry, model-map, or same-protocol passthrough code was touched. Removing the old `args := ""` fallback is intentional: a missing input now yields `{}` rather than an empty arguments string. Go's formatter reports several pre-existing unformatted files (`convert_test.go`, `stream_test.go`, etc.); the four files touched here are gofmt-clean and no unrelated reformatting was included.
- Deferred: the other conversion pairs still have their own tool-input handling (`convert/anthropic_to_gemini.go`, and the Anthropic→Responses helper in `convert/responses.go`) and are outside P0-3. `openAIToolArguments` treats only `map[string]any` as an object; a typed map supplied programmatically would be wrapped as `{"input": ...}`, which is unreachable through JSON decoding and left as is to match the brief.

---

### Task 4: Keep interleaved streamed tool calls separate

**Files:**
- Modify: `convert/types.go` (`streamToolState`)
- Modify: `convert/stream.go` (`StreamConverter`, `handleToolCallDelta`, `inputJSONDelta`, `HandleStreamEnd`)
- Test: `convert/anthropic_openai_p0_test.go` (or `convert/stream_test.go`)

**Interfaces:**
- Consumes: `OpenAIDeltaToolCall`, `streamToolState`, `StreamConverter`.
- Produces:
  - `streamToolState` gains `BlockIndex int`
  - `inputJSONDelta(blockIndex int, partialJSON string) [][]byte`
  - `toolArgsComplete(args string) bool`
  - `StreamConverter` gains `openToolIndex int` and
    `heldTools map[int]*streamToolState`

- [ ] **Step 1: Write the failing test**

Append to `convert/anthropic_openai_p0_test.go`:

```go
func TestP0_StreamInterleavedToolCallsKeepArguments(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	_ = sc.HandleStreamStart()

	chunks := []string{
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"g","arguments":"{\"b\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":null}]}`,
	}
	var out []byte
	for _, chunk := range chunks {
		b, err := sc.HandleChunk([]byte(chunk))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, b...)
	}
	out = append(out, sc.HandleStreamEnd()...)

	args := map[string]string{} // tool ID -> concatenated partial_json
	var currentID string
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}
		var evt struct {
			Type         string `json:"type"`
			ContentBlock struct {
				ID string `json:"id"`
			} `json:"content_block"`
			Delta struct {
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if json.Unmarshal([]byte(line), &evt) != nil {
			continue
		}
		switch evt.Type {
		case "content_block_start":
			currentID = evt.ContentBlock.ID
		case "content_block_delta":
			args[currentID] += evt.Delta.PartialJSON
		}
	}
	if args["call_0"] != `{"a":1}` {
		t.Fatalf("call_0 arguments = %q", args["call_0"])
	}
	if args["call_1"] != `{"b":2}` {
		t.Fatalf("call_1 arguments = %q", args["call_1"])
	}
}
```

Note: the test accumulates the chunk and end-phase bytes, then parses the
resulting SSE. A held call is released in the end phase; a sequential call
streams in its chunk output. Both must reach the same per-call arguments.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd llm-api-converter && go test ./convert -run TestP0_StreamInterleavedToolCallsKeepArguments -v`

Expected: FAIL. `call_1` receives `{"b":1}2}` or `call_0` receives the wrong
fragment.

- [ ] **Step 3: Add state fields and helpers**

In `convert/types.go`, add `BlockIndex int` to `streamToolState`.

In `convert/stream.go`:

```go
type StreamConverter struct {
	// ... existing fields ...
	openToolIndex int
	heldTools     map[int]*streamToolState
}
```

Initialize `openToolIndex: -1` and `heldTools: make(map[int]*streamToolState)`
in `NewStreamConverter`.

Add:

```go
// toolArgsComplete reports whether args is a complete JSON object.
func toolArgsComplete(args string) bool {
	var v map[string]any
	return json.Unmarshal([]byte(args), &v) == nil
}
```

In `inputJSONDelta`, take the block index explicitly and use it instead of
`sc.curBlockIndex`.

- [ ] **Step 4: Change `handleToolCallDelta`**

Rules:

1. For an already-seen call:
   - if it is the currently open tool block, append arguments and emit a
     delta with its own `BlockIndex`;
   - otherwise append the fragment to its state and emit nothing. The call
     stays held and is released at stream end.
2. For a new call:
   - if a tool block is open and its accumulated arguments are not a complete
     JSON object, store the new call in both `toolCallByIndex` and
     `heldTools`, and emit nothing;
   - otherwise close the current block, start the new tool block, set
     `BlockIndex = sc.curBlockIndex`, set `openToolIndex = idx`, and emit
     the initial arguments if present.
3. When closing a tool block, set `openToolIndex = -1`. This must also happen
   inside `ensureBlock` when a text/thinking block replaces an open tool block,
   and in `HandleStreamEnd` when the final block is closed.

- [ ] **Step 5: Release held calls in `HandleStreamEnd`**

After closing the currently open block(s) and before `message_delta`:

- collect `heldTools` keys in ascending numeric order;
- for each held call, create a fresh content block at the next block index,
  emit `content_block_start` with `id`, `name`, and `"input":{}`;
- emit `input_json_delta` with the accumulated arguments, or `{}` when empty;
- emit `content_block_stop`;
- clear `heldTools`.

Do not emit a second block for a call that already had its block closed;
only calls in `heldTools` are released.

- [ ] **Step 6: Run the new test and the existing stream tests**

Run: `cd llm-api-converter && go test ./convert -run 'TestP0_StreamInterleavedToolCallsKeepArguments|TestStream_' -v`

Expected: PASS. Existing sequential tool-call tests must keep their immediate
`content_block_start` behavior.

- [ ] **Step 7: Run the full suite**

Run: `cd llm-api-converter && go test ./...`

Expected: PASS.

- [ ] **Step 8: Fill the implementation record and commit**

```bash
git add convert/types.go convert/stream.go convert/anthropic_openai_p0_test.go docs/plans/2026-11-06-anthropic-openai-chat-p0.md
git commit -m "fix(convert): keep interleaved stream tool calls separate"
```

**Implementation record**

- Status: Done
- Date: 2026-11-06
- Repository / commit: `llm-api-converter`, branch `feat/anthropic-openai-p0`, commit `fix(convert): keep interleaved stream tool calls separate` (2026-11-06), which contains the converter changes, the regression tests, and this record. A commit cannot embed its own hash, so the resolved SHA is recorded in `.superpowers/sdd/2026-11-06-anthropic-openai-chat-p0/task-4-report.md` and can be obtained with `git log -1 --format=%H -- convert/stream.go`.
- Files changed:
  - `convert/types.go` — `streamToolState` gains `BlockIndex`, so each tool call remembers the Anthropic content block index it owns.
  - `convert/stream.go` — `StreamConverter` gains `openToolIndex` (initialized to -1) and `heldTools`; `toolArgsComplete` detects a complete JSON object; `handleToolCallDelta` emits fragments only for the currently open call and holds a new call in both `toolCallByIndex` and `heldTools` while the open call's arguments are incomplete; `inputJSONDelta` takes an explicit block index; `ensureBlock` and `HandleStreamEnd` reset `openToolIndex` whenever a tool block closes; `releaseHeldTools` emits held calls at stream end in ascending index order (`content_block_start` with `"input":{}`, then `input_json_delta` with the accumulated arguments or `{}`, then `content_block_stop`) before `message_delta`.
  - `convert/anthropic_openai_p0_test.go` — added `TestP0_StreamInterleavedToolCallsKeepArguments` (the brief's test, with `\n\n` framing inserted between phase payloads because `HandleChunk` returns no trailing delimiter) and `TestP0_StreamHeldToolCallsReleaseInIndexOrder` covering ascending release order, `"input":{}`, and the empty-arguments `{}` delta.
- Tests / commands:
  - RED: `go test ./convert -run TestP0_StreamInterleavedToolCallsKeepArguments -v` → FAIL, `call_0 arguments = "{\"a\":"`; the later `1}` fragment was written to the block index opened by `call_1`.
  - GREEN: same command after the fix → PASS.
  - `go test ./convert -run 'TestP0_StreamInterleavedToolCallsKeepArguments|TestStream_' -v` → PASS, including the existing sequential tool-call tests that require an immediate `content_block_start`.
  - `go test ./...` → PASS (`convert`, `engine`, `rewriter`, `tests/e2e`).
  - `go test -race ./convert` → PASS; `go vet ./...` → clean.
- Review findings / fixes: self-review found the brief's test as written loses the last event of each phase because `HandleChunk` documents that it returns no trailing `\n\n`; the test harness now inserts the separator between phase payloads (production code already frames phases), and the assertions are unchanged. Sequential calls still open and close immediately once the open call's arguments form a complete object, as required by the existing tests. Verified that no third-party dependency, converter registry, model-map, or same-protocol passthrough code changed. `releaseHeldTools` clears `heldTools`, so a second `HandleStreamEnd` cannot emit duplicate blocks (it is also guarded by `finalized`).
- Deferred: providers that omit `tool_calls[].index` cannot disambiguate a continuation fragment that arrives after another index-less call was held; it attaches to `lastToolIndex`. This is inherent to the index-less stream format and outside P0-4. Released held calls keep their accumulated arguments even if they are not a complete JSON object, matching the brief's "accumulated arguments" rule; clients see the same partial JSON they would have received sequentially.

---

### Task 5: P0 verification and documentation

**Files:**
- Modify: `llm-api-converter/docs/reviews/2026-11-06-anthropic-openai-chat-review.md`
- Modify: `llm-api-converter/docs/plans/2026-11-06-anthropic-openai-chat-p0.md`
- Modify: `/root/code/go-gost/.memory/notes/llm-api-converter-magpie-eval.md`
  (tracked in the workspace-root repo, not in `llm-api-converter`)

**Interfaces:**
- Consumes: the four completed P0 tasks.
- Produces: a verified P0 status, filled implementation records, and a durable
  memory note.

- [ ] **Step 1: Run the full test suite**

Run: `cd llm-api-converter && go test ./...`

Expected: PASS, including the e2e package.

- [ ] **Step 2: Run the race and vet gates**

Run:

```sh
cd llm-api-converter
CGO_ENABLED=1 go test -race ./convert ./rewriter
go vet ./...
```

Expected: PASS, no data race or vet diagnostics.

- [ ] **Step 3: Update the review document**

Update the existing “P0 Status” table in the review document:

```markdown
| ID | Status | Commit |
|---|---|---|
| P0-1 | fixed | <sha> |
| P0-2 | fixed | <sha> |
| P0-3 | fixed | <sha> |
| P0-4 | fixed | <sha> |
```

If any P0 item was deferred, mark it `deferred` and record why.

- [ ] **Step 4: Fill the remaining implementation records**

Ensure Tasks 1–4 have date, commit SHA, actual files, commands/results,
review findings, and deferred items. Do not leave `Pending`.

- [ ] **Step 5: Update project memory**

The 2026-11-06 review observation already exists in
`.memory/notes/llm-api-converter-magpie-eval.md`. After the fixes, update its
status from `open` to `fixed` and add the four commit SHAs. Do not append a
duplicate observation.

No new memory note is needed unless the fixes establish a new convention.

- [ ] **Step 6: Verify the working tree and commit both repositories**

`llm-api-converter` and the workspace root are separate Git repositories.
Commit the docs in `llm-api-converter`, then the memory note in
`/root/code/go-gost`. Do not commit unrelated root memory changes.

Run:

```sh
cd /root/code/go-gost/llm-api-converter
git status --short
```

Expected: only the review doc and plan are modified. Then:

```bash
git add docs/reviews/2026-11-06-anthropic-openai-chat-review.md docs/plans/2026-11-06-anthropic-openai-chat-p0.md
git commit -m "docs(convert): record P0 review status"
```

Then commit the memory note in the workspace-root repo:

```sh
cd /root/code/go-gost
git add .memory/notes/llm-api-converter-magpie-eval.md
git commit -m "docs(memory): record llm-api-converter P0 review"
```

Expected: `git status --short` in each repo shows no unrelated changes from
this task. Record both commit SHAs in the implementation record.

**Implementation record**

- Status: Pending
- Date:
- Repository / commit:
- Files changed:
- Tests / commands:
- Review findings / fixes:
- Deferred:

---

## Self-Review

- **Spec coverage:** P0-1 → Task 1; P0-2 → Task 2; P0-3 → Task 3; P0-4 → Task 4; verification/memory → Task 5. P1/P2 are explicitly out of scope and remain in the review document.
- **Step scan:** Each task has a failing test, an explicit implementation step, a pass check, and a commit. No “handle edge cases” placeholders.
- **Type consistency:** `appendTextBlock`, `anthropicToolInput`, `openAIToolArguments`, `toolArgsComplete`, and `streamToolState.BlockIndex` are named consistently across tasks and interfaces.
- **Review Focus:** The five input classes are pinned in Tasks 1–4 and the fallback empty-message case in Task 1.
- **Proportion:** The plan is task-level with exact signatures and test assertions; no full implementation bodies are transcribed beyond the two small helpers that determine the protocol contract.

## Execution Handoff

Plan complete and saved to `docs/plans/2026-11-06-anthropic-openai-chat-p0.md`.
Please review it. Which execution approach would you prefer?

- **Subagent-driven** — a fresh subagent implements each task and a fresh reviewer checks it before the next; most thorough, higher cost.
- **Native** — I implement every task in this session; fastest, with one final review.

For this plan I recommend **subagent-driven** if the P0 fixes will ship immediately, because Tasks 1–3 touch shared conversion code and benefit from an independent gate; otherwise native is fine for the small, well-bounded changes.
