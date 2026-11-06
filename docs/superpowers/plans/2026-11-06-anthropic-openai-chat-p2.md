# Anthropic ↔ OpenAI Chat P2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the actionable P2 findings from the Anthropic ↔ OpenAI Chat review, plus the Responses truncated-tool and Gemini stream-error gaps, without changing the pairwise converter architecture.

**Architecture:** In-place fixes in `llm-api-converter/convert`. Shared normalization helpers handle stop reasons, tool IDs, reasoning aliases, and stream block shapes; request-field mappings stay in the two request converters; stream-terminal guards live in each stateful stream converter.

**Tech Stack:** Go, `encoding/json`, existing `convert`, `engine`, and `rewriter` packages. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-06-anthropic-openai-chat-p2-design.md`

## Global Constraints

- Do not add third-party dependencies.
- Keep the pairwise converter registry, model-map, and same-protocol passthrough behavior intact.
- Code comments are English.
- Every task must leave `go test -count=1 ./...` green.
- `CGO_ENABLED=1 go test -race -count=1 ./convert` and `go vet ./...` must pass before the branch is considered done.
- Commit only; never push.
- Do not modify `core/` or other go-gost modules.
- Each task fills its **Implementation record** in this plan with date, repository/commit, files, commands/results, review findings, and deferred items, and commits it with the task.
- Follow TDD: write the failing test, capture RED, implement, capture GREEN.
- All commands run from `/root/code/go-gost/llm-api-converter`.

## Review Focus

1. **Truncated Responses tool stream** (`ResponsesStreamConverter`, `AnthropicStreamConverter`): a clean EOF after partial JSON must end in `response.failed`, never `response.completed` or a completed `function_call` item. Task 7.
2. **Tool stop reason with content filtering** (OpenAI → Anthropic): a response containing `tool_use` blocks must keep `stop_reason: tool_use` even when the upstream says `content_filter`, or the client will not run the tool. Task 2.
3. **Invalid tool IDs across the pair** (OpenAI → Anthropic request/response/stream): an ID containing `.`, `/`, or `:` must normalize identically in the assistant call and the following `tool_result`, or the strict Anthropic endpoint rejects the history. Task 5.
4. **Gemini upstream error followed by normal close** (all four stateful Gemini converters): an error chunk must not be followed by `message_stop`, `[DONE]`, or a finish chunk. Task 8.
5. **Optional fields causing new 400s** (request conversion): `top_k` must not reach strict OpenAI models, and `disable_parallel_tool_use` must not be attached to an Anthropic `none` tool choice. Task 6.

---

### Task 1: ReasoningCache concurrency safety

**Files:**
- Modify: `convert/reasoning_cache.go` (`memoryStore.Get`, `memoryStore.Set`)
- Test: `convert/anthropic_openai_p2_test.go` (create)

**Interfaces:**
- Produces: `memoryStore.Get(key string) *cacheEntry` returns a shallow copy.
- Produces: `memoryStore.Set` replaces the map entry pointer instead of mutating the existing entry.
- No exported API changes.

- [x] **Step 1: Write the failing race test**

```go
package convert

import (
	"fmt"
	"sync"
	"testing"
)

func TestP2_ReasoningCacheConcurrentGetPut(t *testing.T) {
	cache := NewReasoningCache(16)
	defer cache.Close()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 2000; j++ {
				cache.Put([]string{"tool_1"}, fmt.Sprintf("reasoning-%d-%d", i, j))
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < 2000; j++ {
				cache.Get([]string{"tool_1"})
			}
		}()
	}
	wg.Wait()
}
```

- [x] **Step 2: Run the race test and capture RED**

Run: `CGO_ENABLED=1 go test -race -count=1 -run TestP2_ReasoningCacheConcurrentGetPut ./convert`

Expected on the base revision: `WARNING: DATA RACE` between `memoryStore.Get` reading `entry.AddedAt`/`entry.Reasoning` and `memoryStore.Set` writing `existing.Reasoning`/`existing.AddedAt`.

- [x] **Step 3: Implement the copy-on-read and replace-on-write behavior**

In `convert/reasoning_cache.go`:

```go
func (ms *memoryStore) Get(key string) *cacheEntry {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	entry := ms.entries[key]
	if entry == nil {
		return nil
	}
	copied := *entry
	return &copied
}

func (ms *memoryStore) Set(key string, entry *cacheEntry) {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	if _, ok := ms.entries[key]; ok {
		ms.entries[key] = entry
		return
	}
	if len(ms.entries) >= ms.maxSize {
		ms.evictLocked()
	}
	ms.entries[key] = entry
	ms.order = append(ms.order, key)
}
```

Do not change `ReasoningCache.getFromStore`; it reads the copied entry safely now.

- [x] **Step 4: Run the race test and the package tests**

Run: `CGO_ENABLED=1 go test -race -count=1 -run TestP2_ReasoningCacheConcurrentGetPut ./convert`

Expected: PASS with no race report.

Run: `go test -count=1 ./...`

Expected: all packages PASS.

- [x] **Step 5: Commit**

```bash
git add convert/reasoning_cache.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): make reasoning cache concurrency-safe"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `1bca7bd`
- Files changed: `convert/reasoning_cache.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED `CGO_ENABLED=1 go test -race -count=1 -run TestP2_ReasoningCacheConcurrentGetPut ./convert` → `WARNING: DATA RACE` between `memoryStore.Set` and `ReasoningCache.getFromStore`; GREEN same command → PASS. `go test -count=1 ./...` → all packages PASS. A stale released `llm-api-converter` process was holding `:18080` and caused an e2e 404; killed it, reran green.
- Review findings/fixes: none.
- Deferred/limitations: none.

---

### Task 2: Stop-reason normalization, refusal content, empty choices

**Files:**
- Modify: `convert/types.go` (`OpenAIMessage.Refusal`, `OpenAIDelta.Refusal`)
- Modify: `convert/openai_to_anthropic.go` (`normalizeOpenAIStopReason`, `mapOpenAIFinishReason`, response conversion, message-to-content)
- Modify: `convert/convert.go` (`mapOpenAIStreamFinish`)
- Modify: `convert/stream.go` (refusal delta, stop inference)
- Modify: `convert/anthropic_to_openai.go` (`mapFinishReason`, `strContentFilter`)
- Modify: `convert/anthropic_openai_p1_test.go` (`TestP1_NonStreamToolStopInference` expectation)
- Test: `convert/anthropic_openai_p2_test.go`

**Interfaces:**
- Produces: `func normalizeOpenAIStopReason(reason string) string`
- Produces: `OpenAIMessage.Refusal string` with `json:"refusal,omitempty"`
- Produces: `OpenAIDelta.Refusal string` with `json:"refusal,omitempty"`
- Produces: `var strContentFilter = "content_filter"`
- Behavior: Anthropic responses with no choices serialize `"content":[]` and `stop_reason:"end_turn"`.

- [x] **Step 1: Write the failing tests**

Append to `convert/anthropic_openai_p2_test.go`:

```go
func TestP2_OpenAIStopReasonNormalization(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{
			name: "function_call",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"function_call","message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
			want: "tool_use",
		},
		{
			name: "content_filter",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","content":"filtered"}}]}`,
			want: "refusal",
		},
		{
			name: "unknown",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"weird","message":{"role":"assistant","content":"ok"}}]}`,
			want: "end_turn",
		},
		{
			name: "missing",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`,
			want: "end_turn",
		},
		{
			name: "content_filter_with_tools",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
			want: "tool_use",
		},
		{
			name: "length_with_tools",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
			want: "max_tokens",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := convertOpenAIResponseToAnthropic([]byte(tc.body), nil)
			if err != nil {
				t.Fatal(err)
			}
			var resp AnthropicResponse
			if err := json.Unmarshal(out, &resp); err != nil {
				t.Fatal(err)
			}
			if resp.StopReason == nil || *resp.StopReason != tc.want {
				t.Fatalf("stop_reason = %v, want %q", resp.StopReason, tc.want)
			}
		})
	}
}

func TestP2_StreamStopReasonNormalization(t *testing.T) {
	t.Run("content_filter", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}`))
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"stop_reason":"refusal"`) {
			t.Fatalf("stream stop = %s, want refusal", end)
		}
	})

	t.Run("unknown", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}`))
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"weird"}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"stop_reason":"end_turn"`) {
			t.Fatalf("stream stop = %s, want end_turn", end)
		}
	})

	t.Run("content_filter_with_tools", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"content_filter"}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"stop_reason":"tool_use"`) {
			t.Fatalf("stream stop = %s, want tool_use", end)
		}
	})
}

func TestP2_AnthropicStopReasonNormalization(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"refusal", `{"id":"msg_1","model":"claude","stop_reason":"refusal","content":[{"type":"text","text":"no"}]}`, "content_filter"},
		{"pause_turn", `{"id":"msg_1","model":"claude","stop_reason":"pause_turn","content":[{"type":"text","text":"wait"}]}`, "stop"},
		{"unknown", `{"id":"msg_1","model":"claude","stop_reason":"weird","content":[{"type":"text","text":"ok"}]}`, "stop"},
		{"missing", `{"id":"msg_1","model":"claude","content":[{"type":"text","text":"ok"}]}`, "stop"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := convertAnthropicResponseToOpenAI([]byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			var resp OpenAIChatResponse
			if err := json.Unmarshal(out, &resp); err != nil {
				t.Fatal(err)
			}
			if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != tc.want {
				t.Fatalf("finish_reason = %v, want %q", resp.Choices[0].FinishReason, tc.want)
			}
		})
	}
}

func TestP2_OpenAIRefusalText(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":null,"refusal":"I cannot help with that."}}]}`
	out, err := convertOpenAIResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var resp AnthropicResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" || resp.Content[0].Text != "I cannot help with that." {
		t.Fatalf("content = %+v, want refusal text", resp.Content)
	}
	if resp.StopReason == nil || *resp.StopReason != "refusal" {
		t.Fatalf("stop_reason = %v, want refusal", resp.StopReason)
	}
}

func TestP2_StreamRefusalDelta(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"refusal":"I cannot"},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	end := string(sc.HandleStreamEnd())
	all := string(out) + end
	if !strings.Contains(all, `"text_delta"`) || !strings.Contains(all, `"I cannot"`) {
		t.Fatalf("refusal delta missing: %s", all)
	}
	if !strings.Contains(end, `"stop_reason":"refusal"`) {
		t.Fatalf("stream stop = %s, want refusal", end)
	}
}

func TestP2_EmptyChoicesContentArray(t *testing.T) {
	out, err := convertOpenAIResponseToAnthropic([]byte(`{"id":"x","model":"gpt-4","choices":[]}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatal(err)
	}
	if string(raw["content"]) != "[]" {
		t.Fatalf("content = %s, want []", raw["content"])
	}
	var resp AnthropicResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.StopReason == nil || *resp.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %v, want end_turn", resp.StopReason)
	}
}
```

- [x] **Step 2: Run the tests and capture RED**

Run: `go test -count=1 -run 'TestP2_(OpenAIStopReasonNormalization|StreamStopReasonNormalization|AnthropicStopReasonNormalization|OpenAIRefusalText|StreamRefusalDelta|EmptyChoicesContentArray)' ./convert`

Expected: failures for `function_call`, `content_filter`, unknown/missing reasons, refusal text, stream refusal, and `content:null`.

- [x] **Step 3: Implement normalization and refusal handling**

In `convert/openai_to_anthropic.go`, add:

```go
func normalizeOpenAIStopReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}
```

Change `mapOpenAIFinishReason` to use it; nil maps to `end_turn`:

```go
func mapOpenAIFinishReason(reason *string) *string {
	if reason == nil {
		s := "end_turn"
		return &s
	}
	s := normalizeOpenAIStopReason(*reason)
	return &s
}
```

In `convertOpenAIResponseToAnthropic`, initialize `Content: []AnthropicContent{}` in the `AnthropicResponse` literal. Replace the tool-stop inference block with:

```go
if hasAnthropicToolUse(anth.Content) && (choice.FinishReason == nil || *choice.FinishReason != "length") {
	stopReason := "tool_use"
	anth.StopReason = &stopReason
}
if msg.Refusal != "" && len(msg.ToolCalls) == 0 &&
	(choice.FinishReason == nil || *choice.FinishReason == "stop") {
	stopReason := "refusal"
	anth.StopReason = &stopReason
}
```

After the `if len(resp.Choices) > 0 { ... }` block, add:

```go
if anth.StopReason == nil {
	stopReason := "end_turn"
	anth.StopReason = &stopReason
}
```

In `convertOpenAIMessageToContent`, after `text := extractTextContent(msg.Content)`:

```go
if text == "" && msg.Refusal != "" {
	text = msg.Refusal
}
```

In `convert/convert.go`, change `mapOpenAIStreamFinish` to:

```go
func mapOpenAIStreamFinish(reason string) string {
	return normalizeOpenAIStopReason(reason)
}
```

In `convert/types.go`, add `Refusal string` to `OpenAIMessage` and `OpenAIDelta`. Extend the `OpenAIDelta.UnmarshalJSON` aux with `Refusal string` and assign `d.Refusal = aux.Refusal`.

In `convert/stream.go`:
- add `sawRefusal bool` to `StreamConverter`
- in `HandleChunk`, after the content block:

```go
if delta.Refusal != "" {
	sc.sawRefusal = true
	sc.accumulatedText += delta.Refusal
	sc.stopThinkingBlock()
	events = append(events, sc.ensureBlock("text", "", "")...)
	events = append(events, sc.textDelta(delta.Refusal)...)
}
```

- replace the stop-reason switch in `HandleStreamEnd` with:

```go
stopReason := "end_turn"
switch {
case len(sc.toolCallByIndex) > 0 && sc.finishReason != "length":
	stopReason = "tool_use"
case sc.sawRefusal && (sc.finishReason == "" || sc.finishReason == "stop"):
	stopReason = "refusal"
case sc.finishReason != "":
	stopReason = mapOpenAIStreamFinish(sc.finishReason)
}
```

In `convert/anthropic_to_openai.go`, add `strContentFilter = "content_filter"` to the var block and change `mapFinishReason`:

```go
func mapFinishReason(reason *string) *string {
	if reason == nil {
		return &strStop
	}
	switch *reason {
	case "end_turn", "stop_sequence", "pause_turn":
		return &strStop
	case "max_tokens":
		return &strLength
	case "tool_use":
		return &strToolCalls
	case "refusal", "content_filter":
		return &strContentFilter
	default:
		return &strStop
	}
}
```

Update `convert/anthropic_openai_p1_test.go` in `TestP1_NonStreamToolStopInference`: the `content_filter` + tool calls case now expects `tool_use`, not `end_turn`.

Update `convert/convert_test.go` in `TestConvert_ContentFilterFinishReason`: Anthropic `content_filter` now maps to OpenAI `content_filter`, matching the stream path, not `stop`.

- [x] **Step 4: Run the focused and full tests**

Run: `go test -count=1 -run 'TestP2_|TestP1_NonStreamToolStopInference' ./convert`

Expected: PASS.

Run: `go test -count=1 ./...`

Expected: all packages PASS. If the contract validator rejects `refusal`, confirm it is in the allowed stop-reason list; it is.

- [x] **Step 5: Commit**

```bash
git add convert/types.go convert/openai_to_anthropic.go convert/convert.go convert/stream.go convert/anthropic_to_openai.go convert/anthropic_openai_p1_test.go convert/convert_test.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): normalize stop reasons and refusal content"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `ae8391a`
- Files changed: `convert/types.go`, `convert/openai_to_anthropic.go`, `convert/convert.go`, `convert/stream.go`, `convert/anthropic_to_openai.go`, `convert/anthropic_openai_p1_test.go`, `convert/convert_test.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED focused run failed on `function_call`, `content_filter`, unknown/missing reasons, refusal text, stream refusal, and `content:null`; GREEN `go test -count=1 -run 'TestP2_|TestP1_NonStreamToolStopInference|TestConvert_ContentFilterFinishReason' ./convert` → PASS; `go test -count=1 ./...` → all packages PASS; `go vet ./...` and `git diff --check` clean.
- Review findings/fixes: existing `TestP1_NonStreamToolStopInference` and `TestConvert_ContentFilterFinishReason` expectations updated to the new consistent mapping.
- Deferred/limitations: annotations remain deferred; empty choices intentionally serialize `[]` rather than inventing text.

---

### Task 3: Accept `reasoning` and `reasoning_details` aliases

**Files:**
- Modify: `convert/types.go` (`OpenAIReasoningDetail`, `OpenAIMessage.UnmarshalJSON`, `OpenAIDelta.UnmarshalJSON`)
- Test: `convert/anthropic_openai_p2_test.go`

**Interfaces:**
- Produces: `type OpenAIReasoningDetail struct { Type, Text, Summary string }`
- Produces: `func normalizeReasoningAliases(reasoningContent, reasoning string, details []OpenAIReasoningDetail) string`
- Behavior: the normalized value is stored in `ReasoningContent`; alias fields are not added to the public structs.

- [x] **Step 1: Write the failing tests**

Append:

```go
func TestP2_ReasoningAliasRequest(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"ok","reasoning":"chain"},
		{"role":"user","content":"go on"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	if err := json.Unmarshal(out, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) == 0 || len(req.Messages[0].Content) == 0 || req.Messages[0].Content[0].Type != "thinking" {
		t.Fatalf("messages = %+v, want a thinking block", req.Messages)
	}
	if req.Messages[0].Content[0].Thinking != "chain" {
		t.Fatalf("thinking = %q, want chain", req.Messages[0].Content[0].Thinking)
	}
}

func TestP2_ReasoningDetailsRequest(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"ok","reasoning_details":[
			{"type":"reasoning.text","text":"a"},
			{"type":"reasoning.summary","summary":"b"}]},
		{"role":"user","content":"go on"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	_ = json.Unmarshal(out, &req)
	if len(req.Messages) == 0 || len(req.Messages[0].Content) == 0 {
		t.Fatalf("messages = %+v", req.Messages)
	}
	if got := req.Messages[0].Content[0].Thinking; got != "a\nb" {
		t.Fatalf("thinking = %q, want a\\nb", got)
	}
}

func TestP2_ReasoningContentWinsOverAlias(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"ok","reasoning_content":"primary","reasoning":"alias"},
		{"role":"user","content":"go on"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), `"alias"`) {
		t.Fatalf("alias leaked: %s", out)
	}
	if !strings.Contains(string(out), `"primary"`) {
		t.Fatalf("primary reasoning lost: %s", out)
	}
}

func TestP2_ReasoningAliasStream(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning":"chain"},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"thinking_delta"`) || !strings.Contains(string(out), `"chain"`) {
		t.Fatalf("reasoning alias missing from stream: %s", out)
	}
}

func TestP2_ReasoningAliasResponse(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok","reasoning":"chain"}}]}`
	out, err := convertOpenAIResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if len(resp.Content) == 0 || resp.Content[0].Type != "thinking" || resp.Content[0].Thinking != "chain" {
		t.Fatalf("content = %+v, want thinking chain", resp.Content)
	}
}
```

- [x] **Step 2: Run the tests and capture RED**

Run: `go test -count=1 -run 'TestP2_Reasoning' ./convert`

Expected: alias tests fail because `reasoning` / `reasoning_details` are unknown fields and `ReasoningContent` stays empty.

- [x] **Step 3: Implement alias normalization**

In `convert/types.go`, add:

```go
type OpenAIReasoningDetail struct {
	Type    string `json:"type,omitempty"`
	Text    string `json:"text,omitempty"`
	Summary string `json:"summary,omitempty"`
}

func normalizeReasoningAliases(reasoningContent, reasoning string, details []OpenAIReasoningDetail) string {
	if reasoningContent != "" {
		return reasoningContent
	}
	if reasoning != "" {
		return reasoning
	}
	var parts []string
	for _, d := range details {
		switch {
		case d.Text != "":
			parts = append(parts, d.Text)
		case d.Summary != "":
			parts = append(parts, d.Summary)
		}
	}
	return strings.Join(parts, "\n")
}
```

Add `OpenAIMessage.UnmarshalJSON`:

```go
func (m *OpenAIMessage) UnmarshalJSON(data []byte) error {
	type shadow OpenAIMessage
	var aux struct {
		shadow
		Reasoning        string                 `json:"reasoning,omitempty"`
		ReasoningDetails []OpenAIReasoningDetail `json:"reasoning_details,omitempty"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*m = OpenAIMessage(aux.shadow)
	m.ReasoningContent = normalizeReasoningAliases(m.ReasoningContent, aux.Reasoning, aux.ReasoningDetails)
	return nil
}
```

Extend the `OpenAIDelta.UnmarshalJSON` aux with `Reasoning` and `ReasoningDetails`, then after assigning existing fields:

```go
d.ReasoningContent = normalizeReasoningAliases(aux.ReasoningContent, aux.Reasoning, aux.ReasoningDetails)
```

Keep `Refusal` in the aux from Task 2.

- [x] **Step 4: Run the tests**

Run: `go test -count=1 -run 'TestP2_Reasoning' ./convert`

Expected: PASS.

Run: `go test -count=1 ./...`

Expected: all packages PASS.

- [x] **Step 5: Commit**

```bash
git add convert/types.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): accept reasoning aliases"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `a84076c`
- Files changed: `convert/types.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED `go test -count=1 -run 'TestP2_Reasoning' ./convert` → alias tests produced text-only/empty thinking; GREEN same command → PASS; `go test -count=1 ./...` → all packages PASS; `go vet ./...` and `git diff --check` clean.
- Review findings/fixes: plan tests assumed the assistant message was at index 0; the P1 normalizer inserts a synthetic user turn for assistant-first history. Ruling: search for the assistant message instead of assuming its index. Cost if wrong: the test would assert the synthetic user turn and miss the actual conversion.
- Deferred/limitations: alias fields are normalized into `ReasoningContent` and not re-emitted; `reasoning_details` signatures/encrypted payloads are not preserved, only text/summary.

---

### Task 4: Complete stream block starts, no-argument tools, interrupted text

**Files:**
- Modify: `convert/stream.go` (`ensureBlock`, new `closeOpenToolBlock`, `handleToolCallDelta`, `HandleStreamEnd`)
- Modify: `convert/stream_gemini.go` (`GeminiToAnthropicStreamConverter.ensureBlock`)
- Modify: `convert/contract_test.go` (`validateAnthropicSSE` block-start fields, negative test)
- Test: `convert/anthropic_openai_p2_test.go`

**Interfaces:**
- Produces: `func (sc *StreamConverter) closeOpenToolBlock() [][]byte`
- Behavior: both Anthropic block-start producers emit `"text":""`, thinking
  `"thinking":""`/`"signature":""`, and tool `"input":{}`.
- Behavior: every tool block close with empty accumulated args emits
  `input_json_delta` `{}` first.

- [x] **Step 1: Write the failing tests**

Append:

```go
func TestP2_StreamBlockStartPayloads(t *testing.T) {
	t.Run("text", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`))
		if !strings.Contains(string(out), `"content_block":{"type":"text","text":""}`) {
			t.Fatalf("text start = %s", out)
		}
	})

	t.Run("thinking", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}`))
		if !strings.Contains(string(out), `"content_block":{"type":"thinking","thinking":"","signature":""}`) {
			t.Fatalf("thinking start = %s", out)
		}
	})

	t.Run("tool", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`))
		if !strings.Contains(string(out), `"input":{}`) {
			t.Fatalf("tool start = %s", out)
		}
	})
}

func TestP2_GeminiAnthropicBlockStarts(t *testing.T) {
	t.Run("text", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		c.HandleStreamStart()
		out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), `"content_block":{"type":"text","text":""`) {
			t.Fatalf("gemini text start = %s", out)
		}
	})

	t.Run("thinking", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		c.HandleStreamStart()
		out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"text":"think","thought":true}],"role":"model"}}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), `"content_block":{"type":"thinking","thinking":"","signature":""`) {
			t.Fatalf("gemini thinking start = %s", out)
		}
	})

	t.Run("tool", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		c.HandleStreamStart()
		out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}],"role":"model"},"finishReason":"STOP"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), `"input":{}`) {
			t.Fatalf("gemini tool start = %s", out)
		}
	})
}

func TestP2_NoArgumentToolEmitsEmptyObject(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	first, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
	end := string(sc.HandleStreamEnd())
	all := string(first) + end
	if !strings.Contains(all, `"partial_json":"{}"`) {
		t.Fatalf("no empty object for no-arg tool: %s", all)
	}
	if strings.Index(all, `"partial_json":"{}"`) > strings.Index(all, `"content_block_stop"`) {
		t.Fatalf("empty object emitted after block stop: %s", all)
	}
}

func TestP2_TwoNoArgumentToolsBothClose(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out1, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
	out2, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"g","arguments":""}}]},"finish_reason":null}]}`))
	end := string(sc.HandleStreamEnd())
	all := string(out1) + string(out2) + end
	if got := strings.Count(all, `"partial_json":"{}"`); got != 2 {
		t.Fatalf("empty object deltas = %d, want 2: %s", got, all)
	}
	if got := strings.Count(all, "event: content_block_start"); got != 2 {
		t.Fatalf("tool block starts = %d, want 2: %s", got, all)
	}
}

func TestP2_InterruptedStreamSingleTextBlock(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`))
	sc.SetInterrupted()
	end := sc.HandleStreamEnd()
	all := string(out) + string(end)
	if got := strings.Count(all, "event: content_block_start"); got != 1 {
		t.Fatalf("text block starts = %d, want 1: %s", got, all)
	}
	if got := strings.Count(all, "event: content_block_stop"); got != 1 {
		t.Fatalf("text block stops = %d, want 1: %s", got, all)
	}
	if !strings.Contains(all, "[stream interrupted]") {
		t.Fatalf("interruption marker missing: %s", all)
	}
}

func TestP2_EmptyInterruptedStreamSingleTextBlock(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	sc.SetInterrupted()
	end := string(sc.HandleStreamEnd())
	if got := strings.Count(end, "event: content_block_start"); got != 1 {
		t.Fatalf("text block starts = %d, want 1: %s", got, end)
	}
	if !strings.Contains(end, "[stream interrupted]") {
		t.Fatalf("interruption marker missing: %s", end)
	}
}
```

- [x] **Step 2: Run the tests and capture RED**

Run: `go test -count=1 -run 'TestP2_(StreamBlockStartPayloads|GeminiAnthropicBlockStarts|NoArgumentToolEmitsEmptyObject|TwoNoArgumentToolsBothClose|InterruptedStreamSingleTextBlock|EmptyInterruptedStreamSingleTextBlock)' ./convert`

Expected: text/thinking start payloads missing (both OpenAI→Anthropic and Gemini→Anthropic), no `{}` delta, two no-arg tools not both closing, and two text blocks on the interrupted stream.

- [x] **Step 3: Implement the stream changes**

In `convert/stream.go`:

1. In `ensureBlock`, use this close/start structure:

```go
if sc.curBlockType != "" {
	if sc.curBlockType == "thinking" {
		events = append(events, sc.thinkingBlockStop())
		sc.thinkingBlockStopped = true
	} else if sc.curBlockType == "tool_use" {
		events = append(events, sc.closeOpenToolBlock()...)
		sc.openToolIndex = -1
	} else {
		events = append(events, sc.contentBlockStop())
	}
}

idx := sc.nextBlockIndex
sc.nextBlockIndex++
sc.curBlockType = blockType
sc.curBlockIndex = idx

if blockType == "tool_use" {
	events = append(events, toolUseBlockStart(idx, toolID, toolName))
	return events
}

var payload []byte
switch blockType {
case "text":
	payload = []byte(`{"type":"text","text":""}`)
case "thinking":
	payload = []byte(`{"type":"thinking","thinking":"","signature":""}`)
default:
	payload = []byte(fmt.Sprintf(`{"type":%q}`, blockType))
}
startPayload := fmt.Sprintf(
	`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":%d,"content_block":%s}`,
	idx, payload,
)
events = append(events, []byte(startPayload))
return events
```

2. Add:

```go
func (sc *StreamConverter) closeOpenToolBlock() [][]byte {
	var events [][]byte
	if state := sc.toolCallByIndex[sc.openToolIndex]; state != nil && strings.TrimSpace(state.Arguments) == "" {
		events = append(events, sc.inputJSONDelta(sc.curBlockIndex, "{}")...)
	}
	events = append(events, sc.contentBlockStop())
	return events
}
```

3. In `handleToolCallDelta`, change the hold check to `!streamToolArgsComplete(open.Arguments)` and replace the explicit tool stop with `sc.closeOpenToolBlock()`.

4. In `HandleStreamEnd`:
   - move the interruption marker block **before** closing the current content block:
     ```go
     if sc.streamInterrupted {
         events = append(events, sc.ensureBlock("text", "", "")...)
         events = append(events, sc.textDelta("\n\n[stream interrupted]")...)
     }
     ```
   - close an open tool block with `sc.closeOpenToolBlock()` instead of `sc.contentBlockStop()`.
   - remove the old interruption block from its later position so the marker is emitted once.
   - keep the empty-text placeholder; because the marker block is already open, it appends to the same block.

5. In `convert/stream_gemini.go`, `GeminiToAnthropicStreamConverter.ensureBlock`, keep the existing string builder but add the literal fields:

```go
startPayload := fmt.Sprintf(`data: {"type":"content_block_start","index":%d,"content_block":{"type":"%s"`, idx, blockType)
switch blockType {
case "text":
	startPayload += `,"text":""`
case "thinking":
	startPayload += `,"thinking":"","signature":""`
case "tool_use":
	if meta != "" {
		startPayload += "," + meta
	}
	startPayload += `,"input":{}`
}
startPayload += `}}`
```

6. In `convert/contract_test.go`, in `validateAnthropicSSE`'s `content_block_start` case, require:
   - text block: `block["text"]` is a string
   - thinking block: `block["thinking"]` and `block["signature"]` are strings
   - tool_use block: `block["input"]` is an object

Add a negative case to `TestContractValidatorsRejectKnownInvalid`:

```go
t.Run("Anthropic stream text start without text", func(t *testing.T) {
	stream := []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"claude\"}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}")
	if errs := validateAnthropicSSE(stream); len(errs) == 0 {
		t.Fatal("validator accepted a text block start without text")
	}
})
```

- [x] **Step 4: Run the focused and contract tests**

Run: `go test -count=1 -run 'TestP2_(StreamBlockStartPayloads|GeminiAnthropicBlockStarts|NoArgumentToolEmitsEmptyObject|TwoNoArgumentToolsBothClose|InterruptedStreamSingleTextBlock|EmptyInterruptedStreamSingleTextBlock)|TestContract' ./convert`

Expected: PASS.

Run: `go test -count=1 ./...`

Expected: all packages PASS.

- [x] **Step 5: Commit**

```bash
git add convert/stream.go convert/stream_gemini.go convert/contract_test.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): complete stream block payloads and tool closes"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `48e248a`
- Files changed: `convert/stream.go`, `convert/stream_gemini.go`, `convert/contract_test.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED focused run showed missing text/thinking start fields, one `{}` delta for two no-arg tools, and two text blocks on the interrupted stream; GREEN focused + `TestContract` → PASS; `go test -count=1 ./...` → all packages PASS; `go vet ./...` and `git diff --check` clean.
- Review findings/fixes: the empty-text placeholder moved before block close so it appends to the interruption marker block; both `StreamConverter` and `GeminiToAnthropicStreamConverter` now emit complete block-start payloads. Contract validator now rejects a text start without `text`.
- Deferred/limitations: none.

---

### Task 5: Sanitize Anthropic tool IDs and align cache keys

**Files:**
- Modify: `convert/convert.go` (new `sanitizeAnthropicToolID`)
- Modify: `convert/openai_to_anthropic.go` (request assistant/result, response content, cache `Put`)
- Modify: `convert/stream.go` (`handleToolCallDelta` tool ID)
- Test: `convert/anthropic_openai_p2_test.go`

**Interfaces:**
- Produces: `func sanitizeAnthropicToolID(id string) string`
- Behavior: empty IDs use `ensureToolID`; valid IDs are unchanged; every other byte becomes `_`.

- [x] **Step 1: Write the failing tests**

Append:

```go
func assertAnthropicToolID(t *testing.T, id string) {
	t.Helper()
	if id == "" {
		t.Fatal("empty tool ID")
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			continue
		}
		t.Fatalf("tool ID %q contains invalid byte %q", id, c)
	}
}

func TestP2_ToolIDRequestPairing(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call.a/b","content":"ok"},
		{"role":"user","content":"done"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "claude-sonnet"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	if err := json.Unmarshal(out, &req); err != nil {
		t.Fatal(err)
	}
	var toolUseID, resultID string
	for _, msg := range req.Messages {
		for _, block := range msg.Content {
			switch block.Type {
			case "tool_use":
				toolUseID = block.ID
			case "tool_result":
				resultID = block.ToolUseID
			}
		}
	}
	assertAnthropicToolID(t, toolUseID)
	if toolUseID != "call_a_b" || resultID != "call_a_b" {
		t.Fatalf("tool_use=%q result=%q, want call_a_b for both", toolUseID, resultID)
	}
}

func TestP2_ToolIDResponseSanitized(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	out, err := convertOpenAIResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if len(resp.Content) == 0 || resp.Content[0].Type != "tool_use" {
		t.Fatalf("content = %+v", resp.Content)
	}
	assertAnthropicToolID(t, resp.Content[0].ID)
	if resp.Content[0].ID != "call_a_b" {
		t.Fatalf("id = %q, want call_a_b", resp.Content[0].ID)
	}
}

func TestP2_ToolIDStreamSanitized(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"id":"call_a_b"`) {
		t.Fatalf("stream tool ID not sanitized: %s", out)
	}
}

func TestP2_ToolIDCacheUsesSanitizedID(t *testing.T) {
	cache := NewReasoningCache(8)
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","reasoning_content":"chain","tool_calls":[{"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	if _, err := convertOpenAIResponseToAnthropic([]byte(body), &ConvertOptions{ReasoningCache: cache}); err != nil {
		t.Fatal(err)
	}
	if got, ok := cache.Get([]string{"call_a_b"}); !ok || got != "chain" {
		t.Fatalf("cache.Get(call_a_b) = %q, %v; want chain, true", got, ok)
	}
	if _, ok := cache.Get([]string{"call.a/b"}); ok {
		t.Fatal("cache stored the raw invalid ID")
	}
}
```

- [x] **Step 2: Run the tests and capture RED**

Run: `go test -count=1 -run 'TestP2_ToolID' ./convert`

Expected: invalid IDs pass through unchanged and the cache is keyed by the raw ID.

- [x] **Step 3: Implement sanitization**

In `convert/convert.go`, next to `ensureToolID`:

```go
func sanitizeAnthropicToolID(id string) string {
	if id == "" {
		return ensureToolID("")
	}
	var b strings.Builder
	b.Grow(len(id))
	for i := 0; i < len(id); i++ {
		c := id[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_', c == '-':
			b.WriteByte(c)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}
```

Apply it:

- `convertAssistantContent`: `id := sanitizeAnthropicToolID(tc.ID)` (remove the empty-only branch).
- `convertToolResultContent`: `ToolUseID: sanitizeAnthropicToolID(msg.ToolCallID)`.
- `convertOpenAIMessageToContent`: `id := sanitizeAnthropicToolID(tc.ID)`.
- `handleToolCallDelta`: `toolID := sanitizeAnthropicToolID(tc.ID)`.
- `convertOpenAIResponseToAnthropic` cache block: `ids[i] = sanitizeAnthropicToolID(tc.ID)`.
- `StreamConverter.HandleStreamEnd` cache block already uses `state.ID`, which is sanitized when the state is created.

- [x] **Step 4: Run the focused and full tests**

Run: `go test -count=1 -run 'TestP2_ToolID' ./convert`

Expected: PASS.

Run: `go test -count=1 ./...`

Expected: all packages PASS.

- [x] **Step 5: Commit**

```bash
git add convert/convert.go convert/openai_to_anthropic.go convert/stream.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): sanitize anthropic tool ids"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `f8d9aec`
- Files changed: `convert/convert.go`, `convert/openai_to_anthropic.go`, `convert/stream.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED `go test -count=1 -run 'TestP2_ToolID' ./convert` → `call.a/b` passed through and cache missed `call_a_b`; GREEN same command → PASS; `go test -count=1 ./...` → all packages PASS; `go vet ./...` and `git diff --check` clean.
- Review findings/fixes: none.
- Deferred/limitations: two distinct invalid IDs that collapse to the same sanitized string are accepted as a known risk; hashing is not used.

---

### Task 6: Request fidelity: developer role, parallel tool calls, top_k

**Files:**
- Modify: `convert/types.go` (`OpenAIChatRequest.ParallelToolCalls`, `AnthropicToolChoice.DisableParallelToolUse`)
- Modify: `convert/openai_to_anthropic.go` (developer role, parallel mapping)
- Modify: `convert/anthropic_to_openai.go` (parallel mapping, top_k gating, helpers)
- Modify: `convert/responses.go` (`ConvertResponsesToAnthropic` parallel mapping)
- Test: `convert/anthropic_openai_p2_test.go`

**Interfaces:**
- Produces: `OpenAIChatRequest.ParallelToolCalls *bool`
- Produces: `AnthropicToolChoice.DisableParallelToolUse *bool`
- Produces: `func supportsTopKPassthrough(model string) bool`
- Produces: `func stripAllProviderPrefixes(model string) string`

- [x] **Step 1: Write the failing tests**

Append:

```go
func TestP2_DeveloperRoleBecomesSystem(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"developer","content":"be brief"},
		{"role":"user","content":"hi"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "claude-sonnet"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	_ = json.Unmarshal(out, &req)
	if len(req.System) != 1 || req.System[0].Text != "be brief" {
		t.Fatalf("system = %+v, want developer text", req.System)
	}
	for _, msg := range req.Messages {
		if msg.Role == "user" && len(msg.Content) == 1 && msg.Content[0].Text == "be brief" {
			t.Fatal("developer text leaked into a user message")
		}
	}
}

func TestP2_ParallelToolCallsChat(t *testing.T) {
	tools := `"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]`
	cases := []struct {
		name        string
		extra       string
		wantDisable *bool
	}{
		{"false", `"parallel_tool_calls":false,`, boolPtr(true)},
		{"true", `"parallel_tool_calls":true,`, boolPtr(false)},
		{"none choice", `"parallel_tool_calls":false,"tool_choice":"none",`, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"gpt-4",` + tc.extra + tools + `,"messages":[{"role":"user","content":"hi"}]}`
			out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "claude-sonnet"})
			if err != nil {
				t.Fatal(err)
			}
			var req AnthropicRequest
			_ = json.Unmarshal(out, &req)
			if tc.wantDisable == nil {
				if req.ToolChoice != nil && req.ToolChoice.DisableParallelToolUse != nil {
					t.Fatalf("disable_parallel_tool_use = %v, want absent", *req.ToolChoice.DisableParallelToolUse)
				}
				return
			}
			if req.ToolChoice == nil || req.ToolChoice.DisableParallelToolUse == nil ||
				*req.ToolChoice.DisableParallelToolUse != *tc.wantDisable {
				t.Fatalf("tool_choice = %+v, want disable=%v", req.ToolChoice, *tc.wantDisable)
			}
		})
	}
}

func TestP2_ParallelToolCallsReverse(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"tool_choice":{"type":"auto","disable_parallel_tool_use":true},
		"tools":[{"name":"f","input_schema":{"type":"object"}}],
		"messages":[{"role":"user","content":"hi"}]}`
	out, err := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	if err != nil {
		t.Fatal(err)
	}
	var req OpenAIChatRequest
	_ = json.Unmarshal(out, &req)
	if req.ParallelToolCalls == nil || *req.ParallelToolCalls != false {
		t.Fatalf("parallel_tool_calls = %v, want false", req.ParallelToolCalls)
	}
}

func TestP2_ResponsesParallelToolCalls(t *testing.T) {
	body := `{"model":"gpt-5","parallel_tool_calls":false,"input":"hi",
		"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]}`
	out, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	_ = json.Unmarshal(out, &req)
	if req.ToolChoice == nil || req.ToolChoice.DisableParallelToolUse == nil ||
		!*req.ToolChoice.DisableParallelToolUse {
		t.Fatalf("tool_choice = %+v, want disable_parallel_tool_use=true", req.ToolChoice)
	}
}

func TestP2_TopKProviderGating(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"groq/llama-3.3-70b", true},
		{"openrouter/meta-llama/llama-3.1-70b", true},
		{"vllm/mistral-7b", true},
		{"openrouter/openai/gpt-5", false},
		{"gpt-4o", false},
		{"o3-mini", false},
		{"deepseek-chat", false},
	}
	body := `{"model":"claude","max_tokens":10,"top_k":40,"messages":[{"role":"user","content":"hi"}]}`
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			out, err := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: tc.model})
			if err != nil {
				t.Fatal(err)
			}
			var req OpenAIChatRequest
			_ = json.Unmarshal(out, &req)
			if (req.TopK != nil) != tc.want {
				t.Fatalf("top_k present = %v, want %v (top_k=%v)", req.TopK != nil, tc.want, req.TopK)
			}
		})
	}
}
```

- [x] **Step 2: Run the tests and capture RED**

Run: `go test -count=1 -run 'TestP2_(DeveloperRoleBecomesSystem|ParallelToolCalls|ResponsesParallelToolCalls|TopKProviderGating)' ./convert`

Expected: developer becomes a user message; parallel fields are dropped; `top_k` is dropped for all models.

- [x] **Step 3: Implement the mappings**

In `convert/types.go`:

```go
type OpenAIChatRequest struct {
	// ...
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
}

type AnthropicToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}
```

In `convert/openai_to_anthropic.go`, change the system branch:

```go
if m.Role == "system" || m.Role == "developer" {
	systemTexts = append(systemTexts, extractTextContent(m.Content))
	continue
}
```

Inside `if len(req.Tools) > 0`, after the tool-choice switch:

```go
if req.ParallelToolCalls != nil {
	if anthropic.ToolChoice == nil {
		anthropic.ToolChoice = &AnthropicToolChoice{Type: "auto"}
	}
	if anthropic.ToolChoice.Type != "none" {
		disable := !*req.ParallelToolCalls
		anthropic.ToolChoice.DisableParallelToolUse = &disable
	}
}
```

In `convert/anthropic_to_openai.go`, after `oai.ToolChoice = anthropicToolChoiceToOpenAi(...)`:

```go
if len(oai.Tools) > 0 && req.ToolChoice != nil && req.ToolChoice.DisableParallelToolUse != nil {
	parallel := !*req.ToolChoice.DisableParallelToolUse
	oai.ParallelToolCalls = &parallel
}
```

Add the top_k condition in `convertAnthropicRequestToOpenAI`:

```go
if req.TopK != nil && !isOpenAIReasoningModel(stripAllProviderPrefixes(outputModel)) &&
	supportsTopKPassthrough(outputModel) {
	oai.TopK = req.TopK
}
```

Add helpers in `convert/anthropic_to_openai.go`:

```go
func stripAllProviderPrefixes(model string) string {
	for {
		i := strings.IndexByte(model, '/')
		if i < 0 {
			return model
		}
		model = model[i+1:]
	}
}

func supportsTopKPassthrough(model string) bool {
	i := strings.IndexByte(model, '/')
	if i < 0 {
		return false
	}
	switch strings.ToLower(model[:i]) {
	case "groq", "openrouter", "together", "fireworks", "vllm", "ollama", "lmstudio":
		return true
	}
	return false
}
```

In `convert/responses.go`, in `ConvertResponsesToAnthropic`, after the
`if len(req.Tools) > 0 { ... }` block:

```go
if len(anth.Tools) > 0 && req.ParallelToolCalls != nil {
	if anth.ToolChoice == nil {
		anth.ToolChoice = &AnthropicToolChoice{Type: "auto"}
	}
	if anth.ToolChoice.Type != "none" {
		disable := !*req.ParallelToolCalls
		anth.ToolChoice.DisableParallelToolUse = &disable
	}
}
```

- [x] **Step 4: Run the focused and full tests**

Run: `go test -count=1 -run 'TestP2_(DeveloperRoleBecomesSystem|ParallelToolCalls|ResponsesParallelToolCalls|TopKProviderGating)' ./convert`

Expected: PASS.

Run: `go test -count=1 ./...`

Expected: all packages PASS.

- [x] **Step 5: Commit**

```bash
git add convert/types.go convert/openai_to_anthropic.go convert/anthropic_to_openai.go convert/responses.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): preserve request fidelity fields"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `7e9f5e4`
- Files changed: `convert/types.go`, `convert/openai_to_anthropic.go`, `convert/anthropic_to_openai.go`, `convert/responses.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED build failed on missing `ParallelToolCalls`/`DisableParallelToolUse` fields; GREEN focused run → PASS; `go test -count=1 ./...` → all packages PASS; `go vet ./...` and `git diff --check` clean.
- Review findings/fixes: `top_k` is gated to known providers and checks all provider prefixes for reasoning models; `disable_parallel_tool_use` is skipped for Anthropic `none` and when no tools exist.
- Deferred/limitations: Responses request → Anthropic is the only Responses direction with a parallel mapping; there is no Anthropic request → Responses request converter.

---

### Task 7: Reject truncated Responses tool streams and emit `{}` for no-argument tools

**Files:**
- Modify: `convert/stream_responses.go` (`HandleStreamEnd`)
- Modify: `convert/stream_anthropic.go` (`HandleStreamEnd`)
- Test: `convert/anthropic_openai_p2_test.go`

**Interfaces:**
- Behavior: both Responses converters call `EmitError("stream ended with incomplete tool call arguments")` before finalizing when any non-empty argument string is not a JSON object.
- Behavior: no-argument function calls emit `"arguments":"{}"` in `response.function_call_arguments.done` and `response.output_item.done`.

- [x] **Step 1: Write the failing tests**

Append:

```go
func TestP2_ResponsesTruncatedToolErrors(t *testing.T) {
	sc := NewResponsesStreamConverter("gpt-4o")
	sc.HandleStreamStart()
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]},"finish_reason":null}]}`))
	end := string(sc.HandleStreamEnd())
	if !strings.Contains(end, "response.failed") || !strings.Contains(end, "incomplete tool call arguments") {
		t.Fatalf("truncated Responses stream end = %s, want response.failed", end)
	}
	if strings.Contains(end, "response.completed") || strings.Contains(end, "response.function_call_arguments.done") {
		t.Fatalf("truncated Responses stream closed normally: %s", end)
	}
}

func TestP2_AnthropicToResponsesTruncatedToolErrors(t *testing.T) {
	sc := NewAnthropicStreamConverter("claude")
	sc.HandleStreamStart()
	_, _ = sc.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f","input":{}}}`))
	_, _ = sc.HandleChunk([]byte(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`))
	end := string(sc.HandleStreamEnd())
	if !strings.Contains(end, "response.failed") || !strings.Contains(end, "incomplete tool call arguments") {
		t.Fatalf("truncated Anthropic→Responses end = %s, want response.failed", end)
	}
	if strings.Contains(end, "response.completed") || strings.Contains(end, "response.function_call_arguments.done") {
		t.Fatalf("truncated Anthropic→Responses stream closed normally: %s", end)
	}
}

func TestP2_ResponsesNoArgumentToolEmitsEmptyObject(t *testing.T) {
	t.Run("chat to responses", func(t *testing.T) {
		sc := NewResponsesStreamConverter("gpt-4o")
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"arguments":"{}"`) {
			t.Fatalf("no-arg Responses tool = %s, want arguments {}", end)
		}
	})

	t.Run("anthropic to responses", func(t *testing.T) {
		sc := NewAnthropicStreamConverter("claude")
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f","input":{}}}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"arguments":"{}"`) {
			t.Fatalf("no-arg Anthropic→Responses tool = %s, want arguments {}", end)
		}
	})
}
```

- [x] **Step 2: Run the tests and capture RED**

Run: `go test -count=1 -run 'TestP2_(ResponsesTruncatedToolErrors|AnthropicToResponsesTruncatedToolErrors|ResponsesNoArgumentToolEmitsEmptyObject)' ./convert`

Expected: truncated streams complete successfully and no-argument items carry empty arguments.

- [x] **Step 3: Implement the guards and `{}` output**

In both `HandleStreamEnd` methods, **before** `sc.finalized = true`:

```go
for _, fc := range sc.toolCallByIndex {
	if !streamToolArgsComplete(fc.Arguments) {
		return sc.EmitError("stream ended with incomplete tool call arguments")
	}
}
```

In both tool-finalization loops, replace the empty-arguments handling with:

```go
args := fc.Arguments
if strings.TrimSpace(args) == "" {
	args = "{}"
} else {
	args = canonicalJSONString(args)
}
```

Use `args` in `response.function_call_arguments.done` and in the
`response.output_item.done` item's `Arguments` field. Emit
`function_call_arguments.done` unconditionally for each tool call (it is now
safe because the guard runs first).

- [x] **Step 4: Run the focused, full, and race tests**

Run: `go test -count=1 -run 'TestP2_(ResponsesTruncatedToolErrors|AnthropicToResponsesTruncatedToolErrors|ResponsesNoArgumentToolEmitsEmptyObject)' ./convert`

Expected: PASS.

Run: `go test -count=1 ./...`

Expected: all packages PASS.

Run: `CGO_ENABLED=1 go test -race -count=1 ./convert`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add convert/stream_responses.go convert/stream_anthropic.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): reject truncated responses tool streams"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `5a50512`
- Files changed: `convert/stream_responses.go`, `convert/stream_anthropic.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED focused run showed truncated streams emitting `function_call_arguments.done` and `response.completed`; GREEN focused run → PASS; `go test -count=1 ./...` → all packages PASS; `CGO_ENABLED=1 go test -race -count=1 ./convert` → PASS; `go vet ./...` and `git diff --check` clean.
- Review findings/fixes: both Responses-bound converters guard before setting `finalized`; no-argument calls now emit `"arguments":"{}"` in both done events.
- Deferred/limitations: none.

---

### Task 8: Make Gemini stream errors JSON-safe and terminal

**Files:**
- Modify: `convert/gemini_types.go` (`GeminiError`, `GeminiChatResponse.Error`)
- Modify: `convert/stream_gemini.go` (both `EmitError` methods, error routing)
- Modify: `convert/stream_to_gemini.go` (both `EmitError` methods, error routing)
- Modify: `convert/convert.go` (end-phase error dispatch)
- Modify: `convert/anthropic_openai_p1_test.go` (replace the non-dispatch test)
- Test: `convert/anthropic_openai_p2_test.go`

**Interfaces:**
- Produces: `type GeminiError struct { Code int; Message, Status string }`
- Produces: `GeminiChatResponse.Error *GeminiError`
- Behavior: every stateful Gemini converter's `EmitError` is JSON-safe, idempotent, and terminal.

- [x] **Step 1: Write the failing tests**

Append:

```go
func TestP2_GeminiEmitErrorSafety(t *testing.T) {
	message := "bad \"quote\"\nline\\slash"
	t.Run("gemini to openai", func(t *testing.T) {
		c := NewGeminiToOpenAIStreamConverter("gemini-2.5-flash")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
		payload := strings.TrimSuffix(strings.TrimPrefix(string(first), "data: "), "\n\n")
		var evt struct{ Error struct{ Message string `json:"message"` } `json:"error"` }
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			t.Fatalf("invalid error JSON %q: %v", payload, err)
		}
		if evt.Error.Message != message {
			t.Fatalf("message = %q, want %q", evt.Error.Message, message)
		}
	})

	t.Run("gemini to anthropic", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
		if !strings.HasPrefix(string(first), "event: error\ndata: ") {
			t.Fatalf("error framing = %q", first)
		}
		var evt struct{ Error struct{ Message string `json:"message"` } `json:"error"` }
		payload := strings.TrimPrefix(string(first), "event: error\ndata: ")
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			t.Fatalf("invalid error JSON %q: %v", payload, err)
		}
		if evt.Error.Message != message {
			t.Fatalf("message = %q, want %q", evt.Error.Message, message)
		}
	})

	t.Run("anthropic to gemini", func(t *testing.T) {
		c := NewAnthropicToGeminiStreamConverter("gemini-2.5-flash")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
	})

	t.Run("openai to gemini", func(t *testing.T) {
		c := NewOpenAIToGeminiStreamConverter("gemini-2.5-flash")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
	})
}

func TestP2_GeminiUpstreamErrorTerminal(t *testing.T) {
	t.Run("gemini to openai", func(t *testing.T) {
		c := NewGeminiToOpenAIStreamConverter("gemini-2.5-flash")
		out, err := c.HandleChunk([]byte(`{"error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})

	t.Run("gemini to anthropic", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		out, err := c.HandleChunk([]byte(`{"error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})

	t.Run("anthropic to gemini", func(t *testing.T) {
		c := NewAnthropicToGeminiStreamConverter("gemini-2.5-flash")
		out, err := c.HandleChunk([]byte(`{"type":"error","error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})

	t.Run("openai to gemini", func(t *testing.T) {
		c := NewOpenAIToGeminiStreamConverter("gemini-2.5-flash")
		out, err := c.HandleChunk([]byte(`{"error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})
}

func TestP2_GeminiEndPhaseStreamErrorDispatched(t *testing.T) {
	cases := []struct {
		name       string
		handler    responsesStreamHandler
		wantError  string
		normalMark string
	}{
		{"gemini to openai", NewGeminiToOpenAIStreamConverter("gemini-2.5-flash"), "data: ", `[DONE]`},
		{"gemini to anthropic", NewGeminiToAnthropicStreamConverter("claude"), "event: error", "message_stop"},
		{"anthropic to gemini", NewAnthropicToGeminiStreamConverter("gemini-2.5-flash"), "data: ", "finishReason"},
		{"openai to gemini", NewOpenAIToGeminiStreamConverter("gemini-2.5-flash"), "data: ", "finishReason"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := NewSessionStore()
			store.Set("g1", &Session{ID: "g1", StreamHandler: tc.handler})
			opts := &ConvertOptions{SessionStore: store, Direction: "response", SID: "g1", ErrorMsg: `upstream "reset"`}
			out, err := HandleSSEEvent("g1", "end", 0, nil, opts)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(out), tc.wantError) {
				t.Fatalf("end-phase error not dispatched: %s", out)
			}
			if strings.Contains(string(out), tc.normalMark) {
				t.Fatalf("normal close followed the error: %s", out)
			}
		})
	}
}
```

Replace `TestP1_GeminiEndPhaseStreamErrorNotDispatched` in
`convert/anthropic_openai_p1_test.go` with the new expectation, or delete it
because `TestP2_GeminiEndPhaseStreamErrorDispatched` covers the behavior.

- [x] **Step 2: Run the tests and capture RED**

Run: `go test -count=1 -run 'TestP2_Gemini' ./convert`

Expected: `GeminiToAnthropicStreamConverter.EmitError` corrupts JSON for quotes, several converters close normally after an error, and end-phase errors are not dispatched to the Gemini handler.

- [x] **Step 3: Implement terminal, JSON-safe errors and routing**

In `convert/gemini_types.go`:

```go
type GeminiError struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Status  string `json:"status,omitempty"`
}
```

Add `Error *GeminiError \`json:"error,omitempty"\`` to `GeminiChatResponse`.

`GeminiToOpenAIStreamConverter.EmitError`:

```go
func (c *GeminiToOpenAIStreamConverter) EmitError(message string) []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true
	b, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": message, "type": "server_error"},
	})
	return []byte("data: " + string(b) + "\n\n")
}
```

`GeminiToAnthropicStreamConverter.EmitError`:

```go
func (c *GeminiToAnthropicStreamConverter) EmitError(message string) []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true
	b, _ := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "stream_error", "message": message},
	})
	return []byte("event: error\ndata: " + string(b))
}
```

`AnthropicToGeminiStreamConverter.EmitError` and
`OpenAIToGeminiStreamConverter.EmitError`: add

```go
if c.finished {
	return nil
}
c.finished = true
```

before the existing JSON marshal and return.

Route detected errors:

- `AnthropicToGeminiStreamConverter.HandleChunk`, `case "error"`: extract the
  message (fallback `"upstream stream error"`) and `return c.EmitError(msg), nil`
  instead of building the chunk inline.
- `OpenAIToGeminiStreamConverter.HandleChunk`, after unmarshal:

  ```go
  if chunk.Error != nil {
      msg := chunk.Error.Message
      if msg == "" {
          msg = "upstream stream error"
      }
      return c.EmitError(msg), nil
  }
  ```

- `GeminiToOpenAIStreamConverter.HandleChunk`, after unmarshal:

  ```go
  if resp.Error != nil {
      msg := resp.Error.Message
      if msg == "" {
          msg = "upstream stream error"
      }
      return c.EmitError(msg), nil
  }
  ```

  Replace the blocked-response inline error chunk with
  `return c.EmitError(fmt.Sprintf("blocked by Gemini safety: %s", ...)), nil`.

- `GeminiToAnthropicStreamConverter.HandleChunk`: add the same `resp.Error`
  guard. Its blocked path already calls `EmitError`.

In `convert/convert.go`, extend the end-phase switch in `handleSSEEvent`:

```go
switch handler.(type) {
case *StreamConverter, *AnthropicToOpenAIStreamConverter,
	*GeminiToOpenAIStreamConverter, *GeminiToAnthropicStreamConverter,
	*AnthropicToGeminiStreamConverter, *OpenAIToGeminiStreamConverter:
	return handler.EmitError(opts.ErrorMsg), nil
}
```

- [x] **Step 4: Run the focused, full, and race tests**

Run: `go test -count=1 -run 'TestP2_Gemini|TestP1_Gemini' ./convert`

Expected: PASS.

Run: `go test -count=1 ./...`

Expected: all packages PASS.

Run: `CGO_ENABLED=1 go test -race -count=1 ./convert`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add convert/gemini_types.go convert/stream_gemini.go convert/stream_to_gemini.go convert/convert.go convert/anthropic_openai_p1_test.go convert/anthropic_openai_p2_test.go
git commit -m "fix(convert): make gemini stream errors terminal"
```

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `c8c5138`
- Files changed: `convert/gemini_types.go`, `convert/stream_gemini.go`, `convert/stream_to_gemini.go`, `convert/convert.go`, `convert/anthropic_openai_p1_test.go`, `convert/anthropic_openai_p2_test.go`
- Tests/commands/results: RED `go test -count=1 -run 'TestP2_Gemini' ./convert` → non-idempotent errors, corrupted JSON, normal close after upstream error, and no end-phase dispatch; GREEN focused → PASS; full `go test -count=1 ./...` → all packages PASS; `CGO_ENABLED=1 go test -race -count=1 ./convert` → PASS; `go vet ./...` and `git diff --check` clean.
- Review findings/fixes: `GeminiToOpenAI` keeps `content_filter` for blocked responses via an internal `emitError(message, type)` helper while `EmitError` uses `server_error`; both are terminal. Deleted the obsolete `TestP1_GeminiEndPhaseStreamErrorNotDispatched`.
- Deferred/limitations: `GeminiStreamHandler` remains stateless and out of scope; it has no lifecycle to terminate.

---

### Final review fix pass (2026-11-06)

The fresh whole-branch review found one Critical and three Important issues.
All were fixed in one pass with regression tests:

1. **Critical: interleaved header-only parallel calls lost arguments.**
   The Task 4 hold-check change to `streamToolArgsComplete` treated an empty
   first fragment as complete, closed the block with `{}`, and dropped later
   fragments. Restored `toolArgsComplete` for the hold check while keeping
   `closeOpenToolBlock` for every close path.
2. **Important: refusal overrode a legacy `function_call` tool stop.**
   The refusal branch checked only `len(msg.ToolCalls) == 0`; changed it to
   `!hasAnthropicToolUse(anth.Content)` so tool content always wins.
3. **Important: streamed `null` tool arguments passed the completeness
   guards.** `toolArgsComplete` now requires a JSON object; `null`, arrays, and
   scalars are incomplete and produce a terminal error.
4. **Important: Anthropic → Responses ignored `content_block_start.text` and
   `content_block_start.input`.** Added both fields, seeded tool arguments from
   the start event, marked them complete so later deltas cannot double-apply,
   and emitted the initial text/tool deltas.

Also strengthened the end-phase error assertion and added blocked-Gemini and
no-tools parallel cases. The plan's Task 8 code-block indentation was fixed so
`git diff --check 60f59e8..HEAD` is clean.

**Files changed:** `convert/stream.go`, `convert/openai_to_anthropic.go`,
`convert/stream_anthropic.go`, `convert/anthropic_openai_p2_test.go`, this
plan.

**Tests:** targeted RED `go test -count=1 -run
'TestP2_(InterleavedHeaderOnlyToolArguments|RefusalWithLegacyFunctionCallKeepsToolStop|NullToolArgumentsRejected|AnthropicToResponsesStartPayloads)'
./convert` failed on the base branch; GREEN after the fixes. Full
`go test -count=1 ./...`, `CGO_ENABLED=1 go test -race -count=1 ./convert`,
`go vet ./...`, and `git diff --check` all pass.

**Implementation record:**
- Date/status: 2026-11-06 / complete
- Repository/commit: `llm-api-converter` `7bf41ee` (code); docs commit follows
- Files changed: `convert/stream.go`, `convert/openai_to_anthropic.go`, `convert/stream_anthropic.go`, `convert/anthropic_openai_p2_test.go`, `docs/superpowers/plans/2026-11-06-anthropic-openai-chat-p2.md`
- Tests/commands/results: targeted RED→GREEN; full suite PASS; race PASS; vet PASS; `git diff --check 60f59e8..HEAD` clean after the docs commit
- Review findings/fixes: Critical hold-check regression restored; legacy function_call refusal stop; `null` arguments rejected; Responses start text/tool input preserved
- Deferred minors: empty-ID legacy function-call cache keys are random on both sides (pre-existing/spec-accepted); Gemini start-event tool names are still string-concatenated (Gemini names are restricted); end-phase test assertion strengthened
- Process note: task records remain in separate `docs(convert): record task N` commits to capture the code commit SHA, matching the P1 precedent; this is an intentional deviation from the "same commit" wording

---

## Final verification

After Task 8:

```bash
go test -count=1 ./...
CGO_ENABLED=1 go test -race -count=1 ./convert
go vet ./...
git diff --check
git status --short
```

Expected: all tests PASS, race clean, vet clean, no whitespace errors, and only intentional files modified.

Update `docs/reviews/2026-11-06-anthropic-openai-chat-review.md`:

- header status: `P0, P1 and P2 fixed and verified; deferred items remain unscheduled.`
- P2 table: mark each fixed item with its task commit, and mark deferred items explicitly
  (`files/audio/documents`, `message_start.usage`, legacy `ConvertSSE`, annotations).
- Add the two extra gaps (Responses truncation, Gemini stream errors) to the status.

Update `.memory/notes/llm-api-converter-magpie-eval.md` with the P2 outcome and
the deferred list, then commit the review/memory changes separately.
