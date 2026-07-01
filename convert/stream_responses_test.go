package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestResponsesStreamConverter_Text(t *testing.T) {
	sc := NewResponsesStreamConverter("deepseek-chat")

	// Stream start
	start := sc.HandleStreamStart()
	if !strings.Contains(string(start), "response.created") {
		t.Fatal("missing response.created in start")
	}
	if !strings.Contains(string(start), "response.in_progress") {
		t.Fatal("missing response.in_progress in start")
	}

	// First chunk
	c1 := `{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`
	out1, err := sc.HandleChunk([]byte(c1))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out1), "response.output_item.added") {
		t.Fatal("missing output_item.added")
	}
	if !strings.Contains(string(out1), "response.content_part.added") {
		t.Fatal("missing content_part.added")
	}
	if !strings.Contains(string(out1), `"delta":"Hello"`) {
		t.Fatal("missing text delta for 'Hello'")
	}

	// Second chunk
	c2 := `{"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`
	out2, err := sc.HandleChunk([]byte(c2))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out2), `"delta":" world"`) {
		t.Fatal("missing text delta for ' world'")
	}

	// Finish chunk
	c3 := `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	out3, err := sc.HandleChunk([]byte(c3))
	if err != nil {
		t.Fatal(err)
	}
	if out3 != nil {
		t.Logf("usage-only chunk produced output: %s", out3)
	}

	// Stream end
	end := sc.HandleStreamEnd()
	if !strings.Contains(string(end), "response.content_part.done") {
		t.Fatal("missing content_part.done")
	}
	if !strings.Contains(string(end), "response.output_item.done") {
		t.Fatal("missing output_item.done")
	}
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
	if !strings.Contains(string(end), `"output_tokens":5`) {
		t.Fatal("missing output_tokens in usage")
	}
}

func TestResponsesStreamConverter_Reasoning(t *testing.T) {
	sc := NewResponsesStreamConverter("deepseek-chat")
	sc.HandleStreamStart()

	c1 := `{"choices":[{"index":0,"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}`
	out1, err := sc.HandleChunk([]byte(c1))
	if err != nil {
		t.Fatal(err)
	}
	// Reasoning emits a separate reasoning output item (not folded into message text).
	if !strings.Contains(string(out1), "output_item.added") {
		t.Fatal("missing output_item.added for reasoning")
	}
	if !strings.Contains(string(out1), "reasoning_summary_part.added") {
		t.Fatal("missing reasoning_summary_part.added")
	}
	if !strings.Contains(string(out1), "reasoning_summary_text.delta") {
		t.Fatal("missing reasoning_summary_text.delta")
	}
	if !strings.Contains(string(out1), `"delta":"thinking..."`) {
		t.Fatal("missing reasoning content in delta")
	}

	// Followed by text — separate message item from reasoning.
	c2 := `{"choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}`
	out2, err := sc.HandleChunk([]byte(c2))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out2), "output_text.delta") {
		t.Fatal("missing output_text.delta for text content")
	}
	if !strings.Contains(string(out2), `"delta":"answer"`) {
		t.Fatal("missing text content in delta")
	}

	end := sc.HandleStreamEnd()
	if !strings.Contains(string(end), "reasoning_summary_text.done") {
		t.Fatal("missing reasoning_summary_text.done")
	}
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
}

func TestResponsesStreamConverter_ToolCalls(t *testing.T) {
	sc := NewResponsesStreamConverter("deepseek-chat")
	sc.HandleStreamStart()

	// Tool call start
	c1 := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`
	out1, err := sc.HandleChunk([]byte(c1))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out1), "output_item.added") {
		t.Fatal("missing output_item.added for function_call")
	}
	if !strings.Contains(string(out1), `"type":"function_call"`) {
		t.Fatal("missing function_call type in output item")
	}

	// Tool call arguments delta
	c2 := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"paris\"}"}}]},"finish_reason":null}]}`
	out2, err := sc.HandleChunk([]byte(c2))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out2), "function_call_arguments.delta") {
		t.Fatal("missing function_call_arguments.delta")
	}

	// Finish reason
	c3 := `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	sc.HandleChunk([]byte(c3))

	end := sc.HandleStreamEnd()
	if !strings.Contains(string(end), "function_call_arguments.done") {
		t.Fatal("missing function_call_arguments.done")
	}
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
}

func TestResponsesStreamConverter_NoOutput(t *testing.T) {
	// Empty stream should produce minimal output.
	sc := NewResponsesStreamConverter("deepseek-chat")
	sc.HandleStreamStart()
	end := sc.HandleStreamEnd()
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
	// Should still have the response completed even with no chunks.
}

func TestResponsesStreamConverter_UsageOnlyChunk(t *testing.T) {
	sc := NewResponsesStreamConverter("deepseek-chat")
	sc.HandleStreamStart()

	// Text content
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`))

	// Usage-only chunk (no choices)
	usageChunk := `{"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}`
	out, err := sc.HandleChunk([]byte(usageChunk))
	if err != nil {
		t.Fatal(err)
	}
	if out != nil {
		t.Logf("usage-only chunk unexpected output: %s", out)
	}

	// Finalize with stop
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`))
	end := sc.HandleStreamEnd()
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
}

func TestResponsesStreamConverter_MultipleToolCalls(t *testing.T) {
	sc := NewResponsesStreamConverter("deepseek-chat")
	sc.HandleStreamStart()

	// Two tool calls starting simultaneously
	c1 := `{"choices":[{"index":0,"delta":{"tool_calls":[
		{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"paris\"}"}},
		{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"UTC\"}"}}
	]},"finish_reason":null}]}`
	out1, err := sc.HandleChunk([]byte(c1))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out1), "output_item.added") {
		t.Fatal("missing output_item.added for tool calls")
	}
	// Should have 2 output_item.added events
	if strings.Count(string(out1), "event: response.output_item.added") != 2 {
		t.Fatalf("expected 2 output_item.added events, got %d", strings.Count(string(out1), "event: response.output_item.added"))
	}

	end := sc.HandleStreamEnd()
	if !strings.Contains(string(end), "function_call_arguments.done") {
		t.Fatal("missing function_call_arguments.done")
	}
}

// Regression: codex silently drops a function_call whose output_item.done item
// lacks `arguments` (ResponseItem::FunctionCall requires it, no serde default).
// The converter must carry the accumulated arguments in the done item.
func TestResponsesStreamConverter_ToolCallDoneItemHasArguments(t *testing.T) {
	sc := NewResponsesStreamConverter("deepseek-chat")
	sc.HandleStreamStart()

	// Tool call start (name + first arg fragment).
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":"}}]},"finish_reason":null}]}`))
	// Second arg fragment — exercises accumulation across deltas.
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls\"}"}}]},"finish_reason":null}]}`))
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))

	end := string(sc.HandleStreamEnd())

	// Find the output_item.done whose item is a function_call and verify it
	// carries the full accumulated arguments.
	lines := strings.Split(end, "\n")
	var doneItem *ResponsesOutputItem
	for i, ln := range lines {
		if !strings.HasPrefix(ln, "data: ") {
			continue
		}
		if !strings.Contains(ln, `"type":"response.output_item.done"`) {
			continue
		}
		var evt ResponsesStreamEvent
		if err := json.Unmarshal([]byte(ln[6:]), &evt); err != nil {
			t.Fatalf("unmarshal output_item.done event: %v\n%s", err, ln)
		}
		if evt.Item != nil && evt.Item.Type == "function_call" {
			doneItem = evt.Item
			break
		}
		_ = i
	}
	if doneItem == nil {
		t.Fatal("no function_call output_item.done event emitted")
	}
	if doneItem.Name != "exec_command" {
		t.Fatalf("done item name = %q, want exec_command", doneItem.Name)
	}
	if doneItem.Arguments == "" {
		t.Fatal("done item arguments empty — codex would silently drop this tool call")
	}
	// Arguments must equal the concatenated, canonicalized deltas.
	var got map[string]any
	if err := json.Unmarshal([]byte(doneItem.Arguments), &got); err != nil {
		t.Fatalf("done item arguments not valid JSON: %v\n%s", err, doneItem.Arguments)
	}
	if got["cmd"] != "ls" {
		t.Fatalf("done item arguments = %v, want {\"cmd\":\"ls\"}", doneItem.Arguments)
	}
}
