package convert

import (
	"strings"
	"testing"
)

func TestAnthropicStreamConverter_Text(t *testing.T) {
	asc := NewAnthropicStreamConverter("claude-sonnet-4-20250514")

	start := asc.HandleStreamStart()
	if !strings.Contains(string(start), "response.created") {
		t.Fatal("missing response.created")
	}
	if !strings.Contains(string(start), "response.in_progress") {
		t.Fatal("missing response.in_progress")
	}

	// message_start (capture usage, no output).
	msgStart := `{"type":"message_start","message":{"id":"msg_abc","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}`
	out, err := asc.HandleChunk([]byte(msgStart))
	if err != nil {
		t.Fatal(err)
	}
	if out != nil {
		t.Fatalf("message_start should produce no events, got: %s", out)
	}

	// content_block_start(text).
	cbs := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	out, err = asc.HandleChunk([]byte(cbs))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "response.output_item.added") {
		t.Fatal("missing output_item.added for text")
	}
	if !strings.Contains(string(out), "response.content_part.added") {
		t.Fatal("missing content_part.added")
	}

	// content_block_delta(text_delta).
	cbd := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`
	out, err = asc.HandleChunk([]byte(cbd))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"delta":"Hello"`) {
		t.Fatalf("missing text delta, got: %s", out)
	}

	cbd2 := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`
	out, err = asc.HandleChunk([]byte(cbd2))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"delta":" world"`) {
		t.Fatalf("missing text delta for ' world', got: %s", out)
	}

	// content_block_stop (no output).
	cbstop := `{"type":"content_block_stop","index":0}`
	out, err = asc.HandleChunk([]byte(cbstop))
	if err != nil {
		t.Fatal(err)
	}
	if out != nil {
		t.Fatalf("content_block_stop should produce no events, got: %s", out)
	}

	// message_delta (capture finish_reason + usage, no output).
	md := `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`
	out, err = asc.HandleChunk([]byte(md))
	if err != nil {
		t.Fatal(err)
	}
	if out != nil {
		t.Fatalf("message_delta should produce no events, got: %s", out)
	}

	// message_stop (no output, finalized by StreamPhaseEnd).
	ms := `{"type":"message_stop"}`
	out, err = asc.HandleChunk([]byte(ms))
	if err != nil {
		t.Fatal(err)
	}
	if out != nil {
		t.Fatalf("message_stop should produce no events, got: %s", out)
	}

	// Stream end.
	end := asc.HandleStreamEnd()
	if !strings.Contains(string(end), "response.content_part.done") {
		t.Fatal("missing content_part.done")
	}
	if !strings.Contains(string(end), "response.output_text.done") {
		t.Fatal("missing output_text.done")
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

func TestAnthropicStreamConverter_ThinkingThenText(t *testing.T) {
	asc := NewAnthropicStreamConverter("claude-sonnet-4-20250514")
	asc.HandleStreamStart()

	// message_start.
	asc.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_think","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":1}}}`))

	// content_block_start(thinking) — should emit separate reasoning output_item.
	cbs := `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"thinking step 1..."}}`
	out, err := asc.HandleChunk([]byte(cbs))
	if err != nil {
		t.Fatal(err)
	}
	// Thinking now emits a separate reasoning item (not folded into message text).
	if !strings.Contains(string(out), "output_item.added") {
		t.Fatalf("missing output_item.added for reasoning, got: %s", out)
	}
	if !strings.Contains(string(out), "reasoning_summary_part.added") {
		t.Fatalf("missing reasoning_summary_part.added, got: %s", out)
	}
	if !strings.Contains(string(out), "reasoning_summary_text.delta") {
		t.Fatalf("missing reasoning_summary_text.delta, got: %s", out)
	}
	if !strings.Contains(string(out), `"delta":"thinking step 1..."`) {
		t.Fatalf("missing thinking content in reasoning delta, got: %s", out)
	}

	// content_block_delta(thinking_delta).
	cbd := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" more thinking"}}`
	out, err = asc.HandleChunk([]byte(cbd))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"delta":" more thinking"`) {
		t.Fatalf("missing thinking delta in reasoning delta, got: %s", out)
	}

	// content_block_stop.
	asc.HandleChunk([]byte(`{"type":"content_block_stop","index":0}`))

	// content_block_start(text).
	cbs2 := `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`
	out, err = asc.HandleChunk([]byte(cbs2))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "output_item.added") {
		t.Fatal("missing output_item.added for text")
	}

	// content_block_delta(text).
	asc.HandleChunk([]byte(`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"final answer"}}`))
	asc.HandleChunk([]byte(`{"type":"content_block_stop","index":1}`))
	asc.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":20}}`))
	asc.HandleChunk([]byte(`{"type":"message_stop"}`))

	end := asc.HandleStreamEnd()
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
	// Reasoning + text = 2 output_item.done events.
	if strings.Count(string(end), "event: response.output_item.done") != 2 {
		t.Fatalf("expected 2 output_item.done (reasoning + text), got %d", strings.Count(string(end), "event: response.output_item.done"))
	}
}

func TestAnthropicStreamConverter_ToolUse(t *testing.T) {
	asc := NewAnthropicStreamConverter("claude-sonnet-4-20250514")
	asc.HandleStreamStart()
	asc.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_tool","type":"message","usage":{"input_tokens":10,"output_tokens":1}}}`))

	// content_block_start(tool_use).
	cbs := `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"get_weather","input":{}}}`
	out, err := asc.HandleChunk([]byte(cbs))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "output_item.added") {
		t.Fatal("missing output_item.added for function_call")
	}
	if !strings.Contains(string(out), `"type":"function_call"`) {
		t.Fatal("missing function_call type")
	}
	if !strings.Contains(string(out), `"name":"get_weather"`) {
		t.Fatal("missing function name")
	}

	// content_block_delta(input_json_delta).
	cbd := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\""}}`
	out, err = asc.HandleChunk([]byte(cbd))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "function_call_arguments.delta") {
		t.Fatalf("missing function_call_arguments.delta, got: %s", out)
	}

	cbd2 := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"paris\"}"}}`
	out, err = asc.HandleChunk([]byte(cbd2))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"arguments":"paris\"}`) {
		t.Fatalf("missing arguments delta, got: %s", out)
	}

	asc.HandleChunk([]byte(`{"type":"content_block_stop","index":0}`))
	asc.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`))
	asc.HandleChunk([]byte(`{"type":"message_stop"}`))

	end := asc.HandleStreamEnd()
	if !strings.Contains(string(end), "function_call_arguments.done") {
		t.Fatal("missing function_call_arguments.done")
	}
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
}

func TestAnthropicStreamConverter_NoOutput(t *testing.T) {
	// Empty stream: just message_start, message_delta, message_stop.
	asc := NewAnthropicStreamConverter("claude-sonnet-4-20250514")
	asc.HandleStreamStart()
	asc.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_empty","usage":{"input_tokens":5,"output_tokens":0}}}`))
	asc.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}`))
	asc.HandleChunk([]byte(`{"type":"message_stop"}`))

	end := asc.HandleStreamEnd()
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
	// Should not have any done events for non-existent items.
	if strings.Contains(string(end), "output_text.done") {
		t.Fatal("unexpected output_text.done with no text")
	}
}

func TestAnthropicStreamConverter_MaxTokensStop(t *testing.T) {
	asc := NewAnthropicStreamConverter("claude-sonnet-4-20250514")
	asc.HandleStreamStart()
	asc.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_max","usage":{"input_tokens":10,"output_tokens":1}}}`))
	asc.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`))
	asc.HandleChunk([]byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`))
	asc.HandleChunk([]byte(`{"type":"content_block_stop","index":0}`))
	asc.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":5}}`))

	end := asc.HandleStreamEnd()
	// max_tokens → incomplete status → response.incomplete event type.
	if !strings.Contains(string(end), "response.incomplete") {
		t.Fatalf("expected response.incomplete for max_tokens, got: %s", end)
	}
}

func TestIsAnthropicStreamEvent(t *testing.T) {
	tests := []struct {
		name string
		data string
		want bool
	}{
		{"message_start", `{"type":"message_start","message":{}}`, true},
		{"content_block_delta", `{"type":"content_block_delta","index":0}`, true},
		{"content_block_start", `{"type":"content_block_start","index":0}`, true},
		{"message_stop", `{"type":"message_stop"}`, true},
		{"ping", `{"type":"ping"}`, true},
		{"OpenAI chunk", `{"choices":[{"index":0,"delta":{"content":"hi"}}]}`, false},
		{"Chat response", `{"id":"chatcmpl","object":"chat.completion","choices":[]}`, false},
		{"empty", ``, false},
		{"invalid JSON", `not json`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isAnthropicStreamEvent([]byte(tt.data))
			if got != tt.want {
				t.Errorf("isAnthropicStreamEvent(%q) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
