package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// tidyChunk runs one OpenAI passthrough chunk and returns its JSON payload.
func tidyChunk(t *testing.T, payload string) map[string]any {
	t.Helper()
	h := NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)
	out, err := h.HandleChunk([]byte("data: " + payload + "\n\n"))
	if err != nil {
		t.Fatalf("handle chunk: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal([]byte(dataPayload(string(out))), &raw); err != nil {
		t.Fatalf("unmarshal output: %v\n%s", err, out)
	}
	return raw
}

func firstChoice(t *testing.T, raw map[string]any) map[string]any {
	t.Helper()
	choices, _ := raw["choices"].([]any)
	if len(choices) == 0 {
		t.Fatalf("no choices: %v", raw)
	}
	c, _ := choices[0].(map[string]any)
	return c
}

func TestOpenAIPassthrough_DropsEmptyReasoningContent(t *testing.T) {
	raw := tidyChunk(t, `{"model":"m","choices":[{"index":0,"delta":{"content":"hi","reasoning_content":""},"finish_reason":null}]}`)
	delta, _ := firstChoice(t, raw)["delta"].(map[string]any)
	if _, ok := delta["reasoning_content"]; ok {
		t.Fatalf("empty reasoning_content kept: %v", delta)
	}
	if delta["content"] != "hi" {
		t.Fatalf("content lost: %v", delta)
	}
}

func TestOpenAIPassthrough_DropsEmptyToolCalls(t *testing.T) {
	raw := tidyChunk(t, `{"model":"m","choices":[{"index":0,"delta":{"content":"hi","tool_calls":[]},"finish_reason":null}]}`)
	delta, _ := firstChoice(t, raw)["delta"].(map[string]any)
	if _, ok := delta["tool_calls"]; ok {
		t.Fatalf("empty tool_calls kept: %v", delta)
	}
}

func TestOpenAIPassthrough_EmptyFinishReasonBecomesNull(t *testing.T) {
	raw := tidyChunk(t, `{"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":""}]}`)
	c := firstChoice(t, raw)
	v, ok := c["finish_reason"]
	if !ok || v != nil {
		t.Fatalf("finish_reason: want null, got %v", v)
	}
}

func TestOpenAIPassthrough_ToolCallIndexFilled(t *testing.T) {
	raw := tidyChunk(t, `{"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":null}]}`)
	delta, _ := firstChoice(t, raw)["delta"].(map[string]any)
	calls, _ := delta["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("want 1 tool call, got %v", delta)
	}
	call, _ := calls[0].(map[string]any)
	if _, ok := call["index"]; !ok {
		t.Fatalf("index missing: %v", call)
	}
}

func TestOpenAIPassthrough_EmptyFunctionNameDropped(t *testing.T) {
	raw := tidyChunk(t, `{"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"","arguments":"{\"p\":1}"}}]},"finish_reason":null}]}`)
	delta, _ := firstChoice(t, raw)["delta"].(map[string]any)
	calls, _ := delta["tool_calls"].([]any)
	call, _ := calls[0].(map[string]any)
	fn, _ := call["function"].(map[string]any)
	if _, ok := fn["name"]; ok {
		t.Fatalf("empty function name kept: %v", fn)
	}
	if fn["arguments"] != `{"p":1}` {
		t.Fatalf("arguments lost: %v", fn)
	}
}

func TestOpenAIPassthrough_StopWithToolCallsBecomesToolCalls(t *testing.T) {
	raw := tidyChunk(t, `{"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{}"}}]},"finish_reason":"stop"}]}`)
	c := firstChoice(t, raw)
	if c["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason: want tool_calls, got %v", c["finish_reason"])
	}
}

func TestOpenAIPassthrough_LeavesCleanChunkAlone(t *testing.T) {
	in := `{"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`
	h := NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)
	out, _ := h.HandleChunk([]byte("data: " + in + "\n\n"))
	if !strings.Contains(string(out), in) {
		t.Fatalf("clean chunk was rewritten: %s", out)
	}
}
