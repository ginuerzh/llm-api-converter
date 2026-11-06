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

func TestP1_PassthroughEmitErrorEscapesJSON(t *testing.T) {
	want := "bad \"quote\"\nline\\backslash"
	h := NewPassthroughStreamHandler("claude", ProtocolAnthropic, false)
	out := string(h.EmitError(want))
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

func TestP1_DeltaContentShapes(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"string", `{"content":"plain"}`, "plain"},
		{"null", `{"content":null}`, ""},
		{"typed text", `{"content":[{"type":"text","text":"a"},{"type":"image","text":"b"}]}`, "a"},
		{"non-text only", `{"content":[{"type":"image_url","image_url":{}}]}`, ""},
		{"object", `{"content":{"type":"text","text":"x"}}`, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var d OpenAIDelta
			if err := json.Unmarshal([]byte(tc.body), &d); err != nil {
				t.Fatal(err)
			}
			if d.Content != tc.want {
				t.Fatalf("content = %q, want %q", d.Content, tc.want)
			}
		})
	}
}

func TestP1_DeltaUnmarshalKeepsOtherFields(t *testing.T) {
	var d OpenAIDelta
	body := `{"role":"assistant","content":"hi","reasoning_content":"think",
		"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}`
	if err := json.Unmarshal([]byte(body), &d); err != nil {
		t.Fatal(err)
	}
	if d.Role != "assistant" || d.Content != "hi" || d.ReasoningContent != "think" {
		t.Fatalf("delta = %+v", d)
	}
	if len(d.ToolCalls) != 1 || d.ToolCalls[0].ID != "a" || d.ToolCalls[0].Function.Name != "f" {
		t.Fatalf("tool calls = %+v", d.ToolCalls)
	}
}
