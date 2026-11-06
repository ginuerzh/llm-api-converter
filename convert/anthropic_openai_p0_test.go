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
