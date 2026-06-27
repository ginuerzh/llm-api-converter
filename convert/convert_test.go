package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

func opts() *ConvertOptions {
	return &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
}

// ---------------------------------------------------------------------------
// Passthrough tests
// ---------------------------------------------------------------------------

func TestConvert_PassthroughEmpty(t *testing.T) {
	b, err := Convert(nil, opts())
	if err != nil {
		t.Fatal(err)
	}
	if b != nil {
		t.Fatal("expected nil")
	}
}

func TestConvert_PassthroughNonJSON(t *testing.T) {
	body := []byte("not json at all")
	b, err := Convert(body, opts())
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != string(body) {
		t.Fatalf("expected %q, got %q", body, b)
	}
}

func TestConvert_PassthroughUnknownJSON(t *testing.T) {
	body := []byte(`{"foo":"bar","count":42}`)
	b, err := Convert(body, opts())
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != string(body) {
		t.Fatalf("expected %q, got %q", body, b)
	}
}

// ---------------------------------------------------------------------------
// OpenAI Request → Anthropic Request
// ---------------------------------------------------------------------------

func TestConvert_SimpleUserMessage(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if a.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("model: want claude-sonnet-4-20250514, got %q", a.Model)
	}
	if a.MaxTokens != 8192 {
		t.Fatalf("max_tokens: want 8192, got %d", a.MaxTokens)
	}
	if len(a.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(a.Messages))
	}
	if a.Messages[0].Role != "user" {
		t.Fatalf("role: want user, got %q", a.Messages[0].Role)
	}
	if len(a.Messages[0].Content) != 1 {
		t.Fatalf("want 1 content block, got %d", len(a.Messages[0].Content))
	}
	if a.Messages[0].Content[0].Type != "text" || a.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("content: want text/hello, got %s/%s", a.Messages[0].Content[0].Type, a.Messages[0].Content[0].Text)
	}
}

func TestConvert_SystemMessage(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(a.System) == 0 {
		t.Fatal("expected system field")
	}
	if a.System[0].Text != "be helpful" {
		t.Fatalf("system text: want 'be helpful', got %q", a.System[0].Text)
	}
	if len(a.Messages) != 1 {
		t.Fatalf("want 1 message (system excluded), got %d", len(a.Messages))
	}
	if a.Messages[0].Role != "user" {
		t.Fatalf("role: want user, got %q", a.Messages[0].Role)
	}
}

func TestConvert_OnlySystemMessage(t *testing.T) {
	body := `{"messages":[{"role":"system","content":"be helpful"}]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	// Anthropic requires at least one message.
	if len(a.Messages) == 0 {
		t.Fatal("expected at least one message")
	}
}

func TestConvert_TemperatureTopP(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_p":0.9}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.Temperature == nil || *a.Temperature != 0.7 {
		t.Fatalf("temperature: want 0.7, got %v", a.Temperature)
	}
	if a.TopP == nil || *a.TopP != 0.9 {
		t.Fatalf("top_p: want 0.9, got %v", a.TopP)
	}
}

func TestConvert_Stream(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.Stream == nil || !*a.Stream {
		t.Fatal("expected stream: true")
	}
}

func TestConvert_StopSequences(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stop":["stop1","stop2"]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.StopSequences) != 2 || a.StopSequences[0] != "stop1" || a.StopSequences[1] != "stop2" {
		t.Fatalf("stop_sequences: want [stop1 stop2], got %v", a.StopSequences)
	}
}

func TestConvert_StopString(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stop":"done"}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.StopSequences) != 1 || a.StopSequences[0] != "done" {
		t.Fatalf("stop_sequences: want [done], got %v", a.StopSequences)
	}
}

func TestConvert_Tools(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"what is the weather?"}],
		"tools":[{
			"type":"function",
			"function":{
				"name":"get_weather",
				"description":"Get the weather",
				"parameters":{"type":"object","properties":{"location":{"type":"string"}}}
			}
		}]
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Tools) != 1 {
		t.Fatalf("want 1 tool, got %d", len(a.Tools))
	}
	if a.Tools[0].Name != "get_weather" {
		t.Fatalf("tool name: want get_weather, got %q", a.Tools[0].Name)
	}
}

func TestConvert_AssistantToolCalls(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[
			{"role":"user","content":"weather?"},
			{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}
			]},
			{"role":"tool","tool_call_id":"call_1","content":"sunny"}
		]
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(a.Messages) != 3 {
		t.Fatalf("want 3 messages, got %d", len(a.Messages))
	}
	// Check assistant → tool_use
	if a.Messages[1].Role != "assistant" {
		t.Fatalf("messages[1] role: want assistant, got %q", a.Messages[1].Role)
	}
	if len(a.Messages[1].Content) != 2 { // empty text + tool_use
		t.Fatalf("assistant content: want 2 blocks, got %d", len(a.Messages[1].Content))
	}
	toolUseFound := false
	for _, c := range a.Messages[1].Content {
		if c.Type == "tool_use" && c.ID == "call_1" && c.Name == "get_weather" {
			toolUseFound = true
		}
	}
	if !toolUseFound {
		t.Fatal("expected tool_use block in assistant message")
	}
	// Check tool → tool_result
	if a.Messages[2].Role != "user" {
		t.Fatalf("messages[2] role: want user, got %q", a.Messages[2].Role)
	}
	if len(a.Messages[2].Content) != 1 || a.Messages[2].Content[0].Type != "tool_result" {
		t.Fatalf("messages[2] content: want tool_result block")
	}
	if a.Messages[2].Content[0].ToolUseID != "call_1" {
		t.Fatalf("tool_result tool_use_id: want call_1, got %q", a.Messages[2].Content[0].ToolUseID)
	}
}

func TestConvert_MultiPartContent(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"what is this?"},
				{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/4AAQ=="}}
			]
		}]
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(a.Messages))
	}
	if len(a.Messages[0].Content) != 2 {
		t.Fatalf("want 2 content blocks, got %d", len(a.Messages[0].Content))
	}
	// Check image block.
	imgBlock := a.Messages[0].Content[1]
	if imgBlock.Type != "image" {
		t.Fatalf("block type: want image, got %q", imgBlock.Type)
	}
	if imgBlock.Source == nil {
		t.Fatal("expected image source")
	}
	if imgBlock.Source.Type != "base64" || imgBlock.Source.MediaType != "image/jpeg" || imgBlock.Source.Data != "/9j/4AAQ==" {
		t.Fatalf("source: got type=%s media=%s data=%s", imgBlock.Source.Type, imgBlock.Source.MediaType, imgBlock.Source.Data[:8])
	}
}

// ---------------------------------------------------------------------------
// Anthropic Response → OpenAI Response
// ---------------------------------------------------------------------------

func TestConvert_AnthropicToOpenAI_Simple(t *testing.T) {
	body := `{
		"id":"msg_01abc",
		"type":"message",
		"role":"assistant",
		"content":[{"type":"text","text":"Hello!"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"stop_sequence":null,
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if !strings.HasPrefix(o.ID, "chatcmpl-") {
		t.Fatalf("id: want chatcmpl- prefix, got %q", o.ID)
	}
	if o.Object != "chat.completion" {
		t.Fatalf("object: want chat.completion, got %q", o.Object)
	}
	if len(o.Choices) != 1 {
		t.Fatalf("want 1 choice, got %d", len(o.Choices))
	}
	msg := o.Choices[0].Message
	if msg.Role != "assistant" {
		t.Fatalf("role: want assistant, got %q", msg.Role)
	}
	content, ok := msg.Content.(string)
	if !ok || content != "Hello!" {
		t.Fatalf("content: want 'Hello!', got %v", msg.Content)
	}
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason: want stop, got %v", o.Choices[0].FinishReason)
	}
	if o.Usage.PromptTokens != 10 || o.Usage.CompletionTokens != 5 || o.Usage.TotalTokens != 15 {
		t.Fatalf("usage: want 10/5/15, got %d/%d/%d", o.Usage.PromptTokens, o.Usage.CompletionTokens, o.Usage.TotalTokens)
	}
}

func TestConvert_AnthropicToOpenAI_ToolUse(t *testing.T) {
	body := `{
		"id":"msg_01",
		"type":"message",
		"role":"assistant",
		"content":[
			{"type":"text","text":"Let me check."},
			{"type":"tool_use","id":"tu_01","name":"get_weather","input":{"loc":"NYC"}}
		],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"tool_use",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason: want tool_calls, got %v", o.Choices[0].FinishReason)
	}
	msg := o.Choices[0].Message
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("want 1 tool_call, got %d", len(msg.ToolCalls))
	}
	tc := msg.ToolCalls[0]
	if tc.ID != "tu_01" || tc.Function.Name != "get_weather" {
		t.Fatalf("tool_call: want id=tu_01 name=get_weather, got id=%s name=%s", tc.ID, tc.Function.Name)
	}
	if !strings.Contains(tc.Function.Arguments, `"NYC"`) {
		t.Fatalf("arguments: expected NYC, got %s", tc.Function.Arguments)
	}
}

func TestConvert_AnthropicToOpenAI_IgnoreThinking(t *testing.T) {
	body := `{
		"id":"msg_01",
		"type":"message",
		"role":"assistant",
		"content":[
			{"type":"thinking","text":"thinking..."},
			{"type":"text","text":"Result"}
		],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	content, _ := o.Choices[0].Message.Content.(string)
	if content != "Result" {
		t.Fatalf("content: want 'Result', got %v", content)
	}
}

func TestConvert_AnthropicToOpenAI_MaxTokensFinish(t *testing.T) {
	body := `{
		"id":"msg_01",
		"type":"message",
		"role":"assistant",
		"content":[{"type":"text","text":"partial"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"max_tokens",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "length" {
		t.Fatalf("finish_reason: want length, got %v", o.Choices[0].FinishReason)
	}
}
