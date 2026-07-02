package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Detection tests (detectSource) — see convert_test.go for the shared detector.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Responses Request → Chat Request
// ---------------------------------------------------------------------------

func TestConvertResponsesToChat_StringInput(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{"model":"gpt-4o","input":"hello world"}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if chat.Model != "gpt-4o" {
		t.Fatalf("model: want gpt-4o, got %q", chat.Model)
	}
	if len(chat.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(chat.Messages))
	}
	if chat.Messages[0].Role != "user" {
		t.Fatalf("role: want user, got %q", chat.Messages[0].Role)
	}
	if want := "hello world"; extractTextContent(chat.Messages[0].Content) != want {
		t.Fatalf("content: want %q, got %q", want, extractTextContent(chat.Messages[0].Content))
	}
}

func TestConvertResponsesToChat_WithInstructions(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{"model":"gpt-4o","instructions":"be concise","input":"hello"}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(chat.Messages) != 2 {
		t.Fatalf("want 2 messages (system + user), got %d", len(chat.Messages))
	}
	if chat.Messages[0].Role != "system" || extractTextContent(chat.Messages[0].Content) != "be concise" {
		t.Fatalf("first message should be system/'be concise', got role=%s content=%s", chat.Messages[0].Role, extractTextContent(chat.Messages[0].Content))
	}
	if chat.Messages[1].Role != "user" {
		t.Fatalf("second message role: want user, got %q", chat.Messages[1].Role)
	}
}

func TestConvertResponsesToChat_MessagesInInput(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"model":"gpt-4o",
		"input":[
			{"type":"message","role":"user","content":"what time is it"},
			{"type":"message","role":"assistant","content":"it's noon"},
			{"type":"message","role":"user","content":"thanks"}
		]
	}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(chat.Messages) != 3 {
		t.Fatalf("want 3 messages, got %d", len(chat.Messages))
	}
	if chat.Messages[0].Role != "user" || extractTextContent(chat.Messages[0].Content) != "what time is it" {
		t.Fatalf("first message: want user/what time is it, got role=%s content=%s", chat.Messages[0].Role, extractTextContent(chat.Messages[0].Content))
	}
	if chat.Messages[1].Role != "assistant" || extractTextContent(chat.Messages[1].Content) != "it's noon" {
		t.Fatalf("second message: want assistant/it's noon, got role=%s content=%s", chat.Messages[1].Role, extractTextContent(chat.Messages[1].Content))
	}
}

func TestConvertResponsesToChat_WithFunctionCallAndOutput(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"model":"gpt-4o",
		"input":[
			{"type":"message","role":"user","content":"what is the weather in paris"},
			{"type":"function_call","id":"call_123","name":"get_weather","arguments":"{\"city\":\"paris\"}"},
			{"type":"function_call_output","call_id":"call_123","output":"\"sunny\""}
		]
	}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(chat.Messages) != 3 {
		t.Fatalf("want 3 messages, got %d", len(chat.Messages))
	}
	// user
	if chat.Messages[0].Role != "user" {
		t.Fatalf("msg[0]: want user, got %s", chat.Messages[0].Role)
	}
	// assistant with tool_calls
	if chat.Messages[1].Role != "assistant" {
		t.Fatalf("msg[1]: want assistant, got %s", chat.Messages[1].Role)
	}
	if len(chat.Messages[1].ToolCalls) != 1 {
		t.Fatalf("msg[1]: want 1 tool_call, got %d", len(chat.Messages[1].ToolCalls))
	}
	tc := chat.Messages[1].ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Fatalf("tool name: want get_weather, got %s", tc.Function.Name)
	}
	// tool result
	if chat.Messages[2].Role != "tool" {
		t.Fatalf("msg[2]: want tool, got %s", chat.Messages[2].Role)
	}
	if extractTextContent(chat.Messages[2].Content) != `"sunny"` {
		t.Fatalf("tool content: want \"sunny\", got %s", extractTextContent(chat.Messages[2].Content))
	}
}

func TestConvertResponsesToChat_StreamOption(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{"model":"gpt-4o","input":"hello","stream":true}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if chat.Stream == nil || !*chat.Stream {
		t.Fatal("chat.Stream should be true")
	}
	if chat.StreamOptions == nil {
		t.Fatal("chat.StreamOptions should be set when streaming")
	}
}

// ---------------------------------------------------------------------------
// Responses Request → Anthropic
// ---------------------------------------------------------------------------

func TestConvertResponsesToAnthropic_StringInput(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{"model":"gpt-4o","input":"hello world"}`
	b, err := ConvertResponsesToAnthropic([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(anth.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(anth.Messages))
	}
	if anth.Messages[0].Role != "user" {
		t.Fatalf("role: want user, got %q", anth.Messages[0].Role)
	}
	if len(anth.Messages[0].Content) != 1 {
		t.Fatalf("want 1 content block, got %d", len(anth.Messages[0].Content))
	}
	if anth.Messages[0].Content[0].Text != "hello world" {
		t.Fatalf("text: want 'hello world', got %q", anth.Messages[0].Content[0].Text)
	}
}

func TestConvertResponsesToAnthropic_WithInstructions(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{"model":"gpt-4o","instructions":"be concise","input":"hello"}`
	b, err := ConvertResponsesToAnthropic([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(anth.System) == 0 {
		t.Fatal("expected system field")
	}
	if anth.System[0].Text != "be concise" {
		t.Fatalf("system text: want 'be concise', got %q", anth.System[0].Text)
	}
}

func TestConvertResponsesToAnthropic_MaxTokens(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	maxTokens := 4096
	body := `{"model":"gpt-4o","input":"hello","max_output_tokens":4096}`
	b, err := ConvertResponsesToAnthropic([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if anth.MaxTokens != maxTokens {
		t.Fatalf("max_tokens: want %d, got %d", maxTokens, anth.MaxTokens)
	}
}

func TestConvertResponsesToAnthropic_WithTools(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"model":"gpt-4o",
		"input":"what is the weather in paris",
		"tools":[{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]
	}`
	b, err := ConvertResponsesToAnthropic([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(anth.Tools) != 1 {
		t.Fatalf("want 1 tool, got %d", len(anth.Tools))
	}
	if anth.Tools[0].Name != "get_weather" {
		t.Fatalf("tool name: want get_weather, got %s", anth.Tools[0].Name)
	}
	if anth.ToolChoice != nil {
		t.Fatalf("tool_choice should be nil when not explicitly set (Anthropic defaults to auto), got %+v", anth.ToolChoice)
	}
}

func TestConvertResponsesToAnthropic_NoToolsSetsToolChoiceNone(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{"model":"gpt-4o","input":"hello"}`
	b, err := ConvertResponsesToAnthropic([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if anth.ToolChoice == nil {
		t.Fatal("expected tool_choice to be set")
	}
	if anth.ToolChoice.Type != "none" {
		t.Fatalf("tool_choice: want none, got %s", anth.ToolChoice.Type)
	}
}

// ---------------------------------------------------------------------------
// Chat Response → Responses Response
// ---------------------------------------------------------------------------

func TestConvertChatToResponses_Text(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"id":"chatcmpl-abc123",
		"object":"chat.completion",
		"created":1718000000,
		"model":"deepseek-chat",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`
	b, err := ConvertChatToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if r.Object != "response" {
		t.Fatalf("object: want response, got %q", r.Object)
	}
	if r.Status != "completed" {
		t.Fatalf("status: want completed, got %q", r.Status)
	}
	if len(r.Output) != 1 {
		t.Fatalf("want 1 output item, got %d", len(r.Output))
	}
	if r.Output[0].Type != "message" {
		t.Fatalf("output[0].type: want message, got %q", r.Output[0].Type)
	}
	if len(r.Output[0].Content) != 1 {
		t.Fatalf("output[0].content: want 1 part, got %d", len(r.Output[0].Content))
	}
	if r.Output[0].Content[0].Text != "hello" {
		t.Fatalf("output[0].content[0].text: want hello, got %q", r.Output[0].Content[0].Text)
	}
	if r.Usage.InputTokens != 10 || r.Usage.OutputTokens != 5 {
		t.Fatalf("usage: want input=10 output=5, got input=%d output=%d", r.Usage.InputTokens, r.Usage.OutputTokens)
	}
	if !strings.HasPrefix(r.ID, "resp_") {
		t.Fatalf("id: want resp_ prefix, got %q", r.ID)
	}
}

func TestConvertChatToResponses_Reasoning(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"id":"chatcmpl-xyz",
		"object":"chat.completion",
		"created":1718000000,
		"model":"deepseek-chat",
		"choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning_content":"thinking..."},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}
	}`
	b, err := ConvertChatToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(r.Output) != 2 {
		t.Fatalf("want 2 output items (reasoning + message), got %d", len(r.Output))
	}
	if r.Output[0].Type != "reasoning" {
		t.Fatalf("output[0].type: want reasoning, got %q", r.Output[0].Type)
	}
	if len(r.Output[0].Summary) != 1 {
		t.Fatalf("output[0].summary: want 1 summary, got %d", len(r.Output[0].Summary))
	}
	if r.Output[0].Summary[0].Text != "thinking..." {
		t.Fatalf("output[0].summary[0].text: want 'thinking...', got %q", r.Output[0].Summary[0].Text)
	}
	if r.Output[1].Type != "message" {
		t.Fatalf("output[1].type: want message, got %q", r.Output[1].Type)
	}
	if r.Output[1].Content[0].Text != "answer" {
		t.Fatalf("output[1].content[0].text: want 'answer', got %q", r.Output[1].Content[0].Text)
	}
}

func TestConvertChatToResponses_ToolCalls(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"id":"chatcmpl-xyz",
		"object":"chat.completion",
		"created":1718000000,
		"model":"deepseek-chat",
		"choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"paris\"}"}}]},"finish_reason":"tool_calls"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`
	b, err := ConvertChatToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(r.Output) != 1 {
		t.Fatalf("want 1 output item (function_call), got %d", len(r.Output))
	}
	if r.Output[0].Type != "function_call" {
		t.Fatalf("output[0].type: want function_call, got %q", r.Output[0].Type)
	}
	if r.Output[0].Name != "get_weather" {
		t.Fatalf("output[0].name: want get_weather, got %q", r.Output[0].Name)
	}
	if r.Output[0].Arguments != `{"city":"paris"}` {
		t.Fatalf("output[0].arguments: want canonical JSON, got %q", r.Output[0].Arguments)
	}
}

func TestConvertChatToResponses_IncompleteLength(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	finish := "length"
	body := `{
		"id":"chatcmpl-xyz",
		"object":"chat.completion",
		"created":1718000000,
		"model":"deepseek-chat",
		"choices":[{"index":0,"message":{"role":"assistant","content":"partial"},"finish_reason":"length"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`
	_ = finish
	b, err := ConvertChatToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if r.Status != "incomplete" {
		t.Fatalf("status: want incomplete, got %q", r.Status)
	}
	if r.IncompleteDetails == nil {
		t.Fatal("expected incomplete_details")
	}
}

// ---------------------------------------------------------------------------
// Anthropic Response → Responses Response
// ---------------------------------------------------------------------------

func TestConvertAnthropicToResponses_Text(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	stopReason := "end_turn"
	body := `{
		"id":"msg_abc123",
		"type":"message",
		"role":"assistant",
		"content":[{"type":"text","text":"hello from claude"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	_ = stopReason
	b, err := ConvertAnthropicToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if r.Object != "response" {
		t.Fatalf("object: want response, got %q", r.Object)
	}
	if r.Status != "completed" {
		t.Fatalf("status: want completed, got %q", r.Status)
	}
	if len(r.Output) != 1 {
		t.Fatalf("want 1 output item, got %d", len(r.Output))
	}
	if r.Output[0].Type != "message" {
		t.Fatalf("output[0].type: want message, got %q", r.Output[0].Type)
	}
	if r.Output[0].Content[0].Text != "hello from claude" {
		t.Fatalf("output[0].content[0].text: want 'hello from claude', got %q", r.Output[0].Content[0].Text)
	}
}

func TestConvertAnthropicToResponses_Thinking(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"id":"msg_xyz",
		"type":"message",
		"role":"assistant",
		"content":[
			{"type":"thinking","thinking":"thought process..."},
			{"type":"text","text":"final answer"}
		],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":10}
	}`
	b, err := ConvertAnthropicToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(r.Output) != 2 {
		t.Fatalf("want 2 output items (reasoning + message), got %d", len(r.Output))
	}
	if r.Output[0].Type != "reasoning" {
		t.Fatalf("output[0].type: want reasoning, got %q", r.Output[0].Type)
	}
	if r.Output[0].Summary[0].Text != "thought process..." {
		t.Fatalf("output[0].summary[0].text: want 'thought process...', got %q", r.Output[0].Summary[0].Text)
	}
	if r.Output[1].Content[0].Text != "final answer" {
		t.Fatalf("output[1].content[0].text: want 'final answer', got %q", r.Output[1].Content[0].Text)
	}
}

func TestConvertAnthropicToResponses_ToolUse(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"id":"msg_xyz",
		"type":"message",
		"role":"assistant",
		"content":[
			{"type":"text","text":"let me check"},
			{"type":"tool_use","id":"toolu_abc","name":"get_weather","input":{"city":"paris"}}
		],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"tool_use",
		"usage":{"input_tokens":10,"output_tokens":8}
	}`
	b, err := ConvertAnthropicToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(r.Output) != 2 {
		t.Fatalf("want 2 output items (message + function_call), got %d", len(r.Output))
	}
	if r.Output[0].Type != "message" {
		t.Fatalf("output[0].type: want message, got %q", r.Output[0].Type)
	}
	if r.Output[1].Type != "function_call" {
		t.Fatalf("output[1].type: want function_call, got %q", r.Output[1].Type)
	}
	if r.Output[1].Name != "get_weather" {
		t.Fatalf("output[1].name: want get_weather, got %q", r.Output[1].Name)
	}
	if r.Output[1].Arguments != `{"city":"paris"}` {
		t.Fatalf("output[1].arguments: want canonical JSON, got %q", r.Output[1].Arguments)
	}
}

func TestConvertAnthropicToResponses_MaxTokens(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	body := `{
		"id":"msg_xyz",
		"type":"message",
		"role":"assistant",
		"content":[{"type":"text","text":"partial"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"max_tokens",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	b, err := ConvertAnthropicToResponses([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if r.Status != "incomplete" {
		t.Fatalf("status: want incomplete, got %q", r.Status)
	}
	if r.IncompleteDetails == nil {
		t.Fatal("expected incomplete_details")
	}
}

// ---------------------------------------------------------------------------
// Full Convert() path integration tests
// ---------------------------------------------------------------------------

func TestConvert_ResponsesToChatRoundTrip(t *testing.T) {
	sid := "test-roundtrip-chat"
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, SID: sid}
	opts.SessionStore = NewSessionStore()
	body := `{"model":"gpt-4o","input":"hello world","max_output_tokens":100}`
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if chat.Model != "gpt-4o" {
		t.Fatalf("model: want gpt-4o, got %q", chat.Model)
	}
	if len(chat.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(chat.Messages))
	}
	if extractTextContent(chat.Messages[0].Content) != "hello world" {
		t.Fatalf("content: want 'hello world', got %q", extractTextContent(chat.Messages[0].Content))
	}
	
}

func TestConvert_ResponsesRequestWithMessagesInput(t *testing.T) {
	body := `{
		"model":"gpt-4o",
		"input":[
			{"type":"message","role":"user","content":"hello"},
			{"type":"function_call","id":"call_1","name":"test","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"\"done\""}
		]
	}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(chat.Messages) != 3 {
		t.Fatalf("want 3 messages (user + assistant/tool_call + tool), got %d", len(chat.Messages))
	}
	if chat.Messages[0].Role != "user" {
		t.Fatalf("msg[0].role: want user, got %s", chat.Messages[0].Role)
	}
	if chat.Messages[1].Role != "assistant" || len(chat.Messages[1].ToolCalls) != 1 {
		t.Fatalf("msg[1]: want assistant with 1 tool_call, got role=%s tools=%d", chat.Messages[1].Role, len(chat.Messages[1].ToolCalls))
	}
	if chat.Messages[2].Role != "tool" {
		t.Fatalf("msg[2].role: want tool, got %s", chat.Messages[2].Role)
	}
}

func TestConvert_ResponsesToAnthropicViaModelMap(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "gpt-4o", TargetModel: "claude-sonnet-4-20250514", Protocol: "anthropic"},
	}
	body := `{"model":"gpt-4o","input":"hello"}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: mm}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if !strings.Contains(anth.Model, "claude-sonnet") {
		t.Fatalf("model: want claude-sonnet, got %q", anth.Model)
	}
	if len(anth.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(anth.Messages))
	}
	if anth.Messages[0].Content[0].Text != "hello" {
		t.Fatalf("content: want 'hello', got %q", anth.Messages[0].Content[0].Text)
	}
}

func TestConvert_ChatResponseToResponsesWithSession(t *testing.T) {
	sid := "test-session-chat-resp"

	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, SID: sid}
	opts.SessionStore = NewSessionStore()
	opts.SessionStore.Set(sid, &Session{ID: sid, IsResponses: true})
	body := `{
		"id":"chatcmpl-abc",
		"object":"chat.completion",
		"created":1718000001,
		"model":"deepseek-chat",
		"choices":[{"index":0,"message":{"role":"assistant","content":"response text"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if r.Object != "response" {
		t.Fatalf("object: want response, got %q", r.Object)
	}
	if r.Status != "completed" {
		t.Fatalf("status: want completed, got %q", r.Status)
	}
	if len(r.Output) != 1 {
		t.Fatalf("want 1 output item, got %d", len(r.Output))
	}
	if r.Output[0].Content[0].Text != "response text" {
		t.Fatalf("output[0].content[0].text: want 'response text', got %q", r.Output[0].Content[0].Text)
	}
}

func TestConvert_AnthropicResponseToResponsesWithSession(t *testing.T) {
	sid := "test-session-anth-resp"

	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, SID: sid}
	opts.SessionStore = NewSessionStore()
	opts.SessionStore.Set(sid, &Session{ID: sid, IsResponses: true})
	body := `{
		"id":"msg_xyz",
		"type":"message",
		"role":"assistant",
		"content":[{"type":"text","text":"hello from claude"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if r.Object != "response" {
		t.Fatalf("object: want response, got %q", r.Object)
	}
	if len(r.Output) != 1 {
		t.Fatalf("want 1 output item, got %d", len(r.Output))
	}
	if r.Output[0].Content[0].Text != "hello from claude" {
		t.Fatalf("output[0].content[0].text: want 'hello from claude', got %q", r.Output[0].Content[0].Text)
	}
}

func TestConvert_PassthroughResponsesResponse(t *testing.T) {
	sid := "test-session-ps"

	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, SID: sid}
	opts.SessionStore = NewSessionStore()
	opts.SessionStore.Set(sid, &Session{ID: sid, IsResponses: true})
	body := `{"object":"response","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"already responses"}]}]}`
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if r.Object != "response" {
		t.Fatalf("object: want response, got %q", r.Object)
	}
	if len(r.Output) != 1 {
		t.Fatalf("want 1 output item, got %d", len(r.Output))
	}
}
