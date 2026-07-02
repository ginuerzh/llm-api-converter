package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func opts() *ConvertOptions {
	return &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, URI: "/v1/chat/completions", Direction: "request"}
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
	b, err := Convert(body, &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192})
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
	if a.Model != "gpt-4" {
		t.Fatalf("model: want gpt-4, got %q", a.Model)
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

func TestConvert_FinishReasonNilReturnsDefault(t *testing.T) {
	body := `{
		"id":"msg_01",
		"type":"message",
		"role":"assistant",
		"content":[{"type":"text","text":"hello"}],
		"model":"claude-sonnet-4-20250514",
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
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason: want 'stop' got %v", o.Choices[0].FinishReason)
	}
}

func TestConvert_ContentFilterFinishReason(t *testing.T) {
	body := `{
		"id":"msg_01",
		"type":"message",
		"role":"assistant",
		"content":[{"type":"text","text":"hello"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"content_filter",
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
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason: want 'stop' got %v", o.Choices[0].FinishReason)
	}
}

// ---------------------------------------------------------------------------
// Thinking / Reasoning Content Round-Trip
// ---------------------------------------------------------------------------

func TestConvert_ThinkingRoundTrip(t *testing.T) {
	// Step 1: Anthropic Response with thinking → OpenAI Response with reasoning_content.
	anthBody := `{
		"id":"msg_01",
		"type":"message",
		"role":"assistant",
		"content":[
			{"type":"thinking","thinking":"hidden reasoning..."},
			{"type":"text","text":"visible answer"}
		],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`
	b, err := Convert([]byte(anthBody), opts())
	if err != nil {
		t.Fatal(err)
	}
	var openAIResp OpenAIChatResponse
	if err := json.Unmarshal(b, &openAIResp); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	msg := openAIResp.Choices[0].Message
	if msg.ReasoningContent != "hidden reasoning..." {
		t.Fatalf("reasoning_content: want 'hidden reasoning...', got %q", msg.ReasoningContent)
	}
	content, _ := msg.Content.(string)
	if content != "visible answer" {
		t.Fatalf("content: want 'visible answer', got %v", content)
	}

	// Step 2: OpenAI Request with reasoning_content → Anthropic Request with thinking.
	openAIReq := `{
		"model":"deepseek-v4",
		"messages":[
			{"role":"user","content":"hello"},
			{"role":"assistant","reasoning_content":"hidden reasoning...","content":"visible answer"}
		]
	}`
	b2, err := Convert([]byte(openAIReq), opts())
	if err != nil {
		t.Fatal(err)
	}
	var anthReq AnthropicRequest
	if err := json.Unmarshal(b2, &anthReq); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b2)
	}
	if len(anthReq.Messages) != 2 {
		t.Fatalf("want 2 messages, got %d", len(anthReq.Messages))
	}
	assistantMsg := anthReq.Messages[1]
	if assistantMsg.Role != "assistant" {
		t.Fatalf("messages[1] role: want assistant, got %q", assistantMsg.Role)
	}
	if len(assistantMsg.Content) != 2 {
		t.Fatalf("assistant content: want 2 blocks (thinking + text), got %d", len(assistantMsg.Content))
	}
	if assistantMsg.Content[0].Type != "thinking" || assistantMsg.Content[0].Thinking != "hidden reasoning..." {
		t.Fatalf("expect thinking block with thinking='hidden reasoning...', got type=%s thinking=%q",
			assistantMsg.Content[0].Type, assistantMsg.Content[0].Thinking)
	}
	if assistantMsg.Content[1].Type != "text" || assistantMsg.Content[1].Text != "visible answer" {
		t.Fatalf("expect text block with 'visible answer', got type=%s text=%q",
			assistantMsg.Content[1].Type, assistantMsg.Content[1].Text)
	}
}

// ---------------------------------------------------------------------------
// Tool Choice Mapping
// ---------------------------------------------------------------------------

func TestConvert_ToolChoiceAuto(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
		"tool_choice":"auto"
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice == nil || a.ToolChoice.Type != "auto" {
		t.Fatalf("tool_choice: want auto, got %+v", a.ToolChoice)
	}
}

func TestConvert_ToolChoiceNone(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
		"tool_choice":"none"
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice == nil || a.ToolChoice.Type != "none" {
		t.Fatalf("tool_choice: want none, got %+v", a.ToolChoice)
	}
}

func TestConvert_ToolChoiceRequired(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
		"tool_choice":"required"
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice == nil || a.ToolChoice.Type != "any" {
		t.Fatalf("tool_choice: want any, got %+v", a.ToolChoice)
	}
}

func TestConvert_ToolChoiceSpecificFunction(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
		"tool_choice":{"type":"function","function":{"name":"f"}}
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice == nil || a.ToolChoice.Type != "tool" || a.ToolChoice.Name != "f" {
		t.Fatalf("tool_choice: want type=tool name=f, got %+v", a.ToolChoice)
	}
}

func TestConvert_NoToolsSetsToolChoiceNone(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"hi"}]
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice == nil || a.ToolChoice.Type != "none" {
		t.Fatalf("tool_choice: want none (no tools), got %+v", a.ToolChoice)
	}
}

// ---------------------------------------------------------------------------
// Tool Sorting and Schema Enforcement
// ---------------------------------------------------------------------------

func TestConvert_ToolsSortedByName(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[
			{"type":"function","function":{"name":"z_tool","parameters":{}}},
			{"type":"function","function":{"name":"a_tool","parameters":{}}}
		]
	}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.Tools) != 2 {
		t.Fatalf("want 2 tools, got %d", len(a.Tools))
	}
	if a.Tools[0].Name != "a_tool" || a.Tools[1].Name != "z_tool" {
		t.Fatalf("tools not sorted by name: got %q, %q", a.Tools[0].Name, a.Tools[1].Name)
	}
}

func TestConvert_InputSchemaEnforcesObjectType(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"properties":{"x":{"type":"string"}}}}}]
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
	schema, ok := a.Tools[0].InputSchema.(map[string]any)
	if !ok {
		t.Fatalf("input_schema is not a map: %T", a.Tools[0].InputSchema)
	}
	if schema["type"] != "object" {
		t.Fatalf("input_schema type: want 'object', got %q", schema["type"])
	}
}

// ---------------------------------------------------------------------------
// Legacy function_call
// ---------------------------------------------------------------------------

func TestConvert_FunctionCallToToolUse(t *testing.T) {
	body := `{
		"model":"gpt-4",
		"messages":[
			{"role":"user","content":"weather?"},
			{"role":"assistant","content":null,"function_call":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}
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
	if len(a.Messages) != 2 {
		t.Fatalf("want 2 messages, got %d", len(a.Messages))
	}
	if a.Messages[1].Role != "assistant" {
		t.Fatalf("messages[1] role: want assistant, got %q", a.Messages[1].Role)
	}
	toolUseFound := false
	for _, c := range a.Messages[1].Content {
		if c.Type == "tool_use" && c.Name == "get_weather" {
			toolUseFound = true
		}
	}
	if !toolUseFound {
		t.Fatal("expected tool_use block for function_call")
	}
}

// ---------------------------------------------------------------------------
// SSE event handling
// ---------------------------------------------------------------------------

func TestIsSSE(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want bool
	}{
		{"empty", nil, false},
		{"data field", []byte("data: hello"), true},
		{"event field", []byte("event: message_start"), true},
		{"id field", []byte("id: msg_001"), true},
		{"data with leading whitespace", []byte("  data: hello"), false},
		{"plain JSON", []byte(`{"model":"gpt-4"}`), false},
		{"plain text", []byte("hello world"), false},
		{"retry only (not a prefix match)", []byte("retry: 3000"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isSSE(tt.data); got != tt.want {
				t.Errorf("isSSE() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseSSEEvent_Basic(t *testing.T) {
	evt := parseSSEEvent([]byte("data: {\"key\":\"val\"}"))
	if evt.Data != `{"key":"val"}` {
		t.Fatalf("Data = %q, want %q", evt.Data, `{"key":"val"}`)
	}
	if evt.Event != "" {
		t.Fatalf("Event = %q, want empty", evt.Event)
	}
}

func TestParseSSEEvent_WithEventType(t *testing.T) {
	evt := parseSSEEvent([]byte("event: message_start\ndata: {\"type\":\"message_start\"}"))
	if evt.Event != "message_start" {
		t.Fatalf("Event = %q, want %q", evt.Event, "message_start")
	}
	if evt.Data != `{"type":"message_start"}` {
		t.Fatalf("Data = %q, want %q", evt.Data, `{"type":"message_start"}`)
	}
}

func TestParseSSEEvent_MultiLineData(t *testing.T) {
	evt := parseSSEEvent([]byte("data: line1\ndata: line2"))
	if evt.Data != "line1\nline2" {
		t.Fatalf("Data = %q, want %q", evt.Data, "line1\nline2")
	}
}

func TestParseSSEEvent_WithID(t *testing.T) {
	evt := parseSSEEvent([]byte("id: msg_001\ndata: {}"))
	if evt.ID != "msg_001" {
		t.Fatalf("ID = %q, want %q", evt.ID, "msg_001")
	}
	if evt.Data != "{}" {
		t.Fatalf("Data = %q, want %q", evt.Data, "{}")
	}
}

func TestParseSSEEvent_WithRetry(t *testing.T) {
	evt := parseSSEEvent([]byte("retry: 3000\ndata: {}"))
	if evt.Retry != 3000 {
		t.Fatalf("Retry = %d, want %d", evt.Retry, 3000)
	}
}

func TestParseSSEEvent_NoSpaceAfterColon(t *testing.T) {
	evt := parseSSEEvent([]byte("data:{}"))
	if evt.Data != "{}" {
		t.Fatalf("Data = %q, want %q", evt.Data, "{}")
	}
}

func TestParseSSEEvent_CommentsIgnored(t *testing.T) {
	// Lines starting with just ":" are SSE comments.
	evt := parseSSEEvent([]byte(":comment\ndata: {}"))
	if evt.Data != "{}" {
		t.Fatalf("Data = %q, want %q", evt.Data, "{}")
	}
}

func TestReconstructSSEEvent(t *testing.T) {
	evt := &SSEEvent{Event: "message_stop", Data: "{}", ID: "msg_01", Retry: 2000}
	raw := reconstructSSEEvent(evt)
	out := string(raw)
	if !strings.Contains(out, "id: msg_01") {
		t.Errorf("missing id field: %s", out)
	}
	if !strings.Contains(out, "event: message_stop") {
		t.Errorf("missing event field: %s", out)
	}
	if !strings.Contains(out, "data: {}") {
		t.Errorf("missing data field: %s", out)
	}
	if !strings.Contains(out, "retry: 2000") {
		t.Errorf("missing retry field: %s", out)
	}
}

func TestReconstructSSEEvent_Order(t *testing.T) {
	// Ensure order is: id, event, data*, retry.
	evt := &SSEEvent{ID: "1", Event: "e", Data: "d", Retry: 100}
	raw := string(reconstructSSEEvent(evt))
	want := "id: 1\nevent: e\ndata: d\nretry: 100"
	if raw != want {
		t.Errorf("order/format:\ngot:  %q\nwant: %q", raw, want)
	}
}

func TestConvertSSE_PassthroughNonJSONData(t *testing.T) {
	raw := []byte("data: plain text")
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != string(raw) {
		t.Fatalf("got %q, want %q", b, raw)
	}
}

func TestConvertSSE_DONEMarker(t *testing.T) {
	raw := []byte("data: [DONE]")
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.Contains(out, "message_stop") {
		t.Fatalf("expected message_stop event, got: %s", out)
	}
}

func TestConvertSSE_Empty(t *testing.T) {
	b, err := ConvertSSE(nil, opts())
	if err != nil {
		t.Fatal(err)
	}
	if b != nil {
		t.Fatal("expected nil")
	}
}

func TestConvertSSE_OpenAIRequest(t *testing.T) {
	raw := []byte(`data: {"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.HasPrefix(out, "data: ") {
		t.Fatalf("SSE framing missing: %s", out)
	}
	// The data portion should be an Anthropic request.
	jsonPart := out[6:] // strip "data: "
	var acr AnthropicRequest
	if err := json.Unmarshal([]byte(jsonPart), &acr); err != nil {
		t.Fatalf("unmarshal Anthropic request: %v\nbody: %s", err, jsonPart)
	}
	if acr.Model != "gpt-4" {
		t.Fatalf("model: want gpt-4, got %q", acr.Model)
	}
}

func TestConvertSSE_AnthropicResponse(t *testing.T) {
	raw := []byte(`data: {"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.HasPrefix(out, "data: ") {
		t.Fatalf("SSE framing missing: %s", out)
	}
	jsonPart := out[6:]
	var ocr OpenAIChatResponse
	if err := json.Unmarshal([]byte(jsonPart), &ocr); err != nil {
		t.Fatalf("unmarshal OpenAI response: %v\nbody: %s", err, jsonPart)
	}
	if len(ocr.Choices) != 1 {
		t.Fatalf("want 1 choice, got %d", len(ocr.Choices))
	}
}

func TestConvertSSE_PreservesEventAndID(t *testing.T) {
	raw := []byte("event: message_start\nid: msg_01\ndata: {\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Hi\"}],\"model\":\"claude\",\"stop_reason\":\"end_turn\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}")
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.Contains(out, "event: message_start") {
		t.Errorf("missing event field: %s", out)
	}
	if !strings.Contains(out, "id: msg_01") {
		t.Errorf("missing id field: %s", out)
	}
	if !strings.HasPrefix(out, "id:") {
		t.Errorf("should start with id: (alphabetical order), got: %s", out)
	}
}

func TestConvertViaAutoDetection_SSE(t *testing.T) {
	// Convert() should auto-detect SSE framing.
	raw := []byte(`data: {"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`)
	b, err := Convert(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(b), "data: ") {
		t.Fatalf("expected SSE output, got: %s", b)
	}
}

func TestConvertViaAutoDetection_PlainJSON(t *testing.T) {
	// Plain JSON should still work (regression check).
	raw := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`)
	b, err := Convert(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
}

// ---------------------------------------------------------------------------
// Detection tests
// ---------------------------------------------------------------------------

func TestDetectSource_AnthropicRequestMarkers(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    Protocol
	}{
		{
			name: "anthropic with thinking block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"..."}]}]}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with tool_use block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"tu1","name":"f","input":{}}]}]}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with tool_result block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"ok"}]}]}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with image block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"aa"}}]}]}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with top-level system",
			json: `{"model":"claude","max_tokens":8192,"system":[{"type":"text","text":"be helpful"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with tool_choice.any",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with tool_choice.tool",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tool_choice":{"type":"tool","name":"f"}}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with stop_sequences",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stop_sequences":["stop"]}`,
			want: ProtocolAnthropic,
		},
		{
			name: "anthropic with flat tools (name+input_schema)",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}]}`,
			want: ProtocolAnthropic,
		},
		{
			name: "openai with frequency_penalty",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.5}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with stop field",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stop":"done"}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with tool role in messages",
			json: `{"model":"gpt-4","messages":[{"role":"tool","content":"result"}]}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with system role in messages",
			json: `{"model":"gpt-4","messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}]}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai tool_choice auto string + function tools",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto"}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with n field",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"n":2}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with presence_penalty",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"presence_penalty":0.5}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with logit_bias",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"logit_bias":{"123":-1}}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with response_format",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai with seed",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"seed":42}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai function role in messages",
			json: `{"model":"gpt-4","messages":[{"role":"function","name":"f","content":"result"}]}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "openai tool_calls in message",
			json: `{"model":"gpt-4","messages":[{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`,
			want: ProtocolOpenAIChat,
		},
		// Minimal request without distinguishing features -> Unknown (not guessed).
		{
			name: "minimal request with max_tokens - no positive markers",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			want: ProtocolUnknown,
		},
		{
			name: "minimal openai request - no positive markers",
			json: `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			want: ProtocolUnknown,
		},
		{
			name: "unknown JSON without features",
			json: `{"foo":"bar"}`,
			want: ProtocolUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal([]byte(tt.json), &raw); err != nil {
				t.Fatal(err)
			}
			if got := detectSource(raw); got != tt.want {
				t.Errorf("detectSource() = %v (%s), want %v (%s)\nbody: %s", got, got.String(), tt.want, tt.want.String(), tt.json)
			}
		})
	}
}

func TestDetectSource_OpenAIResponse(t *testing.T) {
	tests := []struct {
		name string
		json string
		want Protocol
	}{
		{
			name: "full non-streaming response",
			json: `{"id":"chatcmpl-xyz","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "streaming delta chunk",
			json: `{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
			want: ProtocolOpenAIChat,
		},
		{
			name: "anthropic response",
			json: `{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
			want: ProtocolAnthropic,
		},
		{
			name: "responses response",
			json: `{"id":"resp_01","object":"response","status":"completed","output":[]}`,
			want: ProtocolOpenAIResponses,
		},
		{
			name: "unknown JSON without features",
			json: `{"foo":"bar"}`,
			want: ProtocolUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal([]byte(tt.json), &raw); err != nil {
				t.Fatal(err)
			}
			if got := detectSource(raw); got != tt.want {
				t.Errorf("detectSource() = %v, want %v\nbody: %s", got, tt.want, tt.json)
			}
		})
	}
}

func TestIsOpenAIStreamChunk_DetectsCorrectly(t *testing.T) {
	tests := []struct {
		name string
		data string
		want bool
	}{
		{
			name: "delta with content",
			data: `{"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			want: true,
		},
		{
			name: "delta with finish_reason",
			data: `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			want: true,
		},
		{
			name: "non-streaming response (has message not delta)",
			data: `{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`,
			want: false, // isOpenAIStreamChunk checks for delta specifically
		},
		{
			name: "not a chunk",
			data: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isOpenAIStreamChunk([]byte(tt.data)); got != tt.want {
				t.Errorf("isOpenAIStreamChunk() = %v, want %v\ndata: %s", got, tt.want, tt.data)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Detection edge case regression tests
// ---------------------------------------------------------------------------

func TestDetectSource_MinimalRequest_PassthroughWithoutURI(t *testing.T) {
	// A minimal request with no distinguishing features returns Unknown.
	body := `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	var raw map[string]any
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatal(err)
	}
	if got := detectSource(raw); got != ProtocolUnknown {
		t.Fatalf("detectSource() = %v, want Unknown (no positive markers)", got)
	}
}

func TestConvert_MinimalOpenAIReqWithURI(t *testing.T) {
	// Minimal OpenAI request with URI fallback should detect and convert.
	body := `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 8192,
		URI:       "/v1/chat/completions",
		Direction: "request",
	}
	out, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(out, &a); err != nil {
		t.Fatalf("expected valid Anthropic request output, got: %s", out)
	}
	if a.Model != "gpt-4" {
		t.Fatalf("model: want gpt-4, got %q", a.Model)
	}
}

func TestConvert_MinimalAnthropicReqWithURI(t *testing.T) {
	// Minimal Anthropic request with URI fallback should detect and convert.
	body := `{"model":"my-custom-model","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 8192,
		URI:       "/v1/messages",
		Direction: "request",
	}
	out, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(out, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, out)
	}
	if len(o.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(o.Messages))
	}
	if o.Messages[0].Role != "user" {
		t.Fatalf("role: want user, got %q", o.Messages[0].Role)
	}
}

func TestDetectSource_DeepSeekModelWithNoMarkers(t *testing.T) {
	// deepseek model with only max_tokens + array content = Unknown.
	body := `{"model":"deepseek-chat","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	var raw map[string]any
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatal(err)
	}
	if got := detectSource(raw); got != ProtocolUnknown {
		t.Fatalf("detectSource() = %v, want Unknown", got)
	}
}

func TestDetectSource_GLMModelWithNoMarkers(t *testing.T) {
	// glm model with only max_tokens + array content = Unknown.
	body := `{"model":"glm-4.6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	var raw map[string]any
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatal(err)
	}
	if got := detectSource(raw); got != ProtocolUnknown {
		t.Fatalf("detectSource() = %v, want Unknown", got)
	}
}

// ---------------------------------------------------------------------------
// Anthropic Request → OpenAI Request
// ---------------------------------------------------------------------------

func TestConvert_AnthropicReqToOpenAIReq_SimpleUser(t *testing.T) {
	body := `{"model":"claude-sonnet-4-20250514","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if o.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("model: want deepseek-chat, got %q", o.Model)
	}
	if len(o.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(o.Messages))
	}
	if o.Messages[0].Role != "user" {
		t.Fatalf("role: want user, got %q", o.Messages[0].Role)
	}
	content, ok := o.Messages[0].Content.(string)
	if !ok || content != "hello" {
		t.Fatalf("content: want 'hello', got %v", o.Messages[0].Content)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_WithSystem(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"system":[{"type":"text","text":"be helpful"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if len(o.Messages) != 2 {
		t.Fatalf("want 2 messages (system+user), got %d", len(o.Messages))
	}
	if o.Messages[0].Role != "system" {
		t.Fatalf("messages[0] role: want system, got %q", o.Messages[0].Role)
	}
	content, ok := o.Messages[0].Content.(string)
	if !ok || content != "be helpful" {
		t.Fatalf("system content: want 'be helpful', got %v", o.Messages[0].Content)
	}
	if o.Messages[1].Role != "user" {
		t.Fatalf("messages[1] role: want user, got %q", o.Messages[1].Role)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_Thinking(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":[{"type":"thinking","thinking":"thinking steps..."},{"type":"text","text":"answer"}]}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(o.Messages) != 2 {
		t.Fatalf("want 2 messages, got %d", len(o.Messages))
	}
	if o.Messages[1].Role != "assistant" {
		t.Fatalf("messages[1] role: want assistant, got %q", o.Messages[1].Role)
	}
	if o.Messages[1].ReasoningContent != "thinking steps..." {
		t.Fatalf("reasoning_content: want 'thinking steps...', got %q", o.Messages[1].ReasoningContent)
	}
	content, _ := o.Messages[1].Content.(string)
	if content != "answer" {
		t.Fatalf("content: want 'answer', got %v", o.Messages[1].Content)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_ToolUse(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"weather?"}]},{"role":"assistant","content":[{"type":"text","text":""},{"type":"tool_use","id":"tu01","name":"get_weather","input":{"loc":"NYC"}}]}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(o.Messages) != 2 {
		t.Fatalf("want 2 messages, got %d", len(o.Messages))
	}
	assistant := o.Messages[1]
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("want 1 tool_call, got %d", len(assistant.ToolCalls))
	}
	tc := assistant.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Fatalf("tool name: want get_weather, got %q", tc.Function.Name)
	}
	if !strings.Contains(tc.Function.Arguments, `"NYC"`) {
		t.Fatalf("arguments: want NYC, got %s", tc.Function.Arguments)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_ToolResult(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu01","content":"sunny"}]}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	// Orphan tool result (no preceding assistant tool_use) is sanitized to user text.
	if len(o.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(o.Messages))
	}
	if o.Messages[0].Role != "user" {
		t.Fatalf("role: want user (orphan tool converted), got %q", o.Messages[0].Role)
	}
	content, ok := o.Messages[0].Content.(string)
	if !ok || content != "Tool result without a matching tool call (tu01):\nsunny" {
		t.Fatalf("content: want descriptive user text, got %v", o.Messages[0].Content)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_MultiToolResult(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"assistant","content":[{"type":"text","text":"Let me check"},{"type":"tool_use","id":"tu01","name":"get_weather","input":{"loc":"NYC"}},{"type":"tool_use","id":"tu02","name":"get_temp","input":{"loc":"NYC"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu01","content":"sunny"},{"type":"tool_result","tool_use_id":"tu02","content":"72°F"},{"type":"text","text":"Based on that, what should I wear?"}]}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	// Expected messages: assistant (tool_calls) → tool (tu01) → tool (tu02) → user (text)
	if len(o.Messages) != 4 {
		t.Fatalf("want 4 messages, got %d", len(o.Messages))
	}
	// Message 0: assistant with 2 tool_calls
	if o.Messages[0].Role != "assistant" {
		t.Fatalf("msg[0] role: want assistant, got %q", o.Messages[0].Role)
	}
	if len(o.Messages[0].ToolCalls) != 2 {
		t.Fatalf("msg[0] tool_calls: want 2, got %d", len(o.Messages[0].ToolCalls))
	}
	if o.Messages[0].ToolCalls[0].ID != "tu01" || o.Messages[0].ToolCalls[1].ID != "tu02" {
		t.Fatalf("msg[0] tool_call IDs: want tu01,tu02 got %q,%q",
			o.Messages[0].ToolCalls[0].ID, o.Messages[0].ToolCalls[1].ID)
	}
	// Message 1: tool response for tu01
	if o.Messages[1].Role != "tool" {
		t.Fatalf("msg[1] role: want tool, got %q", o.Messages[1].Role)
	}
	if o.Messages[1].ToolCallID != "tu01" {
		t.Fatalf("msg[1] tool_call_id: want tu01, got %q", o.Messages[1].ToolCallID)
	}
	// Message 2: tool response for tu02
	if o.Messages[2].Role != "tool" {
		t.Fatalf("msg[2] role: want tool, got %q", o.Messages[2].Role)
	}
	if o.Messages[2].ToolCallID != "tu02" {
		t.Fatalf("msg[2] tool_call_id: want tu02, got %q", o.Messages[2].ToolCallID)
	}
	// Message 3: user text after tool results
	if o.Messages[3].Role != "user" {
		t.Fatalf("msg[3] role: want user, got %q", o.Messages[3].Role)
	}
	content, ok := o.Messages[3].Content.(string)
	if !ok || content != "Based on that, what should I wear?" {
		t.Fatalf("msg[3] content: want 'Based on that...', got %v", o.Messages[3].Content)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_Tools(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"get_weather","description":"Get the weather","input_schema":{"type":"object","properties":{"loc":{"type":"string"}}}}],"tool_choice":{"type":"auto"}}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(o.Tools) != 1 {
		t.Fatalf("want 1 tool, got %d", len(o.Tools))
	}
	if o.Tools[0].Type != "function" {
		t.Fatalf("tool type: want function, got %q", o.Tools[0].Type)
	}
	if o.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("tool function name: want get_weather, got %q", o.Tools[0].Function.Name)
	}
	if o.ToolChoice != "auto" {
		t.Fatalf("tool_choice: want auto, got %v", o.ToolChoice)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_ToolChoiceAny(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.ToolChoice != "required" {
		t.Fatalf("tool_choice: want required (non-DeepSeek), got %v", o.ToolChoice)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_ToolChoiceTool(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"f"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	tc, ok := o.ToolChoice.(map[string]any)
	if !ok {
		t.Fatalf("tool_choice: expected object (non-DeepSeek), got %T", o.ToolChoice)
	}
	if tc["type"] != "function" {
		t.Fatalf("tool_choice.type: want function, got %q", tc["type"])
	}
}

func TestConvert_AnthropicReqToOpenAIReq_StopSequences(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stop_sequences":["stop1","stop2"]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	stop, ok := o.Stop.([]any)
	if !ok || len(stop) != 2 {
		t.Fatalf("stop: want [stop1 stop2], got %v", o.Stop)
	}
	if stop[0].(string) != "stop1" || stop[1].(string) != "stop2" {
		t.Fatalf("stop values: want stop1, stop2, got %v", stop)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_Stream(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stream":true}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Stream == nil || !*o.Stream {
		t.Fatal("expected stream: true")
	}
}

func TestConvert_AnthropicReqToOpenAIReq_MultiPartContent(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"what is this?"},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"/9j/4AAQ=="}}]}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	parts, ok := o.Messages[0].Content.([]any)
	if !ok {
		t.Fatalf("content: expected array, got %T", o.Messages[0].Content)
	}
	if len(parts) != 2 {
		t.Fatalf("want 2 content parts, got %d", len(parts))
	}
	if txt, _ := parts[0].(map[string]any); txt["type"] != "text" || txt["text"] != "what is this?" {
		t.Fatalf("first part: want text/what is this?, got %v", parts[0])
	}
	if img, _ := parts[1].(map[string]any); img["type"] != "image_url" {
		t.Fatalf("second part: want image_url, got %v", parts[1])
	}
}

// ---------------------------------------------------------------------------
// OpenAI Response → Anthropic Response
// ---------------------------------------------------------------------------

func TestConvert_OpenAIRespToAnthropicResp_Simple(t *testing.T) {
	body := `{"id":"chatcmpl-xyz","object":"chat.completion","created":123,"model":"deepseek","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicResponse
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if a.Type != "message" {
		t.Fatalf("type: want message, got %q", a.Type)
	}
	if !strings.HasPrefix(a.ID, "msg_") {
		t.Fatalf("id: want msg_ prefix, got %q", a.ID)
	}
	if len(a.Content) != 1 {
		t.Fatalf("want 1 content block, got %d", len(a.Content))
	}
	if a.Content[0].Type != "text" || a.Content[0].Text != "Hello!" {
		t.Fatalf("content: want text/Hello!, got %s/%s", a.Content[0].Type, a.Content[0].Text)
	}
	if a.StopReason == nil || *a.StopReason != "end_turn" {
		t.Fatalf("stop_reason: want end_turn, got %v", a.StopReason)
	}
	if a.Usage.InputTokens != 10 || a.Usage.OutputTokens != 5 {
		t.Fatalf("usage: want 10/5, got %d/%d", a.Usage.InputTokens, a.Usage.OutputTokens)
	}
}

func TestConvert_OpenAIRespToAnthropicResp_ToolCalls(t *testing.T) {
	body := `{"id":"chatcmpl-xyz","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Let me check.","tool_calls":[{"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicResponse
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if a.StopReason == nil || *a.StopReason != "tool_use" {
		t.Fatalf("stop_reason: want tool_use, got %v", a.StopReason)
	}
	toolUseFound := false
	for _, block := range a.Content {
		if block.Type == "tool_use" && block.Name == "get_weather" {
			toolUseFound = true
			if block.ID != "call_abc" {
				t.Fatalf("tool_use id: want call_abc, got %q", block.ID)
			}
		}
	}
	if !toolUseFound {
		t.Fatal("expected tool_use block")
	}
}

func TestConvert_OpenAIRespToAnthropicResp_Reasoning(t *testing.T) {
	body := `{"id":"chatcmpl-xyz","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"thinking steps...","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicResponse
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(a.Content) != 2 {
		t.Fatalf("want 2 content blocks (thinking + text), got %d", len(a.Content))
	}
	if a.Content[0].Type != "thinking" || a.Content[0].Thinking != "thinking steps..." {
		t.Fatalf("first block: want thinking/'thinking steps...', got type=%s thinking=%q",
			a.Content[0].Type, a.Content[0].Thinking)
	}
	if a.Content[1].Type != "text" || a.Content[1].Text != "answer" {
		t.Fatalf("second block: want text/'answer', got type=%s text=%q",
			a.Content[1].Type, a.Content[1].Text)
	}
}

func TestConvert_OpenAIRespToAnthropicResp_FinishReasonLength(t *testing.T) {
	body := `{"id":"chatcmpl-xyz","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"partial"},"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicResponse
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.StopReason == nil || *a.StopReason != "max_tokens" {
		t.Fatalf("stop_reason: want max_tokens, got %v", a.StopReason)
	}
}

// ---------------------------------------------------------------------------
// Anthropic Request with string content (deprecated format)
// ---------------------------------------------------------------------------

func TestConvert_AnthropicReqToOpenAIReq_StringContent(t *testing.T) {
	// Anthropic API still accepts content as a plain string (deprecated format).
	body := `{"model":"claude-sonnet-4-20250514","max_tokens":8192,"messages":[{"role":"user","content":"hello"}]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if o.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("model: want deepseek-chat, got %q", o.Model)
	}
	if len(o.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(o.Messages))
	}
	if o.Messages[0].Role != "user" {
		t.Fatalf("role: want user, got %q", o.Messages[0].Role)
	}
	content, ok := o.Messages[0].Content.(string)
	if !ok || content != "hello" {
		t.Fatalf("content: want 'hello', got %v", o.Messages[0].Content)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_MixedStringAndArrayContent(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[
		{"role":"user","content":"hello"},
		{"role":"assistant","content":"hi there"},
		{"role":"user","content":[{"type":"text","text":"how are you?"}]}
	]}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(o.Messages) != 3 {
		t.Fatalf("want 3 messages, got %d", len(o.Messages))
	}
	// First: string content "hello" → should be "hello"
	content0, ok := o.Messages[0].Content.(string)
	if !ok || content0 != "hello" {
		t.Fatalf("messages[0] content: want 'hello', got %v", o.Messages[0].Content)
	}
	// Second: string content "hi there" → should be "hi there"
	content1, ok := o.Messages[1].Content.(string)
	if !ok || content1 != "hi there" {
		t.Fatalf("messages[1] content: want 'hi there', got %v", o.Messages[1].Content)
	}
	// Third: array content "how are you?" → should be "how are you?"
	content2, ok := o.Messages[2].Content.(string)
	if !ok || content2 != "how are you?" {
		t.Fatalf("messages[2] content: want 'how are you?', got %v", o.Messages[2].Content)
	}
}

func TestAnthropicMessage_UnmarshalStringContent(t *testing.T) {
	// Direct unmarshal of the struct — this is what was failing.
	var msg AnthropicMessage
	if err := json.Unmarshal([]byte(`{"role":"user","content":"hello"}`), &msg); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if msg.Role != "user" {
		t.Fatalf("role: want user, got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks: want 1, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != "text" || msg.Content[0].Text != "hello" {
		t.Fatalf("content[0]: want text/hello, got %s/%s", msg.Content[0].Type, msg.Content[0].Text)
	}
}

func TestAnthropicMessage_UnmarshalArrayContent(t *testing.T) {
	var msg AnthropicMessage
	if err := json.Unmarshal([]byte(`{"role":"user","content":[{"type":"text","text":"hello"}]}`), &msg); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if msg.Role != "user" {
		t.Fatalf("role: want user, got %q", msg.Role)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks: want 1, got %d", len(msg.Content))
	}
	if msg.Content[0].Type != "text" || msg.Content[0].Text != "hello" {
		t.Fatalf("content[0]: want text/hello, got %s/%s", msg.Content[0].Type, msg.Content[0].Text)
	}
}

// ---------------------------------------------------------------------------
// SSE streaming: OpenAI delta → Anthropic events
// ---------------------------------------------------------------------------

func TestConvertSSE_OpenAIStream_TextDelta(t *testing.T) {
	raw := []byte(`data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`)
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.Contains(out, "event: content_block_delta") {
		t.Fatalf("expected content_block_delta event, got: %s", out)
	}
	if !strings.Contains(out, "text_delta") || !strings.Contains(out, "Hello") {
		t.Fatalf("expected text_delta with 'Hello', got: %s", out)
	}
}

func TestConvertSSE_OpenAIStream_ReasoningDelta(t *testing.T) {
	raw := []byte(`data: {"choices":[{"index":0,"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}`)
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.Contains(out, "thinking_delta") || !strings.Contains(out, "thinking...") {
		t.Fatalf("expected thinking_delta with 'thinking...', got: %s", out)
	}
}

func TestConvertSSE_OpenAIStream_FinishReason(t *testing.T) {
	raw := []byte(`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.Contains(out, "event: message_delta") {
		t.Fatalf("expected message_delta event, got: %s", out)
	}
	if !strings.Contains(out, "end_turn") {
		t.Fatalf("expected stop_reason=end_turn, got: %s", out)
	}
}

func TestConvertSSE_OpenAIStream_RoleAnnouncement(t *testing.T) {
	// Role-only delta should pass through unchanged.
	raw := []byte(`data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)
	b, err := ConvertSSE(raw, opts())
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if string(b) != string(raw) {
		t.Fatalf("expected passthrough, got: %s", out)
	}
}

// ---------------------------------------------------------------------------
// Convert auto-detection: all 4 paths
// ---------------------------------------------------------------------------

func TestConvertViaAutoDetection_AnthropicReq(t *testing.T) {
	// Convert() should detect Anthropic Request and convert to OpenAI Request.
	body := []byte(`{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)
	b, err := Convert(body, anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if o.Model != "claude" {
		t.Fatalf("model: want claude, got %q", o.Model)
	}
}

func TestConvertViaAutoDetection_OpenAIResp(t *testing.T) {
	// Convert() should detect OpenAI Response and convert to Anthropic Response.
	body := []byte(`{"id":"chatcmpl-xyz","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	b, err := Convert(body, opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicResponse
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if a.Type != "message" {
		t.Fatalf("type: want message, got %q", a.Type)
	}
}

func TestConvertViaAutoDetection_AnthropicRespStillWorks(t *testing.T) {
	// Existing Anthropic Response → OpenAI Response path should still work.
	body := []byte(`{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	b, err := Convert(body, opts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(o.Choices) != 1 {
		t.Fatalf("want 1 choice, got %d", len(o.Choices))
	}
}

func TestConvertViaAutoDetection_OpenAIReqStillWorks(t *testing.T) {
	// Existing OpenAI Request → Anthropic Request path should still work.
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`)
	b, err := Convert(body, opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if a.Model != "gpt-4" {
		t.Fatalf("model: want gpt-4, got %q", a.Model)
	}
}

// ---------------------------------------------------------------------------
// ReasoningCache: store, inject, miss, eviction
// ---------------------------------------------------------------------------

func TestConvert_ReasoningCache_StoreInject(t *testing.T) {
	rc := NewReasoningCache(100)

	// Step 1: OpenAI Response (reasoning + tool_calls) → Anthropic Response (cache store).
	opts1 := &ConvertOptions{
		Model:          "claude-sonnet-4-20250514",
		MaxTokens:      8192,
		ReasoningCache: rc,
	}
	openAIResp := `{
		"id":"chatcmpl-xyz","object":"chat.completion","choices":[
			{"index":0,"message":{
				"role":"assistant",
				"reasoning_content":"thinking steps...",
				"content":"answer",
				"tool_calls":[
					{"id":"call_abc","type":"function","function":{"name":"f","arguments":"{}"}}
				]
			},"finish_reason":"tool_calls"}
		],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`
	b, err := Convert([]byte(openAIResp), opts1)
	if err != nil {
		t.Fatal(err)
	}
	var anthResp AnthropicResponse
	if err := json.Unmarshal(b, &anthResp); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(anthResp.Content) != 3 {
		t.Fatalf("want 3 content blocks (thinking + text + tool_use), got %d", len(anthResp.Content))
	}
	if anthResp.Content[0].Type != "thinking" || anthResp.Content[0].Thinking != "thinking steps..." {
		t.Fatalf("first block: want thinking/'thinking steps...', got type=%s thinking=%q",
			anthResp.Content[0].Type, anthResp.Content[0].Thinking)
	}
	// Cache should have two entries: one by tool call ID, one by assistant text.
	if rc.Len() != 2 {
		t.Fatalf("cache Len: want 2 (tool ID + text), got %d", rc.Len())
	}
	if v, ok := rc.Get([]string{"call_abc"}); !ok || v != "thinking steps..." {
		t.Fatalf("cache Get: want 'thinking steps...', got %q, ok=%v", v, ok)
	}
	// Text-based cache should also have the entry.
	if textV, ok := rc.GetText("answer"); !ok || textV != "thinking steps..." {
		t.Fatalf("cache GetText: want 'thinking steps...', got %q, ok=%v", textV, ok)
	}

	// Step 2: Anthropic Request (tool_use, no thinking)
	opts2 := &ConvertOptions{
		ReasoningCache: rc,
	}
	anthReq := `{
		"model":"claude-sonnet-4-20250514","max_tokens":8192,"messages":[
			{"role":"user","content":[{"type":"text","text":"hi"}]},
			{"role":"assistant","content":[
				{"type":"text","text":""},
				{"type":"tool_use","id":"call_abc","name":"f","input":{}}
			]}
		]
	}`
	b2, err := Convert([]byte(anthReq), opts2)
	if err != nil {
		t.Fatal(err)
	}
	var oaiReq OpenAIChatRequest
	if err := json.Unmarshal(b2, &oaiReq); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b2)
	}
	if len(oaiReq.Messages) != 2 {
		t.Fatalf("want 2 messages, got %d", len(oaiReq.Messages))
	}
	assistant := oaiReq.Messages[1]
	if assistant.ReasoningContent != "thinking steps..." {
		t.Fatalf("reasoning_content: want 'thinking steps...', got %q", assistant.ReasoningContent)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("want 1 tool_call, got %d", len(assistant.ToolCalls))
	}
}

func TestConvert_ReasoningCache_Miss(t *testing.T) {
	rc := NewReasoningCache(100)

	opts := &ConvertOptions{
		ReasoningCache: rc, // empty cache → miss
	}
	anthReq := `{
		"model":"claude-sonnet-4-20250514","max_tokens":8192,"messages":[
			{"role":"user","content":[{"type":"text","text":"hi"}]},
			{"role":"assistant","content":[
				{"type":"text","text":""},
				{"type":"tool_use","id":"call_xyz","name":"f","input":{}}
			]}
		]
	}`
	b, err := Convert([]byte(anthReq), opts)
	if err != nil {
		t.Fatal(err)
	}
	var oaiReq OpenAIChatRequest
	if err := json.Unmarshal(b, &oaiReq); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	assistant := oaiReq.Messages[1]
	// DeepSeek V4 requires reasoning_content when tool_calls are present;
	// placeholder is injected on cache miss.
	if assistant.ReasoningContent != placeholderReasoning {
		t.Fatalf("reasoning_content on cache miss: want placeholder, got %q", assistant.ReasoningContent)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("want 1 tool_call, got %d", len(assistant.ToolCalls))
	}
}

func TestConvert_ReasoningCache_Eviction(t *testing.T) {
	rc := NewReasoningCache(2) // max 2 entries

	rc.Put([]string{"id1"}, "reasoning1")
	rc.Put([]string{"id2"}, "reasoning2")

	if rc.Len() != 2 {
		t.Fatalf("Len: want 2, got %d", rc.Len())
	}

	// Third put should evict id1 (FIFO).
	rc.Put([]string{"id3"}, "reasoning3")

	if rc.Len() != 2 {
		t.Fatalf("Len after eviction: want 2, got %d", rc.Len())
	}

	if _, ok := rc.Get([]string{"id1"}); ok {
		t.Fatal("id1 should be evicted")
	}
	if v, ok := rc.Get([]string{"id2"}); !ok || v != "reasoning2" {
		t.Fatal("id2 should still be present")
	}
	if v, ok := rc.Get([]string{"id3"}); !ok || v != "reasoning3" {
		t.Fatal("id3 should still be present")
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// anthropicOpts returns ConvertOptions configured for Anthropic→OpenAI direction.
func anthropicOpts() *ConvertOptions {
	return &ConvertOptions{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 8192,
		URI:       "/v1/messages",
		Direction: "request",
	}
}

// ---------------------------------------------------------------------------
// DeepSeek model-specific tests
// ---------------------------------------------------------------------------

func TestConvert_DeepSeekToolChoice_AnyStripped(t *testing.T) {
	// DeepSeek models strip forced tool_choice and inject a system instruction.
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"use a tool"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.ToolChoice != nil {
		t.Fatalf("DeepSeek tool_choice 'any': want nil (stripped), got %v", o.ToolChoice)
	}
	// System instruction should contain the tool call requirement.
	if len(o.Messages) == 0 || !strings.Contains(extractTextContent(o.Messages[0].Content), "Call one of the available tools") {
		t.Fatalf("DeepSeek tool_choice: system instruction missing, got first message: %+v", o.Messages[0])
	}
}

func TestConvert_DeepSeekToolChoice_ToolStripped(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"use a tool"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"f"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.ToolChoice != nil {
		t.Fatalf("DeepSeek tool_choice 'tool': want nil (stripped), got %v", o.ToolChoice)
	}
	if len(o.Messages) == 0 || !strings.Contains(extractTextContent(o.Messages[0].Content), "tool named f") {
		t.Fatalf("DeepSeek tool_choice 'tool': system instruction missing tool name, got: %v", o.Messages[0])
	}
}

func TestConvert_DeepSeekThinkingAndEffort(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"enabled","budget_tokens":4096},"output_config":{"effort":"max"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Thinking == nil {
		t.Fatal("DeepSeek: thinking field should be set")
	}
	if o.ReasoningEffort == nil {
		t.Fatal("DeepSeek: reasoning_effort should be set")
	}
}

func TestConvert_NonDeepSeekNoThinkingEffort(t *testing.T) {
	// Non-DeepSeek models should NOT get thinking/reasoning_effort.
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"enabled"},"output_config":{"effort":"max"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Thinking != nil {
		t.Fatal("Non-DeepSeek: thinking field should not be set")
	}
	if o.ReasoningEffort != nil {
		t.Fatal("Non-DeepSeek: reasoning_effort should not be set")
	}
}
func TestConvert_GLMThinkingMapping(t *testing.T) {
	// GLM models should get thinking with budget_tokens (not reasoning_effort).
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"enabled","budget_tokens":4096}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "glm-4.6"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Thinking == nil {
		t.Fatal("GLM: thinking field should be set")
	}
	thinking, ok := o.Thinking.(map[string]any)
	if !ok {
		t.Fatalf("GLM: thinking should be a map, got %T", o.Thinking)
	}
	if typ, _ := thinking["type"].(string); typ != "enabled" {
		t.Fatalf("GLM: thinking.type should be 'enabled', got %q", typ)
	}
	budget, hasBudget := thinking["budget_tokens"]
	if !hasBudget {
		t.Fatal("GLM: thinking.budget_tokens should be set")
	}
	if b, ok := budget.(float64); !ok || int(b) != 4096 {
		t.Fatalf("GLM: thinking.budget_tokens should be 4096, got %v", budget)
	}
	if o.ReasoningEffort != nil {
		t.Fatal("GLM: reasoning_effort should NOT be set")
	}
	// GLM thinking without budget_tokens should still work.
	body2 := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"enabled"}}`
	opts2 := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "glm-4.6"}}}
	b2, err := Convert([]byte(body2), opts2)
	if err != nil {
		t.Fatal(err)
	}
	var o2 OpenAIChatRequest
	if err := json.Unmarshal(b2, &o2); err != nil {
		t.Fatal(err)
	}
	if o2.Thinking == nil {
		t.Fatal("GLM: thinking field should be set even without budget")
	}
	thinking2, _ := o2.Thinking.(map[string]any)
	if _, hasB := thinking2["budget_tokens"]; hasB {
		t.Fatal("GLM: budget_tokens should not be set when Anthropic thinking has no budget")
	}
}

func TestConvert_StreamOptionsIncluded(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, URI: "/v1/messages", Direction: "request"}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.StreamOptions == nil {
		t.Fatal("expected stream_options for streaming request")
	}
}

func TestConvert_CoalesceAdjacentAssistantToolCalls(t *testing.T) {
	messages := []OpenAIMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: nil, ToolCalls: []OpenAIToolCall{
			{ID: "call_1", Type: "function", Function: OpenAIFunctionCall{Name: "f1", Arguments: "{}"}},
		}},
		{Role: "assistant", Content: nil, ToolCalls: []OpenAIToolCall{
			{ID: "call_2", Type: "function", Function: OpenAIFunctionCall{Name: "f2", Arguments: "{}"}},
		}},
	}
	result := coalesceAdjacentAssistantToolCalls(messages)
	if len(result) != 2 {
		t.Fatalf("want 2 messages after coalescing, got %d", len(result))
	}
	if len(result[1].ToolCalls) != 2 {
		t.Fatalf("want 2 tool_calls after coalescing, got %d", len(result[1].ToolCalls))
	}
}

func TestConvert_TextBasedReasoningCache(t *testing.T) {
	rc := NewReasoningCache(100)

	// Store by text.
	rc.PutText("hello world", "deep thoughts")
	v, ok := rc.GetText("hello world")
	if !ok || v != "deep thoughts" {
		t.Fatalf("GetText: want 'deep thoughts', got %q, ok=%v", v, ok)
	}

	// Store by context.
	rc.PutContext([]string{"tool_use:1:read:{}"}, "result", "context reasoning")
	cv, cok := rc.GetContext([]string{"tool_use:1:read:{}"}, "result")
	if !cok || cv != "context reasoning" {
		t.Fatalf("GetContext: want 'context reasoning', got %q, ok=%v", cv, cok)
	}
}

func TestConvert_GetBestMultiTier(t *testing.T) {
	rc := NewReasoningCache(100)

	// Store in text tier only.
	rc.PutText("hello", "text tier reasoning")

	// GetBest should find it via text (tool IDs empty, context empty).
	result := rc.GetBest(nil, nil, "hello")
	if result != "text tier reasoning" {
		t.Fatalf("GetBest: want 'text tier reasoning', got %q", result)
	}

	// Tool call ID tier takes priority.
	rc.Put([]string{"tool_1"}, "tool tier reasoning")
	result = rc.GetBest([]string{"tool_1"}, nil, "hello")
	if result != "tool tier reasoning" {
		t.Fatalf("GetBest: tool tier should take priority, got %q", result)
	}
}

func TestAnthropicRequest_UnmarshalJSON_SystemString(t *testing.T) {
	body := `{"model":"claude-opus-4","max_tokens":4096,"messages":[{"role":"user","content":"hello"}],"system":"you are a helpful assistant"}`
	var req AnthropicRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if len(req.System) != 1 {
		t.Fatalf("want 1 system block, got %d", len(req.System))
	}
	if req.System[0].Type != "text" || req.System[0].Text != "you are a helpful assistant" {
		t.Fatalf("system: want {text, you are a helpful assistant}, got {%s, %s}", req.System[0].Type, req.System[0].Text)
	}
}

func TestAnthropicRequest_UnmarshalJSON_SystemArray(t *testing.T) {
	body := `{"model":"claude-opus-4","max_tokens":4096,"messages":[{"role":"user","content":"hello"}],"system":[{"type":"text","text":"be helpful"},{"type":"text","text":"be concise"}]}`
	var req AnthropicRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if len(req.System) != 2 {
		t.Fatalf("want 2 system blocks, got %d", len(req.System))
	}
	if req.System[0].Text != "be helpful" || req.System[1].Text != "be concise" {
		t.Fatalf("system: want [be helpful, be concise], got [%s, %s]", req.System[0].Text, req.System[1].Text)
	}
}

func TestAnthropicRequest_UnmarshalJSON_NoSystem(t *testing.T) {
	body := `{"model":"claude-opus-4","max_tokens":4096,"messages":[{"role":"user","content":"hello"}]}`
	var req AnthropicRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if req.System != nil {
		t.Fatalf("system should be nil, got %v", req.System)
	}
}

func TestConvert_AnthropicRequestWithStringSystem(t *testing.T) {
	// Anthropic request with string system → OpenAI request
	body := `{"model":"claude-opus-4","max_tokens":4096,"messages":[{"role":"user","content":"hello"}],"system":"you are a helpful assistant"}`
	oaiBody, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var oai OpenAIChatRequest
	if err := json.Unmarshal(oaiBody, &oai); err != nil {
		t.Fatalf("unmarshal OpenAI error: %v\nbody: %s", err, oaiBody)
	}
	// The system string should become the first message with role=system
	if len(oai.Messages) < 2 {
		t.Fatalf("want at least 2 messages, got %d", len(oai.Messages))
	}
	if oai.Messages[0].Role != "system" {
		t.Fatalf("first message role: want system, got %q", oai.Messages[0].Role)
	}
	if oai.Messages[0].Content != "you are a helpful assistant" {
		t.Fatalf("system content: want %q, got %q", "you are a helpful assistant", oai.Messages[0].Content)
	}
}

func TestHandleSSEEvent_AnthropicPassthroughStreaming(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "anthropic"},
		{SourcePrefix: "*", TargetModel: "deepseek-v4-flash", Protocol: "anthropic"},
	}
	store := NewSessionStore()
	opts := &ConvertOptions{
		Model:        "deepseek-chat",
		MaxTokens:    8192,
		ModelMap:     mm,
		SessionStore: store,
		URI:          "/v1/messages",
		Direction:    "response",
	}

	// Stream start: Anthropic message_start should pass through unchanged.
	startData := []byte(`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"msg_abc","type":"message","role":"assistant","model":"deepseek-v4-pro","content":[],"stop_reason":null,"stop_sequence":null}}`)
	out, err := HandleSSEEvent("test-sid", string(StreamPhaseStart), 0, startData, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"model":"claude-opus"`) {
		t.Fatalf("stream start: expected passthrough with source model, got: %s", out)
	}

	// Delta event should pass through unchanged.
	deltaData := []byte(`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hello"}}`)
	out2, err := HandleSSEEvent("test-sid", string(StreamPhaseEvent), 0, deltaData, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out2, []byte("event:")) {
		t.Fatalf("stream event: expected SSE-framed output, got: %s", out2)
	}
	if !strings.Contains(string(out2), "thinking_delta") {
		t.Fatalf("stream event: expected passthrough of thinking delta, got: %s", out2)
	}

	// Stream end: passthrough should return nil.
	out3, err := HandleSSEEvent("test-sid", string(StreamPhaseEnd), 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if out3 != nil {
		t.Fatalf("stream end: expected nil for passthrough, got: %s", out3)
	}

	// Session should be cleaned up.
	if sess := store.Get("test-sid"); sess != nil {
		t.Fatal("session should be deleted after stream end")
	}
}

// ---------------------------------------------------------------------------
// Body-primary detection: custom endpoints must not depend on URI.
// ---------------------------------------------------------------------------

// TestConvert_CustomEndpoint_BodyDetection verifies that an Anthropic-shaped
// request on a non-canonical endpoint (/v1/llm) is detected via body markers
// (top-level system + tools[].input_schema) — URI never participates.
func TestConvert_CustomEndpoint_BodyDetection(t *testing.T) {
	opts := &ConvertOptions{
		Model: "deepseek-chat", MaxTokens: 8192,
		URI: "/v1/llm", Direction: "request",
		ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}},
	}
	body := `{"model":"claude-sonnet-4","max_tokens":100,"system":"s",` +
		`"tools":[{"name":"t","input_schema":{"type":"object","properties":{}}}],` +
		`"messages":[{"role":"user","content":"hi"}]}`
	out, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	// Source detected as Anthropic → converted to OpenAI Chat downstream.
	// Converted request has no top-level "system" and no "input_schema";
	// model is rewritten to the downstream target.
	if bytes.Contains(out, []byte(`"input_schema"`)) || bytes.Contains(out, []byte(`"system":"s"`)) {
		t.Fatalf("expected Anthropic→OpenAI conversion, body still Anthropic-shaped: %s", out)
	}
	if !bytes.Contains(out, []byte(`"deepseek-chat"`)) {
		t.Fatalf("expected model rewritten to deepseek-chat: %s", out)
	}
}

// TestConvert_PathologicalMinimal_Passthrough verifies that a minimal request
// with no distinguishing body markers on a non-canonical endpoint yields
// ProtocolUnknown and is returned completely unchanged (no guess, no rewrite).
func TestConvert_PathologicalMinimal_Passthrough(t *testing.T) {
	opts := &ConvertOptions{
		Model: "deepseek-chat", MaxTokens: 8192,
		URI: "/v1/llm", Direction: "request",
		ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}},
	}
	body := `{"model":"x","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	out, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != body {
		t.Fatalf("expected unchanged passthrough, got: %s", out)
	}
}

// TestConvert_ResponseUsesSessionClientProtocol verifies the request→response
// session carry: client protocol detected on the request is stored in the
// session and read on the response, so a response with an EMPTY URI (as when
// resp.Request==nil) still converts back to the client protocol.
func TestConvert_ResponseUsesSessionClientProtocol(t *testing.T) {
	store := NewSessionStore()
	mm := ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}

	// 1. Request: Anthropic client on /v1/messages → converted to OpenAI downstream.
	reqOpts := &ConvertOptions{
		Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm,
		SID: "sess-rt", SessionStore: store,
		URI: "/v1/messages", Direction: "request",
	}
	anthReq := `{"model":"claude-sonnet-4","max_tokens":100,"system":"s","messages":[{"role":"user","content":"hi"}]}`
	if _, err := Convert([]byte(anthReq), reqOpts); err != nil {
		t.Fatal(err)
	}
	sess := store.Get("sess-rt")
	if sess == nil || sess.From != ProtocolAnthropic {
		t.Fatalf("session.From should be Anthropic after request, got %+v", sess)
	}

	// 2. Response: downstream returns OpenAI Chat, URI EMPTY (resp.Request==nil).
	// Client protocol must come from the session, not the URI.
	respOpts := &ConvertOptions{
		Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm,
		SID: "sess-rt", SessionStore: store,
		URI: "", Direction: "response",
	}
	openaiResp := `{"id":"x","object":"chat.completion","model":"deepseek-chat",` +
		`"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	out, err := Convert([]byte(openaiResp), respOpts)
	if err != nil {
		t.Fatal(err)
	}
	// source=OpenAIChat (choices), client=Anthropic (session) → converted to Anthropic.
	if !bytes.Contains(out, []byte(`"type":"message"`)) {
		t.Fatalf("expected OpenAI→Anthropic conversion using session client proto, got: %s", out)
	}
	// Session must be deleted after the non-streaming response (pair complete).
	if sess := store.Get("sess-rt"); sess != nil {
		t.Fatalf("session should be deleted after response, got %+v", sess)
	}
}

// TestHandleSSEEvent_AsymmetricProtocolOverride_UsesSessionClient verifies the
// streaming path reads the client protocol from the session (stored at request
// time) instead of deriving the target from resolveModel. With a `:openai`
// override and an Anthropic client, the OpenAI stream chunks must convert back
// to Anthropic — not passthrough (which resolveModel's downstream=openai would do).
func TestHandleSSEEvent_AsymmetricProtocolOverride_UsesSessionClient(t *testing.T) {
	mm := ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}
	store := NewSessionStore()

	// 1. Request: Anthropic client → stores session.From=Anthropic.
	reqOpts := &ConvertOptions{
		Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm,
		SessionStore: store, SID: "asym-sid",
		URI: "/v1/messages", Direction: "request",
	}
	anthReq := `{"model":"claude-sonnet-4","max_tokens":100,"system":"s","messages":[{"role":"user","content":"hi"}]}`
	if _, err := Convert([]byte(anthReq), reqOpts); err != nil {
		t.Fatal(err)
	}
	if sess := store.Get("asym-sid"); sess == nil || sess.From != ProtocolAnthropic {
		t.Fatalf("request should store session.From=Anthropic, got %+v", sess)
	}

	// 2. Streaming response: OpenAI chunk. Must convert to Anthropic via session.From.
	respOpts := &ConvertOptions{
		Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm,
		SessionStore: store, SID: "asym-sid",
		URI: "/v1/messages", Direction: "response",
	}
	startData := []byte(`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`)
	out, err := HandleSSEEvent("asym-sid", string(StreamPhaseStart), 0, startData, respOpts)
	if err != nil {
		t.Fatal(err)
	}
	// StreamConverter (OpenAI→Anthropic) emits Anthropic SSE.
	if !bytes.Contains(out, []byte(`message_start`)) {
		t.Fatalf("expected OpenAI→Anthropic stream conversion using session.From, got: %s", out)
	}
	// And it must NOT have chosen a passthrough handler.
	if sess := store.Get("asym-sid"); sess != nil {
		if _, ok := sess.StreamHandler.(*PassthroughStreamHandler); ok {
			t.Fatal("got PassthroughStreamHandler — streaming target did not use session.From")
		}
	}
}
