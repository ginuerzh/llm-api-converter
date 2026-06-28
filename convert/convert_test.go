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
	if acr.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("model: want claude-sonnet-4-20250514, got %q", acr.Model)
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

func TestIsAnthropicRequest_DetectsCorrectly(t *testing.T) {
	tests := []struct {
		name string
		json string
		want bool
	}{
		{
			name: "anthropic with thinking block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"..."}]}]}`,
			want: true,
		},
		{
			name: "anthropic with tool_use block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"tu1","name":"f","input":{}}]}]}`,
			want: true,
		},
		{
			name: "anthropic with tool_result block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"ok"}]}]}`,
			want: true,
		},
		{
			name: "anthropic with image block",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"aa"}}]}]}`,
			want: true,
		},
		{
			name: "anthropic with top-level system",
			json: `{"model":"claude","max_tokens":8192,"system":[{"type":"text","text":"be helpful"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			want: true,
		},
		{
			name: "anthropic with tool_choice.any",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}}`,
			want: true,
		},
		{
			name: "anthropic with max_tokens and array content",
			json: `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			want: true,
		},
		{
			name: "openai with frequency_penalty",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.5}`,
			want: false,
		},
		{
			name: "openai with stop field",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stop":"done"}`,
			want: false,
		},
		{
			name: "openai with tool role in messages",
			json: `{"model":"gpt-4","messages":[{"role":"tool","content":"result"}]}`,
			want: false,
		},
		{
			name: "openai with system role in messages",
			json: `{"model":"gpt-4","messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}]}`,
			want: false,
		},
		{
			name: "openai tool_choice as string",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"auto"}`,
			want: false,
		},
		{
			name: "openai with n field",
			json: `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"n":2}`,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal([]byte(tt.json), &raw); err != nil {
				t.Fatal(err)
			}
			if got := isAnthropicRequest(raw); got != tt.want {
				t.Errorf("isAnthropicRequest() = %v, want %v\nbody: %s", got, tt.want, tt.json)
			}
		})
	}
}

func TestIsOpenAIResponse_DetectsCorrectly(t *testing.T) {
	tests := []struct {
		name string
		json string
		want bool
	}{
		{
			name: "full non-streaming response",
			json: `{"id":"chatcmpl-xyz","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
			want: true,
		},
		{
			name: "streaming delta chunk",
			json: `{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
			want: true,
		},
		{
			name: "unknown JSON without choices",
			json: `{"foo":"bar"}`,
			want: false,
		},
		{
			name: "anthropic response",
			json: `{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal([]byte(tt.json), &raw); err != nil {
				t.Fatal(err)
			}
			if got := isOpenAIResponse(raw); got != tt.want {
				t.Errorf("isOpenAIResponse() = %v, want %v\nbody: %s", got, tt.want, tt.json)
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

func TestIsAnthropicRequest_OpenAIReqWithMaxTokensAndArrayContent(t *testing.T) {
	// Regression: an OpenAI request with max_tokens and array-style content
	// should NOT be classified as Anthropic Request.
	body := `{"model":"gpt-4","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	var raw map[string]any
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatal(err)
	}
	if isAnthropicRequest(raw) {
		t.Fatal("OpenAI request with max_tokens + array content should NOT be Anthropic Request")
	}
	// Should be caught by catch-all as OpenAI Request → Anthropic Request.
	out, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(out, &a); err != nil {
		t.Fatalf("expected valid Anthropic request output, got: %s", out)
	}
	if a.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("model: want claude-sonnet-4-20250514, got %q", a.Model)
	}
}

func TestIsAnthropicRequest_CustomModelName(t *testing.T) {
	// Anthropic request with a non-standard model name should still be detected.
	body := `{"model":"my-custom-model","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	var raw map[string]any
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatal(err)
	}
	if !isAnthropicRequest(raw) {
		t.Fatal("Anthropic request with custom model should be detected as Anthropic Request")
	}
}

func TestIsAnthropicRequest_DeepSeekModelIsOpenAI(t *testing.T) {
	// deepseek-* models are OpenAI-compatible, not Anthropic.
	body := `{"model":"deepseek-chat","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	var raw map[string]any
	if err := json.Unmarshal([]byte(body), &raw); err != nil {
		t.Fatal(err)
	}
	if isAnthropicRequest(raw) {
		t.Fatal("deepseek model should NOT be classified as Anthropic Request")
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
	if o.Model != "deepseek-chat" {
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
	if len(o.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(o.Messages))
	}
	if o.Messages[0].Role != "tool" {
		t.Fatalf("role: want tool, got %q", o.Messages[0].Role)
	}
	if o.Messages[0].ToolCallID != "tu01" {
		t.Fatalf("tool_call_id: want tu01, got %q", o.Messages[0].ToolCallID)
	}
	content, ok := o.Messages[0].Content.(string)
	if !ok || content != "sunny" {
		t.Fatalf("content: want 'sunny', got %v", o.Messages[0].Content)
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
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.ToolChoice != "required" {
		t.Fatalf("tool_choice: want required, got %v", o.ToolChoice)
	}
}

func TestConvert_AnthropicReqToOpenAIReq_ToolChoiceTool(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"f"}}`
	b, err := Convert([]byte(body), anthropicOpts())
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	tc, ok := o.ToolChoice.(map[string]any)
	if !ok {
		t.Fatalf("tool_choice: expected object, got %T", o.ToolChoice)
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
	if o.Model != "deepseek-chat" {
		t.Fatalf("model: want deepseek-chat, got %q", o.Model)
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
	if a.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("model: want claude-sonnet-4-20250514, got %q", a.Model)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// anthropicOpts returns ConvertOptions configured for Anthropic→OpenAI direction.
func anthropicOpts() *ConvertOptions {
	return &ConvertOptions{
		Model:      "claude-sonnet-4-20250514",
		MaxTokens:  8192,
		Downstream: "deepseek-chat",
	}
}
