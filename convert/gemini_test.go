package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func geminiOpts() *ConvertOptions {
	return &ConvertOptions{
		Model:     "gemini-2.5-flash",
		MaxTokens: 8192,
		URI:       "/v1beta/models/gemini-2.5-flash:generateContent",
		Direction: "request",
	}
}

// ---------------------------------------------------------------------------
// OpenAI Chat Request → Gemini Request
// ---------------------------------------------------------------------------

func TestGemini_OpenAIReqToGemini_Simple(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	b, err := convertOpenAIRequestToGemini([]byte(body), geminiOpts())
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Contents) != 1 {
		t.Fatalf("want 1 content, got %d", len(g.Contents))
	}
	if g.Contents[0].Role != "user" {
		t.Fatalf("role: want user, got %q", g.Contents[0].Role)
	}
	if len(g.Contents[0].Parts) != 1 || g.Contents[0].Parts[0].Text != "hello" {
		t.Fatalf("parts: want [hello], got %+v", g.Contents[0].Parts)
	}
	if g.GenerationConfig == nil || g.GenerationConfig.MaxOutputTokens == nil || *g.GenerationConfig.MaxOutputTokens != 8192 {
		t.Fatalf("maxOutputTokens: want 8192, got %+v", g.GenerationConfig)
	}
}

func TestGemini_OpenAIReqToGemini_SystemMessage(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"hi"}]}`
	b, err := convertOpenAIRequestToGemini([]byte(body), geminiOpts())
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if g.SystemInstruction == nil {
		t.Fatal("want systemInstruction")
	}
	if len(g.SystemInstruction.Parts) != 1 || !strings.Contains(g.SystemInstruction.Parts[0].Text, "helpful") {
		t.Fatalf("system text: want 'helpful', got %+v", g.SystemInstruction.Parts)
	}
	if len(g.Contents) != 1 || g.Contents[0].Role != "user" {
		t.Fatalf("want 1 user content, got %+v", g.Contents)
	}
}

func TestGemini_OpenAIReqToGemini_Tools(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object"}}}],"tool_choice":"auto"}`
	b, err := convertOpenAIRequestToGemini([]byte(body), geminiOpts())
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Tools) != 1 || len(g.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("want 1 tool with 1 declaration, got %+v", g.Tools)
	}
	if g.Tools[0].FunctionDeclarations[0].Name != "get_weather" {
		t.Fatalf("name: want get_weather, got %q", g.Tools[0].FunctionDeclarations[0].Name)
	}
	if g.ToolConfig == nil || g.ToolConfig.FunctionCallingConfig == nil {
		t.Fatal("want toolConfig")
	}
	if g.ToolConfig.FunctionCallingConfig.Mode != "AUTO" {
		t.Fatalf("mode: want AUTO, got %q", g.ToolConfig.FunctionCallingConfig.Mode)
	}
}

func TestGemini_OpenAIReqToGemini_AssistantWithToolCalls(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"weather?"},
		{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"72F"}
	]}`
	b, err := convertOpenAIRequestToGemini([]byte(body), geminiOpts())
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Contents) != 3 {
		t.Fatalf("want 3 contents, got %d", len(g.Contents))
	}
	// Assistant message → model role with functionCall part.
	if g.Contents[1].Role != "model" {
		t.Fatalf("want model role for assistant, got %q", g.Contents[1].Role)
	}
	if g.Contents[1].Parts[0].FunctionCall == nil {
		t.Fatal("want functionCall part")
	}
	if g.Contents[1].Parts[0].FunctionCall.Name != "get_weather" {
		t.Fatalf("function name: want get_weather, got %q", g.Contents[1].Parts[0].FunctionCall.Name)
	}
}

func TestGemini_OpenAIReqToGemini_Image(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/4AAQ"}}]}]}`
	b, err := convertOpenAIRequestToGemini([]byte(body), geminiOpts())
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Contents[0].Parts) != 2 {
		t.Fatalf("want 2 parts, got %d", len(g.Contents[0].Parts))
	}
	if g.Contents[0].Parts[1].InlineData == nil {
		t.Fatal("want inlineData for image")
	}
	if g.Contents[0].Parts[1].InlineData.MimeType != "image/jpeg" {
		t.Fatalf("mimeType: want image/jpeg, got %q", g.Contents[0].Parts[1].InlineData.MimeType)
	}
}

// ---------------------------------------------------------------------------
// Gemini Response → OpenAI Chat Response
// ---------------------------------------------------------------------------

func TestGemini_ResponseToOpenAI_Simple(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Hello!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}`
	b, err := convertGeminiResponseToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gemini-2.5-flash"})
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
	if o.Choices[0].Message.Content != "Hello!" {
		t.Fatalf("content: want 'Hello!', got %q", o.Choices[0].Message.Content)
	}
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason: want stop, got %v", o.Choices[0].FinishReason)
	}
	if o.Usage.PromptTokens != 10 || o.Usage.CompletionTokens != 20 {
		t.Fatalf("usage: want 10/20, got %d/%d", o.Usage.PromptTokens, o.Usage.CompletionTokens)
	}
}

func TestGemini_ResponseToOpenAI_WithFunctionCall(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Let me check"},{"functionCall":{"name":"get_weather","args":{"loc":"NYC"}}}],"role":"model"},"finishReason":"STOP","index":0}]}`
	b, err := convertGeminiResponseToOpenAI([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(o.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("want 1 tool_call, got %d", len(o.Choices[0].Message.ToolCalls))
	}
	if o.Choices[0].Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("name: want get_weather, got %q", o.Choices[0].Message.ToolCalls[0].Function.Name)
	}
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason: want tool_calls, got %v", o.Choices[0].FinishReason)
	}
	if !strings.Contains(o.Choices[0].Message.Content.(string), "Let me check") {
		t.Fatalf("content: want 'Let me check', got %q", o.Choices[0].Message.Content)
	}
}

func TestGemini_ResponseToOpenAI_MaxTokens(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"partial"}],"role":"model"},"finishReason":"MAX_TOKENS","index":0}]}`
	b, err := convertGeminiResponseToOpenAI([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "length" {
		t.Fatalf("finish_reason: want length, got %v", o.Choices[0].FinishReason)
	}
}

// ---------------------------------------------------------------------------
// OpenAI Chat Response → Gemini Response (reverse direction)
// ---------------------------------------------------------------------------

func TestGemini_OpenARespToGemini_Simple(t *testing.T) {
	body := `{"id":"chatcmpl-x","object":"chat.completion","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}`
	b, err := convertOpenAIResponseToGemini([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatResponse
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Candidates) != 1 {
		t.Fatalf("want 1 candidate, got %d", len(g.Candidates))
	}
	if len(g.Candidates[0].Content.Parts) != 1 || g.Candidates[0].Content.Parts[0].Text != "Hello!" {
		t.Fatalf("parts: want [Hello!], got %+v", g.Candidates[0].Content.Parts)
	}
	if g.Candidates[0].FinishReason != "STOP" {
		t.Fatalf("finishReason: want STOP, got %q", g.Candidates[0].FinishReason)
	}
	if g.UsageMetadata == nil || g.UsageMetadata.PromptTokenCount != 5 {
		t.Fatalf("usage: want 5, got %+v", g.UsageMetadata)
	}
}

// ---------------------------------------------------------------------------
// Anthropic Request → Gemini Request
// ---------------------------------------------------------------------------

func TestGemini_AnthropicReqToGemini_Simple(t *testing.T) {
	body := `{"model":"claude-sonnet","max_tokens":4096,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`
	b, err := convertAnthropicRequestToGemini([]byte(body), geminiOpts())
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Contents) != 1 || g.Contents[0].Role != "user" {
		t.Fatalf("want 1 user content, got %+v", g.Contents)
	}
	if g.GenerationConfig == nil || g.GenerationConfig.MaxOutputTokens == nil || *g.GenerationConfig.MaxOutputTokens != 4096 {
		t.Fatalf("maxOutputTokens: want 4096, got %+v", g.GenerationConfig)
	}
}

func TestGemini_AnthropicReqToGemini_ToolUse(t *testing.T) {
	body := `{"model":"claude","max_tokens":4096,"messages":[{"role":"user","content":[{"type":"text","text":"weather?"}]}],"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object"}}],"tool_choice":{"type":"auto"}}`
	b, err := convertAnthropicRequestToGemini([]byte(body), geminiOpts())
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Tools) != 1 || len(g.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("want 1 tool, got %+v", g.Tools)
	}
	if g.Tools[0].FunctionDeclarations[0].Name != "get_weather" {
		t.Fatalf("name: want get_weather, got %q", g.Tools[0].FunctionDeclarations[0].Name)
	}
	if g.ToolConfig.FunctionCallingConfig.Mode != "AUTO" {
		t.Fatalf("mode: want AUTO, got %q", g.ToolConfig.FunctionCallingConfig.Mode)
	}
}

// ---------------------------------------------------------------------------
// Gemini Response → Anthropic Response
// ---------------------------------------------------------------------------

func TestGemini_ResponseToAnthropic_Simple(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Hello!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}`
	b, err := convertGeminiResponseToAnthropic([]byte(body), &ConvertOptions{RequestModel: "claude-sonnet"})
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
	if a.StopReason == nil || *a.StopReason != "end_turn" {
		t.Fatalf("stop_reason: want end_turn, got %v", a.StopReason)
	}
	if a.Model != "claude-sonnet" {
		t.Fatalf("model: want claude-sonnet, got %q", a.Model)
	}
	if len(a.Content) != 1 || a.Content[0].Text != "Hello!" {
		t.Fatalf("content: want [Hello!], got %+v", a.Content)
	}
	if a.Usage.InputTokens != 10 || a.Usage.OutputTokens != 20 {
		t.Fatalf("usage: want 10/20, got %d/%d", a.Usage.InputTokens, a.Usage.OutputTokens)
	}
}

func TestGemini_ResponseToAnthropic_FunctionCall(t *testing.T) {
	body := `{"candidates":[{"content":{"parts":[{"text":"Let me check"},{"functionCall":{"name":"get_weather","args":{"loc":"NYC"}}}],"role":"model"},"finishReason":"STOP","index":0}]}`
	b, err := convertGeminiResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicResponse
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if a.StopReason == nil || *a.StopReason != "end_turn" {
		t.Fatalf("stop_reason for tool calls: want end_turn, got %v", a.StopReason)
	}
	if len(a.Content) != 2 {
		t.Fatalf("want 2 content blocks, got %d", len(a.Content))
	}
	if a.Content[0].Type != "text" {
		t.Fatalf("block 0: want text, got %q", a.Content[0].Type)
	}
	if a.Content[1].Type != "tool_use" || a.Content[1].Name != "get_weather" {
		t.Fatalf("block 1: want tool_use/get_weather, got %s/%s", a.Content[1].Type, a.Content[1].Name)
	}
}

// ---------------------------------------------------------------------------
// Anthropic Response → Gemini Response (reverse direction)
// ---------------------------------------------------------------------------

func TestGemini_AnthropicRespToGemini_Simple(t *testing.T) {
	body := `{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude","stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":10}}`
	b, err := convertAnthropicResponseToGemini([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatResponse
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Candidates) != 1 {
		t.Fatalf("want 1 candidate, got %d", len(g.Candidates))
	}
	if g.Candidates[0].FinishReason != "STOP" {
		t.Fatalf("finishReason: want STOP, got %q", g.Candidates[0].FinishReason)
	}
	if len(g.Candidates[0].Content.Parts) != 1 || g.Candidates[0].Content.Parts[0].Text != "Hello!" {
		t.Fatalf("parts: want [Hello!], got %+v", g.Candidates[0].Content.Parts)
	}
}

// ---------------------------------------------------------------------------
// Gemini Request → OpenAI Chat Request (reverse direction)
// ---------------------------------------------------------------------------

func TestGemini_RequestToOpenAI_Simple(t *testing.T) {
	body := `{"contents":[{"parts":[{"text":"hello"}],"role":"user"}],"generationConfig":{"maxOutputTokens":8192}}`
	b, err := convertGeminiRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4"})
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if o.Model != "gpt-4" {
		t.Fatalf("model: want gpt-4, got %q", o.Model)
	}
	if len(o.Messages) != 1 || o.Messages[0].Role != "user" || o.Messages[0].Content.(string) != "hello" {
		t.Fatalf("messages: want [user/hello], got %+v", o.Messages)
	}
}

// ---------------------------------------------------------------------------
// Gemini Request → Anthropic Request (reverse direction)
// ---------------------------------------------------------------------------

func TestGemini_RequestToAnthropic_Simple(t *testing.T) {
	body := `{"contents":[{"parts":[{"text":"hello"}],"role":"user"}],"generationConfig":{"maxOutputTokens":4096}}`
	b, err := convertGeminiRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "claude-sonnet"})
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if a.Model != "claude-sonnet" {
		t.Fatalf("model: want claude-sonnet, got %q", a.Model)
	}
	if len(a.Messages) != 1 || a.Messages[0].Role != "user" {
		t.Fatalf("messages: want 1 user, got %+v", a.Messages)
	}
	if a.MaxTokens != 4096 {
		t.Fatalf("max_tokens: want 4096, got %d", a.MaxTokens)
	}
}

// ---------------------------------------------------------------------------
// Gemini protocol detection
// ---------------------------------------------------------------------------

func TestGemini_DetectRequest(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"contents with parts", `{"contents":[{"parts":[{"text":"hello"}],"role":"user"}]}`},
		{"contents with generationConfig", `{"contents":[{"parts":[{"text":"hi"}],"role":"user"}],"generationConfig":{"temperature":0.7}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal([]byte(tt.body), &raw); err != nil {
				t.Fatal(err)
			}
			if got := detectSource(raw); got != ProtocolGemini {
				t.Fatalf("detectSource: want gemini, got %s", got)
			}
		})
	}
}

func TestGemini_DetectResponse(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"candidates with parts", `{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP","index":0}]}`},
		{"candidates with functionCall", `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"foo"}}],"role":"model"},"finishReason":"STOP","index":0}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var raw map[string]any
			if err := json.Unmarshal([]byte(tt.body), &raw); err != nil {
				t.Fatal(err)
			}
			if got := detectSource(raw); got != ProtocolGemini {
				t.Fatalf("detectSource: want gemini, got %s", got)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// End-to-end Convert routing via model-map
// ---------------------------------------------------------------------------

func TestGemini_E2EOpenAIReqToGemini(t *testing.T) {
	mm := ModelMap{{SourcePrefix: "*", TargetModel: "gemini-2.5-flash", Protocol: "gemini"}}
	opts := &ConvertOptions{
		Model:     "gemini-2.5-flash",
		MaxTokens: 8192,
		ModelMap:  mm,
		URI:       "/v1/chat/completions",
		Direction: "request",
	}
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(g.Contents) != 1 || g.Contents[0].Parts[0].Text != "hello" {
		t.Fatalf("expected Gemini contents, got %+v", g.Contents)
	}
}

func TestGemini_E2EGeminiRespToOpenAI(t *testing.T) {
	store := NewSessionStore()
	store.Set("test-sid", &Session{ID: "test-sid", From: ProtocolOpenAIChat})

	mm := ModelMap{{SourcePrefix: "*", TargetModel: "gpt-4", Protocol: "openai"}}
	opts := &ConvertOptions{
		Model:        "gpt-4",
		MaxTokens:    8192,
		ModelMap:     mm,
		SessionStore: store,
		SID:          "test-sid",
		Direction:    "response",
	}
	body := `{"candidates":[{"content":{"parts":[{"text":"Hello!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}`
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(o.Choices) != 1 || o.Choices[0].Message.Content != "Hello!" {
		t.Fatalf("expected OpenAI response, got %+v", o.Choices)
	}
	if o.Choices[0].FinishReason == nil || *o.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason: want stop, got %v", o.Choices[0].FinishReason)
	}
}

// ---------------------------------------------------------------------------
// URI detection
// ---------------------------------------------------------------------------

func TestGemini_DetectByURI(t *testing.T) {
	tests := []struct {
		uri string
		dir Direction
	}{
		{"/v1beta/models/gemini-2.5-flash:generateContent", DirectionRequest},
		{"/v1/models/gemini-2.5-pro:streamGenerateContent?alt=sse", DirectionRequest},
	}
	for _, tt := range tests {
		t.Run(tt.uri, func(t *testing.T) {
			if got := detectByURI(tt.uri, tt.dir); got != ProtocolGemini {
				t.Fatalf("detectByURI(%q): want gemini, got %s", tt.uri, got)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Protocol string / parse
// ---------------------------------------------------------------------------

func TestGemini_ProtocolString(t *testing.T) {
	if ProtocolGemini.String() != "gemini" {
		t.Fatalf("String: want gemini, got %q", ProtocolGemini.String())
	}
}

func TestGemini_ParseProtocol(t *testing.T) {
	if got := parseProtocol("gemini"); got != ProtocolGemini {
		t.Fatalf("parseProtocol: want gemini, got %s", got)
	}
}

// ---------------------------------------------------------------------------
// Blocked response → error
// ---------------------------------------------------------------------------

func TestGemini_BlockedResponse_ToOpenAI(t *testing.T) {
	body := `{"candidates":[],"promptFeedback":{"blockReason":"SAFETY"}}`
	b, err := convertGeminiResponseToOpenAI([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if raw["error"] == nil {
		t.Fatal("want error response")
	}
	errObj, _ := raw["error"].(map[string]any)
	if errObj["type"] != "content_filter" {
		t.Fatalf("want error.type content_filter, got %v", errObj["type"])
	}
}

func TestGemini_BlockedResponse_ToAnthropic(t *testing.T) {
	body := `{"candidates":[],"promptFeedback":{"blockReason":"SAFETY"}}`
	b, err := convertGeminiResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if raw["type"] != "error" {
		t.Fatalf("want type: error, got %q", raw["type"])
	}
	errObj, _ := raw["error"].(map[string]any)
	if errObj["type"] != "permission_error" {
		t.Fatalf("want error.type permission_error, got %v", errObj["type"])
	}
}

// ---------------------------------------------------------------------------
// SSE streaming via HandleSSEEvent — Gemini → OpenAI delta chunks
// ---------------------------------------------------------------------------

func TestGemini_SSEStream_ToOpenAI(t *testing.T) {
	store := NewSessionStore()
	store.Set("test-sid", &Session{
		ID: "test-sid", From: ProtocolOpenAIChat,
	})

	opts := &ConvertOptions{
		Model:        "gemini-2.5-flash",
		MaxTokens:    8192,
		SessionStore: store,
		Direction:    "response",
	}

	// Chunk 1: text delta.
	chunk1 := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}]}` + "\n"
	out, err := HandleSSEEvent("test-sid", "start", 0, []byte(chunk1), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected non-empty output from start phase")
	}
	// Should be an OpenAI delta: data: {"id":"chatcmpl-gemini-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]}
	if string(out[:5]) != "data:" {
		t.Fatalf("expected data: prefix, got %q", string(out[:5]))
	}
	if !bytes.Contains(out, []byte(`"delta":{"content":"Hello"}`)) {
		t.Fatalf("expected content delta Hello, got %s", out)
	}

	// Chunk 2: more text with finish reason.
	chunk2 := `data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP","index":0}]}` + "\n"
	out, err = HandleSSEEvent("test-sid", "event", 1, []byte(chunk2), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected output from event phase")
	}
	if !bytes.Contains(out, []byte(`"content":" world"`)) {
		t.Fatalf("expected content delta ' world', got %s", out)
	}

	// End phase: finish chunk + [DONE].
	out, err = HandleSSEEvent("test-sid", "end", 0, []byte{}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if out == nil {
		t.Fatal("expected output from end phase")
	}
	if !bytes.Contains(out, []byte(`"finish_reason":"stop"`)) {
		t.Fatalf("expected finish_reason in end phase, got %s", out)
	}
	if !bytes.Contains(out, []byte("data: [DONE]")) {
		t.Fatalf("expected [DONE] in end phase, got %s", out)
	}
	// Session cleaned up.
	if store.Get("test-sid") != nil {
		t.Fatal("session should be deleted after end phase")
	}
}

func TestGemini_SSEStream_ToOpenAI_WithFinishReason(t *testing.T) {
	store := NewSessionStore()
	store.Set("test-sid", &Session{
		ID: "test-sid", From: ProtocolOpenAIChat,
	})

	opts := &ConvertOptions{
		Model:        "gemini-2.5-flash",
		MaxTokens:    8192,
		SessionStore: store,
		Direction:    "response",
	}

	// Single chunk with text and finish reason.
	data := `data: {"candidates":[{"content":{"parts":[{"text":"done"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":10,"totalTokenCount":15}}` + "\n"
	out, err := HandleSSEEvent("test-sid", "start", 0, []byte(data), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"delta":{"content":"done"}`)) {
		t.Fatalf("expected content delta, got %s", out)
	}

	// End phase should emit finish_reason + usage.
	out, err = HandleSSEEvent("test-sid", "end", 0, []byte{}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"finish_reason":"stop"`)) {
		t.Fatalf("expected finish_reason stop, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"prompt_tokens":5`)) {
		t.Fatalf("expected prompt_tokens, got %s", out)
	}
	if !bytes.Contains(out, []byte("data: [DONE]")) {
		t.Fatalf("expected [DONE], got %s", out)
	}
}

// ---------------------------------------------------------------------------
// SSE streaming — Gemini → Anthropic event sequence
// ---------------------------------------------------------------------------

func TestGemini_SSEStream_ToAnthropic(t *testing.T) {
	store := NewSessionStore()
	store.Set("test-sid", &Session{
		ID: "test-sid", From: ProtocolAnthropic,
	})

	opts := &ConvertOptions{
		Model:        "claude-sonnet",
		MaxTokens:    8192,
		SessionStore: store,
		Direction:    "response",
	}

	// Start phase: first chunk → message_start + ping + content_block_start + text delta.
	chunk1 := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}]}` + "\n"
	out, err := HandleSSEEvent("test-sid", "start", 0, []byte(chunk1), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected output from start phase")
	}
	// Should start with message_start.
	if !bytes.Contains(out, []byte(`event: message_start`)) {
		t.Fatalf("expected message_start event, got %s", out)
	}
	// Should have content_block_start for text.
	if !bytes.Contains(out, []byte(`event: content_block_start`)) {
		t.Fatalf("expected content_block_start, got %s", out)
	}
	// Should have text delta.
	if !bytes.Contains(out, []byte(`text_delta`)) {
		t.Fatalf("expected text_delta, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"text":"Hello"`)) {
		t.Fatalf("expected text Hello, got %s", out)
	}

	sess := store.Get("test-sid")
	if sess == nil {
		t.Fatal("session should exist")
	}
	if _, ok := sess.StreamHandler.(*GeminiToAnthropicStreamConverter); !ok {
		t.Fatalf("expected GeminiToAnthropicStreamConverter, got %T", sess.StreamHandler)
	}

	// Event phase: more text → content_block_delta.
	chunk2 := `data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"index":0}]}` + "\n"
	out, err = HandleSSEEvent("test-sid", "event", 1, []byte(chunk2), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`event: content_block_delta`)) {
		t.Fatalf("expected content_block_delta, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"text":" world"`)) {
		t.Fatalf("expected text ' world', got %s", out)
	}

	// End phase: content_block_stop + message_delta + message_stop.
	out, err = HandleSSEEvent("test-sid", "end", 0, []byte{}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if out == nil {
		t.Fatal("expected output from end phase")
	}
	if !bytes.Contains(out, []byte(`event: content_block_stop`)) {
		t.Fatalf("expected content_block_stop, got %s", out)
	}
	if !bytes.Contains(out, []byte(`event: message_delta`)) {
		t.Fatalf("expected message_delta, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"stop_reason":"end_turn"`)) {
		t.Fatalf("expected end_turn, got %s", out)
	}
	if !bytes.Contains(out, []byte(`event: message_stop`)) {
		t.Fatalf("expected message_stop, got %s", out)
	}
	if store.Get("test-sid") != nil {
		t.Fatal("session should be deleted after end phase")
	}
}

func TestGemini_SSEStream_ToOpenAI_ErrorHandler(t *testing.T) {
	store := NewSessionStore()
	handler := NewGeminiToOpenAIStreamConverter("gemini-2.5-flash")
	store.Set("test-sid", &Session{
		ID: "test-sid", From: ProtocolOpenAIChat, To: ProtocolGemini,
		StreamHandler: handler,
	})

	opts := &ConvertOptions{
		SessionStore: store,
		Direction:    "response",
		ErrorMsg:     "test error",
	}

	out, err := HandleSSEEvent("test-sid", "error", 0, []byte{}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected error event output")
	}
	if string(out[:5]) != "data:" {
		t.Fatalf("expected data: prefix for OpenAI error, got %q", string(out[:5]))
	}
}

func TestGemini_SSEStream_ToAnthropic_ErrorHandler(t *testing.T) {
	store := NewSessionStore()
	handler := NewGeminiToAnthropicStreamConverter("claude-sonnet")
	store.Set("test-sid", &Session{
		ID: "test-sid", From: ProtocolAnthropic, To: ProtocolGemini,
		StreamHandler: handler,
	})

	opts := &ConvertOptions{
		SessionStore: store,
		Direction:    "response",
		ErrorMsg:     "test error",
	}

	out, err := HandleSSEEvent("test-sid", "error", 0, []byte{}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("expected error event output")
	}
	if !bytes.Contains(out, []byte("event: error")) {
		t.Fatalf("expected event: error, got %q", out)
	}
}

func TestGemini_SSEStream_ToAnthropic_Blocked(t *testing.T) {
	store := NewSessionStore()
	handler := NewGeminiToAnthropicStreamConverter("claude-sonnet")
	store.Set("test-sid", &Session{
		ID: "test-sid", From: ProtocolAnthropic, To: ProtocolGemini,
		StreamHandler: handler,
	})

	// Process blocked response through the stream converter directly.
	blocked := `{"candidates":[],"promptFeedback":{"blockReason":"SAFETY"}}`
	out, err := handler.HandleChunk([]byte(blocked))
	if err != nil {
		t.Fatal(err)
	}
	if out == nil {
		t.Fatal("expected error output for blocked response")
	}
	if !bytes.Contains(out, []byte("event: error")) {
		t.Fatalf("expected event: error, got %q", out)
	}
}

func TestGemini_SSEStream_ToOpenAI_Blocked(t *testing.T) {
	handler := NewGeminiToOpenAIStreamConverter("gemini-2.5-flash")
	blocked := `{"candidates":[],"promptFeedback":{"blockReason":"SAFETY"}}`
	out, err := handler.HandleChunk([]byte(blocked))
	if err != nil {
		t.Fatal(err)
	}
	if out == nil {
		t.Fatal("expected error output for blocked response")
	}
	if !bytes.Contains(out, []byte("content_filter")) {
		t.Fatalf("expected content_filter error, got %q", out)
	}
}

func TestGemini_SSEStream_ToOpenAI_FunctionCall(t *testing.T) {
	handler := NewGeminiToOpenAIStreamConverter("gemini-2.5-flash")

	// Gemini chunk with function call.
	chunk := `{"candidates":[{"content":{"parts":[{"text":"Checking weather"},{"functionCall":{"name":"get_weather","args":{"loc":"NYC"}}}],"role":"model"},"finishReason":"STOP","index":0}]}`
	out, err := handler.HandleChunk([]byte(chunk))
	if err != nil {
		t.Fatal(err)
	}
	if out == nil {
		t.Fatal("expected output for function call chunk")
	}
	// Should have text delta.
	if !bytes.Contains(out, []byte(`"content":"Checking weather"`)) {
		t.Fatalf("expected content delta, got %s", out)
	}
	// Should have tool call delta.
	if !bytes.Contains(out, []byte(`"tool_calls"`)) {
		t.Fatalf("expected tool_calls delta, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"name":"get_weather"`)) {
		t.Fatalf("expected get_weather, got %s", out)
	}
}

func TestGemini_SSEStream_ToAnthropic_FunctionCall(t *testing.T) {
	handler := NewGeminiToAnthropicStreamConverter("claude-sonnet")

	// message_start needed before chunks.
	_ = handler.HandleStreamStart()

	// Gemini chunk with function call.
	chunk := `{"candidates":[{"content":{"parts":[{"text":"Checking weather"},{"functionCall":{"name":"get_weather","args":{"loc":"NYC"}}}],"role":"model"},"finishReason":"STOP","index":0}]}`
	out, err := handler.HandleChunk([]byte(chunk))
	if err != nil {
		t.Fatal(err)
	}
	if out == nil {
		t.Fatal("expected output for function call chunk")
	}
	// Should have text block.
	if !bytes.Contains(out, []byte(`text_delta`)) {
		t.Fatalf("expected text_delta, got %s", out)
	}
	// Should have tool_use block.
	if !bytes.Contains(out, []byte(`content_block_start`)) {
		t.Fatalf("expected content_block_start for tool_use, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"name":"get_weather"`)) {
		t.Fatalf("expected get_weather, got %s", out)
	}
	if !bytes.Contains(out, []byte(`input_json_delta`)) {
		t.Fatalf("expected input_json_delta, got %s", out)
	}

	// End: should close both blocks.
	out = handler.HandleStreamEnd()
	if !bytes.Contains(out, []byte(`content_block_stop`)) {
		t.Fatalf("expected content_block_stop, got %s", out)
	}
}


// ---------------------------------------------------------------------------
// Multi-chunk streaming via HandleSSEEvent
// ---------------------------------------------------------------------------

func TestGemini_SSEStream_ToOpenAI_MultiChunk(t *testing.T) {
	store := NewSessionStore()
	store.Set("test-sid", &Session{ID: "test-sid", From: ProtocolOpenAIChat})

	opts := &ConvertOptions{
		Model:        "gemini-2.5-flash",
		MaxTokens:    8192,
		SessionStore: store,
		Direction:    "response",
	}

	// Chunk 1: partial text.
	chunk1 := `data: {"candidates":[{"content":{"parts":[{"text":"The"}],"role":"model"},"index":0}]}` + "\n"
	out, err := HandleSSEEvent("test-sid", "start", 0, []byte(chunk1), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"content":"The"`)) {
		t.Fatalf("expected 'The', got %s", out)
	}

	// Chunk 2: continuation.
	chunk2 := `data: {"candidates":[{"content":{"parts":[{"text":" quick"}],"role":"model"},"index":0}]}` + "\n"
	out, err = HandleSSEEvent("test-sid", "event", 1, []byte(chunk2), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"content":" quick"`)) {
		t.Fatalf("expected ' quick', got %s", out)
	}

	// Chunk 3: final text with finish reason.
	chunk3 := `data: {"candidates":[{"content":{"parts":[{"text":" brown fox"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":15,"totalTokenCount":20}}` + "\n"
	out, err = HandleSSEEvent("test-sid", "event", 2, []byte(chunk3), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"content":" brown fox"`)) {
		t.Fatalf("expected ' brown fox', got %s", out)
	}

	// End: final chunk with finish_reason + usage, then [DONE].
	out, err = HandleSSEEvent("test-sid", "end", 0, []byte{}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"finish_reason":"stop"`)) {
		t.Fatalf("expected finish_reason, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"prompt_tokens":5`)) {
		t.Fatalf("expected prompt_tokens, got %s", out)
	}
	if !bytes.Contains(out, []byte("data: [DONE]")) {
		t.Fatalf("expected [DONE], got %s", out)
	}
}

func TestGemini_SSEStream_ToAnthropic_MultiChunk(t *testing.T) {
	store := NewSessionStore()
	store.Set("test-sid", &Session{ID: "test-sid", From: ProtocolAnthropic})

	opts := &ConvertOptions{
		Model:        "claude-sonnet",
		MaxTokens:    8192,
		SessionStore: store,
		Direction:    "response",
	}

	// Chunk 1: first text.
	chunk1 := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}]}` + "\n"
	out, err := HandleSSEEvent("test-sid", "start", 0, []byte(chunk1), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`event: message_start`)) {
		t.Fatalf("expected message_start, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"text":"Hello"`)) {
		t.Fatalf("expected 'Hello', got %s", out)
	}

	// Chunk 2: continuation.
	chunk2 := `data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"index":0}]}` + "\n"
	out, err = HandleSSEEvent("test-sid", "event", 1, []byte(chunk2), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`event: content_block_delta`)) {
		t.Fatalf("expected content_block_delta, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"text":" world"`)) {
		t.Fatalf("expected ' world', got %s", out)
	}

	// Chunk 3: final text with finish reason.
	chunk3 := `data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":12,"totalTokenCount":17}}` + "\n"
	out, err = HandleSSEEvent("test-sid", "event", 2, []byte(chunk3), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`"text":"!"`)) {
		t.Fatalf("expected '!', got %s", out)
	}

	// End: content_block_stop + message_delta + message_stop.
	out, err = HandleSSEEvent("test-sid", "end", 0, []byte{}, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`event: content_block_stop`)) {
		t.Fatalf("expected content_block_stop, got %s", out)
	}
	if !bytes.Contains(out, []byte(`event: message_delta`)) {
		t.Fatalf("expected message_delta, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"stop_reason":"end_turn"`)) {
		t.Fatalf("expected end_turn, got %s", out)
	}
	if !bytes.Contains(out, []byte(`event: message_stop`)) {
		t.Fatalf("expected message_stop, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"input_tokens":5`)) {
		t.Fatalf("expected input_tokens 5, got %s", out)
	}
	if !bytes.Contains(out, []byte(`"output_tokens":12`)) {
		t.Fatalf("expected output_tokens 12, got %s", out)
	}
	if store.Get("test-sid") != nil {
		t.Fatal("session should be deleted after end phase")
	}
}
