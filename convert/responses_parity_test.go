
package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Step 1: input_file / input_audio
// ---------------------------------------------------------------------------

func TestConvertResponsesToChat_InputFile(t *testing.T) {
	opts := &ConvertOptions{Model: "gpt-4o", MaxTokens: 8192}
	body := `{
		"model":"gpt-4o",
		"input":[{"type":"message","role":"user","content":[{"type":"input_file","input_file":{"filename":"doc.txt"}}]}]
	}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(chat.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(chat.Messages))
	}
	text := extractTextContent(chat.Messages[0].Content)
	if !strings.Contains(text, "input_file") {
		t.Fatalf("expected input_file placeholder, got %q", text)
	}
}

func TestConvertResponsesToChat_InputAudio(t *testing.T) {
	opts := &ConvertOptions{Model: "gpt-4o", MaxTokens: 8192}
	body := `{
		"model":"gpt-4o",
		"input":[{"type":"message","role":"user","content":[{"type":"input_audio","input_audio":{"data":"base64..."}}]}]
	}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	text := extractTextContent(chat.Messages[0].Content)
	if !strings.Contains(text, "input_audio") {
		t.Fatalf("expected input_audio placeholder, got %q", text)
	}
}

// ---------------------------------------------------------------------------
// Step 2: billing header stripping
// ---------------------------------------------------------------------------

func TestStripLeadingAnthropicBillingHeader(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"strips leading header", "x-anthropic-billing-header: abc123\nbe concise", "be concise"},
		{"no header unchanged", "be concise", "be concise"},
		{"empty string", "", ""},
		{"only header no newline", "x-anthropic-billing-header: abc123", ""},
		{"preserves subsequent header text", "x-anthropic-billing-header: first\nregular text\nx-anthropic-billing-header: second", "regular text\nx-anthropic-billing-header: second"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripLeadingAnthropicBillingHeader(tt.input)
			if got != tt.want {
				t.Errorf("stripLeadingAnthropicBillingHeader() = %q, want %q", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Step 3: reasoning attachment
// ---------------------------------------------------------------------------

func TestParseResponsesInput_ReasoningDedup(t *testing.T) {
	input := json.RawMessage(`[
		{"type":"message","role":"user","content":"hello"},
		{"type":"reasoning","text":"thinking about this","summary":[{"type":"summary_text","text":"thinking about this"}]},
		{"type":"reasoning","text":"thinking about this","summary":[{"type":"summary_text","text":"thinking about this"}]},
		{"type":"message","role":"assistant","content":"answer"}
	]`)
	result := parseResponsesInput(input)
	if len(result.messages) < 2 {
		t.Fatalf("want at least 2 messages, got %d", len(result.messages))
	}
	asst := result.messages[len(result.messages)-1]
	if asst.ReasoningContent == "" {
		t.Fatal("expected non-empty reasoning_content on assistant")
	}
	if strings.Count(asst.ReasoningContent, "thinking about this") > 1 {
		t.Fatalf("reasoning was duplicated: %q", asst.ReasoningContent)
	}
}

func TestParseResponsesInput_ReasoningAttachToLastAssistant(t *testing.T) {
	input := json.RawMessage(`[
		{"type":"message","role":"user","content":"hello"},
		{"type":"message","role":"assistant","content":"let me think"},
		{"type":"reasoning","text":"trailing thought","summary":[{"type":"summary_text","text":"trailing thought"}]}
	]`)
	result := parseResponsesInput(input)
	if len(result.messages) < 2 {
		t.Fatalf("want at least 2 messages, got %d", len(result.messages))
	}
	asst := result.messages[len(result.messages)-1]
	if asst.Role != "assistant" {
		t.Fatalf("expected assistant as last message, got %s", asst.Role)
	}
	if asst.ReasoningContent != "trailing thought" {
		t.Fatalf("expected reasoning 'trailing thought', got %q", asst.ReasoningContent)
	}
}

func TestParseResponsesInput_ReasoningBackfill(t *testing.T) {
	input := json.RawMessage(`[
		{"type":"message","role":"user","content":"get weather"},
		{"type":"function_call","id":"call_1","name":"get_weather","arguments":"{}"},
		{"type":"function_call_output","call_id":"call_1","output":"\"sunny\""}
	]`)
	result := parseResponsesInput(input)
	if len(result.messages) < 2 {
		t.Fatalf("want at least 2 messages, got %d", len(result.messages))
	}
	found := false
	for _, m := range result.messages {
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			found = true
			if m.ReasoningContent != "tool call" {
				t.Errorf("expected backfill 'tool call', got %q", m.ReasoningContent)
			}
		}
	}
	if !found {
		t.Fatal("no assistant message with tool_calls found")
	}
}

// ---------------------------------------------------------------------------
// Step 4: collapse system messages
// ---------------------------------------------------------------------------

func TestCollapseSystemMessagesToHead(t *testing.T) {
	msgs := []OpenAIMessage{
		{Role: "system", Content: "first instruction"},
		{Role: "user", Content: "hello"},
		{Role: "system", Content: "second instruction"},
		{Role: "assistant", Content: "hi"},
		{Role: "system", Content: ""},
	}
	result := collapseSystemMessagesToHead(msgs)
	if len(result) != 3 {
		t.Fatalf("want 3 messages (1 system + user + assistant), got %d", len(result))
	}
	if result[0].Role != "system" {
		t.Fatalf("first message should be system, got %s", result[0].Role)
	}
	s, ok := result[0].Content.(string)
	if !ok || !strings.Contains(s, "first instruction") || !strings.Contains(s, "second instruction") {
		t.Fatalf("system content should join both, got %q", s)
	}
	if !strings.Contains(s, "\n\n") {
		t.Fatal("system messages should be joined by double newline")
	}
}

// ---------------------------------------------------------------------------
// Step 5: codex tool context
// ---------------------------------------------------------------------------

func TestBuildCodexToolContext(t *testing.T) {
	tools := []ResponsesTool{
		{Name: "get_weather", Type: "function", Description: "Get weather"},
		{Name: "custom_search", Type: "custom", Description: "Custom search"},
		{Name: "web_lookup", Type: "tool_search", Description: "Search web"},
	}
	ctx := buildCodexToolContext(tools)
	if len(ctx.byChatName) != 3 {
		t.Fatalf("want 3 entries, got %d", len(ctx.byChatName))
	}
	if spec := ctx.byChatName["get_weather"]; spec.Kind != "function" {
		t.Errorf("get_weather: want function, got %s", spec.Kind)
	}
	if spec := ctx.byChatName["custom_search"]; spec.Kind != "custom" {
		t.Errorf("custom_search: want custom, got %s", spec.Kind)
	}
	if spec := ctx.byChatName["web_lookup"]; spec.Kind != "tool_search" {
		t.Errorf("web_lookup: want tool_search, got %s", spec.Kind)
	}
	// Reverse mapping: custom tool → custom_tool_call output item.
	tc := OpenAIToolCall{ID: "call_1", Function: OpenAIFunctionCall{Name: "custom_search", Arguments: `{"input":"query"}`}}
	item := ctx.toResponsesOutputItem(tc)
	if item.Type != "custom_tool_call" {
		t.Errorf("custom tool: want custom_tool_call, got %q", item.Type)
	}
	if item.Status != "completed" {
		t.Errorf("custom tool: want completed, got %q", item.Status)
	}
}

func TestConvertResponsesToChat_WithCustomTool(t *testing.T) {
	opts := &ConvertOptions{Model: "gpt-4o", MaxTokens: 8192}
	body := `{
		"model":"gpt-4o",
		"input":"search for something",
		"tools":[{"name":"my_search","type":"custom","description":"Custom search tool","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}]
	}`
	b, err := ConvertResponsesToChat([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var chat OpenAIChatRequest
	if err := json.Unmarshal(b, &chat); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
	}
	if len(chat.Tools) != 1 {
		t.Fatalf("want 1 tool, got %d", len(chat.Tools))
	}
	fn := chat.Tools[0].Function
	if fn.Name != "my_search" {
		t.Fatalf("tool name: want my_search, got %q", fn.Name)
	}
	// Custom tools get a single "input" string parameter schema.
	params, ok := fn.Parameters.(map[string]any)
	if !ok {
		t.Fatal("parameters should be a map")
	}
	if params["type"] != "object" {
		t.Errorf("params type: want object, got %v", params["type"])
	}
	props, _ := params["properties"].(map[string]any)
	if props == nil {
		t.Fatal("params.properties missing")
	}
	inputProp, ok := props["input"].(map[string]any)
	if !ok {
		t.Fatalf("custom tool should have 'input' parameter, got properties: %v", props)
	}
	if inputProp["type"] != "string" {
		t.Errorf("input param type: want string, got %v", inputProp["type"])
	}
}

// ---------------------------------------------------------------------------
// Step 7: error normalization
// ---------------------------------------------------------------------------

func TestChatErrorToResponseError(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"openai error", `{"error":{"message":"bad request","type":"invalid_request_error","code":"invalid_model"}}`},
		{"bare string error", `{"error":"something went wrong"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := chatErrorToResponseError([]byte(tt.body))
			if err != nil {
				t.Fatal(err)
			}
			var r ResponsesResponse
			if err := json.Unmarshal(b, &r); err != nil {
				t.Fatalf("unmarshal error: %v\nbody: %s", err, b)
			}
			if r.Status != "failed" {
				t.Errorf("status: want failed, got %q", r.Status)
			}
			if r.Error == nil {
				t.Fatal("expected non-nil error field")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Step 8: streaming inline think
// ---------------------------------------------------------------------------

func TestResponsesStreamConverter_InlineThink(t *testing.T) {
	sc := NewResponsesStreamConverter("deepseek-chat")
	sc.HandleStreamStart()

	c1 := `{"choices":[{"index":0,"delta":{"content":"<think>let me reason</think>the answer is 42"},"finish_reason":null}]}`
	out1, err := sc.HandleChunk([]byte(c1))
	if err != nil {
		t.Fatal(err)
	}
	outStr := string(out1)
	if !strings.Contains(outStr, "reasoning_summary_text.delta") {
		t.Fatal("missing reasoning_summary_text.delta for inline think")
	}
	if !strings.Contains(outStr, `"delta":"let me reason"`) {
		t.Fatal("missing reasoning content in inline think delta")
	}
	if !strings.Contains(outStr, "output_text.delta") {
		t.Fatal("missing output_text.delta for text after </think>")
	}
	if !strings.Contains(outStr, `"delta":"the answer is 42"`) {
		t.Fatal("missing text content after inline think")
	}

	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	end := sc.HandleStreamEnd()
	if !strings.Contains(string(end), "response.completed") {
		t.Fatal("missing response.completed")
	}
}
