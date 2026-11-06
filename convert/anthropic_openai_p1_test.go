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

// ---------------------------------------------------------------------------
// P1-2: Anthropic message coalescing and tool-pair repair
// ---------------------------------------------------------------------------

func TestP1_ParallelToolResultsCoalesce(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"start"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},
			{"id":"b","type":"function","function":{"name":"g","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"a","content":"ra"},
		{"role":"tool","tool_call_id":"b","content":"rb"},
		{"role":"user","content":"next"}]}`
	b, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatal(err)
	}
	if len(anth.Messages) != 3 {
		t.Fatalf("messages = %d, want 3: %s", len(anth.Messages), b)
	}
	last := anth.Messages[2].Content
	if len(last) != 3 || last[0].Type != "tool_result" || last[1].Type != "tool_result" || last[2].Type != "text" {
		t.Fatalf("last user blocks = %+v, want two tool_result then text", last)
	}
}

func TestP1_UnansweredToolUseGetsSyntheticResult(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"user","content":"continue"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	last := anth.Messages[len(anth.Messages)-1].Content
	if len(last) != 2 || last[0].Type != "tool_result" || last[0].ToolUseID != "a" || !last[0].IsError {
		t.Fatalf("blocks = %+v, want synthetic tool_result first", last)
	}
	if last[1].Type != "text" || last[1].Text != "continue" {
		t.Fatalf("text block = %+v", last[1])
	}
}

func TestP1_OrphanToolResultBecomesText(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"tool","tool_call_id":"missing","content":"orphan text"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	got := anth.Messages[0].Content[0]
	if got.Type != "text" || got.Text != "orphan text" {
		t.Fatalf("orphan result = %+v, want text block", got)
	}
}

func TestP1_TrailingToolUseGetsSyntheticUserResult(t *testing.T) {
	msgs := []AnthropicMessage{
		{Role: "user", Content: []AnthropicContent{{Type: "text", Text: "hi"}}},
		{Role: "assistant", Content: []AnthropicContent{{Type: "tool_use", ID: "a", Name: "f", Input: map[string]any{}}}},
	}
	got := normalizeAnthropicMessages(msgs)
	if len(got) != 3 || got[2].Role != "user" {
		t.Fatalf("messages = %+v, want trailing synthetic user", got)
	}
	if got[2].Content[0].Type != "tool_result" || got[2].Content[0].ToolUseID != "a" {
		t.Fatalf("trailing result = %+v", got[2].Content)
	}
}

func TestP1_HistoryToolsGetStubDeclarations(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
		{"role":"user","content":"go on"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	if len(anth.Tools) != 1 || anth.Tools[0].Name != "lookup" {
		t.Fatalf("tools = %+v, want one lookup stub", anth.Tools)
	}
	if anth.ToolChoice == nil || anth.ToolChoice.Type != "none" {
		t.Fatalf("tool_choice = %+v, want none", anth.ToolChoice)
	}
}

func TestP1_ResponsesToolHistoryNormalized(t *testing.T) {
	body := `{"model":"gpt-5","input":[
		{"type":"message","role":"user","content":"start"},
		{"type":"function_call","call_id":"a","name":"lookup","arguments":"{}"},
		{"type":"function_call_output","call_id":"a","output":"found"},
		{"type":"message","role":"user","content":"next"}]}`
	b, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024})
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatal(err)
	}
	if len(anth.Messages) != 3 {
		t.Fatalf("messages = %d, want 3: %s", len(anth.Messages), b)
	}
	last := anth.Messages[2].Content
	if len(last) != 2 || last[0].Type != "tool_result" || last[0].ToolUseID != "a" || last[1].Type != "text" {
		t.Fatalf("last user blocks = %+v, want tool_result then text", last)
	}
	if len(anth.Tools) != 1 || anth.Tools[0].Name != "lookup" {
		t.Fatalf("tools = %+v, want one lookup stub", anth.Tools)
	}
	if anth.ToolChoice == nil || anth.ToolChoice.Type != "none" {
		t.Fatalf("tool_choice = %+v, want none", anth.ToolChoice)
	}
}

func TestP1_SyntheticAndOrphanResultText(t *testing.T) {
	msgs := []AnthropicMessage{
		{Role: "assistant", Content: []AnthropicContent{{Type: "tool_use", ID: "a", Name: "f"}}},
		{Role: "user", Content: []AnthropicContent{{Type: "tool_result", ToolUseID: "b"}}},
	}
	got := normalizeAnthropicMessages(msgs)
	if len(got) != 2 || len(got[1].Content) != 2 {
		t.Fatalf("messages = %+v", got)
	}
	synthetic := got[1].Content[0]
	if synthetic.Type != "tool_result" || synthetic.ToolUseID != "a" ||
		synthetic.Content != "Tool result missing." || !synthetic.IsError {
		t.Fatalf("synthetic = %+v", synthetic)
	}
	orphan := got[1].Content[1]
	if orphan.Type != "text" || orphan.Text != "Tool result received." {
		t.Fatalf("orphan = %+v", orphan)
	}
}

func TestP1_StubHistoryToolsSchemaAndDedup(t *testing.T) {
	msgs := []AnthropicMessage{{
		Role: "assistant",
		Content: []AnthropicContent{
			{Type: "tool_use", ID: "a", Name: "lookup"},
			{Type: "tool_use", ID: "b", Name: "lookup"},
		},
	}}
	req := &AnthropicRequest{}
	stubHistoryTools(req, msgs)
	if len(req.Tools) != 1 || req.Tools[0].Name != "lookup" {
		t.Fatalf("tools = %+v, want one lookup stub", req.Tools)
	}
	if req.Tools[0].Description != "Tool from earlier conversation. Do not call." {
		t.Fatalf("description = %q", req.Tools[0].Description)
	}
	schema, ok := req.Tools[0].InputSchema.(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("schema = %+v", req.Tools[0].InputSchema)
	}
	if props, ok := schema["properties"].(map[string]any); !ok || len(props) != 0 {
		t.Fatalf("schema properties = %+v", schema["properties"])
	}
	if req.ToolChoice == nil || req.ToolChoice.Type != "none" {
		t.Fatalf("tool_choice = %+v, want none", req.ToolChoice)
	}

	declared := &AnthropicRequest{Tools: []AnthropicTool{{Name: "declared"}}}
	stubHistoryTools(declared, msgs)
	if len(declared.Tools) != 1 || declared.Tools[0].Name != "declared" || declared.ToolChoice != nil {
		t.Fatalf("declared tools changed: tools=%+v choice=%+v", declared.Tools, declared.ToolChoice)
	}
}

func TestP1_UserRunKeepsResultsFirst(t *testing.T) {
	msgs := []AnthropicMessage{
		{Role: "assistant", Content: []AnthropicContent{{Type: "tool_use", ID: "a", Name: "f"}}},
		{Role: "user", Content: []AnthropicContent{{Type: "text", Text: "wait"}}},
		{Role: "user", Content: []AnthropicContent{{Type: "tool_result", ToolUseID: "a", Content: "ok"}}},
	}
	got := normalizeAnthropicMessages(msgs)
	if len(got) != 2 {
		t.Fatalf("messages = %+v, want assistant + merged user", got)
	}
	blocks := got[1].Content
	if len(blocks) != 2 || blocks[0].Type != "tool_result" || blocks[0].ToolUseID != "a" || blocks[1].Text != "wait" {
		t.Fatalf("blocks = %+v, want tool_result then text", blocks)
	}
}
