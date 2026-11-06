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

// ---------------------------------------------------------------------------
// P1-4/P1-5/P1-7/P1-8: thinking, reasoning replay, effort, metadata gating
// ---------------------------------------------------------------------------

func TestP1_UnsignedThinkingDroppedForAnthropic(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"hi","reasoning_content":"chain",
		 "tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024, ResolvedModel: "gpt-4"})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	for _, block := range anth.Messages[0].Content {
		if block.Type == "thinking" {
			t.Fatalf("unsigned thinking sent to Anthropic target: %+v", block)
		}
	}

	b, _ = convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 1024, ResolvedModel: "deepseek-chat"})
	_ = json.Unmarshal(b, &anth)
	found := false
	for _, block := range anth.Messages[0].Content {
		found = found || block.Type == "thinking"
	}
	if !found {
		t.Fatal("DeepSeek target lost its thinking block")
	}
}

func TestP1_ReasoningContentGatedByTargetModel(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"messages":[
		{"role":"assistant","content":[{"type":"thinking","thinking":"chain"},{"type":"tool_use","id":"a","name":"f","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`

	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "" {
		t.Fatalf("gpt-4o received reasoning_content: %q", oai.Messages[0].ReasoningContent)
	}

	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "chain" {
		t.Fatalf("deepseek reasoning = %q, want chain", oai.Messages[0].ReasoningContent)
	}
}

func TestP1_DeepSeekPlaceholderOnlyForDeepSeek(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`

	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o", ReasoningCache: NewReasoningCache(10)})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "" {
		t.Fatalf("non-DeepSeek placeholder: %q", oai.Messages[0].ReasoningContent)
	}

	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat", ReasoningCache: NewReasoningCache(10)})
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent == "" {
		t.Fatal("DeepSeek lost its compatibility placeholder")
	}
}

func TestP1_EffortRequiresThinking(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"output_config":{"effort":"high"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.ReasoningEffort != nil {
		t.Fatalf("effort forwarded without thinking: %v", oai.ReasoningEffort)
	}

	body = `{"model":"claude","max_tokens":10,"thinking":{"type":"enabled","budget_tokens":1024},
		"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`
	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	_ = json.Unmarshal(b, &oai)
	if oai.ReasoningEffort != "high" {
		t.Fatalf("effort = %v, want high", oai.ReasoningEffort)
	}
}

func TestP1_MetadataWhitelist(t *testing.T) {
	body := `{"model":"gpt-4","user":"u2","metadata":{"user_id":"u1","trace":"x"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	m, _ := anth.Metadata.(map[string]any)
	if len(m) != 1 || m["user_id"] != "u1" {
		t.Fatalf("metadata = %+v, want only user_id=u1", anth.Metadata)
	}

	body = `{"model":"gpt-4","user":"u2","metadata":{"trace":"x"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ = convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	_ = json.Unmarshal(b, &anth)
	m, _ = anth.Metadata.(map[string]any)
	if len(m) != 1 || m["user_id"] != "u2" {
		t.Fatalf("metadata = %+v, want fallback user_id=u2", anth.Metadata)
	}
}

func TestP1_ResponsesUnsignedThinkingAndMetadata(t *testing.T) {
	body := `{"model":"gpt-4o","metadata":{"user_id":"u1","trace":"x"},"input":[
		{"type":"reasoning","text":"chain"},
		{"type":"function_call","call_id":"a","name":"f","arguments":"{}"}]}`
	b, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	var anth AnthropicRequest
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatal(err)
	}
	for _, msg := range anth.Messages {
		for _, block := range msg.Content {
			if block.Type == "thinking" {
				t.Fatalf("Responses unsigned thinking sent to Anthropic: %+v", block)
			}
		}
	}
	m, _ := anth.Metadata.(map[string]any)
	if len(m) != 1 || m["user_id"] != "u1" {
		t.Fatalf("Responses metadata = %+v, want only user_id=u1", anth.Metadata)
	}
}

func TestP1_ReplaysReasoningTargets(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"deepseek-chat", true},
		{"deepseek-v4-pro", true},
		{"glm-4.6", true},
		{"gpt-4o", false},
		{"claude-sonnet-4-5", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := replaysReasoning(tc.model); got != tc.want {
			t.Errorf("replaysReasoning(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestP1_AnthropicMetadataHelper(t *testing.T) {
	cases := []struct {
		name     string
		metadata any
		user     string
		want     any
	}{
		{"keeps user_id", map[string]any{"user_id": "u1", "trace": "x"}, "u2", map[string]any{"user_id": "u1"}},
		{"string map", map[string]string{"user_id": "u3", "trace": "x"}, "", map[string]any{"user_id": "u3"}},
		{"falls back to user", map[string]any{"trace": "x"}, "u2", map[string]any{"user_id": "u2"}},
		{"empty user_id falls back", map[string]any{"user_id": ""}, "u4", map[string]any{"user_id": "u4"}},
		{"no source", nil, "", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := anthropicMetadata(tc.metadata, tc.user)
			gotMap, gotOK := got.(map[string]any)
			wantMap, wantOK := tc.want.(map[string]any)
			if gotOK != wantOK || len(gotMap) != len(wantMap) || gotMap["user_id"] != wantMap["user_id"] {
				t.Fatalf("anthropicMetadata(%v, %q) = %v, want %v", tc.metadata, tc.user, got, tc.want)
			}
		})
	}
}

func TestP1_GLMReplaysReasoningWithoutPlaceholder(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"messages":[
		{"role":"assistant","content":[{"type":"thinking","thinking":"chain"},{"type":"tool_use","id":"a","name":"f","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "glm-4.6"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "chain" {
		t.Fatalf("glm reasoning = %q, want chain", oai.Messages[0].ReasoningContent)
	}

	body = `{"model":"claude","max_tokens":10,"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`
	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "glm-4.6", ReasoningCache: NewReasoningCache(10)})
	oai = OpenAIChatRequest{}
	_ = json.Unmarshal(b, &oai)
	if oai.Messages[0].ReasoningContent != "" {
		t.Fatalf("glm placeholder: %q", oai.Messages[0].ReasoningContent)
	}
}

func TestP1_ReasoningModelUsesMaxCompletionTokens(t *testing.T) {
	body := `{"model":"claude","max_tokens":100,"temperature":0.5,"top_p":0.9,
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "o3-mini"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.MaxCompletionTokens == nil || *oai.MaxCompletionTokens != 100 {
		t.Fatalf("max_completion_tokens = %v, want 100", oai.MaxCompletionTokens)
	}
	if oai.MaxTokens != nil || oai.Temperature != nil || oai.TopP != nil {
		t.Fatalf("reasoning model kept unsupported params: %s", b)
	}

	b, _ = convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	_ = json.Unmarshal(b, &oai)
	if oai.MaxTokens == nil || *oai.MaxTokens != 100 {
		t.Fatalf("non-reasoning max_tokens = %v, want 100", oai.MaxTokens)
	}
	if oai.Temperature == nil || oai.TopP == nil {
		t.Fatalf("non-reasoning sampling params dropped: %s", b)
	}
}

func TestP1_NonStreamToolStopInference(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop",
		"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	b, _ := convertOpenAIResponseToAnthropic([]byte(body), nil)
	var anth AnthropicResponse
	_ = json.Unmarshal(b, &anth)
	if anth.StopReason == nil || *anth.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %v, want tool_use", anth.StopReason)
	}

	body = `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"content_filter",
		"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	b, _ = convertOpenAIResponseToAnthropic([]byte(body), nil)
	_ = json.Unmarshal(b, &anth)
	if anth.StopReason == nil || *anth.StopReason != "end_turn" {
		t.Fatalf("content_filter stop_reason = %v, want end_turn", anth.StopReason)
	}

	body = `{"id":"msg_1","model":"claude","stop_reason":"end_turn",
		"content":[{"type":"tool_use","id":"a","name":"f","input":{}}]}`
	b, _ = convertAnthropicResponseToOpenAI([]byte(body))
	var oai OpenAIChatResponse
	_ = json.Unmarshal(b, &oai)
	if oai.Choices[0].FinishReason == nil || *oai.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish_reason = %v, want tool_calls", oai.Choices[0].FinishReason)
	}
}

func TestP1_StreamToolStopInference(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`))
	out := string(sc.HandleStreamEnd())
	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Fatalf("OpenAI→Anthropic stop = %s", out)
	}

	c := NewAnthropicToOpenAIStreamConverter("claude")
	_, _ = c.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_1","model":"claude"}}`))
	_, _ = c.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f"}}`))
	_, _ = c.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`))
	out = string(c.HandleStreamEnd())
	if !strings.Contains(out, `"finish_reason":"tool_calls"`) {
		t.Fatalf("Anthropic→OpenAI finish = %s", out)
	}
}

func TestP1_ReasoningModelDetection(t *testing.T) {
	cases := map[string]bool{
		"o1":             true,
		"o1-mini":        true,
		"o3-mini":        true,
		"o4-mini":        true,
		"gpt-5":          true,
		"gpt-5.1-codex":  true,
		"openai/o3-mini": true,
		"openai/gpt-5":   true,
		"O3-MINI":        true,
		"gpt-4o":         false,
		"gpt-4.1":        false,
		"openai/gpt-4o":  false,
		"claude-opus-4":  false,
		"deepseek-chat":  false,
		"":               false,
	}
	for model, want := range cases {
		if got := isOpenAIReasoningModel(model); got != want {
			t.Errorf("isOpenAIReasoningModel(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestP1_ReasoningModelProviderPrefixMapping(t *testing.T) {
	body := `{"model":"claude","max_tokens":64,"temperature":0.2,
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "openai/gpt-5"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.MaxCompletionTokens == nil || *oai.MaxCompletionTokens != 64 {
		t.Fatalf("max_completion_tokens = %v, want 64: %s", oai.MaxCompletionTokens, b)
	}
	if oai.MaxTokens != nil || oai.Temperature != nil {
		t.Fatalf("provider-prefixed reasoning model kept unsupported params: %s", b)
	}
}

func TestP1_ToolStopExplicitReasonsPreserved(t *testing.T) {
	cases := []struct {
		finishReason string
		want         string
	}{
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
	}
	for _, tc := range cases {
		body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"` + tc.finishReason + `",
			"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
		b, _ := convertOpenAIResponseToAnthropic([]byte(body), nil)
		var anth AnthropicResponse
		_ = json.Unmarshal(b, &anth)
		if anth.StopReason == nil || *anth.StopReason != tc.want {
			t.Fatalf("finish_reason %q → stop_reason %v, want %s", tc.finishReason, anth.StopReason, tc.want)
		}
	}

	// A missing OpenAI finish_reason with tool calls still infers tool_use.
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,
		"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	b, _ := convertOpenAIResponseToAnthropic([]byte(body), nil)
	var anth AnthropicResponse
	_ = json.Unmarshal(b, &anth)
	if anth.StopReason == nil || *anth.StopReason != "tool_use" {
		t.Fatalf("nil finish_reason stop_reason = %v, want tool_use", anth.StopReason)
	}

	// Anthropic max_tokens with tools keeps length.
	body = `{"id":"msg_1","model":"claude","stop_reason":"max_tokens",
		"content":[{"type":"tool_use","id":"a","name":"f","input":{}}]}`
	b, _ = convertAnthropicResponseToOpenAI([]byte(body))
	var oai OpenAIChatResponse
	_ = json.Unmarshal(b, &oai)
	if oai.Choices[0].FinishReason == nil || *oai.Choices[0].FinishReason != "length" {
		t.Fatalf("max_tokens finish_reason = %v, want length", oai.Choices[0].FinishReason)
	}

	// Anthropic missing stop_reason with tools infers tool_calls.
	body = `{"id":"msg_1","model":"claude",
		"content":[{"type":"tool_use","id":"a","name":"f","input":{}}]}`
	b, _ = convertAnthropicResponseToOpenAI([]byte(body))
	oai = OpenAIChatResponse{}
	_ = json.Unmarshal(b, &oai)
	if oai.Choices[0].FinishReason == nil || *oai.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("nil stop_reason finish_reason = %v, want tool_calls", oai.Choices[0].FinishReason)
	}
}

func TestP1_StreamToolStopExplicitReasonsPreserved(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`))
	if out := string(sc.HandleStreamEnd()); !strings.Contains(out, `"stop_reason":"max_tokens"`) {
		t.Fatalf("length stream stop = %s", out)
	}

	c := NewAnthropicToOpenAIStreamConverter("claude")
	_, _ = c.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_1","model":"claude"}}`))
	_, _ = c.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f"}}`))
	_, _ = c.HandleChunk([]byte(`{"type":"message_delta","delta":{"stop_reason":"max_tokens"}}`))
	if out := string(c.HandleStreamEnd()); !strings.Contains(out, `"finish_reason":"length"`) {
		t.Fatalf("max_tokens stream finish = %s", out)
	}
}

// ---------------------------------------------------------------------------
// P1-9/P1-10: URL images and structured output
// ---------------------------------------------------------------------------

func TestP1_OpenAIImageURLPreserved(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":[
		{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	img := anth.Messages[0].Content[1]
	if img.Type != "image" || img.Source == nil || img.Source.Type != "url" || img.Source.URL != "https://example.com/cat.png" {
		t.Fatalf("image = %+v", img)
	}
}

func TestP1_AnthropicURLImagePreserved(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":[
		{"type":"text","text":"see"},{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}]}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	parts, _ := oai.Messages[0].Content.([]any)
	if len(parts) != 2 {
		t.Fatalf("parts = %+v", parts)
	}
	img, _ := parts[1].(map[string]any)
	imageURL, _ := img["image_url"].(map[string]any)
	if imageURL["url"] != "https://example.com/cat.png" {
		t.Fatalf("image part = %+v", img)
	}
}

func TestP1_StructuredOutputJSONSchemaToAnthropic(t *testing.T) {
	body := `{"model":"gpt-4","response_format":{"type":"json_schema","json_schema":{
		"name":"out","schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	if anth.OutputConfig == nil || anth.OutputConfig.Format == nil || anth.OutputConfig.Format.Type != "json_schema" {
		t.Fatalf("output_config = %+v", anth.OutputConfig)
	}
	schema, _ := anth.OutputConfig.Format.Schema.(map[string]any)
	if schema["type"] != "object" {
		t.Fatalf("schema = %+v", schema)
	}
}

func TestP1_AnthropicStructuredOutputToOpenAI(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"output_config":{"format":{
		"type":"json_schema","schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	if oai.ResponseFormat == nil || oai.ResponseFormat.Type != "json_schema" || oai.ResponseFormat.JSONSchema == nil {
		t.Fatalf("response_format = %+v", oai.ResponseFormat)
	}
	if oai.ResponseFormat.JSONSchema.Name != "response" {
		t.Fatalf("json_schema.name = %q, want response", oai.ResponseFormat.JSONSchema.Name)
	}
}

func TestP1_JSONObjectFallsBackToInstruction(t *testing.T) {
	body := `{"model":"gpt-4","response_format":{"type":"json_object"},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	if anth.OutputConfig != nil && anth.OutputConfig.Format != nil {
		t.Fatalf("json_object emitted a schema: %+v", anth.OutputConfig)
	}
	found := false
	for _, block := range anth.System {
		found = found || strings.Contains(strings.ToLower(block.Text), "json")
	}
	if !found {
		t.Fatalf("no JSON instruction in system: %+v", anth.System)
	}
}

func TestP1_JSONSchemaWithoutSchemaFallsBackToInstruction(t *testing.T) {
	body := `{"model":"gpt-4","response_format":{"type":"json_schema","json_schema":{"name":"out"}},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 10})
	var anth AnthropicRequest
	_ = json.Unmarshal(b, &anth)
	if anth.OutputConfig != nil {
		t.Fatalf("schema-less json_schema emitted output_config: %+v", anth.OutputConfig)
	}
	if len(anth.System) != 1 || !strings.Contains(anth.System[0].Text, jsonOutputInstruction) {
		t.Fatalf("system = %+v, want the JSON instruction", anth.System)
	}
}

func TestP1_StructuredInstructionMergesWithToolChoice(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"system":"be brief",
		"tools":[{"name":"f","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"any"},
		"output_config":{"format":{"type":"json_schema"}},
		"messages":[{"role":"user","content":"hi"}]}`
	b, _ := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	var oai OpenAIChatRequest
	_ = json.Unmarshal(b, &oai)
	var systems []string
	for _, msg := range oai.Messages {
		if msg.Role == "system" {
			text, _ := msg.Content.(string)
			systems = append(systems, text)
		}
	}
	if len(systems) != 1 {
		t.Fatalf("system messages = %+v, want exactly one", systems)
	}
	want := "be brief\n\nThe caller requires a tool call for this turn. Call one of the available tools instead of answering directly.\n\n" + jsonOutputInstruction
	if systems[0] != want {
		t.Fatalf("system text = %q, want %q", systems[0], want)
	}
}
