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
	if len(a.Messages) != 1 {
		t.Fatalf("want 1 message, got %d: %s", len(a.Messages), b)
	}
	if a.Messages[0].Role != "user" || len(a.Messages[0].Content) != 1 ||
		a.Messages[0].Content[0].Type != "text" || a.Messages[0].Content[0].Text != "hi" {
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
	if a.Messages[0].Content[0].Type != "text" || strings.TrimSpace(a.Messages[0].Content[0].Text) == "" {
		t.Fatalf("repaired message has no non-empty text: %s", b)
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

func TestP0_OpenAIToAnthropicEmptySystemOmitted(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "empty",
			body: `{"model":"gpt-4","messages":[
				{"role":"system","content":""},
				{"role":"user","content":"hi"}
			]}`,
		},
		{
			name: "whitespace",
			body: `{"model":"gpt-4","messages":[
				{"role":"system","content":"   \n\t"},
				{"role":"user","content":"hi"}
			]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := Convert([]byte(tc.body), opts())
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(b, []byte(`"system"`)) {
				t.Fatalf("empty system emitted: %s", b)
			}
			var a AnthropicRequest
			if err := json.Unmarshal(b, &a); err != nil {
				t.Fatal(err)
			}
			if len(a.System) != 0 {
				t.Fatalf("system = %+v, want empty", a.System)
			}
		})
	}
}

func TestP0_OpenAIToAnthropicSystemKeepsNonWhitespaceText(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"system","content":"   "},
		{"role":"system","content":"be brief"},
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
	if len(a.System) != 1 || a.System[0].Text != "be brief" {
		t.Fatalf("system = %+v, want one 'be brief' block: %s", a.System, b)
	}
}

func TestP0_ResponsesToAnthropicEmptySystemOmitted(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "empty instructions",
			body: `{"model":"gpt-4o","instructions":"","input":"hi"}`,
		},
		{
			name: "whitespace instructions",
			body: `{"model":"gpt-4o","instructions":"   ","input":"hi"}`,
		},
		{
			name: "billing header leaves whitespace",
			body: `{"model":"gpt-4o","instructions":"x-anthropic-billing-header: abc\n   ","input":"hi"}`,
		},
		{
			name: "whitespace system item",
			body: `{"model":"gpt-4o","input":[
				{"type":"message","role":"system","content":"  "},
				{"type":"message","role":"user","content":"hi"}
			]}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := ConvertResponsesToAnthropic([]byte(tc.body), &ConvertOptions{Model: "claude", MaxTokens: 8192})
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(b, []byte(`"system"`)) {
				t.Fatalf("empty system emitted: %s", b)
			}
			var a AnthropicRequest
			if err := json.Unmarshal(b, &a); err != nil {
				t.Fatal(err)
			}
			if len(a.System) != 0 {
				t.Fatalf("system = %+v, want empty", a.System)
			}
		})
	}
}

func TestP0_ResponsesToAnthropicSystemKeepsNonWhitespaceText(t *testing.T) {
	body := `{"model":"gpt-4o","instructions":"x-anthropic-billing-header: abc\n   ","input":[
		{"type":"message","role":"system","content":"be brief"},
		{"type":"message","role":"user","content":"hi"}
	]}`
	b, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 8192})
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if len(a.System) != 1 || a.System[0].Text != "be brief" {
		t.Fatalf("system = %+v, want one 'be brief' block: %s", a.System, b)
	}
}

func TestP0_ResponsesToAnthropicNonFunctionToolsNoToolChoice(t *testing.T) {
	body := `{"model":"gpt-4o","input":"hello","tools":[{"type":"web_search"}],"tool_choice":"auto"}`
	b, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude", MaxTokens: 8192})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(b, []byte(`"tool_choice"`)) {
		t.Fatalf("tool_choice emitted without tools after filtering: %s", b)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	if a.ToolChoice != nil {
		t.Fatalf("tool_choice = %+v, want nil", a.ToolChoice)
	}
	if len(a.Tools) != 0 {
		t.Fatalf("tools = %+v, want empty", a.Tools)
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
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	var sawToolUse bool
	for _, m := range a.Messages {
		for _, c := range m.Content {
			if c.Type == "text" && strings.TrimSpace(c.Text) == "" {
				t.Fatalf("empty text block: %+v\nbody: %s", m, b)
			}
			if c.Type == "tool_use" && c.ID == "call_1" {
				if m.Role != "assistant" {
					t.Fatalf("tool_use role = %q, want assistant", m.Role)
				}
				if c.Name != "get_weather" {
					t.Fatalf("tool_use name = %q, want get_weather", c.Name)
				}
				if _, ok := c.Input.(map[string]any); !ok {
					t.Fatalf("tool_use input type = %T, want object", c.Input)
				}
				sawToolUse = true
			}
		}
	}
	if !sawToolUse {
		t.Fatalf("assistant tool_use block missing from Responses output: %s", b)
	}
}

func TestP0_OpenAIToAnthropicEmptyArgumentsBecomeInputObject(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"f","arguments":""}}
		]},
		{"role":"tool","tool_call_id":"call_1","content":"ok"}
	]}`
	b, err := Convert([]byte(body), opts())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(`"input":{}`)) {
		t.Fatalf("tool_use.input missing: %s", b)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatal(err)
	}
	var input any
	for _, c := range a.Messages[1].Content {
		if c.Type == "tool_use" {
			input = c.Input
		}
	}
	if _, ok := input.(map[string]any); !ok {
		t.Fatalf("input type = %T, want object", input)
	}
}

func TestP0_AnthropicToOpenAINullInputBecomesArgumentsObject(t *testing.T) {
	body := `{"id":"msg_1","type":"message","role":"assistant","model":"claude",
		"content":[{"type":"tool_use","id":"toolu_1","name":"f","input":null}],
		"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}}`
	b, err := convertAnthropicResponseToOpenAI([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	got := o.Choices[0].Message.ToolCalls[0].Function.Arguments
	if got != "{}" {
		t.Fatalf("arguments = %q, want {}", got)
	}
}

func TestP0_StreamInterleavedToolCallsKeepArguments(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	_ = sc.HandleStreamStart()

	chunks := []string{
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"g","arguments":"{\"b\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":null}]}`,
	}
	var out []byte
	for _, chunk := range chunks {
		b, err := sc.HandleChunk([]byte(chunk))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, b...)
		// HandleChunk returns no trailing delimiter; frame each phase
		// payload so consecutive payloads cannot merge into one SSE line.
		out = append(out, '\n', '\n')
	}
	out = append(out, sc.HandleStreamEnd()...)

	args := map[string]string{} // tool ID -> concatenated partial_json
	var currentID string
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}
		var evt struct {
			Type         string `json:"type"`
			ContentBlock struct {
				ID string `json:"id"`
			} `json:"content_block"`
			Delta struct {
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if json.Unmarshal([]byte(line), &evt) != nil {
			continue
		}
		switch evt.Type {
		case "content_block_start":
			currentID = evt.ContentBlock.ID
		case "content_block_delta":
			args[currentID] += evt.Delta.PartialJSON
		}
	}
	if args["call_0"] != `{"a":1}` {
		t.Fatalf("call_0 arguments = %q", args["call_0"])
	}
	if args["call_1"] != `{"b":2}` {
		t.Fatalf("call_1 arguments = %q", args["call_1"])
	}
}

func TestP0_StreamHeldToolCallsReleaseInIndexOrder(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	_ = sc.HandleStreamStart()

	// Call 0 opens with incomplete arguments, so calls 2 and 1 (in that
	// arrival order) are held. Call 1 carries no arguments at all.
	chunks := []string{
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"id":"call_2","type":"function","function":{"name":"h","arguments":"{\"c\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"g","arguments":""}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"function":{"arguments":"3}"}}]},"finish_reason":null}]}`,
	}
	for _, chunk := range chunks {
		if _, err := sc.HandleChunk([]byte(chunk)); err != nil {
			t.Fatal(err)
		}
	}
	end := string(sc.HandleStreamEnd())

	events := parseP0SSEEvents(t, end)
	type releasedCall struct {
		id    string
		index int
	}
	var released []releasedCall
	deltas := map[int]string{}
	stops := map[int]int{}
	for _, evt := range events {
		switch evt.Type {
		case "content_block_start":
			if evt.ContentBlock.Type != "tool_use" {
				t.Fatalf("unexpected start block type %q:\n%s", evt.ContentBlock.Type, end)
			}
			if evt.ContentBlock.Input == nil || len(evt.ContentBlock.Input) != 0 {
				t.Fatalf("start %s input = %#v, want empty object:\n%s", evt.ContentBlock.ID, evt.ContentBlock.Input, end)
			}
			released = append(released, releasedCall{id: evt.ContentBlock.ID, index: evt.Index})
		case "content_block_delta":
			deltas[evt.Index] += evt.Delta.PartialJSON
		case "content_block_stop":
			stops[evt.Index]++
		}
	}

	if len(released) != 2 {
		t.Fatalf("got %d released calls, want 2:\n%s", len(released), end)
	}
	if released[0].id != "call_1" || released[1].id != "call_2" {
		t.Fatalf("released %q then %q, want call_1 then call_2:\n%s", released[0].id, released[1].id, end)
	}
	seen := map[int]bool{}
	for i, call := range released {
		if seen[call.index] {
			t.Fatalf("block index %d reused:\n%s", call.index, end)
		}
		seen[call.index] = true
		if i > 0 && released[i-1].index >= call.index {
			t.Fatalf("block indices not ascending: %d then %d:\n%s", released[i-1].index, call.index, end)
		}
		if stops[call.index] != 1 {
			t.Fatalf("call %s block %d has %d stops, want 1:\n%s", call.id, call.index, stops[call.index], end)
		}
	}
	if got := deltas[released[0].index]; got != "{}" {
		t.Fatalf("call_1 arguments = %q, want {}:\n%s", got, end)
	}
	if got := deltas[released[1].index]; got != `{"c":3}` {
		t.Fatalf("call_2 arguments = %q, want {\"c\":3}:\n%s", got, end)
	}
}

type p0SSEEvent struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock struct {
		Type  string         `json:"type"`
		ID    string         `json:"id"`
		Name  string         `json:"name"`
		Input map[string]any `json:"input"`
	} `json:"content_block"`
	Delta struct {
		Type        string `json:"type"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
}

// parseP0SSEEvents decodes the data lines of an Anthropic SSE payload.
func parseP0SSEEvents(t *testing.T, s string) []p0SSEEvent {
	t.Helper()
	var events []p0SSEEvent
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var evt p0SSEEvent
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &evt); err != nil {
			t.Fatalf("parse SSE event %q: %v", line, err)
		}
		events = append(events, evt)
	}
	return events
}
