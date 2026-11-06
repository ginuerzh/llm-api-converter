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

	call1 := strings.Index(end, `"id":"call_1"`)
	call2 := strings.Index(end, `"id":"call_2"`)
	if call1 < 0 || call2 < 0 {
		t.Fatalf("held calls not released:\n%s", end)
	}
	if call1 > call2 {
		t.Fatalf("held calls released out of index order:\n%s", end)
	}
	if !strings.Contains(end, `"type":"tool_use","id":"call_1","name":"g","input":{}`) {
		t.Fatalf("call_1 start missing empty input object:\n%s", end)
	}
	if !strings.Contains(end, `"partial_json":"{}"`) {
		t.Fatalf("call_1 empty arguments not released as an object:\n%s", end)
	}
	if !strings.Contains(end, `"partial_json":"{\"c\":3}"`) {
		t.Fatalf("call_2 arguments wrong:\n%s", end)
	}
}
