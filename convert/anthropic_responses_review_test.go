package convert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConvertAnthropicRequestToResponsesParallelToolPairing(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"system":[{"type":"text","text":"sys"}],"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_a","name":"a","input":{"x":1}},{"type":"tool_use","id":"toolu_b","name":"b","input":{"y":2}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_a","content":"ra"},{"type":"tool_result","tool_use_id":"toolu_b","content":"rb"}]}]}`
	out, err := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	input := req["input"].([]any)
	want := []struct{ typ, id string }{
		{"function_call", "toolu_a"},
		{"function_call", "toolu_b"},
		{"function_call_output", "toolu_a"},
		{"function_call_output", "toolu_b"},
	}
	if len(input) != len(want) {
		t.Fatalf("input = %+v", input)
	}
	for i, w := range want {
		item := input[i].(map[string]any)
		if item["type"] != w.typ || item["call_id"] != w.id {
			t.Fatalf("input[%d] = %+v, want %+v", i, item, w)
		}
	}
}

func TestConvertAnthropicRequestToResponsesThinkingWithTools(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"system":[{"type":"text","text":"sys"}],"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"think","signature":"sig"},{"type":"text","text":"calling"},{"type":"tool_use","id":"toolu_a","name":"a","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_a","content":"done"}]}]}`
	out, err := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	input := req["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input = %+v", input)
	}
	if input[0].(map[string]any)["type"] != "message" || input[1].(map[string]any)["type"] != "function_call" || input[2].(map[string]any)["type"] != "function_call_output" {
		t.Fatalf("input = %+v", input)
	}
	if input[2].(map[string]any)["call_id"] != "toolu_a" {
		t.Fatalf("tool pairing lost: %+v", input)
	}
}

func TestConvertResponsesResponseToAnthropicZeroArgTool(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"function_call","call_id":"toolu_1","name":"ping","arguments":"{}"}]}`
	out, err := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"input":{}`) {
		t.Fatalf("missing input object: %s", out)
	}
	if errs := validateAnthropicResponseOpts(out, true); len(errs) != 0 {
		t.Fatalf("contract errors: %v", errs)
	}
}

func TestResponsesToAnthropicStreamDoneOnlyArguments(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"t","type":"function_call","call_id":"toolu_1","name":"f"}}`,
		`{"type":"response.function_call_arguments.done","item_id":"t","arguments":"{\"x\":1}"}`,
		`{"type":"response.output_item.done","item":{"id":"t","type":"function_call","call_id":"toolu_1","name":"f","arguments":"{\"x\":1}"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call"}]}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil {
			t.Fatal(err)
		}
		out.Write(b)
	}
	s := out.String()
	if !strings.Contains(s, `"partial_json":"{\"x\":1}"`) || !strings.Contains(s, `"type":"tool_use"`) {
		t.Fatalf("done-only args not emitted: %s", s)
	}
}

func TestResponsesToAnthropicStreamInterleavedDoneBeforeOpen(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a"}}`,
		`{"type":"response.output_item.added","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"a","delta":"{\"x\":1}"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"b","delta":"{\"y\":2}"}`,
		`{"type":"response.output_item.done","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b","arguments":"{\"y\":2}"}}`,
		`{"type":"response.output_item.done","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a","arguments":"{\"x\":1}"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call"}]}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil {
			t.Fatal(err)
		}
		out.Write(b)
	}
	s := out.String()
	if strings.Count(s, `"type":"tool_use"`) != 2 {
		t.Fatalf("want two tool blocks: %s", s)
	}
	if strings.Index(s, `"id":"call_a"`) > strings.Index(s, `"id":"call_b"`) {
		t.Fatalf("calls out of order: %s", s)
	}
}

func TestResponsesToAnthropicStreamTruncatedToolOnCompleted(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"t","type":"function_call","call_id":"toolu_1","name":"f"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"t","delta":"{\"a\":"}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call"}]}}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	s := out.String()
	if !strings.Contains(s, "event: error") || strings.Contains(s, "event: message_stop") {
		t.Fatalf("truncated tool was not terminal: %s", s)
	}
	if errs := validateAnthropicSSE([]byte(s)); len(errs) != 0 {
		t.Fatalf("contract errors: %v\n%s", errs, s)
	}
}

func TestResponsesToAnthropicStreamFirstEventFailed(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	b, err := c.HandleChunk([]byte(`{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"message":"boom"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, "event: message_start") || !strings.Contains(s, "event: error") || !strings.Contains(s, "boom") {
		t.Fatalf("failed event did not produce a contract stream: %s", s)
	}
	if errs := validateAnthropicSSE([]byte(s)); len(errs) != 0 {
		t.Fatalf("contract errors: %v\n%s", errs, s)
	}
}

func TestResponsesToAnthropicStreamAfterTerminal(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_text.delta","item_id":"m","delta":"hi"}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message"}]}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message"}]}}`,
		`{"type":"response.content_part.added","item_id":"m2","part":{"type":"output_text","text":""}}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	s := out.String()
	if strings.Count(s, "event: message_stop") != 1 {
		t.Fatalf("duplicate/terminal output: %s", s)
	}
	if strings.Count(s, `"type":"content_block_start"`) != 1 {
		t.Fatalf("block opened after terminal: %s", s)
	}
}

func TestHandleSSEEventMissingSessionBareError(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SID: "missing", SessionStore: store, URI: "/v1/messages", Direction: "response"}
	out, err := HandleSSEEvent("missing", "start", 0, []byte(`data: {"type":"error","error":{"type":"api_error","message":"boom"}}`), opts)
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	if !strings.Contains(s, "event: error") || !strings.Contains(s, "boom") {
		t.Fatalf("bare error was not forwarded as an Anthropic error: %s", s)
	}
}

func TestHandleSSEEventAnthropicPassthroughError(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolAnthropic, To: ProtocolAnthropic})
	opts := &ConvertOptions{SID: "sid", SessionStore: store, URI: "/v1/messages", Direction: "response"}
	out, err := HandleSSEEvent("sid", "start", 0, []byte(`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`), opts)
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	if !strings.Contains(s, "overloaded_error") || strings.Contains(s, "api_error") {
		t.Fatalf("Anthropic passthrough error was rewritten: %s", s)
	}
}

func TestResponsesFixturesContract(t *testing.T) {
	fixtures := map[string][]string{
		"text.sse":               {`"type":"text"`, `"text_delta"`},
		"tool.sse":               {`"type":"tool_use"`, `"input_json_delta"`},
		"reasoning.sse":          {`"type":"thinking"`, `"thinking_delta"`},
		"parallel.sse":           {`"type":"tool_use"`, `"type":"tool_use"`},
		"deepseek-text.sse":      {`"type":"thinking"`, `"stop_reason":"max_tokens"`},
		"deepseek-reasoning.sse": {`"type":"thinking"`, `"type":"text"`},
		"deepseek-tool.sse":      {`"type":"tool_use"`},
	}
	for name, wants := range fixtures {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join("testdata", "responses", name))
			if err != nil {
				t.Fatal(err)
			}
			c := NewResponsesToAnthropicStreamConverter("claude-opus-5-5", &ConvertOptions{})
			var out strings.Builder
			for _, line := range strings.Split(string(data), "\n") {
				line = strings.TrimSpace(line)
				if !strings.HasPrefix(line, "data:") {
					continue
				}
				payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				if payload == "" || payload == "[DONE]" {
					continue
				}
				b, err := c.HandleChunk([]byte(payload))
				if err != nil {
					t.Fatalf("chunk: %v", err)
				}
				out.Write(b)
			}
			out.Write(c.HandleStreamEnd())
			s := out.String()
			for _, want := range wants {
				if !strings.Contains(s, want) {
					t.Fatalf("missing %s in %s", want, s)
				}
			}
			if errs := validateAnthropicSSE([]byte(s)); len(errs) != 0 {
				t.Fatalf("contract errors: %v\n%s", errs, s)
			}
		})
	}
}
