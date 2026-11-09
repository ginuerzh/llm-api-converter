package convert

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResponsesToAnthropicStreamText(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("claude-opus-5-5", &ConvertOptions{})
	var out strings.Builder
	out.Write(c.HandleStreamStart())
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"deepseek-v4.1-flash-1017"}}`,
		`{"type":"response.output_item.added","item":{"id":"msg_1","type":"message"},"output_index":0}`,
		`{"type":"response.content_part.added","item_id":"msg_1","part":{"type":"output_text","text":""}}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","delta":"Hello"}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","delta":" world"}`,
		`{"type":"response.output_text.done","item_id":"msg_1","text":"Hello world"}`,
		`{"type":"response.content_part.done","item_id":"msg_1"}`,
		`{"type":"response.output_item.done","item":{"id":"msg_1","type":"message"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message"}],"usage":{"input_tokens":10,"output_tokens":2}}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil {
			t.Fatal(err)
		}
		out.Write(b)
	}
	out.Write(c.HandleStreamEnd())
	s := out.String()
	for _, want := range []string{"event: message_start", "event: content_block_start", `"type":"text"`, `"text_delta"`, `"text":"Hello"`, `"text":" world"`, "event: content_block_stop", `"stop_reason":"end_turn"`, `"output_tokens":2`, "event: message_stop"} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}

func TestResponsesToAnthropicStreamToolCall(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("claude-opus-5-5", &ConvertOptions{})
	var out strings.Builder
	out.Write(c.HandleStreamStart())
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"msg_1","type":"function_call","call_id":"toolu_1","name":"get_weather","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"msg_1","delta":"{\"city\":\"Beijing\"}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"msg_1","arguments":"{\"city\":\"Beijing\"}"}`,
		`{"type":"response.output_item.done","item":{"id":"msg_1","type":"function_call","call_id":"toolu_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call"}]}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil {
			t.Fatal(err)
		}
		out.Write(b)
	}
	s := out.String()
	for _, want := range []string{`"type":"tool_use"`, `"id":"toolu_1"`, `"input_json_delta"`, `"partial_json":"{\"city\":\"Beijing\"}"`, `"stop_reason":"tool_use"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}

func TestResponsesToAnthropicStreamReasoningText(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("claude-opus-5-5", &ConvertOptions{})
	var out strings.Builder
	out.Write(c.HandleStreamStart())
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"msg_r","type":"reasoning"}}`,
		`{"type":"response.reasoning_text.delta","item_id":"msg_r","delta":"think"}`,
		`{"type":"response.reasoning_text.done","item_id":"msg_r","text":"think"}`,
		`{"type":"response.output_item.done","item":{"id":"msg_r","type":"reasoning"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil {
			t.Fatal(err)
		}
		out.Write(b)
	}
	s := out.String()
	if !strings.Contains(s, `"type":"thinking"`) || !strings.Contains(s, `"signature":""`) || !strings.Contains(s, `"thinking_delta"`) || !strings.Contains(s, `"thinking":"think"`) {
		t.Fatalf("unexpected reasoning stream: %s", s)
	}
	if strings.Contains(s, "signature_delta") {
		t.Fatalf("unexpected signature_delta: %s", s)
	}
}

func TestResponsesToAnthropicStreamReasoningSummaryVariant(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.output_item.added","item":{"id":"r","type":"reasoning"}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"r","delta":"summary"}`,
		`{"type":"response.reasoning_summary_text.done","item_id":"r","text":"summary"}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	if !strings.Contains(out.String(), `"thinking":"summary"`) {
		t.Fatalf("out = %s", out.String())
	}
}

func TestResponsesToAnthropicStreamReasoningPartAdded(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.output_item.added","item":{"id":"r","type":"reasoning"}}`,
		`{"type":"response.content_part.added","item_id":"r","part":{"type":"reasoning_text","text":""}}`,
		`{"type":"response.reasoning_text.delta","item_id":"r","delta":"think"}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	s := out.String()
	if strings.Contains(s, `"type":"text"`) {
		t.Fatalf("reasoning content part opened a text block: %s", s)
	}
	if !strings.Contains(s, `"type":"thinking"`) || !strings.Contains(s, `"thinking":"think"`) {
		t.Fatalf("unexpected stream: %s", s)
	}
}

func TestResponsesToAnthropicStreamParallelCalls(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"a","delta":"{\"x\":1}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"a","arguments":"{\"x\":1}"}`,
		`{"type":"response.output_item.done","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a","arguments":"{\"x\":1}"}}`,
		`{"type":"response.output_item.added","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"b","delta":"{\"y\":2}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"b","arguments":"{\"y\":2}"}`,
		`{"type":"response.output_item.done","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b","arguments":"{\"y\":2}"}}`,
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
	for _, want := range []string{`"partial_json":"{\"x\":1}"`, `"partial_json":"{\"y\":2}"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}

func TestResponsesToAnthropicStreamInterleavedCalls(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a"}}`,
		`{"type":"response.output_item.added","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"a","delta":"{\"x\":1}"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"b","delta":"{\"y\":2}"}`,
		`{"type":"response.output_item.done","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a","arguments":"{\"x\":1}"}}`,
		`{"type":"response.output_item.done","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b","arguments":"{\"y\":2}"}}`,
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
	for _, want := range []string{`"partial_json":"{\"x\":1}"`, `"partial_json":"{\"y\":2}"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}

func TestResponsesToAnthropicStreamTruncatedToolIsError(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{ErrorMsg: "unexpected EOF"})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.output_item.added","item":{"id":"t","type":"function_call","call_id":"toolu_1","name":"f"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"t","delta":"{\"city\":\"Bei"}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	out.Write(c.HandleStreamEnd())
	s := out.String()
	if !strings.Contains(s, "event: error") || strings.Contains(s, "event: message_stop") {
		t.Fatalf("out = %s", s)
	}
}

func TestResponsesToAnthropicStreamFailedEvent(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	_ = c.HandleStreamStart()
	out, _ := c.HandleChunk([]byte(`{"type":"response.failed","response":{"status":"failed","error":{"message":"upstream failed"}}}`))
	if !strings.Contains(string(out), "event: error") || !strings.Contains(string(out), "upstream failed") {
		t.Fatalf("out = %s", out)
	}
}

func TestResponsesToAnthropicStreamFixtures(t *testing.T) {
	fixtures := []string{
		"text.sse", "tool.sse", "reasoning.sse", "parallel.sse",
		"deepseek-text.sse", "deepseek-reasoning.sse", "deepseek-tool.sse",
	}
	for _, name := range fixtures {
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
			if !strings.Contains(s, "event: message_stop") && !strings.Contains(s, "event: error") {
				t.Fatalf("fixture did not finish: %s", s)
			}
		})
	}
}
