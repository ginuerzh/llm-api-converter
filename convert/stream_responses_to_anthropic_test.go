package convert

import (
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
