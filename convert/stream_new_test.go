package convert

import (
	"strings"
	"testing"
)

func TestStream_SignatureDelta(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil)
	sc.HandleStreamStart()

	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":"think..."},"finish_reason":null}]}`))

	out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`))
	s := string(out)
	if !strings.Contains(s, "signature_delta") {
		t.Fatalf("expected signature_delta when closing thinking block, got: %s", s)
	}
	if !strings.Contains(s, "content_block_stop") {
		t.Fatalf("expected content_block_stop after signature_delta, got: %s", s)
	}
}

func TestStream_UsagePassthrough(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil)
	sc.HandleStreamStart()

	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`))

	end := sc.HandleStreamEnd()
	s := string(end)
	if !strings.Contains(s, `"input_tokens":5`) {
		t.Fatalf("expected input_tokens:5 in message_delta, got: %s", s)
	}
	if !strings.Contains(s, `"output_tokens":3`) {
		t.Fatalf("expected output_tokens:3 in message_delta, got: %s", s)
	}
}

func TestStream_Interrupted(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil)
	sc.HandleStreamStart()

	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}`))

	sc.SetInterrupted()
	end := sc.HandleStreamEnd()
	s := string(end)
	if !strings.Contains(s, "[stream interrupted]") {
		t.Fatalf("expected [stream interrupted] marker, got: %s", s)
	}
}
