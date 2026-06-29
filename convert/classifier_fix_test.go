package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestConvert_DeepSeekReasoningOnlyEmptyContent covers the case where
// DeepSeek-V4-Pro spends all max_tokens on reasoning_content and returns
// empty content + finish_reason:length. The converter must produce a
// schema-valid Anthropic response with at least one non-empty text block.
func TestConvert_DeepSeekReasoningOnlyEmptyContent(t *testing.T) {
	deepseekResp := `{"id":"chatcmpl-test","object":"chat.completion","model":"deepseek-v4-pro","choices":[{"index":0,"message":{"role":"assistant","content":"","reasoning_content":"The user asks: say hi"},"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":85,"completion_tokens":10,"total_tokens":95,"completion_tokens_details":{"reasoning_tokens":10}}}`

	out, err := Convert([]byte(deepseekResp), &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192})
	if err != nil {
		t.Fatal(err)
	}

	var anth map[string]any
	if err := json.Unmarshal(out, &anth); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}

	content, ok := anth["content"].([]any)
	if !ok || len(content) == 0 {
		t.Fatalf("expected non-empty content array, got %v", anth["content"])
	}

	for i, block := range content {
		b := block.(map[string]any)
		if b["type"] == "text" {
			text, ok := b["text"].(string)
			if !ok || text == "" {
				t.Errorf("content[%d]: text block missing non-empty 'text' field: %v", i, b)
			}
		}
	}

	if sr, ok := anth["stop_reason"].(string); !ok || sr != "max_tokens" {
		t.Errorf("stop_reason: want max_tokens, got %v", anth["stop_reason"])
	}
}

// TestStreamConverter_DeepSeekReasoningOnly covers the case where a
// DeepSeek-V4-Pro streaming response produces only reasoning_content deltas
// and no text content. HandleStreamEnd must emit at least one valid text block.
func TestStreamConverter_DeepSeekReasoningOnly(t *testing.T) {
	sc := NewStreamConverter("deepseek-v4-pro", nil, nil)
	_ = sc.HandleStreamStart()

	// Reasoning delta only — no content text.
	if _, err := sc.HandleChunk([]byte(
		`{"choices":[{"index":0,"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}`,
	)); err != nil {
		t.Fatal(err)
	}

	// Final chunk: empty delta, finish=length.
	if _, err := sc.HandleChunk([]byte(
		`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":10}}`,
	)); err != nil {
		t.Fatal(err)
	}

	end := sc.HandleStreamEnd()
	if end == nil {
		t.Fatal("HandleStreamEnd returned nil")
	}

	s := string(end)
	if !strings.Contains(s, `text_delta`) {
		t.Fatalf("expected text_delta in stream output, got:\n%s", s)
	}
	if !strings.Contains(s, `"stop_reason":"max_tokens"`) {
		t.Errorf("expected stop_reason max_tokens in:\n%s", s)
	}
}
