package convert

import (
	"encoding/json"
	"testing"
)

// Claude Code sends thinking:{"type":"adaptive"} (adaptive reasoning mode).
// There is no OpenAI-format equivalent, and reasoning-only models like
// ox-alpha-free reject any non-{enabled,disabled} thinking directive. The
// thinking field must not leak verbatim into the OpenAI request.
func TestConvert_DeepSeekAdaptiveThinkingDropped(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Thinking != nil {
		t.Fatalf("thinking should be dropped for non-OpenAI thinking.type, got %v", o.Thinking)
	}
	if o.ReasoningEffort == nil {
		t.Fatal("reasoning_effort should still be set from output_config.effort")
	}
	if e, _ := o.ReasoningEffort.(string); e != "high" {
		t.Fatalf("reasoning_effort: want 'high', got %v", o.ReasoningEffort)
	}
}

// enabled/disabled remain valid OpenAI-format values and must still pass through.
func TestConvert_DeepSeekThinkingEnabledDisabledPassthrough(t *testing.T) {
	for _, tt := range []struct {
		thinking string
		wantType string
	}{
		{`{"type":"enabled","budget_tokens":4096}`, "enabled"},
		{`{"type":"disabled"}`, "disabled"},
	} {
		body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":` + tt.thinking + `}`
		opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
		b, err := Convert([]byte(body), opts)
		if err != nil {
			t.Fatal(err)
		}
		var o OpenAIChatRequest
		if err := json.Unmarshal(b, &o); err != nil {
			t.Fatal(err)
		}
		if o.Thinking == nil {
			t.Fatalf("thinking should be forwarded for type %s", tt.wantType)
		}
		if m, ok := o.Thinking.(map[string]any); !ok || m["type"] != tt.wantType {
			t.Fatalf("thinking: want type %q, got %v", tt.wantType, o.Thinking)
		}
	}
}

// reasoning_effort values are preserved verbatim (OpenAI-compatible models such
// as DeepSeek/GLM/ox-alpha accept "max"); only xhigh is capped to "max", and
// low/medium must not be collapsed to high.
func TestReasoningEffortToOpenAi(t *testing.T) {
	for _, tt := range []struct {
		effort string
		want   any
	}{
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
		{"max", "max"},
		{"xhigh", "max"},
		{"", nil},
	} {
		got := reasoningEffortToOpenAi(&AnthropicOutputConfig{Effort: tt.effort})
		if got != tt.want {
			t.Fatalf("effort %q: want %v, got %v", tt.effort, tt.want, got)
		}
	}
	if reasoningEffortToOpenAi(nil) != nil {
		t.Fatal("nil config should return nil")
	}
}

// GLM path has the same problem: adaptive must not leak.
func TestConvert_GLMAdaptiveThinkingDropped(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "glm-4.6", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Thinking != nil {
		t.Fatalf("GLM thinking should be dropped for non-OpenAI thinking.type, got %v", o.Thinking)
	}
}

// Anthropic metadata (user_id etc.) must not leak into the OpenAI request —
// strict endpoints reject unknown params ("Invalid API parameter").
func TestConvert_AnthropicMetadataNotForwarded(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"metadata":{"user_id":"u_abc123"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"output_config":{"effort":"high"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-5", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Metadata != nil {
		t.Fatalf("Anthropic metadata should not be forwarded, got %v", o.Metadata)
	}
}