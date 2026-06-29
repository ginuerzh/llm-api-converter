package convert

import (
	"encoding/json"
	"testing"
)

// ---------------------------------------------------------------------------
// Protocol override tests
// ---------------------------------------------------------------------------

func TestModelMapLookupTarget(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "minimax-m3", Protocol: "anthropic"},
		{SourcePrefix: "gpt-4", TargetModel: "deepseek-chat", Protocol: "openai"},
	}

	if p := mm.LookupTarget("minimax-m3"); p != "anthropic" {
		t.Fatalf("LookupTarget(minimax-m3): want anthropic, got %q", p)
	}
	if p := mm.LookupTarget("DEEPSEEK-chat"); p != "openai" {
		t.Fatalf("LookupTarget(DEEPSEEK-chat): want openai (case insensitive), got %q", p)
	}
	if p := mm.LookupTarget("unknown"); p != "" {
		t.Fatalf("LookupTarget(unknown): want empty, got %q", p)
	}
}

func TestModelMapLookupTargetNoMatch(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "minimax-m3", Protocol: "anthropic"},
	}
	if p := mm.LookupTarget("other-model"); p != "" {
		t.Fatalf("LookupTarget(other-model): want empty, got %q", p)
	}
}

func TestModelMapApply_WithProtocol(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
		{SourcePrefix: "claude-sonnet", TargetModel: "deepseek-v4-flash"},
		{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"},
	}

	target, proto, ok := mm.Apply("claude-opus-4")
	if !ok || target != "deepseek-v4-pro" || proto != "openai" {
		t.Fatalf("claude-opus: want (deepseek-v4-pro, openai, true), got (%s, %s, %v)", target, proto, ok)
	}

	target, proto, ok = mm.Apply("claude-sonnet-4")
	if !ok || target != "deepseek-v4-flash" || proto != "" {
		t.Fatalf("claude-sonnet: want (deepseek-v4-flash, '', true), got (%s, %s, %v)", target, proto, ok)
	}

	target, proto, ok = mm.Apply("gpt-4")
	if !ok || target != "deepseek-chat" || proto != "openai" {
		t.Fatalf("gpt-4 via catch-all: want (deepseek-chat, openai, true), got (%s, %s, %v)", target, proto, ok)
	}

	target, proto, ok = mm.Apply("unknown-model")
	if !ok || target != "deepseek-chat" || proto != "openai" {
		t.Fatalf("unknown-model via catch-all: want (deepseek-chat, openai, true), got (%s, %s, %v)", target, proto, ok)
	}
}

func TestConvert_ProtocolOpenAI_OpenAIReqPassthrough(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	// Model should be rewritten but format preserved (no protocol conversion).
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v", err)
	}
	if result["model"] != "deepseek-chat" {
		t.Fatalf("model should be rewritten to deepseek-chat, got %v", result["model"])
	}
	if msgs, ok := result["messages"].([]any); !ok || len(msgs) == 0 {
		t.Fatal("messages should be preserved")
	}
}

func TestConvert_ProtocolOpenAI_AnthropicReqConverted(t *testing.T) {
	body := `{"model":"claude-sonnet-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("should produce valid OpenAI request, err=%v\nbody=%s", err, b)
	}
	if o.Model != "deepseek-chat" {
		t.Fatalf("model should be deepseek-chat, got %s", o.Model)
	}
	if len(o.Messages) == 0 {
		t.Fatal("expected messages in converted request")
	}
}

func TestConvert_ProtocolAnthropic_AnthropicReqPassthrough(t *testing.T) {
	body := `{"model":"claude-sonnet-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "claude-sonnet-4", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "claude-sonnet-4", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != body {
		t.Fatalf("expected passthrough:\n  got:  %s\n  want: %s", b, body)
	}
}

func TestConvert_ProtocolAnthropic_AnthropicReqPassthroughWithRename(t *testing.T) {
	body := `{"model":"claude-opus-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude-opus", TargetModel: "minimax-m3", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v\nbody=%s", err, b)
	}
	if result["model"] != "minimax-m3" {
		t.Fatalf("model should be rewritten to minimax-m3, got %v", result["model"])
	}
	// Should remain Anthropic format (preserve max_tokens, messages with content blocks)
	if _, ok := result["max_tokens"]; !ok {
		t.Fatal("should preserve Anthropic max_tokens field")
	}
	if _, ok := result["messages"]; !ok {
		t.Fatal("should preserve Anthropic messages field")
	}
}

func TestConvert_ProtocolAnthropic_OpenAIReqConverted(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	opts := &ConvertOptions{Model: "claude-sonnet-4", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "claude-sonnet-4", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("should produce valid Anthropic request, err=%v\nbody=%s", err, b)
	}
	if len(a.Messages) == 0 {
		t.Fatal("expected messages in converted request")
	}
}

func TestConvert_ProtocolOpenAI_OpenAIRespPassthrough(t *testing.T) {
	body := `{"id":"chatcmpl-abc","object":"chat.completion","created":1234567890,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v", err)
	}
	if result["model"] != "deepseek-chat" {
		t.Fatalf("model should be rewritten to deepseek-chat, got %v", result["model"])
	}
}

func TestConvert_ProtocolAnthropic_AnthropicRespPassthrough(t *testing.T) {
	body := `{"id":"msg_abc","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":5}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "claude-sonnet-4", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != body {
		t.Fatalf("expected passthrough:\n  got:  %s\n  want: %s", b, body)
	}
}

func TestConvert_ProtocolUnset_CurrentBehavior(t *testing.T) {
	body := `{"model":"claude-opus-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("should produce OpenAI request, err=%v\nbody=%s", err, b)
	}
	if o.Model != "deepseek-v4-pro" {
		t.Fatalf("model should be deepseek-v4-pro, got %s", o.Model)
	}
}

func TestConvert_ProtocolCatchAll(t *testing.T) {
	// Catch-all "*" with protocol=openai should rewrite model for unmatched models.
	mm := ModelMap{
		{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"},
	}
	body := `{"model":"unknown-model","messages":[{"role":"user","content":"hello"}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v", err)
	}
	if result["model"] != "deepseek-chat" {
		t.Fatalf("model should be rewritten to deepseek-chat, got %v", result["model"])
	}
}

// parseModelMap tests are in rewriter/server_test.go
