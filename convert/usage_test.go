package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// usageOf converts body and returns the usage object of the result, whatever
// protocol the result is in.
func usageOf(t *testing.T, body string) map[string]any {
	t.Helper()
	return usageOfWithOpts(t, body, opts())
}

func usageOfWithOpts(t *testing.T, body string, o *ConvertOptions) map[string]any {
	t.Helper()
	out, err := Convert([]byte(body), o)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatalf("unmarshal result: %v\nbody: %s", err, out)
	}
	usage, _ := raw["usage"].(map[string]any)
	if usage == nil {
		usage, _ = raw["usageMetadata"].(map[string]any)
	}
	if usage == nil {
		t.Fatalf("no usage in result: %s", out)
	}
	return usage
}

func num(t *testing.T, m map[string]any, key string) int {
	t.Helper()
	v, ok := m[key]
	if !ok {
		return 0
	}
	f, ok := v.(float64)
	if !ok {
		t.Fatalf("%s: want number, got %T", key, v)
	}
	return int(f)
}

func TestConvert_OpenAIRespToAnthropicResp_CacheRead(t *testing.T) {
	body := `{
		"id":"chatcmpl-1","object":"chat.completion","model":"deepseek",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1050,
			"prompt_tokens_details":{"cached_tokens":800}}
	}`
	u := usageOf(t, body)
	if got := num(t, u, "input_tokens"); got != 200 {
		t.Fatalf("input_tokens: want 200 (prompt minus cached), got %d", got)
	}
	if got := num(t, u, "cache_read_input_tokens"); got != 800 {
		t.Fatalf("cache_read_input_tokens: want 800, got %d", got)
	}
	if got := num(t, u, "output_tokens"); got != 50 {
		t.Fatalf("output_tokens: want 50, got %d", got)
	}
}

func TestConvert_OpenAIRespToAnthropicResp_DeepSeekCacheHit(t *testing.T) {
	body := `{
		"id":"chatcmpl-1","object":"chat.completion","model":"deepseek",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1050,
			"prompt_cache_hit_tokens":700}
	}`
	u := usageOf(t, body)
	if got := num(t, u, "input_tokens"); got != 300 {
		t.Fatalf("input_tokens: want 300, got %d", got)
	}
	if got := num(t, u, "cache_read_input_tokens"); got != 700 {
		t.Fatalf("cache_read_input_tokens: want 700, got %d", got)
	}
}

func TestConvert_OpenAIRespToAnthropicResp_CacheGreaterThanPrompt(t *testing.T) {
	body := `{
		"id":"chatcmpl-1","object":"chat.completion","model":"deepseek",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":100,"completion_tokens":50,
			"prompt_tokens_details":{"cached_tokens":300}}
	}`
	u := usageOf(t, body)
	if got := num(t, u, "input_tokens"); got != 0 {
		t.Fatalf("input_tokens: want 0 (clamped), got %d", got)
	}
	if got := num(t, u, "cache_read_input_tokens"); got != 300 {
		t.Fatalf("cache_read_input_tokens: want 300, got %d", got)
	}
}

func TestConvert_AnthropicToOpenAI_CacheUsage(t *testing.T) {
	body := `{
		"id":"msg_1","type":"message","role":"assistant",
		"content":[{"type":"text","text":"hi"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"usage":{"input_tokens":200,"output_tokens":50,
			"cache_read_input_tokens":800,"cache_creation_input_tokens":100}
	}`
	u := usageOf(t, body)
	if got := num(t, u, "prompt_tokens"); got != 1100 {
		t.Fatalf("prompt_tokens: want 1100 (input+read+write), got %d", got)
	}
	if got := num(t, u, "completion_tokens"); got != 50 {
		t.Fatalf("completion_tokens: want 50, got %d", got)
	}
	if got := num(t, u, "total_tokens"); got != 1150 {
		t.Fatalf("total_tokens: want 1150, got %d", got)
	}
	details, _ := u["prompt_tokens_details"].(map[string]any)
	if details == nil {
		t.Fatalf("prompt_tokens_details missing: %v", u)
	}
	if got := num(t, details, "cached_tokens"); got != 800 {
		t.Fatalf("cached_tokens: want 800, got %d", got)
	}
	if got := num(t, details, "cache_write_tokens"); got != 100 {
		t.Fatalf("cache_write_tokens: want 100, got %d", got)
	}
}

func TestStream_UsageCacheRead(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil, nil)
	sc.HandleStreamStart()

	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":800}}}`))

	s := string(sc.HandleStreamEnd())
	var delta struct {
		Usage map[string]any `json:"usage"`
	}
	idx := indexOfMessageDelta(s)
	if idx < 0 {
		t.Fatalf("no message_delta in: %s", s)
	}
	if err := json.Unmarshal([]byte(dataPayload(s[idx:])), &delta); err != nil {
		t.Fatalf("unmarshal message_delta: %v\n%s", err, s)
	}
	if got := num(t, delta.Usage, "input_tokens"); got != 200 {
		t.Fatalf("input_tokens: want 200, got %d (usage: %v)", got, delta.Usage)
	}
	if got := num(t, delta.Usage, "cache_read_input_tokens"); got != 800 {
		t.Fatalf("cache_read_input_tokens: want 800, got %d (usage: %v)", got, delta.Usage)
	}
	if got := num(t, delta.Usage, "output_tokens"); got != 50 {
		t.Fatalf("output_tokens: want 50, got %d", got)
	}
}

func TestConvert_GeminiRespToAnthropic_CacheRead(t *testing.T) {
	body := `{
		"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":50,
			"totalTokenCount":1050,"cachedContentTokenCount":800}
	}`
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolAnthropic})
	u := usageOfWithOpts(t, body, &ConvertOptions{
		Model:        "claude-sonnet-4-20250514",
		MaxTokens:    8192,
		SessionStore: store,
		SID:          "sid",
		Direction:    "response",
	})
	if got := num(t, u, "input_tokens"); got != 200 {
		t.Fatalf("input_tokens: want 200, got %d", got)
	}
	if got := num(t, u, "cache_read_input_tokens"); got != 800 {
		t.Fatalf("cache_read_input_tokens: want 800, got %d", got)
	}
}

func TestConvert_GeminiRespToOpenAI_CacheRead(t *testing.T) {
	body := `{
		"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":50,
			"totalTokenCount":1050,"cachedContentTokenCount":800}
	}`
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolOpenAIChat})
	u := usageOfWithOpts(t, body, &ConvertOptions{
		Model:        "gpt-4",
		MaxTokens:    8192,
		SessionStore: store,
		SID:          "sid",
		Direction:    "response",
	})
	if got := num(t, u, "prompt_tokens"); got != 1000 {
		t.Fatalf("prompt_tokens: want 1000, got %d", got)
	}
	details, _ := u["prompt_tokens_details"].(map[string]any)
	if details == nil {
		t.Fatalf("prompt_tokens_details missing: %v", u)
	}
	if got := num(t, details, "cached_tokens"); got != 800 {
		t.Fatalf("cached_tokens: want 800, got %d", got)
	}
}

func TestGeminiStreamFinish_CacheRead(t *testing.T) {
	chunk := &GeminiChatResponse{
		Candidates: []GeminiCandidate{{Index: 0, FinishReason: "STOP"}},
		UsageMetadata: &GeminiUsage{
			PromptTokenCount:        1000,
			CandidatesTokenCount:    50,
			TotalTokenCount:         1050,
			CachedContentTokenCount: 800,
		},
	}

	out, err := geminiChunkToOpenAI(nil, chunk, chunk.Candidates[0], "gpt-4", &ConvertOptions{})
	if err != nil {
		t.Fatalf("openai chunk: %v", err)
	}
	var oai struct {
		Usage map[string]any `json:"usage"`
	}
	if err := json.Unmarshal([]byte(dataPayload(string(out))), &oai); err != nil {
		t.Fatalf("unmarshal openai chunk: %v\n%s", err, out)
	}
	details, _ := oai.Usage["prompt_tokens_details"].(map[string]any)
	if details == nil {
		t.Fatalf("prompt_tokens_details missing: %v", oai.Usage)
	}
	if got := num(t, details, "cached_tokens"); got != 800 {
		t.Fatalf("cached_tokens: want 800, got %d", got)
	}

	out, err = geminiChunkToAnthropic(nil, chunk, chunk.Candidates[0], "claude", &ConvertOptions{})
	if err != nil {
		t.Fatalf("anthropic chunk: %v", err)
	}
	if !strings.Contains(string(out), `"input_tokens":200`) || !strings.Contains(string(out), `"cache_read_input_tokens":800`) {
		t.Fatalf("anthropic finish chunk lacks cache usage: %s", out)
	}
}

// indexOfMessageDelta returns the offset of the message_delta event in s, or -1.
func indexOfMessageDelta(s string) int {
	const marker = "event: message_delta"
	for i := 0; i+len(marker) <= len(s); i++ {
		if s[i:i+len(marker)] == marker {
			return i
		}
	}
	return -1
}

// dataPayload returns the data line of the first SSE event in s.
func dataPayload(s string) string {
	for i := 0; i < len(s); i++ {
		if len(s)-i >= 6 && s[i:i+6] == "data: " {
			j := i + 6
			for j < len(s) && s[j] != '\n' {
				j++
			}
			return s[i+6 : j]
		}
	}
	return ""
}
