package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// thoughtSignatureSentinel is now defined in anthropic_to_openai.go (production).
func toolCallSignatures(t *testing.T, body []byte) []map[string]any {
	t.Helper()
	var oai OpenAIChatRequest
	if err := json.Unmarshal(body, &oai); err != nil {
		t.Fatalf("unmarshal outgoing body: %v", err)
	}
	var sigs []map[string]any
	for _, m := range oai.Messages {
		if m.Role != "assistant" {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.ExtraContent != nil {
				sigs = append(sigs, tc.ExtraContent)
			} else {
				sigs = append(sigs, nil)
			}
		}
	}
	return sigs
}

func TestGeminiToolReplayAddsExtraContentWithSentinel(t *testing.T) {
	body := []byte(`{"model":"gemini-4-hydrogen","max_tokens":10,"messages":[
		{"role":"user","content":[{"type":"text","text":"do two things"}]},
		{"role":"assistant","content":[
			{"type":"tool_use","id":"call_a","name":"lookup","input":{"q":"x"}},
			{"type":"tool_use","id":"call_b","name":"list","input":{}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_a","content":"ok"},
			{"type":"tool_result","tool_use_id":"call_b","content":"ok"}]}]}`)

	out, err := convertAnthropicRequestToOpenAI(body, &ConvertOptions{ResolvedModel: "gemini-4-hydrogen"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}

	// Wire format is pinned: snake_case extra_content, nested google object,
	// nested thought_signature. Assert on the raw bytes so a struct-tag slip
	// cannot silently pass.
	raw := string(out)
	want := `"extra_content":{"google":{"thought_signature":"` + thoughtSignatureSentinel + `"}}`
	if !strings.Contains(raw, want) {
		t.Fatalf("outgoing body missing pinned extra_content shape %s\n%s", want, raw)
	}

	sigs := toolCallSignatures(t, out)
	if len(sigs) != 2 {
		t.Fatalf("assistant replay should carry 2 tool calls, got %d\n%s", len(sigs), raw)
	}
	for i, sig := range sigs {
		google, ok := sig["google"].(map[string]any)
		if !ok {
			t.Fatalf("tool call %d extra_content missing google object: %v", i, sig)
		}
		if got := google["thought_signature"]; got != thoughtSignatureSentinel {
			t.Fatalf("tool call %d thought_signature = %v, want %q", i, got, thoughtSignatureSentinel)
		}
	}
}

func TestGeminiToolReplayNonGeminiTargetIsUntouched(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4-pro","max_tokens":10,"messages":[
		{"role":"user","content":[{"type":"text","text":"do two things"}]},
		{"role":"assistant","content":[
			{"type":"tool_use","id":"call_a","name":"lookup","input":{"q":"x"}},
			{"type":"tool_use","id":"call_b","name":"list","input":{}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_a","content":"ok"},
			{"type":"tool_result","tool_use_id":"call_b","content":"ok"}]}]}`)

	out, err := convertAnthropicRequestToOpenAI(body, &ConvertOptions{ResolvedModel: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if strings.Contains(string(out), "extra_content") {
		t.Fatalf("non-Gemini target must not carry extra_content\n%s", out)
	}
	for i, sig := range toolCallSignatures(t, out) {
		if sig != nil {
			t.Fatalf("tool call %d unexpectedly carries extra_content: %v", i, sig)
		}
	}
}

func TestGeminiToolReplayOnlyAssistantTurns(t *testing.T) {
	// A user message (and a text-only assistant message) that merely *mention*
	// the substring "tool_use" must not trip the seam. Only assistant turns
	// with tool_calls are eligible for mutation.
	body := []byte(`{"model":"gemini-4-hydrogen","max_tokens":10,"messages":[
		{"role":"user","content":[{"type":"text","text":"the literal token tool_use appears in this prose"}]},
		{"role":"assistant","content":[{"type":"text","text":"you wrote tool_use in your message"}]}]}`)

	out, err := convertAnthropicRequestToOpenAI(body, &ConvertOptions{ResolvedModel: "gemini-4-hydrogen"})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if strings.Contains(string(out), "extra_content") {
		t.Fatalf("prose containing the substring tool_use must not gain extra_content\n%s", out)
	}
}

// fatalReasoningStore is a ReasoningStore double whose Get/Set t.Fatal the
// test. It exists to make the SID/cache trap loud: the Gemini replay seam is
// only allowed to mutate tool_calls, so any cache read or write against a
// Gemini target fails the test instead of silently succeeding.
type fatalReasoningStore struct {
	t *testing.T
}

func (s fatalReasoningStore) Get(key string) *cacheEntry {
	s.t.Fatalf("ReasoningStore.Get(%q) called during Gemini replay; the seam must not read caches", key)
	return nil
}

func (s fatalReasoningStore) Set(key string, entry *cacheEntry) {
	s.t.Fatalf("ReasoningStore.Set(%q) called during Gemini replay; the seam must not write caches", key)
}

func (s fatalReasoningStore) Delete(key string) {}
func (s fatalReasoningStore) Len() int          { return 0 }
func (s fatalReasoningStore) Close() error      { return nil }

func TestGeminiToolReplaySentinelNeverReadsSessionStore(t *testing.T) {
	body := []byte(`{"model":"gemini-4-hydrogen","max_tokens":10,"messages":[
		{"role":"user","content":[{"type":"text","text":"do two things"}]},
		{"role":"assistant","content":[
			{"type":"tool_use","id":"call_a","name":"lookup","input":{"q":"x"}},
			{"type":"tool_use","id":"call_b","name":"list","input":{}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_a","content":"ok"},
			{"type":"tool_result","tool_use_id":"call_b","content":"ok"}]}]}`)

	// ReasoningCache double: any Get/Set fatals the test. A real SessionStore
	// is passed alongside with a SID set to prove the request path completes
	// without depending on per-session state either. SessionStore is a
	// concrete struct, not an interface, so it cannot be doubled — the fatal
	// ReasoningStore above is the assertion that makes cache/SID reads loud.
	fatal := fatalReasoningStore{t: t}
	cache := &ReasoningCache{toolStore: fatal, textStore: fatal, ctxStore: fatal}
	store := NewSessionStore()

	out, err := convertAnthropicRequestToOpenAI(body, &ConvertOptions{
		ResolvedModel:  "gemini-4-hydrogen",
		ReasoningCache: cache,
		SessionStore:   store,
		SID:            "sid-trap",
		Direction:      "request",
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if !strings.Contains(string(out), thoughtSignatureSentinel) {
		t.Fatalf("convert completed but the sentinel did not land\n%s", out)
	}
}

// TestGeminiToolReplayOpenAIChatPassthrough reproduces the 2026-11-17 fable
// incident: OpenCode speaks OpenAI Chat inbound, so the chat→chat passthrough
// branch never enters convertAnthropicRequestToOpenAI and the Gemini replay
// sentinel is missing. The captured upstream reply was HTTP 400
// "Function call is missing a thought_signature ... position 583".
func TestGeminiToolReplayOpenAIChatPassthrough(t *testing.T) {
	body := []byte(`{"model":"claude-fable-5","messages":[
		{"role":"system","content":"help the user"},
		{"role":"user","content":"read both files"},
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"functions_read_1","type":"function","function":{"name":"read","arguments":"{\"path\":\"/etc/hostname\"}"}},
			{"id":"functions_shell_2","type":"function","function":{"name":"shell","arguments":"{\"command\":\"hostname\"}"}}]},
		{"role":"tool","tool_call_id":"functions_read_1","content":"pi.lan"},
		{"role":"tool","tool_call_id":"functions_shell_2","content":"pi"}]}`)

	opts := &ConvertOptions{
		Direction: "request",
		ModelMap: ModelMap{{
			SourcePrefix: "claude-fable-5",
			TargetModel:  "gemini-4-hydrogen",
			Protocol:     "openai",
		}},
	}
	out, err := Convert(body, opts)
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if !strings.Contains(string(out), `"model":"gemini-4-hydrogen"`) {
		t.Fatalf("target model not rewritten:\n%s", out)
	}
	want := `"extra_content":{"google":{"thought_signature":"` + thoughtSignatureSentinel + `"}}`
	if got := strings.Count(string(out), want); got != 2 {
		t.Fatalf("sentinel tool calls = %d, want 2 (both replayed calls)\n%s", got, out)
	}
	sigs := toolCallSignatures(t, out)
	if len(sigs) != 2 {
		t.Fatalf("assistant replay should carry 2 tool calls, got %d\n%s", len(sigs), out)
	}
}

func TestGeminiToolReplayPassthroughLeavesNonGeminiAndUnknownFieldsAlone(t *testing.T) {
	body := []byte(`{"model":"claude-fable-5","opencode_private":{"keep":true},"messages":[
		{"role":"assistant","content":null,"tool_calls":[
			{"id":"c1","type":"function","function":{"name":"shell","arguments":"{}"}}]}]}`)

	// Non-Gemini target: byte-identical passthrough, no extra_content anywhere.
	out, err := Convert(body, &ConvertOptions{
		Direction: "request",
		ModelMap:  ModelMap{{SourcePrefix: "claude-fable-5", TargetModel: "deepseek-chat", Protocol: "openai"}},
	})
	if err != nil {
		t.Fatalf("convert non-gemini: %v", err)
	}
	if !strings.Contains(string(out), `"opencode_private":{"keep":true}`) {
		t.Fatalf("non-gemini passthrough must stay byte-identical:\n%s", out)
	}
	if strings.Contains(string(out), "extra_content") {
		t.Fatalf("sentinel leaked into non-gemini target:\n%s", out)
	}

	// Gemini target: unknown fields survive the map-based mutation.
	out, err = Convert(body, &ConvertOptions{
		Direction: "request",
		ModelMap:  ModelMap{{SourcePrefix: "claude-fable-5", TargetModel: "gemini-4-hydrogen", Protocol: "openai"}},
	})
	if err != nil {
		t.Fatalf("convert gemini: %v", err)
	}
	if !strings.Contains(string(out), `"opencode_private":{"keep":true}`) {
		t.Fatalf("gemini sentinel injection dropped unknown request fields:\n%s", out)
	}
	if got := strings.Count(string(out), thoughtSignatureSentinel); got != 1 {
		t.Fatalf("sentinel count = %d, want 1\n%s", got, out)
	}
}
