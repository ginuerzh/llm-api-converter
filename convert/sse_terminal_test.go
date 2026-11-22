package convert

import (
	"strings"
	"testing"
)

func chatOpts(store *SessionStore) *ConvertOptions {
	return &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/chat/completions", SID: "s1"}
}

// The production shape: finish_reason + a usage-only chunk, upstream [DONE] absent.
func TestHandleSSEEvent_ChatPassthroughNoUpstreamDoneStillEndsWithDone(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	mustEvent(t, "s1", "event", chatFinishStop, opts)
	mustEvent(t, "s1", "event", "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}", opts)
	end, err := HandleSSEEvent("s1", "end", 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(end); got != "data: [DONE]" {
		t.Fatalf("end = %q, want \"data: [DONE]\"", got)
	}
}

func TestHandleSSEEvent_ChatPassthroughUpstreamDoneEmitsOnce(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	mustEvent(t, "s1", "event", chatFinishTool, opts)
	if mid, err := HandleSSEEvent("s1", "event", 2, []byte("data: [DONE]"), opts); err != nil || len(mid) != 0 {
		t.Fatalf("sentinel event = %q, %v; want swallowed", mid, err)
	}
	end, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	if got := string(end); got != "data: [DONE]" {
		t.Fatalf("end = %q, want exactly one \"data: [DONE]\"", got)
	}
}

func TestHandleSSEEvent_ChatPassthroughTruncatedReportsError(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	mustEvent(t, "s1", "event", chatContentNull, opts)
	end, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	got := string(end)
	if !strings.Contains(got, `"type":"stream_truncated"`) || strings.Contains(got, "[DONE]") {
		t.Fatalf("end = %q, want a stream_truncated error and no [DONE]", got)
	}
}

func TestHandleSSEEvent_ChatPassthroughTransportErrorCarried(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	opts.ErrorMsg = "read tcp: i/o timeout"
	end, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	if !strings.Contains(string(end), "i/o timeout") {
		t.Fatalf("end = %q, want the transport error carried", end)
	}
}

func TestHandleSSEEvent_AnthropicPassthroughNeverGetsDone(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/messages", SID: "a1"}
	mustEvent(t, "a1", "start", "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude\"}}", opts)
	mustEvent(t, "a1", "event", anthropicStop, opts)
	end, _ := HandleSSEEvent("a1", "end", 0, nil, opts)
	if len(end) != 0 {
		t.Fatalf("end = %q, want nothing for a healthy anthropic passthrough", end)
	}
}

func TestHandleSSEEvent_AnthropicPassthroughTruncatedEmitsError(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/messages", SID: "a2"}
	mustEvent(t, "a2", "start", "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude\"}}", opts)
	mustEvent(t, "a2", "event", anthropicContent, opts)
	end, _ := HandleSSEEvent("a2", "end", 0, nil, opts)
	if !strings.HasPrefix(string(end), "event: error") {
		t.Fatalf("end = %q, want an anthropic error event", end)
	}
}

func TestHandleSSEEvent_EndWithoutSessionStaysSilent(t *testing.T) {
	opts := chatOpts(NewSessionStore())
	end, err := HandleSSEEvent("nope", "end", 0, nil, opts)
	if err != nil || len(end) != 0 {
		t.Fatalf("end = %q, %v; want silence", end, err)
	}
}

func mustEvent(t *testing.T, sid, phase, data string, opts *ConvertOptions) []byte {
	t.Helper()
	out, err := HandleSSEEvent(sid, phase, 0, []byte(data), opts)
	if err != nil {
		t.Fatalf("%s %s: %v", sid, phase, err)
	}
	return out
}

// A non-chat, non-anthropic passthrough (Gemini today, Responses on the
// fallback path) has no terminal-marker concept: its end phase must stay
// silent, never emit an Anthropic-shaped error.
func TestHandleSSEEvent_GeminiPassthroughEndStaysSilent(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1beta/models/gemini-2.0:streamGenerateContent", SID: "g1"}
	start := "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}"
	mustEvent(t, "g1", "start", start, opts)
	end, err := HandleSSEEvent("g1", "end", 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(end) != 0 {
		t.Fatalf("end = %q, want nothing for a Gemini passthrough", end)
	}
}

// A stream whose only event is the upstream sentinel must not forward it and
// then synthesize a second one.
func TestHandleSSEEvent_ChatPassthroughSentinelOnlyFirstEvent(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	start := mustEvent(t, "s1", "start", "data: [DONE]", opts)
	if len(start) != 0 {
		t.Fatalf("start = %q, want the sentinel swallowed", start)
	}
	end, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	if got := string(end); got != "data: [DONE]" {
		t.Fatalf("end = %q, want exactly one \"data: [DONE]\"", got)
	}
}

// The same sentinel must never reach an Anthropic client.
func TestHandleSSEEvent_AnthropicPassthroughSentinelNeverReachesClient(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/messages", SID: "a3"}
	start := mustEvent(t, "a3", "start", "data: [DONE]", opts)
	if strings.Contains(string(start), "[DONE]") {
		t.Fatalf("start = %q, sentinel leaked to an anthropic client", start)
	}
	end, _ := HandleSSEEvent("a3", "end", 0, nil, opts)
	if strings.Contains(string(end), "[DONE]") || len(end) != 0 {
		t.Fatalf("end = %q, want nothing for a sentinel-terminated anthropic stream", end)
	}
}
