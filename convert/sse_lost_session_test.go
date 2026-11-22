package convert

import (
	"strings"
	"testing"
)

// A stream can lose its session mid-flight (store eviction, restart, aborted
// peer). The event phase must then keep the client's protocol: a chat chunk
// for a chat client is forwarded unchanged, never converted into Anthropic
// events, and the end phase still closes the stream with one [DONE].
func TestHandleSSEEvent_UnknownSessionKeepsClientProtocol(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/chat/completions", SID: "lost-chat"}

	out, err := HandleSSEEvent("lost-chat", "event", 7, []byte(chatContentNull), opts)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"choices"`) {
		t.Fatalf("event = %q, want the chat chunk forwarded", out)
	}
	if strings.Contains(string(out), "content_block") || strings.HasPrefix(string(out), "event:") {
		t.Fatalf("event = %q, chat chunk was converted to another protocol", out)
	}

	if _, err := HandleSSEEvent("lost-chat", "event", 8, []byte(chatFinishTool), opts); err != nil {
		t.Fatal(err)
	}
	end, err := HandleSSEEvent("lost-chat", "end", 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(end); got != "data: [DONE]" {
		t.Fatalf("end = %q, want \"data: [DONE]\"", got)
	}
}

// The fallback must still convert when the event speaks a different protocol
// than the client: an Anthropic client reading a chat upstream keeps getting
// Anthropic events.
func TestHandleSSEEvent_UnknownSessionStillConvertsAcrossProtocols(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/messages", SID: "lost-anthropic"}

	out, err := HandleSSEEvent("lost-anthropic", "event", 3, []byte(chatContentNull), opts)
	if err != nil {
		t.Fatal(err)
	}
	got := string(out)
	if !strings.Contains(got, "message_start") && !strings.Contains(got, "content_block") {
		t.Fatalf("out = %q, want anthropic events", got)
	}
	if strings.Contains(got, `"choices"`) {
		t.Fatalf("out = %q, chat chunk leaked to an anthropic client", got)
	}
}
