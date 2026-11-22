package convert

import (
	"strings"
	"testing"
)

func chatHandler() *PassthroughStreamHandler {
	return NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)
}

const (
	chatFinishStop   = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	chatFinishTool   = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
	chatContentNull  = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
	chatErrorChunk   = "data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n"
	anthropicStop    = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	anthropicContent = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
)

func TestPassthroughChatEmitsDoneAfterFinishReason(t *testing.T) {
	h := chatHandler()
	if _, err := h.HandleChunk([]byte(chatFinishStop)); err != nil {
		t.Fatal(err)
	}
	if got := string(h.EndStream("")); got != "data: [DONE]" {
		t.Fatalf("end = %q, want \"data: [DONE]\"", got)
	}
}

func TestPassthroughChatTruncatedEmitsErrorOnly(t *testing.T) {
	h := chatHandler()
	if _, err := h.HandleChunk([]byte(chatContentNull)); err != nil {
		t.Fatal(err)
	}
	got := string(h.EndStream(""))
	if !strings.Contains(got, `"type":"stream_truncated"`) {
		t.Fatalf("end = %q, want a stream_truncated error", got)
	}
	if strings.Contains(got, "[DONE]") {
		t.Fatalf("end = %q, must not carry [DONE] after an error", got)
	}
	if !strings.HasPrefix(got, "data: {") {
		t.Fatalf("end = %q, want OpenAI chat data framing", got)
	}
}

func TestPassthroughChatTransportErrorIsCarried(t *testing.T) {
	h := chatHandler()
	got := string(h.EndStream("read tcp: i/o timeout"))
	if !strings.Contains(got, "i/o timeout") {
		t.Fatalf("end = %q, want the transport error in the message", got)
	}
}

func TestPassthroughChatSentinelCountsAsTerminal(t *testing.T) {
	h := chatHandler()
	h.markTerminal()
	if got := string(h.EndStream("")); got != "data: [DONE]" {
		t.Fatalf("end = %q, want \"data: [DONE]\"", got)
	}
}

func TestPassthroughChatErrorChunkIsTerminal(t *testing.T) {
	h := chatHandler()
	if _, err := h.HandleChunk([]byte(chatErrorChunk)); err != nil {
		t.Fatal(err)
	}
	if got := h.EndStream(""); got != nil {
		t.Fatalf("end = %q, want nil (in-band error already terminated the stream)", got)
	}
}

func TestPassthroughAnthropicHealthyNoDone(t *testing.T) {
	h := NewPassthroughStreamHandler("m", ProtocolAnthropic, false)
	if _, err := h.HandleChunk([]byte(anthropicStop)); err != nil {
		t.Fatal(err)
	}
	if got := h.EndStream(""); got != nil {
		t.Fatalf("end = %q, want nil for a healthy anthropic passthrough", got)
	}
}

func TestPassthroughAnthropicTruncatedEmitsEventError(t *testing.T) {
	h := NewPassthroughStreamHandler("m", ProtocolAnthropic, false)
	if _, err := h.HandleChunk([]byte(anthropicContent)); err != nil {
		t.Fatal(err)
	}
	got := string(h.EndStream(""))
	if !strings.HasPrefix(got, "event: error") || !strings.Contains(got, `"type":"stream_error"`) {
		t.Fatalf("end = %q, want an anthropic error event", got)
	}
	if strings.Contains(got, "[DONE]") || strings.Contains(got, "message_stop") {
		t.Fatalf("end = %q, must not fake a normal anthropic close", got)
	}
}

func TestPassthroughEmitErrorShapeFollowsProtocol(t *testing.T) {
	if got := string(chatHandler().EmitError(`a"b`)); !strings.HasPrefix(got, "data: {") || !strings.Contains(got, `a\"b`) {
		t.Fatalf("chat error = %q", got)
	}
	anthropic := NewPassthroughStreamHandler("m", ProtocolAnthropic, false)
	if got := string(anthropic.EmitError(`a"b`)); !strings.HasPrefix(got, "event: error") || !strings.Contains(got, `a\"b`) {
		t.Fatalf("anthropic error = %q", got)
	}
}
