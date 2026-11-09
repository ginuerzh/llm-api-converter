package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDetectResponsesStreamingEvents(t *testing.T) {
	for _, typ := range []string{"response.created", "response.output_item.added", "response.output_text.delta", "response.completed"} {
		if got := detectSource(map[string]any{"type": typ}); got != ProtocolOpenAIResponses {
			t.Fatalf("detectSource(%q) = %v, want Responses", typ, got)
		}
	}
	if got := detectSource(map[string]any{"type": "message_start"}); got != ProtocolAnthropic {
		t.Fatalf("message_start = %v, want Anthropic", got)
	}
}

func TestResponsesRequestReasoningMarshal(t *testing.T) {
	req := ResponsesRequest{
		Model: "m", Stream: true, Store: boolPtr(false),
		Reasoning: &ResponsesReasoning{Effort: "high", Summary: "auto"},
		Text:      &ResponsesText{Format: map[string]any{"type": "json_schema", "name": "response"}},
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{`"reasoning":{"effort":"high","summary":"auto"}`, `"text":{"format":`, `"store":false`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in %s", want, s)
		}
	}
}
