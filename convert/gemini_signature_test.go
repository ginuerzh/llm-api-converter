package convert

import (
	"encoding/json"
	"testing"
)

// geminiSigOpts returns options that convert a Gemini response for a client
// that speaks clientProto, with a session store shared by later requests.
func geminiSigOpts(store *SessionStore, clientProto Protocol) *ConvertOptions {
	store.Set("sig-sid", &Session{ID: "sig-sid", From: clientProto})
	return &ConvertOptions{
		MaxTokens:    100,
		SessionStore: store,
		SID:          "sig-sid",
		Direction:    "response",
		ModelMap:     ModelMap{{SourcePrefix: "*", TargetModel: "gemini-pro", Protocol: "gemini"}},
	}
}

// toolIDsInAnthropic returns the tool_use IDs of an Anthropic response, in
// content order.
func toolIDsInAnthropic(t *testing.T, body []byte) []string {
	t.Helper()
	var resp AnthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("unmarshal Anthropic response: %v\n%s", err, body)
	}
	var ids []string
	for _, c := range resp.Content {
		if c.Type == "tool_use" {
			ids = append(ids, c.ID)
		}
	}
	return ids
}

// thoughtSigForCall converts an Anthropic request carrying a tool_use with
// callID and returns the thoughtSignature the Gemini functionCall was given.
func thoughtSigForCall(t *testing.T, store *SessionStore, callID, name string) string {
	t.Helper()
	req := `{"model":"claude-sonnet-4-20250514","max_tokens":100,"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"` + callID + `","name":"` + name + `","input":{"p":"a"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + callID + `","content":"ok"}]}
	]}`
	out, err := Convert([]byte(req), &ConvertOptions{
		MaxTokens:    100,
		SessionStore: store,
		SID:          "sig-sid",
		Direction:    "request",
		ModelMap:     ModelMap{{SourcePrefix: "*", TargetModel: "gemini-pro", Protocol: "gemini"}},
	})
	if err != nil {
		t.Fatalf("convert request: %v", err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(out, &g); err != nil {
		t.Fatalf("unmarshal Gemini request: %v\n%s", err, out)
	}
	for _, c := range g.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil && p.FunctionCall.Name == name {
				return p.ThoughtSignature
			}
		}
	}
	t.Fatalf("no functionCall %q in: %s", name, out)
	return ""
}

func TestGeminiSignature_KeptPerCallID_Anthropic(t *testing.T) {
	store := NewSessionStore()
	resp := `{"candidates":[{"content":{"parts":[
		{"functionCall":{"name":"read","args":{"p":"a"}},"thoughtSignature":"sigA"},
		{"functionCall":{"name":"read","args":{"p":"b"}},"thoughtSignature":"sigB"}
	],"role":"model"},"finishReason":"STOP"}]}`

	out, err := Convert([]byte(resp), geminiSigOpts(store, ProtocolAnthropic))
	if err != nil {
		t.Fatalf("convert response: %v", err)
	}
	ids := toolIDsInAnthropic(t, out)
	if len(ids) != 2 {
		t.Fatalf("want 2 tool_use ids, got %d: %s", len(ids), out)
	}
	if got := thoughtSigForCall(t, store, ids[0], "read"); got != "sigA" {
		t.Fatalf("first call signature: want sigA, got %q", got)
	}
	if got := thoughtSigForCall(t, store, ids[1], "read"); got != "sigB" {
		t.Fatalf("second call signature: want sigB, got %q", got)
	}
}

func TestGeminiSignature_KeptPerCallID_OpenAIStream(t *testing.T) {
	store := NewSessionStore()
	store.Set("sig-sid", &Session{ID: "sig-sid", From: ProtocolOpenAIChat})

	chunks := []string{
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"p":"a"}},"thoughtSignature":"sigA"}],"role":"model"}}]}`,
		`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"p":"b"}},"thoughtSignature":"sigB"}],"role":"model"}}]}`,
	}
	var ids []string
	for _, chunk := range chunks {
		evt := &SSEEvent{Event: "message", Data: chunk}
		out, err := convertGeminiStreamChunk(evt, &ConvertOptions{
			MaxTokens:    100,
			SessionStore: store,
			SID:          "sig-sid",
			ModelMap:     ModelMap{{SourcePrefix: "*", TargetModel: "gemini-pro", Protocol: "gemini"}},
		})
		if err != nil {
			t.Fatalf("convert chunk: %v", err)
		}
		var delta struct {
			Choices []struct {
				Delta struct {
					ToolCalls []OpenAIToolCall `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(dataPayload(string(out))), &delta); err != nil {
			t.Fatalf("unmarshal chunk: %v\n%s", err, out)
		}
		if len(delta.Choices) == 0 || len(delta.Choices[0].Delta.ToolCalls) == 0 {
			t.Fatalf("no tool call in chunk: %s", out)
		}
		ids = append(ids, delta.Choices[0].Delta.ToolCalls[0].ID)
	}

	req := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":null,"tool_calls":[{"id":"` + ids[0] + `","type":"function","function":{"name":"read","arguments":"{\"p\":\"a\"}"}}]},
		{"role":"tool","tool_call_id":"` + ids[0] + `","content":"ok"}
	]}`
	out, err := Convert([]byte(req), &ConvertOptions{
		MaxTokens:    100,
		SessionStore: store,
		SID:          "sig-sid",
		Direction:    "request",
		ModelMap:     ModelMap{{SourcePrefix: "*", TargetModel: "gemini-pro", Protocol: "gemini"}},
	})
	if err != nil {
		t.Fatalf("convert request: %v", err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(out, &g); err != nil {
		t.Fatalf("unmarshal Gemini request: %v\n%s", err, out)
	}
	var got string
	for _, c := range g.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil && p.FunctionCall.Name == "read" {
				got = p.ThoughtSignature
			}
		}
	}
	if got != "sigA" {
		t.Fatalf("first call signature: want sigA, got %q\nrequest: %s", got, out)
	}
}
