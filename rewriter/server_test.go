package rewriter

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"llm-api-converter/convert"
)

func TestRewriteEndpoint_OpenAIRequest(t *testing.T) {
	opts := &Options{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	srv := newServer(opts)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	reqBody := rewriteRequest{
		Data: []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`),
	}
	resp := doPost(t, ts.URL, reqBody)

	if !resp.OK {
		t.Fatal("expected ok=true")
	}
	if len(resp.Data) == 0 {
		t.Fatal("expected non-empty data")
	}

	// Verify it's valid Anthropic request.
	var acr convert.AnthropicRequest
	if err := json.Unmarshal(resp.Data, &acr); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, resp.Data)
	}
	if acr.Model != "claude-sonnet-4-20250514" {
		t.Fatalf("model: want claude-sonnet-4-20250514, got %q", acr.Model)
	}
	if len(acr.Messages) != 1 {
		t.Fatalf("want 1 message, got %d", len(acr.Messages))
	}
}

func TestRewriteEndpoint_AnthropicResponse(t *testing.T) {
	opts := &Options{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	srv := newServer(opts)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	reqBody := rewriteRequest{
		Data: []byte(`{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude-sonnet-4","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`),
	}
	resp := doPost(t, ts.URL, reqBody)

	if !resp.OK {
		t.Fatal("expected ok=true")
	}

	// Verify it's valid OpenAI response.
	var ocr convert.OpenAIChatResponse
	if err := json.Unmarshal(resp.Data, &ocr); err != nil {
		t.Fatalf("unmarshal error: %v\nbody: %s", err, resp.Data)
	}
	if len(ocr.Choices) != 1 {
		t.Fatalf("want 1 choice, got %d", len(ocr.Choices))
	}
	if ocr.Choices[0].Message.Role != "assistant" {
		t.Fatalf("role: want assistant, got %q", ocr.Choices[0].Message.Role)
	}
}

func TestRewriteEndpoint_Passthrough(t *testing.T) {
	opts := &Options{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	srv := newServer(opts)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	reqBody := rewriteRequest{
		Data: []byte(`{"some":"random json"}`),
	}
	resp := doPost(t, ts.URL, reqBody)

	if !resp.OK {
		t.Fatal("expected ok=true")
	}
	if string(resp.Data) != `{"some":"random json"}` {
		t.Fatalf("expected passthrough, got %s", resp.Data)
	}
}

func TestRewriteEndpoint_EmptyBody(t *testing.T) {
	opts := &Options{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	srv := newServer(opts)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	// Missing data is nil/empty — should return ok=false.
	reqBody := rewriteRequest{}
	resp := doPost(t, ts.URL, reqBody)
	if resp.OK {
		t.Fatal("expected ok=false for empty request")
	}
}

func doPost(t *testing.T, url string, req rewriteRequest) rewriteResponse {
	t.Helper()

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := http.Post(url+"/rewrite", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var rr rewriteResponse
	if err := json.NewDecoder(resp.Body).Decode(&rr); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return rr
}

// -- Benchmark ---------------------------------------------------------------

func BenchmarkRewrite(b *testing.B) {
	// Build a realistic-sized payload: OpenAI request with a few messages.
	req := rewriteRequest{
		Data: []byte(`{"model":"gpt-4","messages":[
			{"role":"system","content":"You are helpful."},
			{"role":"user","content":"Explain quantum computing in simple terms."},
			{"role":"assistant","content":"Quantum computing uses qubits."},
			{"role":"user","content":"Tell me more."}
		],"temperature":0.7,"max_tokens":2048}`),
	}
	payload, _ := json.Marshal(req)

	srv := newServer(&Options{Model: "claude-sonnet-4-20250514", MaxTokens: 8192})
	ts := httptest.NewServer(srv)
	defer ts.Close()
	url := ts.URL + "/rewrite"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, err := http.Post(url, "application/json", bytes.NewReader(payload))
		if err != nil {
			b.Fatal(err)
		}
		_, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
	}
}

// -- Integration smoke test helper ------------------------------------------

func init() {
	// Suppress slog output during tests.
	slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError})))
}

// TestIntegrationSmokeTest verifies the full rewrite cycle.
func TestIntegrationSmokeTest(t *testing.T) {
	opts := &Options{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	srv := newServer(opts)
	ts := httptest.NewServer(srv)
	defer ts.Close()

	req := rewriteRequest{
		Data: []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`),
	}
	payload, _ := json.Marshal(req)
	resp, _ := http.Post(ts.URL+"/rewrite", "application/json", bytes.NewReader(payload))
	var rr rewriteResponse
	json.NewDecoder(resp.Body).Decode(&rr)
	resp.Body.Close()

	if !rr.OK {
		t.Fatal("expected ok=true")
	}
}
