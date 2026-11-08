package engine

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ginuerzh/llm-api-converter/convert"
)

func TestRewriteSkipsNonChatPaths(t *testing.T) {
	e := New(Options{Model: "deepseek-chat", MaxTokens: 8192})
	body := []byte(`{"model":"claude-opus-4-5","messages":[]}`)

	for _, uri := range []string{"/v1/messages/count_tokens", "/v1/models", "/v1/embeddings"} {
		res, err := e.Rewrite(context.Background(), body, Meta{Direction: "request", URI: uri})
		if err != nil {
			t.Fatalf("%s: %v", uri, err)
		}
		if string(res.Data) != string(body) {
			t.Fatalf("%s: body changed: %s", uri, res.Data)
		}
	}
}

func TestRewriteConvertsOpenAIToAnthropic(t *testing.T) {
	e := New(Options{Model: "claude-sonnet-4-20250514", MaxTokens: 8192})
	body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)

	res, err := e.Rewrite(context.Background(), body, Meta{Direction: "request", URI: "/v1/chat/completions"})
	if err != nil {
		t.Fatal(err)
	}
	var acr convert.AnthropicRequest
	if err := json.Unmarshal(res.Data, &acr); err != nil {
		t.Fatalf("unmarshal: %v body=%s", err, res.Data)
	}
	if acr.Model != "gpt-4" {
		t.Fatalf("model = %q, want gpt-4", acr.Model)
	}
	if len(acr.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(acr.Messages))
	}
}

func TestRewriteReplacesModelOnPassthrough(t *testing.T) {
	e := New(Options{
		Model:     "deepseek-chat",
		MaxTokens: 8192,
		ResolveModel: func(client, effort string) (string, string, string) {
			if client == "claude-opus-4-5" {
				return "deepseek-v4-pro", "anthropic", "strong"
			}
			return "", "", ""
		},
	})
	body := []byte(`{"model":"claude-opus-4-5","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)

	res, err := e.Rewrite(context.Background(), body, Meta{Direction: "request", URI: "/v1/messages"})
	if err != nil {
		t.Fatal(err)
	}
	if res.ClientModel != "claude-opus-4-5" || res.TargetModel != "deepseek-v4-pro" {
		t.Fatalf("models = %q -> %q", res.ClientModel, res.TargetModel)
	}

	var got struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(res.Data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Model != "deepseek-v4-pro" {
		t.Fatalf("model = %q, want deepseek-v4-pro (passthrough must still rewrite)", got.Model)
	}
}

func TestResolveModelReceivesEffort(t *testing.T) {
	var gotEffort string
	e := New(Options{
		Model:     "deepseek-chat",
		MaxTokens: 8192,
		ResolveModel: func(client, effort string) (string, string, string) {
			gotEffort = effort
			return "deepseek-v4-flash", "openai", "fast"
		},
	})
	body := []byte(`{"model":"gpt-5.5","thinking":{"type":"enabled","budget_tokens":20000},"messages":[{"role":"user","content":"hi"}]}`)

	if _, err := e.Rewrite(context.Background(), body, Meta{Direction: "request", URI: "/v1/messages"}); err != nil {
		t.Fatal(err)
	}
	if gotEffort != "high" {
		t.Fatalf("effort = %q, want high", gotEffort)
	}
}

func TestNormalizeEffort(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"missing", `{"model":"m"}`, "none"},
		{"openai high", `{"model":"m","reasoning_effort":"high"}`, "high"},
		{"openai xhigh", `{"model":"m","reasoning_effort":"xhigh"}`, "high"},
		{"openai minimal", `{"model":"m","reasoning_effort":"minimal"}`, "low"},
		{"responses", `{"model":"m","reasoning":{"effort":"medium"}}`, "medium"},
		{"anthropic high", `{"model":"m","thinking":{"type":"enabled","budget_tokens":16384}}`, "high"},
		{"anthropic medium", `{"model":"m","thinking":{"type":"enabled","budget_tokens":4096}}`, "medium"},
		{"anthropic low", `{"model":"m","thinking":{"type":"enabled","budget_tokens":1024}}`, "low"},
		{"anthropic disabled", `{"model":"m","thinking":{"type":"disabled","budget_tokens":99999}}`, "none"},
		{"anthropic adaptive effort", `{"model":"m","thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`, "high"},
		{"anthropic max effort", `{"model":"m","output_config":{"effort":"max"}}`, "high"},
		{"anthropic output_config medium", `{"model":"m","output_config":{"effort":"medium"}}`, "medium"},
		{"gemini level", `{"model":"m","generationConfig":{"thinkingConfig":{"thinkingLevel":"HIGH"}}}`, "high"},
		{"gemini budget", `{"model":"m","generationConfig":{"thinkingConfig":{"thinkingBudget":20000}}}`, "high"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, eff := inspectRequest([]byte(tc.body))
			if eff != tc.want {
				t.Fatalf("effort = %q, want %q", eff, tc.want)
			}
		})
	}
}

func TestRewriteEndPhaseStreamError(t *testing.T) {
	e := New(Options{Model: "claude", MaxTokens: 8192})
	start := []byte(`data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)
	if _, err := e.Rewrite(context.Background(), start, Meta{
		Direction: "response", URI: "/v1/messages", SID: "s1", SSEPhase: "start",
	}); err != nil {
		t.Fatal(err)
	}
	res, err := e.Rewrite(context.Background(), nil, Meta{
		Direction: "response", URI: "/v1/messages", SID: "s1", SSEPhase: "end",
		StreamError: "upstream reset",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(res.Data), "event: error") {
		t.Fatalf("end phase = %s", res.Data)
	}
}
