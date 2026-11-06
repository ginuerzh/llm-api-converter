package convert

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
)

func TestP2_ReasoningCacheConcurrentGetPut(t *testing.T) {
	cache := NewReasoningCache(16)
	defer cache.Close()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 2000; j++ {
				cache.Put([]string{"tool_1"}, fmt.Sprintf("reasoning-%d-%d", i, j))
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < 2000; j++ {
				cache.Get([]string{"tool_1"})
			}
		}()
	}
	wg.Wait()
}

func TestP2_OpenAIStopReasonNormalization(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{
			name: "function_call",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"function_call","message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
			want: "tool_use",
		},
		{
			name: "content_filter",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","content":"filtered"}}]}`,
			want: "refusal",
		},
		{
			name: "unknown",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"weird","message":{"role":"assistant","content":"ok"}}]}`,
			want: "end_turn",
		},
		{
			name: "missing",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`,
			want: "end_turn",
		},
		{
			name: "content_filter_with_tools",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"content_filter","message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
			want: "tool_use",
		},
		{
			name: "length_with_tools",
			body: `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`,
			want: "max_tokens",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := convertOpenAIResponseToAnthropic([]byte(tc.body), nil)
			if err != nil {
				t.Fatal(err)
			}
			var resp AnthropicResponse
			if err := json.Unmarshal(out, &resp); err != nil {
				t.Fatal(err)
			}
			if resp.StopReason == nil || *resp.StopReason != tc.want {
				t.Fatalf("stop_reason = %v, want %q", resp.StopReason, tc.want)
			}
		})
	}
}

func TestP2_StreamStopReasonNormalization(t *testing.T) {
	t.Run("content_filter", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}`))
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"stop_reason":"refusal"`) {
			t.Fatalf("stream stop = %s, want refusal", end)
		}
	})

	t.Run("unknown", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}`))
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"weird"}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"stop_reason":"end_turn"`) {
			t.Fatalf("stream stop = %s, want end_turn", end)
		}
	})

	t.Run("content_filter_with_tools", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"content_filter"}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"stop_reason":"tool_use"`) {
			t.Fatalf("stream stop = %s, want tool_use", end)
		}
	})
}

func TestP2_AnthropicStopReasonNormalization(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"refusal", `{"id":"msg_1","model":"claude","stop_reason":"refusal","content":[{"type":"text","text":"no"}]}`, "content_filter"},
		{"pause_turn", `{"id":"msg_1","model":"claude","stop_reason":"pause_turn","content":[{"type":"text","text":"wait"}]}`, "stop"},
		{"unknown", `{"id":"msg_1","model":"claude","stop_reason":"weird","content":[{"type":"text","text":"ok"}]}`, "stop"},
		{"missing", `{"id":"msg_1","model":"claude","content":[{"type":"text","text":"ok"}]}`, "stop"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := convertAnthropicResponseToOpenAI([]byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			var resp OpenAIChatResponse
			if err := json.Unmarshal(out, &resp); err != nil {
				t.Fatal(err)
			}
			if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != tc.want {
				t.Fatalf("finish_reason = %v, want %q", resp.Choices[0].FinishReason, tc.want)
			}
		})
	}
}

func TestP2_OpenAIRefusalText(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":null,"refusal":"I cannot help with that."}}]}`
	out, err := convertOpenAIResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var resp AnthropicResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" || resp.Content[0].Text != "I cannot help with that." {
		t.Fatalf("content = %+v, want refusal text", resp.Content)
	}
	if resp.StopReason == nil || *resp.StopReason != "refusal" {
		t.Fatalf("stop_reason = %v, want refusal", resp.StopReason)
	}
}

func TestP2_StreamRefusalDelta(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"refusal":"I cannot"},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	end := string(sc.HandleStreamEnd())
	all := string(out) + end
	if !strings.Contains(all, `"text_delta"`) || !strings.Contains(all, `"I cannot"`) {
		t.Fatalf("refusal delta missing: %s", all)
	}
	if !strings.Contains(end, `"stop_reason":"refusal"`) {
		t.Fatalf("stream stop = %s, want refusal", end)
	}
}

func TestP2_EmptyChoicesContentArray(t *testing.T) {
	out, err := convertOpenAIResponseToAnthropic([]byte(`{"id":"x","model":"gpt-4","choices":[]}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out, &raw); err != nil {
		t.Fatal(err)
	}
	if string(raw["content"]) != "[]" {
		t.Fatalf("content = %s, want []", raw["content"])
	}
	var resp AnthropicResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.StopReason == nil || *resp.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %v, want end_turn", resp.StopReason)
	}
}
