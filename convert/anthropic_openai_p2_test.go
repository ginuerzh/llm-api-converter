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

func TestP2_ReasoningAliasRequest(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"ok","reasoning":"chain"},
		{"role":"user","content":"go on"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	if err := json.Unmarshal(out, &req); err != nil {
		t.Fatal(err)
	}
	var assistant *AnthropicMessage
	for i := range req.Messages {
		if req.Messages[i].Role == "assistant" {
			assistant = &req.Messages[i]
			break
		}
	}
	if assistant == nil || len(assistant.Content) == 0 || assistant.Content[0].Type != "thinking" {
		t.Fatalf("messages = %+v, want an assistant thinking block", req.Messages)
	}
	if assistant.Content[0].Thinking != "chain" {
		t.Fatalf("thinking = %q, want chain", assistant.Content[0].Thinking)
	}
}

func TestP2_ReasoningDetailsRequest(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"ok","reasoning_details":[
			{"type":"reasoning.text","text":"a"},
			{"type":"reasoning.summary","summary":"b"}]},
		{"role":"user","content":"go on"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	_ = json.Unmarshal(out, &req)
	var assistant *AnthropicMessage
	for i := range req.Messages {
		if req.Messages[i].Role == "assistant" {
			assistant = &req.Messages[i]
			break
		}
	}
	if assistant == nil || len(assistant.Content) == 0 {
		t.Fatalf("messages = %+v", req.Messages)
	}
	if got := assistant.Content[0].Thinking; got != "a\nb" {
		t.Fatalf("thinking = %q, want a\\nb", got)
	}
}

func TestP2_ReasoningContentWinsOverAlias(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"assistant","content":"ok","reasoning_content":"primary","reasoning":"alias"},
		{"role":"user","content":"go on"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "deepseek-chat"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), `"alias"`) {
		t.Fatalf("alias leaked: %s", out)
	}
	if !strings.Contains(string(out), `"primary"`) {
		t.Fatalf("primary reasoning lost: %s", out)
	}
}

func TestP2_ReasoningAliasStream(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning":"chain"},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"thinking_delta"`) || !strings.Contains(string(out), `"chain"`) {
		t.Fatalf("reasoning alias missing from stream: %s", out)
	}
}

func TestP2_ReasoningAliasResponse(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok","reasoning":"chain"}}]}`
	out, err := convertOpenAIResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if len(resp.Content) == 0 || resp.Content[0].Type != "thinking" || resp.Content[0].Thinking != "chain" {
		t.Fatalf("content = %+v, want thinking chain", resp.Content)
	}
}

func TestP2_StreamBlockStartPayloads(t *testing.T) {
	t.Run("text", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`))
		if !strings.Contains(string(out), `"content_block":{"type":"text","text":""}`) {
			t.Fatalf("text start = %s", out)
		}
	})

	t.Run("thinking", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}`))
		if !strings.Contains(string(out), `"content_block":{"type":"thinking","thinking":"","signature":""}`) {
			t.Fatalf("thinking start = %s", out)
		}
	})

	t.Run("tool", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		sc.HandleStreamStart()
		out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`))
		if !strings.Contains(string(out), `"input":{}`) {
			t.Fatalf("tool start = %s", out)
		}
	})
}

func TestP2_GeminiAnthropicBlockStarts(t *testing.T) {
	t.Run("text", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		c.HandleStreamStart()
		out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), `"content_block":{"type":"text","text":""`) {
			t.Fatalf("gemini text start = %s", out)
		}
	})

	t.Run("thinking", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		c.HandleStreamStart()
		out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"text":"think","thought":true}],"role":"model"}}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), `"content_block":{"type":"thinking","thinking":"","signature":""`) {
			t.Fatalf("gemini thinking start = %s", out)
		}
	})

	t.Run("tool", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		c.HandleStreamStart()
		out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}],"role":"model"},"finishReason":"STOP"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), `"input":{}`) {
			t.Fatalf("gemini tool start = %s", out)
		}
	})
}

func TestP2_NoArgumentToolEmitsEmptyObject(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	first, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
	end := string(sc.HandleStreamEnd())
	all := string(first) + end
	if !strings.Contains(all, `"partial_json":"{}"`) {
		t.Fatalf("no empty object for no-arg tool: %s", all)
	}
	if strings.Index(all, `"partial_json":"{}"`) > strings.Index(all, `"content_block_stop"`) {
		t.Fatalf("empty object emitted after block stop: %s", all)
	}
}

func TestP2_TwoNoArgumentToolsBothClose(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out1, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
	out2, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"g","arguments":""}}]},"finish_reason":null}]}`))
	end := string(sc.HandleStreamEnd())
	all := string(out1) + string(out2) + end
	if got := strings.Count(all, `"partial_json":"{}"`); got != 2 {
		t.Fatalf("empty object deltas = %d, want 2: %s", got, all)
	}
	if got := strings.Count(all, "event: content_block_start"); got != 2 {
		t.Fatalf("tool block starts = %d, want 2: %s", got, all)
	}
}

func TestP2_InterruptedStreamSingleTextBlock(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`))
	sc.SetInterrupted()
	end := sc.HandleStreamEnd()
	all := string(out) + string(end)
	if got := strings.Count(all, "event: content_block_start"); got != 1 {
		t.Fatalf("text block starts = %d, want 1: %s", got, all)
	}
	if got := strings.Count(all, "event: content_block_stop"); got != 1 {
		t.Fatalf("text block stops = %d, want 1: %s", got, all)
	}
	if !strings.Contains(all, "[stream interrupted]") {
		t.Fatalf("interruption marker missing: %s", all)
	}
}

func TestP2_EmptyInterruptedStreamSingleTextBlock(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	sc.SetInterrupted()
	end := string(sc.HandleStreamEnd())
	if got := strings.Count(end, "event: content_block_start"); got != 1 {
		t.Fatalf("text block starts = %d, want 1: %s", got, end)
	}
	if !strings.Contains(end, "[stream interrupted]") {
		t.Fatalf("interruption marker missing: %s", end)
	}
}

func assertAnthropicToolID(t *testing.T, id string) {
	t.Helper()
	if id == "" {
		t.Fatal("empty tool ID")
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			continue
		}
		t.Fatalf("tool ID %q contains invalid byte %q", id, c)
	}
}

func TestP2_ToolIDRequestPairing(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call.a/b","content":"ok"},
		{"role":"user","content":"done"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "claude-sonnet"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	if err := json.Unmarshal(out, &req); err != nil {
		t.Fatal(err)
	}
	var toolUseID, resultID string
	for _, msg := range req.Messages {
		for _, block := range msg.Content {
			switch block.Type {
			case "tool_use":
				toolUseID = block.ID
			case "tool_result":
				resultID = block.ToolUseID
			}
		}
	}
	assertAnthropicToolID(t, toolUseID)
	if toolUseID != "call_a_b" || resultID != "call_a_b" {
		t.Fatalf("tool_use=%q result=%q, want call_a_b for both", toolUseID, resultID)
	}
}

func TestP2_ToolIDResponseSanitized(t *testing.T) {
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	out, err := convertOpenAIResponseToAnthropic([]byte(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if len(resp.Content) == 0 || resp.Content[0].Type != "tool_use" {
		t.Fatalf("content = %+v", resp.Content)
	}
	assertAnthropicToolID(t, resp.Content[0].ID)
	if resp.Content[0].ID != "call_a_b" {
		t.Fatalf("id = %q, want call_a_b", resp.Content[0].ID)
	}
}

func TestP2_ToolIDStreamSanitized(t *testing.T) {
	sc := NewStreamConverter("claude", nil, nil)
	sc.HandleStreamStart()
	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), `"id":"call_a_b"`) {
		t.Fatalf("stream tool ID not sanitized: %s", out)
	}
}

func TestP2_ToolIDCacheUsesSanitizedID(t *testing.T) {
	cache := NewReasoningCache(8)
	body := `{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","reasoning_content":"chain","tool_calls":[{"id":"call.a/b","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
	if _, err := convertOpenAIResponseToAnthropic([]byte(body), &ConvertOptions{ReasoningCache: cache}); err != nil {
		t.Fatal(err)
	}
	if got, ok := cache.Get([]string{"call_a_b"}); !ok || got != "chain" {
		t.Fatalf("cache.Get(call_a_b) = %q, %v; want chain, true", got, ok)
	}
	if _, ok := cache.Get([]string{"call.a/b"}); ok {
		t.Fatal("cache stored the raw invalid ID")
	}
}

func TestP2_DeveloperRoleBecomesSystem(t *testing.T) {
	body := `{"model":"gpt-4","messages":[
		{"role":"developer","content":"be brief"},
		{"role":"user","content":"hi"}]}`
	out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "claude-sonnet"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	_ = json.Unmarshal(out, &req)
	if len(req.System) != 1 || req.System[0].Text != "be brief" {
		t.Fatalf("system = %+v, want developer text", req.System)
	}
	for _, msg := range req.Messages {
		if msg.Role == "user" && len(msg.Content) == 1 && msg.Content[0].Text == "be brief" {
			t.Fatal("developer text leaked into a user message")
		}
	}
}

func TestP2_ParallelToolCallsChat(t *testing.T) {
	tools := `"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]`
	cases := []struct {
		name        string
		extra       string
		wantDisable *bool
	}{
		{"false", `"parallel_tool_calls":false,`, boolPtr(true)},
		{"true", `"parallel_tool_calls":true,`, boolPtr(false)},
		{"none choice", `"parallel_tool_calls":false,"tool_choice":"none",`, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"model":"gpt-4",` + tc.extra + tools + `,"messages":[{"role":"user","content":"hi"}]}`
			out, err := convertOpenAIRequestToAnthropic([]byte(body), &ConvertOptions{ResolvedModel: "claude-sonnet"})
			if err != nil {
				t.Fatal(err)
			}
			var req AnthropicRequest
			_ = json.Unmarshal(out, &req)
			if tc.wantDisable == nil {
				if req.ToolChoice != nil && req.ToolChoice.DisableParallelToolUse != nil {
					t.Fatalf("disable_parallel_tool_use = %v, want absent", *req.ToolChoice.DisableParallelToolUse)
				}
				return
			}
			if req.ToolChoice == nil || req.ToolChoice.DisableParallelToolUse == nil ||
				*req.ToolChoice.DisableParallelToolUse != *tc.wantDisable {
				t.Fatalf("tool_choice = %+v, want disable=%v", req.ToolChoice, *tc.wantDisable)
			}
		})
	}
}

func TestP2_ParallelToolCallsReverse(t *testing.T) {
	body := `{"model":"claude","max_tokens":10,"tool_choice":{"type":"auto","disable_parallel_tool_use":true},
		"tools":[{"name":"f","input_schema":{"type":"object"}}],
		"messages":[{"role":"user","content":"hi"}]}`
	out, err := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: "gpt-4o"})
	if err != nil {
		t.Fatal(err)
	}
	var req OpenAIChatRequest
	_ = json.Unmarshal(out, &req)
	if req.ParallelToolCalls == nil || *req.ParallelToolCalls != false {
		t.Fatalf("parallel_tool_calls = %v, want false", req.ParallelToolCalls)
	}
}

func TestP2_ResponsesParallelToolCalls(t *testing.T) {
	body := `{"model":"gpt-5","parallel_tool_calls":false,"input":"hi",
		"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]}`
	out, err := ConvertResponsesToAnthropic([]byte(body), &ConvertOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatal(err)
	}
	var req AnthropicRequest
	_ = json.Unmarshal(out, &req)
	if req.ToolChoice == nil || req.ToolChoice.DisableParallelToolUse == nil ||
		!*req.ToolChoice.DisableParallelToolUse {
		t.Fatalf("tool_choice = %+v, want disable_parallel_tool_use=true", req.ToolChoice)
	}
}

func TestP2_TopKProviderGating(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"groq/llama-3.3-70b", true},
		{"openrouter/meta-llama/llama-3.1-70b", true},
		{"vllm/mistral-7b", true},
		{"openrouter/openai/gpt-5", false},
		{"gpt-4o", false},
		{"o3-mini", false},
		{"deepseek-chat", false},
	}
	body := `{"model":"claude","max_tokens":10,"top_k":40,"messages":[{"role":"user","content":"hi"}]}`
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			out, err := convertAnthropicRequestToOpenAI([]byte(body), &ConvertOptions{ResolvedModel: tc.model})
			if err != nil {
				t.Fatal(err)
			}
			var req OpenAIChatRequest
			_ = json.Unmarshal(out, &req)
			if (req.TopK != nil) != tc.want {
				t.Fatalf("top_k present = %v, want %v (top_k=%v)", req.TopK != nil, tc.want, req.TopK)
			}
		})
	}
}

func TestP2_ResponsesTruncatedToolErrors(t *testing.T) {
	sc := NewResponsesStreamConverter("gpt-4o")
	sc.HandleStreamStart()
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]},"finish_reason":null}]}`))
	end := string(sc.HandleStreamEnd())
	if !strings.Contains(end, "response.failed") || !strings.Contains(end, "incomplete tool call arguments") {
		t.Fatalf("truncated Responses stream end = %s, want response.failed", end)
	}
	if strings.Contains(end, "response.completed") || strings.Contains(end, "response.function_call_arguments.done") {
		t.Fatalf("truncated Responses stream closed normally: %s", end)
	}
}

func TestP2_AnthropicToResponsesTruncatedToolErrors(t *testing.T) {
	sc := NewAnthropicStreamConverter("claude")
	sc.HandleStreamStart()
	_, _ = sc.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f","input":{}}}`))
	_, _ = sc.HandleChunk([]byte(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`))
	end := string(sc.HandleStreamEnd())
	if !strings.Contains(end, "response.failed") || !strings.Contains(end, "incomplete tool call arguments") {
		t.Fatalf("truncated Anthropic→Responses end = %s, want response.failed", end)
	}
	if strings.Contains(end, "response.completed") || strings.Contains(end, "response.function_call_arguments.done") {
		t.Fatalf("truncated Anthropic→Responses stream closed normally: %s", end)
	}
}

func TestP2_ResponsesNoArgumentToolEmitsEmptyObject(t *testing.T) {
	t.Run("chat to responses", func(t *testing.T) {
		sc := NewResponsesStreamConverter("gpt-4o")
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"arguments":"{}"`) {
			t.Fatalf("no-arg Responses tool = %s, want arguments {}", end)
		}
	})

	t.Run("anthropic to responses", func(t *testing.T) {
		sc := NewAnthropicStreamConverter("claude")
		sc.HandleStreamStart()
		_, _ = sc.HandleChunk([]byte(`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f","input":{}}}`))
		end := string(sc.HandleStreamEnd())
		if !strings.Contains(end, `"arguments":"{}"`) {
			t.Fatalf("no-arg Anthropic→Responses tool = %s, want arguments {}", end)
		}
	})
}

func TestP2_GeminiEmitErrorSafety(t *testing.T) {
	message := "bad \"quote\"\nline\\slash"
	t.Run("gemini to openai", func(t *testing.T) {
		c := NewGeminiToOpenAIStreamConverter("gemini-2.5-flash")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
		payload := strings.TrimSuffix(strings.TrimPrefix(string(first), "data: "), "\n\n")
		var evt struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			t.Fatalf("invalid error JSON %q: %v", payload, err)
		}
		if evt.Error.Message != message {
			t.Fatalf("message = %q, want %q", evt.Error.Message, message)
		}
	})

	t.Run("gemini to anthropic", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
		if !strings.HasPrefix(string(first), "event: error\ndata: ") {
			t.Fatalf("error framing = %q", first)
		}
		var evt struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		payload := strings.TrimPrefix(string(first), "event: error\ndata: ")
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			t.Fatalf("invalid error JSON %q: %v", payload, err)
		}
		if evt.Error.Message != message {
			t.Fatalf("message = %q, want %q", evt.Error.Message, message)
		}
	})

	t.Run("anthropic to gemini", func(t *testing.T) {
		c := NewAnthropicToGeminiStreamConverter("gemini-2.5-flash")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
	})

	t.Run("openai to gemini", func(t *testing.T) {
		c := NewOpenAIToGeminiStreamConverter("gemini-2.5-flash")
		first := c.EmitError(message)
		if first == nil {
			t.Fatal("first error = nil")
		}
		if second := c.EmitError("again"); second != nil {
			t.Fatalf("second error = %s", second)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after error = %s", end)
		}
	})
}

func TestP2_GeminiUpstreamErrorTerminal(t *testing.T) {
	t.Run("gemini to openai", func(t *testing.T) {
		c := NewGeminiToOpenAIStreamConverter("gemini-2.5-flash")
		out, err := c.HandleChunk([]byte(`{"error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})

	t.Run("gemini to anthropic", func(t *testing.T) {
		c := NewGeminiToAnthropicStreamConverter("claude")
		out, err := c.HandleChunk([]byte(`{"error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})

	t.Run("anthropic to gemini", func(t *testing.T) {
		c := NewAnthropicToGeminiStreamConverter("gemini-2.5-flash")
		out, err := c.HandleChunk([]byte(`{"type":"error","error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})

	t.Run("openai to gemini", func(t *testing.T) {
		c := NewOpenAIToGeminiStreamConverter("gemini-2.5-flash")
		out, err := c.HandleChunk([]byte(`{"error":{"message":"boom"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), "boom") {
			t.Fatalf("error chunk = %s", out)
		}
		if end := c.HandleStreamEnd(); end != nil {
			t.Fatalf("normal close after upstream error = %s", end)
		}
	})
}

func TestP2_GeminiEndPhaseStreamErrorDispatched(t *testing.T) {
	cases := []struct {
		name       string
		handler    responsesStreamHandler
		wantError  string
		normalMark string
	}{
		{"gemini to openai", NewGeminiToOpenAIStreamConverter("gemini-2.5-flash"), "data: ", `[DONE]`},
		{"gemini to anthropic", NewGeminiToAnthropicStreamConverter("claude"), "event: error", "message_stop"},
		{"anthropic to gemini", NewAnthropicToGeminiStreamConverter("gemini-2.5-flash"), "data: ", "finishReason"},
		{"openai to gemini", NewOpenAIToGeminiStreamConverter("gemini-2.5-flash"), "data: ", "finishReason"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := NewSessionStore()
			store.Set("g1", &Session{ID: "g1", StreamHandler: tc.handler})
			opts := &ConvertOptions{SessionStore: store, Direction: "response", SID: "g1", ErrorMsg: `upstream "reset"`}
			out, err := HandleSSEEvent("g1", "end", 0, nil, opts)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(out), tc.wantError) {
				t.Fatalf("end-phase error not dispatched: %s", out)
			}
			if strings.Contains(string(out), tc.normalMark) {
				t.Fatalf("normal close followed the error: %s", out)
			}
		})
	}
}
