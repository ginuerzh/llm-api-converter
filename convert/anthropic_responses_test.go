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

func TestConvertAnthropicRequestToResponsesTextAndSystem(t *testing.T) {
	body := `{"model":"claude-opus-5-5","max_tokens":128,"system":[{"type":"text","text":"sys"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	out, err := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	if req["instructions"] != "sys" || req["max_output_tokens"] != float64(128) || req["store"] != false {
		t.Fatalf("unexpected request: %s", out)
	}
	if _, ok := req["messages"]; ok {
		t.Fatalf("messages leaked: %s", out)
	}
	input := req["input"].([]any)
	if got := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)["type"]; got != "input_text" {
		t.Fatalf("content type = %v", got)
	}
}

func TestConvertAnthropicRequestToResponsesToolsAndChoice(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"get_weather","description":"weather","input_schema":{"type":"object"}}],"tool_choice":{"type":"any","disable_parallel_tool_use":true}}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	tool := req["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" {
		t.Fatalf("tool = %+v", tool)
	}
	if req["tool_choice"] != "required" || req["parallel_tool_calls"] != false {
		t.Fatalf("choice/parallel = %v/%v", req["tool_choice"], req["parallel_tool_calls"])
	}
}

func TestConvertAnthropicRequestToResponsesToolUseAndResult(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Beijing"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}]}]}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	input := req["input"].([]any)
	call := input[0].(map[string]any)
	result := input[1].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "toolu_1" || call["arguments"] != `{"city":"Beijing"}` {
		t.Fatalf("call = %+v", call)
	}
	if result["type"] != "function_call_output" || result["call_id"] != "toolu_1" || result["output"] != "sunny" {
		t.Fatalf("result = %+v", result)
	}
}

func TestConvertAnthropicRequestToResponsesThinking(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":20000},"output_config":{"effort":"high"}}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	if got := req["reasoning"].(map[string]any); got["effort"] != "high" || got["summary"] != "auto" {
		t.Fatalf("reasoning = %+v", got)
	}
}

func TestConvertAnthropicRequestToResponsesImageAndFormat(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}],"output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	part := req["input"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if part["type"] != "input_image" || part["image_url"] != "data:image/png;base64,abc" {
		t.Fatalf("image = %+v", part)
	}
	if req["text"].(map[string]any)["format"].(map[string]any)["type"] != "json_schema" {
		t.Fatalf("text = %+v", req["text"])
	}
}

func TestConvertResponsesResponseToAnthropicText(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"deepseek-v4.1-flash-1017","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":10,"output_tokens":2}}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{RequestModel: "claude-opus-5-5"})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if resp.Model != "claude-opus-5-5" || resp.Content[0].Type != "text" || resp.Content[0].Text != "Hello" {
		t.Fatalf("unexpected response: %s", out)
	}
	if resp.StopReason == nil || *resp.StopReason != "end_turn" {
		t.Fatalf("stop = %v", resp.StopReason)
	}
}

func TestConvertResponsesResponseToAnthropicReasoning(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"think"}]}]}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	if !strings.Contains(string(out), `"type":"thinking"`) || !strings.Contains(string(out), `"thinking":"think"`) || !strings.Contains(string(out), `"signature":""`) {
		t.Fatalf("unexpected response: %s", out)
	}
}

func TestConvertResponsesResponseToAnthropicToolCall(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"function_call","call_id":"toolu_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}]}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	b := resp.Content[0]
	if b.Type != "tool_use" || b.ID != "toolu_1" || b.Name != "get_weather" {
		t.Fatalf("block = %+v", b)
	}
	if resp.StopReason == nil || *resp.StopReason != "tool_use" {
		t.Fatalf("stop = %v", resp.StopReason)
	}
	if b.Input.(map[string]any)["city"] != "Beijing" {
		t.Fatalf("input = %+v", b.Input)
	}
}

func TestConvertResponsesResponseToAnthropicUsage(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"message","content":[{"type":"output_text","text":"x"}]}],"usage":{"input_tokens":100,"output_tokens":20,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":5}}}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	u := resp.Usage
	if u.InputTokens != 15 || u.CacheReadInputTokens != 80 || u.CacheCreationInputTokens != 5 || u.OutputTokens != 20 {
		t.Fatalf("usage = %+v", u)
	}
}

func TestConvertResponsesResponseToAnthropicIncomplete(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model":"m","output":[]}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if resp.StopReason == nil || *resp.StopReason != "max_tokens" {
		t.Fatalf("stop = %v", resp.StopReason)
	}
}

func TestConvertRoutesAnthropicToResponses(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{
		SID: "sid-1", Direction: "request", SessionStore: store,
		ModelMap: ModelMap{{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4.1-flash-1017", Protocol: "responses"}},
	}
	out, err := Convert([]byte(`{"model":"claude-opus-5-5","max_tokens":8,"system":[{"type":"text","text":"sys"}],"messages":[{"role":"user","content":"hi"}]}`), opts)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), `"messages"`) || !strings.Contains(string(out), `"input"`) {
		t.Fatalf("not converted: %s", out)
	}
	sess := store.Get("sid-1")
	if sess == nil || sess.From != ProtocolAnthropic || sess.To != ProtocolOpenAIResponses {
		t.Fatalf("session = %+v", sess)
	}
}

func TestConvertRoutesResponsesReplyToAnthropicClient(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid-1", &Session{ID: "sid-1", From: ProtocolAnthropic, To: ProtocolOpenAIResponses})
	opts := &ConvertOptions{SID: "sid-1", Direction: "response", SessionStore: store, RequestModel: "claude-opus-5-5"}
	out, err := Convert([]byte(`{"id":"resp_1","object":"response","status":"completed","model":"deepseek-v4.1-flash-1017","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`), opts)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), `"object":"response"`) || !strings.Contains(string(out), `"stop_reason":"end_turn"`) {
		t.Fatalf("not converted: %s", out)
	}
}

func TestConvertResponsesClientStillPassesThrough(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid-1", &Session{ID: "sid-1", From: ProtocolOpenAIResponses, IsResponses: true})
	body := `{"id":"resp_1","object":"response","status":"completed","output":[]}`
	out, _ := Convert([]byte(body), &ConvertOptions{SID: "sid-1", Direction: "response", SessionStore: store})
	if string(out) != body {
		t.Fatalf("passthrough changed body: %s", out)
	}
}

func TestConvertUnknownResponsesPairFails(t *testing.T) {
	opts := &ConvertOptions{
		Direction: "request",
		ModelMap:  ModelMap{{SourcePrefix: "gpt", TargetModel: "deepseek-v4.1-flash-1017", Protocol: "responses"}},
	}
	_, err := Convert([]byte(`{"model":"gpt-5","tool_choice":"auto","messages":[{"role":"user","content":"hi"}]}`), opts)
	if err == nil || !strings.Contains(err.Error(), "no converter") {
		t.Fatalf("err = %v, want a missing-converter diagnostic", err)
	}
}
