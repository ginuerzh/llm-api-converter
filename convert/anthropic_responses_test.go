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
