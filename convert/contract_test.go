package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"testing"
)

// These validators are an independent contract check, not a round trip: they
// encode the parts of the Anthropic and OpenAI wire schemas that strict
// endpoints enforce. A converter output that passes here is at least
// structurally valid even when no real provider is available.

var contractToolIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

func contractAdd(errs *[]string, format string, args ...any) {
	*errs = append(*errs, fmt.Sprintf(format, args...))
}

func contractContains(list []string, want string) bool {
	for _, item := range list {
		if item == want {
			return true
		}
	}
	return false
}

func contractObject(v any) (map[string]any, bool) {
	m, ok := v.(map[string]any)
	return m, ok && m != nil
}

func validateAnthropicRequest(body []byte) []string {
	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return []string{fmt.Sprintf("unmarshal Anthropic request: %v", err)}
	}

	var errs []string
	if req.Model == "" {
		contractAdd(&errs, "model is empty")
	}
	if req.MaxTokens <= 0 {
		contractAdd(&errs, "max_tokens = %d, want > 0", req.MaxTokens)
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 1) {
		contractAdd(&errs, "temperature = %v, want [0,1]", *req.Temperature)
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		contractAdd(&errs, "top_p = %v, want [0,1]", *req.TopP)
	}
	if req.TopK != nil && *req.TopK < 0 {
		contractAdd(&errs, "top_k = %d, want >= 0", *req.TopK)
	}
	if len(req.Messages) == 0 {
		contractAdd(&errs, "messages is empty")
	}
	if len(req.Messages) > 0 && req.Messages[0].Role != "user" {
		contractAdd(&errs, "messages[0].role = %q, want user", req.Messages[0].Role)
	}

	for i, block := range req.System {
		if block.Type != "text" {
			contractAdd(&errs, "system[%d].type = %q, want text", i, block.Type)
		}
		if strings.TrimSpace(block.Text) == "" {
			contractAdd(&errs, "system[%d].text is empty", i)
		}
	}

	toolNames := make(map[string]bool, len(req.Tools))
	for i, tool := range req.Tools {
		if tool.Name == "" {
			contractAdd(&errs, "tools[%d].name is empty", i)
		}
		if toolNames[tool.Name] {
			contractAdd(&errs, "tools[%d].name = %q is duplicated", i, tool.Name)
		}
		toolNames[tool.Name] = true
		if _, ok := contractObject(tool.InputSchema); !ok {
			contractAdd(&errs, "tools[%d].input_schema is not an object", i)
		}
	}
	if req.ToolChoice != nil {
		if len(req.Tools) == 0 {
			contractAdd(&errs, "tool_choice is set without tools")
		}
		switch req.ToolChoice.Type {
		case "auto", "none", "any":
		case "tool":
			if req.ToolChoice.Name == "" {
				contractAdd(&errs, "tool_choice.name is empty")
			} else if !toolNames[req.ToolChoice.Name] {
				contractAdd(&errs, "tool_choice.name = %q is not a declared tool", req.ToolChoice.Name)
			}
		default:
			contractAdd(&errs, "tool_choice.type = %q is invalid", req.ToolChoice.Type)
		}
	}

	if req.Metadata != nil {
		meta, ok := contractObject(req.Metadata)
		if !ok {
			contractAdd(&errs, "metadata is not an object")
		} else {
			for key, value := range meta {
				if key != "user_id" {
					contractAdd(&errs, "metadata.%s is not allowed", key)
					continue
				}
				if s, ok := value.(string); !ok || s == "" {
					contractAdd(&errs, "metadata.user_id is not a non-empty string")
				}
			}
		}
	}

	if req.OutputConfig != nil && req.OutputConfig.Format != nil {
		format := req.OutputConfig.Format
		if format.Type != "json_schema" {
			contractAdd(&errs, "output_config.format.type = %q, want json_schema", format.Type)
		}
		if _, ok := contractObject(format.Schema); !ok {
			contractAdd(&errs, "output_config.format.schema is not an object")
		}
	}

	var pending []string
	for i, msg := range req.Messages {
		prefix := fmt.Sprintf("messages[%d]", i)
		if msg.Role != "user" && msg.Role != "assistant" {
			contractAdd(&errs, "%s.role = %q is invalid", prefix, msg.Role)
		}
		if len(msg.Content) == 0 {
			contractAdd(&errs, "%s.content is empty", prefix)
		}

		if msg.Role == "assistant" {
			if len(pending) > 0 {
				contractAdd(&errs, "%s: assistant turn follows unanswered tool_use", prefix)
			}
			pending = nil
			for j, block := range msg.Content {
				if block.Type == "tool_use" {
					pending = append(pending, block.ID)
					_ = j
				}
			}
			validateAnthropicBlocks(&errs, prefix, req.Model, msg.Content)
			continue
		}

		var answered = make(map[string]bool)
		sawNonResult := false
		for j, block := range msg.Content {
			if block.Type == "tool_result" {
				if sawNonResult {
					contractAdd(&errs, "%s.content[%d]: tool_result must precede text/image blocks", prefix, j)
				}
				if !contractContains(pending, block.ToolUseID) {
					contractAdd(&errs, "%s.content[%d].tool_use_id = %q has no open tool_use", prefix, j, block.ToolUseID)
				}
				if answered[block.ToolUseID] {
					contractAdd(&errs, "%s.content[%d].tool_use_id = %q is duplicated", prefix, j, block.ToolUseID)
				}
				answered[block.ToolUseID] = true
				continue
			}
			sawNonResult = true
		}
		for _, id := range pending {
			if !answered[id] {
				contractAdd(&errs, "%s: missing tool_result for tool_use %q", prefix, id)
			}
		}
		pending = nil
		validateAnthropicBlocks(&errs, prefix, req.Model, msg.Content)
	}
	if len(pending) > 0 {
		contractAdd(&errs, "trailing unanswered tool_use: %v", pending)
	}

	return errs
}

func validateAnthropicBlocks(errs *[]string, prefix, model string, blocks []AnthropicContent) {
	for i, block := range blocks {
		where := fmt.Sprintf("%s.content[%d]", prefix, i)
		switch block.Type {
		case "text":
			if strings.TrimSpace(block.Text) == "" {
				contractAdd(errs, "%s.text is empty", where)
			}
		case "image":
			if block.Source == nil {
				contractAdd(errs, "%s.source is nil", where)
				continue
			}
			switch block.Source.Type {
			case "base64":
				if block.Source.MediaType == "" || block.Source.Data == "" {
					contractAdd(errs, "%s.source has empty media_type/data", where)
				}
			case "url":
				if block.Source.URL == "" {
					contractAdd(errs, "%s.source.url is empty", where)
				}
			default:
				contractAdd(errs, "%s.source.type = %q is invalid", where, block.Source.Type)
			}
		case "thinking":
			if strings.TrimSpace(block.Thinking) == "" {
				contractAdd(errs, "%s.thinking is empty", where)
			}
			if !isDeepSeekModel(model) && block.Signature == "" {
				contractAdd(errs, "%s.thinking has no signature for model %q", where, model)
			}
		case "tool_use":
			if block.ID == "" || !contractToolIDPattern.MatchString(block.ID) {
				contractAdd(errs, "%s.id = %q is invalid", where, block.ID)
			}
			if block.Name == "" {
				contractAdd(errs, "%s.name is empty", where)
			}
			if _, ok := contractObject(block.Input); !ok {
				contractAdd(errs, "%s.input is not an object", where)
			}
		case "tool_result":
			if block.ToolUseID == "" || !contractToolIDPattern.MatchString(block.ToolUseID) {
				contractAdd(errs, "%s.tool_use_id = %q is invalid", where, block.ToolUseID)
			}
			if block.Content == nil {
				contractAdd(errs, "%s.content is nil", where)
			}
		default:
			contractAdd(errs, "%s.type = %q is invalid", where, block.Type)
		}
	}
}

func validateOpenAIChatRequest(body []byte) []string {
	var req OpenAIChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return []string{fmt.Sprintf("unmarshal OpenAI request: %v", err)}
	}

	var errs []string
	if req.Model == "" {
		contractAdd(&errs, "model is empty")
	}
	if req.MaxTokens != nil && req.MaxCompletionTokens != nil {
		contractAdd(&errs, "max_tokens and max_completion_tokens are both set")
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 2) {
		contractAdd(&errs, "temperature = %v, want [0,2]", *req.Temperature)
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		contractAdd(&errs, "top_p = %v, want [0,1]", *req.TopP)
	}
	if len(req.Messages) == 0 {
		contractAdd(&errs, "messages is empty")
	}

	for i, msg := range req.Messages {
		prefix := fmt.Sprintf("messages[%d]", i)
		switch msg.Role {
		case "system", "user", "assistant", "tool", "function", "developer":
		default:
			contractAdd(&errs, "%s.role = %q is invalid", prefix, msg.Role)
		}
		if msg.Content == nil && len(msg.ToolCalls) == 0 && msg.FunctionCall == nil {
			contractAdd(&errs, "%s.content is nil", prefix)
		}
		for j, call := range msg.ToolCalls {
			if call.ID == "" {
				contractAdd(&errs, "%s.tool_calls[%d].id is empty", prefix, j)
			}
			if call.Function.Name == "" {
				contractAdd(&errs, "%s.tool_calls[%d].function.name is empty", prefix, j)
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil || args == nil {
				contractAdd(&errs, "%s.tool_calls[%d].function.arguments is not a JSON object: %q", prefix, j, call.Function.Arguments)
			}
		}
		if msg.Role == "tool" && msg.ToolCallID == "" {
			contractAdd(&errs, "%s.tool_call_id is empty", prefix)
		}
	}

	for i, tool := range req.Tools {
		if tool.Type != "function" {
			contractAdd(&errs, "tools[%d].type = %q, want function", i, tool.Type)
		}
		if tool.Function.Name == "" {
			contractAdd(&errs, "tools[%d].function.name is empty", i)
		}
		if _, ok := contractObject(tool.Function.Parameters); !ok {
			contractAdd(&errs, "tools[%d].function.parameters is not an object", i)
		}
	}
	if req.ToolChoice != nil {
		if len(req.Tools) == 0 {
			contractAdd(&errs, "tool_choice is set without tools")
		}
		switch v := req.ToolChoice.(type) {
		case string:
			if v != "auto" && v != "none" && v != "required" {
				contractAdd(&errs, "tool_choice = %q is invalid", v)
			}
		case map[string]any:
			if v["type"] != "function" {
				contractAdd(&errs, "tool_choice.type = %v, want function", v["type"])
			}
			fn, _ := v["function"].(map[string]any)
			if name, _ := fn["name"].(string); name == "" {
				contractAdd(&errs, "tool_choice.function.name is empty")
			}
		default:
			contractAdd(&errs, "tool_choice has type %T", req.ToolChoice)
		}
	}
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_schema":
			if req.ResponseFormat.JSONSchema == nil {
				contractAdd(&errs, "response_format.json_schema is nil")
			} else {
				if req.ResponseFormat.JSONSchema.Name == "" {
					contractAdd(&errs, "response_format.json_schema.name is empty")
				}
				if _, ok := contractObject(req.ResponseFormat.JSONSchema.Schema); !ok {
					contractAdd(&errs, "response_format.json_schema.schema is not an object")
				}
			}
		case "json_object":
		default:
			contractAdd(&errs, "response_format.type = %q is invalid", req.ResponseFormat.Type)
		}
	}

	return errs
}

func validateAnthropicResponse(body []byte) []string {
	var resp AnthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return []string{fmt.Sprintf("unmarshal Anthropic response: %v", err)}
	}

	var errs []string
	if resp.Type != "message" {
		contractAdd(&errs, "type = %q, want message", resp.Type)
	}
	if resp.Role != "assistant" {
		contractAdd(&errs, "role = %q, want assistant", resp.Role)
	}
	if resp.Model == "" {
		contractAdd(&errs, "model is empty")
	}
	if len(resp.Content) == 0 {
		contractAdd(&errs, "content is empty")
	}
	hasTool := false
	for i, block := range resp.Content {
		where := fmt.Sprintf("content[%d]", i)
		switch block.Type {
		case "text":
			if strings.TrimSpace(block.Text) == "" && block.Text != " " {
				contractAdd(&errs, "%s.text is empty", where)
			}
		case "thinking":
			if strings.TrimSpace(block.Thinking) == "" {
				contractAdd(&errs, "%s.thinking is empty", where)
			}
			if !isDeepSeekModel(resp.Model) && block.Signature == "" {
				contractAdd(&errs, "%s.thinking has no signature", where)
			}
		case "tool_use":
			hasTool = true
			if block.ID == "" || !contractToolIDPattern.MatchString(block.ID) {
				contractAdd(&errs, "%s.id = %q is invalid", where, block.ID)
			}
			if block.Name == "" {
				contractAdd(&errs, "%s.name is empty", where)
			}
			if _, ok := contractObject(block.Input); !ok {
				contractAdd(&errs, "%s.input is not an object", where)
			}
		default:
			contractAdd(&errs, "%s.type = %q is invalid", where, block.Type)
		}
	}
	if resp.StopReason == nil {
		contractAdd(&errs, "stop_reason is nil")
	} else {
		switch *resp.StopReason {
		case "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal":
		default:
			contractAdd(&errs, "stop_reason = %q is invalid", *resp.StopReason)
		}
		if hasTool && *resp.StopReason != "tool_use" {
			contractAdd(&errs, "stop_reason = %q, want tool_use for tool content", *resp.StopReason)
		}
		if !hasTool && *resp.StopReason == "tool_use" {
			contractAdd(&errs, "stop_reason = tool_use without tool content")
		}
	}
	return errs
}

func validateOpenAIChatResponse(body []byte) []string {
	var resp OpenAIChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return []string{fmt.Sprintf("unmarshal OpenAI response: %v", err)}
	}

	var errs []string
	if resp.ID == "" {
		contractAdd(&errs, "id is empty")
	}
	if resp.Model == "" {
		contractAdd(&errs, "model is empty")
	}
	if len(resp.Choices) == 0 {
		contractAdd(&errs, "choices is empty")
	}
	for i, choice := range resp.Choices {
		where := fmt.Sprintf("choices[%d]", i)
		if choice.Message.Role != "assistant" {
			contractAdd(&errs, "%s.message.role = %q, want assistant", where, choice.Message.Role)
		}
		hasTool := len(choice.Message.ToolCalls) > 0
		for j, call := range choice.Message.ToolCalls {
			if call.ID == "" {
				contractAdd(&errs, "%s.message.tool_calls[%d].id is empty", where, j)
			}
			if call.Function.Name == "" {
				contractAdd(&errs, "%s.message.tool_calls[%d].function.name is empty", where, j)
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil || args == nil {
				contractAdd(&errs, "%s.message.tool_calls[%d].function.arguments is not a JSON object", where, j)
			}
		}
		if choice.FinishReason == nil {
			contractAdd(&errs, "%s.finish_reason is nil", where)
		} else {
			switch *choice.FinishReason {
			case "stop", "length", "tool_calls", "content_filter", "function_call":
			default:
				contractAdd(&errs, "%s.finish_reason = %q is invalid", where, *choice.FinishReason)
			}
			if hasTool && *choice.FinishReason == "stop" {
				contractAdd(&errs, "%s.finish_reason = stop with tool_calls, want tool_calls", where)
			}
		}
		text := extractTextContent(choice.Message.Content)
		if text == "" && !hasTool {
			contractAdd(&errs, "%s.message.content is empty", where)
		}
		if text != "" && strings.TrimSpace(text) == "" && text != " " {
			contractAdd(&errs, "%s.message.content is whitespace-only", where)
		}
	}
	if resp.Usage.PromptTokens > 0 || resp.Usage.CompletionTokens > 0 {
		if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
			contractAdd(&errs, "usage.total_tokens = %d, want %d", resp.Usage.TotalTokens,
				resp.Usage.PromptTokens+resp.Usage.CompletionTokens)
		}
	}
	return errs
}

// ---------------------------------------------------------------------------
// SSE contract checks
// ---------------------------------------------------------------------------

func contractSSEBlocks(data []byte) [][]byte {
	var blocks [][]byte
	for _, raw := range strings.Split(string(data), "\n\n") {
		if strings.TrimSpace(raw) == "" {
			continue
		}
		blocks = append(blocks, []byte(raw))
	}
	return blocks
}

func validateAnthropicSSE(data []byte) []string {
	var errs []string
	started := false
	terminal := false
	openBlock := -1
	blockType := map[int]string{}
	blockArgs := map[int]string{}
	hasTool := false

	for i, raw := range contractSSEBlocks(data) {
		where := fmt.Sprintf("event[%d]", i)
		evt := parseSSEEvent(raw)
		var payload map[string]any
		if err := json.Unmarshal([]byte(evt.Data), &payload); err != nil {
			contractAdd(&errs, "%s: invalid JSON: %v", where, err)
			continue
		}
		typ, _ := payload["type"].(string)
		if evt.Event != "" && evt.Event != typ {
			contractAdd(&errs, "%s: event = %q, data.type = %q", where, evt.Event, typ)
		}
		if terminal {
			contractAdd(&errs, "%s: event after terminal event", where)
		}
		switch typ {
		case "message_start":
			if started {
				contractAdd(&errs, "%s: duplicate message_start", where)
			}
			started = true
			if msg, ok := payload["message"].(map[string]any); ok {
				if model, _ := msg["model"].(string); model == "" {
					contractAdd(&errs, "%s: message.model is empty", where)
				}
			}
		case "ping":
		case "content_block_start":
			if !started {
				contractAdd(&errs, "%s: content_block_start before message_start", where)
			}
			if openBlock >= 0 {
				contractAdd(&errs, "%s: content_block_start while block %d is open", where, openBlock)
			}
			idx := int(number(payload["index"]))
			block, _ := payload["content_block"].(map[string]any)
			openBlock = idx
			blockType[idx], _ = block["type"].(string)
			if blockType[idx] == "tool_use" {
				hasTool = true
				id, _ := block["id"].(string)
				if id == "" || !contractToolIDPattern.MatchString(id) {
					contractAdd(&errs, "%s: tool_use id = %q is invalid", where, id)
				}
				if name, _ := block["name"].(string); name == "" {
					contractAdd(&errs, "%s: tool_use name is empty", where)
				}
				if input, ok := block["input"].(map[string]any); ok && len(input) > 0 {
					blockArgs[idx] = string(mustJSON(input))
				}
			}
		case "content_block_delta":
			idx := int(number(payload["index"]))
			if openBlock != idx {
				contractAdd(&errs, "%s: delta index = %d, open block = %d", where, idx, openBlock)
			}
			delta, _ := payload["delta"].(map[string]any)
			dtyp, _ := delta["type"].(string)
			switch dtyp {
			case "text_delta":
				if text, _ := delta["text"].(string); text == "" {
					contractAdd(&errs, "%s: empty text_delta", where)
				}
				if blockType[idx] != "text" {
					contractAdd(&errs, "%s: text_delta on %q block", where, blockType[idx])
				}
			case "thinking_delta":
				if text, _ := delta["thinking"].(string); text == "" {
					contractAdd(&errs, "%s: empty thinking_delta", where)
				}
				if blockType[idx] != "thinking" {
					contractAdd(&errs, "%s: thinking_delta on %q block", where, blockType[idx])
				}
			case "input_json_delta":
				if blockType[idx] != "tool_use" {
					contractAdd(&errs, "%s: input_json_delta on %q block", where, blockType[idx])
				}
				blockArgs[idx] += stringField(delta, "partial_json")
			default:
				contractAdd(&errs, "%s: delta.type = %q is invalid", where, dtyp)
			}
		case "content_block_stop":
			idx := int(number(payload["index"]))
			if idx != openBlock {
				contractAdd(&errs, "%s: stop index = %d, open block = %d", where, idx, openBlock)
			}
			if blockType[idx] == "tool_use" && blockArgs[idx] != "" && !toolArgsComplete(blockArgs[idx]) {
				contractAdd(&errs, "%s: incomplete tool arguments %q", where, blockArgs[idx])
			}
			openBlock = -1
		case "message_delta":
			if openBlock >= 0 {
				contractAdd(&errs, "%s: message_delta while block %d is open", where, openBlock)
			}
			delta, _ := payload["delta"].(map[string]any)
			stop, _ := delta["stop_reason"].(string)
			switch stop {
			case "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal":
			default:
				contractAdd(&errs, "%s: stop_reason = %q is invalid", where, stop)
			}
			if hasTool && stop == "end_turn" {
				contractAdd(&errs, "%s: stop_reason = end_turn with tool_use", where)
			}
			if !hasTool && stop == "tool_use" {
				contractAdd(&errs, "%s: stop_reason = tool_use without tool_use", where)
			}
		case "message_stop":
			if openBlock >= 0 {
				contractAdd(&errs, "%s: message_stop while block %d is open", where, openBlock)
			}
			terminal = true
		case "error":
			terminal = true
		default:
			contractAdd(&errs, "%s: type = %q is invalid", where, typ)
		}
	}
	if !terminal {
		contractAdd(&errs, "stream has no terminal event")
	}
	if !started {
		contractAdd(&errs, "stream has no message_start")
	}
	return errs
}

func validateOpenAISSE(data []byte) []string {
	var errs []string
	terminal := false
	toolArgs := map[int]string{}
	hasTool := false

	for i, raw := range contractSSEBlocks(data) {
		where := fmt.Sprintf("event[%d]", i)
		evt := parseSSEEvent(raw)
		if terminal {
			contractAdd(&errs, "%s: event after terminal event", where)
		}
		if strings.TrimSpace(evt.Data) == "[DONE]" {
			terminal = true
			for idx, args := range toolArgs {
				if args != "" && !toolArgsComplete(args) {
					contractAdd(&errs, "%s: tool_calls[%d] has incomplete arguments %q", where, idx, args)
				}
			}
			continue
		}
		var chunk OpenAIStreamChunk
		if err := json.Unmarshal([]byte(evt.Data), &chunk); err != nil {
			contractAdd(&errs, "%s: invalid JSON: %v", where, err)
			continue
		}
		if chunk.Error != nil {
			terminal = true
			continue
		}
		if len(chunk.Choices) == 0 {
			if chunk.Usage != nil && chunk.Usage.TotalTokens != chunk.Usage.PromptTokens+chunk.Usage.CompletionTokens {
				contractAdd(&errs, "%s: usage.total_tokens = %d, want %d", where,
					chunk.Usage.TotalTokens, chunk.Usage.PromptTokens+chunk.Usage.CompletionTokens)
			}
			continue
		}
		for _, choice := range chunk.Choices {
			for _, call := range choice.Delta.ToolCalls {
				hasTool = true
				if call.Index == nil {
					contractAdd(&errs, "%s: tool_call.index is nil", where)
					continue
				}
				idx := *call.Index
				toolArgs[idx] += call.Function.Arguments
			}
			if choice.FinishReason != nil {
				switch *choice.FinishReason {
				case "stop", "length", "tool_calls", "content_filter", "function_call":
				default:
					contractAdd(&errs, "%s: finish_reason = %q is invalid", where, *choice.FinishReason)
				}
				if hasTool && *choice.FinishReason == "stop" {
					contractAdd(&errs, "%s: finish_reason = stop with tool_calls", where)
				}
			}
		}
	}
	if !terminal {
		contractAdd(&errs, "stream has no [DONE] or error terminal")
	}
	return errs
}

func number(v any) float64 {
	f, _ := v.(float64)
	return f
}

func stringField(m map[string]any, key string) string {
	s, _ := m[key].(string)
	return s
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

// ---------------------------------------------------------------------------
// Positive contract tests over the P1 scenarios
// ---------------------------------------------------------------------------

func assertNoContractErrs(t *testing.T, kind string, errs []string) {
	t.Helper()
	if len(errs) == 0 {
		return
	}
	t.Errorf("%s contract errors:\n- %s", kind, strings.Join(errs, "\n- "))
}

func TestContractAnthropicRequestOutputs(t *testing.T) {
	deepseek := &ConvertOptions{Model: "claude", MaxTokens: 1024, ResolvedModel: "deepseek-chat"}
	claude := &ConvertOptions{Model: "claude", MaxTokens: 1024, ResolvedModel: "claude-sonnet"}

	cases := []struct {
		name string
		body string
		opts *ConvertOptions
	}{
		{
			name: "tool-only assistant",
			body: `{"model":"gpt-4","messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"a","content":"ok"},
				{"role":"user","content":"done"}],
				"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`,
			opts: claude,
		},
		{
			name: "parallel tool results",
			body: `{"model":"gpt-4","messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":null,"tool_calls":[
					{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},
					{"id":"b","type":"function","function":{"name":"g","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"a","content":"ra"},
				{"role":"tool","tool_call_id":"b","content":"rb"},
				{"role":"user","content":"done"}],
				"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}},
					{"type":"function","function":{"name":"g","parameters":{"type":"object"}}}]}`,
			opts: claude,
		},
		{
			name: "legacy function_call",
			body: `{"model":"gpt-4","messages":[
				{"role":"assistant","content":null,"function_call":{"name":"get_weather","arguments":"{\"city\":\"x\"}"}},
				{"role":"user","content":"go on"}]}`,
			opts: claude,
		},
		{
			name: "assistant-first history",
			body: `{"model":"gpt-4","messages":[
				{"role":"assistant","content":"hello"},
				{"role":"user","content":"go on"}]}`,
			opts: claude,
		},
		{
			name: "metadata whitelist",
			body: `{"model":"gpt-4","user":"u2","metadata":{"user_id":"u1","trace":"x"},
				"messages":[{"role":"user","content":"hi"}]}`,
			opts: claude,
		},
		{
			name: "url image",
			body: `{"model":"gpt-4","messages":[{"role":"user","content":[
				{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}`,
			opts: claude,
		},
		{
			name: "structured output",
			body: `{"model":"gpt-4","response_format":{"type":"json_schema","json_schema":{
				"name":"out","schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
				"messages":[{"role":"user","content":"hi"}]}`,
			opts: claude,
		},
		{
			name: "unsigned thinking dropped for Claude",
			body: `{"model":"gpt-4","messages":[
				{"role":"assistant","content":"hi","reasoning_content":"chain",
				 "tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"a","content":"ok"}],
				"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`,
			opts: claude,
		},
		{
			name: "unsigned thinking kept for DeepSeek",
			body: `{"model":"gpt-4","messages":[
				{"role":"assistant","content":"hi","reasoning_content":"chain",
				 "tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},
				{"role":"tool","tool_call_id":"a","content":"ok"}],
				"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`,
			opts: deepseek,
		},
		{
			name: "history tools without declarations",
			body: `{"model":"gpt-4","messages":[
				{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
				{"role":"user","content":"go on"}]}`,
			opts: claude,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := convertOpenAIRequestToAnthropic([]byte(tc.body), tc.opts)
			if err != nil {
				t.Fatalf("convertOpenAIRequestToAnthropic() error = %v", err)
			}
			assertNoContractErrs(t, "Anthropic request", validateAnthropicRequest(out))
		})
	}
}

func TestContractOpenAIChatRequestOutputs(t *testing.T) {
	cases := []struct {
		name string
		body string
		opts *ConvertOptions
	}{
		{
			name: "text",
			body: `{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`,
			opts: &ConvertOptions{ResolvedModel: "gpt-4o"},
		},
		{
			name: "thinking and tool result for GPT",
			body: `{"model":"claude","max_tokens":10,"messages":[
				{"role":"assistant","content":[{"type":"thinking","thinking":"chain"},{"type":"tool_use","id":"a","name":"f","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`,
			opts: &ConvertOptions{ResolvedModel: "gpt-4o"},
		},
		{
			name: "thinking and tool result for DeepSeek",
			body: `{"model":"claude","max_tokens":10,"messages":[
				{"role":"assistant","content":[{"type":"thinking","thinking":"chain"},{"type":"tool_use","id":"a","name":"f","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}]}`,
			opts: &ConvertOptions{ResolvedModel: "deepseek-chat", ReasoningCache: NewReasoningCache(10)},
		},
		{
			name: "url image",
			body: `{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":[
				{"type":"text","text":"see"},{"type":"image","source":{"type":"url","url":"https://example.com/cat.png"}}]}]}`,
			opts: &ConvertOptions{ResolvedModel: "gpt-4o"},
		},
		{
			name: "structured output",
			body: `{"model":"claude","max_tokens":10,"output_config":{"format":{
				"type":"json_schema","schema":{"type":"object","properties":{"a":{"type":"string"}}}}},
				"messages":[{"role":"user","content":"hi"}]}`,
			opts: &ConvertOptions{ResolvedModel: "gpt-4o"},
		},
		{
			name: "reasoning model parameters",
			body: `{"model":"claude","max_tokens":100,"temperature":0.5,"top_p":0.9,
				"messages":[{"role":"user","content":"hi"}]}`,
			opts: &ConvertOptions{ResolvedModel: "o3-mini"},
		},
		{
			name: "effort with thinking",
			body: `{"model":"claude","max_tokens":10,"thinking":{"type":"enabled","budget_tokens":1024},
				"output_config":{"effort":"high"},"messages":[{"role":"user","content":"hi"}]}`,
			opts: &ConvertOptions{ResolvedModel: "deepseek-chat"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, err := convertAnthropicRequestToOpenAI([]byte(tc.body), tc.opts)
			if err != nil {
				t.Fatalf("convertAnthropicRequestToOpenAI() error = %v", err)
			}
			assertNoContractErrs(t, "OpenAI request", validateOpenAIChatRequest(out))
		})
	}
}

func TestContractResponseOutputs(t *testing.T) {
	t.Run("Anthropic text", func(t *testing.T) {
		body := `{"id":"chatcmpl-1","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop",
			"message":{"role":"assistant","content":"hi"}}]}`
		out, _ := convertOpenAIResponseToAnthropic([]byte(body), nil)
		assertNoContractErrs(t, "Anthropic response", validateAnthropicResponse(out))
	})
	t.Run("Anthropic tool stop", func(t *testing.T) {
		body := `{"id":"chatcmpl-1","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop",
			"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`
		out, _ := convertOpenAIResponseToAnthropic([]byte(body), nil)
		assertNoContractErrs(t, "Anthropic response", validateAnthropicResponse(out))
	})
	t.Run("OpenAI text", func(t *testing.T) {
		body := `{"id":"msg_1","model":"claude","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hi"}]}`
		out, _ := convertAnthropicResponseToOpenAI([]byte(body))
		assertNoContractErrs(t, "OpenAI response", validateOpenAIChatResponse(out))
	})
	t.Run("OpenAI tool stop", func(t *testing.T) {
		body := `{"id":"msg_1","model":"claude","stop_reason":"end_turn",
			"content":[{"type":"tool_use","id":"a","name":"f","input":{}}]}`
		out, _ := convertAnthropicResponseToOpenAI([]byte(body))
		assertNoContractErrs(t, "OpenAI response", validateOpenAIChatResponse(out))
	})
}

func TestContractStreamOutputs(t *testing.T) {
	t.Run("OpenAI to Anthropic text and tool", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		var parts [][]byte
		parts = append(parts, sc.HandleStreamStart())
		for _, chunk := range []string{
			`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"stop"}]}`,
		} {
			out, err := sc.HandleChunk([]byte(chunk))
			if err != nil {
				t.Fatalf("HandleChunk() error = %v", err)
			}
			if len(out) > 0 {
				parts = append(parts, out)
			}
		}
		parts = append(parts, sc.HandleStreamEnd())
		assertNoContractErrs(t, "Anthropic SSE", validateAnthropicSSE(bytes.Join(parts, []byte("\n\n"))))
	})

	t.Run("OpenAI to Anthropic error chunk", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		var parts [][]byte
		parts = append(parts, sc.HandleStreamStart())
		out, err := sc.HandleChunk([]byte(`{"error":{"type":"server_error","message":"boom"}}`))
		if err != nil {
			t.Fatalf("HandleChunk() error = %v", err)
		}
		parts = append(parts, out)
		parts = append(parts, sc.HandleStreamEnd())
		assertNoContractErrs(t, "Anthropic SSE", validateAnthropicSSE(bytes.Join(parts, []byte("\n\n"))))
	})

	t.Run("OpenAI to Anthropic truncated tool", func(t *testing.T) {
		sc := NewStreamConverter("claude", nil, nil)
		var parts [][]byte
		parts = append(parts, sc.HandleStreamStart())
		out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f","arguments":"{\"a\":"}}]},"finish_reason":null}]}`))
		if err != nil {
			t.Fatalf("HandleChunk() error = %v", err)
		}
		parts = append(parts, out)
		parts = append(parts, sc.HandleStreamEnd())
		assertNoContractErrs(t, "Anthropic SSE", validateAnthropicSSE(bytes.Join(parts, []byte("\n\n"))))
	})

	t.Run("Anthropic to OpenAI text and tool", func(t *testing.T) {
		c := NewAnthropicToOpenAIStreamConverter("claude")
		var parts [][]byte
		for _, chunk := range []string{
			`{"type":"message_start","message":{"id":"msg_1","model":"claude"}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hi"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"a","name":"f","input":{}}}`,
			`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			`{"type":"message_stop"}`,
		} {
			out, err := c.HandleChunk([]byte(chunk))
			if err != nil {
				t.Fatalf("HandleChunk() error = %v", err)
			}
			if len(out) > 0 {
				parts = append(parts, out)
			}
		}
		parts = append(parts, c.HandleStreamEnd())
		assertNoContractErrs(t, "OpenAI SSE", validateOpenAISSE(bytes.Join(parts, []byte("\n\n"))))
	})

	t.Run("Anthropic to OpenAI error event", func(t *testing.T) {
		c := NewAnthropicToOpenAIStreamConverter("claude")
		out, err := c.HandleChunk([]byte(`{"type":"error","error":{"type":"overloaded_error","message":"boom"}}`))
		if err != nil {
			t.Fatalf("HandleChunk() error = %v", err)
		}
		assertNoContractErrs(t, "OpenAI SSE", validateOpenAISSE(out))
	})

	t.Run("Anthropic to OpenAI truncated tool", func(t *testing.T) {
		c := NewAnthropicToOpenAIStreamConverter("claude")
		var parts [][]byte
		for _, chunk := range []string{
			`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"a","name":"f","input":{}}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}`,
		} {
			out, err := c.HandleChunk([]byte(chunk))
			if err != nil {
				t.Fatalf("HandleChunk() error = %v", err)
			}
			if len(out) > 0 {
				parts = append(parts, out)
			}
		}
		parts = append(parts, c.HandleStreamEnd())
		assertNoContractErrs(t, "OpenAI SSE", validateOpenAISSE(bytes.Join(parts, []byte("\n\n"))))
	})
}

func TestContractValidatorsRejectKnownInvalid(t *testing.T) {
	t.Run("Anthropic empty text", func(t *testing.T) {
		if errs := validateAnthropicRequest([]byte(`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":""}]}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted empty text block")
		}
	})
	t.Run("Anthropic tool_choice without tools", func(t *testing.T) {
		if errs := validateAnthropicRequest([]byte(`{"model":"claude","max_tokens":10,"tool_choice":{"type":"auto"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted tool_choice without tools")
		}
	})
	t.Run("Anthropic orphan tool_result", func(t *testing.T) {
		if errs := validateAnthropicRequest([]byte(`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"x"}]}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted orphan tool_result")
		}
	})
	t.Run("Anthropic missing tool_result", func(t *testing.T) {
		if errs := validateAnthropicRequest([]byte(`{"model":"claude","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]},{"role":"user","content":[{"type":"text","text":"go"}]}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted missing tool_result")
		}
	})
	t.Run("Anthropic metadata extra key", func(t *testing.T) {
		if errs := validateAnthropicRequest([]byte(`{"model":"claude","max_tokens":10,"metadata":{"trace":"x"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted extra metadata key")
		}
	})
	t.Run("OpenAI both max tokens", func(t *testing.T) {
		if errs := validateOpenAIChatRequest([]byte(`{"model":"gpt-4","max_tokens":10,"max_completion_tokens":10,"messages":[{"role":"user","content":"hi"}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted both max_tokens fields")
		}
	})
	t.Run("OpenAI tool call without id", func(t *testing.T) {
		if errs := validateOpenAIChatRequest([]byte(`{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{}"}}]}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted tool call without id")
		}
	})
	t.Run("OpenAI tool_choice without function name", func(t *testing.T) {
		if errs := validateOpenAIChatRequest([]byte(`{"model":"gpt-4","tool_choice":{"type":"function"},"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"hi"}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted function tool_choice without a name")
		}
	})
	t.Run("OpenAI tool arguments not object", func(t *testing.T) {
		if errs := validateOpenAIChatRequest([]byte(`{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"[]"}}]}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted non-object tool arguments")
		}
	})
	t.Run("Anthropic response tool stop", func(t *testing.T) {
		if errs := validateAnthropicResponse([]byte(`{"type":"message","role":"assistant","model":"claude","stop_reason":"end_turn","content":[{"type":"tool_use","id":"a","name":"f","input":{}}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted end_turn with tool_use")
		}
	})
	t.Run("OpenAI response stop with tools", func(t *testing.T) {
		if errs := validateOpenAIChatResponse([]byte(`{"id":"x","model":"gpt-4","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}`)); len(errs) == 0 {
			t.Fatal("validator accepted stop with tool_calls")
		}
	})
}
