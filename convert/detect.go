package convert

import "strings"
func hasStringField(m map[string]any, key string) bool {
	_, ok := m[key].(string)
	return ok
}

func hasArrayField(m map[string]any, key string) bool {
	_, ok := m[key].([]any)
	return ok
}

func isAnthropicResponse(m map[string]any) bool {
	t, ok := m["type"].(string)
	if !ok || t != "message" {
		return false
	}
	// Look for the presence of either stop_reason or usage.
	if _, ok := m["stop_reason"]; ok {
		return true
	}
	if _, ok := m["usage"]; ok {
		return true
	}
	return false
}

// isAnthropicRequest detects an Anthropic Messages API request body.
// Key signals: messages content as typed-block arrays, max_tokens (required),
// Anthropic-specific content block types (thinking, tool_use, tool_result, image).
func isAnthropicRequest(m map[string]any) bool {
	msgs, ok := m["messages"].([]any)
	if !ok || len(msgs) == 0 {
		return false
	}

	// Quick rejection: OpenAI-specific fields that Anthropic never uses.
	if _, ok := m["frequency_penalty"]; ok {
		return false
	}
	if _, ok := m["presence_penalty"]; ok {
		return false
	}
	if _, ok := m["logit_bias"]; ok {
		return false
	}
	if _, ok := m["response_format"]; ok {
		return false
	}
	if _, ok := m["n"]; ok {
		return false
	}
	if _, ok := m["seed"]; ok {
		return false
	}

	// Quick rejection: OpenAI uses "stop" (string/array), Anthropic uses "stop_sequences" (array).
	if _, ok := m["stop"]; ok {
		return false
	}

	// OpenAI tool_choice can be a string (e.g. "auto", "required"); Anthropic is always an object.
	if _, ok := m["tool_choice"].(string); ok {
		return false
	}

	// Anthropic tools have flat name + input_schema at top level, no "function" wrapper.
	if tools, ok := m["tools"].([]any); ok && len(tools) > 0 {
		if firstTool, ok := tools[0].(map[string]any); ok {
			_, hasFunction := firstTool["function"]
			_, hasName := firstTool["name"]
			_, hasInputSchema := firstTool["input_schema"]
			if !hasFunction && hasName && hasInputSchema {
				return true
			}
		}
	}

	// Check messages for OpenAI-only patterns (role: system/tool/function in array).
	for _, msg := range msgs {
		mmsg, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		role, _ := mmsg["role"].(string)
		if role == "system" || role == "tool" || role == "function" {
			return false
		}
	}

	// Check first few messages for Anthropic-specific content block types.
	for i := 0; i < len(msgs) && i < 3; i++ {
		msg, ok := msgs[i].(map[string]any)
		if !ok {
			continue
		}
		content, ok := msg["content"].([]any)
		if !ok {
			continue
		}
		for _, block := range content {
			b, ok := block.(map[string]any)
			if !ok {
				continue
			}
			bt, _ := b["type"].(string)
			switch bt {
			case "thinking", "tool_use", "tool_result", "image":
				// These types only exist in Anthropic content blocks.
				return true
			}
		}
	}

	// Top-level system field is Anthropic-specific.
	if _, ok := m["system"]; ok {
		return true
	}

	// Anthropic's tool_choice uses "any" or "tool" (not "required" or "function").
	if tc, ok := m["tool_choice"].(map[string]any); ok {
		if t, _ := tc["type"].(string); t == "function" {
			return false // OpenAI-style function-type tool_choice
		}
		if t, _ := tc["type"].(string); t == "any" || t == "tool" {
			return true
		}
	}

	// max_tokens is REQUIRED in Anthropic. If present AND the model doesn't
	// look OpenAI-ish (gpt/o1/o3/deepseek/gemini), it's likely an Anthropic
	// request. Accept both array content (current format) and string content
	// (deprecated format still accepted by the API).
	if _, ok := m["max_tokens"]; ok && !hasOpenAIStyleModel(m) {
		for i := 0; i < len(msgs) && i < 2; i++ {
			msg, ok := msgs[i].(map[string]any)
			if !ok {
				continue
			}
			switch c := msg["content"].(type) {
			case []any, string:
				return true
			case nil:
				continue
			default:
				_ = c
			}
		}
	}

	return false
}

// hasOpenAIStyleModel returns true if the model field value looks like an
// OpenAI-compatible model ID rather than Anthropic. Used to prevent false
// positives where an OpenAI request with max_tokens + array content would
// otherwise be classified as an Anthropic request.
func hasOpenAIStyleModel(m map[string]any) bool {
	model, ok := m["model"].(string)
	if !ok || model == "" {
		return false
	}
	// Check for common OpenAI / downstream model prefixes.
	openAIPrefixes := []string{
		"gpt-", "o1", "o3", "deepseek", "gemini-", "glm-",
	}
	modelLower := strings.ToLower(model)
	for _, p := range openAIPrefixes {
		if strings.HasPrefix(modelLower, p) {
			return true
		}
	}
	return false
}

// isOpenAIResponse detects an OpenAI Chat Completions response body.
// Key signals: choices array with message/finish_reason fields.
func isOpenAIResponse(m map[string]any) bool {
	choices, ok := m["choices"].([]any)
	if !ok || len(choices) == 0 {
		return false
	}
	if choice, ok := choices[0].(map[string]any); ok {
		if _, ok := choice["finish_reason"]; ok {
			return true
		}
		if _, ok := choice["message"]; ok {
			return true
		}
		if _, ok := choice["delta"]; ok {
			return true // streaming chunk
		}
	}
	return false
}

// isOpenAIRequest detects an OpenAI Chat Completions request body.
// It checks for model/messages fields while rejecting known non-request formats.
func isOpenAIRequest(m map[string]any) bool {
	if !hasStringField(m, "model") && !hasArrayField(m, "messages") {
		return false
	}
	if isAnthropicRequest(m) {
		return false
	}
	if isAnthropicResponse(m) {
		return false
	}
	if isOpenAIResponse(m) {
		return false
	}
	return true
}

// isOpenAIStreamChunk detects an OpenAI SSE streaming data payload.
// Format: {"choices":[{"index":0,"delta":{...},"finish_reason":null}]}
