package convert

// detectSource detects the protocol from a parsed JSON body using positive
// structural markers only. No negative exclusions, no hardcoded model prefixes.
// Returns ProtocolUnknown for minimal requests that lack distinguishing features.
func detectSource(raw map[string]any) Protocol {
	// -------- 1. Response markers --------

	// OpenAI Chat Response: choices array.
	if _, ok := raw["choices"]; ok {
		return ProtocolOpenAIChat
	}

	// Anthropic Response: type:"message" + stop_reason or usage.
	if t, ok := raw["type"].(string); ok && t == "message" {
		if _, ok := raw["stop_reason"]; ok {
			return ProtocolAnthropic
		}
		if _, ok := raw["usage"]; ok {
			return ProtocolAnthropic
		}
	}

	// OpenAI Responses Response: object:"response" or output[].
	if obj, ok := raw["object"].(string); ok && obj == "response" {
		return ProtocolOpenAIResponses
	}
	if output, ok := raw["output"].([]any); ok && len(output) > 0 {
		if first, ok := output[0].(map[string]any); ok {
			if t, _ := first["type"].(string); t != "" {
				return ProtocolOpenAIResponses
			}
		}
	}
	// Empty output array still qualifies if object is present.
	if _, ok := raw["output"]; ok {
		if _, ok := raw["object"]; ok {
			return ProtocolOpenAIResponses
		}
	}

	// -------- 2. Request markers --------

	// OpenAI Responses Request: input (without messages) or instructions.
	// Checked first — these fields are unique to the Responses API (neither
	// Chat nor Anthropic use them), so they must override weaker heuristics
	// like a bare-string tool_choice, which Responses also sends.
	if _, hasInput := raw["input"]; hasInput {
		if _, hasMessages := raw["messages"]; !hasMessages {
			return ProtocolOpenAIResponses
		}
	}
	if _, ok := raw["instructions"]; ok {
		return ProtocolOpenAIResponses
	}

	// Anthropic: tools[].input_schema (flat name + input_schema, no function wrapper).
	if tools, ok := raw["tools"].([]any); ok && len(tools) > 0 {
		if firstTool, ok := tools[0].(map[string]any); ok {
			_, hasFunction := firstTool["function"]
			_, hasName := firstTool["name"]
			_, hasInputSchema := firstTool["input_schema"]
			if !hasFunction && hasName && hasInputSchema {
				return ProtocolAnthropic
			}
		}
	}

	// OpenAI: tools[].function wrapper.
	if tools, ok := raw["tools"].([]any); ok && len(tools) > 0 {
		if firstTool, ok := tools[0].(map[string]any); ok {
			if _, ok := firstTool["function"]; ok {
				return ProtocolOpenAIChat
			}
		}
	}

	// Anthropic: top-level system field.
	if _, ok := raw["system"]; ok {
		return ProtocolAnthropic
	}

	// Anthropic: tool_choice.type == "any" or "tool".
	if tc, ok := raw["tool_choice"].(map[string]any); ok {
		if t, _ := tc["type"].(string); t == "any" || t == "tool" {
			return ProtocolAnthropic
		}
	}

	// OpenAI: tool_choice.type == "function".
	if tc, ok := raw["tool_choice"].(map[string]any); ok {
		if t, _ := tc["type"].(string); t == "function" {
			return ProtocolOpenAIChat
		}
	}

	// OpenAI: tool_choice as a bare string ("auto"/"required"/"none").
	// Anthropic's tool_choice is always an object, so a string is definitive.
	if _, ok := raw["tool_choice"].(string); ok {
		return ProtocolOpenAIChat
	}

	// Anthropic: stop_sequences field.
	if _, ok := raw["stop_sequences"]; ok {
		return ProtocolAnthropic
	}

	// Check messages for role-based and content-block markers.
	if msgs, ok := raw["messages"].([]any); ok && len(msgs) > 0 {
		// Anthropic: content block types (thinking, tool_use, tool_result, image).
		for i := 0; i < len(msgs) && i < 3; i++ {
			if msg, ok := msgs[i].(map[string]any); ok {
				if content, ok := msg["content"].([]any); ok {
					for _, block := range content {
						if b, ok := block.(map[string]any); ok {
							switch b["type"] {
							case "thinking", "tool_use", "tool_result", "image":
								return ProtocolAnthropic
							}
						}
					}
				}
			}
		}

		// OpenAI: messages[].role ∈ {system, tool, function}.
		for _, msg := range msgs {
			if m, ok := msg.(map[string]any); ok {
				role, _ := m["role"].(string)
				if role == "system" || role == "tool" || role == "function" {
					return ProtocolOpenAIChat
				}
			}
		}

		// OpenAI: messages[].tool_calls.
		for _, msg := range msgs {
			if m, ok := msg.(map[string]any); ok {
				if _, ok := m["tool_calls"]; ok {
					return ProtocolOpenAIChat
				}
			}
		}

		// OpenAI: messages[].function_call (legacy format).
		for _, msg := range msgs {
			if m, ok := msg.(map[string]any); ok {
				if _, ok := m["function_call"]; ok {
					return ProtocolOpenAIChat
				}
			}
		}

		// OpenAI: messages[].reasoning_content (DeepSeek/GLM extension).
		for _, msg := range msgs {
			if m, ok := msg.(map[string]any); ok {
				if _, ok := m["reasoning_content"]; ok {
					return ProtocolOpenAIChat
				}
			}
		}

		// OpenAI: image_url content block type.
		for i := 0; i < len(msgs) && i < 3; i++ {
			if msg, ok := msgs[i].(map[string]any); ok {
				if content, ok := msg["content"].([]any); ok {
					for _, block := range content {
						if b, ok := block.(map[string]any); ok {
							if b["type"] == "image_url" {
								return ProtocolOpenAIChat
							}
						}
					}
				}
			}
		}
	}

	// Anthropic: output_config.
	if _, ok := raw["output_config"]; ok {
		return ProtocolAnthropic
	}

	// OpenAI: stream_options.
	if _, ok := raw["stream_options"]; ok {
		return ProtocolOpenAIChat
	}

	// OpenAI Request: standard OpenAI-only fields.
	for _, k := range []string{"frequency_penalty", "presence_penalty", "logit_bias", "response_format", "n", "seed"} {
		if _, ok := raw[k]; ok {
			return ProtocolOpenAIChat
		}
	}

	// OpenAI: stop field (Anthropic uses stop_sequences).
	if _, ok := raw["stop"]; ok {
		return ProtocolOpenAIChat
	}

	// Minimal request with no distinguishing features → unknown.
	return ProtocolUnknown
}
