package convert

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

const placeholderReasoning = "Compatibility bridge placeholder reasoning for prior assistant history."

// currentToolContextParts extracts the tool call/result context from the most
// recent tool-calling turn in the message list. Returns the signature parts
// for the latest turn (empty if no tool call turn found).
func currentToolContextParts(messages []AnthropicMessage) []string {
	var hadToolCall bool
	var parts []string

	for _, msg := range messages {
		var text string
		var toolResults []AnthropicContent
		var toolUses []AnthropicContent

		for _, block := range msg.Content {
			switch block.Type {
			case "text":
				text += block.Text
			case "tool_result":
				toolResults = append(toolResults, block)
			case "tool_use":
				toolUses = append(toolUses, block)
			}
		}

		if msg.Role == "user" {
			if len(toolResults) > 0 {
				for _, tr := range toolResults {
					if hadToolCall {
						parts = append(parts, toolResultSignature(tr.ToolUseID, tr.Content))
					}
				}
			} else if text != "" {
				hadToolCall = false
				parts = nil
			}
			if len(toolResults) == 0 && text != "" {
				hadToolCall = false
				parts = nil
			}
			continue
		}

		if msg.Role == "assistant" && len(toolUses) > 0 {
			hadToolCall = true
			parts = nil
			for _, tu := range toolUses {
				parts = append(parts, toolUseSignature(tu.ID, tu.Name, tu.Input))
			}
		}
	}

	if hadToolCall {
		return parts
	}
	return nil
}

// anthropicToolChoiceToOpenAi maps Anthropic tool_choice to OpenAI tool_choice.
// For DeepSeek models, forced tool_choice (any/tool) is stripped and handled
// via a system instruction instead.
func anthropicToolChoiceToOpenAi(choice *AnthropicToolChoice, model string) any {
	if choice == nil {
		return nil
	}
	switch choice.Type {
	case "auto":
		return "auto"
	case "none":
		return "none"
	case "any":
		if isDeepSeekModel(model) {
			return nil // softened to system instruction
		}
		return "required"
	case "tool":
		if isDeepSeekModel(model) {
			return nil // softened to system instruction
		}
		return map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": choice.Name,
			},
		}
	}
	return nil
}

// toolChoiceInstruction returns a system instruction for forced tool_choice
// on DeepSeek models (which reject forced function tool_choice).
func toolChoiceInstruction(choice *AnthropicToolChoice, model string) string {
	if choice == nil || !isDeepSeekModel(model) {
		return ""
	}
	switch choice.Type {
	case "any":
		return "The caller requires a tool call for this turn. Call one of the available tools instead of answering directly."
	case "tool":
		if choice.Name != "" {
			return fmt.Sprintf("The caller requires a tool call for this turn. Call the available tool named %s instead of answering directly.", choice.Name)
		}
	}
	return ""
}

// thinkingToOpenAi converts Anthropic thinking config to OpenAI format.
func thinkingToOpenAi(t *AnthropicThinking) any {
	if t == nil {
		return nil
	}
	return map[string]any{"type": t.Type}
}

// thinkingToOpenAiGLM maps Anthropic thinking to GLM's structured thinking field.
// GLM expects {"type":"enabled","budget_tokens":N} with budget_tokens forwarded
// directly from the Anthropic thinking config (not mapped through reasoning_effort).
func thinkingToOpenAiGLM(t *AnthropicThinking) any {
	if t == nil {
		return nil
	}
	result := map[string]any{"type": t.Type}
	if t.Type == "enabled" && t.BudgetTokens > 0 {
		result["budget_tokens"] = t.BudgetTokens
	}
	return result
}

// reasoningEffortToOpenAi maps Anthropic output_config.effort to OpenAI reasoning_effort.
func reasoningEffortToOpenAi(cfg *AnthropicOutputConfig) any {
	if cfg == nil || cfg.Effort == "" {
		return nil
	}
	switch strings.ToLower(cfg.Effort) {
	case "max", "xhigh":
		return "max"
	case "high", "medium", "low":
		return "high"
	}
	return nil
}

// ---------------------------------------------------------------------------
// Anthropic Request → OpenAI Request
// ---------------------------------------------------------------------------

func convertAnthropicRequestToOpenAI(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal Anthropic request", "err", err)
		return body, nil
	}

	// Resolve output model: mapping → passthrough → fallback.
	outputModel, _ := resolveModel(req.Model, opts.Model, opts.ModelMap)
	profile := classifyModel(outputModel)

	oai := OpenAIChatRequest{
		Model:  outputModel,
		Stream: req.Stream,
	}

	if req.MaxTokens > 0 {
		oai.MaxTokens = &req.MaxTokens
	}
	if req.Temperature != nil {
		oai.Temperature = req.Temperature
	}
	if req.TopP != nil {
		oai.TopP = req.TopP
	}
	if req.Metadata != nil {
		oai.Metadata = req.Metadata
	}

	// Stop sequences → OpenAI stop.
	if len(req.StopSequences) > 0 {
		if len(req.StopSequences) == 1 {
			oai.Stop = req.StopSequences[0]
		} else {
			anys := make([]any, len(req.StopSequences))
			for i, s := range req.StopSequences {
				anys[i] = s
			}
			oai.Stop = anys
		}
	}

	// Thinking config → model-specific OpenAI thinking field.
	if profile.isDeepSeek {
		oai.Thinking = thinkingToOpenAi(req.Thinking)
		oai.ReasoningEffort = reasoningEffortToOpenAi(req.OutputConfig)
	} else if isGLMModel(outputModel) {
		oai.Thinking = thinkingToOpenAiGLM(req.Thinking)
	}

	// Tool choice instruction for DeepSeek (softened to system instruction).
	extraInstruction := toolChoiceInstruction(req.ToolChoice, outputModel)

	// Top-level system → prepend as system message.
	if len(req.System) > 0 {
		var sysText string
		for _, block := range req.System {
			sysText += block.Text
		}
		if extraInstruction != "" {
			sysText = sysText + "\n\n" + extraInstruction
		}
		oai.Messages = append(oai.Messages, OpenAIMessage{
			Role:    "system",
			Content: sysText,
		})
	} else if extraInstruction != "" {
		oai.Messages = append(oai.Messages, OpenAIMessage{
			Role:    "system",
			Content: extraInstruction,
		})
	}

	// Derive tool context parts for reasoning cache lookups.
	ctxParts := currentToolContextParts(req.Messages)

	// Messages.
	for _, msg := range req.Messages {
		oai.Messages = append(oai.Messages, convertAnthropicMessage(msg, opts, ctxParts)...)
	}

	// Message sequence sanitization: coalesce adjacent tool calls, pair
	// tool_calls with tool_results, handle orphans.
	oai.Messages = sanitizeOpenAiToolMessageSequence(coalesceAdjacentAssistantToolCalls(oai.Messages))

	// GLM API only accepts string content — flatten multi-part content arrays.
	if isGLMModel(outputModel) {
		flattenContentForGLM(oai.Messages)
	}

	// Edge case: if no messages after conversion, inject minimal user message.
	if len(oai.Messages) == 0 {
		oai.Messages = []OpenAIMessage{
			{Role: "user", Content: "..."},
		}
	}

	// Tools.
	if len(req.Tools) > 0 {
		sortAnthropicTools(req.Tools)
		oai.Tools = make([]OpenAITool, 0, len(req.Tools))
		for _, t := range req.Tools {
			oai.Tools = append(oai.Tools, OpenAITool{
				Type: "function",
				Function: OpenAIFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  ensureObjectSchema(t.InputSchema),
				},
			})
		}

		// Model-aware tool choice mapping.
		oai.ToolChoice = anthropicToolChoiceToOpenAi(req.ToolChoice, outputModel)
	}

	// stream_options for streaming requests (enables usage in final chunk).
	if req.Stream != nil && *req.Stream {
		oai.StreamOptions = map[string]any{"include_usage": true}
	}

	return json.Marshal(oai)
}

func convertAnthropicMessage(msg AnthropicMessage, opts *ConvertOptions, ctxParts []string) []OpenAIMessage {
	switch msg.Role {
	case "user":
		return convertAnthropicUserMessage(msg)
	case "assistant":
		return []OpenAIMessage{convertAnthropicAssistantMessage(msg, opts, ctxParts)}
	default:
		return []OpenAIMessage{{Role: "user", Content: extractTextContent(msg.Content)}}
	}
}

func convertAnthropicUserMessage(msg AnthropicMessage) []OpenAIMessage {
	// Check for tool_result blocks (wrapped in "user" role in Anthropic).
	var toolMsgs []OpenAIMessage
	var textParts []string
	for _, block := range msg.Content {
		if block.Type == "tool_result" {
			content := block.Content
			if content == nil {
				content = ""
			}
			toolMsgs = append(toolMsgs, OpenAIMessage{
				Role:       "tool",
				ToolCallID: block.ToolUseID,
				Content:    content,
			})
		} else if block.Type == "text" && block.Text != "" {
			textParts = append(textParts, block.Text)
		}
	}
	if len(toolMsgs) > 0 {
		if len(textParts) > 0 {
			text := strings.Join(textParts, "\n")
			toolMsgs = append(toolMsgs, OpenAIMessage{Role: "user", Content: text})
		}
		return toolMsgs
	}

	// Normal user content — build multi-part content if needed.
	var parts []map[string]any
	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				parts = append(parts, map[string]any{
					"type": "text",
					"text": block.Text,
				})
			}
		case "image":
			if block.Source != nil && block.Source.Type == "base64" {
				parts = append(parts, map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": "data:" + block.Source.MediaType + ";base64," + block.Source.Data,
					},
				})
			}
		}
	}

	if len(parts) == 1 && parts[0]["type"] == "text" {
		return []OpenAIMessage{{Role: "user", Content: parts[0]["text"].(string)}}
	}
	if len(parts) == 0 {
		return []OpenAIMessage{{Role: "user", Content: ""}}
	}
	contentArr := make([]any, len(parts))
	for i, p := range parts {
		contentArr[i] = p
	}
	return []OpenAIMessage{{Role: "user", Content: contentArr}}
}

func convertAnthropicAssistantMessage(msg AnthropicMessage, opts *ConvertOptions, ctxParts []string) OpenAIMessage {
	var textParts []string
	var reasoning string
	var toolCalls []OpenAIToolCall

	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "thinking":
			t := block.Thinking
			if t == "" {
				t = block.Text
			}
			reasoning += t
		case "tool_use":
			args := ""
			if block.Input != nil {
				if b, err := json.Marshal(block.Input); err == nil {
					args = string(b)
				}
			}
			id := block.ID
			if id == "" {
				id = ensureToolID("")
			}
			toolCalls = append(toolCalls, OpenAIToolCall{
				ID:   id,
				Type: "function",
				Function: OpenAIFunctionCall{
					Name:      block.Name,
					Arguments: args,
				},
			})
		}
	}

	oaiMsg := OpenAIMessage{Role: "assistant"}
	if reasoning != "" {
		oaiMsg.ReasoningContent = reasoning
	}
	content := strings.Join(textParts, "")

	if len(toolCalls) > 0 {
		oaiMsg.ToolCalls = toolCalls
		if content != "" {
			oaiMsg.Content = content
		} else {
			oaiMsg.Content = nil
		}
	} else {
		oaiMsg.Content = content
	}

	// If reasoning is empty but tool_calls exist (Claude Code compressed thinking),
	// try to restore reasoning_content from cache (multi-tier).
	if oaiMsg.ReasoningContent == "" && len(oaiMsg.ToolCalls) > 0 && opts != nil && opts.ReasoningCache != nil {
		ids := make([]string, len(oaiMsg.ToolCalls))
		for i, tc := range oaiMsg.ToolCalls {
			ids[i] = tc.ID
		}
		content := extractTextContent(oaiMsg.Content)
		if cached := opts.ReasoningCache.GetBest(ids, ctxParts, content); cached != "" {
			oaiMsg.ReasoningContent = cached
		} else {
			// Placeholder fallback — DeepSeek requires reasoning_content
			// when tool_calls are present, even if cache is empty.
			oaiMsg.ReasoningContent = placeholderReasoning
		}
	}

	return oaiMsg
}

// ---------------------------------------------------------------------------
// Anthropic Response → OpenAI Response
// ---------------------------------------------------------------------------

func convertAnthropicResponseToOpenAI(body []byte) ([]byte, error) {
	var resp AnthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		slog.Warn("failed to unmarshal Anthropic response", "err", err)
		return body, nil
	}

	openai := OpenAIChatResponse{
		ID:      strings.Replace(resp.ID, "msg_", "chatcmpl-", 1),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
	}

	// Build the assistant message from content blocks.
	msg := OpenAIMessage{
		Role:    "assistant",
		Content: nil,
	}

	var textParts []string
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "thinking":
			msg.ReasoningContent += block.Thinking
			if msg.ReasoningContent == "" {
				msg.ReasoningContent = block.Text
			}
		case "tool_use":
			if msg.ToolCalls == nil {
				msg.ToolCalls = []OpenAIToolCall{}
			}
			argsBytes, _ := json.Marshal(block.Input)
			msg.ToolCalls = append(msg.ToolCalls, OpenAIToolCall{
				ID:   block.ID,
				Type: "function",
				Function: OpenAIFunctionCall{
					Name:      block.Name,
					Arguments: string(argsBytes),
				},
			})
			// signature and other blocks are ignored.
		}
	}

	// Set content — if there are tool_calls, content should be null or empty string.
	if len(msg.ToolCalls) > 0 {
		if len(textParts) > 0 {
			msg.Content = strings.Join(textParts, "")
		} else {
			msg.Content = nil
		}
	} else {
		msg.Content = strings.Join(textParts, "")
	}

	openai.Choices = []OpenAIChoice{
		{
			Index:   0,
			Message: msg,
			FinishReason: mapFinishReason(resp.StopReason),
		},
	}

	// Usage.
	openai.Usage = OpenAIUsage{
		PromptTokens:     resp.Usage.InputTokens,
		CompletionTokens: resp.Usage.OutputTokens,
		TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
	}

	return json.Marshal(openai)
}

func mapFinishReason(reason *string) *string {
	if reason == nil {
		return &strStop // default to "stop" instead of nil to avoid NPE
	}
	switch *reason {
	case "end_turn":
		return &strStop
	case "max_tokens":
		return &strLength
	case "tool_use":
		return &strToolCalls
	case "stop_sequence":
		return &strStop
	case "content_filter":
		return &strStop
	default:
		return reason
	}
}

var (
	strStop      = "stop"
	strLength    = "length"
	strToolCalls = "tool_calls"
)
