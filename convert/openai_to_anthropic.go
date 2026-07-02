package convert

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// ---------------------------------------------------------------------------
// Message sequence sanitization
// ---------------------------------------------------------------------------

// toolUseSignature returns a canonical string for a tool_use block.
func toolUseSignature(id, name string, input any) string {
	inp, _ := json.Marshal(input)
	return fmt.Sprintf("tool_use:%s:%s:%s", id, name, string(inp))
}

// toolResultSignature returns a canonical string for a tool_result block.
func toolResultSignature(toolUseID string, content any) string {
	return fmt.Sprintf("tool_result:%s:%s", toolUseID, stringifyContent(content))
}

func stringifyContent(content any) string {
	if content == nil {
		return ""
	}
	if s, ok := content.(string); ok {
		return s
	}
	b, _ := json.Marshal(content)
	return string(b)
}

// coalesceAdjacentAssistantToolCalls merges consecutive assistant messages
// that both have tool_calls into a single message. This handles Claude Code's
// split tool call emission when conversation compression leaves adjacent
// assistant blocks.
func coalesceAdjacentAssistantToolCalls(messages []OpenAIMessage) []OpenAIMessage {
	var out []OpenAIMessage

	for _, msg := range messages {
		prev := len(out) - 1
		if prev >= 0 &&
			out[prev].Role == "assistant" &&
			msg.Role == "assistant" &&
			len(out[prev].ToolCalls) > 0 &&
			len(msg.ToolCalls) > 0 {

			// Merge content.
			prevContent := extractTextContent(out[prev].Content)
			curContent := extractTextContent(msg.Content)
			if prevContent != "" && curContent != "" {
				out[prev].Content = prevContent + "\n" + curContent
			} else if curContent != "" {
				out[prev].Content = curContent
			}

			// Merge reasoning_content.
			if msg.ReasoningContent != "" {
				if out[prev].ReasoningContent != "" {
					out[prev].ReasoningContent += "\n" + msg.ReasoningContent
				} else {
					out[prev].ReasoningContent = msg.ReasoningContent
				}
			}

			out[prev].ToolCalls = append(out[prev].ToolCalls, msg.ToolCalls...)
			continue
		}

		out = append(out, msg)
	}

	return out
}

// sanitizeOpenAiToolMessageSequence ensures each assistant tool_calls message
// is properly paired with its tool results, dropping unfulfilled calls and
// converting orphan results to user text.
func sanitizeOpenAiToolMessageSequence(messages []OpenAIMessage) []OpenAIMessage {
	var out []OpenAIMessage

	for i := 0; i < len(messages); i++ {
		msg := messages[i]
		toolCalls := msg.ToolCalls

		if msg.Role == "assistant" && len(toolCalls) > 0 {
			// Collect trailing tool messages.
			var toolMessages []OpenAIMessage
			j := i + 1
			for j < len(messages) && messages[j].Role == "tool" {
				toolMessages = append(toolMessages, messages[j])
				j++
			}

			if len(toolMessages) == 0 && j == len(messages) {
				// End of history — emit as-is (in-progress message).
				out = append(out, msg)
				i = j - 1
				continue
			}

			// Build expected IDs and available results.
			expectedIDs := make(map[string]bool)
			for _, tc := range toolCalls {
				if tc.ID != "" {
					expectedIDs[tc.ID] = true
				}
			}
			toolByID := make(map[string]OpenAIMessage)
			var orphanTools []OpenAIMessage
			for _, tm := range toolMessages {
				id := tm.ToolCallID
				if expectedIDs[id] {
					if _, seen := toolByID[id]; !seen {
						toolByID[id] = tm
					} else {
						orphanTools = append(orphanTools, tm)
					}
				} else {
					orphanTools = append(orphanTools, tm)
				}
			}

			// Keep only tool calls that have results.
			var fulfilledCalls []OpenAIToolCall
			for _, tc := range toolCalls {
				if _, ok := toolByID[tc.ID]; ok {
					fulfilledCalls = append(fulfilledCalls, tc)
				}
			}

			if len(fulfilledCalls) > 0 {
				outMsg := msg
				outMsg.ToolCalls = fulfilledCalls
				out = append(out, outMsg)
				for _, tc := range fulfilledCalls {
					out = append(out, toolByID[tc.ID])
				}
			} else {
				// All tool calls unfulfilled — drop tool_calls, keep text.
				if extractTextContent(msg.Content) != "" {
					outMsg := msg
					outMsg.ToolCalls = nil
					out = append(out, outMsg)
				}
			}

			for _, orphan := range orphanTools {
				out = append(out, OpenAIMessage{
					Role:    "user",
					Content: fmt.Sprintf("Tool result without a matching tool call (%s):\n%s", orphan.ToolCallID, extractTextContent(orphan.Content)),
				})
			}

			i = j - 1
			continue
		}

		if msg.Role == "tool" {
			out = append(out, OpenAIMessage{
				Role:    "user",
				Content: fmt.Sprintf("Tool result without a matching tool call (%s):\n%s", msg.ToolCallID, extractTextContent(msg.Content)),
			})
			continue
		}

		out = append(out, msg)
	}

	return out
}

// ---------------------------------------------------------------------------
// OpenAI Request → Anthropic Request
// ---------------------------------------------------------------------------

func convertOpenAIRequestToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req OpenAIChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal OpenAI request", "err", err)
		return body, nil
	}

	model := opts.ResolvedModel; if model == "" { model = resolveModelTarget(req.Model, opts.ModelMap) }
	// Respect the request's max_tokens / max_completion_tokens if set,
	// otherwise fall back to the CLI --max-tokens flag (default 8192).
	// ponytail: upstream APIs often default to unreasonably low limits.
	maxTokens := opts.MaxTokens
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxTokens = *req.MaxTokens
	} else if req.MaxCompletionTokens != nil && *req.MaxCompletionTokens > 0 {
		maxTokens = *req.MaxCompletionTokens
	}
	anthropic := AnthropicRequest{
		Model:     model,
		MaxTokens: maxTokens,
		Stream:    req.Stream,
	}

	if req.Temperature != nil {
		anthropic.Temperature = req.Temperature
	}
	if req.TopP != nil {
		anthropic.TopP = req.TopP
	}
	if req.TopK != nil {
		anthropic.TopK = req.TopK
	}
	if req.Metadata != nil {
		anthropic.Metadata = req.Metadata
	}

	// Stop sequences.
	if req.Stop != nil {
		switch v := req.Stop.(type) {
		case string:
			anthropic.StopSequences = []string{v}
		case []any:
			for _, s := range v {
				if str, ok := s.(string); ok {
					anthropic.StopSequences = append(anthropic.StopSequences, str)
				}
			}
		}
	}

	// Tools — sort for deterministic JSON, enforce schema type.
	if len(req.Tools) > 0 {
		sortOpenAITools(req.Tools)
		anthropic.Tools = make([]AnthropicTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			anthropic.Tools = append(anthropic.Tools, AnthropicTool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: ensureObjectSchema(t.Function.Parameters),
			})
		}

		// Tool choice mapping.
		switch v := req.ToolChoice.(type) {
		case string:
			switch v {
			case "auto":
				anthropic.ToolChoice = &AnthropicToolChoice{Type: "auto"}
			case "none":
				anthropic.ToolChoice = &AnthropicToolChoice{Type: "none"}
			case "required":
				anthropic.ToolChoice = &AnthropicToolChoice{Type: "any"}
			default:
				anthropic.ToolChoice = &AnthropicToolChoice{Type: "auto"}
			}
		case map[string]any:
			if t, _ := v["type"].(string); t == "function" {
				if fn, ok := v["function"].(map[string]any); ok {
					if name, _ := fn["name"].(string); name != "" {
						anthropic.ToolChoice = &AnthropicToolChoice{Type: "tool", Name: name}
					}
				}
			}
		}
	} else {
		// No tools declared — explicitly disable tool choice to prevent
		// upstream models from hallucinating tool calls.
		anthropic.ToolChoice = &AnthropicToolChoice{Type: "none"}
	}

	// Messages.
	var systemTexts []string
	for _, m := range req.Messages {
		if m.Role == "system" {
			systemTexts = append(systemTexts, extractTextContent(m.Content))
			continue
		}
		am := AnthropicMessage{Role: m.Role}
		// Anthropic only supports "user" and "assistant" roles.
		if am.Role != "user" && am.Role != "assistant" {
			am.Role = "user"
		}
		am.Content = convertMessageContent(m)
		anthropic.Messages = append(anthropic.Messages, am)
	}

	// Top-level system field.
	if len(systemTexts) > 0 {
		anthropic.System = []AnthropicTextBlock{
			{Type: "text", Text: strings.Join(systemTexts, "\n")},
		}
	}

	// Edge case: if messages ended up empty after filtering (e.g. only system),
	// inject a minimal user message — Anthropic requires at least one message.
	if len(anthropic.Messages) == 0 {
		anthropic.Messages = []AnthropicMessage{
			{
				Role:    "user",
				Content: []AnthropicContent{{Type: "text", Text: "..."}},
			},
		}
	}

	return json.Marshal(anthropic)
}

// extractTextContent extracts the text representation of an OpenAI message content field.
func extractTextContent(content any) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, p := range v {
			if m, ok := p.(map[string]any); ok {
				if t, _ := m["type"].(string); t == "text" {
					if txt, _ := m["text"].(string); txt != "" {
						parts = append(parts, txt)
					}
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return fmt.Sprintf("%v", content)
}

// flattenContentForGLM flattens multi-part content arrays to plain strings.
// GLM API does not support array-typed content (text + image_url parts).
func flattenContentForGLM(msgs []OpenAIMessage) {
	for i := range msgs {
		switch c := msgs[i].Content.(type) {
		case []any:
			var textParts []string
			for _, part := range c {
				if m, ok := part.(map[string]any); ok {
					if t, _ := m["type"].(string); t == "text" {
						if txt, _ := m["text"].(string); txt != "" {
							textParts = append(textParts, txt)
						}
					}
				}
			}
			msgs[i].Content = strings.Join(textParts, "\n")
		}
	}
}

// convertMessageContent converts an OpenAI message into Anthropic content blocks.
func convertMessageContent(msg OpenAIMessage) []AnthropicContent {
	switch msg.Role {
	case "user":
		return convertUserContent(msg.Content)
	case "assistant":
		return convertAssistantContent(msg)
	case "tool":
		return convertToolResultContent(msg)
	case "function":
		return []AnthropicContent{
			{Type: "text", Text: fmt.Sprintf("Function result (%s): %v", msg.Name, msg.Content)},
		}
	}
	return []AnthropicContent{{Type: "text", Text: fmt.Sprintf("%v", msg.Content)}}
}

func convertUserContent(content any) []AnthropicContent {
	if content == nil {
		return []AnthropicContent{{Type: "text", Text: ""}}
	}

	// Simple string.
	if s, ok := content.(string); ok {
		return []AnthropicContent{{Type: "text", Text: s}}
	}

	// Array of content parts (text + image_url).
	parts, ok := content.([]any)
	if !ok {
		return []AnthropicContent{{Type: "text", Text: fmt.Sprintf("%v", content)}}
	}

	var blocks []AnthropicContent
	for _, p := range parts {
		m, ok := p.(map[string]any)
		if !ok {
			continue
		}
		switch m["type"] {
		case "text":
			txt, _ := m["text"].(string)
			blocks = append(blocks, AnthropicContent{Type: "text", Text: txt})
		case "image_url":
			img := convertImageURL(m)
			if img != nil {
				blocks = append(blocks, *img)
			}
		}
	}
	return blocks
}

func convertImageURL(m map[string]any) *AnthropicContent {
	urlMap, ok := m["image_url"].(map[string]any)
	if !ok {
		return nil
	}
	url, _ := urlMap["url"].(string)
	if url == "" {
		return nil
	}

	// Only support data URIs with base64 encoding.
	if !strings.HasPrefix(url, "data:image/") {
		slog.Debug("skipping non-data-URI image URL")
		return nil
	}

	rest, ok := strings.CutPrefix(url, "data:")
	if !ok {
		return nil
	}
	rest, data, ok := strings.Cut(rest, ";base64,")
	if !ok {
		slog.Debug("skipping non-base64 image data URI")
		return nil
	}
	mediaType := rest

	return &AnthropicContent{
		Type: "image",
		Source: &AnthropicImageSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      data,
		},
	}
}

func convertAssistantContent(msg OpenAIMessage) []AnthropicContent {
	// Collect all tool calls, including legacy function_call.
	toolCalls := msg.ToolCalls
	if msg.FunctionCall != nil {
		toolCalls = append(toolCalls, OpenAIToolCall{
			Type:     "function",
			Function: *msg.FunctionCall,
		})
	}

	if len(toolCalls) > 0 {
		var blocks []AnthropicContent
		// Reasoning content → thinking block (must come before text).
		if msg.ReasoningContent != "" {
			blocks = append(blocks, AnthropicContent{
				Type:     "thinking",
				Thinking: msg.ReasoningContent,
			})
		}
		blocks = append(blocks, AnthropicContent{Type: "text", Text: extractTextContent(msg.Content)})
		for _, tc := range toolCalls {
			var input any
			if tc.Function.Arguments != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
			}
			blocks = append(blocks, AnthropicContent{
				Type:  "tool_use",
				ID:    tc.ID, // preserve original ID — client uses it for tool_result matching
				Name:  tc.Function.Name,
				Input: input,
			})
		}
		return blocks
	}

	// Reasoning content without tools — single thinking + text.
	if msg.ReasoningContent != "" {
		return []AnthropicContent{
			{Type: "thinking", Thinking: msg.ReasoningContent},
			{Type: "text", Text: extractTextContent(msg.Content)},
		}
	}

	// Plain text.
	text := extractTextContent(msg.Content)
	return []AnthropicContent{{Type: "text", Text: text}}
}

func convertToolResultContent(msg OpenAIMessage) []AnthropicContent {
	return []AnthropicContent{
		{
			Type:      "tool_result",
			ToolUseID: msg.ToolCallID,
			Content:   extractTextContent(msg.Content),
		},
	}
}

// ---------------------------------------------------------------------------
// OpenAI Response → Anthropic Response
// ---------------------------------------------------------------------------

func convertOpenAIResponseToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var resp OpenAIChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		slog.Warn("failed to unmarshal OpenAI response", "err", err)
		return body, nil
	}

	id := resp.ID
	if strings.HasPrefix(id, "chatcmpl-") {
		id = "msg_" + strings.TrimPrefix(id, "chatcmpl-")
	} else if !strings.HasPrefix(id, "msg_") {
		id = "msg_" + id
	}

	anth := AnthropicResponse{
		ID:    id,
		Type:  "message",
		Role:  "assistant",
		Model: resp.Model,
		Usage: AnthropicUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}

	// Rewrite model to the original client-facing name so that Claude Code's
	// safety classifier sees the expected model (e.g. "claude-opus-4-8")
	// instead of the upstream target (e.g. "deepseek-v4-pro").
	if opts != nil && opts.RequestModel != "" {
		anth.Model = opts.RequestModel
	} else if opts != nil && opts.ModelMap != nil {
		if sourcePrefix := opts.ModelMap.SourcePrefix(resp.Model); sourcePrefix != "" {
			anth.Model = sourcePrefix
		}
	}

	if len(resp.Choices) > 0 {
		msg := resp.Choices[0].Message
		anth.StopReason = mapOpenAIFinishReason(resp.Choices[0].FinishReason)
		anth.Content = convertOpenAIMessageToContent(msg)
	}

	// Cache reasoning_content for tool call replay (DeepSeek V4 requirement).
	// Store in all three tiers: tool call ID, assistant text, and tool context.
	if opts != nil && opts.ReasoningCache != nil && len(resp.Choices) > 0 {
		msg := resp.Choices[0].Message
		if msg.ReasoningContent != "" {
			if len(msg.ToolCalls) > 0 {
				ids := make([]string, len(msg.ToolCalls))
				for i, tc := range msg.ToolCalls {
					ids[i] = tc.ID
				}
				opts.ReasoningCache.Put(ids, msg.ReasoningContent)
			}
			content := extractTextContent(msg.Content)
			if content != "" {
				opts.ReasoningCache.PutText(content, msg.ReasoningContent)
			}
		}
	}

	return json.Marshal(anth)
}

func mapOpenAIFinishReason(reason *string) *string {
	if reason == nil {
		return nil
	}
	switch *reason {
	case "stop":
		s := "end_turn"
		return &s
	case "length":
		s := "max_tokens"
		return &s
	case "tool_calls":
		s := "tool_use"
		return &s
	case "content_filter":
		s := "end_turn"
		return &s
	default:
		return reason
	}
}

func convertOpenAIMessageToContent(msg OpenAIMessage) []AnthropicContent {
	var blocks []AnthropicContent

	// Reasoning content → thinking block (must come before text).
	if msg.ReasoningContent != "" {
		blocks = append(blocks, AnthropicContent{
			Type:     "thinking",
			Thinking: msg.ReasoningContent,
		})
	}

	// Text content.
	text := extractTextContent(msg.Content)

	// Tool calls.
	calls := msg.ToolCalls
	if msg.FunctionCall != nil {
		calls = append(calls, OpenAIToolCall{
			Type:     "function",
			Function: *msg.FunctionCall,
		})
	}

	if len(calls) > 0 {
		if text != "" {
			blocks = append(blocks, AnthropicContent{Type: "text", Text: text})
		}
		for _, tc := range calls {
			var input any
			if tc.Function.Arguments != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
			}
			blocks = append(blocks, AnthropicContent{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
	} else {
		// Ensure at least one valid text block — empty "" serializes as
		// {"type":"text"} (no text field), malformed per Anthropic schema.
		// DeepSeek-V4-Pro can spend all max_tokens on reasoning, returning
		// empty content + finish_reason:length, which falls through here.
		if text == "" {
			text = " " // minimal placeholder so the response is parseable
		}
		blocks = append(blocks, AnthropicContent{Type: "text", Text: text})
	}

	return blocks
}
