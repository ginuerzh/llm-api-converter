package convert

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// Convert detects the input body format and performs bidirectional
// conversion between OpenAI Chat Completions and Anthropic Messages formats.
func Convert(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	}

	if len(body) == 0 {
		return body, nil
	}

	// Try parsing as a generic object to detect format.
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		slog.Debug("not JSON, passing through", "err", err)
		return body, nil
	}

	// Detect: Anthropic Messages Response (type "message" and stop_reason/usage).
	// Must check BEFORE OpenAI request since Anthropic responses also have a "model" field.
	if isAnthropicResponse(raw) {
		slog.Debug("detected Anthropic Response → converting to OpenAI")
		return convertAnthropicResponseToOpenAI(body)
	}

	// Detect: OpenAI Chat Completions Request (model or messages field)
	if hasStringField(raw, "model") || hasArrayField(raw, "messages") {
		slog.Debug("detected OpenAI Chat Request → converting to Anthropic")
		return convertOpenAIRequestToAnthropic(body, opts)
	}

	slog.Debug("unknown format, passing through")
	return body, nil
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// OpenAI Request → Anthropic Request
// ---------------------------------------------------------------------------

func convertOpenAIRequestToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req OpenAIChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal OpenAI request", "err", err)
		return body, nil
	}

	anthropic := AnthropicRequest{
		Model:     opts.Model,
		MaxTokens: opts.MaxTokens,
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

	// Tools.
	if len(req.Tools) > 0 {
		anthropic.Tools = make([]AnthropicTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			anthropic.Tools = append(anthropic.Tools, AnthropicTool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: t.Function.Parameters,
			})
		}
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
	// If there are tool_calls, produce tool_use blocks.
	if len(msg.ToolCalls) > 0 {
		blocks := []AnthropicContent{
			{Type: "text", Text: extractTextContent(msg.Content)},
		}
		for _, tc := range msg.ToolCalls {
			var input any
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
			blocks = append(blocks, AnthropicContent{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
		return blocks
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
		// thinking and signature blocks are ignored.
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
		return nil
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
	default:
		return reason
	}
}

var (
	strStop      = "stop"
	strLength    = "length"
	strToolCalls = "tool_calls"
)
