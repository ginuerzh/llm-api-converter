package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
)

// ---------------------------------------------------------------------------
// Anthropic Request → Gemini generateContent Request
// ---------------------------------------------------------------------------

func convertAnthropicRequestToGemini(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal Anthropic request", "err", err)
		return body, nil
	}

	gemini := GeminiChatRequest{
		SafetySettings: defaultGeminiSafety(),
	}

	// Generation config.
	gc := &GeminiGenerationConfig{}
	if req.Temperature != nil {
		gc.Temperature = req.Temperature
	}
	if req.TopP != nil {
		gc.TopP = req.TopP
	}
	if req.TopK != nil {
		v := float64(*req.TopK)
		gc.TopK = &v
	}
	if req.MaxTokens > 0 {
		gc.MaxOutputTokens = &req.MaxTokens
	}
	if len(req.StopSequences) > 0 {
		gc.StopSequences = req.StopSequences
	}
	// Thinking → simplified: budget maps to maxOutputTokens if not already set.
	if req.Thinking != nil && req.Thinking.Type == "enabled" {
		if req.MaxTokens <= 0 && req.Thinking.BudgetTokens > 0 {
			gc.MaxOutputTokens = &req.Thinking.BudgetTokens
		}
	}
	gemini.GenerationConfig = gc

	// System → systemInstruction.
	if len(req.System) > 0 {
		var parts []GeminiPart
		for _, block := range req.System {
			if block.Text != "" {
				parts = append(parts, GeminiPart{Text: stripLeadingAnthropicBillingHeader(block.Text)})
			}
		}
		if len(parts) > 0 {
			gemini.SystemInstruction = &GeminiContent{Parts: parts, Role: "user"}
		}
	}

	// Messages → contents.
	for _, msg := range req.Messages {
		content := convertAnthropicMsgToGeminiContent(msg, opts)
		if content != nil {
			gemini.Contents = append(gemini.Contents, *content)
		}
	}

	// Tools → functionDeclarations.
	if len(req.Tools) > 0 {
		sortAnthropicTools(req.Tools)
		var fds []GeminiFunctionDeclaration
		for _, t := range req.Tools {
			fds = append(fds, GeminiFunctionDeclaration{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  sanitizeGeminiSchema(t.InputSchema),
			})
		}
		gemini.Tools = []GeminiTool{{FunctionDeclarations: fds}}

		// Tool choice → toolConfig.
		mode := "AUTO"
		var allowedNames []string
		if req.ToolChoice != nil {
			switch req.ToolChoice.Type {
			case "auto":
				mode = "AUTO"
			case "any":
				mode = "ANY"
			case "none":
				mode = "NONE"
			case "tool":
				mode = "ANY"
				if req.ToolChoice.Name != "" {
					allowedNames = []string{req.ToolChoice.Name}
				}
			}
		}
		gemini.ToolConfig = &GeminiToolConfig{
			FunctionCallingConfig: &GeminiFunctionCallingConfig{
				Mode:                 mode,
				AllowedFunctionNames: allowedNames,
			},
		}
	}

	return json.Marshal(gemini)
}

func convertAnthropicMsgToGeminiContent(msg AnthropicMessage, opts *ConvertOptions) *GeminiContent {
	role := msg.Role
	if role == "assistant" {
		role = "model"
	} else if role != "user" {
		role = "user"
	}

	var parts []GeminiPart
	var toolResults []GeminiPart

	for _, block := range msg.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				parts = append(parts, GeminiPart{Text: block.Text})
			}
		case "thinking":
			t := block.Thinking
			if t == "" {
				t = block.Text
			}
			if t != "" {
				parts = append(parts, GeminiPart{Text: t})
			}
		case "tool_use":
			var args any
			if block.Input != nil {
				args = block.Input
			}
			part := GeminiPart{
				FunctionCall: &GeminiFunctionCall{
					Name: block.Name,
					Args: args,
				},
			}
			if opts != nil && opts.SessionStore != nil && opts.SID != "" {
				if sig := opts.SessionStore.GetThoughtSig(opts.SID, block.Name); sig != "" {
					part.ThoughtSignature = sig
				}
			}
			parts = append(parts, part)
		case "tool_result":
			toolResults = append(toolResults, GeminiPart{
				FunctionResponse: &GeminiFunctionResponse{
					Name:     block.ToolUseID,
					Response: normalizeToolResponse(block.Content),
				},
			})
		case "image":
			if block.Source != nil && block.Source.Type == "base64" {
				parts = append(parts, GeminiPart{
					InlineData: &GeminiInlineData{
						MimeType: block.Source.MediaType,
						Data:     block.Source.Data,
					},
				})
			}
		}
	}

	if len(parts) == 0 && len(toolResults) == 0 {
		return nil
	}

	// Tool results go in a separate user-role content block.
	if len(toolResults) > 0 {
		parts = append(parts, toolResults...)
	}

	return &GeminiContent{Parts: parts, Role: role}
}

// ---------------------------------------------------------------------------
// Gemini Response → Anthropic Response
// ---------------------------------------------------------------------------

func convertGeminiResponseToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var resp GeminiChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		slog.Warn("failed to unmarshal Gemini response", "err", err)
		return body, nil
	}

	// Empty candidates → error response (either blocked by safety or API error).
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			slog.Warn("Gemini response blocked", "blockReason", resp.PromptFeedback.BlockReason)
			b, _ := json.Marshal(map[string]any{
				"type": "error",
				"error": map[string]any{
					"type":    "permission_error",
					"message": fmt.Sprintf("blocked by Gemini safety: %s", resp.PromptFeedback.BlockReason),
				},
			})
			return b, nil
		}
		// No block reason — likely a Gemini API error (e.g. 429 quota).
		var raw map[string]any
		if json.Unmarshal(body, &raw) == nil {
			if errVal, ok := raw["error"]; ok {
				msg, code := extractGeminiError(errVal)
				slog.Warn("Gemini API error in response", "msg", msg, "code", code)
				b, _ := json.Marshal(map[string]any{
					"type": "error",
					"error": map[string]any{
						"type":    "api_error",
						"message": fmt.Sprintf("Gemini API error (HTTP %d): %s", code, msg),
					},
				})
				return b, nil
			}
		}
	}

	anth := AnthropicResponse{
		ID:    "msg_gemini_" + randHex(16),
		Type:  "message",
		Role:  "assistant",
		Model: "gemini",
	}

	// Model rewrite.
	if opts != nil && opts.RequestModel != "" {
		anth.Model = opts.RequestModel
	} else if opts != nil && opts.ModelMap != nil {
		if sourcePrefix := opts.ModelMap.SourcePrefix(anth.Model); sourcePrefix != "" {
			anth.Model = sourcePrefix
		}
	}

	// Usage.
	if resp.UsageMetadata != nil {
		anth.Usage = AnthropicUsage{
			InputTokens:  resp.UsageMetadata.PromptTokenCount,
			OutputTokens: resp.UsageMetadata.CandidatesTokenCount,
		}
	}

	// Candidates → content blocks.
	if len(resp.Candidates) > 0 {
		c := resp.Candidates[0]
		anth.StopReason = mapGeminiFinishToAnthropic(c.FinishReason)

		// Capture thoughtSignature from any functionCall parts for session reuse.
		for _, part := range c.Content.Parts {
			if part.FunctionCall != nil && part.ThoughtSignature != "" {
				if opts != nil && opts.SessionStore != nil && opts.SID != "" {
					opts.SessionStore.SetThoughtSig(opts.SID, part.FunctionCall.Name, part.ThoughtSignature)
				}
			}
		}

		for _, part := range c.Content.Parts {
			switch {
			case part.Text != "" && part.FunctionCall == nil:
				anth.Content = append(anth.Content, AnthropicContent{
					Type: "text",
					Text: part.Text,
				})
			case part.FunctionCall != nil:
				id := ensureToolID("")
				anth.Content = append(anth.Content, AnthropicContent{
					Type:  "tool_use",
					ID:    id,
					Name:  part.FunctionCall.Name,
					Input: part.FunctionCall.Args,
				})
			case part.InlineData != nil:
				anth.Content = append(anth.Content, AnthropicContent{
					Type: "image",
					Source: &AnthropicImageSource{
						Type:      "base64",
						MediaType: part.InlineData.MimeType,
						Data:      part.InlineData.Data,
					},
				})
			}
		}

		if len(anth.Content) == 0 {
			anth.Content = []AnthropicContent{{Type: "text", Text: " "}}
		}
	}

	return json.Marshal(anth)
}

// anthropicResponseToSSE wraps a complete Anthropic JSON response into
// the SSE event sequence the Anthropic streaming protocol requires.
// The input is a single JSON object from convertGeminiResponseToAnthropic();
// the output is SSE events ready for a client that sent stream: true.
func anthropicResponseToSSE(body []byte) ([]byte, error) {
	var resp AnthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return body, nil
	}

	var buf bytes.Buffer

	// message_start with empty content (blocks arrive separately).
	writeSSEEvent(&buf, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            resp.ID,
			"type":          "message",
			"role":          resp.Role,
			"content":       []any{},
			"model":         resp.Model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  resp.Usage.InputTokens,
				"output_tokens": 0,
			},
		},
	})

	// ping
	writeSSEEvent(&buf, "ping", map[string]string{"type": "ping"})

	// Content blocks.
	for i, block := range resp.Content {
		writeSSEEvent(&buf, "content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         i,
			"content_block": block,
		})
		writeSSEEvent(&buf, "content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": i,
		})
	}

	// message_delta with stop_reason and usage.
	stopReason := "end_turn"
	if resp.StopReason != nil {
		stopReason = *resp.StopReason
	}
	writeSSEEvent(&buf, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"input_tokens":  resp.Usage.InputTokens,
			"output_tokens": resp.Usage.OutputTokens,
		},
	})

	// message_stop
	writeSSEEvent(&buf, "message_stop", map[string]string{"type": "message_stop"})

	return buf.Bytes(), nil
}

// writeSSEEvent writes a single SSE event to buf in the standard format:
//   event: <name>
//   data: <json>
//   (blank line)
func writeSSEEvent(buf *bytes.Buffer, event string, data any) {
	b, _ := json.Marshal(data)
	buf.WriteString("event: ")
	buf.WriteString(event)
	buf.WriteByte('\n')
	buf.WriteString("data: ")
	buf.Write(b)
	buf.WriteString("\n\n")
}

func mapGeminiFinishToAnthropic(reason string) *string {
	switch reason {
	case "STOP":
		s := "end_turn"
		return &s
	case "MAX_TOKENS":
		s := "max_tokens"
		return &s
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "OTHER":
		s := "end_turn"
		return &s
	default:
		if reason != "" {
			return &reason
		}
		return nil
	}
}

// ---------------------------------------------------------------------------
// Gemini Request → Anthropic Request
// ---------------------------------------------------------------------------

func convertGeminiRequestToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req GeminiChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal Gemini request", "err", err)
		return body, nil
	}

	anth := AnthropicRequest{
		Model:   opts.ResolvedModel,
		MaxTokens: 8192,
	}

	if req.GenerationConfig != nil {
		if req.GenerationConfig.Temperature != nil {
			anth.Temperature = req.GenerationConfig.Temperature
		}
		if req.GenerationConfig.TopP != nil {
			anth.TopP = req.GenerationConfig.TopP
		}
		if req.GenerationConfig.MaxOutputTokens != nil {
			anth.MaxTokens = *req.GenerationConfig.MaxOutputTokens
		}
		if len(req.GenerationConfig.StopSequences) > 0 {
			anth.StopSequences = req.GenerationConfig.StopSequences
		}
	}

	// System instruction → system text blocks.
	if req.SystemInstruction != nil {
		for _, p := range req.SystemInstruction.Parts {
			if p.Text != "" {
				anth.System = append(anth.System, AnthropicTextBlock{
					Type: "text",
					Text: p.Text,
				})
			}
		}
	}

	// Contents → messages.
	for _, c := range req.Contents {
		role := c.Role
		if role == "model" {
			role = "assistant"
		}
		msg := AnthropicMessage{Role: role}
		msg.Content = convertGeminiPartsToAnthropic(c.Parts)
		if len(msg.Content) > 0 {
			anth.Messages = append(anth.Messages, msg)
		}
	}

	// Tools → Anthropic format.
	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			for _, fd := range t.FunctionDeclarations {
				anth.Tools = append(anth.Tools, AnthropicTool{
					Name:        fd.Name,
					Description: fd.Description,
					InputSchema: fd.Parameters,
				})
			}
		}
	}

	// Tool config → tool_choice.
	if req.ToolConfig != nil && req.ToolConfig.FunctionCallingConfig != nil {
		fcc := req.ToolConfig.FunctionCallingConfig
		switch fcc.Mode {
		case "NONE":
			anth.ToolChoice = &AnthropicToolChoice{Type: "none"}
		case "ANY":
			if len(fcc.AllowedFunctionNames) == 1 {
				anth.ToolChoice = &AnthropicToolChoice{
					Type: "tool",
					Name: fcc.AllowedFunctionNames[0],
				}
			} else {
				anth.ToolChoice = &AnthropicToolChoice{Type: "any"}
			}
		default:
			anth.ToolChoice = &AnthropicToolChoice{Type: "auto"}
		}
	}

	if len(anth.Messages) == 0 {
		anth.Messages = []AnthropicMessage{
			{Role: "user", Content: []AnthropicContent{{Type: "text", Text: "..."}}},
		}
	}

	return json.Marshal(anth)
}

func convertGeminiPartsToAnthropic(parts []GeminiPart) []AnthropicContent {
	var blocks []AnthropicContent
	for _, p := range parts {
		switch {
		case p.Text != "" && p.FunctionCall == nil && p.FunctionResponse == nil:
			blocks = append(blocks, AnthropicContent{Type: "text", Text: p.Text})
		case p.FunctionCall != nil:
			blocks = append(blocks, AnthropicContent{
				Type:  "tool_use",
				ID:    ensureToolID(""),
				Name:  p.FunctionCall.Name,
				Input: p.FunctionCall.Args,
			})
		case p.FunctionResponse != nil:
			blocks = append(blocks, AnthropicContent{
				Type:      "tool_result",
				ToolUseID: p.FunctionResponse.Name,
				Content:   p.FunctionResponse.Response,
			})
		case p.InlineData != nil:
			blocks = append(blocks, AnthropicContent{
				Type: "image",
				Source: &AnthropicImageSource{
					Type:      "base64",
					MediaType: p.InlineData.MimeType,
					Data:      p.InlineData.Data,
				},
			})
		}
	}
	return blocks
}

// ---------------------------------------------------------------------------
// Anthropic Response → Gemini Response
// ---------------------------------------------------------------------------

func convertAnthropicResponseToGemini(body []byte, _ *ConvertOptions) ([]byte, error) {
	var resp AnthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		slog.Warn("failed to unmarshal Anthropic response", "err", err)
		return body, nil
	}

	gemini := GeminiChatResponse{}
	if resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0 {
		gemini.UsageMetadata = &GeminiUsage{
			PromptTokenCount:     resp.Usage.InputTokens,
			CandidatesTokenCount: resp.Usage.OutputTokens,
			TotalTokenCount:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
	}

	c := GeminiCandidate{
		Index:        0,
		FinishReason: mapAnthropicFinishToGemini(resp.StopReason),
		Content:      GeminiContent{Role: "model"},
	}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				c.Content.Parts = append(c.Content.Parts, GeminiPart{Text: block.Text})
			}
		case "thinking":
			t := block.Thinking
			if t == "" {
				t = block.Text
			}
			if t != "" {
				c.Content.Parts = append(c.Content.Parts, GeminiPart{Text: t})
			}
		case "tool_use":
			c.Content.Parts = append(c.Content.Parts, GeminiPart{
				FunctionCall: &GeminiFunctionCall{
					Name: block.Name,
					Args: block.Input,
				},
			})
		case "image":
			if block.Source != nil && block.Source.Type == "base64" {
				c.Content.Parts = append(c.Content.Parts, GeminiPart{
					InlineData: &GeminiInlineData{
						MimeType: block.Source.MediaType,
						Data:     block.Source.Data,
					},
				})
			}
		}
	}

	if len(c.Content.Parts) == 0 {
		c.Content.Parts = []GeminiPart{{Text: " "}}
	}

	gemini.Candidates = []GeminiCandidate{c}
	return json.Marshal(gemini)
}

func mapAnthropicFinishToGemini(reason *string) string {
	if reason == nil {
		return "STOP"
	}
	switch *reason {
	case "end_turn":
		return "STOP"
	case "max_tokens":
		return "MAX_TOKENS"
	case "tool_use":
		return "STOP"
	case "stop_sequence":
		return "STOP"
	default:
		return "STOP"
	}
}

// ---------------------------------------------------------------------------
// Helpers
