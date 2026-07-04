package convert

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// ---------------------------------------------------------------------------
// OpenAI Chat Request → Gemini generateContent Request
// ---------------------------------------------------------------------------

func convertOpenAIRequestToGemini(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req OpenAIChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal OpenAI request", "err", err)
		return body, nil
	}
	rawBody := make(map[string]any)
	json.Unmarshal(body, &rawBody) // best-effort, for optional fields

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
	maxTokens := opts.MaxTokens
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxTokens = *req.MaxTokens
	} else if req.MaxCompletionTokens != nil && *req.MaxCompletionTokens > 0 {
		maxTokens = *req.MaxCompletionTokens
	}
	if maxTokens > 0 {
		gc.MaxOutputTokens = &maxTokens
	}
	if req.Stop != nil {
		switch v := req.Stop.(type) {
		case string:
			gc.StopSequences = []string{v}
		case []any:
			for _, s := range v {
				if str, ok := s.(string); ok {
					gc.StopSequences = append(gc.StopSequences, str)
				}
			}
		}
	}
	if seed, _ := rawBody["seed"].(float64); seed != 0 {
		s := int(seed)
		gc.Seed = &s
	}
	// response_format → responseMimeType + responseSchema.
	if reqFmt, ok := rawBody["response_format"].(map[string]any); ok {
		if t, _ := reqFmt["type"].(string); t != "" {
			switch t {
			case "json_object":
				gc.ResponseMimeType = "application/json"
			case "json_schema":
				gc.ResponseMimeType = "application/json"
				if s, ok := reqFmt["schema"]; ok {
					gc.ResponseSchema = s
				}
			}
		}
	}
	gemini.GenerationConfig = gc

	// Build tool_call_id → function_name mapping for tool result resolution.
	toolCallIDToName := make(map[string]string)
	for _, m := range req.Messages {
		if m.Role == "assistant" {
			for _, tc := range m.ToolCalls {
				toolCallIDToName[tc.ID] = tc.Function.Name
			}
		}
	}

	// Messages → contents + systemInstruction.
	var sysTexts []string
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			sysTexts = append(sysTexts, extractTextContent(m.Content))
		default:
			content := convertOpenAIMsgToGeminiContent(m, toolCallIDToName)
			if content != nil {
				gemini.Contents = append(gemini.Contents, *content)
			}
		}
	}
	if len(sysTexts) > 0 {
		gemini.SystemInstruction = &GeminiContent{
			Parts: []GeminiPart{{Text: strings.Join(sysTexts, "\n")}},
			Role:  "user",
		}
	}

	// Tools → functionDeclarations.
	if len(req.Tools) > 0 {
		sortOpenAITools(req.Tools)
		var fds []GeminiFunctionDeclaration
		for _, t := range req.Tools {
			fds = append(fds, GeminiFunctionDeclaration{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
			})
		}
		gemini.Tools = []GeminiTool{{FunctionDeclarations: fds}}

		// Tool choice → toolConfig.
		mode := "AUTO"
		if tc, ok := req.ToolChoice.(string); ok {
			switch tc {
			case "none":
				mode = "NONE"
			case "auto":
				mode = "AUTO"
			case "required":
				mode = "ANY"
			}
		} else if tc, ok := req.ToolChoice.(map[string]any); ok {
			if t, _ := tc["type"].(string); t == "function" {
				if fn, ok := tc["function"].(map[string]any); ok {
					if name, _ := fn["name"].(string); name != "" {
						mode = "ANY"
						gemini.ToolConfig = &GeminiToolConfig{
							FunctionCallingConfig: &GeminiFunctionCallingConfig{
								Mode:                 mode,
								AllowedFunctionNames: []string{name},
							},
						}
					}
				}
			}
		}
		if gemini.ToolConfig == nil {
			gemini.ToolConfig = &GeminiToolConfig{
				FunctionCallingConfig: &GeminiFunctionCallingConfig{Mode: mode},
			}
		}
	}

	return json.Marshal(gemini)
}

// convertOpenAIMsgToGeminiContent converts a single OpenAI message to Gemini content.
// toolCallIDToName maps tool_call_id → function name for tool result resolution.
func convertOpenAIMsgToGeminiContent(msg OpenAIMessage, toolCallIDToName map[string]string) *GeminiContent {
	role := msg.Role
	switch role {
	case "assistant":
		role = "model"
	case "tool", "function":
		role = "user"
	case "user":
		// keep
	default:
		role = "user"
	}

	var parts []GeminiPart

	// Text content.
	if msg.Content != nil {
		switch c := msg.Content.(type) {
		case string:
			if c != "" {
				parts = append(parts, GeminiPart{Text: c})
			}
		case []any:
			for _, p := range c {
				if m, ok := p.(map[string]any); ok {
					switch m["type"] {
					case "text":
						if txt, _ := m["text"].(string); txt != "" {
							parts = append(parts, GeminiPart{Text: txt})
						}
					case "image_url":
						if img := convertOpenAIImageURLToGemini(m); img != nil {
							parts = append(parts, GeminiPart{InlineData: img})
						}
					}
				}
			}
		}
	}

	// Reasoning content → text part (no native thought marker in Gemini).
	if msg.ReasoningContent != "" {
		parts = append([]GeminiPart{{Text: msg.ReasoningContent}}, parts...)
	}

	// Tool calls → functionCall parts.
	for _, tc := range msg.ToolCalls {
		var args any
		if tc.Function.Arguments != "" {
			json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		parts = append(parts, GeminiPart{
			FunctionCall: &GeminiFunctionCall{
				Name: tc.Function.Name,
				Args: args,
			},
		})
	}

	// Legacy function_call.
	if msg.FunctionCall != nil {
		var args any
		if msg.FunctionCall.Arguments != "" {
			json.Unmarshal([]byte(msg.FunctionCall.Arguments), &args)
		}
		parts = append(parts, GeminiPart{
			FunctionCall: &GeminiFunctionCall{
				Name: msg.FunctionCall.Name,
				Args: args,
			},
		})
	}

	// Tool result → functionResponse part.
	if msg.Role == "tool" && msg.ToolCallID != "" {
		name := msg.ToolCallID
		if fn, ok := toolCallIDToName[msg.ToolCallID]; ok {
			name = fn
		}
		parts = append(parts, GeminiPart{
			FunctionResponse: &GeminiFunctionResponse{
				Name:     name,
				Response: extractTextContent(msg.Content),
			},
		})
	}

	if len(parts) == 0 {
		return nil
	}
	return &GeminiContent{Parts: parts, Role: role}
}

// convertOpenAIImageURLToGemini extracts inlineData from an OpenAI image_url part.
func convertOpenAIImageURLToGemini(m map[string]any) *GeminiInlineData {
	urlMap, ok := m["image_url"].(map[string]any)
	if !ok {
		return nil
	}
	url, _ := urlMap["url"].(string)
	if url == "" || !strings.HasPrefix(url, "data:") {
		return nil
	}
	rest, ok := strings.CutPrefix(url, "data:")
	if !ok {
		return nil
	}
	rest, data, ok := strings.Cut(rest, ";base64,")
	if !ok {
		return nil
	}
	return &GeminiInlineData{MimeType: rest, Data: data}
}

// ---------------------------------------------------------------------------
// Gemini Response → OpenAI Chat Response
// ---------------------------------------------------------------------------

func convertGeminiResponseToOpenAI(body []byte, opts *ConvertOptions) ([]byte, error) {
	var resp GeminiChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		slog.Warn("failed to unmarshal Gemini response", "err", err)
		return body, nil
	}

	// Empty candidates with block reason → error response.
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			slog.Warn("Gemini response blocked", "blockReason", resp.PromptFeedback.BlockReason)
			b, _ := json.Marshal(map[string]any{
				"error": map[string]any{
					"message": fmt.Sprintf("blocked by Gemini safety: %s", resp.PromptFeedback.BlockReason),
					"type":    "content_filter",
				},
			})
			return b, nil
		}
	}

	openai := OpenAIChatResponse{
		ID:      "chatcmpl-gemini-" + randHex(16),
		Object:  "chat.completion",
		Created: nowUnix(),
	}

	// Model rewrite.
	if opts != nil && opts.RequestModel != "" {
		openai.Model = opts.RequestModel
	} else if opts != nil && opts.ModelMap != nil && opts.ResolvedModel != "" {
		openai.Model = opts.ResolvedModel
	}

	// Usage.
	if resp.UsageMetadata != nil {
		openai.Usage = OpenAIUsage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		}
	}

	// Candidates → choices.
	if len(resp.Candidates) > 0 {
		c := resp.Candidates[0]
		msg := OpenAIMessage{Role: "assistant"}
		var textParts []string

		for _, part := range c.Content.Parts {
			switch {
			case part.Text != "" && part.FunctionCall == nil:
				textParts = append(textParts, part.Text)
			case part.FunctionCall != nil:
				args := "{}"
				if part.FunctionCall.Args != nil {
					if b, err := json.Marshal(part.FunctionCall.Args); err == nil {
						args = string(b)
					}
				}
				msg.ToolCalls = append(msg.ToolCalls, OpenAIToolCall{
					ID:   "call_" + randHex(16),
					Type: "function",
					Function: OpenAIFunctionCall{
						Name:      part.FunctionCall.Name,
						Arguments: args,
					},
				})
			case part.InlineData != nil:
				textParts = append(textParts, fmt.Sprintf("![image](data:%s;base64,%s)", part.InlineData.MimeType, part.InlineData.Data))
			case part.ExecutableCode != nil:
				textParts = append(textParts, fmt.Sprintf("```%s\n%s\n```", part.ExecutableCode.Language, part.ExecutableCode.Code))
			case part.CodeExecutionResult != nil:
				textParts = append(textParts, fmt.Sprintf("```output\n%s\n```", part.CodeExecutionResult.Output))
			}
		}

		content := strings.Join(textParts, "\n")
		if len(msg.ToolCalls) > 0 {
			if content != "" {
				msg.Content = content
			}
		} else {
			msg.Content = content
		}

		openai.Choices = []OpenAIChoice{{
			Index:        c.Index,
			Message:      msg,
			FinishReason: mapGeminiFinishToOpenAI(c.FinishReason, len(msg.ToolCalls) > 0),
		}}
	}

	return json.Marshal(openai)
}

func mapGeminiFinishToOpenAI(reason string, hasToolCalls bool) *string {
	if hasToolCalls && reason == "STOP" {
		return &strToolCalls
	}
	switch reason {
	case "STOP":
		return &strStop
	case "MAX_TOKENS":
		return &strLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "OTHER":
		s := "content_filter"
		return &s
	default:
		if reason != "" {
			return &reason
		}
		return &strStop
	}
}

// ---------------------------------------------------------------------------
// Gemini Request → OpenAI Chat Request
// ---------------------------------------------------------------------------

func convertGeminiRequestToOpenAI(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req GeminiChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal Gemini request", "err", err)
		return body, nil
	}

	oai := OpenAIChatRequest{
		Model: opts.ResolvedModel,
	}

	if req.GenerationConfig != nil {
		if req.GenerationConfig.Temperature != nil {
			oai.Temperature = req.GenerationConfig.Temperature
		}
		if req.GenerationConfig.TopP != nil {
			oai.TopP = req.GenerationConfig.TopP
		}
		if req.GenerationConfig.MaxOutputTokens != nil {
			v := *req.GenerationConfig.MaxOutputTokens
			oai.MaxTokens = &v
		}
		if len(req.GenerationConfig.StopSequences) > 0 {
			if len(req.GenerationConfig.StopSequences) == 1 {
				oai.Stop = req.GenerationConfig.StopSequences[0]
			} else {
				anys := make([]any, len(req.GenerationConfig.StopSequences))
				for i, s := range req.GenerationConfig.StopSequences {
					anys[i] = s
				}
				oai.Stop = anys
			}
		}
		if req.GenerationConfig.Seed != nil {
			// ponytail: seed maps to a non-standard field, drop silently
		}
	}

	// System instruction → system message.
	if req.SystemInstruction != nil {
		var sb strings.Builder
		for _, p := range req.SystemInstruction.Parts {
			sb.WriteString(p.Text)
		}
		oai.Messages = append(oai.Messages, OpenAIMessage{Role: "system", Content: sb.String()})
	}

	// Contents → messages.
	for _, c := range req.Contents {
		role := c.Role
		if role == "model" {
			role = "assistant"
		}
		oaiMsg := OpenAIMessage{Role: role}
		var textParts []string
		for _, p := range c.Parts {
			switch {
			case p.Text != "" && p.FunctionCall == nil && p.FunctionResponse == nil:
				textParts = append(textParts, p.Text)
			case p.FunctionCall != nil:
				args := "{}"
				if p.FunctionCall.Args != nil {
					if b, err := json.Marshal(p.FunctionCall.Args); err == nil {
						args = string(b)
					}
				}
				oaiMsg.ToolCalls = append(oaiMsg.ToolCalls, OpenAIToolCall{
					ID:   "call_" + randHex(16),
					Type: "function",
					Function: OpenAIFunctionCall{
						Name:      p.FunctionCall.Name,
						Arguments: args,
					},
				})
			case p.FunctionResponse != nil:
				oaiMsg.Role = "tool"
				resp, _ := json.Marshal(p.FunctionResponse.Response)
				oaiMsg.Content = string(resp)
				oaiMsg.ToolCallID = p.FunctionResponse.Name
			case p.InlineData != nil:
				textParts = append(textParts, fmt.Sprintf("![image](data:%s;base64,%s)", p.InlineData.MimeType, p.InlineData.Data))
			}
		}
		content := strings.Join(textParts, "\n")
		if len(oaiMsg.ToolCalls) > 0 {
			if content != "" {
				oaiMsg.Content = content
			}
		} else if oaiMsg.Role != "tool" {
			oaiMsg.Content = content
		}
		oai.Messages = append(oai.Messages, oaiMsg)
	}

	// Tools → OpenAI format.
	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			for _, fd := range t.FunctionDeclarations {
				oai.Tools = append(oai.Tools, OpenAITool{
					Type: "function",
					Function: OpenAIFunction{
						Name:        fd.Name,
						Description: fd.Description,
						Parameters:  fd.Parameters,
					},
				})
			}
		}
	}

	return json.Marshal(oai)
}

// ---------------------------------------------------------------------------
// OpenAI Chat Response → Gemini Response
// ---------------------------------------------------------------------------

func convertOpenAIResponseToGemini(body []byte, _ *ConvertOptions) ([]byte, error) {
	var resp OpenAIChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		slog.Warn("failed to unmarshal OpenAI response", "err", err)
		return body, nil
	}

	gemini := GeminiChatResponse{}
	if resp.Usage.PromptTokens > 0 || resp.Usage.CompletionTokens > 0 {
		gemini.UsageMetadata = &GeminiUsage{
			PromptTokenCount:     resp.Usage.PromptTokens,
			CandidatesTokenCount: resp.Usage.CompletionTokens,
			TotalTokenCount:      resp.Usage.TotalTokens,
		}
	}

	if len(resp.Choices) > 0 {
		ch := resp.Choices[0]
		c := GeminiCandidate{
			Index:        ch.Index,
			FinishReason: mapOpenAIFinishToGemini(ch.FinishReason),
			Content:      GeminiContent{Role: "model"},
		}

		msg := ch.Message
		var parts []GeminiPart

		// Reasoning content → text part first (mimics thinking order).
		if msg.ReasoningContent != "" {
			parts = append(parts, GeminiPart{Text: msg.ReasoningContent})
		}

		// Text content.
		text := extractTextContent(msg.Content)
		if text != "" {
			parts = append(parts, GeminiPart{Text: text})
		}

		// Tool calls → functionCall parts.
		for _, tc := range msg.ToolCalls {
			var args any
			if tc.Function.Arguments != "" {
				json.Unmarshal([]byte(tc.Function.Arguments), &args)
			}
			parts = append(parts, GeminiPart{
				FunctionCall: &GeminiFunctionCall{
					Name: tc.Function.Name,
					Args: args,
				},
			})
		}

		c.Content.Parts = parts
		gemini.Candidates = []GeminiCandidate{c}
	}

	return json.Marshal(gemini)
}

func mapOpenAIFinishToGemini(reason *string) string {
	if reason == nil {
		return "STOP"
	}
	switch *reason {
	case "stop":
		return "STOP"
	case "length":
		return "MAX_TOKENS"
	case "tool_calls":
		return "STOP"
	case "content_filter":
		return "SAFETY"
	default:
		return "STOP"
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func defaultGeminiSafety() []GeminiSafetySetting {
	return []GeminiSafetySetting{
		{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE"},
		{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_NONE"},
		{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "BLOCK_NONE"},
		{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_NONE"},
	}
}

func nowUnix() int64 {
	return 1749000000 // ponytail: static timestamp, replace with time.Now() if ordering matters
}
