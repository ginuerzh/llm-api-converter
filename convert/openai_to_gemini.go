package convert

import (
	"bytes"
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
			content := convertOpenAIMsgToGeminiContent(m, toolCallIDToName, opts)
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
				Parameters:  sanitizeGeminiSchema(t.Function.Parameters),
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
func convertOpenAIMsgToGeminiContent(msg OpenAIMessage, toolCallIDToName map[string]string, opts *ConvertOptions) *GeminiContent {
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
		part := GeminiPart{
			FunctionCall: &GeminiFunctionCall{
				Name: tc.Function.Name,
				Args: args,
			},
		}
		if opts != nil && opts.SessionStore != nil && opts.SID != "" {
			if sig := opts.SessionStore.GetThoughtSig(opts.SID, tc.Function.Name); sig != "" {
				part.ThoughtSignature = sig
			} else {
				part.ThoughtSignature = "skip_thought_signature_validator"
			}
		}
		parts = append(parts, part)
	}

	// Legacy function_call.
	if msg.FunctionCall != nil {
		var args any
		if msg.FunctionCall.Arguments != "" {
			json.Unmarshal([]byte(msg.FunctionCall.Arguments), &args)
		}
		part := GeminiPart{
			FunctionCall: &GeminiFunctionCall{
				Name: msg.FunctionCall.Name,
				Args: args,
			},
		}
		if opts != nil && opts.SessionStore != nil && opts.SID != "" {
			if sig := opts.SessionStore.GetThoughtSig(opts.SID, msg.FunctionCall.Name); sig != "" {
				part.ThoughtSignature = sig
			} else {
				part.ThoughtSignature = "skip_thought_signature_validator"
			}
		}
		parts = append(parts, part)
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
				Response: normalizeToolResponse(extractTextContent(msg.Content)),
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

	// Empty candidates → error response (either blocked by safety or API error).
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
		// No block reason — likely a Gemini API error (e.g. 429 quota).
		var raw map[string]any
		if json.Unmarshal(body, &raw) == nil {
			if errVal, ok := raw["error"]; ok {
				msg, code := extractGeminiError(errVal)
				slog.Warn("Gemini API error in response", "msg", msg, "code", code)
				b, _ := json.Marshal(map[string]any{
					"error": map[string]any{
						"message": fmt.Sprintf("Gemini API error (HTTP %d): %s", code, msg),
						"type":    "server_error",
						"code":    code,
					},
				})
				return b, nil
			}
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

func convertOpenAIResponseToGemini(body []byte, opts *ConvertOptions) ([]byte, error) {
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
			part := GeminiPart{
				FunctionCall: &GeminiFunctionCall{
					Name: tc.Function.Name,
					Args: args,
				},
			}
			if opts != nil && opts.SessionStore != nil && opts.SID != "" {
				if sig := opts.SessionStore.GetThoughtSig(opts.SID, tc.Function.Name); sig != "" {
					part.ThoughtSignature = sig
				} else {
					part.ThoughtSignature = "skip_thought_signature_validator"
				}
			}
			parts = append(parts, part)
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

// ---------------------------------------------------------------------------
// Gemini NDJSON-array response (streamGenerateContent without alt=sse)
// ---------------------------------------------------------------------------

// parseJSONArray tries to parse body as a top-level JSON array of objects.
func parseJSONArray(body []byte) ([]map[string]any, bool) {
	var raw []map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, false
	}
	return raw, len(raw) > 0
}

// parseNDJSONChunks splits body into newline-separated JSON objects and
// returns them if they look like Gemini streaming chunks (first object has
// "candidates"). Returns nil for single-JSON bodies (handled by normal
// path) or non-Gemini content.
func parseNDJSONChunks(body []byte) []map[string]any {
	body = bytes.TrimSpace(body)
	if len(body) == 0 {
		return nil
	}
	lines := bytes.Split(body, []byte("\n"))
	var chunks []map[string]any
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal(line, &obj); err != nil {
			return nil
		}
		chunks = append(chunks, obj)
	}
	if len(chunks) < 2 {
		return nil
	}
	if _, ok := chunks[0]["candidates"]; !ok {
		return nil
	}
	return chunks
}

// convertGeminiNDJSONArray converts a JSON array of Gemini response chunks
// into the target client protocol response. Since Gemini streaming chunks
// are cumulative snapshots, we merge all parts and build a single response
// from the last chunk (which has the most complete state including
// finishReason). The merged text/functionCall parts are extracted from
// the first candidate of each chunk.
func convertGeminiNDJSONArray(chunks []map[string]any, opts *ConvertOptions) ([]byte, error) {
	if len(chunks) == 0 {
		return []byte(`{"candidates":[],"usageMetadata":{}}`), nil
	}
	// Determine target client protocol from session or model map.
	target := ProtocolOpenAIChat
	if opts != nil && opts.SessionStore != nil && opts.SID != "" {
		if sess := opts.SessionStore.Get(opts.SID); sess != nil {
			target = sess.From
		}
	}
	if target == ProtocolUnknown && opts != nil && opts.ModelMap != nil {
		_, target = resolveModel("", ProtocolGemini, opts.ModelMap)
		if target == ProtocolUnknown {
			target = ProtocolOpenAIChat
		}
	}

	// Check for Gemini API error (e.g. 429 quota exceeded) before processing chunks.
	for _, raw := range chunks {
		if errVal, ok := raw["error"]; ok {
			msg, code := extractGeminiError(errVal)
			slog.Warn("Gemini API error in response", "msg", msg, "code", code)
			return formatProviderError(target, code, msg)
		}
	}

	// Merge parts from each chunk into a single GeminiChatResponse.
	// Each chunk is cumulative, so we collect all unique text + functionCall
	// parts from the first candidate, taking the last chunk's finishReason.
	merged := GeminiChatResponse{}
	var seenTexts []string
	seenFns := make(map[string]bool)
	for _, raw := range chunks {
		var resp GeminiChatResponse
		b, _ := json.Marshal(raw)
		if err := json.Unmarshal(b, &resp); err != nil {
			continue
		}
		if len(resp.Candidates) == 0 {
			continue
		}
		c := resp.Candidates[0]
		for _, p := range c.Content.Parts {
			if p.Text != "" && p.FunctionCall == nil {
				seenTexts = append(seenTexts, p.Text)
			}
			if p.FunctionCall != nil && !seenFns[p.FunctionCall.Name] {
				seenFns[p.FunctionCall.Name] = true
				merged.Candidates = []GeminiCandidate{{Content: GeminiContent{Role: "model"}}}
				merged.Candidates[0].Content.Parts = append(merged.Candidates[0].Content.Parts, p)
			}
		}
		merged.UsageMetadata = resp.UsageMetadata
		if c.FinishReason != "" {
			merged.Candidates = []GeminiCandidate{{
				Content:      GeminiContent{Role: "model"},
				FinishReason: c.FinishReason,
				Index:        c.Index,
			}}
		}
	}
	// Assemble text parts into merged response.
	if len(merged.Candidates) == 0 {
		merged.Candidates = []GeminiCandidate{{Content: GeminiContent{Role: "model"}}}
	}
	// Concatenate text fragments from all chunks (Gemini stream chunks are
	// not cumulative — each chunk's text is a fragment that builds the response).
	var fullText string
	for _, t := range seenTexts {
		fullText += t
	}
	if fullText != "" {
		merged.Candidates[0].Content.Parts = append(
			[]GeminiPart{{Text: fullText}},
			merged.Candidates[0].Content.Parts...,
		)
	}
	if len(merged.Candidates[0].Content.Parts) == 0 && merged.Candidates[0].FinishReason == "" {
		// ponytail: empty response, return minimal structure
		merged.Candidates[0].Content.Parts = []GeminiPart{{Text: " "}}
	}

	b, err := json.Marshal(merged)
	if err != nil {
		return nil, err
	}

	// Return plain JSON for non-streaming response (Client didn't send stream:true).
	// SSE wrapping (anthropicResponseToSSE) only belongs in the streaming path.
	switch target {
	case ProtocolAnthropic:
		return convertGeminiResponseToAnthropic(b, opts)
	default:
		return convertGeminiResponseToOpenAI(b, opts)
	}
}

// sanitizeGeminiSchema recursively strips JSON Schema keywords that Gemini's
// functionDeclarations.parameters rejects: $schema, additionalProperties,
// propertyNames, const, exclusiveMinimum (numeric), and any_of (not supported).
// normalizeToolResponse ensures the tool result response is a JSON Object.
// Gemini rejects plain strings / arrays in function_response.response
// (type.googleapis.com/google.protobuf.Struct); wrap them in an object.
func normalizeToolResponse(v any) any {
	switch x := v.(type) {
	case string:
		return map[string]any{"output": x}
	case []any:
		return map[string]any{"output": x}
	default:
		return v
	}
}

// Gemini's schema model is an OpenAPI 3.0 subset; anything outside it fails.
func sanitizeGeminiSchema(v any) any {
	switch m := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(m))
		for k, val := range m {
			switch k {
			case "$schema", "additionalProperties", "propertyNames", "const":
				continue
			case "exclusiveMinimum":
				if _, ok := val.(float64); ok {
					continue // Gemini rejects numeric exclusiveMinimum
				}
				out[k] = sanitizeGeminiSchema(val)
			case "any_of":
				continue // Gemini doesn't support any_of
			default:
				out[k] = sanitizeGeminiSchema(val)
			}
		}
		return out
	case []any:
		out := make([]any, len(m))
		for i, val := range m {
			out[i] = sanitizeGeminiSchema(val)
		}
		return out
	default:
		return v
	}
}

// extractGeminiError extracts error message and code from a Gemini API error value.
// The error value is typically a map with "code", "message", and "status" keys.
func extractGeminiError(errVal any) (msg string, code int) {
	code = 500 // default
	switch m := errVal.(type) {
	case map[string]any:
		if c, ok := m["code"].(float64); ok {
			code = int(c)
		}
		if s, ok := m["message"].(string); ok {
			msg = s
		}
		if msg == "" {
			if s, ok := m["status"].(string); ok {
				msg = s
			}
		}
	default:
		msg = fmt.Sprintf("%v", errVal)
	}
	return
}

// formatProviderError formats an error as the target provider's error response format.
// Anthropic: {"type":"error","error":{"type":"api_error","message":"..."}}
// OpenAI: {"error":{"message":"...","type":"server_error","code":429}}
func formatProviderError(target Protocol, code int, msg string) ([]byte, error) {
	switch target {
	case ProtocolAnthropic:
		b, err := json.Marshal(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "api_error",
				"message": fmt.Sprintf("Gemini API error (HTTP %d): %s", code, msg),
			},
		})
		return b, err
	default: // OpenAI
		b, err := json.Marshal(map[string]any{
			"error": map[string]any{
				"message": fmt.Sprintf("Gemini API error (HTTP %d): %s", code, msg),
				"type":    "server_error",
				"code":    code,
			},
		})
		return b, err
	}
}
