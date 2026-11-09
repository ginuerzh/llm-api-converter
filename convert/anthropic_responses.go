package convert

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// convertAnthropicRequestToResponses converts an Anthropic Messages request
// body into an OpenAI Responses request body.
func convertAnthropicRequestToResponses(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("unmarshal Anthropic request: %w", err)
	}
	if _, ok := raw["messages"]; !ok {
		return body, nil
	}
	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("unmarshal Anthropic request: %w", err)
	}
	return anthropicRequestToResponses(&req, opts)
}

// anthropicRequestToResponses builds the Responses body. It is a pure function
// so the pair can later be replaced without touching the routing layer.
func anthropicRequestToResponses(req *AnthropicRequest, opts *ConvertOptions) ([]byte, error) {
	out := ResponsesRequest{
		Model:    requestTargetModel(req.Model, opts),
		Stream:   req.Stream != nil && *req.Stream,
		Store:    boolPtr(false),
		Metadata: req.Metadata,
	}
	if req.MaxTokens > 0 {
		maxTokens := req.MaxTokens
		out.MaxOutputTokens = &maxTokens
	} else if opts != nil && opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		out.MaxOutputTokens = &maxTokens
	}
	out.Temperature = req.Temperature
	out.TopP = req.TopP

	if system := anthropicSystemText(req.System); system != "" {
		b, err := json.Marshal(system)
		if err != nil {
			return nil, fmt.Errorf("marshal instructions: %w", err)
		}
		out.Instructions = b
	}

	input := anthropicMessagesToResponsesInput(req.Messages)
	b, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal input: %w", err)
	}
	out.Input = b

	for _, t := range req.Tools {
		schema := t.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out.Tools = append(out.Tools, ResponsesTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  schema,
		})
	}

	if req.ToolChoice != nil {
		switch req.ToolChoice.Type {
		case "auto", "none":
			out.ToolChoice = req.ToolChoice.Type
		case "any":
			out.ToolChoice = "required"
		case "tool":
			out.ToolChoice = map[string]any{"type": "function", "name": req.ToolChoice.Name}
		}
		if req.ToolChoice.DisableParallelToolUse != nil && *req.ToolChoice.DisableParallelToolUse {
			out.ParallelToolCalls = boolPtr(false)
		}
	}

	out.Reasoning = responsesReasoningFromThinking(req.Thinking, req.OutputConfig)
	out.Text = responsesTextFromAnthropic(req.OutputConfig)

	if len(req.StopSequences) > 0 {
		slog.Debug("responses: dropping stop_sequences", "count", len(req.StopSequences))
	}
	if req.TopK != nil {
		slog.Debug("responses: dropping top_k", "top_k", *req.TopK)
	}

	return json.Marshal(out)
}

func requestTargetModel(model string, opts *ConvertOptions) string {
	if opts == nil {
		return model
	}
	if opts.ResolvedModel != "" {
		return opts.ResolvedModel
	}
	if target := resolveModelTarget(model, opts.ModelMap); target != "" {
		return target
	}
	return model
}

func anthropicSystemText(blocks []AnthropicTextBlock) string {
	var parts []string
	for _, b := range blocks {
		if strings.TrimSpace(b.Text) != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n")
}

func anthropicMessagesToResponsesInput(messages []AnthropicMessage) []any {
	input := make([]any, 0, len(messages))
	for _, m := range messages {
		role := m.Role
		if role != "assistant" {
			role = "user"
		}
		var content []map[string]any
		flush := func() {
			if len(content) == 0 {
				return
			}
			input = append(input, map[string]any{
				"type":    "message",
				"role":    role,
				"content": content,
			})
			content = nil
		}
		for _, block := range m.Content {
			switch block.Type {
			case "text":
				if block.Text == "" {
					continue
				}
				typ := "input_text"
				if role == "assistant" {
					typ = "output_text"
				}
				content = append(content, map[string]any{"type": typ, "text": block.Text})
			case "image":
				if role == "assistant" || block.Source == nil {
					continue
				}
				if url := anthropicImageURL(block.Source); url != "" {
					content = append(content, map[string]any{
						"type":      "input_image",
						"detail":    "auto",
						"image_url": url,
					})
				}
			case "tool_use":
				if role != "assistant" {
					continue
				}
				flush()
				input = append(input, map[string]any{
					"type":      "function_call",
					"call_id":   block.ID,
					"name":      block.Name,
					"arguments": toolInputJSON(block.Input),
				})
			case "tool_result":
				flush()
				input = append(input, map[string]any{
					"type":    "function_call_output",
					"call_id": block.ToolUseID,
					"output":  toolResultOutput(block.Content),
				})
			case "thinking", "redacted_thinking":
				slog.Debug("responses: dropping Anthropic thinking block")
			}
		}
		flush()
	}
	return input
}

func anthropicImageURL(src *AnthropicImageSource) string {
	if src == nil {
		return ""
	}
	switch src.Type {
	case "base64":
		if src.MediaType == "" || src.Data == "" {
			return ""
		}
		return "data:" + src.MediaType + ";base64," + src.Data
	case "url":
		return src.URL
	}
	return ""
}

func toolInputJSON(input any) string {
	if input == nil {
		return "{}"
	}
	b, err := json.Marshal(input)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func toolResultOutput(content any) any {
	switch v := content.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		var parts []map[string]any
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch m["type"] {
			case "text":
				if text, _ := m["text"].(string); text != "" {
					parts = append(parts, map[string]any{"type": "input_text", "text": text})
				}
			case "image":
				if src, ok := m["source"].(map[string]any); ok {
					if url := imageURLFromMap(src); url != "" {
						parts = append(parts, map[string]any{
							"type":      "input_image",
							"detail":    "auto",
							"image_url": url,
						})
					}
				}
			}
		}
		if len(parts) == 1 && parts[0]["type"] == "input_text" {
			return parts[0]["text"]
		}
		return parts
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

func imageURLFromMap(src map[string]any) string {
	switch typ, _ := src["type"].(string); typ {
	case "base64":
		mediaType, _ := src["media_type"].(string)
		data, _ := src["data"].(string)
		if mediaType == "" || data == "" {
			return ""
		}
		return "data:" + mediaType + ";base64," + data
	case "url":
		url, _ := src["url"].(string)
		return url
	}
	return ""
}

func responsesReasoningFromThinking(th *AnthropicThinking, oc *AnthropicOutputConfig) *ResponsesReasoning {
	if th == nil || th.Type == "disabled" || (th.Type != "enabled" && th.Type != "adaptive") {
		return nil
	}
	effort := ""
	if oc != nil {
		effort = responsesEffort(oc.Effort)
	}
	if effort == "" {
		effort = responsesEffortFromBudget(th.BudgetTokens)
	}
	if effort == "" {
		effort = "medium"
	}
	return &ResponsesReasoning{Effort: effort, Summary: "auto"}
}

func responsesEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal", "low":
		return "low"
	case "medium":
		return "medium"
	case "high":
		return "high"
	case "xhigh", "max":
		return "high"
	}
	return ""
}

func responsesEffortFromBudget(n int) string {
	switch {
	case n <= 0:
		return ""
	case n <= 4096:
		return "low"
	case n <= 12000:
		return "medium"
	case n <= 24000:
		return "high"
	default:
		return "high"
	}
}

func responsesTextFromAnthropic(oc *AnthropicOutputConfig) *ResponsesText {
	if oc == nil || oc.Format == nil || oc.Format.Type != "json_schema" || oc.Format.Schema == nil {
		return nil
	}
	return &ResponsesText{Format: map[string]any{
		"type":   "json_schema",
		"name":   "response",
		"schema": oc.Format.Schema,
	}}
}

// anthropicOutBlock is the Anthropic content block used by the translated
// response. It differs from AnthropicContent only in being able to emit an
// explicit empty signature on thinking blocks.
type anthropicOutBlock struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	Thinking  string         `json:"thinking,omitempty"`
	Signature *string        `json:"signature,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
}

type anthropicTranslatedResponse struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Role         string              `json:"role"`
	Content      []anthropicOutBlock `json:"content"`
	Model        string              `json:"model"`
	StopReason   *string             `json:"stop_reason"`
	StopSequence *string             `json:"stop_sequence"`
	Usage        AnthropicUsage      `json:"usage"`
}

// convertResponsesResponseToAnthropic converts a Responses reply body into an
// Anthropic Messages reply body.
func convertResponsesResponseToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("unmarshal Responses reply: %w", err)
	}
	var object string
	_ = json.Unmarshal(raw["object"], &object)
	if object != "response" {
		return body, nil
	}
	var resp ResponsesResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal Responses reply: %w", err)
	}
	return responsesResponseToAnthropic(&resp, opts)
}

func responsesResponseToAnthropic(resp *ResponsesResponse, opts *ConvertOptions) ([]byte, error) {
	content := make([]anthropicOutBlock, 0, len(resp.Output))
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					content = append(content, anthropicOutBlock{Type: "text", Text: part.Text})
				}
			}
		case "reasoning":
			if text := responsesReasoningText(item); text != "" {
				sig := ""
				content = append(content, anthropicOutBlock{Type: "thinking", Thinking: text, Signature: &sig})
			}
		case "function_call":
			id := item.CallID
			if id == "" {
				id = item.ID
			}
			content = append(content, anthropicOutBlock{
				Type:  "tool_use",
				ID:    id,
				Name:  item.Name,
				Input: parseResponsesArguments(item.Arguments),
			})
		default:
			slog.Debug("responses: skipping unsupported output item", "type", item.Type)
		}
	}

	stop := responsesStopReason(resp)
	model := resp.Model
	if opts != nil {
		model = resolveResponseModel(resp.Model, opts)
	}
	out := anthropicTranslatedResponse{
		ID:           resp.ID,
		Type:         "message",
		Role:         "assistant",
		Content:      content,
		Model:        model,
		StopReason:   &stop,
		StopSequence: nil,
		Usage:        anthropicUsageFromResponses(resp.Usage),
	}
	return json.Marshal(out)
}

func responsesReasoningText(item ResponsesOutputItem) string {
	var parts []string
	for _, part := range item.Content {
		if part.Type == "reasoning_text" && part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	if len(parts) == 0 {
		for _, summary := range item.Summary {
			if summary.Text != "" {
				parts = append(parts, summary.Text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func responsesStopReason(resp *ResponsesResponse) string {
	if resp.Status == "incomplete" {
		if details, ok := resp.IncompleteDetails.(map[string]any); ok {
			switch details["reason"] {
			case "max_output_tokens":
				return "max_tokens"
			case "content_filter":
				return "refusal"
			}
		}
	}
	for _, item := range resp.Output {
		if item.Type == "function_call" {
			return "tool_use"
		}
	}
	return "end_turn"
}

func anthropicUsageFromResponses(u *ResponsesUsage) AnthropicUsage {
	if u == nil {
		return AnthropicUsage{}
	}
	cached, cacheWrite := 0, 0
	if u.InputTokensDetails != nil {
		cached = u.InputTokensDetails.CachedTokens
		cacheWrite = u.InputTokensDetails.CacheWriteTokens
	}
	input := u.InputTokens - cached - cacheWrite
	if input < 0 {
		input = 0
	}
	return AnthropicUsage{
		InputTokens:              input,
		OutputTokens:             u.OutputTokens,
		CacheCreationInputTokens: cacheWrite,
		CacheReadInputTokens:     cached,
	}
}

func parseResponsesArguments(arguments string) map[string]any {
	out := map[string]any{}
	if strings.TrimSpace(arguments) == "" {
		return out
	}
	if err := json.Unmarshal([]byte(arguments), &out); err != nil || out == nil {
		return map[string]any{}
	}
	return out
}
