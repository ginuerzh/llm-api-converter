package convert

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// -------- Responses API (OpenAI /v1/responses) --------

// ResponsesRequest is a POST /v1/responses request body.
type ResponsesRequest struct {
	Model             string          `json:"model"`
	Instructions      json.RawMessage `json:"instructions,omitempty"`
	Input             json.RawMessage `json:"input,omitempty"` // string or []inputItem
	MaxOutputTokens   *int            `json:"max_output_tokens,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	StreamOptions     any             `json:"stream_options,omitempty"`
	Tools             []ResponsesTool `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID string        `json:"previous_response_id,omitempty"`
	Store              *bool          `json:"store,omitempty"`
	Truncation         string         `json:"truncation,omitempty"`
	Metadata           any            `json:"metadata,omitempty"`
}

// responsesInputItem is an element in the input array.
type responsesInputItem struct {
	Type      string          `json:"type"` // "message", "function_call", "function_call_output", "reasoning"
	Role      string          `json:"role,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // string or []contentPart
	ID        string          `json:"id,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Input     string          `json:"input,omitempty"`
	Output    json.RawMessage `json:"output,omitempty"`
	Summary   []ResponsesReasoningSummary `json:"summary,omitempty"`
	Text      json.RawMessage `json:"text,omitempty"`
}

// ResponsesOutputItem is an element in the output array.
type ResponsesOutputItem struct {
	ID        string                     `json:"id"`
	Type      string                     `json:"type"` // "message", "reasoning", "function_call"
	Status    string                     `json:"status,omitempty"`
	Role      string                     `json:"role,omitempty"`
	Content   []ResponsesContentPart     `json:"content,omitempty"`
	Summary   []ResponsesReasoningSummary `json:"summary,omitempty"`
	CallID    string                     `json:"call_id,omitempty"`
	Name      string                     `json:"name,omitempty"`
	Arguments string                     `json:"arguments,omitempty"`
}

// ResponsesContentPart is a typed content part inside an output item.
type ResponsesContentPart struct {
	Type        string `json:"type"` // "output_text"
	Text        string `json:"text,omitempty"`
	Annotations []any  `json:"annotations,omitempty"`
}

// ResponsesReasoningSummary is a summary inside a reasoning output item.
type ResponsesReasoningSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

// ResponsesTool represents a tool declaration in a Responses API request.
type ResponsesTool struct {
	Name        string                `json:"name,omitempty"`
	Description string                `json:"description,omitempty"`
	Parameters  any                   `json:"parameters,omitempty"`
	Strict      *bool                 `json:"strict,omitempty"`
	Type        string                `json:"type,omitempty"`       // "function" (OpenAI format)
	Function    *ResponsesToolFunction `json:"function,omitempty"` // alternative wrapping
}

// ResponsesToolFunction is the function wrapper inside an OpenAI-style tool definition.
type ResponsesToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// ResponsesResponse is the full /v1/responses response body.
type ResponsesResponse struct {
	ID                string                `json:"id"`
	Object            string                `json:"object"` // "response"
	CreatedAt         int64                 `json:"created_at"`
	CompletedAt       *int64                `json:"completed_at,omitempty"`
	Status            string                `json:"status"` // "completed", "incomplete", "failed", "cancelled"
	Error             any                   `json:"error,omitempty"`
	IncompleteDetails any                   `json:"incomplete_details,omitempty"`
	Model             string                `json:"model"`
	Output            []ResponsesOutputItem `json:"output"`
	ParallelToolCalls *bool                 `json:"parallel_tool_calls,omitempty"`
	Reasoning         map[string]any        `json:"reasoning,omitempty"`
	Store             *bool                 `json:"store,omitempty"`
	Temperature       *float64              `json:"temperature,omitempty"`
	Text              map[string]any        `json:"text,omitempty"`
	ToolChoice        any                   `json:"tool_choice,omitempty"`
	Tools             []any                 `json:"tools,omitempty"`
	TopP              *float64              `json:"top_p,omitempty"`
	Truncation        string                `json:"truncation,omitempty"`
	Usage             *ResponsesUsage       `json:"usage,omitempty"`
	Metadata          map[string]any        `json:"metadata,omitempty"`
}

// ResponsesUsage represents token usage in a Responses API response.
type ResponsesUsage struct {
	InputTokens          int                   `json:"input_tokens"`
	OutputTokens         int                   `json:"output_tokens"`
	TotalTokens          int                   `json:"total_tokens"`
	InputTokensDetails   *ResponsesUsageDetails `json:"input_tokens_details,omitempty"`
	OutputTokensDetails  *ResponsesUsageDetails `json:"output_tokens_details,omitempty"`
}

// ResponsesUsageDetails holds breakdown details for token usage.
type ResponsesUsageDetails struct {
	CachedTokens    int `json:"cached_tokens,omitempty"`
	TextTokens      int `json:"text_tokens,omitempty"`
	AudioTokens     int `json:"audio_tokens,omitempty"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// -------- Responses API Streaming Types --------

// ResponsesStreamEventType enumerates Responses API SSE event types.
type ResponsesStreamEventType string

const (
	ResponseCreated                ResponsesStreamEventType = "response.created"
	ResponseInProgress             ResponsesStreamEventType = "response.in_progress"
	ResponseCompleted              ResponsesStreamEventType = "response.completed"
	ResponseOutputItemAdded        ResponsesStreamEventType = "response.output_item.added"
	ResponseOutputItemDone         ResponsesStreamEventType = "response.output_item.done"
	ResponseContentPartAdded       ResponsesStreamEventType = "response.content_part.added"
	ResponseContentPartDone        ResponsesStreamEventType = "response.content_part.done"
	ResponseOutputTextDelta        ResponsesStreamEventType = "response.output_text.delta"
	ResponseOutputTextDone         ResponsesStreamEventType = "response.output_text.done"
	ResponseReasoningSummaryPartAdded ResponsesStreamEventType = "response.reasoning_summary_part.added"
	ResponseReasoningSummaryDelta     ResponsesStreamEventType = "response.reasoning_summary_text.delta"
	ResponseReasoningSummaryDone      ResponsesStreamEventType = "response.reasoning_summary_text.done"
	ResponseReasoningSummaryPartDone  ResponsesStreamEventType = "response.reasoning_summary_part.done"
	ResponseFunctionCallArgumentsDelta  ResponsesStreamEventType = "response.function_call_arguments.delta"
	ResponseFunctionCallArgumentsDone    ResponsesStreamEventType = "response.function_call_arguments.done"
	ResponseIncomplete                ResponsesStreamEventType = "response.incomplete"
	ResponseFailed                   ResponsesStreamEventType = "response.failed"
)

// ResponsesStreamEvent is a generic Responses API stream event envelope.
type ResponsesStreamEvent struct {
	Type           ResponsesStreamEventType `json:"type"`
	SequenceNumber *int                     `json:"sequence_number,omitempty"`
	Response     *ResponsesResponse       `json:"response,omitempty"`
	Item         *ResponsesOutputItem     `json:"item,omitempty"`
	OutputIndex  *int                     `json:"output_index,omitempty"`
	ContentIndex *int                     `json:"content_index,omitempty"`
	PartIndex    *int                     `json:"part_index,omitempty"`
	ItemID       string                   `json:"item_id,omitempty"`
	SummaryIndex *int                     `json:"summary_index,omitempty"`
	Part         any                      `json:"part,omitempty"`
	Delta        string                   `json:"delta,omitempty"`
	TextDone     string                   `json:"text,omitempty"`
	Arguments    string                   `json:"arguments,omitempty"`
	SummaryText  string                   `json:"summary_text,omitempty"`
}

// -------- Detection --------

// isResponsesRequest detects an OpenAI Responses API request body.
// Key signals: has "input" field, does NOT have "messages".
func isResponsesRequest(m map[string]any) bool {
	if _, ok := m["messages"]; ok {
		return false
	}
	if _, ok := m["input"]; ok {
		return true
	}
	return false
}

// isResponsesResponse detects an OpenAI Responses API response body.
// Key signals: object is "response" or has "output" array of typed items.
func isResponsesResponse(m map[string]any) bool {
	if obj, ok := m["object"].(string); ok && obj == "response" {
		return true
	}
	if output, ok := m["output"].([]any); ok && len(output) > 0 {
		if first, ok := output[0].(map[string]any); ok {
			if t, _ := first["type"].(string); t != "" {
				return true
			}
		}
	}
	// Empty output array is still a Responses response if object is present.
	if _, ok := m["output"]; ok {
		if _, ok := m["object"]; ok {
			return true
		}
	}
	return false
}

// -------- Request Conversion: Responses → Chat --------

// ConvertResponsesToChat converts a Responses API request body to an OpenAI Chat request.
func ConvertResponsesToChat(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("unmarshal Responses request: %w", err)
	}

	model, _ := resolveModel(req.Model, opts.Model, opts.ModelMap)
	chat := OpenAIChatRequest{
		Model: model,
		Stream: &req.Stream,
	}
	if req.MaxOutputTokens != nil {
		chat.MaxTokens = req.MaxOutputTokens
	} else if opts.MaxTokens > 0 {
		maxTokens := opts.MaxTokens
		chat.MaxTokens = &maxTokens
	}
	if req.Temperature != nil {
		chat.Temperature = req.Temperature
	}
	if req.TopP != nil {
		chat.TopP = req.TopP
	}
	if req.Stream {
		chat.StreamOptions = map[string]any{"include_usage": true}
	}
	if req.Metadata != nil {
		chat.Metadata = req.Metadata
	}

	// Instructions → system message.
	result := parseResponsesInput(req.Input)
	if len(req.Instructions) > 0 {
		var instrText string
		if err := json.Unmarshal(req.Instructions, &instrText); err == nil && instrText != "" {
			instrText = stripLeadingAnthropicBillingHeader(instrText)
				result.systemTexts = append(result.systemTexts, instrText)
		}
	}
	if len(result.systemTexts) > 0 {
		chat.Messages = append(chat.Messages, OpenAIMessage{
			Role:    "system",
			Content: strings.Join(result.systemTexts, "\n"),
		})
	}
	chat.Messages = append(chat.Messages, result.messages...)
	chat.Messages = collapseSystemMessagesToHead(chat.Messages)

	// Tools.
	if len(req.Tools) > 0 {
		chat.Tools = make([]OpenAITool, 0, len(req.Tools))
		for _, t := range req.Tools {
			name, desc, params := extractToolParts(t)
			if name == "" {
				// Non-function tool (e.g. web_search_preview)
				// — cannot be represented in Chat Completions format.
				slog.Debug("skipping non-function Responses tool", "type", t.Type)
				continue
			}
				// ponytail: custom tools get a single "input" string param schema
				if t.Type == "custom" {
					params = customToolInputSchema()
				}
			chat.Tools = append(chat.Tools, OpenAITool{
				Type: "function",
				Function: OpenAIFunction{
					Name:        name,
					Description: desc,
					Parameters:  ensureObjectSchema(params),
				},
			})
		}
		sortOpenAITools(chat.Tools)
		chat.ToolChoice = mapResponsesToolChoice(req.ToolChoice)
	}

	if opts != nil {
		opts.CodexToolContext = buildCodexToolContext(req.Tools)
	}
	return json.Marshal(chat)
}

// -------- Request Conversion: Responses → Anthropic --------

// ConvertResponsesToAnthropic converts a Responses API request body to an Anthropic request.
func ConvertResponsesToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("unmarshal Responses request: %w", err)
	}

	model, _ := resolveModel(req.Model, opts.Model, opts.ModelMap)
	maxTokens := 4096
	if req.MaxOutputTokens != nil {
		maxTokens = *req.MaxOutputTokens
	} else if opts.MaxTokens > 0 {
		maxTokens = opts.MaxTokens
	}

	anth := AnthropicRequest{
		Model:     model,
		MaxTokens: maxTokens,
		Stream:    &req.Stream,
	}
	if req.Temperature != nil {
		anth.Temperature = req.Temperature
	}
	if req.TopP != nil {
		anth.TopP = req.TopP
	}
	if req.Metadata != nil {
		anth.Metadata = req.Metadata
	}

	// Parse input + instructions → Anthropic messages + system.
	result := parseResponsesInput(req.Input)
	if len(req.Instructions) > 0 {
		var instrText string
		if err := json.Unmarshal(req.Instructions, &instrText); err == nil && instrText != "" {
				instrText = stripLeadingAnthropicBillingHeader(instrText)
				result.systemTexts = append(result.systemTexts, instrText)
		}
	}
	if len(result.systemTexts) > 0 {
		anth.System = []AnthropicTextBlock{
			{Type: "text", Text: strings.Join(result.systemTexts, "\n")},
		}
	}

	// Convert OpenAI messages → Anthropic messages.
	for _, msg := range result.messages {
		anth.Messages = append(anth.Messages, convertChatMessageToAnthropic(msg))
	}

	// Tools.
	if len(req.Tools) > 0 {
		anth.Tools = make([]AnthropicTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			name, desc, params := extractToolParts(t)
			if name == "" {
				slog.Debug("skipping non-function Responses tool", "type", t.Type)
				continue
			}
			anth.Tools = append(anth.Tools, AnthropicTool{
				Name:        name,
				Description: desc,
				InputSchema: ensureObjectSchema(params),
			})
		}
		sortAnthropicTools(anth.Tools)
		if req.ToolChoice != nil {
			anth.ToolChoice = mapResponsesToolChoiceAnthropic(req.ToolChoice)
		}
	} else {
		anth.ToolChoice = &AnthropicToolChoice{Type: "none"}
	}

	// Edge case: Anthropic requires at least one message.
	if len(anth.Messages) == 0 {
		anth.Messages = []AnthropicMessage{
			{Role: "user", Content: []AnthropicContent{{Type: "text", Text: "..."}}},
		}
	}

	return json.Marshal(anth)
}

// -------- Response Conversion: Chat → Responses --------

// ConvertChatToResponses converts an OpenAI Chat response body to Responses API format.
func ConvertChatToResponses(body []byte, opts *ConvertOptions) ([]byte, error) {
	var resp OpenAIChatResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal OpenAI Chat response: %w", err)
	}

	r := newResponsesResponse()
	r.ID = "resp_" + resp.ID
	if rest, ok := strings.CutPrefix(resp.ID, "chatcmpl-"); ok {
		r.ID = "resp_" + rest
	}
	r.CreatedAt = resp.Created
	r.Model = resolveResponseModel(resp.Model, opts)

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		r.Status = responsesFinishStatus(choice.FinishReason)

		msg := choice.Message
		if msg.ReasoningContent != "" {
			r.Output = append(r.Output, completedReasoningItem(msg.ReasoningContent))
		}
		if text := extractTextContent(msg.Content); text != "" {
			r.Output = append(r.Output, completedMessageItem(text, r.ID))
		}
		for _, tc := range msg.ToolCalls {
			r.Output = append(r.Output, completedFunctionCallItem(tc))
		}

		// Incomplete details.
		if choice.FinishReason != nil && *choice.FinishReason == "length" {
			r.IncompleteDetails = map[string]string{"reason": "max_output_tokens"}
		}
	}

	r.Usage = chatUsageToResponsesUsage(&resp, body)

	return json.Marshal(r)
}

// -------- Response Conversion: Anthropic → Responses --------

// ConvertAnthropicToResponses converts an Anthropic response body to Responses API format.
func ConvertAnthropicToResponses(body []byte, opts *ConvertOptions) ([]byte, error) {
	var resp AnthropicResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal Anthropic response: %w", err)
	}

	r := newResponsesResponse()
	r.ID = "resp_" + resp.ID
	r.CreatedAt = 0 // Anthropic doesn't provide a creation timestamp.
	r.Model = resolveResponseModel(resp.Model, opts)

	// Map stop reason to status.
	if resp.StopReason != nil {
		switch *resp.StopReason {
		case "end_turn", "stop_sequence":
			r.Status = "completed"
		case "max_tokens":
			r.Status = "incomplete"
			r.IncompleteDetails = map[string]string{"reason": "max_output_tokens"}
		case "tool_use":
			r.Status = "completed"
		default:
			r.Status = "completed"
		}
	} else {
		r.Status = "completed"
	}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				r.Output = append(r.Output, completedMessageItem(block.Text, r.ID))
			}
		case "thinking":
			if block.Thinking != "" {
				r.Output = append(r.Output, completedReasoningItem(block.Thinking))
			}
		case "tool_use":
			inputStr := ""
			if block.Input != nil {
				if b, err := json.Marshal(block.Input); err == nil {
					inputStr = string(b)
				}
			}
			r.Output = append(r.Output, ResponsesOutputItem{
				ID:        block.ID,
				Type:      "function_call",
				Status:    "completed",
				CallID:    block.ID,
				Name:      block.Name,
				Arguments: inputStr,
			})
		}
	}

	r.Usage = &ResponsesUsage{
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
		OutputTokensDetails: &ResponsesUsageDetails{
			TextTokens: resp.Usage.OutputTokens,
		},
	}
	return json.Marshal(r)
}

// -------- Helpers --------

// parseResponsesInput parses the Responses API input field (string or []inputItem)
// into OpenAI messages.
func parseResponsesInput(input json.RawMessage) responsesInputParseResult {
	var result responsesInputParseResult
	if len(input) == 0 || string(input) == "null" {
		return result
	}

	// Try string first.
	var text string
	if err := json.Unmarshal(input, &text); err == nil {
		if text != "" {
			result.messages = append(result.messages, OpenAIMessage{
				Role:    "user",
				Content: text,
			})
		}
		return result
	}

	// Parse as array of input items.
	var items []responsesInputItem
	if err := json.Unmarshal(input, &items); err != nil {
		slog.Debug("responses: failed to parse input array, treating as empty", "err", err)
		return result
	}

	// Track pending reasoning for the next assistant message.
	var pendingReasoning string

	for _, item := range items {
		switch item.Type {
		case "message":
			role := item.Role
			if role != "user" && role != "assistant" && role != "system" && role != "developer" {
				role = "user"
			}

			// Developer/system → system texts.
			if role == "developer" || role == "system" {
				systemText := responsesRawText(item.Content)
				if systemText != "" {
					result.systemTexts = append(result.systemTexts, systemText)
				}
				continue
			}

			msg := OpenAIMessage{Role: role}
			msg.Content = responsesContentToOpenAI(item.Content)
			if role == "assistant" && pendingReasoning != "" {
				msg.ReasoningContent = pendingReasoning
				pendingReasoning = ""
			}
			result.messages = append(result.messages, msg)

		case "function_call":
			// Convert to assistant message with tool_calls.
			args := item.Arguments
			if args == "" {
				args = "{}"
			}
			toolID := item.ID
			if toolID == "" {
				toolID = ensureToolID("")
			}

			msg := OpenAIMessage{
				Role: "assistant",
				ToolCalls: []OpenAIToolCall{{
					ID:   toolID,
					Type: "function",
					Function: OpenAIFunctionCall{
						Name:      item.Name,
						Arguments: canonicalJSONString(args),
					},
				}},
			}
			if pendingReasoning != "" {
				msg.ReasoningContent = pendingReasoning
				pendingReasoning = ""
			}
			result.messages = append(result.messages, msg)

		case "function_call_output":
			content := ""
			if len(item.Output) > 0 {
				content = string(item.Output)
				// Try to extract as plain string first.
				var s string
				if err := json.Unmarshal(item.Output, &s); err == nil {
					content = s
				}
			}
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			if callID == "" {
				callID = "call_" + randHex(24)
			}
			result.messages = append(result.messages, OpenAIMessage{
				Role:       "tool",
				ToolCallID: callID,
				Content:    content,
			})

		case "custom_tool_call":
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			if callID == "" {
				callID = ensureToolID("")
			}
			inputStr := item.Input
			if inputStr == "" {
				inputStr = "{}"
			}
			msg := OpenAIMessage{
				Role: "assistant",
				ToolCalls: []OpenAIToolCall{{
					ID:   callID,
					Type: "function",
					Function: OpenAIFunctionCall{
						Name:      item.Name,
						Arguments: canonicalJSONString(inputStr),
					},
				}},
			}
			if pendingReasoning != "" {
				msg.ReasoningContent = pendingReasoning
				pendingReasoning = ""
			}
			result.messages = append(result.messages, msg)

		case "custom_tool_call_output":
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			if callID == "" {
				callID = "call_" + randHex(24)
			}
			content := ""
			if len(item.Output) > 0 {
				content = string(item.Output)
				var s string
				if err := json.Unmarshal(item.Output, &s); err == nil {
					content = s
				}
			}
			result.messages = append(result.messages, OpenAIMessage{
				Role:       "tool",
				ToolCallID: callID,
				Content:    content,
			})

		case "reasoning":
			textContent := responsesRawText(item.Text)
			if textContent != "" {
				pendingReasoning = appendDedupReasoning(pendingReasoning, textContent)
			}
			for _, s := range item.Summary {
				if s.Text != "" {
					pendingReasoning = appendDedupReasoning(pendingReasoning, s.Text)
				}
			}
		}
	}

	// Trailing reasoning → prior assistant: attach to last assistant message.
	if pendingReasoning != "" && len(result.messages) > 0 {
		last := &result.messages[len(result.messages)-1]
		if last.Role == "assistant" && last.ReasoningContent == "" {
			last.ReasoningContent = pendingReasoning
			pendingReasoning = ""
		}
	}

	// Backfill tool call reasoning placeholders: assistant messages with
	// tool_calls but empty reasoning get "tool call". Required by
	// kimi/DeepSeek models.
	for i := range result.messages {
		if result.messages[i].Role == "assistant" &&
			len(result.messages[i].ToolCalls) > 0 &&
			result.messages[i].ReasoningContent == "" {
			result.messages[i].ReasoningContent = "tool call"
		}
	}

	return result
}

// convertChatMessageToAnthropic converts an OpenAI message to Anthropic format.
func convertChatMessageToAnthropic(msg OpenAIMessage) AnthropicMessage {
	role := msg.Role
	if role != "user" && role != "assistant" {
		role = "user"
	}

	am := AnthropicMessage{Role: role}

	switch msg.Role {
	case "user":
		am.Content = convertUserContent(msg.Content)
	case "assistant":
		am.Content = convertAssistantContent(msg)
	case "tool":
		am.Content = convertToolResultContent(msg)
	default:
		am.Content = []AnthropicContent{{Type: "text", Text: fmt.Sprintf("%v", msg.Content)}}
	}

	return am
}

// responsesContentToOpenAI converts Responses API content to OpenAI format.
// Returns a plain string for simple text content, or []OpenAIContentPart for
// mixed content (e.g. text + images).
func responsesContentToOpenAI(content json.RawMessage) any {
	if len(content) == 0 || string(content) == "null" {
		return ""
	}
	// Try string first.
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return s
	}
	// Try array of content parts.
	var parts []map[string]any
	if err := json.Unmarshal(content, &parts); err != nil {
		return ""
	}
	// Check if all parts are text (can return as simple string).
	allText := true
	for _, p := range parts {
		t, _ := p["type"].(string)
		if t != "text" && t != "input_text" && t != "output_text" {
			allText = false
			break
		}
	}
	if allText {
		var texts []string
		for _, p := range parts {
			if txt, _ := p["text"].(string); txt != "" {
				texts = append(texts, txt)
			}
		}
		return strings.Join(texts, "\n")
	}
	// Mixed content: convert to OpenAI content parts.
	converted := make([]OpenAIContentPart, 0, len(parts))
	for _, p := range parts {
		t, _ := p["type"].(string)
		switch {
		case t == "text" || t == "input_text" || t == "output_text":
			txt, _ := p["text"].(string)
			converted = append(converted, OpenAIContentPart{Type: "text", Text: txt})
		case t == "input_image":
			if imageURL, _ := p["image_url"].(string); imageURL != "" {
				converted = append(converted, OpenAIContentPart{
					Type:     "image_url",
					ImageURL: &OpenAIImageURL{URL: imageURL},
				})
			}
			case t == "input_file":
				converted = append(converted, OpenAIContentPart{
					Type: "text",
					Text: fmt.Sprintf("[input_file: %v]", p["input_file"]),
				})
			case t == "input_audio":
				converted = append(converted, OpenAIContentPart{
					Type: "text",
					Text: fmt.Sprintf("[input_audio: %v]", p["input_audio"]),
				})
		}
	}
	if len(converted) == 1 && converted[0].Type == "text" {
		return converted[0].Text // unwrap single text
	}
	return converted
}

// responsesRawText extracts plain text from content JSON (string or array).
func responsesRawText(content json.RawMessage) string {
	if len(content) == 0 || string(content) == "null" {
		return ""
	}

	// Try string.
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return s
	}

	// Try array of content parts.
	var parts []map[string]any
	if err := json.Unmarshal(content, &parts); err == nil {
		var texts []string
		for _, p := range parts {
			if t, ok := p["type"].(string); ok && (t == "text" || t == "input_text" || t == "output_text") {
				if txt, _ := p["text"].(string); txt != "" {
					texts = append(texts, txt)
				}
			}
		}
		return strings.Join(texts, "\n")
	}

	return ""
}

// extractToolParts extracts name, description, and parameters from a ResponsesTool.
func extractToolParts(t ResponsesTool) (string, string, any) {
	if t.Name != "" {
		return t.Name, t.Description, t.Parameters
	}
	if t.Function != nil {
		return t.Function.Name, t.Function.Description, t.Function.Parameters
	}
	return "", "", nil
}

// mapResponsesToolChoice maps Responses API tool_choice to Chat tool_choice.
func mapResponsesToolChoice(tc any) any {
	if tc == nil {
		return nil
	}
	switch v := tc.(type) {
	case string:
		switch v {
		case "auto", "required":
			return v
		case "none":
			return "none"
		default:
			return "auto"
		}
	case map[string]any:
		if t, _ := v["type"].(string); t == "function" {
			if fn, ok := v["function"].(map[string]any); ok {
				if name, _ := fn["name"].(string); name != "" {
					return map[string]any{
						"type": "function",
						"function": map[string]any{
							"name": name,
						},
					}
				}
			}
		}
	}
	return "auto"
}

// mapResponsesToolChoiceAnthropic maps Responses API tool_choice to Anthropic tool_choice.
func mapResponsesToolChoiceAnthropic(tc any) *AnthropicToolChoice {
	if tc == nil {
		return nil
	}
	switch v := tc.(type) {
	case string:
		switch v {
		case "auto":
			return &AnthropicToolChoice{Type: "auto"}
		case "none":
			return &AnthropicToolChoice{Type: "none"}
		case "required":
			return &AnthropicToolChoice{Type: "any"}
		default:
			return &AnthropicToolChoice{Type: "auto"}
		}
	case map[string]any:
		if t, _ := v["type"].(string); t == "function" {
			if fn, ok := v["function"].(map[string]any); ok {
				if name, _ := fn["name"].(string); name != "" {
					return &AnthropicToolChoice{Type: "tool", Name: name}
				}
			}
		}
		if t, _ := v["type"].(string); t == "any" || t == "tool" {
			name, _ := v["name"].(string)
			return &AnthropicToolChoice{Type: t, Name: name}
		}
	}
	return &AnthropicToolChoice{Type: "auto"}
}

// responsesFinishStatus maps Chat finish_reason to Responses API status.
func responsesFinishStatus(reason *string) string {
	if reason == nil {
		return "incomplete"
	}
	switch *reason {
	case "stop", "tool_calls":
		return "completed"
	case "length":
		return "incomplete"
	case "content_filter":
		return "failed"
	default:
		return "incomplete"
	}
}

// newResponsesResponse creates a ResponsesResponse with Object="response".
func newResponsesResponse() ResponsesResponse {
	return ResponsesResponse{
		Object: "response",
		Status: "completed",
	}
}

// completedMessageItem creates a completed message output item.
func completedMessageItem(text, responseID string) ResponsesOutputItem {
	itemID := responseID
	if itemID == "" {
		itemID = "resp_unknown"
	}
	return ResponsesOutputItem{
		ID:     itemID + ".msg",
		Type:   "message",
		Status: "completed",
		Role:   "assistant",
		Content: []ResponsesContentPart{{
			Type: "output_text",
			Text: text,
		}},
	}
}

// completedReasoningItem creates a completed reasoning output item.
func completedReasoningItem(reasoning string) ResponsesOutputItem {
	return ResponsesOutputItem{
		ID:     "reasoning_1",
		Type:   "reasoning",
		Status: "completed",
		Summary: []ResponsesReasoningSummary{{
			Type: "summary_text",
			Text: reasoning,
		}},
	}
}

// completedFunctionCallItem creates a completed function_call output item from a tool call.
func completedFunctionCallItem(tc OpenAIToolCall) ResponsesOutputItem {
	item := ResponsesOutputItem{
		ID:     tc.ID,
		Type:   "function_call",
		Status: "completed",
		CallID: tc.ID,
		Name:   tc.Function.Name,
	}
	if tc.Function.Arguments != "" {
		item.Arguments = canonicalJSONString(tc.Function.Arguments)
	}
	return item
}

// canonicalJSONString re-marshals a JSON string to normalize formatting.
func canonicalJSONString(s string) string {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return s
	}
	return string(b)
}

// resolveResponseModel resolves the model for a Responses response using the model map.
func resolveResponseModel(upstreamModel string, opts *ConvertOptions) string {
	if opts == nil {
		return upstreamModel
	}
	if opts.RequestModel != "" {
		return opts.RequestModel
	}
	if opts.ModelMap != nil {
		if sourcePrefix := opts.ModelMap.SourcePrefix(upstreamModel); sourcePrefix != "" {
			return sourcePrefix
		}
	}
	return upstreamModel
}

// convertResponsesRequest routes a Responses API request to the appropriate
// upstream protocol (Chat or Anthropic) based on model-map resolution.
func convertResponsesRequest(body []byte, opts *ConvertOptions) ([]byte, error) {
	model := extractModelFromData(body)
	_, proto := resolveModel(model, opts.Model, opts.ModelMap)
	if proto == "anthropic" {
		return ConvertResponsesToAnthropic(body, opts)
	}
	return ConvertResponsesToChat(body, opts)
}

// convertToResponsesResponse converts an upstream Chat or Anthropic response
// back to Responses API format.
func convertToResponsesResponse(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	// Detect upstream error and normalize to Responses error format.
	if _, ok := raw["error"]; ok {
		return chatErrorToResponseError(body)
	}
	if isOpenAIResponse(raw) {
		return ConvertChatToResponses(body, opts)
	}
	if isAnthropicResponse(raw) {
		return ConvertAnthropicToResponses(body, opts)
	}
	if isResponsesResponse(raw) {
		return body, nil
	}
	return body, nil
}

// responsesInputParseResult holds the result of parsing a Responses input.
type responsesInputParseResult struct {
	messages       []OpenAIMessage
	systemTexts    []string
}

// stripLeadingAnthropicBillingHeader removes the first leading
// "x-anthropic-billing-header:" line from s, preserving subsequent
// occurrences. Same behavior as cc-switch.
func stripLeadingAnthropicBillingHeader(s string) string {
	const prefix = "x-anthropic-billing-header:"
	if strings.HasPrefix(s, prefix) {
		if idx := strings.IndexByte(s, '\n'); idx >= 0 {
			rest := s[idx+1:]
			// ponytail: strip trailing blank lines (handles \r\n and \n).
			rest = strings.TrimLeft(rest, "\r\n")
			return rest
		}
		return ""
	}
	return s
}

// appendDedupReasoning appends text to pending reasoning, skipping if text
// is already a substring of the accumulated reasoning.
func appendDedupReasoning(current, text string) string {
	if current == "" {
		return text
	}
	if strings.Contains(current, text) {
		return current
	}
	return current + "\n" + text
}

// collapseSystemMessagesToHead merges all system messages into a single
// first message, joining content with "\n\n". Required for MiniMax
// compatibility.
func collapseSystemMessagesToHead(msgs []OpenAIMessage) []OpenAIMessage {
	var texts []string
	out := make([]OpenAIMessage, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == "system" {
			if s, ok := m.Content.(string); ok && s != "" {
				texts = append(texts, s)
			}
			continue
		}
		out = append(out, m)
	}
	if len(texts) > 0 {
		out = append([]OpenAIMessage{{Role: "system", Content: strings.Join(texts, "\n\n")}}, out...)
	}
	return out
}

// chatUsageToResponsesUsage extracts Chat API usage into Responses API usage,
// including optional cached_tokens and reasoning_tokens details.
func chatUsageToResponsesUsage(resp *OpenAIChatResponse, body []byte) *ResponsesUsage {
	u := &ResponsesUsage{
		InputTokens:  resp.Usage.PromptTokens,
		OutputTokens: resp.Usage.CompletionTokens,
		TotalTokens:  resp.Usage.TotalTokens,
	}
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return u
	}
	usageRaw, _ := raw["usage"].(map[string]any)
	if usageRaw == nil {
		return u
	}
	if pt, _ := usageRaw["prompt_tokens_details"].(map[string]any); pt != nil {
		if ct, _ := pt["cached_tokens"].(float64); ct > 0 {
			u.InputTokensDetails = &ResponsesUsageDetails{CachedTokens: int(ct)}
		}
	}
	if ct, _ := usageRaw["completion_tokens_details"].(map[string]any); ct != nil {
		od := &ResponsesUsageDetails{}
		if rt, _ := ct["reasoning_tokens"].(float64); rt > 0 {
			od.ReasoningTokens = int(rt)
		}
		u.OutputTokensDetails = od
	}
	if cr, _ := usageRaw["cache_read_input_tokens"].(float64); cr > 0 {
		if u.InputTokensDetails == nil {
			u.InputTokensDetails = &ResponsesUsageDetails{}
		}
		u.InputTokensDetails.CachedTokens = int(cr)
	}
	return u
}

// chatErrorToResponseError normalizes upstream Chat API errors into Responses
// API error format. Handles standard OpenAI format, MiniMax base_resp, and
// bare strings.
func chatErrorToResponseError(body []byte) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	errVal := raw["error"]
	if errMap, ok := errVal.(map[string]any); ok {
		resp := newResponsesResponse()
		resp.Status = "failed"
		resp.Error = errMap
		return json.Marshal(resp)
	}
	if errStr, ok := errVal.(string); ok {
		resp := newResponsesResponse()
		resp.Status = "failed"
		resp.Error = map[string]string{"message": errStr}
		return json.Marshal(resp)
	}
	return body, nil
}

// codexToolSpec describes a single tool as seen by Codex, capturing
// its kind (function/namespace/custom/tool_search) and flattened name.
type codexToolSpec struct {
	Kind      string
	Name      string
	Namespace string
}

// codexToolContext maps flat chat-format tool names back to their original
// Responses API tool specification. Built during request conversion and
// used during response conversion to emit correct output item types.
type codexToolContext struct {
	byChatName map[string]codexToolSpec
}

// buildCodexToolContext scans Responses API tools and builds the context.
func buildCodexToolContext(tools []ResponsesTool) *codexToolContext {
	ctx := &codexToolContext{byChatName: make(map[string]codexToolSpec)}
	for _, t := range tools {
		name, _, _ := extractToolParts(t)
		switch t.Type {
		case "custom":
			ctx.byChatName[name] = codexToolSpec{Kind: "custom", Name: name}
		case "tool_search":
			ctx.byChatName[name] = codexToolSpec{Kind: "tool_search", Name: name}
		default:
			ctx.byChatName[name] = codexToolSpec{Kind: "function", Name: name}
		}
	}
	return ctx
}

// toResponsesOutputItem maps a chat-format tool call back to a Responses API
// output item, using the tool context to determine the correct type.
func (ctx *codexToolContext) toResponsesOutputItem(tc OpenAIToolCall) ResponsesOutputItem {
	spec, ok := ctx.byChatName[tc.Function.Name]
	if !ok {
		spec = codexToolSpec{Kind: "function", Name: tc.Function.Name}
	}
	item := ResponsesOutputItem{
		ID:     tc.ID,
		CallID: tc.ID,
		Name:   tc.Function.Name,
	}
	if tc.Function.Arguments != "" {
		item.Arguments = canonicalJSONString(tc.Function.Arguments)
	}
	switch spec.Kind {
	case "custom":
		item.Type = "custom_tool_call"
		item.Status = "completed"
	case "tool_search":
		item.Type = "tool_search_call"
		item.Status = "completed"
	default:
		item.Type = "function_call"
		item.Status = "completed"
		if spec.Namespace != "" {
			item.Type = "function_call"
		}
	}
	return item
}

// customToolInputSchema returns a JSON Schema for a single "input" string
// parameter, used as the function definition for custom tools in Chat API.
// ponytail: downstream Chat APIs don't understand custom tools, so we give
// them a generic string input.
func customToolInputSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"input": map[string]any{"type": "string"},
		},
		"required": []string{"input"},
	}
}
