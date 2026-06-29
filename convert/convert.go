package convert

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// SSE event handling
// ---------------------------------------------------------------------------

// isSSE reports whether body appears to be an SSE event by checking for
// known SSE field prefixes at the start of the data.
func isSSE(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	switch {
	case bytes.HasPrefix(body, []byte("data:")):
		return true
	case bytes.HasPrefix(body, []byte("event:")):
		return true
	case bytes.HasPrefix(body, []byte("id:")):
		return true
	}
	return false
}

// parseSSEEvent parses raw SSE event bytes into fields. It handles
// event:, data:, id:, retry: fields and ignores unknown lines.
func parseSSEEvent(raw []byte) *SSEEvent {
	evt := &SSEEvent{}
	for _, line := range bytes.Split(raw, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		key, value, found := bytes.Cut(line, []byte(":"))
		if !found {
			continue
		}
		// SSE convention: "field: value" — strip one leading space if present.
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}

		switch string(key) {
		case "event":
			evt.Event = string(value)
		case "data":
			if evt.Data != "" {
				evt.Data += "\n"
			}
			evt.Data += string(value)
		case "id":
			evt.ID = string(value)
		case "retry":
			if ms, err := strconv.Atoi(string(value)); err == nil && ms > 0 {
				evt.Retry = ms
			}
		}
	}
	return evt
}

// reconstructSSEEvent builds SSE-formatted bytes from the event fields.
// It does NOT include the trailing \n\n event delimiter — the caller
// (sniffer_sse.go) appends \n\n after the rewritten bytes.
func reconstructSSEEvent(evt *SSEEvent) []byte {
	var buf bytes.Buffer
	if evt.ID != "" {
		buf.WriteString("id: ")
		buf.WriteString(evt.ID)
		buf.WriteByte('\n')
	}
	if evt.Event != "" {
		buf.WriteString("event: ")
		buf.WriteString(evt.Event)
		buf.WriteByte('\n')
	}
	for _, line := range strings.Split(evt.Data, "\n") {
		buf.WriteString("data: ")
		buf.WriteString(line)
		buf.WriteByte('\n')
	}
	if evt.Retry > 0 {
		buf.WriteString("retry: ")
		buf.WriteString(strconv.Itoa(evt.Retry))
		buf.WriteByte('\n')
	}
	// Strip trailing newline — the GOST proxy appends \n\n as the event delimiter.
	raw := buf.Bytes()
	if len(raw) > 0 && raw[len(raw)-1] == '\n' {
		raw = raw[:len(raw)-1]
	}
	return raw
}

// convertSSEEvent converts the JSON data payload inside an SSE event
// using the existing Convert function. Returns true if the data changed.
func convertSSEEvent(evt *SSEEvent, opts *ConvertOptions) bool {
	if evt.Data == "" {
		return false
	}
	converted, err := Convert([]byte(evt.Data), opts)
	if err != nil {
		slog.Debug("sse convert error, using original data", "err", err)
		return false
	}
	convertedStr := string(converted)
	if convertedStr == evt.Data {
		return false
	}
	evt.Data = convertedStr
	return true
}

// ConvertSSE parses SSE-formatted body, converts the JSON data payload,
// and reconstructs the SSE framing. Non-SSE content that looks like SSE
// (e.g. "[DONE]") passes through unchanged.
func ConvertSSE(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, Downstream: "deepseek-chat"}
	}
	if len(body) == 0 {
		return body, nil
	}

	// "[DONE]" is a common SSE data marker — convert to Anthropic message_stop.
	if bytes.Equal(bytes.TrimSpace(body), []byte("data: [DONE]")) {
		return []byte("event: message_stop\ndata: {\"type\":\"message_stop\"}"), nil
	}

	evt := parseSSEEvent(body)

	// Check for OpenAI streaming chunk: data payload with choices[].delta.
	// This must be detected before the generic Convert() path since OpenAI
	// streaming chunks have a different structure from non-streaming responses.
	if evt.Data != "" && isOpenAIStreamChunk([]byte(evt.Data)) {
		return convertOpenAIStreamChunkToAnthropic(evt)
	}

	// Default: convert data payload with Convert() (handles Anthropic SSE → OpenAI data).
	convertSSEEvent(evt, opts)
	return reconstructSSEEvent(evt), nil
}

// convertOpenAIStreamChunkToAnthropic converts an OpenAI SSE streaming delta
// to an Anthropic SSE content_block_delta / message_delta / message_stop event.
func convertOpenAIStreamChunkToAnthropic(evt *SSEEvent) ([]byte, error) {
	var chunk struct {
		Choices []struct {
			Index        int              `json:"index"`
			Delta        map[string]any   `json:"delta"`
			FinishReason *string          `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(evt.Data), &chunk); err != nil || len(chunk.Choices) == 0 {
		// Unrecognised format — pass through unchanged.
		return reconstructSSEEvent(evt), nil
	}

	choice := chunk.Choices[0]

	// Finish reason → message_delta (stream end).
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		stopReason := mapOpenAIStreamFinish(*choice.FinishReason)
		deltaJSON, _ := json.Marshal(map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   stopReason,
				"stop_sequence": nil,
			},
			"usage": map[string]any{
				"output_tokens": 0,
			},
		})
		return []byte("event: message_delta\ndata: " + string(deltaJSON)), nil
	}

	// Content delta → text_delta.
	if content, ok := choice.Delta["content"].(string); ok && content != "" {
		deltaJSON, _ := json.Marshal(map[string]any{
			"type": "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type": "text_delta",
				"text": content,
			},
		})
		return []byte("event: content_block_delta\ndata: " + string(deltaJSON)), nil
	}

	// Reasoning content → thinking_delta.
	if reasoning, ok := choice.Delta["reasoning_content"].(string); ok && reasoning != "" {
		deltaJSON, _ := json.Marshal(map[string]any{
			"type": "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type":     "thinking_delta",
				"thinking": reasoning,
			},
		})
		return []byte("event: content_block_delta\ndata: " + string(deltaJSON)), nil
	}

	// Empty delta (role announcement, e.g. delta:{"role":"assistant"}) — pass through as-is.
	// The Claude SDK ignores unrecognised data: lines.
	return reconstructSSEEvent(evt), nil
}

// mapOpenAIStreamFinish maps OpenAI finish_reason to Anthropic stop_reason.
func mapOpenAIStreamFinish(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return reason
	}
}

// Convert detects the input body format and performs bidirectional
// conversion between OpenAI Chat Completions and Anthropic Messages formats.
func Convert(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, Downstream: "deepseek-chat"}
	}

	if len(body) == 0 {
		return body, nil
	}

	// Auto-detect SSE framing: if the body starts with data:, event:,
	// or id:, route through SSE-aware handling.
	if isSSE(body) {
		slog.Debug("detected SSE framing → converting via ConvertSSE")
		return ConvertSSE(body, opts)
	}

	// Try parsing as a generic object to detect format.
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		slog.Debug("not JSON, passing through", "err", err)
		return body, nil
	}

	// Detect: Anthropic Messages Request (messages + max_tokens or Anthropic content blocks).
	if isAnthropicRequest(raw) {
		slog.Debug("detected Anthropic Request → converting to OpenAI")
		return convertAnthropicRequestToOpenAI(body, opts)
	}

	// Detect: OpenAI Chat Completions Response (choices array).
	if isOpenAIResponse(raw) {
		slog.Debug("detected OpenAI Response → converting to Anthropic")
		return convertOpenAIResponseToAnthropic(body, opts)
	}

	// Detect: Anthropic Messages Response (type "message" and stop_reason/usage).
	if isAnthropicResponse(raw) {
		slog.Debug("detected Anthropic Response → converting to OpenAI")
		return convertAnthropicResponseToOpenAI(body)
	}

	// Detect: OpenAI Chat Completions Request (model or messages field).
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
		"gpt-", "o1", "o3", "deepseek", "gemini-",
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

// isOpenAIStreamChunk detects an OpenAI SSE streaming data payload.
// Format: {"choices":[{"index":0,"delta":{...},"finish_reason":null}]}
func isOpenAIStreamChunk(data []byte) bool {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return false
	}
	choices, ok := raw["choices"].([]any)
	if !ok || len(choices) == 0 {
		return false
	}
	if choice, ok := choices[0].(map[string]any); ok {
		if _, ok := choice["delta"]; ok {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Message sequence sanitization
// ---------------------------------------------------------------------------

const placeholderReasoning = "Compatibility bridge placeholder reasoning for prior assistant history."

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
// Anthropic Request → OpenAI Request
// ---------------------------------------------------------------------------

func convertAnthropicRequestToOpenAI(body []byte, opts *ConvertOptions) ([]byte, error) {
	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("failed to unmarshal Anthropic request", "err", err)
		return body, nil
	}

	profile := classifyModel(opts.Downstream)

	oai := OpenAIChatRequest{
		Model:  opts.Downstream,
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

	// Thinking config → OpenAI thinking (DeepSeek-compatible models only).
	if profile.isDeepSeek {
		oai.Thinking = thinkingToOpenAi(req.Thinking)
		oai.ReasoningEffort = reasoningEffortToOpenAi(req.OutputConfig)
	}

	// Tool choice instruction for DeepSeek (softened to system instruction).
	extraInstruction := toolChoiceInstruction(req.ToolChoice, opts.Downstream)

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
		oai.ToolChoice = anthropicToolChoiceToOpenAi(req.ToolChoice, opts.Downstream)
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

// ---------------------------------------------------------------------------
// SSE stream lifecycle management
// ---------------------------------------------------------------------------

// streamStates maps GOST session IDs to active StreamConverter instances.
// Each stream's state is stored when a "start" phase event arrives and
// deleted when the "end" phase event is processed.
var streamStates sync.Map // sid (string) -> *StreamConverter

// extractDeclaredTools returns declared tool names from ConvertOptions, or nil.
func extractDeclaredTools(opts *ConvertOptions) []string {
	if opts != nil && len(opts.DeclaredTools) > 0 {
		return opts.DeclaredTools
	}
	return nil
}

// HandleSSEEvent processes an SSE stream lifecycle event.
// It routes to the appropriate StreamConverter method based on the phase.
func HandleSSEEvent(sid, phase string, eventIndex int, data []byte, opts *ConvertOptions) ([]byte, error) {
	switch StreamPhase(phase) {
	case StreamPhaseStart:
		sc := NewStreamConverter(opts.Model, opts.ReasoningCache, extractDeclaredTools(opts))
		streamStates.Store(sid, sc)
		startData := sc.HandleStreamStart()
		// First event data is now attached to the start phase signal
		// (sniffer sends the first real SSE event with sse_phase:"start").
		if len(data) > 0 {
			payload := extractSSEPayload(data)
			if payload != nil {
				chunkData, err := sc.HandleChunk(payload)
				if err != nil {
					return startData, err
				}
				if len(chunkData) > 0 {
					return append(startData, append([]byte("\n\n"), chunkData...)...), nil
				}
			}
		}
		return startData, nil

	case StreamPhaseEvent:
		payload := extractSSEPayload(data)
		if payload == nil {
			return nil, nil // [DONE] marker, skip silently.
		}

		v, ok := streamStates.Load(sid)
		if !ok {
			slog.Warn("HandleSSEEvent: unknown stream, starting new", "sid", sid)
			sc := NewStreamConverter(opts.Model, opts.ReasoningCache, extractDeclaredTools(opts))
			streamStates.Store(sid, sc)
			startData := sc.HandleStreamStart()
			chunkData, err := sc.HandleChunk(payload)
			if err != nil {
				return startData, err
			}
			if len(chunkData) > 0 {
				return append(startData, append([]byte("\n\n"), chunkData...)...), nil
			}
			return startData, nil
		}
		sc := v.(*StreamConverter)
		return sc.HandleChunk(payload)

	case StreamPhaseEnd:
		v, ok := streamStates.Load(sid)
		if !ok {
			slog.Warn("HandleSSEEvent: unknown stream for end phase", "sid", sid)
			return nil, nil
		}
		sc := v.(*StreamConverter)
		streamStates.Delete(sid)
		return sc.HandleStreamEnd(), nil
	}

	return nil, fmt.Errorf("unknown sse_phase: %s", phase)
}

// extractSSEPayload parses SSE event text and returns just the data payload.
// The GOST sniffer sends raw SSE event text (e.g., "data: {...}") to the
// plugin, but HandleChunk needs just the JSON payload without SSE framing.
// "[DONE]" markers are consumed silently (return nil, no error).
func extractSSEPayload(data []byte) []byte {
	evt := parseSSEEvent(data)
	if evt.Data == "[DONE]" {
		return nil
	}
	if evt.Data != "" {
		return []byte(evt.Data)
	}
	return data
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
		blocks = append(blocks, AnthropicContent{Type: "text", Text: text})
	}

	return blocks
}

// ---------------------------------------------------------------------------
// Utility helpers (canonical JSON, tool sorting, ID normalization)
// ---------------------------------------------------------------------------

// ensureObjectSchema guarantees the schema is a JSON object with "type":"object".
// OpenAI spec requires {"type":"object", ...} but some schemas omit it.
func ensureObjectSchema(raw any) any {
	m, ok := raw.(map[string]any)
	if !ok || m == nil {
		return map[string]any{"type": "object"}
	}
	if _, exists := m["type"]; !exists {
		m["type"] = "object"
	}
	return m
}

func sortOpenAITools(tools []OpenAITool) {
	sort.SliceStable(tools, func(i, j int) bool {
		return tools[i].Function.Name < tools[j].Function.Name
	})
}

func sortAnthropicTools(tools []AnthropicTool) {
	sort.SliceStable(tools, func(i, j int) bool {
		return tools[i].Name < tools[j].Name
	})
}

// ensureToolID normalises a tool call ID to Anthropic's "toolu_" prefix.
// OpenAI IDs like "call_xxx" are not recognised by the Anthropic SDK.
func ensureToolID(id string) string {
	if strings.HasPrefix(id, "toolu_") {
		return id
	}
	return "toolu_" + randHex(24)
}

// randHex returns n crypto-random hex characters.
func randHex(n int) string {
	b := make([]byte, (n+1)/2)
	if _, err := rand.Read(b); err != nil {
		for i := range b {
			b[i] = byte(i * 7)
		}
	}
	const hexd = "0123456789abcdef"
	out := make([]byte, n)
	for i := 0; i < n; i++ {
		out[i] = hexd[b[i/2]>>uint((i%2)*4)&0xf]
	}
	return string(out)
}
