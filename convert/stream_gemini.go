package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// ---------------------------------------------------------------------------
// Gemini → OpenAI: stateful stream converter
// Gemini SSE delivers a complete GeminiChatResponse per chunk with incremental
// text in parts[]. Each chunk is independently mapped to an OpenAI delta chunk.
// ---------------------------------------------------------------------------

type GeminiToOpenAIStreamConverter struct {
	model        string
	id           string
	finalized    bool
	finishReason string
	usage        *OpenAIUsage
}

func NewGeminiToOpenAIStreamConverter(model string) *GeminiToOpenAIStreamConverter {
	return &GeminiToOpenAIStreamConverter{
		model: model,
		id:    "chatcmpl-gemini-" + randHex(16),
	}
}

// HandleStreamStart returns nil — OpenAI SSE has no start-of-stream event.
func (c *GeminiToOpenAIStreamConverter) HandleStreamStart() []byte { return nil }

func (c *GeminiToOpenAIStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if c.finalized {
		return nil, nil
	}
	var resp GeminiChatResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("gemini stream chunk: %w", err)
	}

	// Blocked response.
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			b, _ := json.Marshal(map[string]any{
				"error": map[string]any{
					"message": fmt.Sprintf("blocked by Gemini safety: %s", resp.PromptFeedback.BlockReason),
					"type":    "content_filter",
				},
			})
			return append([]byte("data: "), append(b, '\n')...), nil
		}
		return nil, nil
	}

	cand := resp.Candidates[0]

	if cand.FinishReason != "" {
		c.finishReason = cand.FinishReason
	}
	if resp.UsageMetadata != nil {
		c.usage = &OpenAIUsage{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		}
	}

	// Build delta from parts.
	delta := OpenAIDelta{}
	for _, part := range cand.Content.Parts {
		if part.Text != "" {
			delta.Content += part.Text
		}
		if part.FunctionCall != nil {
			args := "{}"
			if part.FunctionCall.Args != nil {
				if b, err := json.Marshal(part.FunctionCall.Args); err == nil {
					args = string(b)
				}
			}
			delta.ToolCalls = append(delta.ToolCalls, OpenAIDeltaToolCall{
				Index: len(delta.ToolCalls),
				ID:    "call_" + randHex(16),
				Type:  "function",
				Function: OpenAIFunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: args,
				},
			})
		}
	}

	if delta.Content == "" && len(delta.ToolCalls) == 0 {
		return nil, nil
	}

	chunk := OpenAIStreamChunk{
		ID:      c.id,
		Object:  "chat.completion.chunk",
		Model:   c.model,
		Choices: []OpenAIStreamChoice{{Index: 0, Delta: delta}},
	}
	b, _ := json.Marshal(chunk)
	return append([]byte("data: "), append(b, '\n')...), nil
}

func (c *GeminiToOpenAIStreamConverter) HandleStreamEnd() []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true

	var result [][]byte

	// Emit final chunk with finish reason + usage.
	if c.finishReason != "" || c.usage != nil {
		fr := mapGeminiFinishToOpenAI(c.finishReason, false)
		chunk := OpenAIStreamChunk{
			ID:      c.id,
			Object:  "chat.completion.chunk",
			Choices: []OpenAIStreamChoice{{Index: 0, Delta: OpenAIDelta{}, FinishReason: fr}},
		}
		if c.usage != nil {
			chunk.Usage = c.usage
		}
		b, _ := json.Marshal(chunk)
		result = append(result, []byte("data: "+string(b)))
	}

	// [DONE] marker.
	result = append(result, []byte("data: [DONE]"))
	return bytes.Join(result, []byte("\n\n"))
}

func (c *GeminiToOpenAIStreamConverter) EmitError(message string) []byte {
	b, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": message},
	})
	return append([]byte("data: "), append(b, '\n')...)
}

// ---------------------------------------------------------------------------
// Gemini → Anthropic: stateful stream converter
// Gemini SSE chunks are mapped to the Anthropic event sequence:
//
//	message_start → ping → content_block_start → content_block_delta*
//	→ content_block_stop → message_delta → message_stop
// ---------------------------------------------------------------------------

type GeminiToAnthropicStreamConverter struct {
	model          string
	msgID          string
	curBlockType   string // "text" or "tool_use"
	curBlockIndex  int
	nextBlockIndex int
	finishReason   string
	usage          *AnthropicUsage
	started        bool
	finalized      bool
}

func NewGeminiToAnthropicStreamConverter(model string) *GeminiToAnthropicStreamConverter {
	return &GeminiToAnthropicStreamConverter{
		model: model,
		msgID: "msg_gemini_" + randHex(16),
	}
}

func (c *GeminiToAnthropicStreamConverter) HandleStreamStart() []byte {
	c.started = true
	msgStart := fmt.Sprintf(
		`event: message_start`+"\n"+`data: {"type":"message_start","message":{"id":"%s","type":"message","role":"assistant","content":[],"model":"%s","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}`,
		c.msgID, c.model,
	)
	ping := `event: ping` + "\n" + `data: {"type":"ping"}`
	return []byte(msgStart + "\n\n" + ping)
}

func (c *GeminiToAnthropicStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if c.finalized {
		return nil, nil
	}
	var resp GeminiChatResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("gemini stream chunk: %w", err)
	}

	// Blocked response → error event.
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			return c.EmitError(fmt.Sprintf("blocked by Gemini safety: %s", resp.PromptFeedback.BlockReason)), nil
		}
		return nil, nil
	}

	cand := resp.Candidates[0]
	if cand.FinishReason != "" {
		c.finishReason = cand.FinishReason
	}
	if resp.UsageMetadata != nil {
		c.usage = &AnthropicUsage{
			InputTokens:  resp.UsageMetadata.PromptTokenCount,
			OutputTokens: resp.UsageMetadata.CandidatesTokenCount,
		}
	}

	var events [][]byte

	for _, part := range cand.Content.Parts {
		switch {
		case part.Text != "":
			if c.curBlockType == "tool_use" {
				events = append(events, c.contentBlockStop())
			}
			if c.curBlockType == "" || c.curBlockType == "tool_use" {
				events = append(events, c.ensureBlock("text", "")...)
			}
			events = append(events, c.textDelta(part.Text)...)

		case part.FunctionCall != nil:
			if c.curBlockType == "text" {
				events = append(events, c.contentBlockStop())
			}
			meta := fmt.Sprintf(`"id":"%s","name":"%s"`, "toolu_"+randHex(24), part.FunctionCall.Name)
			events = append(events, c.ensureBlock("tool_use", meta)...)

			if part.FunctionCall.Args != nil {
				argsJSON, _ := json.Marshal(part.FunctionCall.Args)
				events = append(events, c.inputJSONDelta(string(argsJSON))...)
			}
		}
	}

	if len(events) == 0 {
		return nil, nil
	}
	return bytes.Join(events, []byte("\n\n")), nil
}

func (c *GeminiToAnthropicStreamConverter) HandleStreamEnd() []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true

	var events [][]byte

	// Close the current content block.
	if c.curBlockType != "" {
		events = append(events, c.contentBlockStop())
		c.curBlockType = ""
	}

	// message_delta with stop_reason.
	stopReason := "end_turn"
	if c.finishReason != "" {
		stopReason = mapGeminiFinishToAnthropicStr(c.finishReason)
	}
	md := fmt.Sprintf(
		`event: message_delta`+"\n"+`data: {"type":"message_delta","delta":{"stop_reason":"%s","stop_sequence":null},"usage":{"input_tokens":%d,"output_tokens":%d}}`,
		stopReason, c.usageInputTokens(), c.usageOutputTokens(),
	)
	events = append(events, []byte(md))

	// message_stop.
	events = append(events, []byte(`event: message_stop`+"\n"+`data: {"type":"message_stop"}`))

	return bytes.Join(events, []byte("\n\n"))
}

func (c *GeminiToAnthropicStreamConverter) EmitError(message string) []byte {
	return []byte(fmt.Sprintf(
		`event: error`+"\n"+`data: {"type":"error","error":{"type":"stream_error","message":"%s"}}`,
		message,
	))
}

// ensureBlock transitions to a new content block.
func (c *GeminiToAnthropicStreamConverter) ensureBlock(blockType, meta string) [][]byte {
	if c.curBlockType == blockType {
		return nil
	}
	idx := c.nextBlockIndex
	c.nextBlockIndex++
	c.curBlockType = blockType
	c.curBlockIndex = idx
	startPayload := fmt.Sprintf(
		`data: {"type":"content_block_start","index":%d,"content_block":{"type":"%s"`,
		idx, blockType,
	)
	if meta != "" {
		startPayload += "," + meta
	}
	startPayload += `}}`
	return [][]byte{[]byte("event: content_block_start\n" + startPayload)}
}

func (c *GeminiToAnthropicStreamConverter) contentBlockStop() []byte {
	return []byte(fmt.Sprintf(
		`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":%d}`,
		c.curBlockIndex,
	))
}

func (c *GeminiToAnthropicStreamConverter) textDelta(text string) [][]byte {
	escaped, _ := json.Marshal(text)
	raw := string(escaped)
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":"%s"}}`,
		c.curBlockIndex, raw,
	)
	return [][]byte{[]byte(evt)}
}

func (c *GeminiToAnthropicStreamConverter) inputJSONDelta(partialJSON string) [][]byte {
	escaped, _ := json.Marshal(partialJSON)
	raw := string(escaped)
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":"%s"}}`,
		c.curBlockIndex, raw,
	)
	return [][]byte{[]byte(evt)}
}

// mapGeminiFinishToAnthropicStr maps Gemini finishReason to Anthropic stop_reason.
func mapGeminiFinishToAnthropicStr(reason string) string {
	switch reason {
	case "STOP":
		return "end_turn"
	case "MAX_TOKENS":
		return "max_tokens"
	default:
		return "end_turn"
	}
}

func (c *GeminiToAnthropicStreamConverter) usageInputTokens() int {
	if c.usage != nil {
		return c.usage.InputTokens
	}
	return 0
}

func (c *GeminiToAnthropicStreamConverter) usageOutputTokens() int {
	if c.usage != nil {
		return c.usage.OutputTokens
	}
	return 0
}
