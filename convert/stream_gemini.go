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
	opts         *ConvertOptions
	finalized    bool
	finishReason string
	usage        *OpenAIUsage

	// tool accumulates a functionCall whose arguments may arrive over
	// several chunks; nextTool is the next OpenAI index.
	tool     *geminiPendingTool
	nextTool int
}

// geminiPendingTool is a functionCall being assembled from one or more Gemini
// chunks. Gemini may repeat the call with more fields as the arguments grow.
type geminiPendingTool struct {
	name string
	id   string
	args map[string]any
	sig  string
}

func (p *geminiPendingTool) merge(fc *GeminiFunctionCall, sig string) {
	if p.args == nil {
		p.args = make(map[string]any)
	}
	p.args = mergeGeminiArgs(p.args, fc.Args)
	if p.id == "" {
		p.id = fc.ID
	}
	if sig != "" {
		p.sig = sig
	}
}

func sameGeminiTool(p *geminiPendingTool, fc *GeminiFunctionCall) bool {
	if p == nil || p.name != fc.Name {
		return false
	}
	return p.id == "" || fc.ID == "" || p.id == fc.ID
}

func mergeGeminiArgs(dst map[string]any, src any) map[string]any {
	m, ok := src.(map[string]any)
	if !ok {
		return dst
	}
	if dst == nil {
		dst = make(map[string]any)
	}
	for k, v := range m {
		if dm, ok := dst[k].(map[string]any); ok {
			if sm, ok := v.(map[string]any); ok {
				dst[k] = mergeGeminiArgs(dm, sm)
				continue
			}
		}
		dst[k] = v
	}
	return dst
}

func NewGeminiToOpenAIStreamConverter(model string, opts ...*ConvertOptions) *GeminiToOpenAIStreamConverter {
	c := &GeminiToOpenAIStreamConverter{
		model: model,
		id:    "chatcmpl-gemini-" + randHex(16),
	}
	if len(opts) > 0 {
		c.opts = opts[0]
	}
	return c
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
	if resp.Error != nil {
		msg := resp.Error.Message
		if msg == "" {
			msg = "upstream stream error"
		}
		return c.EmitError(msg), nil
	}

	// Blocked response.
	if len(resp.Candidates) == 0 {
		if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
			return c.emitError(fmt.Sprintf("blocked by Gemini safety: %s", resp.PromptFeedback.BlockReason), "content_filter"), nil
		}
		return nil, nil
	}

	cand := resp.Candidates[0]

	if cand.FinishReason != "" {
		c.finishReason = cand.FinishReason
	}
	if resp.UsageMetadata != nil {
		u := openAIUsageFromGemini(resp.UsageMetadata)
		c.usage = &u
	}

	// Emit text immediately; buffer function calls until their arguments
	// stop growing (text, another call, finish or end of stream).
	var events [][]byte
	for _, part := range cand.Content.Parts {
		if part.Text != "" {
			events = append(events, c.flushTool()...)
			d := OpenAIDelta{}
			if part.Thought {
				d.ReasoningContent = part.Text
			} else {
				d.Content = part.Text
			}
			events = append(events, c.chunkEvent(d, nil))
		}
		if part.FunctionCall != nil {
			events = append(events, c.bufferTool(part.FunctionCall, part.ThoughtSignature)...)
		}
	}
	if cand.FinishReason != "" {
		events = append(events, c.flushTool()...)
	}
	return joinSSE(events), nil
}

func (c *GeminiToOpenAIStreamConverter) chunkEvent(delta OpenAIDelta, finish *string) []byte {
	chunk := OpenAIStreamChunk{
		ID:      c.id,
		Object:  "chat.completion.chunk",
		Model:   c.model,
		Choices: []OpenAIStreamChoice{{Index: 0, Delta: delta, FinishReason: finish}},
	}
	b, _ := json.Marshal(chunk)
	return []byte("data: " + string(b) + "\n\n")
}

func (c *GeminiToOpenAIStreamConverter) bufferTool(fc *GeminiFunctionCall, sig string) [][]byte {
	var events [][]byte
	if c.tool != nil && !sameGeminiTool(c.tool, fc) {
		events = append(events, c.flushTool()...)
	}
	if c.tool == nil {
		c.tool = &geminiPendingTool{name: fc.Name, id: fc.ID}
	}
	c.tool.merge(fc, sig)
	return events
}

func (c *GeminiToOpenAIStreamConverter) flushTool() [][]byte {
	if c.tool == nil {
		return nil
	}
	args := "{}"
	if b, err := json.Marshal(c.tool.args); err == nil && len(b) > 0 {
		args = string(b)
	}
	id := "call_" + randHex(16)
	name := c.tool.name
	if c.opts != nil && c.opts.SessionStore != nil && c.opts.SID != "" {
		c.opts.SessionStore.SetThoughtSigFor(c.opts.SID, id, name, c.tool.sig)
	}
	idx := c.nextTool
	c.nextTool++
	c.tool = nil
	return [][]byte{c.chunkEvent(OpenAIDelta{ToolCalls: []OpenAIDeltaToolCall{{
		Index: intPtr(idx),
		ID:    id,
		Type:  "function",
		Function: OpenAIFunctionCall{
			Name:      name,
			Arguments: args,
		},
	}}}, nil)}
}

func (c *GeminiToOpenAIStreamConverter) HandleStreamEnd() []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true

	var result [][]byte
	result = append(result, c.flushTool()...)

	// Emit final chunk with finish reason + usage.
	if c.finishReason != "" || c.usage != nil {
		fr := mapGeminiFinishToOpenAI(c.finishReason, false)
		chunk := OpenAIStreamChunk{
			ID:      c.id,
			Object:  "chat.completion.chunk",
			Model:   c.model,
			Choices: []OpenAIStreamChoice{{Index: 0, Delta: OpenAIDelta{}, FinishReason: fr}},
			Usage:   c.usage,
		}
		b, _ := json.Marshal(chunk)
		result = append(result, []byte("data: "+string(b)+"\n\n"))
	}

	// [DONE] marker.
	result = append(result, []byte("data: [DONE]\n\n"))
	return joinSSE(result)
}

func (c *GeminiToOpenAIStreamConverter) EmitError(message string) []byte {
	return c.emitError(message, "server_error")
}

// emitError builds a terminal OpenAI error chunk with the given error type.
func (c *GeminiToOpenAIStreamConverter) emitError(message, typ string) []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true
	b, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": message, "type": typ},
	})
	return []byte("data: " + string(b) + "\n\n")
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
	opts           *ConvertOptions
	curBlockType   string // "text", "thinking" or "tool_use"
	curBlockIndex  int
	nextBlockIndex int
	finishReason   string
	usage          *GeminiUsage
	started        bool
	finalized      bool
	tool           *geminiPendingTool
}

func NewGeminiToAnthropicStreamConverter(model string, opts ...*ConvertOptions) *GeminiToAnthropicStreamConverter {
	c := &GeminiToAnthropicStreamConverter{
		model: model,
		msgID: "msg_gemini_" + randHex(16),
	}
	if len(opts) > 0 {
		c.opts = opts[0]
	}
	return c
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
	if resp.Error != nil {
		msg := resp.Error.Message
		if msg == "" {
			msg = "upstream stream error"
		}
		return c.EmitError(msg), nil
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
		c.usage = resp.UsageMetadata
	}

	var events [][]byte

	for _, part := range cand.Content.Parts {
		switch {
		case part.Text != "":
			events = append(events, c.flushTool()...)
			if part.Thought {
				if c.curBlockType != "thinking" {
					if c.curBlockType != "" {
						events = append(events, c.contentBlockStop())
						c.curBlockType = ""
					}
					events = append(events, c.ensureBlock("thinking", "")...)
				}
				events = append(events, c.thinkingDelta(part.Text)...)
				if part.ThoughtSignature != "" {
					events = append(events, c.signatureDelta(part.ThoughtSignature)...)
				}
			} else {
				if c.curBlockType == "thinking" || c.curBlockType == "tool_use" {
					events = append(events, c.contentBlockStop())
					c.curBlockType = ""
				}
				if c.curBlockType == "" {
					events = append(events, c.ensureBlock("text", "")...)
				}
				events = append(events, c.textDelta(part.Text)...)
			}

		case part.FunctionCall != nil:
			events = append(events, c.bufferTool(part.FunctionCall, part.ThoughtSignature)...)
		}
	}

	if cand.FinishReason != "" {
		events = append(events, c.flushTool()...)
	}

	if len(events) == 0 {
		return nil, nil
	}
	return bytes.Join(events, []byte("\n\n")), nil
}

func (c *GeminiToAnthropicStreamConverter) bufferTool(fc *GeminiFunctionCall, sig string) [][]byte {
	var events [][]byte
	if c.tool != nil && !sameGeminiTool(c.tool, fc) {
		events = append(events, c.flushTool()...)
	}
	if c.tool == nil {
		c.tool = &geminiPendingTool{name: fc.Name, id: fc.ID}
	}
	c.tool.merge(fc, sig)
	return events
}

func (c *GeminiToAnthropicStreamConverter) flushTool() [][]byte {
	if c.tool == nil {
		return nil
	}
	var events [][]byte
	if c.curBlockType != "" {
		events = append(events, c.contentBlockStop())
		c.curBlockType = ""
	}
	id := "toolu_" + randHex(24)
	name := c.tool.name
	if c.opts != nil && c.opts.SessionStore != nil && c.opts.SID != "" {
		c.opts.SessionStore.SetThoughtSigFor(c.opts.SID, id, name, c.tool.sig)
	}
	meta := fmt.Sprintf(`"id":"%s","name":"%s"`, id, name)
	events = append(events, c.ensureBlock("tool_use", meta)...)
	argsJSON, _ := json.Marshal(c.tool.args)
	if len(argsJSON) == 0 {
		argsJSON = []byte("{}")
	}
	events = append(events, c.inputJSONDelta(string(argsJSON))...)
	c.curBlockType = "tool_use"
	c.tool = nil
	return events
}

func (c *GeminiToAnthropicStreamConverter) HandleStreamEnd() []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true

	var events [][]byte
	events = append(events, c.flushTool()...)

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
	usageJSON, _ := json.Marshal(geminiAnthropicUsage(c.usage))
	md := fmt.Sprintf(
		`event: message_delta`+"\n"+`data: {"type":"message_delta","delta":{"stop_reason":"%s","stop_sequence":null},"usage":%s}`,
		stopReason, usageJSON,
	)
	events = append(events, []byte(md))

	// message_stop.
	events = append(events, []byte(`event: message_stop`+"\n"+`data: {"type":"message_stop"}`))

	return bytes.Join(events, []byte("\n\n"))
}

func (c *GeminiToAnthropicStreamConverter) EmitError(message string) []byte {
	if c.finalized {
		return nil
	}
	c.finalized = true
	b, _ := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "stream_error", "message": message},
	})
	return []byte("event: error\ndata: " + string(b))
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
	switch blockType {
	case "text":
		startPayload += `,"text":""`
	case "thinking":
		startPayload += `,"thinking":"","signature":""`
	case "tool_use":
		if meta != "" {
			startPayload += "," + meta
		}
		startPayload += `,"input":{}`
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

func (c *GeminiToAnthropicStreamConverter) thinkingDelta(text string) [][]byte {
	escaped, _ := json.Marshal(text)
	raw := string(escaped)
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":"%s"}}`,
		c.curBlockIndex, raw,
	)
	return [][]byte{[]byte(evt)}
}

func (c *GeminiToAnthropicStreamConverter) signatureDelta(sig string) [][]byte {
	escaped, _ := json.Marshal(sig)
	raw := string(escaped)
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":"%s"}}`,
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
