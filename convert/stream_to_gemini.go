package convert

import (
	"encoding/json"
	"strings"
)

// ---------------------------------------------------------------------------
// Anthropic / OpenAI SSE -> Gemini streamGenerateContent SSE
//
// These are the reverse of the Gemini decoders above: a Gemini-speaking
// client sends its request through the converter to an Anthropic or OpenAI
// upstream, and the upstream's stream is written back as Gemini chunks.
// ---------------------------------------------------------------------------

// geminiSSEChunk writes one Gemini streamGenerateContent event. Gemini SSE
// has no event names; each data line is a complete response object.
func geminiSSEChunk(model string, parts []any, finish string, usage *GeminiUsage) []byte {
	if len(parts) == 0 && finish == "" && usage == nil {
		return nil
	}
	if parts == nil {
		parts = []any{}
	}
	cand := map[string]any{
		"content": map[string]any{"role": "model", "parts": parts},
		"index":   0,
	}
	if finish != "" {
		cand["finishReason"] = finish
	}
	msg := map[string]any{"candidates": []any{cand}}
	if model != "" {
		msg["modelVersion"] = model
	}
	if usage != nil {
		msg["usageMetadata"] = usage
	}
	b, _ := json.Marshal(msg)
	return []byte("data: " + string(b) + "\n\n")
}

func geminiTextParts(text string, thought bool) []any {
	part := map[string]any{"text": text}
	if thought {
		part["thought"] = true
	}
	return []any{part}
}

func geminiArgs(raw string) any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]any{}
	}
	var v any
	if json.Unmarshal([]byte(raw), &v) == nil {
		if _, ok := v.(map[string]any); ok {
			return v
		}
		return map[string]any{"input": v}
	}
	return map[string]any{"input": raw}
}

// ---------------------------------------------------------------------------
// Anthropic SSE -> Gemini
// ---------------------------------------------------------------------------

type AnthropicToGeminiStreamConverter struct {
	model string
	opts  *ConvertOptions

	usage    AnthropicUsage
	hasUsage bool
	stop     string
	finished bool

	tool *anthropicGeminiTool
}

type anthropicGeminiTool struct {
	name string
	args strings.Builder
	sig  string
}

func NewAnthropicToGeminiStreamConverter(model string, opts ...*ConvertOptions) *AnthropicToGeminiStreamConverter {
	c := &AnthropicToGeminiStreamConverter{model: model}
	if len(opts) > 0 {
		c.opts = opts[0]
	}
	return c
}

func (c *AnthropicToGeminiStreamConverter) HandleStreamStart() []byte { return nil }

func (c *AnthropicToGeminiStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if c.finished {
		return nil, nil
	}
	var evt struct {
		Type    string `json:"type"`
		Message *struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			} `json:"usage"`
		} `json:"message"`
		ContentBlock struct {
			Type string `json:"type"`
			Name string `json:"name,omitempty"`
		} `json:"content_block"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text,omitempty"`
			Thinking    string `json:"thinking,omitempty"`
			PartialJSON string `json:"partial_json,omitempty"`
			Signature   string `json:"signature,omitempty"`
		} `json:"delta"`
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, nil
	}

	var events [][]byte
	switch evt.Type {
	case "message_start":
		if evt.Message != nil {
			if c.opts != nil && c.opts.RequestModel != "" {
				c.model = c.opts.RequestModel
			} else if evt.Message.Model != "" {
				c.model = evt.Message.Model
			}
			c.setUsage(evt.Message.Usage.InputTokens, evt.Message.Usage.OutputTokens,
				evt.Message.Usage.CacheReadInputTokens, evt.Message.Usage.CacheCreationInputTokens)
		}

	case "content_block_start":
		if evt.ContentBlock.Type == "tool_use" {
			c.tool = &anthropicGeminiTool{name: evt.ContentBlock.Name}
		}

	case "content_block_delta":
		switch evt.Delta.Type {
		case "text_delta":
			events = append(events, geminiSSEChunk(c.model, geminiTextParts(evt.Delta.Text, false), "", nil))
		case "thinking_delta":
			events = append(events, geminiSSEChunk(c.model, geminiTextParts(evt.Delta.Thinking, true), "", nil))
		case "input_json_delta":
			if c.tool != nil {
				c.tool.args.WriteString(evt.Delta.PartialJSON)
			}
		case "signature_delta":
			if c.tool != nil {
				c.tool.sig = evt.Delta.Signature
			}
		}

	case "content_block_stop":
		events = append(events, c.flushTool()...)

	case "message_delta":
		var md struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			} `json:"usage"`
		}
		_ = json.Unmarshal(data, &md)
		if md.Delta.StopReason != "" {
			c.stop = md.Delta.StopReason
		}
		c.setUsage(md.Usage.InputTokens, md.Usage.OutputTokens,
			md.Usage.CacheReadInputTokens, md.Usage.CacheCreationInputTokens)

	case "message_stop":
		events = append(events, c.finish()...)

	case "error":
		var e struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.Unmarshal(data, &e)
		msg := e.Error.Message
		if msg == "" {
			msg = "upstream stream error"
		}
		return c.EmitError(msg), nil
	}
	return joinSSE(events), nil
}

func (c *AnthropicToGeminiStreamConverter) HandleStreamEnd() []byte {
	if c.finished {
		return nil
	}
	return joinSSE(c.finish())
}

func (c *AnthropicToGeminiStreamConverter) EmitError(message string) []byte {
	if c.finished {
		return nil
	}
	c.finished = true
	b, _ := json.Marshal(map[string]any{"error": map[string]any{
		"code": 502, "message": message, "status": "UNAVAILABLE",
	}})
	return []byte("data: " + string(b) + "\n\n")
}

func (c *AnthropicToGeminiStreamConverter) setUsage(input, output, cacheRead, cacheCreate int) {
	if input == 0 && output == 0 && cacheRead == 0 && cacheCreate == 0 {
		return
	}
	c.usage.InputTokens = input
	c.usage.OutputTokens = output
	c.usage.CacheReadInputTokens = cacheRead
	c.usage.CacheCreationInputTokens = cacheCreate
	c.hasUsage = true
}

func (c *AnthropicToGeminiStreamConverter) flushTool() [][]byte {
	if c.tool == nil {
		return nil
	}
	part := map[string]any{"functionCall": map[string]any{
		"name": c.tool.name,
		"args": geminiArgs(c.tool.args.String()),
	}}
	if c.tool.sig != "" {
		part["thoughtSignature"] = c.tool.sig
	}
	c.tool = nil
	return [][]byte{geminiSSEChunk(c.model, []any{part}, "", nil)}
}

func (c *AnthropicToGeminiStreamConverter) finish() [][]byte {
	if c.finished {
		return nil
	}
	c.finished = true
	var events [][]byte
	events = append(events, c.flushTool()...)
	stop := mapAnthropicFinishToGemini(&c.stop)
	var usage *GeminiUsage
	if c.hasUsage {
		usage = geminiUsageFromAnthropic(c.usage)
	}
	events = append(events, geminiSSEChunk(c.model, nil, stop, usage))
	return events
}

// ---------------------------------------------------------------------------
// OpenAI Chat SSE -> Gemini
// ---------------------------------------------------------------------------

type OpenAIToGeminiStreamConverter struct {
	model string
	opts  *ConvertOptions

	finish   string
	usage    *OpenAIUsage
	finished bool

	tools    map[int]*openAIGeminiTool
	order    []int
	nextTool int
	lastTool int
	openID   string
}

type openAIGeminiTool struct {
	name string
	id   string
	args string
	sig  string
}

func NewOpenAIToGeminiStreamConverter(model string, opts ...*ConvertOptions) *OpenAIToGeminiStreamConverter {
	c := &OpenAIToGeminiStreamConverter{
		model:    model,
		tools:    make(map[int]*openAIGeminiTool),
		lastTool: -1,
	}
	if len(opts) > 0 {
		c.opts = opts[0]
	}
	return c
}

func (c *OpenAIToGeminiStreamConverter) HandleStreamStart() []byte { return nil }

func (c *OpenAIToGeminiStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if c.finished {
		return nil, nil
	}
	var chunk OpenAIStreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil, nil
	}
	if chunk.Error != nil {
		msg := chunk.Error.Message
		if msg == "" {
			msg = "upstream stream error"
		}
		return c.EmitError(msg), nil
	}
	var events [][]byte
	if chunk.Usage != nil {
		c.usage = chunk.Usage
	}
	if len(chunk.Choices) == 0 {
		return nil, nil
	}
	choice := chunk.Choices[0]
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		c.finish = *choice.FinishReason
	}
	delta := choice.Delta
	if delta.Content != "" {
		events = append(events, c.flushTools()...)
		events = append(events, geminiSSEChunk(c.model, geminiTextParts(delta.Content, false), "", nil))
	}
	if delta.ReasoningContent != "" {
		events = append(events, c.flushTools()...)
		events = append(events, geminiSSEChunk(c.model, geminiTextParts(delta.ReasoningContent, true), "", nil))
	}
	for _, tc := range delta.ToolCalls {
		idx := c.toolIndex(tc)
		t := c.tools[idx]
		if t == nil {
			t = &openAIGeminiTool{}
			c.tools[idx] = t
			c.order = append(c.order, idx)
		}
		if tc.ID != "" {
			t.id = tc.ID
		}
		if tc.Function.Name != "" {
			t.name = tc.Function.Name
		}
		t.args += tc.Function.Arguments
	}
	if c.finish != "" {
		events = append(events, c.flushTools()...)
	}
	return joinSSE(events), nil
}

func (c *OpenAIToGeminiStreamConverter) toolIndex(tc OpenAIDeltaToolCall) int {
	if tc.Index != nil {
		idx := *tc.Index
		if idx >= c.nextTool {
			c.nextTool = idx + 1
		}
		c.lastTool = idx
		if tc.ID != "" {
			c.openID = tc.ID
		}
		return idx
	}
	if tc.ID != "" && tc.ID != c.openID {
		idx := c.nextTool
		c.nextTool++
		c.lastTool = idx
		c.openID = tc.ID
		return idx
	}
	if c.lastTool < 0 {
		c.lastTool = c.nextTool
		c.nextTool++
	}
	return c.lastTool
}

func (c *OpenAIToGeminiStreamConverter) flushTools() [][]byte {
	if len(c.order) == 0 {
		return nil
	}
	parts := make([]any, 0, len(c.order))
	for _, idx := range c.order {
		t := c.tools[idx]
		if t == nil {
			continue
		}
		fc := map[string]any{"name": t.name, "args": geminiArgs(t.args)}
		if t.sig != "" {
			fc["thoughtSignature"] = t.sig
		}
		parts = append(parts, map[string]any{"functionCall": fc})
	}
	c.tools = make(map[int]*openAIGeminiTool)
	c.order = nil
	if len(parts) == 0 {
		return nil
	}
	return [][]byte{geminiSSEChunk(c.model, parts, "", nil)}
}

func (c *OpenAIToGeminiStreamConverter) finishStream() [][]byte {
	if c.finished {
		return nil
	}
	c.finished = true
	var events [][]byte
	events = append(events, c.flushTools()...)
	stop := mapOpenAIFinishToGemini(&c.finish)
	var usage *GeminiUsage
	if c.usage != nil {
		usage = geminiUsageFromOpenAI(c.usage)
	}
	events = append(events, geminiSSEChunk(c.model, nil, stop, usage))
	return events
}

func (c *OpenAIToGeminiStreamConverter) HandleStreamEnd() []byte {
	if c.finished {
		return nil
	}
	return joinSSE(c.finishStream())
}

func (c *OpenAIToGeminiStreamConverter) EmitError(message string) []byte {
	if c.finished {
		return nil
	}
	c.finished = true
	b, _ := json.Marshal(map[string]any{"error": map[string]any{
		"code": 502, "message": message, "status": "UNAVAILABLE",
	}})
	return []byte("data: " + string(b) + "\n\n")
}
