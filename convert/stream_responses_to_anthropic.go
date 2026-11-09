package convert

import (
	"encoding/json"
	"fmt"
	"strings"
)

// responsesAnthropicBlock is one open Anthropic content block.
type responsesAnthropicBlock struct {
	index    int
	kind     string // text, thinking, tool_use
	itemID   string
	args     strings.Builder
	argsSent bool
	closed   bool
}

// ResponsesToAnthropicStreamConverter turns a Responses SSE stream into the
// Anthropic event sequence a Messages client expects.
type ResponsesToAnthropicStreamConverter struct {
	model      string
	opts       *ConvertOptions
	started    bool
	finished   bool
	messageID  string
	blockIndex int
	open       *responsesAnthropicBlock
	blocks     map[string]*responsesAnthropicBlock
	usage      AnthropicUsage
	stopReason string
	sawTool    bool
}

// NewResponsesToAnthropicStreamConverter creates a stream converter.
func NewResponsesToAnthropicStreamConverter(model string, opts *ConvertOptions) *ResponsesToAnthropicStreamConverter {
	if opts == nil {
		opts = &ConvertOptions{}
	}
	return &ResponsesToAnthropicStreamConverter{
		model:      model,
		opts:       opts,
		blocks:     make(map[string]*responsesAnthropicBlock),
		stopReason: "end_turn",
	}
}

// HandleStreamStart emits nothing: message_start is produced from the first
// response.created event, or lazily by the first content event.
func (c *ResponsesToAnthropicStreamConverter) HandleStreamStart() []byte {
	return nil
}

// HandleChunk processes one Responses SSE data payload.
func (c *ResponsesToAnthropicStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	var ev ResponsesStreamEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("unmarshal Responses stream event: %w", err)
	}

	switch ev.Type {
	case ResponseCreated:
		if ev.Response != nil {
			c.messageID = ev.Response.ID
			if ev.Response.Usage != nil {
				c.usage = anthropicUsageFromResponses(ev.Response.Usage)
			}
		}
		return c.start(), nil

	case ResponseInProgress:
		return c.start(), nil

	case ResponseOutputItemAdded:
		out := c.start()
		if ev.Item != nil && ev.Item.Type == "message" {
			return out, nil
		}
		return out, nil

	case ResponseContentPartAdded:
		out := c.start()
		itemID, partType := contentPartAdded(ev)
		if partType == "output_text" {
			out = append(out, c.ensureText(itemID)...)
		}
		return out, nil

	case ResponseOutputTextDelta:
		out := c.start()
		out = append(out, c.ensureText(ev.ItemID)...)
		if ev.Delta != "" {
			out = append(out, c.delta("text_delta", map[string]any{"text": ev.Delta})...)
		}
		return out, nil

	case ResponseOutputTextDone, ResponseContentPartDone, ResponseOutputItemDone:
		out := c.start()
		if c.open != nil && c.open.kind == "text" && matchesItem(c.open.itemID, ev) {
			out = append(out, c.closeOpen()...)
		}
		return out, nil

	case ResponseCompleted:
		out := c.start()
		if ev.Response != nil {
			c.usage = anthropicUsageFromResponses(ev.Response.Usage)
			c.stopReason = "end_turn"
		}
		return c.finish(out), nil

	default:
		return c.start(), nil
	}
}

// HandleStreamEnd closes an open block and finishes a stream that never saw a
// terminal event.
func (c *ResponsesToAnthropicStreamConverter) HandleStreamEnd() []byte {
	if c.finished {
		return nil
	}
	out := c.start()
	return c.finish(out)
}

// EmitError emits a terminal Anthropic error event.
func (c *ResponsesToAnthropicStreamConverter) EmitError(message string) []byte {
	if c.finished {
		return nil
	}
	c.finished = true
	return anthropicSSE("error", map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "api_error", "message": message},
	})
}

func (c *ResponsesToAnthropicStreamConverter) start() []byte {
	if c.started {
		return nil
	}
	c.started = true
	msg := map[string]any{
		"id":            c.messageID,
		"type":          "message",
		"role":          "assistant",
		"content":       []any{},
		"model":         c.model,
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  c.usage.InputTokens,
			"output_tokens": 0,
		},
	}
	return anthropicSSE("message_start", map[string]any{
		"type":    "message_start",
		"message": msg,
	})
}

func (c *ResponsesToAnthropicStreamConverter) ensureText(itemID string) []byte {
	if c.open != nil && c.open.kind == "text" && !c.open.closed {
		return nil
	}
	out := c.closeOpen()
	out = append(out, c.openBlock("text", itemID, map[string]any{"type": "text", "text": ""})...)
	return out
}

func (c *ResponsesToAnthropicStreamConverter) openBlock(kind, itemID string, content map[string]any) []byte {
	b := &responsesAnthropicBlock{index: c.blockIndex, kind: kind, itemID: itemID}
	c.blockIndex++
	c.open = b
	if itemID != "" {
		c.blocks[itemID] = b
	}
	return anthropicSSE("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         b.index,
		"content_block": content,
	})
}

func (c *ResponsesToAnthropicStreamConverter) closeOpen() []byte {
	if c.open == nil || c.open.closed {
		c.open = nil
		return nil
	}
	c.open.closed = true
	out := anthropicSSE("content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": c.open.index,
	})
	c.open = nil
	return out
}

func (c *ResponsesToAnthropicStreamConverter) delta(deltaType string, fields map[string]any) []byte {
	if c.open == nil || c.open.closed {
		return nil
	}
	delta := map[string]any{"type": deltaType}
	for k, v := range fields {
		delta[k] = v
	}
	return anthropicSSE("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": c.open.index,
		"delta": delta,
	})
}

func (c *ResponsesToAnthropicStreamConverter) finish(out []byte) []byte {
	out = append(out, c.closeOpen()...)
	c.finished = true
	if c.stopReason == "" {
		c.stopReason = "end_turn"
	}
	usage := map[string]any{
		"input_tokens":  c.usage.InputTokens,
		"output_tokens": c.usage.OutputTokens,
	}
	if c.usage.CacheReadInputTokens > 0 {
		usage["cache_read_input_tokens"] = c.usage.CacheReadInputTokens
	}
	if c.usage.CacheCreationInputTokens > 0 {
		usage["cache_creation_input_tokens"] = c.usage.CacheCreationInputTokens
	}
	out = append(out, anthropicSSE("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": c.stopReason, "stop_sequence": nil},
		"usage": usage,
	})...)
	out = append(out, anthropicSSE("message_stop", map[string]any{"type": "message_stop"})...)
	return out
}

func contentPartAdded(ev ResponsesStreamEvent) (itemID, partType string) {
	itemID = ev.ItemID
	if ev.Part == nil {
		return itemID, ""
	}
	part, ok := ev.Part.(map[string]any)
	if !ok {
		return itemID, ""
	}
	partType, _ = part["type"].(string)
	return itemID, partType
}

func matchesItem(itemID string, ev ResponsesStreamEvent) bool {
	if itemID == "" || ev.ItemID == "" {
		return true
	}
	return itemID == ev.ItemID
}

func anthropicSSE(event string, payload any) []byte {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil
	}
	return []byte("event: " + event + "\ndata: " + string(b) + "\n\n")
}
