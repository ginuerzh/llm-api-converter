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
	sent     bool // a text or thinking delta was emitted
	closed   bool
}

// pendingResponsesCall is a function call added while another call was open.
// Its events are replayed after the open call closes so Anthropic blocks stay
// sequential.
type pendingResponsesCall struct {
	added ResponsesStreamEvent
	lines [][]byte
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
	pending    []*pendingResponsesCall
	pendingBy  map[string]*pendingResponsesCall
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
		pendingBy:  make(map[string]*pendingResponsesCall),
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
	if c.finished {
		return nil, nil
	}
	var ev ResponsesStreamEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("unmarshal Responses stream event: %w", err)
	}

	// A call added while another call is open waits its turn. Its own events
	// are buffered raw and replayed after the open call closes.
	if ev.Type != ResponseOutputItemAdded {
		if call := c.pendingFor(eventItemID(ev)); call != nil {
			call.lines = append(call.lines, append([]byte(nil), data...))
			return nil, nil
		}
	}
	if ev.Type == ResponseOutputItemAdded && ev.Item != nil && ev.Item.Type == "function_call" &&
		c.open != nil && c.open.kind == "tool_use" && c.open.itemID != ev.Item.ID {
		call := &pendingResponsesCall{added: ev}
		c.pending = append(c.pending, call)
		c.pendingBy[ev.Item.ID] = call
		return nil, nil
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
		if ev.Item == nil {
			return out, nil
		}
		switch ev.Item.Type {
		case "message":
			return out, nil
		case "reasoning":
			out = append(out, c.closeOpen()...)
			out = append(out, c.openBlock("thinking", ev.Item.ID, map[string]any{
				"type": "thinking", "thinking": "", "signature": "",
			})...)
		case "function_call":
			out = append(out, c.closeOpen()...)
			id := ev.Item.CallID
			if id == "" {
				id = ev.Item.ID
			}
			out = append(out, c.openBlock("tool_use", ev.Item.ID, map[string]any{
				"type": "tool_use", "id": id, "name": ev.Item.Name, "input": map[string]any{},
			})...)
			c.sawTool = true
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
			c.markSent()
		}
		return out, nil

	case ResponseOutputTextDone:
		out := c.start()
		if c.open != nil && c.open.kind == "text" && matchesItem(c.open.itemID, ev) {
			if !c.open.sent && ev.TextDone != "" {
				out = append(out, c.delta("text_delta", map[string]any{"text": ev.TextDone})...)
			}
			out = append(out, c.closeOpen()...)
		}
		return out, nil

	case ResponseContentPartDone:
		out := c.start()
		if c.open != nil && c.open.kind == "text" && matchesItem(c.open.itemID, ev) {
			out = append(out, c.closeOpen()...)
		}
		return out, nil

	case ResponseFunctionCallArgumentsDelta:
		out := c.start()
		if b := c.blockFor(ev.ItemID); b != nil && b.kind == "tool_use" {
			b.args.WriteString(ev.Delta)
			if ev.Delta != "" {
				out = append(out, c.delta("input_json_delta", map[string]any{"partial_json": ev.Delta})...)
				b.argsSent = true
			}
		}
		return out, nil

	case ResponseFunctionCallArgumentsDone:
		out := c.start()
		if b := c.blockFor(ev.ItemID); b != nil && b.kind == "tool_use" && !b.argsSent && ev.Arguments != "" {
			b.args.WriteString(ev.Arguments)
			out = append(out, c.delta("input_json_delta", map[string]any{"partial_json": ev.Arguments})...)
			b.argsSent = true
		}
		return out, nil

	case ResponseReasoningSummaryPartAdded:
		out := c.start()
		out = append(out, c.ensureThinking(ev.ItemID)...)
		return out, nil

	case ResponseReasoningSummaryDelta, ResponseReasoningTextDelta:
		out := c.start()
		out = append(out, c.ensureThinking(ev.ItemID)...)
		if ev.Delta != "" {
			out = append(out, c.delta("thinking_delta", map[string]any{"thinking": ev.Delta})...)
			c.markSent()
		}
		return out, nil

	case ResponseReasoningSummaryDone, ResponseReasoningTextDone:
		out := c.start()
		if c.open != nil && c.open.kind == "thinking" && matchesItem(c.open.itemID, ev) {
			text := ev.TextDone
			if !c.open.sent && text != "" {
				out = append(out, c.delta("thinking_delta", map[string]any{"thinking": text})...)
			}
			out = append(out, c.closeOpen()...)
		}
		return out, nil

	case ResponseOutputItemDone:
		out := c.start()
		if ev.Item != nil && ev.Item.Type == "function_call" {
			out = append(out, c.finishTool(ev)...)
			c.flushPending(&out)
			return out, nil
		}
		if c.open != nil && matchesItem(c.open.itemID, ev) {
			out = append(out, c.closeOpen()...)
		}
		return out, nil

	case ResponseCompleted:
		out := c.start()
		if ev.Response != nil {
			c.usage = anthropicUsageFromResponses(ev.Response.Usage)
		}
		c.stopReason = ""
		return c.finish(out), nil

	case ResponseIncomplete:
		out := c.start()
		if ev.Response != nil {
			c.usage = anthropicUsageFromResponses(ev.Response.Usage)
			c.stopReason = responsesIncompleteStopReason(ev.Response, c.sawTool)
		}
		return c.finish(out), nil

	case ResponseFailed:
		out := c.start()
		return append(out, c.EmitError(responseErrorMessage(ev))...), nil

	case "error":
		msg := ev.Message
		if msg == "" {
			msg = nestedErrorMessage(ev.Error)
		}
		if msg == "" && ev.Response != nil {
			msg = nestedErrorMessage(ev.Response.Error)
		}
		if msg == "" {
			msg = "upstream error"
		}
		out := c.start()
		return append(out, c.EmitError(msg)...), nil

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
	if c.opts.ErrorMsg != "" {
		return c.EmitError(c.opts.ErrorMsg)
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
	c.flushPending(&out)
	out = append(out, c.openBlock("text", itemID, map[string]any{"type": "text", "text": ""})...)
	return out
}

func (c *ResponsesToAnthropicStreamConverter) ensureThinking(itemID string) []byte {
	if c.open != nil && c.open.kind == "thinking" && !c.open.closed && (itemID == "" || c.open.itemID == itemID) {
		return nil
	}
	out := c.closeOpen()
	c.flushPending(&out)
	out = append(out, c.openBlock("thinking", itemID, map[string]any{
		"type": "thinking", "thinking": "", "signature": "",
	})...)
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

// closeOpenForTerminal closes an open tool block, emitting the empty object a
// no-argument tool call needs before the block stops. A partially written tool
// block is left for truncatedToolArgs to reject.
func (c *ResponsesToAnthropicStreamConverter) closeOpenForTerminal() []byte {
	if c.open == nil || c.open.closed || c.open.kind != "tool_use" || c.open.argsSent {
		return c.closeOpen()
	}
	args := c.open.args.String()
	if args != "" {
		return c.closeOpen()
	}
	c.open.args.WriteString("{}")
	out := c.delta("input_json_delta", map[string]any{"partial_json": "{}"})
	c.open.argsSent = true
	return append(out, c.closeOpen()...)
}

// truncatedToolArgs reports the first tool block whose accumulated arguments
// are not a complete JSON object. Empty arguments are a valid no-arg call.
func (c *ResponsesToAnthropicStreamConverter) truncatedToolArgs() string {
	check := func(b *responsesAnthropicBlock) string {
		if b == nil || b.kind != "tool_use" {
			return ""
		}
		args := b.args.String()
		if args == "" || toolArgsComplete(args) {
			return ""
		}
		return "truncated tool arguments"
	}
	if msg := check(c.open); msg != "" {
		return msg
	}
	for _, b := range c.blocks {
		if msg := check(b); msg != "" {
			return msg
		}
	}
	return ""
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

func (c *ResponsesToAnthropicStreamConverter) markSent() {
	if c.open != nil {
		c.open.sent = true
	}
}

func (c *ResponsesToAnthropicStreamConverter) blockFor(itemID string) *responsesAnthropicBlock {
	if itemID != "" {
		if b := c.blocks[itemID]; b != nil && !b.closed {
			return b
		}
	}
	if c.open != nil && !c.open.closed {
		return c.open
	}
	return nil
}

func (c *ResponsesToAnthropicStreamConverter) finishTool(ev ResponsesStreamEvent) []byte {
	if ev.Item == nil || c.open == nil || c.open.kind != "tool_use" || c.open.itemID != ev.Item.ID {
		return nil
	}
	out := []byte{}
	if !c.open.argsSent {
		args := ev.Item.Arguments
		if args == "" {
			args = "{}"
		}
		c.open.args.WriteString(args)
		out = append(out, c.delta("input_json_delta", map[string]any{"partial_json": args})...)
		c.open.argsSent = true
	}
	out = append(out, c.closeOpen()...)
	c.sawTool = true
	return out
}

// eventItemID is the item a Responses event refers to. Delta and done events
// use the top-level item_id, while output_item.done carries only item.id.
func eventItemID(ev ResponsesStreamEvent) string {
	if ev.ItemID != "" {
		return ev.ItemID
	}
	if ev.Item != nil {
		return ev.Item.ID
	}
	return ""
}

func (c *ResponsesToAnthropicStreamConverter) pendingFor(itemID string) *pendingResponsesCall {
	if itemID == "" {
		return nil
	}
	return c.pendingBy[itemID]
}

func (c *ResponsesToAnthropicStreamConverter) flushPending(out *[]byte) {
	for len(c.pending) > 0 {
		// Never replay a deferred call over an open block: its added event
		// would close that block early. The caller closes first, then calls us.
		if c.open != nil && !c.open.closed {
			return
		}
		call := c.pending[0]
		c.pending = c.pending[1:]
		delete(c.pendingBy, call.added.Item.ID)
		if data, err := json.Marshal(call.added); err == nil {
			if b, err := c.HandleChunk(data); err == nil {
				*out = append(*out, b...)
			}
		}
		for _, line := range call.lines {
			if len(line) == 0 {
				continue
			}
			if b, err := c.HandleChunk(line); err == nil {
				*out = append(*out, b...)
			}
		}
	}
}

func (c *ResponsesToAnthropicStreamConverter) finish(out []byte) []byte {
	// Validate before closing anything: an incomplete tool block must not
	// emit content_block_stop. Then close the open block, drain deferred
	// calls one at a time, and validate again after each one.
	if msg := c.truncatedToolArgs(); msg != "" {
		return append(out, c.EmitError(msg)...)
	}
	out = append(out, c.closeOpenForTerminal()...)
	for len(c.pending) > 0 {
		c.flushPending(&out)
		if msg := c.truncatedToolArgs(); msg != "" {
			return append(out, c.EmitError(msg)...)
		}
		out = append(out, c.closeOpenForTerminal()...)
	}
	if msg := c.truncatedToolArgs(); msg != "" {
		return append(out, c.EmitError(msg)...)
	}
	c.finished = true
	if c.stopReason == "" || (c.stopReason == "end_turn" && c.sawTool) {
		if c.sawTool {
			c.stopReason = "tool_use"
		} else if c.stopReason == "" {
			c.stopReason = "end_turn"
		}
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

func responsesIncompleteStopReason(resp *ResponsesResponse, sawTool bool) string {
	if details, ok := resp.IncompleteDetails.(map[string]any); ok {
		switch details["reason"] {
		case "max_output_tokens":
			return "max_tokens"
		case "content_filter":
			return "refusal"
		}
	}
	if sawTool {
		return "tool_use"
	}
	return "end_turn"
}

func responseErrorMessage(ev ResponsesStreamEvent) string {
	if ev.Message != "" {
		return ev.Message
	}
	if msg := nestedErrorMessage(ev.Error); msg != "" {
		return msg
	}
	if ev.Response != nil {
		if msg := nestedErrorMessage(ev.Response.Error); msg != "" {
			return msg
		}
	}
	return "upstream failed"
}

func nestedErrorMessage(v any) string {
	m, ok := v.(map[string]any)
	if !ok {
		return ""
	}
	msg, _ := m["message"].(string)
	return msg
}

func anthropicSSE(event string, payload any) []byte {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil
	}
	return []byte("event: " + event + "\ndata: " + string(b) + "\n\n")
}
