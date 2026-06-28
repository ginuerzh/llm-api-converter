package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// StreamConverter converts OpenAI streaming delta chunks into the proper
// Anthropic SSE event sequence (message_start → ping → content_block_start
// → content_block_delta* → content_block_stop → message_delta → message_stop).
//
// It is stateful per stream: it tracks which content blocks have been
// started and accumulates partial tool call arguments across deltas.
type StreamConverter struct {
	model string

	// Content block state.
	curBlockType   string // "text", "thinking", "tool_use", or "" (none)
	curBlockIndex  int
	nextBlockIndex int

	// Tool call accumulation: map of OpenAI tool_call index → accumulated state.
	toolCallByIndex map[int]*streamToolState

	// Finish reason seen from latest chunk.
	finishReason string
	finalized    bool
}

// NewStreamConverter creates a StreamConverter for the given Anthropic model name.
func NewStreamConverter(model string) *StreamConverter {
	return &StreamConverter{
		model:           model,
		nextBlockIndex:  0,
		toolCallByIndex: make(map[int]*streamToolState),
	}
}

// HandleStreamStart returns the Anthropic message_start + ping SSE events
// (with internal \n\n separator, plus terminal \n\n).
func (sc *StreamConverter) HandleStreamStart() []byte {
	msgStart := fmt.Sprintf(
		`event: message_start`+"\n"+`data: {"type":"message_start","message":{"id":"msg_%s","type":"message","role":"assistant","content":[],"model":"%s","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}`,
		randHex(16), sc.model,
	)
	ping := `event: ping` + "\n" + `data: {"type":"ping"}`
	return []byte(msgStart + "\n\n" + ping)
}

// HandleChunk processes one OpenAI streaming delta chunk and returns the
// corresponding Anthropic SSE event(s) as a byte slice. Returns nil when
// the chunk produces no Anthropic event (e.g. role announcement).
// The returned bytes do NOT include a trailing \n\n delimiter.
func (sc *StreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if sc.finalized {
		return nil, nil
	}

	var chunk OpenAIStreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil, fmt.Errorf("unmarshal stream chunk: %w", err)
	}
	if len(chunk.Choices) == 0 {
		return nil, nil
	}

	choice := chunk.Choices[0]

	// Track finish_reason for message_delta emission.
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		sc.finishReason = *choice.FinishReason
	}

	var events [][]byte

	// Process delta fields in order: content, reasoning_content, tool_calls.
	delta := choice.Delta

	// Content text delta.
	if delta.Content != "" {
		events = append(events, sc.ensureBlock("text", "")...)
		events = append(events, sc.textDelta(delta.Content)...)
	}

	// Reasoning content → thinking delta.
	if delta.ReasoningContent != "" {
		events = append(events, sc.ensureBlock("thinking", "")...)
		events = append(events, sc.thinkingDelta(delta.ReasoningContent)...)
	}

	// Tool calls.
	if len(delta.ToolCalls) > 0 {
		for _, tc := range delta.ToolCalls {
			events = append(events, sc.handleToolCallDelta(tc)...)
		}
	}
	if delta.FunctionCall != nil {
		events = append(events, sc.handleToolCallDelta(OpenAIDeltaToolCall{
			Index:    0,
			Type:     "function",
			Function: *delta.FunctionCall,
		})...)
	}

	if len(events) == 0 {
		return nil, nil
	}
	return bytes.Join(events, []byte("\n\n")), nil
}

// HandleStreamEnd returns the remaining Anthropic events to close the
// stream: content_block_stop (if a block is open) + message_delta + message_stop.
// The returned bytes are separated by \n\n (with terminal \n\n).
func (sc *StreamConverter) HandleStreamEnd() []byte {
	if sc.finalized {
		return nil
	}
	sc.finalized = true

	var events [][]byte

	// Close the current content block if one is open.
	if sc.curBlockType != "" {
		events = append(events, sc.contentBlockStop())
		sc.curBlockType = ""
	}

	// message_delta with stop_reason.
	stopReason := "end_turn"
	if sc.finishReason != "" {
		stopReason = mapOpenAIStreamFinish(sc.finishReason)
	}
	md := fmt.Sprintf(
		`event: message_delta`+"\n"+`data: {"type":"message_delta","delta":{"stop_reason":"%s","stop_sequence":null},"usage":{"output_tokens":0}}`,
		stopReason,
	)
	events = append(events, []byte(md))

	// message_stop.
	events = append(events, []byte(`event: message_stop`+"\n"+`data: {"type":"message_stop"}`))

	return bytes.Join(events, []byte("\n\n"))
}

// ensureBlock transitions to a new content block if needed.
// If the current block type differs from the requested type, the current
// block is stopped and a new content_block_start is emitted.
// Returns the events (may be empty if already on the right block type).
func (sc *StreamConverter) ensureBlock(blockType, toolMeta string) [][]byte {
	var events [][]byte

	if sc.curBlockType == blockType {
		return nil
	}

	// Stop current block if any.
	if sc.curBlockType != "" {
		events = append(events, sc.contentBlockStop())
	}

	// Start new block.
	idx := sc.nextBlockIndex
	sc.nextBlockIndex++
	sc.curBlockType = blockType
	sc.curBlockIndex = idx

	startPayload := fmt.Sprintf(
		`data: {"type":"content_block_start","index":%d,"content_block":{"type":"%s"`,
		idx, blockType,
	)
	if toolMeta != "" {
		startPayload += "," + toolMeta
	}
	startPayload += `}}`

	events = append(events, []byte("event: content_block_start\n"+startPayload))

	return events
}

// textDelta returns a content_block_delta with text_delta for the given text.
func (sc *StreamConverter) textDelta(text string) [][]byte {
	// JSON-escape the text.
	escaped, _ := json.Marshal(text)
	// Unquote to get the raw escaped string.
	rawText := string(escaped)
	if len(rawText) >= 2 && rawText[0] == '"' && rawText[len(rawText)-1] == '"' {
		rawText = rawText[1 : len(rawText)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":"%s"}}`,
		sc.curBlockIndex, rawText,
	)
	return [][]byte{[]byte(evt)}
}

// thinkingDelta returns a content_block_delta with thinking_delta.
func (sc *StreamConverter) thinkingDelta(content string) [][]byte {
	escaped, _ := json.Marshal(content)
	raw := string(escaped)
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":"%s"}}`,
		sc.curBlockIndex, raw,
	)
	return [][]byte{[]byte(evt)}
}

// handleToolCallDelta processes a single tool call delta entry.
func (sc *StreamConverter) handleToolCallDelta(tc OpenAIDeltaToolCall) [][]byte {
	idx := tc.Index

	// Check if we've already seen this tool call index.
	existing, seen := sc.toolCallByIndex[idx]

	if !seen {
		// First appearance of this tool call — start a tool_use block.
		toolID := tc.ID
		if toolID == "" {
			toolID = ensureToolID("")
		}
		name := tc.Function.Name

		sc.toolCallByIndex[idx] = &streamToolState{
			ID:   toolID,
			Name: name,
		}

		meta := fmt.Sprintf(`"id":"%s","name":"%s"`, toolID, name)

		// If we're already in a tool_use block for a different tool call,
		// stop it before starting the new one.
		var events [][]byte
		if sc.curBlockType == "tool_use" {
			events = append(events, sc.contentBlockStop())
			sc.curBlockType = ""
		}

		events = append(events, sc.ensureBlock("tool_use", meta)...)

		// If there's initial arguments, emit them as input_json_delta.
		if tc.Function.Arguments != "" {
			sc.toolCallByIndex[idx].Arguments = tc.Function.Arguments
			events = append(events, sc.inputJSONDelta(tc.Function.Arguments)...)
		}

		return events
	}

	// Accumulate arguments to existing tool call state.
	existing.Arguments += tc.Function.Arguments
	return sc.inputJSONDelta(tc.Function.Arguments)
}

// inputJSONDelta returns content_block_delta with input_json_delta.
func (sc *StreamConverter) inputJSONDelta(partialJSON string) [][]byte {
	escaped, _ := json.Marshal(partialJSON)
	raw := string(escaped)
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":"%s"}}`,
		sc.curBlockIndex, raw,
	)
	return [][]byte{[]byte(evt)}
}

// contentBlockStop returns a content_block_stop event.
func (sc *StreamConverter) contentBlockStop() []byte {
	return []byte(fmt.Sprintf(
		`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":%d}`,
		sc.curBlockIndex,
	))
}