package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
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

	// Thinking block has been stopped (signature_delta + stop emitted).
	thinkingBlockStopped bool

	// Tool call accumulation: map of OpenAI tool_call index → accumulated state.
	toolCallByIndex map[int]*streamToolState

	// Tool call positions when the provider omits index: nextToolIndex is
	// the next free slot, lastToolIndex the call a fragment without an id
	// continues (-1 before the first call).
	nextToolIndex int
	lastToolIndex int
	openToolID    string

	// The tool call index whose content block is currently open (-1 when
	// none), and calls that arrived while that block's arguments were still
	// incomplete. Held calls are released when the stream ends.
	openToolIndex int
	heldTools     map[int]*streamToolState

	// Tool calls suppressed by declaredTools, so later fragments of the same
	// call cannot open an unnamed tool_use block.
	skippedCalls map[int]bool

	// Finish reason seen from latest chunk.
	finishReason string
	finalized    bool

	// Accumulated reasoning and text across streaming deltas.
	accumulatedReasoning string
	accumulatedText      string

	// Usage accumulated from the final chunk.
	usage *OpenAIUsage

	// Optional cache for reasoning_content replay (DeepSeek V4).
	reasoningCache *ReasoningCache

	// Whether the stream was interrupted (upstream error).
	streamInterrupted bool

	// Tool restriction: only emit tool_use blocks for declared tool names.
	// When restrictTools is true and a tool name is not in declaredTools, the
	// tool call delta is silently skipped to prevent tool hallucination.
	restrictTools bool
	declaredTools map[string]struct{}
}

// NewStreamConverter creates a StreamConverter for the given Anthropic model name
// and optional reasoning cache. declaredTools, if non-empty, restricts emitted
// tool_use blocks to only those tool names.
func NewStreamConverter(model string, reasoningCache *ReasoningCache, declaredTools []string) *StreamConverter {
	sc := &StreamConverter{
		model:           model,
		nextBlockIndex:  0,
		toolCallByIndex: make(map[int]*streamToolState),
		skippedCalls:    make(map[int]bool),
		lastToolIndex:   -1,
		openToolIndex:   -1,
		heldTools:       make(map[int]*streamToolState),
		reasoningCache:  reasoningCache,
	}
	if len(declaredTools) > 0 {
		sc.restrictTools = true
		sc.declaredTools = make(map[string]struct{}, len(declaredTools))
		for _, name := range declaredTools {
			if name != "" {
				sc.declaredTools[name] = struct{}{}
			}
		}
	}
	return sc
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

	// An error chunk is terminal: emit the Anthropic error event and suppress
	// the normal closing sequence (message_delta/message_stop), so a failed
	// upstream is never reported as a successful empty response.
	if chunk.Error != nil {
		msg := chunk.Error.Message
		if msg == "" {
			msg = "upstream stream error"
		}
		evt := sc.EmitError(msg)
		sc.finalized = true
		return evt, nil
	}

	if len(chunk.Choices) == 0 {
		// Check for usage-only chunk (no choices, just usage).
		if chunk.Usage != nil {
			sc.usage = chunk.Usage
		}
		return nil, nil
	}

	choice := chunk.Choices[0]

	// Track usage if present.
	if chunk.Usage != nil {
		sc.usage = chunk.Usage
	}

	// Track finish_reason for message_delta emission.
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		sc.finishReason = *choice.FinishReason
	}

	var events [][]byte

	// Process delta fields in order: content, reasoning_content, tool_calls.
	delta := choice.Delta

	// Content text delta.
	if delta.Content != "" {
		sc.accumulatedText += delta.Content
		sc.stopThinkingBlock()
		events = append(events, sc.ensureBlock("text", "", "")...)
		events = append(events, sc.textDelta(delta.Content)...)
	}

	// Reasoning content → thinking delta.
	if delta.ReasoningContent != "" {
		sc.accumulatedReasoning += delta.ReasoningContent
		if !sc.thinkingBlockStopped {
			events = append(events, sc.ensureBlock("thinking", "", "")...)
			events = append(events, sc.thinkingDelta(delta.ReasoningContent)...)
		}
	}

	// Tool calls.
	if len(delta.ToolCalls) > 0 {
		for _, tc := range delta.ToolCalls {
			events = append(events, sc.handleToolCallDelta(tc)...)
		}
	}
	if delta.FunctionCall != nil {
		events = append(events, sc.handleToolCallDelta(OpenAIDeltaToolCall{
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

	// Close the thinking block if still open.
	if sc.curBlockType == "thinking" {
		events = append(events, sc.thinkingBlockStop())
		sc.curBlockType = ""
		sc.thinkingBlockStopped = true
	}

	// Close the current content block if one is open.
	if sc.curBlockType != "" {
		events = append(events, sc.contentBlockStop())
		sc.curBlockType = ""
		sc.openToolIndex = -1
	}

	// Release tool calls that interleaved with a tool block whose arguments
	// were still incomplete when they arrived.
	events = append(events, sc.releaseHeldTools()...)

	// Cache accumulated reasoning for tool call replay (DeepSeek V4).
	// Store in both tool call ID and text-based caches.
	if sc.accumulatedReasoning != "" && sc.reasoningCache != nil {
		// By tool call IDs.
		if len(sc.toolCallByIndex) > 0 {
			ids := make([]string, 0, len(sc.toolCallByIndex))
			for _, state := range sc.toolCallByIndex {
				if state.ID != "" {
					ids = append(ids, state.ID)
				}
			}
			if len(ids) > 0 {
				sc.reasoningCache.Put(ids, sc.accumulatedReasoning)
			}
		}
		// By assistant text.
		if sc.accumulatedText != "" {
			sc.reasoningCache.PutText(sc.accumulatedText, sc.accumulatedReasoning)
		}
	}

	// Stream interrupted marker.
	if sc.streamInterrupted {
		events = append(events, sc.ensureBlock("text", "", "")...)
		events = append(events, sc.textDelta("\n\n[stream interrupted]")...)
	}

	// Ensure at least one valid text block — a reasoning-only stream
	// (DeepSeek-V4-Pro with a small max_tokens budget) produces a thinking
	// block but no text, leaving the final content array empty. Claude Code's
	// safety classifier cannot parse an empty/no-content response.
	if sc.accumulatedText == "" && len(sc.toolCallByIndex) == 0 {
		events = append(events, sc.ensureBlock("text", "", "")...)
		events = append(events, sc.textDelta(" ")...)
		events = append(events, sc.contentBlockStop())
	}

	// message_delta with stop_reason. When tools were emitted, an empty or
	// "stop" upstream finish reason is ambiguous: OpenAI-compatible backends
	// often report it for a tool-calling turn, so infer tool_use. Explicit
	// reasons such as length or content_filter are preserved.
	stopReason := "end_turn"
	switch {
	case len(sc.toolCallByIndex) > 0 && (sc.finishReason == "" || sc.finishReason == "stop"):
		stopReason = "tool_use"
	case sc.finishReason != "":
		stopReason = mapOpenAIStreamFinish(sc.finishReason)
	}
	usageMap := map[string]any{
		"output_tokens": 0,
	}
	if sc.usage != nil {
		au := anthropicUsageFromOpenAI(sc.usage)
		usageMap = map[string]any{
			"input_tokens":  au.InputTokens,
			"output_tokens": au.OutputTokens,
		}
		if au.CacheReadInputTokens > 0 {
			usageMap["cache_read_input_tokens"] = au.CacheReadInputTokens
		}
		if au.CacheCreationInputTokens > 0 {
			usageMap["cache_creation_input_tokens"] = au.CacheCreationInputTokens
		}
	}
	md := fmt.Sprintf(
		`event: message_delta`+"\n"+`data: {"type":"message_delta","delta":{"stop_reason":"%s","stop_sequence":null},"usage":%s}`,
		stopReason, toJSON(usageMap),
	)
	events = append(events, []byte(md))

	// message_stop.
	events = append(events, []byte(`event: message_stop`+"\n"+`data: {"type":"message_stop"}`))

	return bytes.Join(events, []byte("\n\n"))
}

// SetInterrupted marks the stream as interrupted (upstream error).
func (sc *StreamConverter) SetInterrupted() {
	sc.streamInterrupted = true
}

// EmitError returns an Anthropic-formatted error SSE event. The payload is
// JSON-encoded so quotes, newlines, and backslashes in the message cannot
// corrupt the event.
func (sc *StreamConverter) EmitError(message string) []byte {
	if sc.finalized {
		return nil
	}
	b, _ := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "stream_error", "message": message},
	})
	return []byte("event: error\ndata: " + string(b))
}

// ensureBlock transitions to a new content block if needed.
// If the current block type differs from the requested type, the current
// block is stopped and a new content_block_start is emitted.
// toolID and toolName are only used when blockType is "tool_use".
// Returns the events (may be empty if already on the right block type).
func (sc *StreamConverter) ensureBlock(blockType, toolID, toolName string) [][]byte {
	var events [][]byte

	if sc.curBlockType == blockType {
		return nil
	}

	// Stop current block if any.
	if sc.curBlockType != "" {
		if sc.curBlockType == "thinking" {
			events = append(events, sc.thinkingBlockStop())
			sc.thinkingBlockStopped = true
		} else {
			events = append(events, sc.contentBlockStop())
			if sc.curBlockType == "tool_use" {
				sc.openToolIndex = -1
			}
		}
	}

	// Start new block.
	idx := sc.nextBlockIndex
	sc.nextBlockIndex++
	sc.curBlockType = blockType
	sc.curBlockIndex = idx

	if blockType == "tool_use" {
		events = append(events, toolUseBlockStart(idx, toolID, toolName))
		return events
	}

	startPayload := fmt.Sprintf(
		`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":%d,"content_block":{"type":"%s"}}`,
		idx, blockType,
	)
	events = append(events, []byte(startPayload))

	return events
}

// toolUseBlockStart returns a content_block_start event for a tool_use block.
// id and name are JSON-encoded so provider-supplied values cannot corrupt the
// SSE payload. input is always an empty object; argument fragments follow in
// input_json_delta events.
func toolUseBlockStart(index int, id, name string) []byte {
	block, err := json.Marshal(struct {
		Type  string         `json:"type"`
		ID    string         `json:"id"`
		Name  string         `json:"name"`
		Input map[string]any `json:"input"`
	}{
		Type:  "tool_use",
		ID:    id,
		Name:  name,
		Input: map[string]any{},
	})
	if err != nil {
		block = []byte(`{"type":"tool_use","id":"","name":"","input":{}}`)
	}
	return []byte(fmt.Sprintf(
		`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":%d,"content_block":%s}`,
		index, block,
	))
}

// stopThinkingBlock sends a thinking block signature_delta + stop if the
// thinking block is still open. Used before transitioning to text/tool blocks.
func (sc *StreamConverter) stopThinkingBlock() {
	if sc.curBlockType != "thinking" || sc.thinkingBlockStopped {
		return
	}
	sc.thinkingBlockStopped = true
}

// thinkingBlockStop returns a signature_delta + content_block_stop for the thinking block.
func (sc *StreamConverter) thinkingBlockStop() []byte {
	// Find the current events buffer or return a standalone event set.
	// signature_delta must precede content_block_stop per Anthropic SSE protocol.
	sigDelta := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":""}}`,
		sc.curBlockIndex,
	)
	blockStop := sc.contentBlockStop()
	return []byte(string(sigDelta) + "\n\n" + string(blockStop))
}

// textDelta returns a content_block_delta with text_delta for the given text.
func (sc *StreamConverter) textDelta(text string) [][]byte {
	escaped, _ := json.Marshal(text)
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
// If tool restriction is enabled and the tool name is not declared, the delta
// is silently skipped to prevent tool hallucination.
// toolIndex is the position of tc's call. A provider that sends no index
// starts a new call when the fragment names a new id, and continues the
// last call otherwise.
func (sc *StreamConverter) toolIndex(tc OpenAIDeltaToolCall) int {
	if tc.Index != nil {
		idx := *tc.Index
		if idx >= sc.nextToolIndex {
			sc.nextToolIndex = idx + 1
		}
		sc.lastToolIndex = idx
		if tc.ID != "" {
			sc.openToolID = tc.ID
		}
		return idx
	}
	if tc.ID != "" && tc.ID != sc.openToolID {
		idx := sc.nextToolIndex
		sc.nextToolIndex++
		sc.lastToolIndex = idx
		sc.openToolID = tc.ID
		return idx
	}
	if sc.lastToolIndex < 0 {
		sc.lastToolIndex = sc.nextToolIndex
		sc.nextToolIndex++
	}
	return sc.lastToolIndex
}

// toolArgsComplete reports whether args is a complete JSON object.
func toolArgsComplete(args string) bool {
	var v map[string]any
	return json.Unmarshal([]byte(args), &v) == nil
}

func (sc *StreamConverter) handleToolCallDelta(tc OpenAIDeltaToolCall) [][]byte {
	idx := sc.toolIndex(tc)

	// A call suppressed by declaredTools stays suppressed: its later
	// fragments carry no name and would otherwise open an empty tool_use.
	if sc.skippedCalls[idx] {
		return nil
	}

	// Tool restriction: skip undeclared tool names on the first delta for a call.
	name := tc.Function.Name
	if sc.restrictTools && name != "" {
		if _, ok := sc.declaredTools[name]; !ok {
			sc.skippedCalls[idx] = true
			return nil
		}
	}

	// Check if we've already seen this tool call index.
	existing, seen := sc.toolCallByIndex[idx]

	if seen {
		// Fragments reach the wire only for the currently open block. A held
		// call keeps accumulating silently and is released at stream end; a
		// call whose block already closed keeps accumulating but is never
		// re-emitted.
		existing.Arguments += tc.Function.Arguments
		if idx != sc.openToolIndex || sc.curBlockType != "tool_use" {
			return nil
		}
		return sc.inputJSONDelta(existing.BlockIndex, tc.Function.Arguments)
	}

	// First appearance of this tool call.
	toolID := tc.ID
	if toolID == "" {
		toolID = ensureToolID("")
	}

	state := &streamToolState{
		ID:        toolID,
		Name:      name,
		Arguments: tc.Function.Arguments,
	}

	// Anthropic content blocks are sequential: a call that interleaves with
	// a tool block whose arguments are still incomplete cannot open its own
	// block yet, because doing so would send its fragments to the open
	// block's index. Hold it and release it when the stream ends.
	if sc.curBlockType == "tool_use" && sc.openToolIndex >= 0 {
		if open := sc.toolCallByIndex[sc.openToolIndex]; open != nil && !toolArgsComplete(open.Arguments) {
			sc.toolCallByIndex[idx] = state
			sc.heldTools[idx] = state
			return nil
		}
	}

	sc.toolCallByIndex[idx] = state

	// If we're in a thinking block, stop it before tool_use.
	var events [][]byte
	if sc.curBlockType == "thinking" {
		events = append(events, sc.thinkingBlockStop())
		sc.thinkingBlockStopped = true
		sc.curBlockType = ""
	}

	// If we're already in a tool_use block for a different tool call,
	// stop it before starting the new one.
	if sc.curBlockType == "tool_use" {
		events = append(events, sc.contentBlockStop())
		sc.curBlockType = ""
		sc.openToolIndex = -1
	}

	events = append(events, sc.ensureBlock("tool_use", toolID, name)...)
	state.BlockIndex = sc.curBlockIndex
	sc.openToolIndex = idx

	// If there's initial arguments, emit them as input_json_delta.
	if state.Arguments != "" {
		events = append(events, sc.inputJSONDelta(state.BlockIndex, state.Arguments)...)
	}

	return events
}

// releaseHeldTools emits content blocks for the calls that were held back
// while an earlier tool block was still streaming incomplete arguments.
// Calls are released in ascending OpenAI index order at stream end.
func (sc *StreamConverter) releaseHeldTools() [][]byte {
	if len(sc.heldTools) == 0 {
		return nil
	}

	indices := make([]int, 0, len(sc.heldTools))
	for idx := range sc.heldTools {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	var events [][]byte
	for _, idx := range indices {
		state := sc.heldTools[idx]
		blockIndex := sc.nextBlockIndex
		sc.nextBlockIndex++
		state.BlockIndex = blockIndex

		events = append(events, toolUseBlockStart(blockIndex, state.ID, state.Name))

		args := state.Arguments
		if args == "" {
			args = "{}"
		}
		events = append(events, sc.inputJSONDelta(blockIndex, args)...)
		events = append(events, contentBlockStopAt(blockIndex))
	}
	clear(sc.heldTools)

	return events
}

// inputJSONDelta returns content_block_delta with input_json_delta for the
// given content block index.
func (sc *StreamConverter) inputJSONDelta(blockIndex int, partialJSON string) [][]byte {
	escaped, _ := json.Marshal(partialJSON)
	raw := string(escaped)
	if len(raw) >= 2 && raw[0] == '"' && raw[len(raw)-1] == '"' {
		raw = raw[1 : len(raw)-1]
	}
	evt := fmt.Sprintf(
		`event: content_block_delta`+"\n"+`data: {"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":"%s"}}`,
		blockIndex, raw,
	)
	return [][]byte{[]byte(evt)}
}

// contentBlockStopAt returns a content_block_stop event for the given index.
func contentBlockStopAt(index int) []byte {
	return []byte(fmt.Sprintf(
		`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":%d}`,
		index,
	))
}

// contentBlockStop returns a content_block_stop event.
func (sc *StreamConverter) contentBlockStop() []byte {
	return contentBlockStopAt(sc.curBlockIndex)
}

// toJSON marshals v to a JSON string, returns "{}" on error.
func toJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}
