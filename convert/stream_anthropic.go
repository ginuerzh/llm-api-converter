package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"
)

// responsesStreamHandler is implemented by both ResponsesStreamConverter
// and AnthropicStreamConverter for routing SSE events.
type responsesStreamHandler interface {
	HandleStreamStart() []byte
	HandleChunk(data []byte) ([]byte, error)
	HandleStreamEnd() []byte
	EmitError(message string) []byte
}

// AnthropicStreamConverter converts Anthropic SSE stream events to Responses
// API SSE events. Maps the Anthropic event sequence (message_start,
// content_block_start/content_block_delta/content_block_stop, message_delta,
// message_stop) to the Responses event sequence (output_item.added,
// output_text.delta, reasoning_summary_text.delta,
// function_call_arguments.delta, done events, response.completed).
type AnthropicStreamConverter struct {
	model    string
	response ResponsesResponse

	started   bool
	finalized bool

	accText      string
	accReasoning string

	itemIndex   int
	partIndex   int
	textItemIx  int // -1 = not started
	textPartIx  int // -1 = not started
	reasonItemIx int // -1 = not started
	reasonPartIx int // -1 = not started

	contentBlockTypes map[int]string // Anthropic block index → "text", "tool_use", "thinking"
	toolCallByIndex   map[int]*fcState
}

// NewAnthropicStreamConverter creates a new converter for Anthropic SSE →
// Responses SSE conversion.
func NewAnthropicStreamConverter(model string) *AnthropicStreamConverter {
	return &AnthropicStreamConverter{
		model: model,
		response: ResponsesResponse{
			ID:        "resp_" + randHex(16),
			Object:    "response",
			CreatedAt: time.Now().Unix(),
			Status:    "in_progress",
			Model:     model,
		},
		contentBlockTypes: make(map[int]string),
		toolCallByIndex:   make(map[int]*fcState),
		textItemIx:        -1,
		textPartIx:        -1,
		reasonItemIx:      -1,
		reasonPartIx:      -1,
		itemIndex:         -1,
		partIndex:         -1,
	}
}

// HandleStreamStart emits response.created and response.in_progress events.
func (sc *AnthropicStreamConverter) HandleStreamStart() []byte {
	if sc.started {
		return nil
	}
	sc.started = true
	created, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseCreated, Response: &sc.response})
	inProg, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseInProgress, Response: &sc.response})
	return []byte("event: response.created\ndata: " + string(created) + "\n\nevent: response.in_progress\ndata: " + string(inProg))
}

// isAnthropicStreamEvent reports whether data is an Anthropic SSE event payload
// (message_start, content_block_start, content_block_delta, etc.).
func isAnthropicStreamEvent(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	var raw struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &raw); err != nil || raw.Type == "" {
		return false
	}
	switch raw.Type {
	case "message_start", "content_block_start", "content_block_delta",
		"content_block_stop", "message_delta", "message_stop", "ping":
		return true
	}
	return false
}

// HandleChunk processes one Anthropic SSE event data payload and returns
// the corresponding Responses SSE events.
func (sc *AnthropicStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if sc.finalized {
		return nil, nil
	}
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, nil
	}

	switch envelope.Type {
	case "message_start":
		return sc.handleMessageStart(data)
	case "content_block_start":
		return sc.handleContentBlockStart(data)
	case "content_block_delta":
		return sc.handleContentBlockDelta(data)
	case "content_block_stop":
		return nil, nil
	case "message_delta":
		return sc.handleMessageDelta(data)
	case "message_stop":
		return nil, nil
	case "ping":
		return nil, nil
	default:
		return nil, nil
	}
}

// HandleStreamEnd finalizes the stream and emits done events + response.completed.
func (sc *AnthropicStreamConverter) HandleStreamEnd() []byte {
	if sc.finalized {
		return nil
	}
	sc.finalized = true
	var events [][]byte

	// Finalize text content part.
	if sc.textPartIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseContentPartDone, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), OutputIndex: &sc.textItemIx, ContentIndex: &sc.textPartIx, PartIndex: &sc.textPartIx})
		events = append(events, eventBytes(ResponseContentPartDone, string(evt)))
		evt, _ = json.Marshal(ResponsesStreamEvent{Type: ResponseOutputTextDone, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), TextDone: sc.accText})
		events = append(events, eventBytes(ResponseOutputTextDone, string(evt)))
	}
	if sc.textItemIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemDone, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), OutputIndex: &sc.textItemIx})
		events = append(events, eventBytes(ResponseOutputItemDone, string(evt)))
	}
	// Finalize reasoning item.
	if sc.reasonItemIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryDone,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			SummaryText:  sc.accReasoning,
		})
		events = append(events, eventBytes(ResponseReasoningSummaryDone, string(evt)))
		evt, _ = json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryPartDone,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			Part:         ResponsesReasoningSummary{Type: "summary_text", Text: sc.accReasoning},
		})
		events = append(events, eventBytes(ResponseReasoningSummaryPartDone, string(evt)))
		evt, _ = json.Marshal(ResponsesStreamEvent{
			Type:        ResponseOutputItemDone,
			OutputIndex: &sc.reasonItemIx,
			Item: &ResponsesOutputItem{
				ID:      fmt.Sprintf("%s.reasoning", sc.response.ID),
				Type:    "reasoning",
				Status:  "completed",
				Summary: []ResponsesReasoningSummary{{Type: "summary_text", Text: sc.accReasoning}},
			},
		})
		events = append(events, eventBytes(ResponseOutputItemDone, string(evt)))
	}
	// Finalize tool calls.
	for _, fc := range sc.toolCallByIndex {
		if fc.Arguments != "" {
			evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDone, ItemID: fc.ID, Arguments: canonicalJSONString(fc.Arguments)})
			events = append(events, eventBytes(ResponseFunctionCallArgumentsDone, string(evt)))
		}
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemDone, ItemID: fc.ID, OutputIndex: &fc.ItemIx})
		events = append(events, eventBytes(ResponseOutputItemDone, string(evt)))
	}

	// response.completed or response.incomplete.
	now := time.Now().Unix()
	sc.response.CompletedAt = &now
	if sc.response.Status == "in_progress" {
		sc.response.Status = "completed"
	}
	eventType := ResponseCompleted
	if sc.response.Status == "incomplete" {
		eventType = ResponseIncomplete
	}
	completed, _ := json.Marshal(ResponsesStreamEvent{Type: eventType, Response: &sc.response})
	events = append(events, eventBytes(eventType, string(completed)))
	return bytes.Join(events, []byte("\n\n"))
}

// handleMessageStart captures usage from the initial message event.
func (sc *AnthropicStreamConverter) handleMessageStart(data []byte) ([]byte, error) {
	var evt struct {
		Message struct {
			ID    string `json:"id"`
			Model string `json:"model"`
			Usage struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		} `json:"message"`
	}
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, nil
	}
	sc.response.ID = "resp_" + evt.Message.ID
	if evt.Message.Model != "" {
		// Keep the client-facing model but capture for reference.
	}
	sc.response.Usage = &ResponsesUsage{
		InputTokens:  evt.Message.Usage.InputTokens,
		OutputTokens: evt.Message.Usage.OutputTokens,
		TotalTokens:  evt.Message.Usage.InputTokens + evt.Message.Usage.OutputTokens,
	}
	return nil, nil
}

// handleContentBlockStart emits output_item.added and content_part.added
// for text blocks, output_item.added for tool_use, output_item.added for thinking.
func (sc *AnthropicStreamConverter) handleContentBlockStart(data []byte) ([]byte, error) {
	var evt struct {
		Index int `json:"index"`
		ContentBlock struct {
			Type     string `json:"type"` // "text", "tool_use", "thinking"
			ID       string `json:"id,omitempty"`
			Name     string `json:"name,omitempty"`
			Thinking string `json:"thinking,omitempty"`
		} `json:"content_block"`
	}
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, nil
	}

	sc.contentBlockTypes[evt.Index] = evt.ContentBlock.Type
	var events [][]byte

	switch evt.ContentBlock.Type {
	case "text":
		sc.itemIndex++
		sc.textItemIx = sc.itemIndex
		sc.partIndex++
		sc.textPartIx = sc.partIndex
		item := ResponsesOutputItem{
			ID: fmt.Sprintf("%s.msg", sc.response.ID), Type: "message",
			Status: "in_progress", Role: "assistant",
		}
		sc.response.Output = append(sc.response.Output, item)
		evt1, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemAdded, Item: &item})
		events = append(events, eventBytes(ResponseOutputItemAdded, string(evt1)))
		evt2, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseContentPartAdded, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), PartIndex: &sc.textPartIx})
		events = append(events, eventBytes(ResponseContentPartAdded, string(evt2)))

	case "tool_use":
		sc.itemIndex++
		toolID := evt.ContentBlock.ID
		if toolID == "" {
			toolID = ensureToolID("")
		}
		sc.toolCallByIndex[evt.Index] = &fcState{
			ID: toolID, Name: evt.ContentBlock.Name, ItemIx: sc.itemIndex,
		}
		item := ResponsesOutputItem{
			ID: toolID, Type: "function_call", Status: "in_progress",
			CallID: toolID, Name: evt.ContentBlock.Name,
		}
		sc.response.Output = append(sc.response.Output, item)
		evt1, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemAdded, Item: &item})
		events = append(events, eventBytes(ResponseOutputItemAdded, string(evt1)))

	case "thinking":
		sc.itemIndex++
		sc.reasonItemIx = sc.itemIndex
		item := ResponsesOutputItem{
			ID: fmt.Sprintf("%s.reasoning", sc.response.ID), Type: "reasoning",
			Status: "in_progress",
		}
		sc.response.Output = append(sc.response.Output, item)
		evt1, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemAdded, OutputIndex: &sc.reasonItemIx, Item: &item})
		events = append(events, eventBytes(ResponseOutputItemAdded, string(evt1)))

		// Emit reasoning_summary_part.added.
		sc.reasonPartIx = 0
		evt2, _ := json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryPartAdded,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			Part:         ResponsesReasoningSummary{Type: "summary_text", Text: ""},
		})
		events = append(events, eventBytes(ResponseReasoningSummaryPartAdded, string(evt2)))

		// Anthropic sends initial thinking content at block start.
		if evt.ContentBlock.Thinking != "" {
			sc.accReasoning += evt.ContentBlock.Thinking
			evt3, _ := json.Marshal(ResponsesStreamEvent{
				Type:         ResponseReasoningSummaryDelta,
				ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
				OutputIndex:  &sc.reasonItemIx,
				SummaryIndex: &sc.reasonPartIx,
				Delta:        evt.ContentBlock.Thinking,
			})
			events = append(events, eventBytes(ResponseReasoningSummaryDelta, string(evt3)))
		}
	}

	if len(events) == 0 {
		return nil, nil
	}
	return bytes.Join(events, []byte("\n\n")), nil
}

// handleContentBlockDelta emits text deltas, function_call argument deltas,
// or reasoning summary deltas based on the block type.
func (sc *AnthropicStreamConverter) handleContentBlockDelta(data []byte) ([]byte, error) {
	var evt struct {
		Index int             `json:"index"`
		Delta json.RawMessage `json:"delta"`
	}
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, nil
	}

	blockType := sc.contentBlockTypes[evt.Index]

	switch blockType {
	case "text":
		var delta struct {
			Type string `json:"type"` // "text_delta"
			Text string `json:"text"`
		}
		if err := json.Unmarshal(evt.Delta, &delta); err != nil || delta.Text == "" {
			return nil, nil
		}
		sc.accText += delta.Text
		evtBytes, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputTextDelta, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), Delta: delta.Text})
		return eventBytes(ResponseOutputTextDelta, string(evtBytes)), nil

	case "tool_use":
		var delta struct {
			Type        string `json:"type"` // "input_json_delta"
			PartialJSON string `json:"partial_json"`
		}
		if err := json.Unmarshal(evt.Delta, &delta); err != nil || delta.PartialJSON == "" {
			return nil, nil
		}
		if fc, ok := sc.toolCallByIndex[evt.Index]; ok {
			fc.Arguments += delta.PartialJSON
		}
		itemID := ""
		if fc, ok := sc.toolCallByIndex[evt.Index]; ok {
			itemID = fc.ID
		}
		evtBytes, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDelta, ItemID: itemID, Arguments: delta.PartialJSON})
		return eventBytes(ResponseFunctionCallArgumentsDelta, string(evtBytes)), nil

	case "thinking":
		var delta struct {
			Type     string `json:"type"` // "thinking_delta"
			Thinking string `json:"thinking"`
		}
		if err := json.Unmarshal(evt.Delta, &delta); err != nil || delta.Thinking == "" {
			return nil, nil
		}
		sc.accReasoning += delta.Thinking
		evtBytes, _ := json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryDelta,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			Delta:        delta.Thinking,
		})
		return eventBytes(ResponseReasoningSummaryDelta, string(evtBytes)), nil
	}

	return nil, nil
}

// handleMessageDelta captures finish_reason and output token usage.
// EmitError emits a response.failed event and marks the stream as finalized.
func (sc *AnthropicStreamConverter) EmitError(message string) []byte {
	if sc.finalized {
		return nil
	}
	sc.finalized = true
	sc.response.Status = "failed"
	sc.response.Error = map[string]string{
		"code":    "server_error",
		"message": message,
	}
	evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFailed, Response: &sc.response})
	return eventBytes(ResponseFailed, string(evt))
}

func (sc *AnthropicStreamConverter) handleMessageDelta(data []byte) ([]byte, error) {
	var evt struct {
		Delta struct {
			StopReason   string `json:"stop_reason"`
			StopSequence string `json:"stop_sequence"`
		} `json:"delta"`
		Usage struct {
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &evt); err != nil {
		return nil, nil
	}

	if evt.Delta.StopReason != "" {
		switch evt.Delta.StopReason {
		case "end_turn", "stop_sequence":
			sc.response.Status = "completed"
		case "max_tokens":
			sc.response.Status = "incomplete"
		case "tool_use":
			sc.response.Status = "completed"
		default:
			sc.response.Status = "completed"
		}
	}

	if evt.Usage.OutputTokens > 0 && sc.response.Usage != nil {
		sc.response.Usage.OutputTokens = evt.Usage.OutputTokens
		sc.response.Usage.TotalTokens = sc.response.Usage.InputTokens + evt.Usage.OutputTokens
	}
	return nil, nil
}
