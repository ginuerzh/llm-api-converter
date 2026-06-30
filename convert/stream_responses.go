package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"
)

// ResponsesStreamConverter converts OpenAI streaming delta chunks into the
// Responses API SSE event sequence.
type ResponsesStreamConverter struct {
	model    string
	response ResponsesResponse

	started   bool
	finalized bool

	accText      string
	accReasoning string

	itemIndex     int
	partIndex     int
	textItemIx    int // -1 = none started
	textPartIx    int // -1 = none started
	reasonItemIx  int // -1 = none
	reasonPartIx  int // -1 = none
	toolCallByIndex map[int]*fcState
}

type fcState struct {
	ID        string
	Name      string
	Arguments string
	ItemIx    int
}

// NewResponsesStreamConverter creates a new converter for the given model.
func NewResponsesStreamConverter(model string) *ResponsesStreamConverter {
	return &ResponsesStreamConverter{
		model: model,
		response: ResponsesResponse{
			ID:     "resp_" + randHex(16),
			Object: "response",
			Status: "in_progress",
			Model:  model,
		},
		toolCallByIndex: make(map[int]*fcState),
		textItemIx:      -1,
		textPartIx:      -1,
		reasonItemIx:    -1,
		reasonPartIx:    -1,
		itemIndex:       -1,
		partIndex:       -1,
	}
}

// HandleStreamStart returns response.created and response.in_progress events.
func (sc *ResponsesStreamConverter) HandleStreamStart() []byte {
	if sc.started {
		return nil
	}
	sc.started = true
	created, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseCreated, Response: &sc.response})
	inProg, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseInProgress, Response: &sc.response})
	return []byte("event: response.created\ndata: " + string(created) + "\n\nevent: response.in_progress\ndata: " + string(inProg))
}

// HandleChunk processes one streaming delta chunk and returns Responses SSE events.
func (sc *ResponsesStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if sc.finalized {
		return nil, nil
	}
	var chunk OpenAIStreamChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil, fmt.Errorf("unmarshal stream chunk: %w", err)
	}
	if len(chunk.Choices) == 0 {
		if chunk.Usage != nil {
			sc.setUsage(chunk.Usage)
		}
		return []byte{}, nil // usage-only/cost chunk, consumed
	}

	choice := chunk.Choices[0]
	if chunk.Usage != nil {
		sc.setUsage(chunk.Usage)
	}
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		sc.response.Status = responsesFinishStatus(choice.FinishReason)
		if *choice.FinishReason == "tool_calls" {
			sc.response.Status = "completed"
		}
	}

	delta := choice.Delta
	var events [][]byte

	if delta.Content != "" {
		sc.accText += delta.Content
		if sc.textItemIx < 0 {
			events = append(events, sc.startMessageItem()...)
		}
		if sc.textPartIx < 0 {
			events = append(events, sc.startContentPart()...)
		}
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputTextDelta, ContentIndex: &sc.textPartIx, Delta: delta.Content})
		events = append(events, eventBytes(ResponseOutputTextDelta, string(evt)))
	}
	if delta.ReasoningContent != "" {
		sc.accReasoning += delta.ReasoningContent
		if sc.reasonItemIx < 0 {
			events = append(events, sc.startReasoningItem()...)
			// Emit reasoning_summary_part.added.
			sc.reasonPartIx = 0
			pevt, _ := json.Marshal(ResponsesStreamEvent{
				Type:         ResponseReasoningSummaryPartAdded,
				ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
				OutputIndex:  &sc.reasonItemIx,
				SummaryIndex: &sc.reasonPartIx,
				Part:         ResponsesReasoningSummary{Type: "summary_text", Text: ""},
			})
			events = append(events, eventBytes(ResponseReasoningSummaryPartAdded, string(pevt)))
		}
		evt, _ := json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryDelta,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			Delta:        delta.ReasoningContent,
		})
		events = append(events, eventBytes(ResponseReasoningSummaryDelta, string(evt)))
	}
	for _, tc := range delta.ToolCalls {
		events = append(events, sc.handleToolCall(tc)...)
	}

	if len(events) == 0 {
		// Chunk consumed (e.g., finish_reason, usage-only, empty content) but
		// no Responses SSE events to emit. Return empty non-nil to swallow.
		return []byte{}, nil
	}
	return bytes.Join(events, []byte("\n\n")), nil
}

// HandleStreamEnd returns finalization events and response.completed.
func (sc *ResponsesStreamConverter) HandleStreamEnd() []byte {
	if sc.finalized {
		return nil
	}
	sc.finalized = true
	var events [][]byte

	if sc.textPartIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseContentPartDone, OutputIndex: &sc.textItemIx, ContentIndex: &sc.textPartIx, PartIndex: &sc.textPartIx})
		events = append(events, eventBytes(ResponseContentPartDone, string(evt)))
		evt, _ = json.Marshal(ResponsesStreamEvent{Type: ResponseOutputTextDone, ContentIndex: &sc.textPartIx, TextDone: sc.accText})
		events = append(events, eventBytes(ResponseOutputTextDone, string(evt)))
	}
	if sc.textItemIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemDone, OutputIndex: &sc.textItemIx})
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
	for _, fc := range sc.toolCallByIndex {
		if fc.Arguments != "" {
			evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDone, Arguments: canonicalJSONString(fc.Arguments)})
			events = append(events, eventBytes(ResponseFunctionCallArgumentsDone, string(evt)))
		}
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemDone, OutputIndex: &fc.ItemIx})
		events = append(events, eventBytes(ResponseOutputItemDone, string(evt)))
	}

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

func (sc *ResponsesStreamConverter) startMessageItem() [][]byte {
	sc.itemIndex++
	sc.textItemIx = sc.itemIndex
	item := ResponsesOutputItem{
		ID: fmt.Sprintf("%s.msg", sc.response.ID), Type: "message",
		Status: "in_progress", Role: "assistant",
	}
	sc.response.Output = append(sc.response.Output, item)
	evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemAdded, OutputIndex: &sc.textItemIx, Item: &item})
	return [][]byte{eventBytes(ResponseOutputItemAdded, string(evt))}
}

func (sc *ResponsesStreamConverter) startContentPart() [][]byte {
	sc.partIndex++
	sc.textPartIx = sc.partIndex
	evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseContentPartAdded, OutputIndex: &sc.textItemIx, ContentIndex: &sc.textPartIx, PartIndex: &sc.textPartIx})
	return [][]byte{eventBytes(ResponseContentPartAdded, string(evt))}
}

func (sc *ResponsesStreamConverter) startReasoningItem() [][]byte {
	sc.itemIndex++
	sc.reasonItemIx = sc.itemIndex
	item := ResponsesOutputItem{
		ID: fmt.Sprintf("%s.reasoning", sc.response.ID), Type: "reasoning", Status: "in_progress",
	}
	sc.response.Output = append(sc.response.Output, item)
	evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemAdded, OutputIndex: &sc.reasonItemIx, Item: &item})
	return [][]byte{eventBytes(ResponseOutputItemAdded, string(evt))}
}

func (sc *ResponsesStreamConverter) handleToolCall(tc OpenAIDeltaToolCall) [][]byte {
	idx := tc.Index
	if existing, seen := sc.toolCallByIndex[idx]; seen {
		existing.Arguments += tc.Function.Arguments
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDelta, Arguments: tc.Function.Arguments})
		return [][]byte{eventBytes(ResponseFunctionCallArgumentsDelta, string(evt))}
	}

	toolID := tc.ID
	if toolID == "" {
		toolID = ensureToolID("")
	}
	sc.itemIndex++
	sc.toolCallByIndex[idx] = &fcState{ID: toolID, Name: tc.Function.Name, ItemIx: sc.itemIndex}
	item := ResponsesOutputItem{
		ID: toolID, Type: "function_call", Status: "in_progress",
		CallID: toolID, Name: tc.Function.Name,
	}
	sc.response.Output = append(sc.response.Output, item)
	evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemAdded, OutputIndex: &sc.itemIndex, Item: &item})
	events := [][]byte{eventBytes(ResponseOutputItemAdded, string(evt))}

	if tc.Function.Arguments != "" {
		sc.toolCallByIndex[idx].Arguments = tc.Function.Arguments
		devt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDelta, Arguments: tc.Function.Arguments})
		events = append(events, eventBytes(ResponseFunctionCallArgumentsDelta, string(devt)))
	}
	return events
}

func (sc *ResponsesStreamConverter) setUsage(usage *OpenAIUsage) {
	sc.response.Usage = &ResponsesUsage{
		InputTokens: usage.PromptTokens, OutputTokens: usage.CompletionTokens,
		TotalTokens: usage.TotalTokens,
	}
}

func eventBytes(evtType ResponsesStreamEventType, data string) []byte {
	return []byte("event: " + string(evtType) + "\ndata: " + data)
}
