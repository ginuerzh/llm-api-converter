package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
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
	seqNum         int // auto-incremented per emitted event

	toolSpecs map[string]codexToolSpec // tool name → spec for output item type mapping
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
			ID:                "resp_" + randHex(16),
			Object:            "response",
			CreatedAt:         time.Now().Unix(),
			Status:            "in_progress",
			Model:             model,
			Text:              map[string]any{"format": map[string]any{"type": "text"}},
			Truncation:        "disabled",
			ToolChoice:        "auto",
			ParallelToolCalls: boolPtr(true),
				Reasoning:         map[string]any{"effort": nil, "summary": nil},
		},
		toolCallByIndex: make(map[int]*fcState),
		toolSpecs:       make(map[string]codexToolSpec),
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
	e1 := sc.makeEvent(ResponseCreated, string(created))
	e2 := sc.makeEvent(ResponseInProgress, string(inProg))
	return append(append(e1, '\n', '\n'), e2...)
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
		text, thinkText := splitInlineThink(delta.Content)
		if thinkText != "" {
			sc.accReasoning += thinkText
			if sc.reasonItemIx < 0 {
				events = append(events, sc.startReasoningItem()...)
				sc.reasonPartIx = 0
				pevt, _ := json.Marshal(ResponsesStreamEvent{
					Type:         ResponseReasoningSummaryPartAdded,
					ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
					OutputIndex:  &sc.reasonItemIx,
					SummaryIndex: &sc.reasonPartIx,
					Part:         ResponsesReasoningSummary{Type: "summary_text", Text: ""},
				})
				events = append(events, sc.makeEvent(ResponseReasoningSummaryPartAdded, string(pevt)))
			}
			evt, _ := json.Marshal(ResponsesStreamEvent{
				Type:         ResponseReasoningSummaryDelta,
				ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
				OutputIndex:  &sc.reasonItemIx,
				SummaryIndex: &sc.reasonPartIx,
				Delta:        thinkText,
			})
			events = append(events, sc.makeEvent(ResponseReasoningSummaryDelta, string(evt)))
		}
		if text != "" {
			sc.accText += text
			if sc.textItemIx < 0 {
				events = append(events, sc.startMessageItem()...)
			}
			if sc.textPartIx < 0 {
				events = append(events, sc.startContentPart()...)
			}
			evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputTextDelta, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), OutputIndex: &sc.textItemIx, ContentIndex: &sc.textPartIx, Delta: text})
			events = append(events, sc.makeEvent(ResponseOutputTextDelta, string(evt)))
		}
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
			events = append(events, sc.makeEvent(ResponseReasoningSummaryPartAdded, string(pevt)))
		}
		evt, _ := json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryDelta,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			Delta:        delta.ReasoningContent,
		})
		events = append(events, sc.makeEvent(ResponseReasoningSummaryDelta, string(evt)))
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

	// Finalize reasoning item.
	if sc.reasonItemIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryDone,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			SummaryText:  sc.accReasoning,
		})
		events = append(events, sc.makeEvent(ResponseReasoningSummaryDone, string(evt)))
		evt, _ = json.Marshal(ResponsesStreamEvent{
			Type:         ResponseReasoningSummaryPartDone,
			ItemID:       fmt.Sprintf("%s.reasoning", sc.response.ID),
			OutputIndex:  &sc.reasonItemIx,
			SummaryIndex: &sc.reasonPartIx,
			Part:         ResponsesReasoningSummary{Type: "summary_text", Text: sc.accReasoning},
		})
		events = append(events, sc.makeEvent(ResponseReasoningSummaryPartDone, string(evt)))
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
		events = append(events, sc.makeEvent(ResponseOutputItemDone, string(evt)))
	}
	// Finalize message item (output_index 1 after output_index 0).
	if sc.textPartIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseContentPartDone, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), OutputIndex: &sc.textItemIx, ContentIndex: &sc.textPartIx, PartIndex: &sc.textPartIx})
		events = append(events, sc.makeEvent(ResponseContentPartDone, string(evt)))
		evt, _ = json.Marshal(ResponsesStreamEvent{Type: ResponseOutputTextDone, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), OutputIndex: &sc.textItemIx, ContentIndex: &sc.textPartIx, TextDone: sc.accText})
		events = append(events, sc.makeEvent(ResponseOutputTextDone, string(evt)))
	}
	if sc.textItemIx >= 0 {
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemDone, OutputIndex: &sc.textItemIx, Item: &ResponsesOutputItem{ID: fmt.Sprintf("%s.msg", sc.response.ID), Type: "message", Status: "completed", Role: "assistant", Content: []ResponsesContentPart{{Type: "output_text", Text: sc.accText}}}})
		events = append(events, sc.makeEvent(ResponseOutputItemDone, string(evt)))
	}
	for _, fc := range sc.toolCallByIndex {
		if fc.Arguments != "" {
			evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDone, ItemID: fc.ID, OutputIndex: &fc.ItemIx, Arguments: canonicalJSONString(fc.Arguments)})
			events = append(events, sc.makeEvent(ResponseFunctionCallArgumentsDone, string(evt)))
		}
		// ponytail: codex requires a complete, deserializable function_call item
		// (with arguments) in output_item.done — it ignores *_arguments.delta/.done
		// and silently drops items that fail to deserialize, losing the tool call.
		item := ResponsesOutputItem{ID: fc.ID, Type: "function_call", Status: "completed", CallID: fc.ID, Name: fc.Name, Arguments: canonicalJSONString(fc.Arguments)}
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemDone, OutputIndex: &fc.ItemIx, Item: &item})
		events = append(events, sc.makeEvent(ResponseOutputItemDone, string(evt)))
	}

	// Update output item statuses before building response.completed.
	// Items are stored with "in_progress" when first created; individual
	// response.output_item.done events were emitted, but sc.response.Output
	// was never updated — so response.completed would report every item as
	// "in_progress", breaking clients that validate item completion.
	for i := range sc.response.Output {
		item := &sc.response.Output[i]
		if item.Status == "in_progress" {
			item.Status = "completed"
		}
		if item.Type == "message" && sc.accText != "" {
			item.Content = []ResponsesContentPart{{Type: "output_text", Text: sc.accText}}
		}
		if item.Type == "function_call" && item.Name != "" {
			item.Type = sc.toolTypeFromSpec(item.Name)
		}
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
	events = append(events, sc.makeEvent(eventType, string(completed)))
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
	return [][]byte{sc.makeEvent(ResponseOutputItemAdded, string(evt))}
}

func (sc *ResponsesStreamConverter) startContentPart() [][]byte {
	sc.partIndex++
	sc.textPartIx = sc.partIndex
	evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseContentPartAdded, ItemID: fmt.Sprintf("%s.msg", sc.response.ID), OutputIndex: &sc.textItemIx, ContentIndex: &sc.textPartIx, PartIndex: &sc.textPartIx})
	return [][]byte{sc.makeEvent(ResponseContentPartAdded, string(evt))}
}

func (sc *ResponsesStreamConverter) startReasoningItem() [][]byte {
	sc.itemIndex++
	sc.reasonItemIx = sc.itemIndex
	item := ResponsesOutputItem{
		ID: fmt.Sprintf("%s.reasoning", sc.response.ID), Type: "reasoning", Status: "in_progress",
	}
	sc.response.Output = append(sc.response.Output, item)
	evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemAdded, OutputIndex: &sc.reasonItemIx, Item: &item})
	return [][]byte{sc.makeEvent(ResponseOutputItemAdded, string(evt))}
}

// EmitError emits a response.failed event and marks the stream as finalized.
func (sc *ResponsesStreamConverter) EmitError(message string) []byte {
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
	return sc.makeEvent(ResponseFailed, string(evt))
}

func (sc *ResponsesStreamConverter) handleToolCall(tc OpenAIDeltaToolCall) [][]byte {
	idx := tc.Index
	if existing, seen := sc.toolCallByIndex[idx]; seen {
		existing.Arguments += tc.Function.Arguments
		evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDelta, ItemID: existing.ID, OutputIndex: &existing.ItemIx, Arguments: tc.Function.Arguments})
		return [][]byte{sc.makeEvent(ResponseFunctionCallArgumentsDelta, string(evt))}
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
	events := [][]byte{sc.makeEvent(ResponseOutputItemAdded, string(evt))}

	if tc.Function.Arguments != "" {
		sc.toolCallByIndex[idx].Arguments = tc.Function.Arguments
		devt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDelta, ItemID: toolID, OutputIndex: &sc.itemIndex, Arguments: tc.Function.Arguments})
		events = append(events, sc.makeEvent(ResponseFunctionCallArgumentsDelta, string(devt)))
	}
	return events
}

func (sc *ResponsesStreamConverter) setUsage(usage *OpenAIUsage) {
	sc.response.Usage = &ResponsesUsage{
		InputTokens: usage.PromptTokens, OutputTokens: usage.CompletionTokens,
		TotalTokens: usage.TotalTokens,
	}
}

// makeEvent builds an SSE data line with an auto-incremented sequence_number.
func (sc *ResponsesStreamConverter) makeEvent(evtType ResponsesStreamEventType, data string) []byte {
	seq := sc.seqNum
	sc.seqNum++
	// Inject sequence_number after the opening brace.
	injected := fmt.Sprintf(`{"sequence_number":%d,%s`, seq, data[1:])
	return []byte("event: " + string(evtType) + "\ndata: " + injected)
}

func eventBytes(evtType ResponsesStreamEventType, data string) []byte {
	return []byte("event: " + string(evtType) + "\ndata: " + data)
}

func boolPtr(b bool) *bool { return &b }

// splitInlineThink splits content on <think>...</think> prefix into
// the think text and any remaining text. Returns (remaining, thinkContent).
func splitInlineThink(content string) (string, string) {
	const thinkOpen = "<think>"
	const thinkClose = "</think>"
	if !strings.HasPrefix(content, thinkOpen) {
		return content, ""
	}
	rest := content[len(thinkOpen):]
	closeIdx := strings.Index(rest, thinkClose)
	if closeIdx < 0 {
		// Opening think without close — treat whole rest as thinking.
		return "", rest
	}
	return rest[closeIdx+len(thinkClose):], rest[:closeIdx]
}


// SetToolSpecs sets the tool specs for output item type mapping.
func (sc *ResponsesStreamConverter) SetToolSpecs(specs map[string]codexToolSpec) {
	sc.toolSpecs = specs
}

// toolTypeFromSpec returns the Responses API output item type for a tool name.
func (sc *ResponsesStreamConverter) toolTypeFromSpec(name string) string {
	if sc.toolSpecs == nil {
		return "function_call"
	}
	spec, ok := sc.toolSpecs[name]
	if !ok {
		return "function_call"
	}
	switch spec.Kind {
	case "custom":
		return "custom_tool_call"
	case "tool_search":
		return "tool_search_call"
	default:
		return "function_call"
	}
}
