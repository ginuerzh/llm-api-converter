package convert

import (
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// StreamConverter: HandleStreamStart
// ---------------------------------------------------------------------------

func TestStream_Start(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	out := sc.HandleStreamStart()

	if !strings.Contains(string(out), "event: message_start") {
		t.Fatalf("expected message_start event, got: %s", out)
	}
	if !strings.Contains(string(out), "event: ping") {
		t.Fatalf("expected ping event, got: %s", out)
	}
	if !strings.Contains(string(out), `"model":"claude-sonnet-4-20250514"`) {
		t.Fatalf("expected model in message_start, got: %s", out)
	}
}

// ---------------------------------------------------------------------------
// StreamConverter: HandleChunk with text content
// ---------------------------------------------------------------------------

func TestStream_TextDelta(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart() // establish state

	chunk := `{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`
	out, err := sc.HandleChunk([]byte(chunk))
	if err != nil {
		t.Fatal(err)
	}

	// Should emit content_block_start + content_block_delta
	s := string(out)
	if !strings.Contains(s, "content_block_start") {
		t.Fatalf("expected content_block_start, got: %s", s)
	}
	if !strings.Contains(s, "content_block_delta") {
		t.Fatalf("expected content_block_delta, got: %s", s)
	}
	if !strings.Contains(s, "text_delta") || !strings.Contains(s, "Hello") {
		t.Fatalf("expected text_delta with Hello, got: %s", s)
	}
}

func TestStream_TextMultipleChunks(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	chunk1 := `{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`
	out1, _ := sc.HandleChunk([]byte(chunk1))
	if !strings.Contains(string(out1), "content_block_start") {
		t.Fatalf("first chunk should start block, got: %s", out1)
	}

	// Second text chunk — should only be content_block_delta, no new start.
	chunk2 := `{"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`
	out2, _ := sc.HandleChunk([]byte(chunk2))
	if strings.Contains(string(out2), "content_block_start") {
		t.Fatalf("second chunk should NOT start a new block, got: %s", out2)
	}
	if !strings.Contains(string(out2), "text_delta") || !strings.Contains(string(out2), " world") {
		t.Fatalf("expected text_delta with ' world', got: %s", out2)
	}

	// Third text chunk — same.
	chunk3 := `{"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}`
	out3, _ := sc.HandleChunk([]byte(chunk3))
	if strings.Contains(string(out3), "content_block_start") {
		t.Fatalf("third chunk should NOT start a new block, got: %s", out3)
	}
}

// ---------------------------------------------------------------------------
// StreamConverter: HandleChunk with finish_reason
// ---------------------------------------------------------------------------

func TestStream_FinishReason(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// Text chunk.
	chunk1 := `{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`
	_, _ = sc.HandleChunk([]byte(chunk1))

	// Finish chunk.
	chunk2 := `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	_, _ = sc.HandleChunk([]byte(chunk2))

	// HandleStreamEnd should include message_delta with end_turn (mapped from stop).
	endOut := sc.HandleStreamEnd()
	s := string(endOut)
	if !strings.Contains(s, "message_delta") {
		t.Fatalf("expected message_delta, got: %s", s)
	}
	if !strings.Contains(s, "end_turn") {
		t.Fatalf("expected end_turn stop_reason, got: %s", s)
	}
	if !strings.Contains(s, "message_stop") {
		t.Fatalf("expected message_stop, got: %s", s)
	}
	// Should also have content_block_stop for the text block.
	if !strings.Contains(s, "content_block_stop") {
		t.Fatalf("expected content_block_stop before message_delta, got: %s", s)
	}
}

func TestStream_FinishReasonToolCalls(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// Text then tool_calls then finish.
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"Let me check"},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`))

	endOut := sc.HandleStreamEnd()
	s := string(endOut)
	if !strings.Contains(s, "tool_use") {
		t.Fatalf("expected tool_use stop_reason in message_delta, got: %s", s)
	}
}

// ---------------------------------------------------------------------------
// StreamConverter: thinking → text transition
// ---------------------------------------------------------------------------

func TestStream_ThinkingToText(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// Thinking delta.
	chunk1 := `{"choices":[{"index":0,"delta":{"reasoning_content":"thinking steps..."},"finish_reason":null}]}`
	out1, _ := sc.HandleChunk([]byte(chunk1))
	if !strings.Contains(string(out1), "thinking_delta") {
		t.Fatalf("expected thinking_delta, got: %s", out1)
	}

	// Text delta — should trigger content_block_stop for thinking + content_block_start for text.
	chunk2 := `{"choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}`
	out2, _ := sc.HandleChunk([]byte(chunk2))
	s := string(out2)
	if !strings.Contains(s, "content_block_stop") {
		t.Fatalf("expected content_block_stop for thinking block, got: %s", s)
	}
	if !strings.Contains(s, "content_block_start") {
		t.Fatalf("expected content_block_start for text block, got: %s", s)
	}
	if !strings.Contains(s, "text_delta") {
		t.Fatalf("expected text_delta, got: %s", s)
	}
}

// ---------------------------------------------------------------------------
// StreamConverter: tool call accumulation
// ---------------------------------------------------------------------------

func TestStream_ToolCallAccumulation(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// First delta: tool call start with ID, name, empty arguments.
	chunk1 := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`
	out1, _ := sc.HandleChunk([]byte(chunk1))
	s1 := string(out1)
	if !strings.Contains(s1, "content_block_start") {
		t.Fatalf("expected content_block_start for tool call, got: %s", s1)
	}
	if !strings.Contains(s1, "tool_use") {
		t.Fatalf("expected tool_use block, got: %s", s1)
	}
	if !strings.Contains(s1, "get_weather") {
		t.Fatalf("expected tool name get_weather, got: %s", s1)
	}

	// Second delta: more arguments.
	chunk2 := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc\":"}}]},"finish_reason":null}]}`
	out2, _ := sc.HandleChunk([]byte(chunk2))
	s2 := string(out2)
	if !strings.Contains(s2, "input_json_delta") {
		t.Fatalf("expected input_json_delta, got: %s", s2)
	}

	// Third delta: final arguments.
	chunk3 := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"NYC\"}"}}]},"finish_reason":null}]}`
	out3, _ := sc.HandleChunk([]byte(chunk3))
	s3 := string(out3)
	if !strings.Contains(s3, "input_json_delta") {
		t.Fatalf("expected input_json_delta, got: %s", s3)
	}
	if !strings.Contains(s3, "NYC") {
		t.Fatalf("expected NYC in input_json_delta, got: %s", s3)
	}

	// End the stream.
	endOut := sc.HandleStreamEnd()
	if !strings.Contains(string(endOut), "content_block_stop") {
		t.Fatalf("expected content_block_stop at end, got: %s", endOut)
	}
}

func TestStream_MultipleToolCalls(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// Tool call 0 starts.
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"fn_a","arguments":""}}]},"finish_reason":null}]}`))
	// Arguments for tool call 0.
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]},"finish_reason":null}]}`))

	// Tool call 1 starts (same chunk as tool call 0 continues but with index 1).
	chunk3 := `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"fn_b","arguments":"{\"y\":2}"}}]},"finish_reason":null}]}`
	out3, _ := sc.HandleChunk([]byte(chunk3))
	s3 := string(out3)
	// Should have content_block_stop for tool call 0's block, then content_block_start for tool call 1.
	if !strings.Contains(s3, "content_block_stop") {
		t.Fatalf("expected content_block_stop when switching tool calls, got: %s", s3)
	}
	if !strings.Contains(s3, "content_block_start") {
		t.Fatalf("expected content_block_start for new tool call, got: %s", s3)
	}
	if !strings.Contains(s3, "fn_b") {
		t.Fatalf("expected fn_b, got: %s", s3)
	}
}

// ---------------------------------------------------------------------------
// StreamConverter: edge cases
// ---------------------------------------------------------------------------

func TestStream_EmptyOrRoleDelta(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// Role-only delta (role announcement) — should return nil.
	chunk := `{"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`
	out, err := sc.HandleChunk([]byte(chunk))
	if err != nil {
		t.Fatal(err)
	}
	if out != nil {
		t.Fatalf("expected nil for role-only delta, got: %s", out)
	}
}

func TestStream_AfterFinalized(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// End the stream.
	end := sc.HandleStreamEnd()
	if len(end) == 0 {
		t.Fatal("expected non-empty end events")
	}

	// Second HandleStreamEnd should return nil.
	secondEnd := sc.HandleStreamEnd()
	if secondEnd != nil {
		t.Fatalf("expected nil after finalized, got: %s", secondEnd)
	}

	// HandleChunk after finalized should return nil.
	chunk := `{"choices":[{"index":0,"delta":{"content":"more"},"finish_reason":null}]}`
	out, _ := sc.HandleChunk([]byte(chunk))
	if out != nil {
		t.Fatalf("expected nil after finalized, got: %s", out)
	}
}

// ---------------------------------------------------------------------------
// HandleSSEEvent integration
// ---------------------------------------------------------------------------

func TestHandleSSEEvent_TextStream(t *testing.T) {
	sid := "test_sid_001"
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, SessionStore: NewSessionStore()}

	// Start phase.
	startOut, err := HandleSSEEvent(sid, "start", 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(startOut), "event: message_start") {
		t.Fatalf("expected message_start, got: %s", startOut)
	}
	if !strings.Contains(string(startOut), "event: ping") {
		t.Fatalf("expected ping, got: %s", startOut)
	}

	// Event phase: text chunk.
	chunk := []byte(`{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`)
	eventOut, err := HandleSSEEvent(sid, "event", 0, chunk, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(eventOut), "text_delta") {
		t.Fatalf("expected text_delta, got: %s", eventOut)
	}

	// Event phase: finish.
	finishChunk := []byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	_, _ = HandleSSEEvent(sid, "event", 1, finishChunk, opts)

	// End phase.
	endOut, err := HandleSSEEvent(sid, "end", 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(endOut), "message_delta") {
		t.Fatalf("expected message_delta, got: %s", endOut)
	}
	if !strings.Contains(string(endOut), "message_stop") {
		t.Fatalf("expected message_stop, got: %s", endOut)
	}
}

func TestHandleSSEEvent_UnknownSID(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, SessionStore: NewSessionStore()}

	// Event for unknown SID should auto-create the stream.
	chunk := []byte(`{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`)
	out, err := HandleSSEEvent("unknown_sid", "event", 0, chunk, opts)
	if err != nil {
		t.Fatal(err)
	}
	// Should include start events + chunk events.
	if !strings.Contains(string(out), "message_start") {
		t.Fatalf("expected message_start for auto-created stream, got: %s", out)
	}
	if !strings.Contains(string(out), "content_block_delta") {
		t.Fatalf("expected content_block_delta, got: %s", out)
	}

	// End should work even though it wasn't explicitly started.
	endOut, err := HandleSSEEvent("unknown_sid", "end", 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(endOut), "message_stop") {
		t.Fatalf("expected message_stop, got: %s", endOut)
	}
}

func TestHandleSSEEvent_InvalidPhase(t *testing.T) {
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192}
	_, err := HandleSSEEvent("sid", "invalid", 0, nil, opts)
	if err == nil {
		t.Fatal("expected error for invalid phase")
	}
}

// ---------------------------------------------------------------------------
// Block index assignment
// ---------------------------------------------------------------------------

func TestStream_BlockIndices(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	// Thinking block should get index 0.
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}`))
	if sc.curBlockIndex != 0 || sc.curBlockType != "thinking" {
		t.Fatalf("expected thinking block index 0, got type=%s index=%d", sc.curBlockType, sc.curBlockIndex)
	}

	// Text block should get index 1.
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"text"},"finish_reason":null}]}`))
	if sc.curBlockIndex != 1 || sc.curBlockType != "text" {
		t.Fatalf("expected text block index 1, got type=%s index=%d", sc.curBlockType, sc.curBlockIndex)
	}

	// Tool call should get index 2.
	sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
	if sc.curBlockIndex != 2 || sc.curBlockType != "tool_use" {
		t.Fatalf("expected tool_use block index 2, got type=%s index=%d", sc.curBlockType, sc.curBlockIndex)
	}
}

func TestStream_ThinkingTextToolCallsEnd(t *testing.T) {
	// Full sequence: thinking → text → tool_calls → finish
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil,
	nil)
	sc.HandleStreamStart()

	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":"reasoning..."},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}]},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`))

	end := sc.HandleStreamEnd()
	s := string(end)
	// Expect 3 content_block_stop events (for thinking, text, tool_use) — but only the current (tool_use)
	// block gets a stop. The earlier blocks already got stopped when transitioning.
	// So 1 content_block_stop at the end.
	if !strings.Contains(s, "content_block_stop") {
		t.Fatalf("expected content_block_stop, got: %s", s)
	}
	if !strings.Contains(s, "message_delta") {
		t.Fatalf("expected message_delta, got: %s", s)
	}
	if !strings.Contains(s, "message_stop") {
		t.Fatalf("expected message_stop, got: %s", s)
	}
}

// ---------------------------------------------------------------------------
// ReasoningCache: stream accumulation
// ---------------------------------------------------------------------------

func TestStream_ReasoningCacheOnEnd(t *testing.T) {
	rc := NewReasoningCache(100)
	sc := NewStreamConverter("claude-sonnet-4-20250514", rc, nil)
	sc.HandleStreamStart()

	// Reasoning deltas.
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":"thinking "},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":"steps..."},"finish_reason":null}]}`))

	// Tool call deltas.
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}`))
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":null}]}`))

	// Finish.
	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`))

	// End stream → should trigger cache store.
	sc.HandleStreamEnd()

	// Verify cache has accumulated reasoning for the tool call ID.
	if v, ok := rc.Get([]string{"call_xyz"}); !ok {
		t.Fatal("expected reasoning in cache after stream end")
	} else if v != "thinking steps..." {
		t.Fatalf("want 'thinking steps...', got %q", v)
	}
}
