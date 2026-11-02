package convert

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// ---------------------------------------------------------------------------
// Gemini SSE response chunk detection and conversion
//
// Gemini streaming responses come as SSE chunks with candidates[].content.parts
// in the data payload. This file detects those chunks and converts each one
// independently to the target protocol's SSE format, so the non-streaming
// response rewriter path can still produce correct streaming output.
// ---------------------------------------------------------------------------

// isGeminiStreamChunk returns true if the data payload looks like a Gemini
// generateContent streaming response chunk: an object with candidates[] whose
// content.parts carries incremental text or function calls.
func isGeminiStreamChunk(data []byte) bool {
	var probe struct {
		Candidates []struct {
			Content struct {
				Parts []GeminiPart `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return false
	}
	return len(probe.Candidates) > 0
}

// convertGeminiStreamChunk converts a single Gemini SSE response chunk to the
// target protocol's SSE format (Anthropic event sequence or OpenAI delta chunk),
// based on the model-map resolution.
func convertGeminiStreamChunk(evt *SSEEvent, opts *ConvertOptions) ([]byte, error) {
	var chunk GeminiChatResponse
	if err := json.Unmarshal([]byte(evt.Data), &chunk); err != nil {
		slog.Debug("gemini sse: unmarshal failed, passing through", "err", err)
		return reconstructSSEEvent(evt), nil
	}

	// Empty candidates → content_filter / safety stop.
	if len(chunk.Candidates) == 0 {
		slog.Debug("gemini sse: empty candidates")
		return []byte{}, nil
	}

	c := chunk.Candidates[0]

	// Determine target protocol from model map or fallback to OpenAI delta.
	target := ProtocolOpenAIChat
	if opts != nil && opts.ModelMap != nil {
		model := extractModelFromData([]byte(evt.Data))
		_, resolved := resolveModel(model, ProtocolGemini, opts.ModelMap)
		if resolved != ProtocolUnknown {
			target = resolved
		}
	}
	// Session-based override: if the request was from Anthropic client,
	// stream back Anthropic events.
	if opts != nil && opts.SessionStore != nil && opts.SID != "" {
		if sess := opts.SessionStore.Get(opts.SID); sess != nil && sess.From == ProtocolAnthropic {
			target = ProtocolAnthropic
		}
	}

	model := ""
	if opts != nil {
		model = opts.RequestModel
	}

	switch target {
	case ProtocolAnthropic:
		return geminiChunkToAnthropic(evt, &chunk, c, model, opts)
	default:
		return geminiChunkToOpenAI(evt, &chunk, c, model, opts)
	}
}

// geminiChunkToOpenAI converts a Gemini SSE chunk to an OpenAI delta chunk.
func geminiChunkToOpenAI(evt *SSEEvent, chunk *GeminiChatResponse, c GeminiCandidate, model string, opts *ConvertOptions) ([]byte, error) {
	if len(c.Content.Parts) == 0 {
		if c.FinishReason != "" {
			finish := mapGeminiFinishToOpenAI(c.FinishReason, false)
			delta := map[string]any{
				"id":      "gemini-" + randHex(16),
				"object":  "chat.completion.chunk",
				"model":   model,
				"choices": []map[string]any{{"index": c.Index, "delta": map[string]any{}, "finish_reason": finish}},
			}
			if chunk.UsageMetadata != nil {
				u := openAIUsageFromGemini(chunk.UsageMetadata)
				usage := map[string]any{
					"prompt_tokens":     u.PromptTokens,
					"completion_tokens": u.CompletionTokens,
					"total_tokens":      u.TotalTokens,
				}
				if u.PromptTokensDetails != nil {
					usage["prompt_tokens_details"] = u.PromptTokensDetails
				}
				delta["usage"] = usage
			}
			b, _ := json.Marshal(delta)
			return []byte("data: " + string(b)), nil
		}
		return []byte{}, nil
	}

	var parts []string
	var reasoning []string
	var toolCalls []OpenAIToolCall
	for _, part := range c.Content.Parts {
		switch {
		case part.Text != "" && part.FunctionCall == nil:
			if part.Thought {
				reasoning = append(reasoning, part.Text)
			} else {
				parts = append(parts, part.Text)
			}
		case part.FunctionCall != nil:
			args := "{}"
			if part.FunctionCall.Args != nil {
				if b, err := json.Marshal(part.FunctionCall.Args); err == nil {
					args = string(b)
				}
			}
			id := "call_" + randHex(16)
			if opts != nil && opts.SessionStore != nil && opts.SID != "" {
				opts.SessionStore.SetThoughtSigFor(opts.SID, id, part.FunctionCall.Name, part.ThoughtSignature)
			}
			toolCalls = append(toolCalls, OpenAIToolCall{
				ID:   id,
				Type: "function",
				Function: OpenAIFunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: args,
				},
			})
		}
	}

	content := strings.Join(parts, "")
	reasoningContent := strings.Join(reasoning, "")
	if content == "" && reasoningContent == "" && len(toolCalls) == 0 {
		return []byte{}, nil
	}

	deltaObj := map[string]any{}
	if content != "" {
		deltaObj["content"] = content
	}
	if reasoningContent != "" {
		deltaObj["reasoning_content"] = reasoningContent
	}
	if len(toolCalls) > 0 {
		// For streaming, tool_calls require index-based deltas.
		tcDeltas := make([]map[string]any, 0, len(toolCalls))
		for i, tc := range toolCalls {
			tcDeltas = append(tcDeltas, map[string]any{
				"index":    i,
				"id":       tc.ID,
				"type":     "function",
				"function": map[string]any{"name": tc.Function.Name, "arguments": tc.Function.Arguments},
			})
		}
		deltaObj["tool_calls"] = tcDeltas
	}

	delta := map[string]any{
		"id":      "gemini-" + randHex(16),
		"object":  "chat.completion.chunk",
		"model":   model,
		"choices": []map[string]any{{"index": c.Index, "delta": deltaObj}},
	}
	b, _ := json.Marshal(delta)
	return []byte("data: " + string(b)), nil
}

// geminiChunkToAnthropic converts a Gemini SSE chunk to an Anthropic SSE event
// sequence (content_block_delta with text_delta, or content_block_start/stop
// for tool_use transitions).
func geminiChunkToAnthropic(evt *SSEEvent, chunk *GeminiChatResponse, c GeminiCandidate, _ string, opts *ConvertOptions) ([]byte, error) {
	if len(c.Content.Parts) == 0 {
		if c.FinishReason != "" {
			stopReason := mapGeminiFinishToAnthropic(c.FinishReason)
			delta, _ := json.Marshal(map[string]any{
				"type": "message_delta",
				"delta": map[string]any{
					"stop_reason":   stopReason,
					"stop_sequence": nil,
				},
				"usage": geminiAnthropicUsage(chunk.UsageMetadata),
			})
			return append(
				[]byte("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"),
				[]byte("event: message_delta\ndata: "+string(delta)+"\n\n")...,
			), nil
		}
		return []byte{}, nil
	}

	var buf []byte
	hasToolUse := false

	for _, part := range c.Content.Parts {
		switch {
		case part.Text != "" && part.Thought && part.FunctionCall == nil:
			start, _ := json.Marshal(map[string]any{
				"type":          "content_block_start",
				"index":         0,
				"content_block": map[string]any{"type": "thinking"},
			})
			buf = append(buf, []byte("event: content_block_start\ndata: "+string(start)+"\n\n")...)
			delta, _ := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{"type": "thinking_delta", "thinking": part.Text},
			})
			buf = append(buf, []byte("event: content_block_delta\ndata: "+string(delta)+"\n\n")...)
			if part.ThoughtSignature != "" {
				sig, _ := json.Marshal(map[string]any{
					"type":  "content_block_delta",
					"index": 0,
					"delta": map[string]any{"type": "signature_delta", "signature": part.ThoughtSignature},
				})
				buf = append(buf, []byte("event: content_block_delta\ndata: "+string(sig)+"\n\n")...)
			}
			buf = append(buf, []byte("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")...)
		case part.Text != "" && part.FunctionCall == nil:
			if part.Text != "" {
				delta, _ := json.Marshal(map[string]any{
					"type":  "content_block_delta",
					"index": 0,
					"delta": map[string]any{"type": "text_delta", "text": part.Text},
				})
				buf = append(buf, []byte("event: content_block_delta\ndata: "+string(delta)+"\n\n")...)
			}
		case part.FunctionCall != nil:
			hasToolUse = true
			id := ensureToolID("")
			if opts != nil && opts.SessionStore != nil && opts.SID != "" {
				opts.SessionStore.SetThoughtSigFor(opts.SID, id, part.FunctionCall.Name, part.ThoughtSignature)
			}
			buf = append(buf, []byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\""+id+"\",\"name\":\""+part.FunctionCall.Name+"\"}}\n\n")...)
			args := "{}"
			if part.FunctionCall.Args != nil {
				if b, err := json.Marshal(part.FunctionCall.Args); err == nil {
					args = string(b)
				}
			}
			delta, _ := json.Marshal(map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
			})
			buf = append(buf, []byte("event: content_block_delta\ndata: "+string(delta)+"\n\n")...)
		}
	}

	if hasToolUse {
		buf = append(buf, []byte("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")...)
	}

	// If finish_reason is present, emit final events.
	if c.FinishReason != "" {
		if !hasToolUse {
			buf = append(buf, []byte("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")...)
		}
		stopReason := mapGeminiFinishToAnthropic(c.FinishReason)
		delta, _ := json.Marshal(map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   stopReason,
				"stop_sequence": nil,
			},
			"usage": geminiAnthropicUsage(chunk.UsageMetadata),
		})
		buf = append(buf, []byte("event: message_delta\ndata: "+string(delta)+"\n\n")...)
	}

	if len(buf) == 0 {
		return []byte{}, nil
	}

	return buf, nil
}

// fmtGeminiStreamChunk formats the part for logging.
func fmtGeminiStreamChunk(c GeminiCandidate) string {
	texts := make([]string, 0, len(c.Content.Parts))
	for _, p := range c.Content.Parts {
		if p.Text != "" {
			texts = append(texts, fmt.Sprintf("text=%q", truncStr(p.Text, 40)))
		}
		if p.FunctionCall != nil {
			texts = append(texts, fmt.Sprintf("fn=%s", p.FunctionCall.Name))
		}
	}
	return strings.Join(texts, ", ")
}

func truncStr(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
