package convert

import (
	"bytes"
	"encoding/json"
	"strings"
	"time"
)

// AnthropicToOpenAIStreamConverter turns an Anthropic Messages SSE stream
// into OpenAI Chat Completions delta chunks. It is used when an OpenAI Chat
// client talks to an Anthropic upstream, the reverse of StreamConverter.
//
// The mapping is lossy in one direction: Anthropic's thinking signatures have
// no OpenAI field, so they are dropped here. Thinking text itself is carried
// as reasoning_content.
type AnthropicToOpenAIStreamConverter struct {
	model string
	opts  *ConvertOptions

	id      string
	created int64
	started bool

	usage    AnthropicUsage
	hasUsage bool
	stop     string
	finished bool

	// Anthropic content block index -> OpenAI tool_calls index.
	toolByBlock map[int]int
	nextTool    int
	// Blocks whose tool input arrived complete in content_block_start; later
	// input_json_delta events for those blocks are ignored.
	toolInputComplete map[int]bool
}

func NewAnthropicToOpenAIStreamConverter(model string, opts ...*ConvertOptions) *AnthropicToOpenAIStreamConverter {
	c := &AnthropicToOpenAIStreamConverter{
		model:             model,
		id:                "chatcmpl-" + randHex(16),
		created:           time.Now().Unix(),
		toolByBlock:       make(map[int]int),
		toolInputComplete: make(map[int]bool),
	}
	if len(opts) > 0 {
		c.opts = opts[0]
	}
	return c
}

func (c *AnthropicToOpenAIStreamConverter) HandleStreamStart() []byte { return nil }

func (c *AnthropicToOpenAIStreamConverter) HandleChunk(data []byte) ([]byte, error) {
	if c.finished {
		return nil, nil
	}
	var evt struct {
		Type    string `json:"type"`
		Index   int    `json:"index"`
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
			Type      string          `json:"type"`
			ID        string          `json:"id,omitempty"`
			Name      string          `json:"name,omitempty"`
			Text      string          `json:"text,omitempty"`
			Thinking  string          `json:"thinking,omitempty"`
			Signature string          `json:"signature,omitempty"`
			Input     json.RawMessage `json:"input,omitempty"`
		} `json:"content_block"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text,omitempty"`
			Thinking    string `json:"thinking,omitempty"`
			PartialJSON string `json:"partial_json,omitempty"`
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
	emit := func(chunk OpenAIStreamChunk) {
		b, _ := json.Marshal(chunk)
		events = append(events, []byte("data: "+string(b)+"\n\n"))
	}

	switch evt.Type {
	case "message_start":
		if evt.Message != nil {
			if evt.Message.ID != "" {
				c.id = "chatcmpl-" + strings.TrimPrefix(evt.Message.ID, "msg_")
			}
			if c.opts != nil && c.opts.RequestModel != "" {
				c.model = c.opts.RequestModel
			} else if evt.Message.Model != "" {
				c.model = evt.Message.Model
			}
			c.setUsage(evt.Message.Usage.InputTokens, evt.Message.Usage.OutputTokens,
				evt.Message.Usage.CacheReadInputTokens, evt.Message.Usage.CacheCreationInputTokens)
		}
		c.emitRole(&events)

	case "content_block_start":
		if !c.started {
			c.emitRole(&events)
		}
		switch evt.ContentBlock.Type {
		case "text":
			// Some Anthropic-compatible relays put the first text in the start
			// event instead of a following text_delta.
			if evt.ContentBlock.Text != "" {
				emit(OpenAIStreamChunk{
					ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
					Choices: []OpenAIStreamChoice{{Delta: OpenAIDelta{Content: evt.ContentBlock.Text}}},
				})
			}
		case "thinking":
			if evt.ContentBlock.Thinking != "" {
				emit(OpenAIStreamChunk{
					ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
					Choices: []OpenAIStreamChoice{{Delta: OpenAIDelta{ReasoningContent: evt.ContentBlock.Thinking}}},
				})
			}
		case "tool_use":
			idx := c.nextTool
			c.nextTool++
			c.toolByBlock[evt.Index] = idx
			args := ""
			if anthropicToolStartInput(evt.ContentBlock.Input) {
				args = string(evt.ContentBlock.Input)
				c.toolInputComplete[evt.Index] = true
			}
			emit(OpenAIStreamChunk{
				ID:      c.id,
				Object:  "chat.completion.chunk",
				Created: c.created,
				Model:   c.model,
				Choices: []OpenAIStreamChoice{{
					Index: 0,
					Delta: OpenAIDelta{ToolCalls: []OpenAIDeltaToolCall{{
						Index: intPtr(idx),
						ID:    evt.ContentBlock.ID,
						Type:  "function",
						Function: OpenAIFunctionCall{
							Name:      evt.ContentBlock.Name,
							Arguments: args,
						},
					}}},
				}},
			})
		}

	case "content_block_delta":
		if !c.started {
			c.emitRole(&events)
		}
		switch evt.Delta.Type {
		case "text_delta":
			if evt.Delta.Text != "" {
				emit(OpenAIStreamChunk{
					ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
					Choices: []OpenAIStreamChoice{{Delta: OpenAIDelta{Content: evt.Delta.Text}}},
				})
			}
		case "thinking_delta":
			if evt.Delta.Thinking != "" {
				emit(OpenAIStreamChunk{
					ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
					Choices: []OpenAIStreamChoice{{Delta: OpenAIDelta{ReasoningContent: evt.Delta.Thinking}}},
				})
			}
		case "input_json_delta":
			if c.toolInputComplete[evt.Index] {
				break
			}
			idx, ok := c.toolByBlock[evt.Index]
			if !ok {
				idx = c.nextTool
				c.nextTool++
				c.toolByBlock[evt.Index] = idx
			}
			emit(OpenAIStreamChunk{
				ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
				Choices: []OpenAIStreamChoice{{Delta: OpenAIDelta{ToolCalls: []OpenAIDeltaToolCall{{
					Index:    intPtr(idx),
					Function: OpenAIFunctionCall{Arguments: evt.Delta.PartialJSON},
				}}}}},
			})
		}

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
		b, _ := json.Marshal(map[string]any{"error": map[string]any{
			"type": e.Error.Type, "message": e.Error.Message,
		}})
		events = append(events, []byte("data: "+string(b)+"\n\n"))
		// An upstream error is terminal: suppress the normal finish chunk and
		// [DONE] so the client sees the error instead of a successful finish.
		c.finished = true
	}

	return joinSSE(events), nil
}

func (c *AnthropicToOpenAIStreamConverter) HandleStreamEnd() []byte {
	if c.finished {
		return nil
	}
	return joinSSE(c.finish())
}

// EmitError returns an OpenAI-formatted error chunk and marks the stream
// finished. It is terminal: once a normal finish or an upstream error has
// ended the stream it returns nil, so a late end-phase error cannot follow a
// completed response.
func (c *AnthropicToOpenAIStreamConverter) EmitError(message string) []byte {
	if c.finished {
		return nil
	}
	c.finished = true
	b, _ := json.Marshal(map[string]any{"error": map[string]any{
		"type": "stream_error", "message": message,
	}})
	return []byte("data: " + string(b) + "\n\n")
}

// setUsage merges the usage fields reported by one Anthropic stream event.
// Providers spread usage across message_start (input and cache counts) and
// message_delta (output count), and often omit fields in later events, so each
// field is updated only when the new value is non-zero. Zero means "not
// reported here", not "reset to zero".
func (c *AnthropicToOpenAIStreamConverter) setUsage(input, output, cacheRead, cacheCreate int) {
	if input > 0 {
		c.usage.InputTokens = input
	}
	if output > 0 {
		c.usage.OutputTokens = output
	}
	if cacheRead > 0 {
		c.usage.CacheReadInputTokens = cacheRead
	}
	if cacheCreate > 0 {
		c.usage.CacheCreationInputTokens = cacheCreate
	}
	if input > 0 || output > 0 || cacheRead > 0 || cacheCreate > 0 {
		c.hasUsage = true
	}
}

func (c *AnthropicToOpenAIStreamConverter) emitRole(events *[][]byte) {
	if c.started {
		return
	}
	c.started = true
	b, _ := json.Marshal(OpenAIStreamChunk{
		ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
		Choices: []OpenAIStreamChoice{{Delta: OpenAIDelta{Role: "assistant", Content: ""}}},
	})
	*events = append(*events, []byte("data: "+string(b)+"\n\n"))
}

// finish emits the finish chunk, the usage chunk (when the upstream reported
// usage) and [DONE].
func (c *AnthropicToOpenAIStreamConverter) finish() [][]byte {
	if c.finished {
		return nil
	}
	c.finished = true

	var events [][]byte
	if !c.started {
		c.emitRole(&events)
	}
	// Anthropic-compatible upstreams sometimes end a tool-calling turn with
	// end_turn (or omit the stop reason). OpenAI clients only run tools when
	// finish_reason is tool_calls, so infer it from the emitted calls while
	// preserving explicit reasons such as max_tokens.
	fr := mapAnthropicStopToOpenAI(c.stop)
	if c.nextTool > 0 && (c.stop == "" || c.stop == "end_turn") {
		fr = "tool_calls"
	}
	b, _ := json.Marshal(OpenAIStreamChunk{
		ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
		Choices: []OpenAIStreamChoice{{Delta: OpenAIDelta{}, FinishReason: &fr}},
	})
	events = append(events, []byte("data: "+string(b)+"\n\n"))

	if c.hasUsage {
		u := openAIUsageFromAnthropic(c.usage)
		b, _ := json.Marshal(OpenAIStreamChunk{
			ID: c.id, Object: "chat.completion.chunk", Created: c.created, Model: c.model,
			Choices: []OpenAIStreamChoice{},
			Usage:   &u,
		})
		events = append(events, []byte("data: "+string(b)+"\n\n"))
	}

	events = append(events, []byte("data: [DONE]\n\n"))
	return events
}

// anthropicToolStartInput reports whether a content_block_start tool_use input
// already carries the complete arguments object. Anthropic normally sends {}
// and streams the arguments in later input_json_delta events, but some relays
// put the full input in the start event.
func anthropicToolStartInput(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return false
	}
	return !bytes.Equal(trimmed, []byte("{}")) && !bytes.Equal(trimmed, []byte("null"))
}

func mapAnthropicStopToOpenAI(s string) string {
	switch s {
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal", "content_filter":
		return "content_filter"
	default:
		return "stop"
	}
}

func joinSSE(events [][]byte) []byte {
	if len(events) == 0 {
		return nil
	}
	out := make([]string, len(events))
	for i, e := range events {
		out[i] = strings.TrimSuffix(string(e), "\n\n")
	}
	return []byte(strings.Join(out, "\n\n") + "\n\n")
}
