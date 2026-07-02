package convert

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// SSE event parsing
// ---------------------------------------------------------------------------

// isSSE reports whether body appears to be an SSE event by checking for
// known SSE field prefixes at the start of the data.
func isSSE(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	switch {
	case bytes.HasPrefix(body, []byte("data:")):
		return true
	case bytes.HasPrefix(body, []byte("event:")):
		return true
	case bytes.HasPrefix(body, []byte("id:")):
		return true
	}
	return false
}

// parseSSEEvent parses raw SSE event bytes into fields.
func parseSSEEvent(raw []byte) *SSEEvent {
	evt := &SSEEvent{}
	for _, line := range bytes.Split(raw, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		key, value, found := bytes.Cut(line, []byte(":"))
		if !found {
			continue
		}
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}
		switch string(key) {
		case "event":
			evt.Event = string(value)
		case "data":
			if evt.Data != "" {
				evt.Data += "\n"
			}
			evt.Data += string(value)
		case "id":
			evt.ID = string(value)
		case "retry":
			if ms, err := strconv.Atoi(string(value)); err == nil && ms > 0 {
				evt.Retry = ms
			}
		}
	}
	return evt
}

// reconstructSSEEvent builds SSE-formatted bytes from the event fields.
func reconstructSSEEvent(evt *SSEEvent) []byte {
	var buf bytes.Buffer
	if evt.ID != "" {
		buf.WriteString("id: ")
		buf.WriteString(evt.ID)
		buf.WriteByte('\n')
	}
	if evt.Event != "" {
		buf.WriteString("event: ")
		buf.WriteString(evt.Event)
		buf.WriteByte('\n')
	}
	for _, line := range strings.Split(evt.Data, "\n") {
		buf.WriteString("data: ")
		buf.WriteString(line)
		buf.WriteByte('\n')
	}
	if evt.Retry > 0 {
		buf.WriteString("retry: ")
		buf.WriteString(strconv.Itoa(evt.Retry))
		buf.WriteByte('\n')
	}
	raw := buf.Bytes()
	if len(raw) > 0 && raw[len(raw)-1] == '\n' {
		raw = raw[:len(raw)-1]
	}
	return raw
}

// convertSSEEvent converts the JSON data payload inside an SSE event
// using Convert. Returns true if the data changed.
func convertSSEEvent(evt *SSEEvent, opts *ConvertOptions) bool {
	if evt.Data == "" {
		return false
	}
	converted, err := Convert([]byte(evt.Data), opts)
	if err != nil {
		slog.Debug("sse convert error, using original data", "err", err)
		return false
	}
	convertedStr := string(converted)
	if convertedStr == evt.Data {
		return false
	}
	evt.Data = convertedStr
	return true
}

// ---------------------------------------------------------------------------
// SSE conversion
// ---------------------------------------------------------------------------

// ConvertSSE parses SSE-formatted body, converts the JSON data payload,
// and reconstructs the SSE framing.
func ConvertSSE(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192}
	}
	if len(body) == 0 {
		return body, nil
	}

	// "[DONE]" → message_stop.
	if bytes.Equal(bytes.TrimSpace(body), []byte("data: [DONE]")) {
		return []byte("event: message_stop\ndata: {\"type\":\"message_stop\"}"), nil
	}

	evt := parseSSEEvent(body)
	if evt.Data == "" {
		return body, nil
	}

	// Protocol passthrough: when model-map declares a downstream protocol that
	// matches the data format, pass through with model rewrite instead of
	// converting. This handles SSE events from old GOST sniffers that don't
	// send sse_phase metadata. Stream lifecycle events (HandleSSEEvent) have
	// their own passthrough path in StreamPhaseStart.
	if opts.ModelMap != nil {
		var raw map[string]any
		if err := json.Unmarshal([]byte(evt.Data), &raw); err == nil {
			from, _ := detectByBody(raw, opts)
			// Body-based detection misses SSE payloads (e.g. message_start
			// has type:"message_start" not type:"message"). Fall back to
			// the model map when the data carries a known model name.
			if from == ProtocolUnknown {
				model := extractModelFromData([]byte(evt.Data))
				_, to := resolveModel(model, ProtocolUnknown, opts.ModelMap)
				if to != ProtocolUnknown {
					from = to
				}
			}
			if from != ProtocolUnknown {
				model, _ := raw["model"].(string)
				targetModel, to := resolveModel(model, from, opts.ModelMap)
				if from == to {
					// Same protocol — rewrite model if needed.
					if targetModel != "" && targetModel != model {
						raw["model"] = targetModel
						if newData, err := json.Marshal(raw); err == nil {
							evt.Data = string(newData)
							slog.Debug("sse: protocol passthrough with model rewrite", "model", targetModel)
							return reconstructSSEEvent(evt), nil
						}
					}
					slog.Debug("sse: protocol passthrough")
					return body, nil
				}
			}
		}
	}

	// Check for OpenAI streaming chunk: data payload with choices[].delta.
	if isOpenAIStreamChunk([]byte(evt.Data)) {
		return convertOpenAIStreamChunkToAnthropic(evt)
	}

	// Default: convert data payload with Convert() (handles protocol detection internally).
	convertSSEEvent(evt, opts)
	return reconstructSSEEvent(evt), nil
}

// convertOpenAIStreamChunkToAnthropic converts an OpenAI SSE streaming delta
// to an Anthropic SSE event.
func convertOpenAIStreamChunkToAnthropic(evt *SSEEvent) ([]byte, error) {
	var chunk struct {
		Choices []struct {
			Index        int             `json:"index"`
			Delta        map[string]any  `json:"delta"`
			FinishReason *string         `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(evt.Data), &chunk); err != nil || len(chunk.Choices) == 0 {
		return reconstructSSEEvent(evt), nil
	}

	choice := chunk.Choices[0]

	if choice.FinishReason != nil && *choice.FinishReason != "" {
		stopReason := mapOpenAIStreamFinish(*choice.FinishReason)
		deltaJSON, _ := json.Marshal(map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   stopReason,
				"stop_sequence": nil,
			},
			"usage": map[string]any{"output_tokens": 0},
		})
		return []byte("event: message_delta\ndata: " + string(deltaJSON)), nil
	}

	if content, ok := choice.Delta["content"].(string); ok && content != "" {
		deltaJSON, _ := json.Marshal(map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{"type": "text_delta", "text": content},
		})
		return []byte("event: content_block_delta\ndata: " + string(deltaJSON)), nil
	}

	if reasoning, ok := choice.Delta["reasoning_content"].(string); ok && reasoning != "" {
		deltaJSON, _ := json.Marshal(map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{"type": "thinking_delta", "thinking": reasoning},
		})
		return []byte("event: content_block_delta\ndata: " + string(deltaJSON)), nil
	}

	return reconstructSSEEvent(evt), nil
}

func mapOpenAIStreamFinish(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return reason
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// StripProviderPrefix removes a leading "provider/" segment from model names.
func StripProviderPrefix(model string) string {
	if i := strings.IndexByte(model, '/'); i >= 0 {
		return model[i+1:]
	}
	return model
}

// extractModelFromData extracts the "model" field from raw JSON data.
// Handles nested model in "message" field (Anthropic message_start event).
func extractModelFromData(data []byte) string {
	var raw struct {
		Model   string `json:"model"`
		Message struct {
			Model string `json:"model"`
		} `json:"message"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return ""
	}
	if raw.Model != "" {
		return raw.Model
	}
	return raw.Message.Model
}

// isOpenAIStreamChunk detects an OpenAI SSE streaming data payload.
func isOpenAIStreamChunk(data []byte) bool {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return false
	}
	choices, ok := raw["choices"].([]any)
	if !ok || len(choices) == 0 {
		return false
	}
	if choice, ok := choices[0].(map[string]any); ok {
		if _, ok := choice["delta"]; ok {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Convert — main entry point
// ---------------------------------------------------------------------------

// Convert detects the input format via GOST metadata (URI + direction) and
// converts between protocols. Falls back to body-based detection when
// metadata is unavailable (backward compatibility).
func Convert(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192}
	}

	if len(body) == 0 {
		return body, nil
	}

	// SSE framing → ConvertSSE.
	if isSSE(body) {
		slog.Debug("SSE framing → ConvertSSE")
		return ConvertSSE(body, opts)
	}

	// Parse as JSON.
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		slog.Debug("not JSON, passing through", "err", err)
		return body, nil
	}

	// Log error fields.
	if errVal, ok := raw["error"]; ok {
		switch v := errVal.(type) {
		case map[string]any:
			if msg, _ := v["message"].(string); msg != "" {
				slog.Warn("response contains error", "message", msg, "type", v["type"])
			} else {
				slog.Warn("response contains error", "error", errVal)
			}
		default:
			slog.Warn("response contains error", "error", errVal)
		}
	}

	// Detect protocol from metadata.
	from, dir, ok := detectProtocol(opts.URI, opts.Direction)
	if !ok {
		// Fallback: body-based detection for backward compatibility.
		from, dir = detectByBody(raw, opts)
	}

	if from == ProtocolUnknown {
		slog.Debug("unknown format, passing through")
		return body, nil
	}

	slog.Debug("detected protocol", "protocol", from.String(), "direction", dir)

	// On the response path, the body format may differ from the
	// URI-expected format when the downstream speaks a different
	// protocol than the client. Track the client protocol so we
	// can redirect conversion back to it.
	var clientProto Protocol
	if dir == DirectionResponse && ok {
		if bodyFrom, bodyDir := detectByBody(raw, opts); bodyFrom != ProtocolUnknown && bodyFrom != from {
			clientProto = from
			from = bodyFrom
			dir = bodyDir
			slog.Debug("asymmetric response: body format differs from client",
				"body", from.String(), "client", clientProto.String())
		}
	}

	// Responses API session routing — check BEFORE protocol-specific handling.
	// The body format of upstream responses doesn't match the session's protocol
	// (e.g. Anthropic response body in a Responses session → needs conversion back).
	if opts.SessionStore != nil && opts.SID != "" {
		if sess := opts.SessionStore.Get(opts.SID); sess != nil && sess.IsResponses {
			slog.Debug("Responses session upstream response → converting back")
			return convertToResponsesResponse(body, opts)
		}
	}

	// Responses API uses its own routing (model-map protocol decides Chat vs Anthropic).
	if from == ProtocolOpenAIResponses {
		if dir == DirectionRequest {
			slog.Debug("Responses request → converting")
			if opts.SID != "" && opts.SessionStore != nil {
				sess := &Session{ID: opts.SID, From: from, IsResponses: true}
				opts.SessionStore.Set(opts.SID, sess)
			}
			return convertResponsesRequest(body, opts)
		}
		// Responses response (not in a session) — passthrough.
		return body, nil
	}

	// Resolve downstream protocol from model map.
	model, _ := raw["model"].(string)
	targetModel, to := resolveModel(model, from, opts.ModelMap)
	opts.ResolvedModel = targetModel

	// Redirect asymmetric response: convert body format → client format.
	if clientProto != ProtocolUnknown && from == to {
		to = clientProto
	}

	// Passthrough: same protocol → model rewrite only.
	if from == to {
		return passthrough(raw, targetModel, opts, body)
	}

	// Conversion via registry.
	if convertFn, ok := conversions[ConversionKey{from, to}]; ok {
		slog.Debug("converting", "from", from.String(), "to", to.String(), "model", targetModel)
		converted, err := convertFn(body, opts)
		if err != nil {
			return nil, fmt.Errorf("convert %s→%s: %w", from.String(), to.String(), err)
		}
		return converted, nil
	}

	// No converter registered — passthrough with model rewrite.
	slog.Debug("no converter for pair, passing through", "from", from.String(), "to", to.String())
	return passthrough(raw, targetModel, opts, body)
}

// detectByBody is the fallback body-based format detection.
func detectByBody(raw map[string]any, opts *ConvertOptions) (Protocol, Direction) {
	// Responses API request.
	if isResponsesRequest(raw) {
		return ProtocolOpenAIResponses, DirectionRequest
	}
	// Responses API response.
	if isResponsesResponse(raw) {
		return ProtocolOpenAIResponses, DirectionResponse
	}
	// Anthropic request (messages with typed content blocks, max_tokens).
	if isAnthropicRequest(raw) {
		return ProtocolAnthropic, DirectionRequest
	}
	// OpenAI response (choices array).
	if isOpenAIResponse(raw) {
		return ProtocolOpenAIChat, DirectionResponse
	}
	// Anthropic response (type:"message" + stop_reason/usage).
	if isAnthropicResponse(raw) {
		return ProtocolAnthropic, DirectionResponse
	}
	// OpenAI request (model or messages field).
	if _, ok := raw["model"].(string); ok {
		return ProtocolOpenAIChat, DirectionRequest
	}
	if _, ok := raw["messages"].([]any); ok {
		return ProtocolOpenAIChat, DirectionRequest
	}
	return ProtocolUnknown, 0
}

// passthrough rewrites the model field and injects max_completion_tokens.
// Returns the original body unchanged when nothing is modified.
func passthrough(raw map[string]any, targetModel string, opts *ConvertOptions, originalBody []byte) ([]byte, error) {
	modelChanged := false
	if targetModel != "" && raw["model"] != targetModel {
		raw["model"] = targetModel
		modelChanged = true
	}

	tokensInjected := false
	if opts.MaxTokens > 0 {
		if _, hasMT := raw["max_tokens"]; !hasMT {
			if _, hasMCT := raw["max_completion_tokens"]; !hasMCT {
				if _, hasMsg := raw["messages"]; hasMsg {
					raw["max_completion_tokens"] = opts.MaxTokens
					tokensInjected = true
				}
			}
		}
	}

	if modelChanged || tokensInjected {
		if modified, err := json.Marshal(raw); err == nil {
			slog.Debug("passthrough with rewrite", "model", targetModel)
			return modified, nil
		}
	}
	slog.Debug("passthrough unchanged")
	return originalBody, nil
}


// ---------------------------------------------------------------------------
// SSE stream lifecycle
// ---------------------------------------------------------------------------

// extractSSEPayload parses SSE event text and returns just the data payload.
func extractSSEPayload(data []byte) []byte {
	evt := parseSSEEvent(data)
	if evt.Data == "[DONE]" {
		return nil
	}
	if evt.Data != "" {
		return []byte(evt.Data)
	}
	return data
}

// newStreamConverterFromData resolves the model from data and returns a StreamConverter.
func newStreamConverterFromData(data []byte, opts *ConvertOptions) *StreamConverter {
	streamModel := opts.Model
	if model := extractModelFromData(data); model != "" {
		streamModel = model
		if opts.ModelMap != nil {
			if target, _ := resolveModel(model, ProtocolOpenAIChat, opts.ModelMap); target != "" {
				streamModel = target
			}
		}
	}
	if opts.RequestModel != "" {
		streamModel = opts.RequestModel
	} else if opts.ModelMap != nil {
		if sourcePrefix := opts.ModelMap.SourcePrefix(streamModel); sourcePrefix != "" {
			streamModel = sourcePrefix
		}
	}
	return NewStreamConverter(streamModel, opts.ReasoningCache, opts.DeclaredTools)
}

// HandleSSEEvent processes an SSE stream lifecycle event.
func HandleSSEEvent(sid, phase string, eventIndex int, data []byte, opts *ConvertOptions) ([]byte, error) {
	store := opts.SessionStore

	// Responses API sessions use their own stream handlers.
	if store != nil {
		if sess := store.Get(sid); sess != nil && sess.IsResponses {
			return handleResponsesSSEEvent(sid, phase, eventIndex, data, opts)
		}
	}

	switch StreamPhase(phase) {
	case StreamPhaseStart:
		// Detect protocol from metadata.
		from, _, _ := detectProtocol(opts.URI, opts.Direction)
		if from == ProtocolUnknown {
			// Fallback for old sniffer: detect from data payload.
			payload := extractSSEPayload(data)
			var raw map[string]any
			if err := json.Unmarshal(payload, &raw); err == nil {
				from, _ = detectByBody(raw, opts)
			}
		}

		// Resolve downstream protocol.
		payload := extractSSEPayload(data)
		model := extractModelFromData(payload)
		_, to := resolveModel(model, from, opts.ModelMap)

		// When the sniffer doesn't send URI metadata and body-based
		// detection can't classify the data (e.g. message_start events
		// have type:"message_start" not type:"message"), fall back to
		// the protocol resolved from the model map.
		if from == ProtocolUnknown && to != ProtocolUnknown {
			from = to
		}

		// Passthrough: same protocol.
		if from != ProtocolUnknown && from == to {
			handler := NewPassthroughStreamHandler(opts.RequestModel, to)
			if store != nil {
				store.Set(sid, &Session{ID: sid, From: from, To: to, StreamHandler: handler})
			}
			if len(data) > 0 {
				return handler.HandleChunk(data)
			}
			return nil, nil
		}

		// Create stream converter.
		sc := newStreamConverterFromData(data, opts)
		if store != nil {
			store.Set(sid, &Session{ID: sid, From: from, To: to, StreamHandler: sc})
		}

		startData := sc.HandleStreamStart()
		if len(data) > 0 {
			payload := extractSSEPayload(data)
			if payload != nil {
				chunkData, err := sc.HandleChunk(payload)
				if err != nil {
					return startData, err
				}
				if len(chunkData) > 0 {
					return append(startData, append([]byte("\n\n"), chunkData...)...), nil
				}
			}
		}
		return startData, nil

	case StreamPhaseEvent:
		payload := extractSSEPayload(data)
		if payload == nil {
			return []byte{}, nil // [DONE] marker, swallow.
		}

		var handler responsesStreamHandler
		if store != nil {
			if sess := store.Get(sid); sess != nil {
				handler = sess.StreamHandler
			}
		}

		if handler == nil {
			slog.Warn("HandleSSEEvent: unknown stream, creating from data", "sid", sid)
			sc := newStreamConverterFromData(payload, opts)
			if store != nil {
				store.Set(sid, &Session{ID: sid, StreamHandler: sc})
			}
			handler = sc
			startData := sc.HandleStreamStart()
			chunkData, err := sc.HandleChunk(payload)
			if err != nil {
				return startData, err
			}
			if len(chunkData) > 0 {
				return append(startData, append([]byte("\n\n"), chunkData...)...), nil
			}
			return startData, nil
		}

		out, err := handler.HandleChunk(payload)
		if err != nil {
			return nil, err
		}
		if out == nil {
			return []byte{}, nil
		}
		return out, nil

	case StreamPhaseEnd:
		var handler responsesStreamHandler
		if store != nil {
			if sess := store.Get(sid); sess != nil {
				handler = sess.StreamHandler
			}
			store.Delete(sid)
		}
		if handler == nil {
			slog.Warn("HandleSSEEvent: unknown stream for end phase", "sid", sid)
			return nil, nil
		}
		// Passhthrough: no synthesized closing events needed.
		if _, ok := handler.(*PassthroughStreamHandler); ok {
			return nil, nil
		}
		return handler.HandleStreamEnd(), nil

	case StreamPhaseError:
		var handler responsesStreamHandler
		if store != nil {
			if sess := store.Get(sid); sess != nil {
				handler = sess.StreamHandler
			}
			store.Delete(sid)
		}
		msg := "stream error"
		if opts.ErrorMsg != "" {
			msg = opts.ErrorMsg
		}
		if handler != nil {
			return handler.EmitError(msg), nil
		}
		return nil, nil
	}

	return nil, fmt.Errorf("unknown sse_phase: %s", phase)
}

// handleResponsesSSEEvent routes SSE lifecycle events for Responses API sessions.
func handleResponsesSSEEvent(sid, phase string, _ int, data []byte, opts *ConvertOptions) ([]byte, error) {
	store := opts.SessionStore

	switch StreamPhase(phase) {
	case StreamPhaseStart:
		payload := extractSSEPayload(data)
		model := opts.RequestModel
		if model == "" {
			if m := extractModelFromData(payload); m != "" {
				model, _ = resolveModel(m, ProtocolOpenAIResponses, opts.ModelMap)
			}
		}
		if model == "" {
			model = opts.Model
		}

		var handler responsesStreamHandler
		if isAnthropicStreamEvent(payload) {
			handler = NewAnthropicStreamConverter(model)
		} else {
			handler = NewResponsesStreamConverter(model)
			if opts.CodexToolContext != nil {
				handler.(*ResponsesStreamConverter).SetToolSpecs(opts.CodexToolContext.byChatName)
			}
		}
		if store != nil {
			store.Set(sid, &Session{ID: sid, IsResponses: true, StreamHandler: handler})
		}
		out := handler.HandleStreamStart()
		if payload != nil {
			chunk, err := handler.HandleChunk(payload)
			if err != nil {
				return out, err
			}
			if len(chunk) > 0 {
				return append(out, append([]byte("\n\n"), chunk...)...), nil
			}
		}
		return out, nil

	case StreamPhaseEvent:
		payload := extractSSEPayload(data)
		if payload == nil {
			return []byte{}, nil
		}
		var handler responsesStreamHandler
		if store != nil {
			if sess := store.Get(sid); sess != nil {
				handler = sess.StreamHandler
			}
		}
		if handler == nil {
			slog.Warn("HandleSSEEvent: unknown Responses stream, creating from data", "sid", sid)
			model := opts.RequestModel
			if model == "" {
				if m := extractModelFromData(payload); m != "" {
					model, _ = resolveModel(m, ProtocolOpenAIResponses, opts.ModelMap)
				}
			}
			if model == "" {
				model = opts.Model
			}
			if isAnthropicStreamEvent(payload) {
				handler = NewAnthropicStreamConverter(model)
			} else {
				handler = NewResponsesStreamConverter(model)
				if opts.CodexToolContext != nil {
					handler.(*ResponsesStreamConverter).SetToolSpecs(opts.CodexToolContext.byChatName)
				}
			}
			if store != nil {
				store.Set(sid, &Session{ID: sid, IsResponses: true, StreamHandler: handler})
			}
			startData := handler.HandleStreamStart()
			chunk, err := handler.HandleChunk(payload)
			if err != nil {
				return startData, err
			}
			if len(chunk) > 0 {
				return append(startData, append([]byte("\n\n"), chunk...)...), nil
			}
			return startData, nil
		}
		out, err := handler.HandleChunk(payload)
		if err != nil {
			return nil, err
		}
		if out == nil {
			return []byte{}, nil
		}
		return out, nil

	case StreamPhaseEnd:
		var handler responsesStreamHandler
		if store != nil {
			if sess := store.Get(sid); sess != nil {
				handler = sess.StreamHandler
			}
			store.Delete(sid)
		}
		if handler == nil {
			return nil, nil
		}
		return handler.HandleStreamEnd(), nil

	case StreamPhaseError:
		var handler responsesStreamHandler
		if store != nil {
			if sess := store.Get(sid); sess != nil {
				handler = sess.StreamHandler
			}
			store.Delete(sid)
		}
		msg := "stream error"
		if opts.ErrorMsg != "" {
			msg = opts.ErrorMsg
		}
		if handler != nil {
			return handler.EmitError(msg), nil
		}
		return nil, nil
	}
	return nil, nil
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

func ensureObjectSchema(raw any) any {
	m, ok := raw.(map[string]any)
	if !ok || m == nil {
		return map[string]any{"type": "object"}
	}
	if _, exists := m["type"]; !exists {
		m["type"] = "object"
	}
	return m
}

func sortOpenAITools(tools []OpenAITool) {
	sort.SliceStable(tools, func(i, j int) bool {
		return tools[i].Function.Name < tools[j].Function.Name
	})
}

func sortAnthropicTools(tools []AnthropicTool) {
	sort.SliceStable(tools, func(i, j int) bool {
		return tools[i].Name < tools[j].Name
	})
}

func ensureToolID(id string) string {
	if strings.HasPrefix(id, "toolu_") {
		return id
	}
	return "toolu_" + randHex(24)
}

func randHex(n int) string {
	b := make([]byte, (n+1)/2)
	if _, err := rand.Read(b); err != nil {
		for i := range b {
			b[i] = byte(i * 7)
		}
	}
	const hexd = "0123456789abcdef"
	out := make([]byte, n)
	for i := 0; i < n; i++ {
		out[i] = hexd[b[i/2]>>uint((i%2)*4)&0xf]
	}
	return string(out)
}
