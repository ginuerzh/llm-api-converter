package convert

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
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

	// "[DONE]" => message_stop.
	if bytes.Equal(bytes.TrimSpace(body), []byte("data: [DONE]")) {
		return []byte("event: message_stop\ndata: {\"type\":\"message_stop\"}"), nil
	}

	evt := parseSSEEvent(body)
	if evt.Data == "" {
		return body, nil
	}

	// Protocol passthrough: when model-map declares a downstream protocol that
	// matches the data format, pass through with model rewrite instead of
	// converting. Uses body-primary detection with model map fallback for
	// SSE event payloads (e.g. message_start has type:"message_start").
	if opts.ModelMap != nil {
		var raw map[string]any
		if err := json.Unmarshal([]byte(evt.Data), &raw); err == nil {
			from := detectSource(raw)
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

	// Default: convert data payload with Convert().
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

// Convert detects the input format via body structure (primary) with URI
// metadata as optional fallback, then converts between protocols.
// On request: stores the detected source protocol in Session.From for the
// response path. On response: reads Session.From as the client protocol.
func Convert(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192}
	}

	if len(body) == 0 {
		return body, nil
	}

	// SSE framing => ConvertSSE.
	if isSSE(body) {
		slog.Debug("SSE framing => ConvertSSE")
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

	// Body-primary protocol detection (authoritative).
	source := detectSource(raw)
	dir := DirectionRequest
	if opts.Direction == "response" {
		dir = DirectionResponse
	}

	// Optional URI fallback for minimal requests with no distinguishing features.
	if source == ProtocolUnknown {
		source = detectByURI(opts.URI, dir)
	}

	if source == ProtocolUnknown {
		slog.Debug("unknown format, passing through")
		return body, nil
	}

	slog.Debug("detected protocol", "protocol", source.String(), "direction", opts.Direction)

	// Responses API session routing — check before protocol-specific handling.
	if opts.SessionStore != nil && opts.SID != "" {
		if sess := opts.SessionStore.Get(opts.SID); sess != nil && sess.IsResponses {
			slog.Debug("Responses session upstream response => converting back")
			return convertToResponsesResponse(body, opts)
		}
	}

	// Responses API uses its own routing.
	if source == ProtocolOpenAIResponses {
		if dir == DirectionRequest {
			slog.Debug("Responses request => converting")
			if opts.SID != "" && opts.SessionStore != nil {
				sess := &Session{ID: opts.SID, From: source, IsResponses: true}
				opts.SessionStore.Set(opts.SID, sess)
			}
			return convertResponsesRequest(body, opts)
		}
		return body, nil
	}

	// Resolve downstream protocol from model map.
	model, _ := raw["model"].(string)
	targetModel, downstream := resolveModel(model, source, opts.ModelMap)
	opts.ResolvedModel = targetModel

	if dir == DirectionRequest {
		// Request path: store client protocol in session for response path.
		if opts.SID != "" && opts.SessionStore != nil {
			opts.SessionStore.Set(opts.SID, &Session{ID: opts.SID, From: source})
		}

		// Passthrough if source matches downstream protocol.
		if source == downstream || downstream == ProtocolUnknown {
			return passthrough(raw, targetModel, opts, body)
		}

		// Convert source => downstream protocol.
		if convertFn, ok := conversions[ConversionKey{source, downstream}]; ok {
			slog.Debug("converting request", "from", source.String(), "to", downstream.String(), "model", targetModel)
			converted, err := convertFn(body, opts)
			if err != nil {
				return nil, fmt.Errorf("convert %s=>%s: %w", source.String(), downstream.String(), err)
			}
			return converted, nil
		}

		slog.Debug("no converter for request pair, passing through", "from", source.String(), "to", downstream.String())
		return passthrough(raw, targetModel, opts, body)
	}

	// Response path: source is the downstream protocol (detected from body).
	// Determine client protocol from session or URI fallback.
	client := ProtocolUnknown
	if opts.SessionStore != nil && opts.SID != "" {
		if sess := opts.SessionStore.Get(opts.SID); sess != nil {
			client = sess.From
		}
		// Non-streaming request/response pair is complete; free the entry.
		// The next request on this SID (HTTP keep-alive) re-stores before use.
		opts.SessionStore.Delete(opts.SID)
	}
	if client == ProtocolUnknown {
		client = detectByURI(opts.URI, DirectionRequest)
	}

	// Passthrough: body format matches client protocol.
	if client != ProtocolUnknown && source == client {
		return passthrough(raw, targetModel, opts, body)
	}

	// Convert if source differs from client.
	if client != ProtocolUnknown {
		if convertFn, ok := conversions[ConversionKey{source, client}]; ok {
			slog.Debug("converting response", "from", source.String(), "to", client.String(), "model", targetModel)
			converted, err := convertFn(body, opts)
			if err != nil {
				return nil, fmt.Errorf("convert %s=>%s: %w", source.String(), client.String(), err)
			}
			return converted, nil
		}
	}

	// Fallback passthrough.
	return passthrough(raw, targetModel, opts, body)
}

// passthrough rewrites the model field and injects max_completion_tokens.
// Model replacement is done at byte level to preserve JSON field order.
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
		// Model-only: replace in original bytes to preserve field order.
		if modelChanged && !tokensInjected {
			result := modelValueRe.ReplaceAll(originalBody, []byte(`${1}"`+targetModel+`"`))
			slog.Debug("passthrough with model rewrite (byte-level)", "model", targetModel)
			return result, nil
		}
		if modified, err := json.Marshal(raw); err == nil {
			slog.Debug("passthrough with rewrite", "model", targetModel)
			return modified, nil
		}
	}
	slog.Debug("passthrough unchanged")
	return originalBody, nil
}

var modelValueRe = regexp.MustCompile(`("model"\s*:\s*)"([^"\\]|\\.)*"`)

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
		// Body-primary detection from the initial SSE data payload.
		payload := extractSSEPayload(data)
		from := ProtocolUnknown
		if payload != nil {
			var raw map[string]any
			if err := json.Unmarshal(payload, &raw); err == nil {
				from = detectSource(raw)
			}
		}

		// URI fallback for minimal event payloads (e.g. message_start).
		dir := DirectionRequest
		if opts.Direction == "response" {
			dir = DirectionResponse
		}
		if from == ProtocolUnknown {
			from = detectByURI(opts.URI, dir)
		}

		// Target protocol = the client protocol. Prefer the value stored on the
		// request path (authoritative — makes streaming symmetric with the non-
		// streaming response path); fall back to model-map resolution when no
		// request session exists (old sniffer, or stateless usage).
		target := ProtocolUnknown
		if store != nil {
			if sess := store.Get(sid); sess != nil {
				target = sess.From
			}
		}
		model := extractModelFromData(payload)
		if target == ProtocolUnknown {
			_, target = resolveModel(model, from, opts.ModelMap)
		}

		// Model map fallback when body + URI can't classify the event.
		if from == ProtocolUnknown && target != ProtocolUnknown {
			from = target
		}

		// Passthrough: chunk format matches client protocol.
		if from != ProtocolUnknown && from == target {
			sourceModel := opts.RequestModel
			if sourceModel == "" && model != "" && opts.ModelMap != nil {
				if prefix := opts.ModelMap.SourcePrefix(model); prefix != "" {
					sourceModel = prefix
				}
			}
			handler := NewPassthroughStreamHandler(sourceModel, target)
			if store != nil {
				store.Set(sid, &Session{ID: sid, From: target, To: from, StreamHandler: handler})
			}
			if len(data) > 0 {
				return handler.HandleChunk(data)
			}
			return nil, nil
		}

		// Create stream converter (OpenAI deltas => Anthropic SSE).
		sc := newStreamConverterFromData(data, opts)
		if store != nil {
			store.Set(sid, &Session{ID: sid, From: target, To: from, StreamHandler: sc})
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

		// Passthrough handlers expect SSE-framed input from the start
		// phase, but extractSSEPayload strips framing. Pass the raw
		// SSE event to preserve proper formatting for the client.
		chunkInput := payload
		if _, ok := handler.(*PassthroughStreamHandler); ok {
			chunkInput = data
		}
		out, err := handler.HandleChunk(chunkInput)
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
		// Passthrough: no synthesized closing events needed.
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
