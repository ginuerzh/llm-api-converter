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
	"sync"
)

// ---------------------------------------------------------------------------
// Responses API session tracking
// ---------------------------------------------------------------------------

// responsesSessions tracks which SSE sessions are Responses API sessions.
var responsesSessions sync.Map // sid → bool

// responsesStreamStates tracks active Responses stream converters per session.
var responsesStreamStates sync.Map // sid → *ResponsesStreamConverter

func markResponsesSession(sid string)      { responsesSessions.Store(sid, true) }
func IsResponsesSession(sid string) bool {
	_, ok := responsesSessions.Load(sid)
	return ok
}

func unmarkResponsesSession(sid string) {
	responsesSessions.Delete(sid)
	responsesStreamStates.Delete(sid)
}

// ---------------------------------------------------------------------------
// SSE event handling
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

// parseSSEEvent parses raw SSE event bytes into fields. It handles
// event:, data:, id:, retry: fields and ignores unknown lines.
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
		// SSE convention: "field: value" — strip one leading space if present.
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
// It does NOT include the trailing \n\n event delimiter — the caller
// (sniffer_sse.go) appends \n\n after the rewritten bytes.
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
	// Strip trailing newline — the GOST proxy appends \n\n as the event delimiter.
	raw := buf.Bytes()
	if len(raw) > 0 && raw[len(raw)-1] == '\n' {
		raw = raw[:len(raw)-1]
	}
	return raw
}

// convertSSEEvent converts the JSON data payload inside an SSE event
// using the existing Convert function. Returns true if the data changed.
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

// ConvertSSE parses SSE-formatted body, converts the JSON data payload,
// and reconstructs the SSE framing. Non-SSE content that looks like SSE
// (e.g. "[DONE]") passes through unchanged.
func ConvertSSE(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192}
	}
	if len(body) == 0 {
		return body, nil
	}

	// "[DONE]" is a common SSE data marker — convert to Anthropic message_stop.
	if bytes.Equal(bytes.TrimSpace(body), []byte("data: [DONE]")) {
		return []byte("event: message_stop\ndata: {\"type\":\"message_stop\"}"), nil
	}

	evt := parseSSEEvent(body)

	// Protocol passthrough for SSE events: check if data payload already
	// matches the downstream protocol and bypass conversion.
	if opts.ModelMap != nil && evt.Data != "" {
		var raw map[string]any
		if err := json.Unmarshal([]byte(evt.Data), &raw); err == nil {
			model := extractModelFromData([]byte(evt.Data))
			targetModel, proto := resolveModel(model, opts.Model, opts.ModelMap)

			// Reverse lookup: if the source prefix didn't match but the
			// model appears as a target (e.g. response from downstream API),
			// use the target's protocol.
			if proto == "" && model != "" {
				if lp := opts.ModelMap.LookupTarget(model); lp != "" {
					proto = lp
				}
			}

			passthrough := false
			if proto == "openai" && (isOpenAIStreamChunk([]byte(evt.Data)) || isOpenAIResponse(raw)) {
				passthrough = true
			}
			if proto == "anthropic" && isAnthropicResponse(raw) {
				passthrough = true
			}

			if passthrough {
				if targetModel != "" {
					if origModel, ok := raw["model"].(string); ok && origModel != targetModel {
						raw["model"] = targetModel
						if newData, err := json.Marshal(raw); err == nil {
							evt.Data = string(newData)
							slog.Debug("sse: protocol passthrough with model rewrite", "model", targetModel)
							return reconstructSSEEvent(evt), nil
						}
					}
				}
				slog.Debug("sse: protocol passthrough")
				return body, nil
			}
		}
	}

	// Check for OpenAI streaming chunk: data payload with choices[].delta.
	// This must be detected before the generic Convert() path since OpenAI
	// streaming chunks have a different structure from non-streaming responses.
	if evt.Data != "" && isOpenAIStreamChunk([]byte(evt.Data)) {
		return convertOpenAIStreamChunkToAnthropic(evt)
	}

	// Default: convert data payload with Convert() (handles Anthropic SSE → OpenAI data).
	convertSSEEvent(evt, opts)
	return reconstructSSEEvent(evt), nil
}

// convertOpenAIStreamChunkToAnthropic converts an OpenAI SSE streaming delta
// to an Anthropic SSE content_block_delta / message_delta / message_stop event.
func convertOpenAIStreamChunkToAnthropic(evt *SSEEvent) ([]byte, error) {
	var chunk struct {
		Choices []struct {
			Index        int              `json:"index"`
			Delta        map[string]any   `json:"delta"`
			FinishReason *string          `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(evt.Data), &chunk); err != nil || len(chunk.Choices) == 0 {
		// Unrecognised format — pass through unchanged.
		return reconstructSSEEvent(evt), nil
	}

	choice := chunk.Choices[0]

	// Finish reason → message_delta (stream end).
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		stopReason := mapOpenAIStreamFinish(*choice.FinishReason)
		deltaJSON, _ := json.Marshal(map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   stopReason,
				"stop_sequence": nil,
			},
			"usage": map[string]any{
				"output_tokens": 0,
			},
		})
		return []byte("event: message_delta\ndata: " + string(deltaJSON)), nil
	}

	// Content delta → text_delta.
	if content, ok := choice.Delta["content"].(string); ok && content != "" {
		deltaJSON, _ := json.Marshal(map[string]any{
			"type": "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type": "text_delta",
				"text": content,
			},
		})
		return []byte("event: content_block_delta\ndata: " + string(deltaJSON)), nil
	}

	// Reasoning content → thinking_delta.
	if reasoning, ok := choice.Delta["reasoning_content"].(string); ok && reasoning != "" {
		deltaJSON, _ := json.Marshal(map[string]any{
			"type": "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type":     "thinking_delta",
				"thinking": reasoning,
			},
		})
		return []byte("event: content_block_delta\ndata: " + string(deltaJSON)), nil
	}

	// Empty delta (role announcement, e.g. delta:{"role":"assistant"}) — pass through as-is.
	// The Claude SDK ignores unrecognised data: lines.
	return reconstructSSEEvent(evt), nil
}

// mapOpenAIStreamFinish maps OpenAI finish_reason to Anthropic stop_reason.
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

// StripProviderPrefix removes a leading "provider/" segment (e.g. "anthropic/")
// from model names. Claude Code sends "anthropic/claude-opus-4-8" but mappings
// and upstreams expect the bare model id.
func StripProviderPrefix(model string) string {
	if i := strings.IndexByte(model, '/'); i >= 0 {
		return model[i+1:]
	}
	return model
}

// resolveModel determines the output model ID and downstream protocol override.
// Priority: mapping (prefix match or catch-all) → passthrough (original input) → fallback.
func resolveModel(inputModel, fallback string, mapping ModelMap) (string, string) {
	if inputModel != "" {
		// Strip provider prefix before matching (Claude Code sends "anthropic/claude-opus-4-8").
		bare := StripProviderPrefix(inputModel)
		if target, proto, ok := mapping.Apply(bare); ok {
			return target, proto
		}
		// Also try matching with the original (includes prefix) for backwards compatibility.
		if target, proto, ok := mapping.Apply(inputModel); ok {
			return target, proto
		}
		return inputModel, ""
	}
	return fallback, ""
}

// extractModelFromData extracts the "model" field from raw JSON request data.
func extractModelFromData(data []byte) string {
	var raw struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return ""
	}
	return raw.Model
}

// Convert detects the input body format and performs bidirectional
// conversion between OpenAI Chat Completions and Anthropic Messages formats.
func Convert(body []byte, opts *ConvertOptions) ([]byte, error) {
	if opts == nil {
		opts = &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192}
	}

	if len(body) == 0 {
		return body, nil
	}

	// Auto-detect SSE framing: if the body starts with data:, event:,
	// or id:, route through SSE-aware handling.
	if isSSE(body) {
		slog.Debug("detected SSE framing → converting via ConvertSSE")
		return ConvertSSE(body, opts)
	}

	// Try parsing as a generic object to detect format.
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		slog.Debug("not JSON, passing through", "err", err)
		return body, nil
	}

	// Warn when the response contains an error field (OpenAI or Anthropic format).
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

	// Detect: Responses API request (input field, no messages field).
	// Must run BEFORE the protocol-passthrough block below: Responses requests
	// have a "model" string field, which makes isOpenAIRequest return true,
	// incorrectly triggering passthrough instead of full conversion to Chat/Anthropic.
	if isResponsesRequest(raw) {
		slog.Debug("detected Responses API Request → converting to Chat/Anthropic")
		if opts.SID != "" {
			markResponsesSession(opts.SID)
		}
		return convertResponsesRequest(body, opts)
	}

	// Detect: session is a Responses session — upstream response needs
	// converting back to Responses API format.
	if opts != nil && opts.SID != "" && IsResponsesSession(opts.SID) {
		slog.Debug("Responses session upstream response → converting to Responses format")
		return convertToResponsesResponse(body, opts)
	}

	// Resolve downstream protocol override from model mapping.
	// If set and the input format matches, rewrite model name and pass through.
	if opts.ModelMap != nil {
		downstreamProtocol := ""
		var targetModel string
		if m, ok := raw["model"].(string); ok {
			targetModel, downstreamProtocol = resolveModel(m, opts.Model, opts.ModelMap)
		}

		passthrough := false
		if downstreamProtocol == "openai" && (isOpenAIRequest(raw) || isOpenAIResponse(raw)) {
			passthrough = true
		}
		if downstreamProtocol == "anthropic" && (isAnthropicRequest(raw) || isAnthropicResponse(raw)) {
			passthrough = true
		}

		if passthrough {
			modelChanged := targetModel != "" && raw["model"] != targetModel
			if modelChanged {
				raw["model"] = targetModel
			}

			// Inject max_completion_tokens for requests missing it.
			// Codex never sends max_tokens / max_completion_tokens, and
			// upstream APIs default to very low limits (~100-200 tokens),
			// causing the model to be cut off mid-turn.
			// ponytail: only inject for request-like bodies, not responses.
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
				modified, err := json.Marshal(raw)
				if err == nil {
					slog.Debug("protocol passthrough with model rewrite", "model", targetModel)
					return modified, nil
				}
			}
			slog.Debug("protocol passthrough")
			return body, nil
		}
	}



	// Detect: Anthropic Messages Request (messages + max_tokens or Anthropic content blocks).
	if isAnthropicRequest(raw) {
		slog.Debug("detected Anthropic Request → converting to OpenAI")
		return convertAnthropicRequestToOpenAI(body, opts)
	}

	// Detect: OpenAI Chat Completions Response (choices array).
	if isOpenAIResponse(raw) {
		slog.Debug("detected OpenAI Response → converting to Anthropic")
		return convertOpenAIResponseToAnthropic(body, opts)
	}

	// Detect: Anthropic Messages Response (type "message" and stop_reason/usage).
	if isAnthropicResponse(raw) {
		slog.Debug("detected Anthropic Response → converting to OpenAI")
		return convertAnthropicResponseToOpenAI(body)
	}

	// Detect: OpenAI Chat Completions Request (model or messages field).
	if hasStringField(raw, "model") || hasArrayField(raw, "messages") {
		slog.Debug("detected OpenAI Chat Request → converting to Anthropic")
		return convertOpenAIRequestToAnthropic(body, opts)
	}

	slog.Debug("unknown format, passing through")
	return body, nil
}

// ---------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------

func hasStringField(m map[string]any, key string) bool {
	_, ok := m[key].(string)
	return ok
}

func hasArrayField(m map[string]any, key string) bool {
	_, ok := m[key].([]any)
	return ok
}

func isAnthropicResponse(m map[string]any) bool {
	t, ok := m["type"].(string)
	if !ok || t != "message" {
		return false
	}
	// Look for the presence of either stop_reason or usage.
	if _, ok := m["stop_reason"]; ok {
		return true
	}
	if _, ok := m["usage"]; ok {
		return true
	}
	return false
}

// isAnthropicRequest detects an Anthropic Messages API request body.
// Key signals: messages content as typed-block arrays, max_tokens (required),
// Anthropic-specific content block types (thinking, tool_use, tool_result, image).
func isAnthropicRequest(m map[string]any) bool {
	msgs, ok := m["messages"].([]any)
	if !ok || len(msgs) == 0 {
		return false
	}

	// Quick rejection: OpenAI-specific fields that Anthropic never uses.
	if _, ok := m["frequency_penalty"]; ok {
		return false
	}
	if _, ok := m["presence_penalty"]; ok {
		return false
	}
	if _, ok := m["logit_bias"]; ok {
		return false
	}
	if _, ok := m["response_format"]; ok {
		return false
	}
	if _, ok := m["n"]; ok {
		return false
	}
	if _, ok := m["seed"]; ok {
		return false
	}

	// Quick rejection: OpenAI uses "stop" (string/array), Anthropic uses "stop_sequences" (array).
	if _, ok := m["stop"]; ok {
		return false
	}

	// OpenAI tool_choice can be a string (e.g. "auto", "required"); Anthropic is always an object.
	if _, ok := m["tool_choice"].(string); ok {
		return false
	}

	// Anthropic tools have flat name + input_schema at top level, no "function" wrapper.
	if tools, ok := m["tools"].([]any); ok && len(tools) > 0 {
		if firstTool, ok := tools[0].(map[string]any); ok {
			_, hasFunction := firstTool["function"]
			_, hasName := firstTool["name"]
			_, hasInputSchema := firstTool["input_schema"]
			if !hasFunction && hasName && hasInputSchema {
				return true
			}
		}
	}

	// Check messages for OpenAI-only patterns (role: system/tool/function in array).
	for _, msg := range msgs {
		mmsg, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		role, _ := mmsg["role"].(string)
		if role == "system" || role == "tool" || role == "function" {
			return false
		}
	}

	// Check first few messages for Anthropic-specific content block types.
	for i := 0; i < len(msgs) && i < 3; i++ {
		msg, ok := msgs[i].(map[string]any)
		if !ok {
			continue
		}
		content, ok := msg["content"].([]any)
		if !ok {
			continue
		}
		for _, block := range content {
			b, ok := block.(map[string]any)
			if !ok {
				continue
			}
			bt, _ := b["type"].(string)
			switch bt {
			case "thinking", "tool_use", "tool_result", "image":
				// These types only exist in Anthropic content blocks.
				return true
			}
		}
	}

	// Top-level system field is Anthropic-specific.
	if _, ok := m["system"]; ok {
		return true
	}

	// Anthropic's tool_choice uses "any" or "tool" (not "required" or "function").
	if tc, ok := m["tool_choice"].(map[string]any); ok {
		if t, _ := tc["type"].(string); t == "function" {
			return false // OpenAI-style function-type tool_choice
		}
		if t, _ := tc["type"].(string); t == "any" || t == "tool" {
			return true
		}
	}

	// max_tokens is REQUIRED in Anthropic. If present AND the model doesn't
	// look OpenAI-ish (gpt/o1/o3/deepseek/gemini), it's likely an Anthropic
	// request. Accept both array content (current format) and string content
	// (deprecated format still accepted by the API).
	if _, ok := m["max_tokens"]; ok && !hasOpenAIStyleModel(m) {
		for i := 0; i < len(msgs) && i < 2; i++ {
			msg, ok := msgs[i].(map[string]any)
			if !ok {
				continue
			}
			switch c := msg["content"].(type) {
			case []any, string:
				return true
			case nil:
				continue
			default:
				_ = c
			}
		}
	}

	return false
}

// hasOpenAIStyleModel returns true if the model field value looks like an
// OpenAI-compatible model ID rather than Anthropic. Used to prevent false
// positives where an OpenAI request with max_tokens + array content would
// otherwise be classified as an Anthropic request.
func hasOpenAIStyleModel(m map[string]any) bool {
	model, ok := m["model"].(string)
	if !ok || model == "" {
		return false
	}
	// Check for common OpenAI / downstream model prefixes.
	openAIPrefixes := []string{
		"gpt-", "o1", "o3", "deepseek", "gemini-", "glm-",
	}
	modelLower := strings.ToLower(model)
	for _, p := range openAIPrefixes {
		if strings.HasPrefix(modelLower, p) {
			return true
		}
	}
	return false
}

// isOpenAIResponse detects an OpenAI Chat Completions response body.
// Key signals: choices array with message/finish_reason fields.
func isOpenAIResponse(m map[string]any) bool {
	choices, ok := m["choices"].([]any)
	if !ok || len(choices) == 0 {
		return false
	}
	if choice, ok := choices[0].(map[string]any); ok {
		if _, ok := choice["finish_reason"]; ok {
			return true
		}
		if _, ok := choice["message"]; ok {
			return true
		}
		if _, ok := choice["delta"]; ok {
			return true // streaming chunk
		}
	}
	return false
}

// isOpenAIRequest detects an OpenAI Chat Completions request body.
// It checks for model/messages fields while rejecting known non-request formats.
func isOpenAIRequest(m map[string]any) bool {
	if !hasStringField(m, "model") && !hasArrayField(m, "messages") {
		return false
	}
	if isAnthropicRequest(m) {
		return false
	}
	if isAnthropicResponse(m) {
		return false
	}
	if isOpenAIResponse(m) {
		return false
	}
	return true
}

// isOpenAIStreamChunk detects an OpenAI SSE streaming data payload.
// Format: {"choices":[{"index":0,"delta":{...},"finish_reason":null}]}
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
// SSE stream lifecycle management
// ---------------------------------------------------------------------------

// streamStates maps GOST session IDs to active StreamConverter instances.
// Each stream's state is stored when a "start" phase event arrives and
// deleted when the "end" phase event is processed.
var streamStates sync.Map // sid (string) -> *StreamConverter

// extractDeclaredTools returns declared tool names from ConvertOptions, or nil.
func extractDeclaredTools(opts *ConvertOptions) []string {
	if opts != nil && len(opts.DeclaredTools) > 0 {
		return opts.DeclaredTools
	}
	return nil
}

// newStreamConverterFromData resolves the model from raw data bytes, does a
// reverse model-map lookup for the safety classifier, and returns a configured
// StreamConverter.
func newStreamConverterFromData(data []byte, opts *ConvertOptions) *StreamConverter {
	streamModel := opts.Model
	downstreamProtocol := ""
	// Strip SSE framing so we can parse the model from the chunk payload.
	payload := extractSSEPayload(data)
	if model := extractModelFromData(payload); model != "" {
		streamModel, downstreamProtocol = resolveModel(model, opts.Model, opts.ModelMap)
	}
	if opts.RequestModel != "" {
		streamModel = opts.RequestModel
	} else if opts.ModelMap != nil {
		if sourcePrefix := opts.ModelMap.SourcePrefix(streamModel); sourcePrefix != "" {
			streamModel = sourcePrefix
		}
	}
	sc := NewStreamConverter(streamModel, opts.ReasoningCache, extractDeclaredTools(opts))
	sc.downstreamProtocol = downstreamProtocol
	return sc
}

// HandleSSEEvent processes an SSE stream lifecycle event.
// It routes to the appropriate StreamConverter method based on the phase.
func HandleSSEEvent(sid, phase string, eventIndex int, data []byte, opts *ConvertOptions) ([]byte, error) {
	// Route Responses API sessions to ResponsesStreamConverter.
	if IsResponsesSession(sid) {
		return handleResponsesSSEEvent(sid, phase, eventIndex, data, opts)
	}

	switch StreamPhase(phase) {
	case StreamPhaseStart:
		sc := newStreamConverterFromData(data, opts)
		streamStates.Store(sid, sc)
		startData := sc.HandleStreamStart()
		// First event data is now attached to the start phase signal
		// (sniffer sends the first real SSE event with sse_phase:"start").
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

		v, ok := streamStates.Load(sid)
		if !ok {
			slog.Warn("HandleSSEEvent: unknown stream, starting new", "sid", sid)
			sc := newStreamConverterFromData(payload, opts)
			streamStates.Store(sid, sc)
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
		sc := v.(*StreamConverter)

		// Protocol passthrough: if the payload format matches the downstream
		// protocol, rewrite model in chunk and pass through.
		if sc.downstreamProtocol == "openai" && isOpenAIStreamChunk(payload) {
			if sc.model != "" {
				var chunkRaw map[string]any
				if err := json.Unmarshal(payload, &chunkRaw); err == nil {
					if old, ok := chunkRaw["model"].(string); ok && old != sc.model {
						chunkRaw["model"] = sc.model
						if newPayload, err := json.Marshal(chunkRaw); err == nil {
							evt := parseSSEEvent(data)
							evt.Data = string(newPayload)
							slog.Debug("stream: protocol=openai passthrough with model rewrite", "model", sc.model)
							return reconstructSSEEvent(evt), nil
						}
					}
				}
			}
			slog.Debug("stream: protocol=openai, OpenAI chunk -> passthrough")
			return data, nil
		}

		out, err := sc.HandleChunk(payload)
		if err != nil {
			return nil, err
		}
		if out == nil {
			return []byte{}, nil // consumed, swallow original
		}
		return out, nil

	case StreamPhaseEnd:
		v, ok := streamStates.Load(sid)
		if !ok {
			slog.Warn("HandleSSEEvent: unknown stream for end phase", "sid", sid)
			return nil, nil
		}
		sc := v.(*StreamConverter)
		streamStates.Delete(sid)
		return sc.HandleStreamEnd(), nil
	}

	return nil, fmt.Errorf("unknown sse_phase: %s", phase)
}

// extractSSEPayload parses SSE event text and returns just the data payload.
// The GOST sniffer sends raw SSE event text (e.g., "data: {...}") to the
// plugin, but HandleChunk needs just the JSON payload without SSE framing.
// "[DONE]" markers are consumed silently (return nil, no error).
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

// handleResponsesSSEEvent routes SSE lifecycle events for Responses API sessions.
func handleResponsesSSEEvent(sid, phase string, _ int, data []byte, opts *ConvertOptions) ([]byte, error) {
	switch StreamPhase(phase) {
	case StreamPhaseStart:
		model := opts.Model
		payload := extractSSEPayload(data)
			if opts.RequestModel != "" {
				model = opts.RequestModel
			} else if m := extractModelFromData(payload); m != "" {
		} else if m := extractModelFromData(payload); m != "" {
			model, _ = resolveModel(m, opts.Model, opts.ModelMap)
		}
		var handler responsesStreamHandler
		if isAnthropicStreamEvent(payload) {
			handler = NewAnthropicStreamConverter(model)
		} else {
			handler = NewResponsesStreamConverter(model)
			if opts != nil && opts.CodexToolContext != nil {
				handler.(*ResponsesStreamConverter).SetToolSpecs(opts.CodexToolContext.byChatName)
			}
		}
		responsesStreamStates.Store(sid, handler)
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
			return []byte{}, nil // [DONE] marker, swallow.
		}
		v, ok := responsesStreamStates.Load(sid)
		if !ok {
			// Lazy-create converter from first event.
			model := opts.Model
			if opts.RequestModel != "" {
				model = opts.RequestModel
			} else if m := extractModelFromData(payload); m != "" {
			} else if m := extractModelFromData(payload); m != "" {
				model, _ = resolveModel(m, opts.Model, opts.ModelMap)
			}
			var handler responsesStreamHandler
			if isAnthropicStreamEvent(payload) {
				handler = NewAnthropicStreamConverter(model)
			} else {
				handler = NewResponsesStreamConverter(model)
				if opts != nil && opts.CodexToolContext != nil {
					handler.(*ResponsesStreamConverter).SetToolSpecs(opts.CodexToolContext.byChatName)
				}
			}
			responsesStreamStates.Store(sid, handler)
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
		out, err := v.(responsesStreamHandler).HandleChunk(payload)
		if err != nil {
			return nil, err
		}
		if out == nil {
			return []byte{}, nil // consumed, swallow original
		}
		return out, nil

	case StreamPhaseEnd:
		v, ok := responsesStreamStates.Load(sid)
		if !ok {
			unmarkResponsesSession(sid)
			return nil, nil
		}
		handler := v.(responsesStreamHandler)
		responsesStreamStates.Delete(sid)
		unmarkResponsesSession(sid)
		return handler.HandleStreamEnd(), nil

	case StreamPhaseError:
		v, ok := responsesStreamStates.Load(sid)
		if !ok {
			unmarkResponsesSession(sid)
			return nil, nil
		}
		handler := v.(responsesStreamHandler)
		responsesStreamStates.Delete(sid)
		unmarkResponsesSession(sid)
		msg := "stream error"
		if opts != nil && opts.StreamErrorMsg != "" {
			msg = opts.StreamErrorMsg
		}
		return handler.EmitError(msg), nil
	}
	return nil, nil
}

// ---------------------------------------------------------------------------
// Utility helpers (canonical JSON, tool sorting, ID normalization)
// ---------------------------------------------------------------------------

// ensureObjectSchema guarantees the schema is a JSON object with "type":"object".
// OpenAI spec requires {"type":"object", ...} but some schemas omit it.
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

// ensureToolID normalises a tool call ID to Anthropic's "toolu_" prefix.
// OpenAI IDs like "call_xxx" are not recognised by the Anthropic SDK.
func ensureToolID(id string) string {
	if strings.HasPrefix(id, "toolu_") {
		return id
	}
	return "toolu_" + randHex(24)
}

// randHex returns n crypto-random hex characters.
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
