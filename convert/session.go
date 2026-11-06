package convert

import (
	"encoding/json"
	"log/slog"
	"sync"
)

// SessionStore holds per-session state with max-size FIFO eviction.
// The rewriter server owns this and passes it via ConvertOptions.
type SessionStore struct {
	mu       sync.RWMutex
	sessions map[string]*Session
	order    []string
	maxSize  int

	// ThoughtSignatures persists functionCall → thoughtSignature mappings across
	// session lifecycles (Delete/Set clears the Session but signatures must survive).
	ThoughtSignatures map[string]map[string]string // sid → funcName → thoughtSig
	// sigOrder bounds ThoughtSignatures: the oldest sid is dropped once the
	// store holds maxSize of them.
	sigOrder []string
}

// NewSessionStore creates a session store with the given max size.
func NewSessionStore() *SessionStore {
	return &SessionStore{
		sessions: make(map[string]*Session),
		order:    make([]string, 0, 1000),
		maxSize:  1000,
	}
}

// Session holds per-session streaming state.
type Session struct {
	ID            string
	From          Protocol
	To            Protocol
	IsResponses   bool
	StreamHandler responsesStreamHandler
}

// Get returns the session by ID, or nil.
func (s *SessionStore) Get(sid string) *Session {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sessions[sid]
}

// Set stores a session, replacing any existing entry. Session is treated as
// write-once: pointer replacement (never in-place mutation) so a caller holding
// a Get-returned pointer sees a stable snapshot — no race with a concurrent Set.
func (s *SessionStore) Set(sid string, sess *Session) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.sessions[sid]; !exists {
		if len(s.sessions) >= s.maxSize {
			s.evictLocked()
		}
		s.order = append(s.order, sid)
	}
	s.sessions[sid] = sess
}

// Delete removes a session. It prunes the order slice too — otherwise long-running
// streaming traffic (Set on start, Delete on end) grows order without bound: the map
// stays small, eviction never fires, and order leaks one entry per stream.
func (s *SessionStore) Delete(sid string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, sid)
	for i, id := range s.order {
		if id == sid {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
}

// GetThoughtSig returns the stored thought signature for a (sid, funcName) pair.
func (s *SessionStore) GetThoughtSig(sid, funcName string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.ThoughtSignatures == nil {
		return ""
	}
	return s.ThoughtSignatures[sid][funcName]
}

// SetThoughtSig stores a thought signature for a (sid, funcName) pair.
func (s *SessionStore) SetThoughtSig(sid, funcName, sig string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ThoughtSignatures == nil {
		s.ThoughtSignatures = make(map[string]map[string]string)
	}
	if s.ThoughtSignatures[sid] == nil {
		s.ThoughtSignatures[sid] = make(map[string]string)
		s.sigOrder = append(s.sigOrder, sid)
		if s.maxSize > 0 && len(s.sigOrder) > s.maxSize {
			oldest := s.sigOrder[0]
			s.sigOrder = s.sigOrder[1:]
			delete(s.ThoughtSignatures, oldest)
		}
	}
	s.ThoughtSignatures[sid][funcName] = sig
}

// GetThoughtSigFor returns the signature stored for a tool call: the call's
// own ID first, then the function name as a fallback for clients that drop
// or rewrite the ID.
func (s *SessionStore) GetThoughtSigFor(sid, callID, name string) string {
	if callID != "" {
		if sig := s.GetThoughtSig(sid, callID); sig != "" {
			return sig
		}
	}
	if name == "" {
		return ""
	}
	return s.GetThoughtSig(sid, name)
}

// SetThoughtSigFor stores sig under the call's ID and its function name. The
// ID is what the client sends back with the tool result; the name is only a
// fallback for clients that lose it.
func (s *SessionStore) SetThoughtSigFor(sid, callID, name, sig string) {
	if sig == "" {
		return
	}
	if callID != "" {
		s.SetThoughtSig(sid, callID, sig)
	}
	if name != "" {
		s.SetThoughtSig(sid, name, sig)
	}
}

func (s *SessionStore) evictLocked() {
	for len(s.order) > 0 {
		oldest := s.order[0]
		s.order = s.order[1:]
		if _, ok := s.sessions[oldest]; ok {
			delete(s.sessions, oldest)
			return
		}
		// Stale entry (already removed via Delete) — skip.
	}
}

// PassthroughStreamHandler passes through events with optional model rewrite.
type PassthroughStreamHandler struct {
	model           string
	protocol        Protocol
	filterRedacted  bool
	redactedIndices map[int]bool // content block indices whose start was filtered
}

// NewPassthroughStreamHandler creates a passthrough handler for the target model.
func NewPassthroughStreamHandler(model string, protocol Protocol, filterRedacted bool) *PassthroughStreamHandler {
	return &PassthroughStreamHandler{
		model:           model,
		protocol:        protocol,
		filterRedacted:  filterRedacted,
		redactedIndices: make(map[int]bool),
	}
}

func (p *PassthroughStreamHandler) HandleStreamStart() []byte { return nil }

func (p *PassthroughStreamHandler) HandleChunk(data []byte) ([]byte, error) {
	if p.protocol == ProtocolAnthropic {
		return p.anthropicPassthrough(data), nil
	}
	if p.protocol == ProtocolOpenAIChat {
		return p.openaiPassthrough(data), nil
	}
	return data, nil
}

func (p *PassthroughStreamHandler) HandleStreamEnd() []byte { return nil }

// EmitError returns an Anthropic-formatted error SSE event. The payload is
// JSON-encoded so quotes, newlines, and backslashes in the message cannot
// corrupt the event.
func (p *PassthroughStreamHandler) EmitError(message string) []byte {
	b, _ := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "stream_error", "message": message},
	})
	return []byte("event: error\ndata: " + string(b))
}

// ---------------------------------------------------------------------------
// GeminiStreamHandler — converts Gemini SSE chunks to the target protocol.
// Gemini SSE responses deliver complete GeminiChatResponse objects per chunk,
// so each chunk is independently convertible — no state machine needed.
// ---------------------------------------------------------------------------

type GeminiStreamHandler struct {
	convertFn func([]byte, *ConvertOptions) ([]byte, error)
	opts      *ConvertOptions
}

func NewGeminiStreamHandler(convertFn func([]byte, *ConvertOptions) ([]byte, error), opts *ConvertOptions) *GeminiStreamHandler {
	return &GeminiStreamHandler{convertFn: convertFn, opts: opts}
}

func (h *GeminiStreamHandler) HandleStreamStart() []byte { return nil }

func (h *GeminiStreamHandler) HandleChunk(data []byte) ([]byte, error) {
	out, err := h.convertFn(data, h.opts)
	if err != nil {
		return nil, err
	}
	if out == nil {
		return nil, nil
	}
	// Wrap in SSE data: framing — Gemini SSE delivers complete JSON objects
	// per chunk, so the converted result is also a complete response.
	return append([]byte("data: "), append(out, '\n')...), nil
}

func (h *GeminiStreamHandler) HandleStreamEnd() []byte { return nil }

func (h *GeminiStreamHandler) EmitError(message string) []byte {
	b, _ := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"message": message},
	})
	return append([]byte("event: error\ndata: "), append(b, '\n')...)
}

// anthropicPassthrough rewrites model in message_start events and
// filters redacted_thinking content blocks when enabled.
func (p *PassthroughStreamHandler) anthropicPassthrough(data []byte) []byte {
	evt := parseSSEEvent(data)
	if evt.Data == "" {
		return data
	}
	var raw map[string]any
	if err := json.Unmarshal([]byte(evt.Data), &raw); err != nil {
		return data
	}
	t, _ := raw["type"].(string)

	// Filter OpenRouter processing events (non-standard, no protocol value).
	if t == "processing" {
		return nil
	}

	// Filter redacted_thinking content blocks when enabled.
	if p.filterRedacted {
		switch t {
		case "content_block_start":
			if cb, ok := raw["content_block"].(map[string]any); ok {
				if cbt, _ := cb["type"].(string); cbt == "redacted_thinking" {
					if idx, ok := raw["index"].(float64); ok {
						p.redactedIndices[int(idx)] = true
					}
					return nil
				}
			}
		case "content_block_delta":
			if d, ok := raw["delta"].(map[string]any); ok {
				if dt, _ := d["type"].(string); dt == "redacted_thinking_delta" {
					return nil
				}
			}
		case "content_block_stop":
			if idx, ok := raw["index"].(float64); ok {
				if p.redactedIndices[int(idx)] {
					delete(p.redactedIndices, int(idx))
					return nil
				}
			}
		}
	}

	// Model rewrite in message_start events.
	if t == "message_start" {
		if msg, ok := raw["message"].(map[string]any); ok {
			if old, _ := msg["model"].(string); p.model != "" && old != "" && old != p.model {
				msg["model"] = p.model
				raw["message"] = msg
				if newData, err := json.Marshal(raw); err == nil {
					evt.Data = string(newData)
					return reconstructSSEEvent(evt)
				}
			}
		}
	}
	return data
}

// openaiPassthrough rewrites model in the chunk payload.
func (p *PassthroughStreamHandler) openaiPassthrough(data []byte) []byte {
	payload := extractSSEPayload(data)
	if payload == nil {
		return data
	}
	var raw map[string]any
	if err := json.Unmarshal(payload, &raw); err != nil {
		return data
	}
	changed := false
	if old, ok := raw["model"].(string); ok && old != p.model {
		raw["model"] = p.model
		changed = true
	}
	if tidyOpenAIChunk(raw) {
		changed = true
	}
	if changed {
		if newPayload, err := json.Marshal(raw); err == nil {
			evt := parseSSEEvent(data)
			evt.Data = string(newPayload)
			slog.Debug("stream: passthrough with tidy", "model", p.model)
			return reconstructSSEEvent(evt)
		}
	}
	return data
}
