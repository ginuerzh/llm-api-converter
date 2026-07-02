package convert

import (
	"encoding/json"
	"fmt"
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
	model    string
	protocol Protocol
}

// NewPassthroughStreamHandler creates a passthrough handler for the target model.
func NewPassthroughStreamHandler(model string, protocol Protocol) *PassthroughStreamHandler {
	return &PassthroughStreamHandler{model: model, protocol: protocol}
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

func (p *PassthroughStreamHandler) EmitError(message string) []byte {
	return []byte(fmt.Sprintf(
		`event: error`+"\n"+`data: {"type":"error","error":{"type":"stream_error","message":"%s"}}`,
		message,
	))
}

// anthropicPassthrough rewrites model in message_start events only.
func (p *PassthroughStreamHandler) anthropicPassthrough(data []byte) []byte {
	evt := parseSSEEvent(data)
	if evt.Data == "" {
		return data
	}
	var raw map[string]any
	if err := json.Unmarshal([]byte(evt.Data), &raw); err != nil {
		return data
	}
	if t, _ := raw["type"].(string); t == "message_start" {
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
	if old, ok := raw["model"].(string); ok && old != p.model {
		raw["model"] = p.model
		if newPayload, err := json.Marshal(raw); err == nil {
			evt := parseSSEEvent(data)
			evt.Data = string(newPayload)
			slog.Debug("stream: passthrough with model rewrite", "model", p.model)
			return reconstructSSEEvent(evt)
		}
	}
	return data
}
