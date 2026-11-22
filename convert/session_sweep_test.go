package convert

import (
	"fmt"
	"testing"
	"time"
)

// A full store must drop leaked sessions (an aborted peer never reaches the end
// phase, so its entry lingers) instead of evicting a stream that is still being
// written — evicting a live one makes the event phase lose its protocol and
// corrupt the rest of the answer.
func TestSessionStoreSweepsLeakedSessionsBeforeLiveOnes(t *testing.T) {
	store := NewSessionStore()
	old := time.Now().Add(-time.Hour)
	for i := 0; i < store.maxSize-1; i++ {
		sid := fmt.Sprintf("leaked-%d", i)
		store.Set(sid, &Session{ID: sid, Created: old, StreamHandler: NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)})
	}
	live := &Session{ID: "live", StreamHandler: NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)}
	store.Set("live", live)

	// Reaching capacity again triggers the sweep: the leaked entries go first.
	store.Set("new", &Session{ID: "new", StreamHandler: NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)})

	if store.Get("live") == nil {
		t.Fatal("live session was evicted in favour of a leaked one")
	}
	if store.Get("leaked-0") != nil {
		t.Fatal("leaked session survived the sweep")
	}
	if store.Get("new") == nil {
		t.Fatal("new session was not stored")
	}
}

// A live session that is merely slow must not be swept: only entries older than
// staleAfter are treated as leaked.
func TestSessionStoreKeepsRecentSessions(t *testing.T) {
	store := NewSessionStore()
	store.Set("recent", &Session{ID: "recent", StreamHandler: NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)})
	for i := 0; i < store.maxSize; i++ {
		store.Set(fmt.Sprintf("leaked-%d", i), &Session{ID: "x", Created: time.Now().Add(-time.Hour), StreamHandler: NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)})
	}
	if store.Get("recent") == nil {
		t.Fatal("recent session was swept")
	}
}
