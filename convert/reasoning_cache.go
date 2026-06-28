package convert

import (
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// ReasoningCache is a thread-safe FIFO-limited cache that maps sorted tool call
// ID hashes to their associated reasoning_content from the model response.
//
// DeepSeek V4 requires that if a prior response contained both reasoning_content
// AND tool_calls, any subsequent request replaying that assistant message MUST
// include the original reasoning_content. This cache stores those mappings so
// the converter can re-inject them when Claude Code compresses conversations
// (dropping thinking blocks but preserving tool_use blocks).
type ReasoningCache struct {
	mu      sync.RWMutex
	entries map[string]*cacheEntry
	order   []string // FIFO insertion order for eviction
	maxSize int
}

type cacheEntry struct {
	Reasoning string
	AddedAt   time.Time
}

// NewReasoningCache creates a cache with the given maxSize (minimum 1).
func NewReasoningCache(maxSize int) *ReasoningCache {
	if maxSize < 1 {
		maxSize = 1
	}
	return &ReasoningCache{
		entries: make(map[string]*cacheEntry),
		order:   make([]string, 0, maxSize),
		maxSize: maxSize,
	}
}

// cacheKey returns a deterministic SHA256 hex hash of the sorted tool call IDs.
func (c *ReasoningCache) cacheKey(toolIDs []string) string {
	sorted := make([]string, len(toolIDs))
	copy(sorted, toolIDs)
	sort.Strings(sorted)
	h := sha256.Sum256([]byte(strings.Join(sorted, "")))
	return fmt.Sprintf("%x", h)
}

// Put stores the reasoning content keyed by the given tool call IDs.
// If the cache is at capacity, the oldest entry is evicted (FIFO).
func (c *ReasoningCache) Put(toolIDs []string, reasoning string) {
	if reasoning == "" || len(toolIDs) == 0 {
		return
	}
	key := c.cacheKey(toolIDs)

	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.entries[key]; exists {
		c.entries[key].Reasoning = reasoning
		c.entries[key].AddedAt = time.Now()
		return
	}

	if len(c.entries) >= c.maxSize {
		c.evictLocked()
	}

	c.entries[key] = &cacheEntry{
		Reasoning: reasoning,
		AddedAt:   time.Now(),
	}
	c.order = append(c.order, key)
}

// Get retrieves the cached reasoning content for the given tool call IDs.
// Returns false if no entry exists.
func (c *ReasoningCache) Get(toolIDs []string) (string, bool) {
	if len(toolIDs) == 0 {
		return "", false
	}
	key := c.cacheKey(toolIDs)

	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.entries[key]
	if !ok {
		return "", false
	}
	return entry.Reasoning, true
}

// Delete removes the entry for the given tool call IDs.
func (c *ReasoningCache) Delete(toolIDs []string) {
	if len(toolIDs) == 0 {
		return
	}
	key := c.cacheKey(toolIDs)

	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.entries, key)
	c.removeOrderLocked(key)
}

// Len returns the number of entries in the cache.
func (c *ReasoningCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// evictLocked removes the oldest entry (FIFO). Must be called with c.mu held.
func (c *ReasoningCache) evictLocked() {
	if len(c.order) == 0 {
		return
	}
	oldest := c.order[0]
	c.order = c.order[1:]
	delete(c.entries, oldest)
}

// removeOrderLocked removes the given key from the order slice.
// Must be called with c.mu held.
func (c *ReasoningCache) removeOrderLocked(key string) {
	for i, k := range c.order {
		if k == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}
