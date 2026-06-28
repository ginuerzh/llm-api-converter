package convert

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ReasoningCache is a thread-safe cache that maps reasoning_content to tool
// call IDs, assistant text, and tool context for replay across requests.
//
// DeepSeek V4 requires that if a prior response contained both reasoning_content
// AND tool_calls / text, any subsequent request replaying that assistant message
// MUST include the original reasoning_content. This cache stores those mappings
// so the converter can re-inject them when Claude Code compresses conversations.
type ReasoningCache struct {
	mu      sync.RWMutex
	entries map[string]*cacheEntry
	order   []string // FIFO insertion order for eviction
	maxSize int

	// Text-keyed reasoning (assistant text SHA256 → reasoning).
	textEntries map[string]*cacheEntry
	textOrder   []string

	// Tool-context-keyed reasoning (SHA256(toolCtx+assistantText) → reasoning).
	ctxEntries map[string]*cacheEntry
	ctxOrder   []string

	// Persistence.
	path    string
	dirty   bool
	closeCh chan struct{}

	// Eviction policy.
	maxAge    time.Duration
	maxBytes  int // 0 = unlimited
}

// cacheEntry holds a reasoning value with metadata.
type cacheEntry struct {
	Reasoning string    `json:"reasoning"`
	AddedAt   time.Time `json:"added_at"`
}

// reasoningCachePersisted is the on-disk format.
type reasoningCachePersisted struct {
	Version              int                  `json:"version"`
	Note                 string               `json:"note"`
	UpdatedAt            int64                `json:"updated_at"`
	ToolCallReasoning    map[string]cacheEntry `json:"tool_call_reasoning,omitempty"`
	AssistantReasoning   map[string]cacheEntry `json:"assistant_reasoning,omitempty"`
	ContextReasoning     map[string]cacheEntry `json:"context_reasoning,omitempty"`
}

// NewReasoningCache creates a cache with the given maxSize (minimum 1).
func NewReasoningCache(maxSize int) *ReasoningCache {
	if maxSize < 1 {
		maxSize = 1
	}
	return &ReasoningCache{
		entries:     make(map[string]*cacheEntry),
		order:       make([]string, 0, maxSize),
		maxSize:     maxSize,
		textEntries: make(map[string]*cacheEntry),
		textOrder:   make([]string, 0, maxSize),
		ctxEntries:  make(map[string]*cacheEntry),
		ctxOrder:    make([]string, 0, maxSize),
		closeCh:     make(chan struct{}),
		maxAge:      30 * 24 * time.Hour, // 30 days default
	}
}

// ---- Tool call ID keyed ----

// cacheKey returns a deterministic SHA256 hex hash of the sorted tool call IDs.
func (c *ReasoningCache) cacheKey(toolIDs []string) string {
	sorted := make([]string, len(toolIDs))
	copy(sorted, toolIDs)
	sort.Strings(sorted)
	h := sha256.Sum256([]byte(strings.Join(sorted, "")))
	return fmt.Sprintf("%x", h)
}

// Put stores the reasoning content keyed by the given tool call IDs.
func (c *ReasoningCache) Put(toolIDs []string, reasoning string) {
	if reasoning == "" || len(toolIDs) == 0 {
		return
	}
	key := c.cacheKey(toolIDs)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.putLocked(c.entries, c.order, &c.order, key, reasoning)
}

// Get retrieves the cached reasoning content for the given tool call IDs.
func (c *ReasoningCache) Get(toolIDs []string) (string, bool) {
	if len(toolIDs) == 0 {
		return "", false
	}
	key := c.cacheKey(toolIDs)
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.getLocked(c.entries, key)
}

// ---- Assistant text keyed ----

// PutText stores reasoning keyed by the SHA256 of the assistant text content.
func (c *ReasoningCache) PutText(text, reasoning string) {
	if text == "" || reasoning == "" {
		return
	}
	key := sha256Key(text)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.putLocked(c.textEntries, c.textOrder, &c.textOrder, key, reasoning)
	c.markDirtyLocked()
}

// GetText retrieves reasoning by assistant text content.
func (c *ReasoningCache) GetText(text string) (string, bool) {
	if text == "" {
		return "", false
	}
	key := sha256Key(text)
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.getLocked(c.textEntries, key)
}

// ---- Tool context keyed ----

// PutContext stores reasoning keyed by tool context + assistant text.
func (c *ReasoningCache) PutContext(ctxParts []string, assistantText, reasoning string) {
	key := contextKey(ctxParts, assistantText)
	if key == "" || reasoning == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.putLocked(c.ctxEntries, c.ctxOrder, &c.ctxOrder, key, reasoning)
	c.markDirtyLocked()
}

// GetContext retrieves reasoning by tool context + assistant text.
func (c *ReasoningCache) GetContext(ctxParts []string, assistantText string) (string, bool) {
	key := contextKey(ctxParts, assistantText)
	if key == "" {
		return "", false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.getLocked(c.ctxEntries, key)
}

// ---- Internal helpers ----

func (c *ReasoningCache) putLocked(entries map[string]*cacheEntry, order []string, orderPtr *[]string, key, reasoning string) {
	now := time.Now()
	if existing, ok := entries[key]; ok {
		existing.Reasoning = reasoning
		existing.AddedAt = now
		return
	}
	if len(entries) >= c.maxSize {
		evictLocked(entries, orderPtr)
	}
	entries[key] = &cacheEntry{Reasoning: reasoning, AddedAt: now}
	*orderPtr = append(*orderPtr, key)
}

func (c *ReasoningCache) getLocked(entries map[string]*cacheEntry, key string) (string, bool) {
	entry, ok := entries[key]
	if !ok {
		return "", false
	}
	if c.maxAge > 0 && time.Since(entry.AddedAt) > c.maxAge {
		delete(entries, key)
		return "", false
	}
	return entry.Reasoning, true
}

func evictLocked(entries map[string]*cacheEntry, orderPtr *[]string) {
	order := *orderPtr
	if len(order) == 0 {
		return
	}
	oldest := order[0]
	*orderPtr = order[1:]
	delete(entries, oldest)
}

func sha256Key(s string) string {
	h := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%x", h)
}

func contextKey(ctxParts []string, assistantText string) string {
	if len(ctxParts) == 0 || assistantText == "" {
		return ""
	}
	raw := strings.Join(ctxParts, "\n") + "\nassistant:" + assistantText
	return sha256Key(raw)
}

// ---- Composite retrieval ----

// GetBest retrieves reasoning from any available cache tier.
// Priority: tool call ID > tool context > assistant text.
func (c *ReasoningCache) GetBest(toolIDs []string, ctxParts []string, assistantText string) string {
	if reasoning, ok := c.Get(toolIDs); ok && reasoning != "" {
		return reasoning
	}
	if reasoning, ok := c.GetContext(ctxParts, assistantText); ok && reasoning != "" {
		return reasoning
	}
	if reasoning, ok := c.GetText(assistantText); ok && reasoning != "" {
		return reasoning
	}
	return ""
}

// ---- Bulk operations ----

// Delete removes the entry for the given tool call IDs.
func (c *ReasoningCache) Delete(toolIDs []string) {
	if len(toolIDs) == 0 {
		return
	}
	key := c.cacheKey(toolIDs)
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.entries, key)
}

// Len returns the total number of entries across all tiers.
func (c *ReasoningCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries) + len(c.textEntries) + len(c.ctxEntries)
}

// ---- Persistence ----

// SetPersistence configures file-backed persistence. If path is non-empty,
// the cache is loaded from the file and periodic saves are started.
func (c *ReasoningCache) SetPersistence(path string) {
	c.mu.Lock()
	c.path = path
	c.mu.Unlock()
	if path != "" {
		c.loadFromFile()
	}
}

// LoadFromFile loads cache state from the configured file path.
func (c *ReasoningCache) loadFromFile() {
	c.mu.Lock()
	path := c.path
	c.mu.Unlock()
	if path == "" {
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return // file not found or unreadable — start fresh
	}
	var p reasoningCachePersisted
	if err := json.Unmarshal(data, &p); err != nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	for id, entry := range p.ToolCallReasoning {
		c.entries[id] = &cacheEntry{Reasoning: entry.Reasoning, AddedAt: entry.AddedAt}
		c.order = append(c.order, id)
	}
	for text, entry := range p.AssistantReasoning {
		c.textEntries[text] = &cacheEntry{Reasoning: entry.Reasoning, AddedAt: entry.AddedAt}
		c.textOrder = append(c.textOrder, text)
	}
	for ctx, entry := range p.ContextReasoning {
		c.ctxEntries[ctx] = &cacheEntry{Reasoning: entry.Reasoning, AddedAt: entry.AddedAt}
		c.ctxOrder = append(c.ctxOrder, ctx)
	}
}

// SaveNow flushes dirty entries to disk immediately.
func (c *ReasoningCache) SaveNow() error {
	c.mu.Lock()
	path := c.path
	if path == "" {
		c.mu.Unlock()
		return nil
	}
	p := c.snapshotLocked()
	c.dirty = false
	c.mu.Unlock()

	data, err := json.Marshal(p)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (c *ReasoningCache) snapshotLocked() *reasoningCachePersisted {
	tc := make(map[string]cacheEntry, len(c.entries))
	for k, v := range c.entries {
		tc[k] = *v
	}
	at := make(map[string]cacheEntry, len(c.textEntries))
	for k, v := range c.textEntries {
		at[k] = *v
	}
	ct := make(map[string]cacheEntry, len(c.ctxEntries))
	for k, v := range c.ctxEntries {
		ct[k] = *v
	}
	return &reasoningCachePersisted{
		Version:            2,
		Note:               "DeepSeek V4 reasoning_content cache for llm-api-converter",
		UpdatedAt:          time.Now().UnixMilli(),
		ToolCallReasoning:   tc,
		AssistantReasoning:  at,
		ContextReasoning:    ct,
	}
}

func (c *ReasoningCache) markDirtyLocked() {
	c.dirty = true
}
