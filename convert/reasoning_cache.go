package convert

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log/slog"
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
	toolStore ReasoningStore
	textStore ReasoningStore
	ctxStore  ReasoningStore

	// Eviction policy.
	maxAge time.Duration
}

// ReasoningStore is a thread-safe key-value store for one cache tier.
type ReasoningStore interface {
	Get(key string) *cacheEntry // nil if not found
	Set(key string, entry *cacheEntry)
	Delete(key string)
	Len() int
	Close() error // memoryStore: no-op; fileStore: final save + stop goroutine
}

// cacheEntry holds a reasoning value with metadata.
type cacheEntry struct {
	Reasoning string    `json:"reasoning"`
	AddedAt   time.Time `json:"added_at"`
}

// NewReasoningCache creates a cache with the given maxSize (minimum 1).
// Uses in-memory storage (memoryStore) — no persistence.
func NewReasoningCache(maxSize int) *ReasoningCache {
	if maxSize < 1 {
		maxSize = 1
	}
	return &ReasoningCache{
		toolStore: newMemoryStore(maxSize),
		textStore: newMemoryStore(maxSize),
		ctxStore:  newMemoryStore(maxSize),
		maxAge:    30 * 24 * time.Hour, // 30 days default
	}
}

// NewReasoningCacheWithFile creates a cache with file-backed persistence.
// Each tier is stored in its own JSON file: <path>.tool.json, .text.json, .context.json.
func NewReasoningCacheWithFile(path string, maxSize int) *ReasoningCache {
	rc := NewReasoningCache(maxSize)

	// Replace memory stores with file stores.
	rc.toolStore = newFileStore(path+".tool.json", maxSize)
	rc.textStore = newFileStore(path+".text.json", maxSize)
	rc.ctxStore = newFileStore(path+".context.json", maxSize)

	return rc
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
	c.toolStore.Set(key, &cacheEntry{Reasoning: reasoning, AddedAt: time.Now()})
}

// Get retrieves the cached reasoning content for the given tool call IDs.
func (c *ReasoningCache) Get(toolIDs []string) (string, bool) {
	if len(toolIDs) == 0 {
		return "", false
	}
	key := c.cacheKey(toolIDs)
	return c.getFromStore(c.toolStore, key)
}

// ---- Assistant text keyed ----

// PutText stores reasoning keyed by the SHA256 of the assistant text content.
func (c *ReasoningCache) PutText(text, reasoning string) {
	if text == "" || reasoning == "" {
		return
	}
	key := sha256Key(text)
	c.textStore.Set(key, &cacheEntry{Reasoning: reasoning, AddedAt: time.Now()})
}

// GetText retrieves reasoning by assistant text content.
func (c *ReasoningCache) GetText(text string) (string, bool) {
	if text == "" {
		return "", false
	}
	key := sha256Key(text)
	return c.getFromStore(c.textStore, key)
}

// ---- Tool context keyed ----

// PutContext stores reasoning keyed by tool context + assistant text.
func (c *ReasoningCache) PutContext(ctxParts []string, assistantText, reasoning string) {
	key := contextKey(ctxParts, assistantText)
	if key == "" || reasoning == "" {
		return
	}
	c.ctxStore.Set(key, &cacheEntry{Reasoning: reasoning, AddedAt: time.Now()})
}

// GetContext retrieves reasoning by tool context + assistant text.
func (c *ReasoningCache) GetContext(ctxParts []string, assistantText string) (string, bool) {
	key := contextKey(ctxParts, assistantText)
	if key == "" {
		return "", false
	}
	return c.getFromStore(c.ctxStore, key)
}

// ---- Internal helpers ----

func (c *ReasoningCache) getFromStore(s ReasoningStore, key string) (string, bool) {
	entry := s.Get(key)
	if entry == nil {
		return "", false
	}
	if c.maxAge > 0 && time.Since(entry.AddedAt) > c.maxAge {
		s.Delete(key)
		return "", false
	}
	return entry.Reasoning, true
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
	c.toolStore.Delete(key)
}

// Len returns the total number of entries across all tiers.
func (c *ReasoningCache) Len() int {
	return c.toolStore.Len() + c.textStore.Len() + c.ctxStore.Len()
}

// Close shuts down all stores. For fileStore, this triggers a final save
// and stops the auto-save goroutine. For memoryStore, it's a no-op.
func (c *ReasoningCache) Close() error {
	var firstErr error
	for _, s := range []ReasoningStore{c.toolStore, c.textStore, c.ctxStore} {
		if err := s.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// ---- memoryStore ----

// memoryStore is an in-memory ReasoningStore with FIFO eviction.
type memoryStore struct {
	mu      sync.RWMutex
	entries map[string]*cacheEntry
	order   []string
	maxSize int
}

func newMemoryStore(maxSize int) *memoryStore {
	return &memoryStore{
		entries: make(map[string]*cacheEntry),
		order:   make([]string, 0, maxSize),
		maxSize: maxSize,
	}
}

func (ms *memoryStore) Get(key string) *cacheEntry {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return ms.entries[key]
}

func (ms *memoryStore) Set(key string, entry *cacheEntry) {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	if existing, ok := ms.entries[key]; ok {
		existing.Reasoning = entry.Reasoning
		existing.AddedAt = entry.AddedAt
		return
	}
	if len(ms.entries) >= ms.maxSize {
		ms.evictLocked()
	}
	ms.entries[key] = entry
	ms.order = append(ms.order, key)
}

func (ms *memoryStore) Delete(key string) {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	delete(ms.entries, key)
}

func (ms *memoryStore) Len() int {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	return len(ms.entries)
}

func (ms *memoryStore) Close() error { return nil }

func (ms *memoryStore) evictLocked() {
	if len(ms.order) == 0 {
		return
	}
	oldest := ms.order[0]
	ms.order = ms.order[1:]
	delete(ms.entries, oldest)
}

// snapshot returns a copy of all entries (for fileStore persistence).
func (ms *memoryStore) snapshot() map[string]cacheEntry {
	ms.mu.RLock()
	defer ms.mu.RUnlock()
	snap := make(map[string]cacheEntry, len(ms.entries))
	for k, v := range ms.entries {
		snap[k] = *v
	}
	return snap
}

// load replaces entries from a snapshot (for fileStore persistence).
func (ms *memoryStore) load(entries map[string]cacheEntry) {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	ms.entries = make(map[string]*cacheEntry, len(entries))
	ms.order = make([]string, 0, len(entries))
	for k, v := range entries {
		entry := v // copy
		ms.entries[k] = &entry
		ms.order = append(ms.order, k)
	}
}

// ---- fileStore ----

// fileStore is a ReasoningStore backed by a memoryStore with JSON file persistence.
type fileStore struct {
	ms      *memoryStore
	path    string
	dirty   bool
	mu      sync.Mutex // protects dirty flag + file I/O
	closeCh chan struct{}
	stopped bool
}

type fileStorePersisted struct {
	Version   int                    `json:"version"`
	UpdatedAt int64                  `json:"updated_at"`
	Entries   map[string]cacheEntry  `json:"entries"`
}

func newFileStore(path string, maxSize int) *fileStore {
	fs := &fileStore{
		ms:      newMemoryStore(maxSize),
		path:    path,
		closeCh: make(chan struct{}),
	}
	fs.loadFromFile()
	go fs.autoSave()
	return fs
}

func (fs *fileStore) Get(key string) *cacheEntry { return fs.ms.Get(key) }

func (fs *fileStore) Set(key string, entry *cacheEntry) {
	fs.ms.Set(key, entry)
	fs.mu.Lock()
	fs.dirty = true
	fs.mu.Unlock()
}

func (fs *fileStore) Delete(key string) {
	fs.ms.Delete(key)
	fs.mu.Lock()
	fs.dirty = true
	fs.mu.Unlock()
}

func (fs *fileStore) Len() int { return fs.ms.Len() }

func (fs *fileStore) Close() error {
	fs.mu.Lock()
	if fs.stopped {
		fs.mu.Unlock()
		return nil
	}
	fs.stopped = true
	close(fs.closeCh)
	fs.mu.Unlock()
	return fs.save()
}

func (fs *fileStore) autoSave() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-fs.closeCh:
			return
		case <-ticker.C:
			fs.mu.Lock()
			dirty := fs.dirty
			fs.dirty = false
			fs.mu.Unlock()
			if dirty {
				if err := fs.save(); err != nil {
					slog.Warn("cache: auto-save failed", "path", fs.path, "err", err)
				}
			}
		}
	}
}

func (fs *fileStore) save() error {
	entries := fs.ms.snapshot()
	p := fileStorePersisted{
		Version:   2,
		UpdatedAt: time.Now().UnixMilli(),
		Entries:   entries,
	}
	data, err := json.Marshal(p)
	if err != nil {
		return err
	}
	dir := filepath.Dir(fs.path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	tmp := fs.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, fs.path)
}

func (fs *fileStore) loadFromFile() {
	data, err := os.ReadFile(fs.path)
	if err != nil {
		return // file not found or unreadable — start fresh
	}
	var p fileStorePersisted
	if err := json.Unmarshal(data, &p); err != nil {
		return
	}
	fs.ms.load(p.Entries)
}
