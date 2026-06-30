# ReasoningCache Pluggable Backend — Implementation Plan

## Context

The `ReasoningCache` currently hard-codes two things:
1. **Storage**: 3 tiers of in-memory maps + FIFO slices
2. **Persistence**: JSON file dump/load + auto-save goroutine

The user wants storage and persistence decoupled so different backends (memory, file, redis, sqlite) can be swapped in. The 3-tier business logic (key derivation, `GetBest` priority, TTL) should stay in `ReasoningCache`.

## Design

Extract a `ReasoningStore` interface per tier. ReasoningCache holds 3 stores (one per tier). Each store handles its own locking and eviction.

```
ReasoningCache (business logic: key derivation, GetBest priority, maxAge TTL)
  ├── toolStore  ReasoningStore
  ├── textStore  ReasoningStore
  └── ctxStore   ReasoningStore
       │
       ├── memoryStore      (default, no persistence)
       └── fileStore        (maps + JSON file auto-save)
```

### Interface

```go
// ReasoningStore is a thread-safe key-value store for one cache tier.
type ReasoningStore interface {
    Get(key string) *cacheEntry   // nil if not found
    Set(key string, entry *cacheEntry) // replaces if exists
    Delete(key string)
    Len() int
    Close() error                 // memoryStore: no-op; fileStore: final save + stop goroutine
}
```

No `error` returns on Get/Set/Delete/Len — memory store is infallible; file store logs and swallows I/O errors on those paths. `Close()` returns error for the same reason (final save can fail).

### Implementations

**`memoryStore`** (~50 lines): `map[string]*cacheEntry` + `[]string` FIFO order + `sync.RWMutex` + `maxSize`. FIFO eviction on Set. This is the default.

**`fileStore`** (~70 lines): same as memoryStore plus JSON dump/load + auto-save goroutine (5s interval). Each tier gets its own JSON file (`<path>.tool.json`, `<path>.text.json`, `<path>.context.json`). Has `Close()` on the concrete type for goroutine shutdown + final save.

### What moves where

| Concern | Before | After |
|---------|--------|-------|
| map + FIFO + maxSize | `ReasoningCache.entries/order` | store implementation |
| Thread safety per tier | `ReasoningCache.mu` (single lock) | store's own `sync.RWMutex` |
| File persistence + auto-save | `ReasoningCache` methods | `fileStore` |
| `ReasoningCache.SetPersistence/SaveNow` | public API | removed |
| `ReasoningCache.Stop()` → `Close()` | public API | fans out to `store.Close()` |
| `putLocked`/`getLocked` | internal helpers | delegate to store + TTL check |

### What stays in ReasoningCache

- 3-tier business logic: `Put`, `Get`, `PutText`, `GetText`, `PutContext`, `GetContext`, `GetBest`, `Delete`
- Key derivation: `cacheKey`, `sha256Key`, `contextKey`
- `maxAge` TTL check (done on Get, before returning)
- `Close() error` — fans out to all 3 stores
- Public method signatures unchanged — zero test changes

## Files to modify

### 1. `convert/reasoning_cache.go` — major refactor

**Add:**
- `ReasoningStore` interface (~6 lines)
- `memoryStore` + `NewMemoryStore(maxSize)` (~50 lines)
- `fileStore` + `NewFileStore(path, maxSize)` (~70 lines): JSON per-tier file, auto-save goroutine, `Close()`
- `NewReasoningCacheWithFile(path, maxSize)` constructor: creates 3 fileStores, loads from disk, starts auto-save

**Remove from ReasoningCache:**
- `mu`, `entries`, `order`, `textEntries`, `textOrder`, `ctxEntries`, `ctxOrder`
- `path`, `dirty`, `closeCh`, `stopped`, `maxSize`, `maxBytes`
- `SetPersistence`, `startAutoSave`, `Stop`, `SaveNow`, `loadFromFile`, `snapshotLocked`, `markDirtyLocked`
- `reasoningCachePersisted` struct
- `putLocked`, `getLocked` → replaced by single-line delegations to store

**Modify:**
- `NewReasoningCache(maxSize)` → creates 3 `memoryStore` instances (signature unchanged)
- `Close()` → fans out to all 3 stores; `Stop()` becomes `Close()` internally
- `Put`/`Get`/`PutText`/`GetText`/`PutContext`/`GetContext` → delegate to `store.Set`/`store.Get`
- `GetBest`/`Delete`/`Len` → delegate to stores

### 2. `cmd/root.go` — replace `--cache-file` with `--cache`

Remove `cacheFile` variable. Add `cacheBackend` string.

```go
rootCmd.PersistentFlags().StringVar(&cacheBackend, "cache", "memory",
    "reasoning cache backend: memory, file:<path>")
```

Options wiring: `Cache: cacheBackend` (was `CacheFile: cacheFile`).

### 3. `rewriter/server.go` — parse `--cache`, dispatch backend

Replace `CacheFile string` with `Cache string` in Options.

```go
func newReasoningCache(cacheSpec string, maxSize int) *convert.ReasoningCache {
    typ, option, _ := strings.Cut(cacheSpec, ":")
    switch strings.ToLower(strings.TrimSpace(typ)) {
    case "memory", "":
        return convert.NewReasoningCache(maxSize)
    case "file":
        if option == "" {
            slog.Warn("cache: file backend requires a path, falling back to memory")
            return convert.NewReasoningCache(maxSize)
        }
        return convert.NewReasoningCacheWithFile(option, maxSize)
    default:
        slog.Warn("cache: unknown backend, falling back to memory", "type", typ)
        return convert.NewReasoningCache(maxSize)
    }
}
```

Replace the current setup:
```go
// Before:
rc := convert.NewReasoningCache(1000)
if opts.CacheFile != "" {
    rc.SetPersistence(opts.CacheFile)
}

// After:
rc := newReasoningCache(opts.Cache, 1000)
```

### 4. `convert/convert_test.go` — zero changes

All tests use `NewReasoningCache(N)` which still works (creates memory stores internally).

## Verification

```bash
cd llm-api-converter
go build ./...
go vet ./...
go test ./... -v -count=1 -race
```

All existing tests must pass unchanged.

## What's skipped

- **Redis/SQLite stores**: Not implemented. The `ReasoningStore` interface makes them trivial: implement 4 methods against a Redis/SQLite client. No ReasoningCache changes needed. The `--cache` flag format extends naturally: `redis://host:port?db=0`, `sqlite:/path/to/cache.db`.
- **`maxBytes` field**: Dead code (never read), deleted.
- **Graceful shutdown wiring**: Adds minimal SIGINT/SIGTERM handler to call `fileStore.Close()` on shutdown.
- **Per-tier config**: All 3 tiers share the same maxSize. If needed later, add parameters to constructors.
