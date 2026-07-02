# Protocol Conversion Matrix Redesign

## Context

3 client protocols × 2 downstream protocols. Three root problems:
- ~200 lines of fragile heuristic detection (negative exclusion, hardcoded model prefixes, ordering constraints)
- Passthrough vs conversion decision is implicit and scattered across 3 sites
- Global mutable state with no cleanup

GOST sniffer will provide `direction` + `uri` metadata — detection becomes a lookup table. No fallback compatibility needed.

## Model-map protocol: passthrough by default

| Entry | Behavior |
|-------|----------|
| `prefix=target` | Passthrough: rewrite model name, no conversion |
| `prefix=target:openai` | Convert to OpenAI Chat format downstream |
| `prefix=target:anthropic` | Convert to Anthropic format downstream |

Unset protocol = `input protocol` (same as detected) → registry finds `from == to` → model rewrite only. This eliminates `ClientProtocol` tracking, asymmetric passthrough guards, and `LookupTarget` reverse lookup for passthrough decisions.

## Design

### 1. `protocol.go` (~30L)

```go
type Protocol int
const (
    ProtocolUnknown    Protocol = iota
    ProtocolAnthropic
    ProtocolOpenAIChat
    ProtocolOpenAIResponses
)

type Direction int
const ( DirectionRequest Direction = iota; DirectionResponse )

var uriTable = map[string]struct{ Req, Resp Protocol }{
    "/v1/messages":          {ProtocolAnthropic, ProtocolAnthropic},
    "/v1/chat/completions":  {ProtocolOpenAIChat, ProtocolOpenAIChat},
    "/v1/responses":         {ProtocolOpenAIResponses, ProtocolOpenAIResponses},
}

func detectProtocol(uri, direction string) (Protocol, Direction, bool) { ... }
```

### 2. `registry.go` (~30L)

```go
type ConversionKey struct{ From, To Protocol }

var conversions = map[ConversionKey]func([]byte, *ConvertOptions) ([]byte, error){...}
var streamFactories = map[ConversionKey]func(string, *ConvertOptions) StreamConverter{...}
```

Passthrough = `from == to` → model rewrite, no converter lookup needed.

### 3. `session.go` (~70L)

```go
type SessionStore struct {
    mu       sync.RWMutex
    sessions map[string]*Session
}
type Session struct {
    ID              string
    Protocol        Protocol
    StreamConverter StreamConverter
}
```

Replaces 3 `sync.Map` globals. Server owns it, passes via `ConvertOptions.SessionStore`.

### 4. `PassthroughStreamConverter`

`StreamConverter` impl: identity passthrough + model rewrite on `message_start` (Anthropic) / chunk `model` (OpenAI).

### 5. Slim `Convert()` (~40L)

```go
func Convert(body []byte, opts *ConvertOptions) ([]byte, error) {
    // 1. Empty/non-JSON/SSE → unchanged
    // 2. from, _, _ := detectProtocol(opts.URI, opts.Direction)
    // 3. to := resolveDownstream(from, model, opts.ModelMap)
    // 4. if from == to → rewrite model, return
    // 5. conversions[{from, to}](body, opts)
}
```

### 6. `resolveModel` simplified

```go
// Returns (targetModel, downstreamProtocol).
// Unset protocol → returns inputProtocol (triggers passthrough).
func resolveModel(inputModel string, inputProtocol Protocol, mapping ModelMap) (targetModel string, downstreamProtocol Protocol) {
    if entry, ok := mapping.Apply(inputModel); ok {
        targetModel = entry.TargetModel
        if entry.Protocol == "" {
            downstreamProtocol = inputProtocol  // passthrough
        } else {
            downstreamProtocol = parseProtocol(entry.Protocol)
        }
        return
    }
    return inputModel, inputProtocol  // no mapping → passthrough
}
```

## Deletions (~450 lines)

| Code | Lines |
|------|-------|
| `isAnthropicRequest`, `hasOpenAIStyleModel`, `isOpenAIRequest`, `isOpenAIResponse`, `isAnthropicResponse`, `isResponsesRequest`, `isResponsesResponse` | ~200 |
| Passthrough blocks in `Convert()`, `ConvertSSE()`, `HandleSSEEvent()` | ~150 |
| `ClientProtocol` field + tracking logic | ~30 |
| Asymmetric guard (`catchAllTarget`, `LookupTarget` passthrough logic) | ~20 |
| `streamStates`, `responsesSessions`, `responsesStreamStates`, mark/unmark helpers | ~30 |
| Duplicate `else if` bug | ~10 |
| ModelMap `catchAllTarget()`, `LookupTarget()` (no longer needed for passthrough) | ~10 |

## Files

| File | Change |
|------|--------|
| `protocol.go` (~30L) | **New** |
| `registry.go` (~30L) | **New** |
| `session.go` (~70L) | **New** |
| `convert.go` (~100L) | Rewritten (was ~1100) |
| `stream.go` | Minus passthrough methods |
| `types.go` | +`Protocol`/`Direction`/`URI`/`SessionStore` on `ConvertOptions`; -`ClientProtocol`, -`StreamErrorMsg`; simplify `ModelMap` |
| `rewriter/server.go` | +SessionStore, +metadata parse, -`requestModels`/`requestProtocols` maps, -`ClientProtocol` tracking |

Unchanged: conversion functions, `ReasoningCache`, `responsesStreamHandler`, file layout of `convert_*.go` / `stream_*.go`.

## Phases

### 1: Protocol + detection
- `protocol.go`, parse `direction`/`uri` metadata, `detectProtocol()`, delete old detectors
- **Verify**: tests pass (URI injected in test opts)

### 2: Model-map → passthrough by default
- `resolveModel`: unset protocol → `inputProtocol`
- Simply `ModelMap`: remove `catchAllTarget()`, trim `LookupTarget` to only `SourcePrefix` reverse lookup (for response model name recovery)
- Delete `ClientProtocol` field and tracking
- **Verify**: fuzz tests pass

### 3: Registry
- `registry.go`, register conversion pairs, `Convert()` → detect → resolve → lookup → call
- **Verify**: tests pass

### 4: Session store + passthrough unification
- `session.go`, `PassthroughStreamConverter`, migrate `HandleSSEEvent`, delete globals, delete scattered passthrough blocks
- **Verify**: E2E streaming tests pass

### 5: Cleanup
- Fix duplicate `else if` bug, split `convert.go`
- **Verify**: `go build ./... && go vet ./... && go test -race ./... -count=1`

## Verification

```bash
cd llm-api-converter && CGO_ENABLED=1 go test -race ./... -count=1
cd llm-api-converter && go test ./convert/ -fuzz=FuzzConvert_Matrix -fuzztime=10s
cd llm-api-converter && go test ./tests/e2e/ -v -timeout 5m
```
