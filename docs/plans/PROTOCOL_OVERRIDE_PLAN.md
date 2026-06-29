# Plan: Downstream Protocol Type for Model Map Entries

## Context

The `llm-api-converter` currently auto-detects the input format (Anthropic vs OpenAI) and always converts to the opposite format. Users want to specify the downstream protocol per model-map entry — e.g., `protocol=openai` means "the backend speaks OpenAI." If the input already matches that protocol, skip conversion entirely. This enables model routing without protocol conversion (OpenAI→OpenAI, Anthropic→Anthropic) while still supporting cross-protocol use cases.

## Implementation (5 files, ~120 lines)

### 1. `convert/types.go` — Add `Protocol` field + update `Apply()` signature

**`ModelMapEntry` (line 344)**: Add `Protocol string` field.
```go
type ModelMapEntry struct {
    SourcePrefix string
    TargetModel  string
    Protocol     string // "" = auto-detect, "openai", "anthropic"
}
```

**`ModelMap.Apply()` (line 355)**: Change return signature from `(string, bool)` to `(string, string, bool)` — returns `(targetModel, protocol, ok)`. Internally, add `protocol` to the existing return logic alongside `target`.

### 2. `convert/convert.go` — Protocol passthrough logic + `resolveModel()` update

**`resolveModel()` (line 236)**: Change return signature from `string` to `(string, string)` — returns `(targetModel, protocol)`. Update internal `mapping.Apply()` call to capture the new 3-value return.

**New helper `isOpenAIRequest()`**: Detect an OpenAI Chat Completions request body from a raw JSON map. Checks for `model`/`messages` fields while rejecting Anthropic/OpenAI response signatures. Used by the protocol passthrough check.

BEFORE `isOpenAIRequest(raw)` is needed: the current `Convert()` dispatch detects OpenAI requests as a fallthrough at line 301 (`hasStringField(raw, "model") || hasArrayField(raw, "messages")`). We need a standalone function for the protocol check.

**`Convert()` (line 259)**: After parsing JSON into `raw` (line 277) and BEFORE the dispatch chain (line 282), insert the protocol passthrough check:

1. Extract `model` field from `raw`, resolve protocol via `resolveModel()`
2. If protocol is set AND the detected format matches it → log + return `body, nil` (passthrough)
3. Four checks: `isAnthropicRequest && proto=="anthropic"`, `isOpenAIRequest && proto=="openai"`, `isOpenAIResponse && proto=="openai"`, `isAnthropicResponse && proto=="anthropic"`
4. If protocol is `""` (unset) → all four conditions are false → existing dispatch runs unchanged

**`ConvertSSE()` (line 129)**: Same pattern — parse the SSE event's data payload as JSON, resolve protocol, and if the payload format matches the protocol → return the event unchanged.

**`HandleSSEEvent()` (line 1553)**: During stream start phase, store resolved protocol on the stream state. During event phase, if the payload format matches the stored protocol → passthrough unchanged.

**Callers of `resolveModel()`**: Three call sites (lines 849, 1164, 1558). Update each to capture the `(string, string)` return — assign the protocol to a local or ignore with `_, _`. The conversion functions themselves don't use the protocol (it's only used in `Convert()`/`ConvertSSE()`/`HandleSSEEvent()` for passthrough).

### 3. `convert/stream.go` — Store protocol on StreamConverter

**`StreamConverter` struct (line 15)**: Add `downstreamProtocol string` field.

### 4. `rewriter/server.go` — Extended CLI format

**`parseModelMap()` (line 158)**: Support new format `prefix=target:protocol`.
```go
target, protocol, _ := strings.Cut(rest, ":")
```
- `"claude-opus=deepseek-chat:openai"` → Protocol: "openai"
- `"claude-opus=deepseek-chat"` → Protocol: "" (backward compatible, unset)
- Validate protocol values: only `""`, `"openai"`, `"anthropic"` accepted; others logged + reset to `""`

### 5. `cmd/root.go` — Updated help text

Update `--model-map` flag description: `"model mapping: prefix=target[:protocol],... (* for catch-all, protocol: openai|anthropic)"`

## Tests (~10 new tests)

All in `convert/convert_test.go`:

| Test | Scenario |
|------|----------|
| `TestConvert_ProtocolOpenAI_OpenAIReqPassthrough` | OpenAI req + proto=openai → unchanged |
| `TestConvert_ProtocolOpenAI_AnthropicReqConverted` | Anthropic req + proto=openai → still converts to OpenAI |
| `TestConvert_ProtocolAnthropic_AnthropicReqPassthrough` | Anthropic req + proto=anthropic → unchanged |
| `TestConvert_ProtocolAnthropic_OpenAIReqConverted` | OpenAI req + proto=anthropic → converts to Anthropic |
| `TestConvert_ProtocolOpenAI_OpenAIRespPassthrough` | OpenAI response + proto=openai → unchanged |
| `TestConvert_ProtocolAnthropic_AnthropicRespPassthrough` | Anthropic response + proto=anthropic → unchanged |
| `TestConvert_ProtocolUnset_CurrentBehavior` | Proto="" behaves identically to current |
| `TestConvert_ProtocolCatchAll` | `*` catch-all with protocol |
| `TestModelMapApply_WithProtocol` | Apply() returns correct (target, protocol, ok) |
| `TestParseModelMap_WithProtocol` | CLI string `a=b:openai,c=d` parses correctly |

## Verification

```bash
cd llm-api-converter && go build ./... && go vet ./... && go test ./... -v -count=1 -race
```

Expected: ~165 tests pass (existing 155 + ~10 new), no vet warnings, clean build.
