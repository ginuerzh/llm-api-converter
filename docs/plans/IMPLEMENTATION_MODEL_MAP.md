# Model passthrough + mapping table for llm-api-converter

## Context

Currently, llm-api-converter has two model CLI flags:
- `--model` — Anthropic target model (OpenAI Request → Anthropic Request direction)
- `--downstream` — OpenAI target model (Anthropic Request → OpenAI Request direction)

Both hard-overwrite the input model ID. The user wants:
1. **Default: passthrough** — preserve the original model ID from the request
2. **Optional: prefix-based model mapping table** — e.g. `claude-opus-*` → `deepseek-v4-pro`, `claude-sonnet-*` → `deepseek-v4-flash`
3. **Simplify CLI** — remove `--downstream`, keep only `--model` as the universal fallback for both directions

Key constraint: `classifyModel()` currently gates DeepSeek-specific paths (thinking/effort mapping, tool_choice softening) based on `opts.Downstream`. After passthrough/mapping, the behavior classification must use the **actual output model**, not a fixed CLI flag.

## Design

### 1. ModelMap type (`convert/types.go`)

New type for prefix-based mapping rules with optional catch-all:

```go
type ModelMapEntry struct {
    SourcePrefix string // lowercase prefix (e.g. "claude-opus"), or "*" for catch-all
    TargetModel  string // target model (e.g. "deepseek-v4-pro")
}

type ModelMap []ModelMapEntry

// Apply checks sourceModel against all entries; returns target + true on match.
// Catch-all ("*") is checked last after all specific prefixes.
func (mm ModelMap) Apply(sourceModel string) (string, bool)
```

`Apply()` lowercases `sourceModel`, checks `strings.HasPrefix` against each non-catch-all entry in order. Catch-all (`*`) is checked last only if no specific prefix matched.

CLI format:
```
--model-map "claude-opus=deepseek-v4-pro,claude-sonnet=deepseek-v4-flash,*=deepseek-v4-base"
```

### 2. Simplify ConvertOptions (`convert/types.go`)

Remove `Downstream` field. `Model` becomes the universal fallback:

```go
type ConvertOptions struct {
    Model     string   // fallback for both directions, when mapping misses + input empty
    MaxTokens int
    // ... other fields unchanged ...
    ModelMap  ModelMap // empty = passthrough only
}
```

### 3. Modified conversion logic (`convert/convert.go`)

Helper:
```
resolveModel(inputModel, fallback, mapping):
    if inputModel != "" && mapping.Apply(inputModel) matches → return mapped target
    if inputModel != "" → return inputModel  (passthrough)
    return fallback
```

#### In `convertAnthropicRequestToOpenAI()`:

```go
// Before:
profile := classifyModel(opts.Downstream)
oai.Model = opts.Downstream

// After:
outputModel := resolveModel(req.Model, opts.Model, opts.ModelMap)
profile := classifyModel(outputModel)
oai.Model = outputModel
```

#### In `convertOpenAIRequestToAnthropic()`:

```go
// Before:
anthropic.Model = opts.Model

// After:
anthropic.Model = resolveModel(req.Model, opts.Model, opts.ModelMap)
```

#### Model behavior classification

`classifyModel()` runs on the **resolved output model**, so DeepSeek paths activate when mapped to `deepseek-v4-*`.

### 4. CLI changes (`cmd/root.go`)

- **Remove:** `--downstream` flag + `downstream` variable
- **Keep:** `--model` as universal fallback (default changes from `claude-sonnet-4-20250514` to `deepseek-chat`)
- **New:** `--model-map` flag (`prefix1=target1,prefix2=target2,...`)

Parsing: split on `,`, each pair split on first `=`. Empty = no mapping. Malformed entries logged and skipped.

### 5. Default passthrough behavior

When `--model-map` is empty, `resolveModel` returns the input model directly. `--model` is only used when input has no `model` field.

### 6. Options and wiring (`rewriter/server.go`)

```go
type Options struct {
    Model     string
    MaxTokens int
    ModelMap  string  // raw string from --model-map flag
}
```

Add `parseModelMap(modelMapStr) convert.ModelMap` helper. Parse once at startup, pass to `ConvertOptions.ModelMap`.

### 7. Streaming impact

`HandleSSEEvent` start phase calls `NewStreamConverter(opts.Model, ...)` (line 1474). Extract model from the "start" phase `data` payload (the original Anthropic request body) and resolve through `resolveModel` before passing to `NewStreamConverter`.

### 8. Files to modify

| File | Change |
|------|--------|
| `convert/types.go` | Add `ModelMapEntry`, `ModelMap`, `Apply()`. Remove `Downstream` from `ConvertOptions`, add `ModelMap`. |
| `convert/convert.go` | Add `resolveModel()`. Update `convertAnthropicRequestToOpenAI()` and `convertOpenAIRequestToAnthropic()`. Update `HandleSSEEvent` start phase. |
| `cmd/root.go` | Remove `--downstream` flag. Add `--model-map` flag. Change `--model` default to `deepseek-chat`. |
| `rewriter/server.go` | Remove `Downstream` from `Options`. Add `ModelMap string`. Add `parseModelMap()`. Update `ConvertOptions` construction. |

### 9. Verification

1. **Existing tests:** assertions on `oai.Model == "deepseek-chat"` will break — input model now passthrough. Update expected values.
2. **Passthrough tests:** input model preserved when no mapping matches.
3. **Mapping tests:** prefix match, catch-all match, no-match-passthrough.
4. **Tool suite:**
   ```bash
   go test ./... -v -count=1
   CGO_ENABLED=1 go test -race ./...
   ```
