# Plan: Codex Responses API Protocol Conversion Parity

## Context

`llm-api-converter` already supports basic Responses API conversion (Responses ↔ OpenAI Chat / Anthropic Messages), but compared to cc-switch, several Codex-specific input item types, reasoning attachment logic, and content type handling are missing. In practice (e.g. `play/llm-proxy.yaml` already routes `/v1/responses`), these gaps cause:

- `custom_tool_call` reverse conversion produces wrong output item types
- Reasoning blocks fail to attach to the correct prior assistant message
- Billing header leaks degrade prefix cache hit rates
- Streaming usage metadata (cache tokens, reasoning tokens) is lost

## Approach

Fill gaps by priority. Each step is the minimum viable diff, following existing code patterns (function style in [`convert/responses.go`](../../convert/responses.go), inline JSON tests).

---

### Step 1: Add missing content part types — `input_file`, `input_audio`

**File:** `convert/responses.go`

**Change ~20 lines:** Add `input_file` and `input_audio` handling in `responsesContentToOpenAI`, after the `input_image` branch. Also update the type filter list in `responsesRawText`.

Use `map[string]any` passthrough instead of expanding the `OpenAIContentPart` struct:

```go
case "input_file":
    parts = append(parts, map[string]any{
        "type": "input_file",
        "input_file": block["input_file"],
    })
case "input_audio":
    parts = append(parts, map[string]any{
        "type": "input_audio",
        "input_audio": block["input_audio"],
    })
```

**Tests:** `TestConvertResponsesToChat_InputFile` / `TestConvertResponsesToChat_InputAudio`

---

### Step 2: Strip billing header from system prompts

**File:** `convert/responses.go`

**Change ~15 lines:** Add `stripLeadingAnthropicBillingHeader(s string) string`. Call it when extracting `instructions` in `ConvertResponsesToChat` and when extracting system prompt in `ConvertResponsesToAnthropic`. Only strip the **first leading** `x-anthropic-billing-header:` line (preserve subsequent non-leading header text — same behavior as cc-switch).

**Tests:** `TestStripLeadingAnthropicBillingHeader`

---

### Step 3: Improve reasoning attachment logic

**File:** `convert/responses.go`

**Change ~50 lines:** Three improvements in `parseResponsesInput`:

1. **Dedup:** `appendDedupReasoning` — skip when the new reasoning text is already a substring of `pendingReasoning`
2. **Trailing reasoning → prior assistant:** After the item loop, if `pendingReasoning` is non-empty and the last message is an assistant (including tool-calls-only assistant), attach the reasoning to it
3. **Tool-call placeholder:** `backfillToolCallReasoningPlaceholders` — after the loop, set `ReasoningContent` to `"tool call"` on any assistant message that has `tool_calls` but empty reasoning (required by kimi/DeepSeek models)

**Tests:** `TestParseResponsesInput_ReasoningAttachToLastAssistant` / `TestParseResponsesInput_ReasoningDedup` / `TestParseResponsesInput_ReasoningBackfill`

---

### Step 4: Collapse system messages to head

**File:** `convert/responses.go`

**Change ~20 lines:** `collapseSystemMessagesToHead` — merge all `role:"system"` messages into a single first message, joining content with `"\n\n"`. Required for MiniMax compatibility. Call it after building messages in `ConvertResponsesToChat`.

**Tests:** `TestCollapseSystemMessagesToHead`

---

### Step 5: Tool context system (core)

**File:** `convert/responses.go`

**Change ~120 lines:**

Add `codexToolContext` and `codexToolSpec`:

```go
type codexToolSpec struct {
    Kind      string // "function", "namespace", "custom", "tool_search"
    Name      string
    Namespace string
}

type codexToolContext struct {
    byChatName map[string]codexToolSpec // flat chat name -> original spec
}
```

Methods:
- `buildCodexToolContext(tools []ResponsesTool) *codexToolContext` — scan tools: `custom` type recorded to byChatName (embed original definition in description), `namespace` type recursively expand children
- `toResponsesOutputItem(chatName string, tc OpenAIToolCall) ResponsesOutputItem` — reverse mapping: custom → `custom_tool_call`, namespace → `function_call` + `namespace` field, default → `function_call`

Modify `ConvertResponsesToChat`: call `buildCodexToolContext`, store context via `opts` or return it. Custom tool functions get a single `input` string parameter schema.

Modify `ConvertChatToResponses`: accept `*codexToolContext` parameter (via `opts`), use `toResponsesOutputItem` when converting tool calls to output items.

**Note:** If session-scoped context passing is too complex for streaming, skip it there initially — `function_call` items still carry correct `call_id` values so Codex can match them.

**Tests:** `TestBuildCodexToolContext` / `TestConvertResponsesToChat_WithCustomTool` / `TestConvertResponsesToChat_WithNamespace`

---

### Step 6: Improve chat usage → Responses usage mapping

**File:** `convert/responses.go`

**Change ~35 lines:** Replace naive usage mapping in `ConvertChatToResponses` with a `chatUsageToResponsesUsage` helper. Add:
- `prompt_tokens_details.cached_tokens` → `input_tokens_details.cached_tokens`
- `completion_tokens_details` → `output_tokens_details` (default `reasoning_tokens` to 0)
- `cache_read_input_tokens` / `cache_creation_input_tokens` passthrough

Also update `ConvertAnthropicToResponses` usage mapping to add `output_tokens_details.reasoning_tokens`.

**Tests:** Extend `TestConvertChatToResponses_Text` to verify cached_tokens and reasoning_tokens.

---

### Step 7: Chat error → Responses error normalization

**File:** `convert/responses.go`

**Change ~30 lines:** `chatErrorToResponseError` — normalize upstream Chat API errors into Responses API error format `{"error": {"message", "type", "code", "param"}}`. Handle standard OpenAI format, MiniMax `base_resp` format, and bare strings.

Wire into `convertToResponsesResponse`: when upstream body contains an `"error"` key, route to `chatErrorToResponseError`.

**Tests:** `TestChatErrorToResponseError`

---

### Step 8: Streaming tool type support

**File:** `convert/stream_responses.go`

**Change ~50 lines:** Add `toolSpecs map[string]codexToolSpec` field to `ResponsesStreamConverter`. In `HandleStreamEnd`, for each tool call, look up spec by tool name:
- `custom` → emit `custom_tool_call` output item
- `tool_search` → emit `tool_search_call` output item
- `namespace` → emit `function_call` item with `namespace` field
- default → `function_call`

Also add inline `<think>` detection in `HandleChunk`: when `delta.Content` contains a `\<think\>...\</think\>` prefix, split into reasoning + text outputs.

**Tests:** `TestResponsesStreamConverter_CustomToolCall` / `TestResponsesStreamConverter_InlineThink`

---

### Step 9: E2E Responses integration test

**File:** `tests/e2e/rewriter_e2e_test.go`

**Change ~40 lines:** Add `TestRewriterE2E_ResponsesNonStreaming` — simulate a Codex client sending a `/v1/responses` request (with `instructions`, `input` array containing multiple item types, `tools` array with a custom tool), through the GOST plugin converting to Chat format, mock returning a Chat response, and conversion back to Responses format.

---

## Verification

After each step:
```bash
cd /config/workspace/go-gost/llm-api-converter && go test ./convert/... -count=1 -race
```

After all steps:
```bash
cd /config/workspace/go-gost/llm-api-converter && go build ./... && go vet ./... && go test ./... -count=1 -race
```

## Skipped

- **Codex OAuth (ChatGPT Plus/Pro proxy):** `store: false`, `service_tier: "priority"`, `include: ["reasoning.encrypted_content"]` — these are cc-switch desktop-app-proxy-specific constraints, not applicable to the general GOST plugin scenario
- **Dynamic tool loading from tool_search output** (`collect_tool_search_output_tools`): requires cross-turn tracking of tool_search_call → tool_search_output → tools, complex and low-frequency; defer until a real request pattern triggers it
- **cache_control verification:** The Anthropic → OpenAI path already implicitly drops cache_control during conversion; no explicit verification needed
- **Separate codex_tool_context.go file:** All new code goes in `responses.go`; keep the file count minimal
