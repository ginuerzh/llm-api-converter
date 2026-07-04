# Plan: Add Google Gemini generateContent Protocol Conversion

## Context

The `llm-api-converter` currently converts between Anthropic Messages and OpenAI Chat/Responses APIs. This plan adds **Google Gemini `generateContent` API** as a new target protocol, enabling bidirectional conversion:

- **Anthropic Messages ↔ Gemini generateContent**
- **OpenAI Chat ↔ Gemini generateContent**

The `new-api` project (local clone at `new-api/`) provides a proven reference implementation for these exact mappings.

## What Exists

| File | Role |
|------|------|
| `convert/types.go` | All data types, `ConvertOptions` |
| `convert/protocol.go` | `Protocol` enum (3 values), `parseProtocol()`, `uriTable` |
| `convert/detect.go` | Body-primary protocol detection (~22 rules) |
| `convert/registry.go` | `ConversionKey` → converter function dispatch map |
| `convert/convert.go` | `Convert()` + `ConvertSSE()` + `HandleSSEEvent()` routing |
| `convert/anthropic_to_openai.go` | Anthropic ↔ OpenAI (request + response) |
| `convert/openai_to_anthropic.go` | OpenAI ↔ Anthropic (request + response) |
| `convert/stream.go` | `StreamConverter` (OpenAI delta → Anthropic SSE) |
| `convert/stream_anthropic.go` | `responsesStreamHandler` interface, `AnthropicStreamConverter` |
| `convert/session.go` | `SessionStore`, `PassthroughStreamHandler` |
| `rewriter/server.go` | HTTP `/rewrite` endpoint, model-map parsing |

## Design Decisions

1. **Two conversion files** (not 4): `openai_to_gemini.go` and `anthropic_to_gemini.go`. Each handles both request and response directions for one protocol pair — matching the existing pattern.

2. **Gemini streaming** handled inline in each file via a Gemini-specific `responsesStreamHandler`, not a separate `stream_gemini.go`. This keeps related streaming logic with its protocol converter.

3. **Anthropic↔Gemini is direct** (not composed through OpenAI). While new-api chains Claude→OpenAI→Gemini, that requires two API hops. The llm-api-converter operates as an inline body rewriter, so direct conversion avoids fidelity loss and is simpler to reason about.

4. **Types in dedicated file** `convert/gemini_types.go` — matches the pattern of `types.go` containing protocol-specific structs.

5. **Gemini streaming uses SSE** (`alt=sse` query parameter on `:streamGenerateContent`). The Gemini SSE response is a sequence of `GeminiChatResponse` JSON objects, each containing incremental `candidates[].content.parts[]`.

## Files to Create

### `convert/gemini_types.go`
Gemini protocol types (adapted from `new-api/dto/gemini.go`):
- `GeminiChatRequest` — `contents[]`, `systemInstruction`, `tools[]`, `toolConfig`, `safetySettings`, `generationConfig`
- `GeminiChatContent` — `role` ("user"/"model"/"function"), `parts[]`
- `GeminiPart` — `text`, `inlineData` (base64 media), `functionCall`, `functionResponse`, `thought`, `executableCode`, `codeExecutionResult`, `fileData`
- `GeminiChatGenerationConfig` — `temperature`, `topP`, `topK`, `maxOutputTokens`, `stopSequences`, `responseMimeType`, `responseSchema`, `thinkingConfig`, `seed`
- `GeminiChatResponse` — `candidates[]`, `promptFeedback`, `usageMetadata`
- `GeminiChatCandidate` — `content`, `finishReason`, `index`, `safetyRatings`
- `GeminiUsageMetadata` — token counts with modality breakdown
- `GeminiThinkingConfig` — `includeThoughts`, `thinkingBudget`, `thinkingLevel`
- Helper types: `FunctionCall`, `GeminiFunctionResponse`, `GeminiInlineData`, `GeminiFileData`, `GeminiChatTool`, `ToolConfig`, `FunctionCallingConfig`

### `convert/openai_to_gemini.go`
Core conversion functions (reference: `new-api/relay/channel/gemini/relay-gemini.go`):

**`convertOpenAIRequestToGemini(body, opts)`**
- System/developer messages → `systemInstruction.parts[].text`
- Messages: role "assistant" → "model", "tool"/"function" → functionResponse parts
- Content: text parts, image_url → `inlineData` (base64+mimeType)
- Tool calls → `functionCall` parts (with `name`+`args`)
- Tools → `functionDeclarations` in `GeminiChatTool`
- Special tools: `googleSearch`, `codeExecution` → corresponding GeminiTool fields
- `tool_choice`: "auto"→AUTO, "none"→NONE, "required"→ANY, specific function→ANY+allowedFunctionNames
- Generation config: `temperature`, `top_p`→`topP`, `max_tokens`→`maxOutputTokens`, `stop`→`stopSequences`, `seed`
- `response_format` (json_schema/json_object) → `responseMimeType`+"application/json"+`responseSchema`
- `stream_options` → sets `alt=sse` streaming flag via opts
- Safety settings: all categories set to `BLOCK_NONE` (least restrictive, matching new-api)
- Thinking: when model has `-thinking` suffix or `reasoning_effort` is set → `thinkingConfig`

**`convertGeminiResponseToOpenAI(body, opts)`**
- `candidates[]` → `choices[]`
- Parts mapping:
  - `text` → message content (plain text or reasoning_content if `thought:true`)
  - `functionCall` → `tool_calls[]` with generated `call_<uuid>` ID
  - `inlineData` (image) → `![image](data:mime;base64,...)` markdown in content
  - `executableCode` → ` ```lang\ncode\n``` ` in content
  - `codeExecutionResult` → ` ```output\nresult\n``` ` in content
- `finishReason` mapping: STOP→stop, MAX_TOKENS→length, SAFETY/RECITATION/BLOCKLIST/PROHIBITED_CONTENT/SPII/OTHER→content_filter
- `usageMetadata` → `usage` (promptTokens, completionTokens, totalTokens with modality breakdown)
- Edge case: empty candidates + promptFeedback.blockReason → error response

**`convertGeminiSSEToOpenAI(body, opts)`**
- Each SSE `data:` line is a `GeminiChatResponse` JSON
- Accumulates incremental parts and emits OpenAI streaming chunks
- Same part mappings as non-streaming but as deltas
- Tool call index tracking across chunks

### `convert/anthropic_to_gemini.go`
Core conversion functions:

**`convertAnthropicRequestToGemini(body, opts)`**
- `system` (string or array) → `systemInstruction.parts[].text`
- Messages: role "assistant" → "model"
- Content blocks → parts:
  - `text` → `text` part
  - `image`/`image_url` → `inlineData` part
  - `tool_use` → `functionCall` part
  - `tool_result` → `functionResponse` part (wrapped in user content)
  - `thinking` → text part with `thought:true`
- Tools: `input_schema` → `functionDeclarations`
- `tool_choice`: type:auto→AUTO, type:any→ANY, type:tool→ANY+allowedFunctionNames
- Generation config: `temperature`, `top_p`→`topP`, `top_k`→`topK`, `max_tokens`→`maxOutputTokens`, `stop_sequences`→`stopSequences`
- Thinking: `thinking.budget_tokens` → `thinkingConfig.thinkingBudget`, `thinking.type:"enabled"` → `includeThoughts:true`
- Safety settings: all `BLOCK_NONE`

**`convertGeminiResponseToAnthropic(body, opts)`**
- `candidates[]` → single Anthropic response
- Parts → content blocks:
  - `text` (non-thought) → `text` block
  - `text` (thought:true) → `thinking` block
  - `functionCall` → `tool_use` block (with generated ID)
  - `inlineData` (image) → `image` block with base64 data
- `finishReason`: STOP→end_turn, MAX_TOKENS→max_tokens, tool calls→tool_use, safety→error
- `usageMetadata` → `usage` (input_tokens, output_tokens)
- Empty candidates + block → Anthropic error shape

**`convertGeminiSSEToAnthropic(body, opts)`**
- Accumulates incremental parts into Anthropic SSE events
- Uses same state machine as `StreamConverter` (`message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` → `message_stop`)

## Files to Modify

### `convert/protocol.go`
- Add `ProtocolGemini Protocol = 4`
- Update `String()`: add `case ProtocolGemini: return "gemini"`
- Update `parseProtocol()`: add `"gemini"` → `ProtocolGemini`
- Add URI detection: `/v1beta/models/` or `generateContent` or `streamGenerateContent` → `ProtocolGemini`

### `convert/detect.go`
Add detection rules for Gemini request/response:
- **Response**: `candidates` field (with `content.parts` structure)
- **Request**: `contents` array (with `parts[]` containing `text`/`inlineData`/`functionCall`), `generationConfig`, `safetySettings`, `systemInstruction`

### `convert/registry.go`
Register new conversion pairs in `init()`:
```go
conversions[ConversionKey{ProtocolOpenAIChat, ProtocolGemini}] = convertOpenAIBodyToGemini
conversions[ConversionKey{ProtocolGemini, ProtocolOpenAIChat}] = convertGeminiBodyToOpenAI
conversions[ConversionKey{ProtocolAnthropic, ProtocolGemini}] = convertAnthropicBodyToGemini
conversions[ConversionKey{ProtocolGemini, ProtocolAnthropic}] = convertGeminiBodyToAnthropic
```

### `convert/convert.go`
- In `Convert()`: wire Gemini routing alongside existing Anthropic/OpenAI/Responses paths
- In `HandleSSEEvent()`: create `GeminiToOpenAIStreamConverter` or `GeminiToAnthropicStreamConverter` for Gemini SSE streams

### `rewriter/server.go`
- In `parseModelMap()`: allow `"gemini"` protocol suffix on model-map entries
- Update `--model-map` help text

### `cmd/root.go`
- Update `--model-map` flag help text: add `gemini` to protocol list

## Conversion Mapping Reference

### Role Mapping
| Anthropic | OpenAI | Gemini |
|-----------|--------|--------|
| `user` | `user` | `user` |
| `assistant` | `assistant` | `model` |
| — | `system`/`developer` | `systemInstruction` (top-level) |
| — | `tool` | `user` (containing functionResponse) |

### Content/Part Mapping
| Anthropic | OpenAI | Gemini |
|-----------|--------|--------|
| `text` block | `text` content | `text` part |
| `image`/`image_url` block | `image_url` content | `inlineData` part |
| `tool_use` block | `tool_calls[]` | `functionCall` part |
| `tool_result` block | `tool` role message | `functionResponse` part |
| `thinking` block | `reasoning_content` | `text` part with `thought:true` |

### Generation Config
| Anthropic | OpenAI | Gemini |
|-----------|--------|--------|
| `temperature` | `temperature` | `temperature` |
| `top_p` | `top_p` | `topP` |
| `top_k` | — | `topK` |
| `max_tokens` | `max_tokens`/`max_completion_tokens` | `maxOutputTokens` |
| `stop_sequences` | `stop` | `stopSequences` |
| — | `seed` | `seed` |

### Finish Reason (Response)
| Gemini | OpenAI | Anthropic |
|--------|--------|-----------|
| `STOP` | `stop` | `end_turn` |
| `MAX_TOKENS` | `length` | `max_tokens` |
| `SAFETY`/`RECITATION`/etc. | `content_filter` | error response |
| (tool calls present) | `tool_calls` | `tool_use` |

## Verification

1. **Build**: `cd llm-api-converter && go build ./...`
2. **Vet**: `cd llm-api-converter && go vet ./...`
3. **Unit tests**: Write table-driven tests for each conversion direction:
   - OpenAI Chat request → Gemini request (verify system→systemInstruction, role mapping, tool conversion, generation config)
   - Gemini response → OpenAI Chat response (verify parts→choices, finishReason mapping, usage mapping)
   - Anthropic request → Gemini request (verify system, content blocks→parts, tool mapping)
   - Gemini response → Anthropic response (verify parts→content blocks, stop_reason mapping)
4. **E2E test**: Run `go test ./tests/e2e/ -v -run TestRewriter` to verify the rewriter plugin still works
5. **Manual test**: Start gost with llm-api-converter plugin, configure model-map entry like `gemini-2.5-flash=gemini-2.5-flash:gemini`, send an OpenAI-format request and verify it reaches Gemini API correctly
