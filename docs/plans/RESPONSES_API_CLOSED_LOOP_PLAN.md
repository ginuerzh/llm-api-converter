# Responses API 闭环实现计划

## Context

在 llm-api-converter 中新增 OpenAI Responses API (`/v1/responses`) 支持，仅做闭环转换：
**Responses API → Chat/Anthropic → Responses API**。与 opencode-cc 架构一致。

## 实现策略：5 阶段

按依赖关系排列，每阶段可独立测试。

### Phase 1: 类型定义 (`convert/responses.go` 新文件)

新增 `ResponsesRequest`, `ResponsesResponse`, `ResponsesOutputItem`, `responsesInputItem` 等类型。

```go
// ResponsesRequest — POST /v1/responses 请求体（Codex 使用）
type ResponsesRequest struct {
    Model             string          `json:"model"`
    Instructions      json.RawMessage `json:"instructions"`
    Input             json.RawMessage `json:"input"`           // string 或 []Item
    MaxOutputTokens   *int            `json:"max_output_tokens,omitempty"`
    Temperature       *float64        `json:"temperature,omitempty"`
    TopP              *float64        `json:"top_p,omitempty"`
    Stream            bool            `json:"stream,omitempty"`
    Tools             []ResponsesTool `json:"tools,omitempty"`
    ToolChoice        any             `json:"tool_choice,omitempty"`
    ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
    // 以下字段接受但忽略（上游不支持）
    PreviousResponseID string `json:"previous_response_id,omitempty"`
    Store              *bool  `json:"store,omitempty"`
    Truncation         string `json:"truncation,omitempty"`
    Background         *bool  `json:"background,omitempty"`
}

// responsesInputItem — input[] 中有多种 Item 类型
type responsesInputItem struct {
    Type      string          `json:"type"` // "message", "function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output", "reasoning"
    Role      string          `json:"role,omitempty"`
    Content   json.RawMessage `json:"content,omitempty"`   // string 或 []contentPart
    ID        string          `json:"id,omitempty"`
    CallID    string          `json:"call_id,omitempty"`
    Name      string          `json:"name,omitempty"`
    Arguments string          `json:"arguments,omitempty"`
    Input     string          `json:"input,omitempty"`      // custom_tool_call
    Output    json.RawMessage `json:"output,omitempty"`
    Summary   []ResponsesReasoningSummary `json:"summary,omitempty"`
}

// ResponsesOutputItem — output[] 中的 Item
type ResponsesOutputItem struct {
    ID        string                      `json:"id"`
    Type      string                      `json:"type"` // "message", "reasoning", "function_call"
    Status    string                      `json:"status,omitempty"`
    Role      string                      `json:"role,omitempty"`
    Content   []ResponsesContentPart      `json:"content,omitempty"`
    Summary   []ResponsesReasoningSummary `json:"summary,omitempty"`
    CallID    string                      `json:"call_id,omitempty"`
    Name      string                      `json:"name,omitempty"`
    Arguments string                      `json:"arguments,omitempty"`
}

type ResponsesContentPart struct {
    Type        string `json:"type"` // "output_text"
    Text        string `json:"text"`
    Annotations []any  `json:"annotations"`
}

type ResponsesReasoningSummary struct {
    Type string `json:"type"` // "summary_text"
    Text string `json:"text"`
}

// ResponsesResponse — 完整响应
type ResponsesResponse struct {
    ID                string                `json:"id"`
    Object            string                `json:"object"` // "response"
    CreatedAt         int64                 `json:"created_at"`
    CompletedAt       *int64                `json:"completed_at"`
    Status            string                `json:"status"` // "completed", "incomplete", "failed", "cancelled"
    Error             any                   `json:"error"`
    IncompleteDetails any                   `json:"incomplete_details"`
    Model             string                `json:"model"`
    Output            []ResponsesOutputItem `json:"output"`
    ParallelToolCalls bool                  `json:"parallel_tool_calls"`
    Reasoning         map[string]any        `json:"reasoning"`
    Store             bool                  `json:"store"`
    Temperature       *float64              `json:"temperature,omitempty"`
    Text              map[string]any        `json:"text"`
    ToolChoice        any                   `json:"tool_choice"`
    Tools             []any                 `json:"tools"`
    TopP              *float64              `json:"top_p,omitempty"`
    Truncation        string                `json:"truncation"`
    Usage             *ResponsesUsage       `json:"usage"`
    Metadata          map[string]any        `json:"metadata"`
}

type ResponsesUsage struct { ... }
```

参考: opencode-cc `responses.go:15-70`, `674-707`. 直接复用其类型定义。

**验证**: `go build ./...`

### Phase 2: 非流式请求转换

在 `convert/responses.go` 中实现两个请求转换函数：

#### 2a. `ConvertResponsesToChat(req *ResponsesRequest, resolveModel func(string) string) (*OpenAIRequest, error)`

```
ResponsesRequest → OpenAIChatRequest 映射:
  .model              → .model (通过 resolveModel)
  .input (string)     → messages[0] = {role:"user", content: string}
  .input ([]item)     → 遍历:
    message(role=developer/system) → messages: {role:"system", content}
    message(role=user)            → messages: {role:"user", content}
    message(role=assistant)       → messages: {role:"assistant", content, reasoning_content}
    function_call                 → messages: {role:"assistant", tool_calls, reasoning_content}
    function_call_output          → messages: {role:"tool", tool_call_id, content}
    custom_tool_call              → tool_use 同样处理 (arguments=wrapped input)
    custom_tool_call_output       → tool 同样处理
    reasoning                     → pendingReasoning (附加到下一 function_call)
  .instructions       → messages 前插入 system message (或合并)
  .max_output_tokens  → .max_tokens
  .tools[{name,parameters}] → .tools[{type:"function", function:{name,parameters}}]
  .tool_choice        → 映射 (required→"required", auto→"auto", {type:"function",name}→{type:"function",function:{name}})
  .stream             → .stream, + stream_options.include_usage
  .temperature, .top_p, .parallel_tool_calls → 直接映射
```

参考: opencode-cc `responses.go:72-124`, `420-632`

#### 2b. `ConvertResponsesToAnthropic(req *ResponsesRequest, ...) (*AnthropicRequest, error)`

```
ResponsesRequest → AnthropicRequest 映射:
  基本同 2a, 差异:
  .instructions       → system blocks (顶层)
  developer/system    → system blocks (顶层)
  message(role=user)  → messages: {role:"user", content}
  function_call       → content block: {type:"tool_use", id, name, input}
  custom_tool_call    → 同上 (input = {"input": item.Input})
  function_call_output → content block: {type:"tool_result", tool_use_id, content}
  reasoning           → content block: {type:"thinking", thinking}
  .max_output_tokens  → .max_tokens (默认 4096)
  .tools[{name,parameters}] → .tools[{name, description, input_schema}]
```

参考: opencode-cc `responses.go:128-382`

**辅助函数** (提取公共逻辑, ~200 行):
- `responsesInputToMessages()` — input JSON → Chat messages
- `responsesInputToAnthropicMessages()` — input JSON → Anthropic messages + system blocks
- `rawText()` — content JSON → 纯文本提取
- `ensureObjectSchema()` — parameters 确保为 object schema
- `canonicalJSONString()` — arguments 规范化

**验证**: 写 P1 测试 `responses_test.go:TestConvertResponsesToChat`, `TestConvertResponsesToAnthropic`

### Phase 3: 非流式响应转换

#### 3a. `ConvertChatToResponses(resp *OpenAIResponse, requestModel string) *ResponsesResponse`

```
OpenAIResponse → ResponsesResponse:
  .choices[0].message.content       → output[].type="message", content=[{type:"output_text", text}]
  .choices[0].message.reasoning_content → output[].type="reasoning", summary=[{type:"summary_text", text}]
  .choices[0].message.tool_calls[]  → output[].type="function_call", call_id, name, arguments
  .choices[0].finish_reason="stop"  → status="completed"
  .choices[0].finish_reason="length" → status="incomplete", incomplete_details={reason:"max_output_tokens"}
  .usage                            → responsesUsage
  .model                            → requestModel (客户端看到的模型名)
```

参考: opencode-cc `responses.go:719-758`

#### 3b. `ConvertAnthropicToResponses(resp *AnthropicResponse, requestModel string) *ResponsesResponse`

```
AnthropicResponse → ResponsesResponse:
  .content[].type="thinking"        → output[].type="reasoning"
  .content[].type="text"            → output[].type="message", content=[{type:"output_text", text}]
  .content[].type="tool_use"        → output[].type="function_call"
  .stop_reason="end_turn"/stop_sequence → status="completed"
  .stop_reason="max_tokens"         → status="incomplete"
  .stop_reason="tool_use"           → status="completed" (工具调用已作为 output items 返回)
  .usage                            → responsesUsage
```

参考: opencode-cc `responses.go:762-816`

**辅助函数** (~80 行):
- `newResponsesResponse()` — 构造 ResponsesResponse 模板
- `completedMessageItem()` / `completedReasoningItem()` — 构造 output items

**验证**: 测试 P2 `TestConvertChatToResponses`, `TestConvertAnthropicToResponses`

### Phase 4: 流式转换 (核心复杂度)

#### 4a. Chat 流式 → Responses SSE (`convert/stream_responses.go`, ~500 行)

`ResponsesStreamConverter` 状态机，将 Chat delta chunks 转为 Responses 类型化 SSE 事件：

```
Chat chunk delta.content    → response.output_text.delta
Chat chunk delta.tool_calls → response.function_call_arguments.delta
Chat chunk delta.reasoning  → response.reasoning_summary_text.delta
Chat chunk finish_reason    → 记录
Chat chunk usage            → 记录
Stream end → response.output_text.done → response.output_item.done → response.completed
```

事件发射顺序 (文本响应):
```
response.created          (首次调用时)
response.in_progress       (首次调用时)
response.output_item.added  (第一个 text delta 触发)
response.content_part.added (第一个 text delta 触发)
response.output_text.delta  (每个 content delta)
response.output_text.done   (Finalize)
response.content_part.done  (Finalize)
response.output_item.done   (Finalize)
response.completed          (Finalize)
```

工具调用响应:
```
response.output_item.added  (function_call item)
response.function_call_arguments.delta (每个参数 delta)
response.function_call_arguments.done (Finalize)
response.output_item.done  (Finalize)
```

推理响应:
```
response.output_item.added  (reasoning item)
response.reasoning_summary_part.added
response.reasoning_summary_text.delta (每个 reasoning delta)
response.reasoning_summary_text.done (Finalize)
response.reasoning_summary_part.done (Finalize)
response.output_item.done  (Finalize)
```

直接复用 opencode-cc 的 `ResponsesStreamConverter` 设计 (`responses.go:917-1358`)。

与 opencode-cc 的唯一差异: opencode-cc 通过 `bufio.Writer` 直接写 HTTP response body, llm-api-converter 需要适配 GOST 的 **3-phase SSE 生命周期** (start / event / end)。需要:

1. `HandleStreamStart(model string)` → 构造 converter, 返回 `response.created` + `response.in_progress` SSE 事件字节
2. `HandleStreamEvent(chunk *OpenAIStreamChunk)` → 返回此 delta 对应的 Responses SSE 事件字节
3. `HandleStreamEnd()` → 调用 Finalize, 返回所有 done 事件

状态通过 `ConvertOptions.SID` 关联的 `sync.Map` 跨 phase 保存。

#### 4b. Anthropic 流式 → Responses SSE (`convert/stream_anthropic.go`, ~400 行)

当上游是 Anthropic 时，GOST 会收到 Anthropic SSE 事件。需要解析这些事件并映射到 Responses SSE：

```
Anthropic SSE event        → Responses SSE event
────────────────────────────────────────────────
message_start              → SetUsage(inputTokens, outputTokens) + SetCachedInputTokens
content_block_start(text)  → 第一个时: response.output_item.added + response.content_part.added
content_block_delta(text_delta) → HandleTextDelta → response.output_text.delta
content_block_start(tool_use) → HandleFunctionCallDelta(init) → response.output_item.added(function_call)
content_block_delta(input_json_delta) → HandleFunctionCallDelta(delta) → response.function_call_arguments.delta
message_delta (usage)      → SetUsage
message_delta (stop_reason)→ SetFinishReason
message_stop               → Finalize → 所有 done 事件 + response.completed
```

参考: opencode-cc `responses_proxy.go:421-513`

在 GOST 的 sniffer_sse.go 中，Anthropic SSE 事件已经被解析为 `event:` + `data:` 分离的形式。llm-api-converter 的 `HandleSSEEvent()` 可以检测到这是 Anthropic 事件（非 Chat delta），然后路由到 Anthropic→Responses 转换路径。

### Phase 5: 集成 + 测试

#### 5a. Session 跟踪机制

通过 `sync.Map` 跟踪哪些 session 是 Responses API 会话（复用现有 SSE session 跟踪模式）：

```go
// 在 convert.go 中新增
var responsesSessions sync.Map // sid → (bool) 标记此 session 为 Responses API

func markResponsesSession(sid string) {
    responsesSessions.Store(sid, true)
}

func isResponsesSession(sid string) bool {
    _, ok := responsesSessions.Load(sid)
    return ok
}
```

**rewriteRequest 时**: 检测到 Responses API 请求 → `markResponsesSession(sid)` → 定向转换
**rewriteResponse 时**: `isResponsesSession(sid)` → 上游响应 → 转换回 Responses 格式

#### 5b. 修改 `convert/convert.go`

在 `Convert()` 中插入 Responses 检测分支（必须在现有 OpenAI/Anthropic 检测之前）：

```go
func Convert(body []byte, opts *ConvertOptions) ([]byte, error) {
    // ... preamble ...

    // NEW: detect Responses API request
    if isResponsesRequest(raw) {
        if opts.SID != "" {
            markResponsesSession(opts.SID)
        }
        return convertResponsesRequest(body, opts)
    }

    // NEW: if this session started as Responses, the response body
    // is an upstream Chat/Anthropic response → convert back to Responses
    if isResponsesSession(opts.SID) {
        return convertToResponsesResponse(body, opts)
    }

    // ... existing detection chain ...
}
```

`convertResponsesRequest()` 根据 model-map 的下游协议选择:
- `protocol == "anthropic"` → `ConvertResponsesToAnthropic()` → Anthropic request JSON
- `protocol == "openai"` 或默认 → `ConvertResponsesToChat()` → Chat request JSON

`convertToResponsesResponse()` 自动检测上游响应格式:
- `choices[]` 存在 → Chat 响应 → `ConvertChatToResponses()`
- `type: "message"` 存在 → Anthropic 响应 → `ConvertAnthropicToResponses()`

#### 5c. SSE 流式路由 (`rewriter/server.go` 修改)

现有 `HandleSSEEvent()` 处理 Chat delta → Anthropic SSE 的流式转换。需要新增两个流式路径:

1. **Chat delta → Responses SSE**: session 标记为 Responses 且 `isOpenAIStreamChunk` true → 路由到 `ResponsesStreamConverter`
2. **Anthropic SSE → Responses SSE**: session 标记为 Responses 且 payload `type: "message"` → 路由到 Anthropic→Responses 流式映射

修改 `rewriter/server.go` 的 SSE dispatch:
```go
case "event":
    if isResponsesSession(sid) {
        if isOpenAIStreamChunk([]byte(evtData)) {
            return h.handleResponsesStreamEvent(sid, evtIndex, evtData, opts)
        }
        if isAnthropicResponse(raw) {
            return h.handleResponsesAnthropicSSE(sid, evtIndex, evtData, opts)
        }
        // passthrough for non-JSON SSE events
    }
    // ... existing dispatch ...
```

不修改 `convert/types.go` — 使用 `sync.Map` session 跟踪替代 ConvertOptions flag，与现有 SSE session 跟踪 (`streamSessions`) 模式一致。

#### 5d. 测试计划

| 文件 | 测试函数 | 行估算 |
|------|---------|--------|
| `convert/responses_test.go` (新) | `TestConvertResponsesToChat` — 基本文本消息, tool calls, images, instructions, developer role | ~300 |
| | `TestConvertResponsesToAnthropic` — 同上 Anthropic 方向 | ~200 |
| | `TestConvertChatToResponses` — 文本, tool calls, reasoning, finish_reason=length | ~150 |
| | `TestConvertAnthropicToResponses` — text, thinking, tool_use, stop_reason mapping | ~150 |
| | `TestInputItemMapping` — 各种 Input Item 类型组合 | ~100 |
| `convert/stream_responses_test.go` (新) | `TestResponsesStreamConverter_Text` — 纯文本流 | ~150 |
| | `TestResponsesStreamConverter_ToolCalls` — 工具调用流 | ~100 |
| | `TestResponsesStreamConverter_Reasoning` — 推理流 | ~100 |
| | `TestResponsesStreamConverter_Combined` — 混合流 | ~50 |
| `convert/stream_anthropic_test.go` (新) | `TestAnthropicToResponsesSSE` — Anthropic SSE 事件序列 | ~150 |
| E2E (现有 `tests/e2e/rewriter_e2e_test.go`) | Responses API 端到端 | ~100 |

**测试总计: ~1,550 行**

---

## 文件清单

| 文件 | 动作 | 预估行数 |
|------|------|---------|
| `convert/responses.go` | 新建 | ~800 |
| `convert/stream_responses.go` | 新建 | ~500 |
| `convert/stream_anthropic.go` | 新建 | ~400 |
| `convert/convert.go` | 修改 (~+60) | 60 |
| `rewriter/server.go` | 修改 (~+30) | 30 |
| `convert/responses_test.go` | 新建 | ~900 |
| `convert/stream_responses_test.go` | 新建 | ~400 |
| `convert/stream_anthropic_test.go` | 新建 | ~150 |
| `tests/e2e/rewriter_e2e_test.go` | 修改 (~+100) | 100 |
| **总计** | | **~3,375** |

---

## 验证方式

每个 phase 完成后运行:
```bash
cd llm-api-converter && go build ./... && go test ./convert/... -v -count=1 -race
```

最终 E2E 验证:
```bash
cd llm-api-converter && go test ./tests/e2e/ -v -timeout 5m
```

运行验证: 用真实 Codex CLI 或 curl 发送 Responses API 请求，通过 GOST + llm-api-converter 代理到下游 API。
