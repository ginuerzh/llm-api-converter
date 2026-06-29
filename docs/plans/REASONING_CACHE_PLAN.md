# DeepSeek V4 reasoning_content Cache/Replay for llm-api-converter

## 问题

DeepSeek V4 要求：如果响应包含 `reasoning_content` + `tool_calls`，后续请求（Claude Code 对话压缩后）包含相同 assistant 消息时，**必须**带上原始的 `reasoning_content`，否则报错 `"reasoning_content must be passed back"`。

`llm-api-converter` 当前是无状态的，只转换每个 HTTP 请求/响应中存在的内容。当 Claude Code 压缩对话（丢弃 `thinking` 块但保留 `tool_use` 块），后续 Anthropic 请求中 assistant 消息缺少 `thinking`，转换为 OpenAI 格式时也缺失 `reasoning_content`。

## 方案

引入一个线程安全的 `ReasoningCache`，以**工具调用 ID 的排序哈希**为键，按以下流程工作：

```
[响应路径] OpenAI 响应 (with reasoning_content + tool_calls) 
    → convertOpenAIResponseToAnthropic / StreamConverter
    → 缓存 reasoning_content (keyed by tool call ID hash)
    → Anthropic 响应 (with thinking + tool_use blocks) → to Claude Code

[请求路径] Anthropic 请求 (tool_use blocks but NO thinking 块)
    → convertAnthropicAssistantMessage
    → 查缓存 → 注入 reasoning_content
    → OpenAI 请求 (with reasoning_content + tool_calls) → to DeepSeek ✅
```

## 改动文件

### 1. 新建 `convert/reasoning_cache.go`

```go
type ReasoningCache struct {
    mu      sync.RWMutex
    entries map[string]*cacheEntry  // key = sha256(sorted tool IDs)
    order   []string                // FIFO eviction
    maxSize int
}
type cacheEntry struct {
    Reasoning string
    AddedAt   time.Time
}
func NewReasoningCache(maxSize int) *ReasoningCache
func (c *ReasoningCache) Put(toolIDs []string, reasoning string)
func (c *ReasoningCache) Get(toolIDs []string) (string, bool)
func (c *ReasoningCache) Delete(toolIDs []string)
func (c *ReasoningCache) Len() int
```

- 键 = SHA256(排序后的 tool call ID 拼接)，保证确定性
- 默认 maxSize = 1000，FIFO 淘汰
- 所有方法线程安全

### 2. 修改 `convert/types.go`

`ConvertOptions` 增加字段：

```go
type ConvertOptions struct {
    // ... 现有字段 ...
    ReasoningCache *ReasoningCache  // 可选，nil 时不启用缓存
}
```

`NewStreamConverter` 签名改为接受 `*ConvertOptions`（或增加 `reasoningCache` 参数）。

### 3. 修改 `convert/convert.go` —— 2 处改动

**a) `convertAnthropicAssistantMessage()` (line 946)** —— 注入

在函数末尾，set `oaiMsg.ReasoningContent` 之前，增加：

```go
// 如果 reasoning 为空但有 tool_calls (Claude Code 可能压缩了 thinking 块)，
// 尝试从缓存中恢复 reasoning_content。
if reasoning == "" && len(toolCalls) > 0 && opts != nil && opts.ReasoningCache != nil {
    var ids []string
    for _, tc := range toolCalls {
        ids = append(ids, tc.ID)
    }
    if cached, ok := opts.ReasoningCache.Get(ids); ok {
        reasoning = cached
    }
}
```

**b) `convertOpenAIResponseToAnthropic()` (line 1184)** —— 存储

在 `convertOpenAIMessageToContent(msg)` 之后，检查结果中是否有 `thinking` + `tool_use` 块，有则缓存：

```go
if opts != nil && opts.ReasoningCache != nil {
    if reasoning != "" && len(toolIDs) > 0 {
        opts.ReasoningCache.Put(toolIDs, reasoning)
    }
}
```

需要从 `resp.Choices[0].Message` 中提取 `ReasoningContent` + `ToolCalls` ID（non-streaming 响应路径数据齐全）。

### 4. 修改 `convert/stream.go` —— StreamConverter 缓存

- 增加字段 `accumulatedReasoning string`，在 `HandleChunk` 中 `delta.ReasoningContent` 非空时累加
- `HandleStreamEnd()` 时，若 `accumulatedReasoning != ""` 且有 `toolCallByIndex` entries，则从中提取 tool call IDs 并调用 `reasoningCache.Put(ids, accumulatedReasoning)`
- `NewStreamConverter` 接受 `*ReasoningCache` 参数（或通过 `*ConvertOptions` 传入）

### 5. 修改 `rewriter/server.go` —— 连接服务器级别的缓存

```go
type rewriteHandler struct {
    opts           *Options
    reasoningCache *convert.ReasoningCache  // 所有请求共享
}

func newServer(opts *Options) http.Handler {
    rc := convert.NewReasoningCache(1000)
    mux.Handle("/rewrite", &rewriteHandler{opts: opts, reasoningCache: rc})
}
```

在 `ServeHTTP` 中构造 `ConvertOptions` 时设置 `ReasoningCache`：

```go
opts := &convert.ConvertOptions{
    Model:          h.opts.Model,
    MaxTokens:      h.opts.MaxTokens,
    Downstream:     h.opts.Downstream,
    ReasoningCache: h.reasoningCache,
}
```

### 6. 新增测试

`convert/convert_test.go`:
- `TestConvert_ReasoningCache_StoreInject`：构造 OpenAI 响应（reasoning_content + tool_calls）→ 转换为 Anthropic → 验证缓存命中；再构造 Anthropic 请求（只有 tool_use，无 thinking）→ 转换为 OpenAI → 验证 reasoning_content 被注入
- `TestConvert_ReasoningCache_Miss`：相同流程但确保`Get`未命中时无注入
- `TestConvert_ReasoningCache_Eviction`：注入超过 maxSize 条目，验证旧条目被淘汰

`convert/stream_test.go`:
- `TestStream_ReasoningCacheOnEnd`：构造一系列流 chunk（reasoning_content deltas + tool_calls deltas）→ `HandleChunk` → `HandleStreamEnd` → 验证缓存中存在对应的 reasoning_content

## 影响范围

| 操作 | 文件 | 改动量 |
|------|------|--------|
| 新建 | `llm-api-converter/convert/reasoning_cache.go` | ~80 行 |
| 修改 | `llm-api-converter/convert/types.go` | +2 行（ConvertOptions 加字段） |
| 修改 | `llm-api-converter/convert/convert.go` | ~20 行（2 处逻辑改动） |
| 修改 | `llm-api-converter/convert/stream.go` | ~15 行（累加 + HandleStreamEnd 缓存） |
| 修改 | `llm-api-converter/rewriter/server.go` | ~5 行（创建缓存 + 传入 opts） |
| 增改 | `llm-api-converter/convert/convert_test.go` | ~80 行 |
| 增改 | `llm-api-converter/convert/stream_test.go` | ~60 行 |

## 验证

```bash
cd llm-api-converter
go build ./... && go vet ./...
CGO_ENABLED=1 go test -race ./convert/... -v -run 'TestConvert_ReasoningCache|TestStream_ReasoningCache'
CGO_ENABLED=1 go test -race ./...
```
