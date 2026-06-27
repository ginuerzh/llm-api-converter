# llm-api-converter — GOST Rewriter HTTP 插件设计

## Context

OpenCode go 的 DeepSeek API 仅支持 OpenAI format，若要接入 Claude Code (Anthropic API)，需要一个协议转换层。本项目遵循 `gost-plugins` 的模式，实现一个 GOST Rewriter HTTP 插件。

核心能力：GOST HTTP forward handler 拦截请求和响应 body，调用 rewriter 插件做双向协议转换：

- **Request 方向**: OpenAI Chat Completions 请求体 → Anthropic Messages 请求体
- **Response 方向**: Anthropic Messages 响应体 → OpenAI Chat Completions 响应体

Reference: [deepseek-v4-opencode-claude-code-bridge](https://github.com/superheroYu/deepseek-v4-opencode-claude-code-bridge), [musistudio/llms](https://github.com/musistudio/llms)

## Phase 1 范围

双向转换：

1. **OpenAI Request → Anthropic Request** (请求 body 转换)
2. **Anthropic Response → OpenAI Response** (响应 body 转换)

### 数据流

```
opencode go (OpenAI SDK)
    ↓ POST api/chat/completions  (OpenAI 格式请求体)
GOST HTTP forward handler
    ↓ rewriter plugin 拦截请求 body → POST /rewrite
    │
    ├── llm-api-converter 检测: 含 model/messages → OpenAI Request
    │   → 转换为 Anthropic Request body
    │   → {"ok":true, "data":"<Anthropic 请求体>"}
    │
    ↓ GOST 用转换后的 body 转发
Anthropic API
    ↓ (Anthropic 格式响应体)
GOST HTTP forward handler
    ↓ rewriter plugin 拦截响应 body → POST /rewrite
    │
    ├── llm-api-converter 检测: 含 type:"message"/stop_reason → Anthropic Response
    │   → 转换为 OpenAI Response body
    │   → {"ok":true, "data":"<OpenAI 响应体>"}
    │
    ↓ GOST 用转换后的 body 返回给客户端
opencode go (OpenAI SDK) 收到 OpenAI 格式响应
```

### 自动检测规则

解析 JSON body，三种情况：

| 检测依据 | 判定 | 处理 |
|---|---|---|
| 含 `model` (string) 或 `messages` (array) | OpenAI Request | → Anthropic Request |
| 含 `type:"message"` 且 `stop_reason`/`usage` | Anthropic Response | → OpenAI Response |
| 都不匹配 或 JSON 解析失败 | 未知格式 | 直接透传 |

### 类型定义

#### OpenAI Request (Chat Completions)

```go
type OpenAIChatRequest struct {
    Model            string            `json:"model"`
    Messages         []OpenAIMessage   `json:"messages"`
    MaxTokens        *int              `json:"max_tokens,omitempty"`
    Temperature      *float64          `json:"temperature,omitempty"`
    TopP             *float64          `json:"top_p,omitempty"`
    Stop             any               `json:"stop,omitempty"`
    Stream           *bool             `json:"stream,omitempty"`
    Tools            []OpenAITool      `json:"tools,omitempty"`
    TopK             *int              `json:"top_k,omitempty"`
    Metadata         map[string]any    `json:"metadata,omitempty"`
    // 忽略: tool_choice, frequency_penalty, presence_penalty, n, logit_bias, user, stream_options
}
```

#### OpenAI Response (Chat Completions)

```go
type OpenAIChatResponse struct {
    ID      string         `json:"id"`
    Object  string         `json:"object"`
    Created int64          `json:"created"`
    Model   string         `json:"model"`
    Choices []OpenAIChoice `json:"choices"`
    Usage   OpenAIUsage    `json:"usage"`
}
type OpenAIChoice struct {
    Index        int            `json:"index"`
    Message      OpenAIMessage  `json:"message"`
    FinishReason *string        `json:"finish_reason"`
}
type OpenAIMessage struct {
    Role      string           `json:"role"`
    Content   any              `json:"content"`
    ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
}
type OpenAIToolCall struct {
    ID       string             `json:"id"`
    Type     string             `json:"type"`
    Function OpenAIFunctionCall `json:"function"`
}
type OpenAIFunctionCall struct {
    Name      string `json:"name"`
    Arguments string `json:"arguments"`
}
type OpenAITool struct {
    Type     string         `json:"type"`
    Function OpenAIFunction `json:"function"`
}
type OpenAIFunction struct {
    Name        string `json:"name"`
    Description string `json:"description,omitempty"`
    Parameters  any    `json:"parameters"`
}
type OpenAIUsage struct {
    PromptTokens     int `json:"prompt_tokens"`
    CompletionTokens int `json:"completion_tokens"`
    TotalTokens      int `json:"total_tokens"`
}
```

#### Anthropic Request (Messages)

```go
type AnthropicRequest struct {
    Model         string               `json:"model"`
    Messages      []AnthropicMessage   `json:"messages"`
    System        []AnthropicTextBlock `json:"system,omitempty"`
    MaxTokens     int                  `json:"max_tokens"`
    Temperature   *float64             `json:"temperature,omitempty"`
    TopP          *float64             `json:"top_p,omitempty"`
    TopK          *int                 `json:"top_k,omitempty"`
    StopSequences []string             `json:"stop_sequences,omitempty"`
    Stream        *bool                `json:"stream,omitempty"`
    Tools         []AnthropicTool      `json:"tools,omitempty"`
    Metadata      map[string]any       `json:"metadata,omitempty"`
}
type AnthropicMessage struct {
    Role    string              `json:"role"`
    Content []AnthropicContent  `json:"content"`
}
type AnthropicContent struct {
    Type      string                 `json:"type"`
    Text      string                 `json:"text,omitempty"`
    Source    *AnthropicImageSource  `json:"source,omitempty"`    // image
    ID        string                 `json:"id,omitempty"`        // tool_use
    Name      string                 `json:"name,omitempty"`      // tool_use
    Input     any                    `json:"input,omitempty"`     // tool_use
    ToolUseID string                 `json:"tool_use_id,omitempty"` // tool_result
    Content   any                    `json:"content,omitempty"`   // tool_result (string or []Content)
}
type AnthropicTextBlock struct {
    Type string `json:"type"`
    Text string `json:"text"`
}
type AnthropicImageSource struct {
    Type      string `json:"type"`       // "base64"
    MediaType string `json:"media_type"` // "image/jpeg" etc
    Data      string `json:"data"`
}
type AnthropicTool struct {
    Name        string `json:"name"`
    Description string `json:"description,omitempty"`
    InputSchema any    `json:"input_schema"`
}
```

#### Anthropic Response (Messages)

```go
type AnthropicResponse struct {
    ID           string              `json:"id"`
    Type         string              `json:"type"`
    Role         string              `json:"role"`
    Content      []AnthropicContent  `json:"content"`
    Model        string              `json:"model"`
    StopReason   *string             `json:"stop_reason"`
    StopSequence *string             `json:"stop_sequence"`
    Usage        AnthropicUsage      `json:"usage"`
}
type AnthropicUsage struct {
    InputTokens              int `json:"input_tokens"`
    OutputTokens             int `json:"output_tokens"`
    CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
    CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}
```

### 字段映射: OpenAI Request → Anthropic Request

| OpenAI Chat Completions | Anthropic Messages | 处理 |
|---|---|---|
| `model` | `model` | 用 `--model` flag 值覆盖 |
| `messages` | `messages` + 顶层 `system` | 逐条映射 |
| `max_tokens` | `max_tokens` | 同名，默认 8192 |
| `temperature` | `temperature` | 同名 |
| `top_p` | `top_p` | 同名 |
| `top_k` | `top_k` | 同名 |
| `stop` | `stop_sequences` | string 或 []string 统一转 []string |
| `stream` | `stream` | 同名 |
| `tools[]` | `tools[]` | function→{name,description,input_schema} |
| 其余(忽略) | — | tool_choice, frequency_penalty, presence_penalty, n, logit_bias, user, stream_options |

#### Request Messages 逐条映射

| OpenAI message | Anthropic message |
|---|---|
| `role: system` | 顶层 `system: [{type:"text",text:...}]` |
| `role: user, content: string` | `{role:"user", content:[{type:"text",text:"..."}]}` |
| `role: user, content: [{type:"text"\|"image_url"}]` | image_url(data:base64) → source:{type:"base64",media_type,data} |
| `role: assistant, content: string` | `{role:"assistant", content:[{type:"text",text:...}]}` |
| `role: assistant, content:null + tool_calls[]` | tool_calls→{type:"tool_use",id,name,input} |
| `role: tool` | `{role:"user", content:[{type:"tool_result",tool_use_id,content}]}` |
| `role: function` | `{role:"user"}` + "Function result: ..." text |

### 字段映射: Anthropic Response → OpenAI Response

| Anthropic Response | OpenAI Response | 处理 |
|---|---|---|
| `id` | `id` | `msg_` → `chatcmpl-` |
| — | `object` | 固定 `"chat.completion"` |
| — | `created` | `time.Now().Unix()` |
| `model` | `model` | 透传 |
| — | `choices[0].index` | 0 |
| — | `choices[0].message.role` | `"assistant"` |
| content[].text | `content` | 拼接 text blocks |
| content[].tool_use | `tool_calls[]` | + function call |
| content[].thinking | — | 忽略 |
| content[].signature | — | 忽略 |
| `stop_reason` | `finish_reason` | end_turn→stop, max_tokens→length, tool_use→tool_calls, stop_sequence→stop |
| `usage.input_tokens` | `prompt_tokens` | |
| `usage.output_tokens` | `completion_tokens` | |
| — | `total_tokens` | prompt + completion |

### 自动检测流程 (convert.go)

```
Convert(body []byte) ([]byte, error)
  ├─ json.Unmarshal 失败 → 透传
  ├─ OpenAI Request (model/messages) → Anthropic Request → marshal
  ├─ Anthropic Response (type+stop_reason) → OpenAI Response → marshal
  └─ 都不匹配 → 透传
```

### 项目结构

```
llm-api-converter/
├── go.mod
├── main.go
├── cmd/
│   └── root.go
├── rewriter/
│   └── server.go
├── convert/
│   ├── convert.go
│   ├── types.go
│   └── convert_test.go
```

### CLI 参数

```bash
go run . rewriter --addr :8000 --model "claude-sonnet-4-20250514" --max-tokens 8192
```

| Flag | 默认值 | 说明 |
|---|---|---|
| `--addr` | `:8000` | 监听地址 |
| `--model` | `claude-sonnet-4-20250514` | Anthropic model ID |
| `--max-tokens` | `8192` | 默认 max_tokens |
| `--log.level` | `info` | 日志级别 |
| `--log.format` | `text` | 日志格式 |

### 关键实现细节

1. **Content blocks**: Anthropic content 总是数组。OpenAI content string/array 需自动适配。
2. **image_url**: 仅 `data:image/...;base64,...` 支持；URL 引用跳过。
3. **tool_result.content**: 可以是 string 或 []Content，统一处理。
4. **Thinking blocks**: 响应转换中忽略，不进入 OpenAI 响应。
5. **零三方依赖**: 仅 `github.com/spf13/cobra`。
6. **不 panic**: 异常 → `{"ok":false}`。

### 实施顺序

1. `go.mod` + `main.go` + `cmd/root.go`
2. `convert/types.go` — 全部类型定义
3. `convert/convert.go` — Convert() + 双向转换逻辑
4. `convert/convert_test.go` — 单元测试
5. `rewriter/server.go` — HTTP server + /rewrite
6. 构建 + vet + test 验证
