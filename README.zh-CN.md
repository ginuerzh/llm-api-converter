# llm-api-converter

[English](README.md) | 简体中文

一个 GOST Rewriter HTTP 插件，在 **OpenAI Chat Completions** 与 **Anthropic Messages API** 两种格式之间双向转换。专为 Claude Code、Codex CLI、OpenCode 以及其他使用这两种协议的 LLM 客户端设计。

## 目录

- [工作原理](#工作原理)
- [快速开始](#快速开始)
  - [使用 Docker Compose](#使用-docker-compose)
  - [Claude Code → DeepSeek（经 opencode-go）](#claude-code--deepseek经-opencode-go)
  - [Codex CLI → DeepSeek（经 opencode-go）](#codex-cli--deepseek经-opencode-go)
- [功能特性](#功能特性)
  - [协议转换](#协议转换)
  - [流式处理](#流式处理)
  - [多层推理缓存（DeepSeek V4）](#多层推理缓存deepseek-v4)
  - [消息序列清洗](#消息序列清洗)
  - [内容支持](#内容支持)
- [命令行参数](#命令行参数)
- [项目结构](#项目结构)
- [测试](#测试)
- [相关项目](#相关项目)
- [许可证](#许可证)

## 工作原理

作为 GOST rewriter 插件部署，在前向代理中拦截 HTTP 请求/响应体，在两种线路格式之间透明转换：

```
Anthropic SDK 客户端 → GOST → llm-api-converter → OpenAI 兼容 API
                       ↕（协议转换）            ↕
                 Anthropic 格式              OpenAI 格式
```

转换器仅使用**正向结构标记**自动检测输入格式——不做反向排除，不硬编码模型名前缀。`SessionStore` 在请求/响应成对出现时跟踪客户端协议，以保证双向路由正确。对于缺少区分特征的极简请求，使用基于 URI 的回退检测。

## 快速开始

```bash
# 构建
go build -o llm-api-converter .

# 独立运行
./llm-api-converter --addr :8000 --model-map "claude-opus=deepseek-v4-pro:openai,claude-sonnet=deepseek-v4-flash,*=deepseek-v4-flash:openai"

# 配合 GOST 运行
gost -C gost.yaml
```

### 使用 Docker Compose

将转换器作为容器与 GOST 一起运行。已发布镜像为 `ginuerzh/llm-api-converter`（多架构：amd64/arm64/arm v6/v7）。由于镜像的 `ENTRYPOINT` 就是二进制本身，通过 `command:` 提供 CLI 参数。

```yaml
# docker-compose.yml
services:
  llm-converter:
    image: ginuerzh/llm-api-converter:latest
    command:
      - --addr
      - :8000
      - --model
      - deepseek-v4-flash
      - --model-map
      - claude-opus=deepseek-v4-pro:openai,claude-sonnet=deepseek-v4-flash,*=deepseek-v4-flash:openai
    ports:
      - "8000:8000"
    restart: unless-stopped

  gost:
    image: gogost/gost:latest
    command: -C /etc/gost/gost.yaml
    volumes:
      - ./gost.yaml:/etc/gost/gost.yaml:ro
    ports:
      - "8787:8787"
    depends_on:
      - llm-converter
    restart: unless-stopped
```

将 GOST rewriter 插件指向转换器的容器地址：

```yaml
# 在 gost.yaml 中
rewriters:
- name: llm-converter
  plugin:
    type: http
    addr: http://llm-converter:8000/rewrite
```

```bash
docker compose up -d
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude
```

本地构建镜像而不是拉取：

```bash
docker build -t ginuerzh/llm-api-converter .
# 或者使用 .github/workflows/buildx.yml 中的多架构 buildx 工作流
```

### Claude Code → DeepSeek（经 opencode-go）

此方案让 Claude Code（Anthropic 协议）通过转换器调用 DeepSeek 模型（OpenAI 协议）：

```
Claude Code → GOST（代理）→ llm-api-converter → opencode-go API → DeepSeek
```

**1. 启动转换器：**

```bash
./llm-api-converter \
  --addr :8000 \
  --model deepseek-v4-flash \
  --model-map "claude-opus=deepseek-v4-pro:openai,*=deepseek-v4-flash:openai"
```

**2. 配置 GOST 拦截 Anthropic API 调用并通过转换器转发：**

```yaml
# gost.yaml
services:
- name: claude-code-proxy
  addr: :8787
  handler:
    type: tcp
    metadata:
      sniffing: true
  listener:
    type: tcp
  forwarder:
    nodes:
    - name: opencode-go
      addr: opencode.ai:443
      tls:
        secure: true
        serverName: opencode.ai
      http:
        host: opencode.ai
        rewriteURL:
        # Anthropic /v1/messages → OpenAI /v1/chat/completions
        - match: /v1/messages
          replacement: /zen/go/v1/chat/completions
        requestHeader:
          Authorization: "Bearer your-oc-apikey"
          x-api-key: "your-oc-apikey"
        rewriteRequestBody:
        - rewriter: llm-converter
          type: application/json
        rewriteResponseBody:
        - rewriter: llm-converter
          type: "*"

rewriters:
- name: llm-converter
  plugin:
    type: http
    addr: http://127.0.0.1:8000/rewrite
```

**3. 将 Claude Code 指向代理：**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
claude
```

来自 Claude Code 的所有 Anthropic 流量被 GOST 拦截，由插件转换为 OpenAI Chat Completions 格式，转发到 opencode-go API 进行 DeepSeek 推理。响应和 SSE 流在返回时被透明地转换回 Anthropic 格式。

**模型映射说明：**

- `claude-opus=deepseek-v4-pro:openai`：将模型名以 `claude-opus` 开头的请求路由到 DeepSeek V4 Pro，并执行 Anthropic→OpenAI 转换
- `*=deepseek-v4-flash:openai`：兜底项，匹配任何未命中的模型前缀，同样转换为 OpenAI
- **下游协议覆盖**：在 target 后追加 `:openai` 或 `:anthropic` 来声明后端使用的协议格式（`prefix=target:protocol`）。**不带协议后缀时，默认是透传**——报文仅重写模型名后原样通过，不做格式转换。带 `:openai`/`:anthropic` 时，若传入协议与声明的协议不同则执行转换；若相同则仅重写模型名。该覆盖在请求和响应两个方向都生效（通过按会话跟踪客户端协议实现）。示例：`claude-opus=deepseek-v4-pro:openai`——传入 Anthropic 与 `:openai` 不同，执行 Anthropic→OpenAI 转换；`claude-opus=deepseek-v4-pro:anthropic`——传入 Anthropic 与之匹配，仅重写模型名。
- 注意：`:responses` 不是合法的覆盖值（只支持 `openai`/`anthropic`）；Responses API 流量通过报文标记和会话存储检测并路由，不走 model map。空 target（如 `claude-opus=:openai`）会在解析阶段被拒绝。

请根据你的 opencode-go 部署可用的模型，相应调整 `--model-map`。

### Codex CLI → DeepSeek（经 opencode-go）

此方案让 Codex CLI（OpenAI Responses API 协议）通过转换器调用 DeepSeek 模型（OpenAI Chat Completions 协议）：

```
Codex CLI → GOST（代理）→ llm-api-converter → opencode-go API → DeepSeek
```

Codex CLI 发送 Responses API 格式（`POST /v1/responses`，包含 `{model, input, ...}`）；转换器将其翻译为 Chat Completions 格式（`POST /v1/chat/completions`，包含 `{model, messages, ...}`）发给 opencode-go，并在响应返回时反向转换。

**1. 启动转换器：**

```bash
./llm-api-converter \
  --addr :8000 \
  --model deepseek-v4-flash \
  --model-map "gpt-4=deepseek-v4-pro:openai,*=deepseek-v4-flash:openai"
```

`:openai` 协议覆盖声明下游使用 OpenAI Chat Completions。Responses API 检测先于透传判断执行，因此请求仍会走完整转换（Responses → Chat）；在响应路径上，它防止 Chat 响应在返回时被错误地转换为 Anthropic 格式。

**2. 配置 GOST 拦截 Codex CLI 的 API 调用并通过转换器转发：**

```yaml
# gost.yaml
services:
- name: codex-cli-proxy
  addr: :8787
  handler:
    type: tcp
    metadata:
      sniffing: true
  listener:
    type: tcp
  forwarder:
    nodes:
    - name: opencode-go
      addr: opencode.ai:443
      tls:
        secure: true
        serverName: opencode.ai
      http:
        host: opencode.ai
        rewriteURL:
        # Responses API /v1/responses → Chat Completions /v1/chat/completions
        - match: /v1/responses
          replacement: /zen/go/v1/chat/completions
        requestHeader:
          Authorization: "Bearer your-oc-apikey"
          x-api-key: "your-oc-apikey"
        rewriteRequestBody:
        - rewriter: llm-converter
          type: application/json
        rewriteResponseBody:
        - rewriter: llm-converter
          type: "*"

rewriters:
- name: llm-converter
  plugin:
    type: http
    addr: http://127.0.0.1:8000/rewrite
```

**3. 将 Codex CLI 指向代理：**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
codex
```

Codex CLI 将 Responses API 请求发送到 `/v1/responses`；GOST 拦截后，转换器把报文重写为 Chat Completions 格式（并重映射模型名），请求被转发到 opencode-go，URL 重写为 `/zen/go/v1/chat/completions`。上游的 Chat Completions 响应在返回时被透明地转换回 Responses API 格式。

## 功能特性

### 协议转换

| 方向 | 说明 |
|------|------|
| OpenAI Request → Anthropic Request | 用于转发到 Anthropic API |
| OpenAI Response → Anthropic Response | 用于向客户端返回 Anthropic 格式的响应 |
| Anthropic Request → OpenAI Request | 用于转发到 OpenAI 兼容的下游（DeepSeek 等） |
| Anthropic Response → OpenAI Response | 用于向客户端返回 OpenAI 格式的响应 |
| Responses API Request → Chat Completions Request | 用于将 Codex CLI（Responses API）转发到 OpenAI Chat Completions 后端 |
| Chat Completions Response → Responses API Response | 将上游 Chat 响应转换回 Responses API 格式 |
| Chat Completions Request → Responses API Request | 将任意 Chat 请求转换为 Responses API 格式 |

### 流式处理

**Anthropic SSE** —— 将 OpenAI 流式 delta 分块转换为标准的 Anthropic SSE 事件序列：

```
message_start → ping → content_block_start → content_block_delta* → content_block_stop → message_delta → message_stop
```

支持文本、思考（reasoning）和工具调用 delta，正确处理内容块切换、思考块的 signature_delta，并限制工具名以防止工具幻觉。

**Responses API SSE** —— 将 Chat Completions 流式 delta 转换为 Responses API SSE 事件序列：

```
response.created → response.in_progress → output_item.added → content_part.added → response.output_text.delta* → response.output_text.done → output_item.finished → response.completed
```

处理流式推理内容（`thinking` 不是 Responses API 的一等概念；推理以带 `type: "reasoning"` 标注的 `response.output_text.delta` 合并）、文本 delta、跨分块的工具调用累积，以及错误传播。

### 多层推理缓存（DeepSeek V4）

处理 DeepSeek V4 在存在工具调用时必须保留 `reasoning_content` 的要求。缓存分三层存储推理内容：

1. **工具调用 ID** —— 精确的工具调用重放
2. **工具上下文** —— 相同工具模式、不同 ID
3. **助手文本** —— 基于文本的兜底

支持可选的文件持久化、30 天 TTL 和 FIFO 淘汰。

缓存后端通过 `ReasoningStore` 接口（`Get`、`Set`、`Delete`、`Len`）可插拔，允许在默认内存 map 之外实现自定义存储。

### 消息序列清洗

- 将工具调用与其结果配对，丢弃未满足的工具调用
- 将孤立的工具结果转为用户文本消息
- 合并连续的助手工具调用消息（Claude Code 对话压缩导致）
- 当 DeepSeek V4 需要推理内容但缓存为空时，注入占位推理

### 内容支持

- 文本和多部分内容块
- 图片 data URI（`data:image/...;base64,...`）
- tool use / tool result 块
- 扩展思考 / 推理内容
- 系统消息

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--addr` | `:8000` | 监听地址 |
| `--model` | `deepseek-chat` | 默认兜底模型 ID |
| `--max-tokens` | `8192` | 默认 max_tokens |
| `--model-map` | `` | 模型映射：`prefix=target[:protocol],...`（`*` 为兜底，protocol: openai\|anthropic） |
| `--cache` | `memory` | 推理缓存后端：`memory` 或 `file:<path>` |
| `--log.level` | `info` | 日志级别 |
| `--log.format` | `json` | 日志格式（text 或 json） |

## 项目结构

```
llm-api-converter/
├── main.go              # 入口
├── cmd/root.go          # Cobra CLI
├── convert/             # 核心转换逻辑
│   ├── types.go                              # OpenAI、Anthropic、Responses API、SSE 数据类型
│   ├── convert.go                            # 入口：Convert + ConvertSSE 分发
│   ├── detect.go                             # 基于报文的协议检测（正向标记）
│   ├── protocol.go                           # Protocol 类型 + URI 回退 + resolveModel
│   ├── registry.go                           # ConversionKey → 转换函数 map
│   ├── session.go                            # SessionStore（按会话状态，FIFO 淘汰）
│   ├── anthropic_to_openai.go                # Anthropic → OpenAI Chat Completions
│   ├── openai_to_anthropic.go                # OpenAI Chat Completions → Anthropic
│   ├── responses.go                          # Responses API ↔ Chat Completions
│   ├── stream.go                             # SSE 流工具
│   ├── stream_anthropic.go                   # Anthropic SSE 状态机（OpenAI → Anthropic 流式）
│   ├── stream_responses.go                   # Responses API SSE 状态机
│   ├── reasoning_cache.go                    # 3 层推理缓存 + ReasoningStore 接口
│   └── *_test.go                             # 测试
├── rewriter/
│   ├── server.go                             # HTTP 插件服务
│   └── server_test.go
├── tests/e2e/                                # 集成测试
└── docs/plans/                               # 历史设计文档
```

## 测试

```bash
go test ./... -v -count=1
go test ./... -race
go test ./tests/e2e/ -v -timeout 5m
```

## 相关项目

- [deepseek-v4-opencode-claude-code-bridge](https://github.com/superheroYu/deepseek-v4-opencode-claude-code-bridge) —— OpenCode 和 Claude Code 的 DeepSeek V4 适配器
- [opencode-cc](https://github.com/Kiowx/opencode-cc) —— OpenCode Claude Code 桥接
- [cc-switch](https://github.com/farion1231/cc-switch) —— Claude Code 供应商/配置切换工具

## 许可证

属于 [GOST](https://github.com/go-gost/gost) 项目的一部分。
