# 协议检测:URI 主导 → 结构主导 + Session 传递 client 协议

## Context(背景)

当前协议检测以 GOST 的 `uri` 元数据为**主**、`detectByBody`(结构检测)为**兜底**(见 [convert.go:347-351](../../convert/convert.go#L347-L351))。用户提出:单纯靠 URI 判断 client protocol 不保险,需要基于数据结构自动检测或更优方案。

调研后确认 URI 主导有三个真实缺陷:

1. **响应路径 URI 会缺失且语义错误**。GOST 在 [sniffer_rewrite.go:48-55](../../../../go-gost/x/internal/util/forwarder/sniffer_rewrite.go#L48-L55) 用 `resp.Request.RequestURI` 填充响应 URI —— 当 `resp.Request == nil`(中间件注入的合成响应/错误)时为空;**且即使非空,响应 URI 只表示 client 端点,绝不表示响应 body 的实际格式**(响应 body 是 downstream 协议格式)。URI 主导导致 [convert.go:364-373](../../convert/convert.go#L364-L373) 不得不再次调用 `detectByBody` 来补救(asymmetric hack)。
2. **脆弱的 200 行启发式检测仍作为兜底存在**。[detect.go](../../convert/detect.go) 里 `isAnthropicRequest` 靠**负向排除**(`frequency_penalty`/`presence_penalty`/`n`/`seed`/`stop`/`tool_choice` as string…)和 `hasOpenAIStyleModel` 的**硬编码模型前缀**(`gpt-`/`o1`/`o3`/`deepseek`/`gemini-`/`glm-`)来区分 —— 这正是上一轮 matrix 重设计计划要删除、却实际保留的代码。
3. **流式与非流式路径各自重复检测逻辑**,且非流式响应路径从不复用请求时已得到的 client 协议。

各信号的真实语义(已核实):

| 信号 | 请求 | 响应 |
|---|---|---|
| `sid` | 恒有 | 恒有,**天然关联请求↔响应** |
| `direction` | `"request"` | `"response"`(调用点硬编码,恒可靠) |
| `uri` | `req.RequestURI`,恒有,= client 端点 | `resp.Request.RequestURI`,**`resp.Request==nil` 时为空**;非空时仍只 = client 端点,**非 body 格式** |

**关键洞察**:source(这批字节是什么格式)只能由 **body** 决定;target(要转成什么)中,响应的 target = client 协议,而 **SID 把请求和响应天然关联** —— 请求时(URI 可靠)检测出 client 协议并存入 Session,响应时直接读取,比依赖响应 URI 稳健得多。URI 仅在**病态最小请求**(body 无任何生态特征)且端点为规范 URI 时作为可选兜底。

预期结果:body 结构主导 source 检测、URI 降级为可选兜底、Session 在请求→响应间传递 client 协议;asymmetric hack 与脆弱启发式一并删除。

## 设计

### 0. 检测层级(总览)

| 层级 | 信号 | 适用 |
|---|---|---|
| 1(主) | **body 正向标记** | 所有真实流量,**含自定义端点**(`/v1/llm` 等)—— 真实客户端总会带生态特征 |
| 2(可选兜底) | **URI** | 仅当 body Unknown 且 URI 为规范端点(`/v1/messages` 等)的最小请求 |
| 3(安全兜底) | **passthrough 原样放行** | 病态最小请求(body 无特征 + URI 自定义);**拒绝猜测** |

关键洞察:真实客户端的请求体即使打任意端点也几乎总带生态标记 —— Claude Code 有顶层 `system` + `tools[].input_schema`,OpenAI 客户端有 `messages[].role:"system"` + `tools[].function`,Codex 有 `input`。所以 body 检测在第一层就解决 `/v1/llm`,URI 退为可选兜底。

**有意行为变更**:当前代码对病态最小请求用 `hasOpenAIStyleModel` 硬编码模型前缀(`gpt-`/`o1`/`deepseek`...)去猜;新设计拒绝猜测、安全放行(猜错破坏 body,比放行更糟)。若运维确需在自定义端点上转换此类请求,可后续加 `--client-protocol` 声明开关(YAGNI,先不加)。

### 1. `detectSource(body) Protocol` —— 正向标记,替换 detect.go

重写 [convert/detect.go](../../convert/detect.go) 为**正向标记**检测,删除全部负向排除与硬编码模型前缀:

- 响应:`choices[]` → OpenAIChat;`type:"message"`+`usage`/`stop_reason` → Anthropic;`object:"response"`/`output[]` 带类型 → Responses
- 请求 Anthropic:`tools[].input_schema` / content block `tool_use`/`tool_result`/`thinking`/`image` / **顶层 `system` 字段** / `tool_choice.type=="any"|"tool"`
- 请求 OpenAIChat:**`messages[].role ∈ {system, tool, function}`** / `tools[].function` 包装 / `messages[].tool_calls` / `n`/`frequency_penalty`/`presence_penalty`/`response_format`/`logit_bias`
- 请求 Responses:`input`(无 `messages`)、`instructions`
- **病态最小请求**(无 system、无 tools、无特征角色 —— 仅 `{model, messages:[{user}], max_tokens}`)→ `ProtocolUnknown`,**不猜**

direction 不再从 body 推断 —— 直接用 `opts.direction`(恒可靠)。`detectByBody` 删除,改为 `detectSource(raw) Protocol`(只返回协议)。

### 2. URI 降级为可选兜底

`detectByURI(uri) Protocol` 复用现有 [uriTable](../../convert/protocol.go#L39-L43)。**仅在 `detectSource == Unknown` 时调用**;匹配不到(自定义端点如 `/v1/llm`)→ Unknown,进入 passthrough。响应路径 URI 不作 source 依据(它本就不表示 body 格式)。

### 3. Session 传递 client 协议(用户已选方案)

`Session` 结构已有 `From`/`To` 字段([session.go:27-28](../../convert/session.go#L27-L28)),`From` 即 client 协议,**复用、不新增字段**。

- **请求时**:`source = detectSource(body) or detectByURI(uri)`;`session.From = source`;`store.Set(sid, session)`;`target = resolveModel(source, modelMap)`(downstream 协议)。
- **响应时**:`source = detectSource(body)`(权威);`client = session.From or detectByURI(uri)`(兜底);`delete(session)`(非流式请求-响应对此刻完成);按 `source==client` 判定 passthrough,否则 `conversions[{source,client}]`。

### 4. Session 清理(防泄漏)

非流式请求现在也建 session 条目。复用 [reasoning_cache.go](../../convert/reasoning_cache.go) 里已有的 `memoryStore` FIFO 模式(ladder:复用仓库内已验证代码),给 `SessionStore` 加 **max-size + FIFO 淘汰** 作为安全网 —— 正常路径在响应时 `Delete`,泄漏(响应未到)由容量上限兜底。无需引入 TTL/goroutine。

### 5. 死代码删除

| 代码 | 位置 |
|---|---|
| asymmetric-response hack(再次 detectByBody + 翻转 from/clientProto) | [convert.go:364-373, 404-407](../../convert/convert.go#L364-L373) |
| `detectByBody` 及 `isOpenAIRequest`/`hasOpenAIStyleModel`/负向排除 | [detect.go](../../convert/detect.go)(整体重写) |
| `ConvertSSE` 内嵌 body 检测块 | [convert.go:148-180](../../convert/convert.go#L148-L180)(简化为 detectSource) |
| `HandleSSEEvent` Start 阶段 detectProtocol→detectByBody 兜底链 | [convert.go:553-574](../../convert/convert.go#L553-L574)(改为 detectSource + session.Client) |

## 改动文件

| 文件 | 改动 |
|---|---|
| `convert/detect.go` | 重写:`detectSource` 正向标记;删 `detectByBody`/`isOpenAIRequest`/`hasOpenAIStyleModel`/负向排除 |
| `convert/protocol.go` | `detectProtocol`→重命名/收窄为 `detectByURI`(仅 URI→Protocol);`detectSource` 不在此文件 |
| `convert/convert.go` | `Convert()`:请求存 session.From、响应读 session.From;删 asymmetric hack、删 detectByBody 兜底;`ConvertSSE`/`HandleSSEEvent` 改用 detectSource |
| `convert/session.go` | `SessionStore` 加 max-size + FIFO(复用 memoryStore 模式);非流式响应路径 `Delete` |
| 测试 `convert/convert_test.go` | 现有 matrix fuzz(664 组合)回归;新增:自定义端点 `/v1/llm` body 第一层判定、病态最小请求 passthrough、空 URI 响应靠 session、请求存/响应读 session |

不变:转换函数、`ReasoningCache`、`responsesStreamHandler`、`convert_*.go`/`stream_*.go` 文件布局、模型名改写逻辑(`passthrough`/`PassthroughStreamHandler`)。

## 验证

```bash
cd llm-api-converter
CGO_ENABLED=1 go test -race ./... -count=1          # 含 664 组合 matrix fuzz 回归
go test ./convert/ -fuzz=FuzzConvert_Matrix -fuzztime=10s
go test ./tests/e2e/ -v -timeout 5m                  # 真实 GOST→plugin→mock 链路
go build ./... && go vet ./...
```

重点回归场景:
1. Claude Code(`/v1/messages`,Anthropic 请求)→ DeepSeek(OpenAI 响应):请求存 client=Anthropic,响应 source=OpenAI、target=Anthropic,正确转回。
2. **自定义端点 `/v1/llm`**:Claude Code 请求(顶层 `system`+`tools[].input_schema`)→ body 第一层判定 Anthropic,URI 不参与;OpenAI 客户端请求(`role:"system"`+`tools[].function`)→ body 判定 OpenAI。证明自定义端点不依赖 URI。
3. **病态最小请求** `{model,messages:[{user}],max_tokens}` 打 `/v1/llm`:body Unknown + URI Unknown → passthrough 原样放行(不猜)。
4. 最小请求打规范端点(`/v1/messages` vs `/v1/chat/completions`):body Unknown,URI 第二层打破平局。
5. 下游返回合成错误响应(`resp.Request==nil`,空 URI):session.From 仍提供 client 协议,不依赖响应 URI。
6. 流式:StreamPhaseStart 用 detectSource 并存 session.From;Event/End 读 session(已有路径,仅切换检测函数)。
