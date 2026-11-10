# Anthropic Client → Responses Upstream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an Anthropic Messages client (Claude Code) use a Responses-only upstream by converting the request, the non-streaming response, and the SSE stream.

**Architecture:** Keep the pairwise registry. Add two one-way conversion functions and one stateful Responses→Anthropic stream encoder. Route the response path by the client protocol recorded on the session, and add a `response.*` detection marker for stream events.

**Tech Stack:** Go, existing `convert` package, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-09-anthropic-responses-upstream-design.md`

## Global Constraints

- No new Go dependency and no `replace` directive.
- Keep the pairwise architecture; do not introduce an IR.
- Code comments are English.
- Existing Anthropic↔Chat, Chat↔Gemini and Anthropic↔Gemini behavior must not change.
- `go test -count=1 ./...`, `go vet ./...` and `CGO_ENABLED=1 go test -race -count=1 ./convert` must pass.
- Commit after every task with its implementation record. Never push without explicit approval.
- The cuttle dependency bump uses a published tag; tagging and pushing are human-gated.
- Record the llm-api-converter commit and, when the cuttle bump happens, the cuttle commit separately.

## Review Focus

These are the inputs most likely to hurt a real Claude Code user; each has a test in the owning task.

1. A tool turn with parallel calls must keep every `call_id` paired with its result, in order — Task 2 and Task 6.
2. Claude Code sends previous thinking blocks back; dropping them must not lose tool pairing or trigger a 400 — Task 2.
3. Bailian streams reasoning as `reasoning_text` and may send tool arguments only in the done event; both must still produce complete Anthropic blocks — Task 6.
4. A truncated stream or truncated tool arguments must end as an error, never as a successful `message_stop` — Task 6 and Task 7.
5. A missing session or a first event that is only `{"type":"error"}` must not be treated as an Anthropic passthrough — Task 7.

---

### Task 1: Responses types and stream detection

**Files:**
- Modify: `convert/responses.go`
- Modify: `convert/detect.go`
- Test: `convert/anthropic_responses_test.go` (create)

**Interfaces:**
- Consumes: existing `ResponsesRequest`, `ResponsesStreamEvent`, `detectSource`.
- Produces: `ResponsesReasoning{Effort, Summary string}`, `ResponsesText{Format map[string]any}`, `ResponsesUsageDetails.CacheWriteTokens`, `ResponsesStreamEvent.Name`.

- [x] **Step 1: Write the failing tests**

```go
func TestDetectResponsesStreamingEvents(t *testing.T) {
	for _, typ := range []string{"response.created", "response.output_item.added", "response.output_text.delta", "response.completed"} {
		if got := detectSource(map[string]any{"type": typ}); got != ProtocolOpenAIResponses {
			t.Fatalf("detectSource(%q) = %v, want Responses", typ, got)
		}
	}
	if got := detectSource(map[string]any{"type": "message_start"}); got != ProtocolAnthropic {
		t.Fatalf("message_start = %v, want Anthropic", got)
	}
}

func TestResponsesRequestReasoningMarshal(t *testing.T) {
	req := ResponsesRequest{
		Model: "m", Stream: true, Store: boolPtr(false),
		Reasoning: &ResponsesReasoning{Effort: "high", Summary: "auto"},
		Text:      &ResponsesText{Format: map[string]any{"type": "json_schema", "name": "response"}},
	}
	b, err := json.Marshal(req)
	if err != nil { t.Fatal(err) }
	s := string(b)
	for _, want := range []string{`"reasoning":{"effort":"high","summary":"auto"}`, `"text":{"format":`, `"store":false`} {
		if !strings.Contains(s, want) { t.Fatalf("missing %s in %s", want, s) }
	}
}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestDetectResponsesStreamingEvents|TestResponsesRequestReasoningMarshal' -v`
Expected: FAIL — undefined `ResponsesReasoning`, `ResponsesText`, or detection returns Anthropic/unknown.

- [x] **Step 3: Add the types and detection branch**

In `convert/responses.go`, add:

```go
type ResponsesReasoning struct {
	Effort  string `json:"effort,omitempty"`
	Summary string `json:"summary,omitempty"`
}

type ResponsesText struct {
	Format map[string]any `json:"format,omitempty"`
}
```

Add `Reasoning *ResponsesReasoning` and `Text *ResponsesText` to `ResponsesRequest`; add `CacheWriteTokens int` to `ResponsesUsageDetails`; add `Name string \`json:"name,omitempty"\`` to `ResponsesStreamEvent`.

In `convert/detect.go`, import `strings` and add this before the Anthropic streaming-event switch:

```go
if t, ok := raw["type"].(string); ok && strings.HasPrefix(t, "response.") {
	return ProtocolOpenAIResponses
}
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -run 'TestDetectResponsesStreamingEvents|TestResponsesRequestReasoningMarshal' -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add convert/responses.go convert/detect.go convert/anthropic_responses_test.go
git commit -m "feat(convert): add Responses reasoning types and stream detection"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `3a0f336`. Files: `convert/responses.go`, `convert/detect.go`, `convert/anthropic_responses_test.go`. Tests: `GOWORK=/tmp/opencode/ws/go.work go test ./convert -run 'TestDetectResponsesStreamingEvents|TestResponsesRequestReasoningMarshal' -v` → PASS. Review findings: none. Deferred: none.

---

### Task 2: Anthropic request → Responses request

**Files:**
- Create: `convert/anthropic_responses.go`
- Modify: `convert/registry.go`
- Test: `convert/anthropic_responses_test.go`

**Interfaces:**
- Consumes: `AnthropicRequest`, `AnthropicContent`, `AnthropicTool`, `ResponsesRequest`, `ResponsesTool`, `ConvertOptions`.
- Produces:
  - `func convertAnthropicRequestToResponses(body []byte, opts *ConvertOptions) ([]byte, error)`
  - `func anthropicRequestToResponses(req *AnthropicRequest, opts *ConvertOptions) ([]byte, error)`
  - `func responsesReasoningFromThinking(th *AnthropicThinking, oc *AnthropicOutputConfig) *ResponsesReasoning`
  - `func anthropicImageURL(src *AnthropicImageSource) string`

- [x] **Step 1: Write the failing tests**

```go
func TestConvertAnthropicRequestToResponsesTextAndSystem(t *testing.T) {
	body := `{"model":"claude-opus-5-5","max_tokens":128,"system":[{"type":"text","text":"sys"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	out, err := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	if err != nil { t.Fatal(err) }
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	if req["instructions"] != "sys" || req["max_output_tokens"] != float64(128) || req["store"] != false {
		t.Fatalf("unexpected request: %s", out)
	}
	if _, ok := req["messages"]; ok { t.Fatalf("messages leaked: %s", out) }
	input := req["input"].([]any)
	if got := input[0].(map[string]any)["content"].([]any)[0].(map[string]any)["type"]; got != "input_text" {
		t.Fatalf("content type = %v", got)
	}
}

func TestConvertAnthropicRequestToResponsesToolsAndChoice(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"get_weather","description":"weather","input_schema":{"type":"object"}}],"tool_choice":{"type":"any","disable_parallel_tool_use":true}}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	tool := req["tools"].([]any)[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" {
		t.Fatalf("tool = %+v", tool)
	}
	if req["tool_choice"] != "required" || req["parallel_tool_calls"] != false {
		t.Fatalf("choice/parallel = %v/%v", req["tool_choice"], req["parallel_tool_calls"])
	}
}

func TestConvertAnthropicRequestToResponsesToolUseAndResult(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Beijing"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}]}]}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	input := req["input"].([]any)
	call := input[0].(map[string]any)
	result := input[1].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "toolu_1" || call["arguments"] != `{"city":"Beijing"}` {
		t.Fatalf("call = %+v", call)
	}
	if result["type"] != "function_call_output" || result["call_id"] != "toolu_1" || result["output"] != "sunny" {
		t.Fatalf("result = %+v", result)
	}
}

func TestConvertAnthropicRequestToResponsesThinking(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":20000},"output_config":{"effort":"high"}}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	if got := req["reasoning"].(map[string]any); got["effort"] != "high" || got["summary"] != "auto" {
		t.Fatalf("reasoning = %+v", got)
	}
}

func TestConvertAnthropicRequestToResponsesImageAndFormat(t *testing.T) {
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}]}],"output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}}`
	out, _ := convertAnthropicRequestToResponses([]byte(body), &ConvertOptions{})
	var req map[string]any
	_ = json.Unmarshal(out, &req)
	part := req["input"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if part["type"] != "input_image" || part["image_url"] != "data:image/png;base64,abc" {
		t.Fatalf("image = %+v", part)
	}
	if req["text"].(map[string]any)["format"].(map[string]any)["type"] != "json_schema" {
		t.Fatalf("text = %+v", req["text"])
	}
}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestConvertAnthropicRequestToResponses' -v`
Expected: FAIL — `convertAnthropicRequestToResponses` undefined.

- [x] **Step 3: Implement the converter and register the direction**

`convertAnthropicRequestToResponses` unmarshals `AnthropicRequest`, calls `anthropicRequestToResponses`, and returns `json.Marshal`. It returns the body unchanged when the JSON has no top-level `messages` field, so a misrouted response body cannot be fed to the request converter.

`anthropicRequestToResponses` builds a `ResponsesRequest` with:

- `Model`: `opts.ResolvedModel`, else `resolveModelTarget(req.Model, opts.ModelMap)`, else `req.Model`.
- `Instructions`: join non-empty system text blocks with `\n\n`, JSON-marshal the string.
- `Input`: a `[]any` built in message/block order:
  - user text → `{"type":"message","role":"user","content":[{"type":"input_text","text":...}]}`
  - user image → the same message content with `{"type":"input_image","detail":"auto","image_url":...}`
  - user `tool_result` → a separate `{"type":"function_call_output","call_id":tool_use_id,"output":...}` item; output is a string, or a list of `input_text`/`input_image` for block content. `AnthropicContent.Content` is `any`, so handle a `[]any` of block maps, not just a `[]AnthropicContent`.
  - assistant text → `{"type":"message","role":"assistant","content":[{"type":"output_text","text":...}]}`
  - assistant `tool_use` → `{"type":"function_call","call_id":id,"name":name,"arguments":json(input)}`; absent input → `{}`.
  - assistant `thinking` → skipped (debug log).
- `Tools`: `{Type:"function", Name, Description, Parameters: input_schema}`.
- `ToolChoice`: auto→"auto", none→"none", any→"required", tool→`{type:"function", name}`.
- `ParallelToolCalls`: `false` when `disable_parallel_tool_use` is true; otherwise nil.
- `MaxOutputTokens`: `req.MaxTokens`; `Temperature`, `TopP`; `Stream`; `Metadata`.
- `Reasoning`: `responsesReasoningFromThinking`. Return nil when thinking is absent or `type=="disabled"`; otherwise pick `output_config.effort` when set, else map `budget_tokens` (<=4096 low, <=12000 medium, <=24000 high, else high), map `minimal`/`low` to low, `xhigh`/`max` to high, and set `summary:"auto"`.
- `Text`: when `output_config.format` is a json_schema, `{format:{type:"json_schema",name:"response",schema:...}}`.
- `Store`: `boolPtr(false)`.
- Drop `stop_sequences` and `top_k` with a debug log.

Register in `convert/registry.go`:

```go
conversions[ConversionKey{ProtocolAnthropic, ProtocolOpenAIResponses}] = convertAnthropicRequestToResponses
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -run 'TestConvertAnthropicRequestToResponses' -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add convert/anthropic_responses.go convert/registry.go convert/anthropic_responses_test.go
git commit -m "feat(convert): translate Anthropic requests to Responses"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `41a92f1`. Files: `convert/anthropic_responses.go` (new), `convert/registry.go`, `convert/anthropic_responses_test.go`. Tests: `go test ./convert -run 'TestConvertAnthropicRequestToResponses' -v` → PASS; `go test ./convert -count=1` → PASS. Review findings: none. Deferred: thinking replay, stop_sequences/top_k unsupported.

---

### Task 3: Responses response → Anthropic response

**Files:**
- Modify: `convert/anthropic_responses.go`
- Modify: `convert/registry.go`
- Test: `convert/anthropic_responses_test.go`

**Interfaces:**
- Consumes: `ResponsesResponse`, `ResponsesOutputItem`, `ResponsesUsage`, `AnthropicResponse`, `resolveResponseModel`.
- Produces:
  - `func convertResponsesResponseToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error)`
  - `func responsesResponseToAnthropic(resp *ResponsesResponse, opts *ConvertOptions) ([]byte, error)`
  - `func responsesStopReason(resp *ResponsesResponse) string`
  - `func anthropicUsageFromResponses(u *ResponsesUsage) AnthropicUsage`
  - `func parseResponsesArguments(s string) map[string]any`

- [x] **Step 1: Write the failing tests**

```go
func TestConvertResponsesResponseToAnthropicText(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"deepseek-v4.1-flash-1017","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}],"usage":{"input_tokens":10,"output_tokens":2}}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{RequestModel:"claude-opus-5-5"})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if resp.Model != "claude-opus-5-5" || resp.Content[0].Type != "text" || resp.Content[0].Text != "Hello" {
		t.Fatalf("unexpected response: %s", out)
	}
	if resp.StopReason == nil || *resp.StopReason != "end_turn" { t.Fatalf("stop = %v", resp.StopReason) }
}

func TestConvertResponsesResponseToAnthropicReasoning(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"think"}]}]}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	if !strings.Contains(string(out), `"type":"thinking"`) || !strings.Contains(string(out), `"thinking":"think"`) || !strings.Contains(string(out), `"signature":""`) {
		t.Fatalf("unexpected response: %s", out)
	}
}

func TestConvertResponsesResponseToAnthropicToolCall(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"function_call","call_id":"toolu_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}]}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	b := resp.Content[0]
	if b.Type != "tool_use" || b.ID != "toolu_1" || b.Name != "get_weather" {
		t.Fatalf("block = %+v", b)
	}
	if resp.StopReason == nil || *resp.StopReason != "tool_use" {
		t.Fatalf("stop = %v", resp.StopReason)
	}
	if b.Input.(map[string]any)["city"] != "Beijing" {
		t.Fatalf("input = %+v", b.Input)
	}
}

func TestConvertResponsesResponseToAnthropicUsage(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"message","content":[{"type":"output_text","text":"x"}]}],"usage":{"input_tokens":100,"output_tokens":20,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":5}}}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	u := resp.Usage
	if u.InputTokens != 15 || u.CacheReadInputTokens != 80 || u.CacheCreationInputTokens != 5 || u.OutputTokens != 20 {
		t.Fatalf("usage = %+v", u)
	}
}

func TestConvertResponsesResponseToAnthropicIncomplete(t *testing.T) {
	body := `{"id":"resp_1","object":"response","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model":"m","output":[]}`
	out, _ := convertResponsesResponseToAnthropic([]byte(body), &ConvertOptions{})
	var resp AnthropicResponse
	_ = json.Unmarshal(out, &resp)
	if resp.StopReason == nil || *resp.StopReason != "max_tokens" {
		t.Fatalf("stop = %v", resp.StopReason)
	}
}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestConvertResponsesResponseToAnthropic' -v`
Expected: FAIL — undefined converter.

- [x] **Step 3: Implement the converter and register the direction**

`responsesResponseToAnthropic` builds an `AnthropicResponse` with `Type:"message"`, `Role:"assistant"`, `Model: resolveResponseModel(resp.Model, opts)` and one block per output item:

- `message` content parts with `type:"output_text"` → text blocks.
- `reasoning` → thinking block; join `content[].text` where `type=="reasoning_text"`, else join `summary[].text`; emit with `"signature":""`.
- `function_call` → tool_use; `id` is `call_id` or the item ID; `input` is `parseResponsesArguments(arguments)` returning `map[string]any{}` for invalid/non-object JSON.
- Unknown item types → debug log and skip.

Use a local output envelope and block type so the empty signature is emitted only for thinking blocks, without changing the shared `AnthropicContent` tags:

```go
type anthropicOutBlock struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	Thinking  string         `json:"thinking,omitempty"`
	Signature *string        `json:"signature,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
}

type anthropicTranslatedResponse struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Role         string              `json:"role"`
	Content      []anthropicOutBlock `json:"content"`
	Model        string              `json:"model"`
	StopReason   *string             `json:"stop_reason"`
	StopSequence *string             `json:"stop_sequence"`
	Usage        AnthropicUsage      `json:"usage"`
}
```

`responsesStopReason` returns `tool_use` when any output item is a function call, `max_tokens` for incomplete `max_output_tokens`, `refusal` for incomplete `content_filter`, otherwise `end_turn`.

`anthropicUsageFromResponses` maps input minus cached minus cache-write (clamped at 0), cached to `CacheReadInputTokens`, cache-write to `CacheCreationInputTokens`, output to `OutputTokens`; a nil usage returns the zero value.

Register in `convert/registry.go`:

```go
conversions[ConversionKey{ProtocolOpenAIResponses, ProtocolAnthropic}] = convertResponsesResponseToAnthropic
```

The public function guards by body shape: if the body has no `"object":"response"`, return it unchanged.

- [x] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -run 'TestConvertResponsesResponseToAnthropic' -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add convert/anthropic_responses.go convert/registry.go convert/anthropic_responses_test.go
git commit -m "feat(convert): translate Responses replies to Anthropic"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `92fe052`. Files: `convert/anthropic_responses.go`, `convert/registry.go`, `convert/anthropic_responses_test.go`. Tests: `go test ./convert -run 'TestConvertResponsesResponseToAnthropic' -v` → PASS; `go test ./convert -count=1` → PASS. Review findings: none. Deferred: encrypted reasoning replay, web search and custom tools.

---

### Task 4: Route requests and non-streaming responses

**Files:**
- Modify: `convert/convert.go`
- Test: `convert/anthropic_responses_test.go` (routing tests)

**Interfaces:**
- Consumes: the two converters from Tasks 2 and 3; `Session{From, To}`.
- Produces: `Convert()` behavior; `Session.To` populated on the request path.

- [x] **Step 1: Write the failing tests**

```go
func TestConvertRoutesAnthropicToResponses(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{
		SID: "sid-1", Direction: "request", SessionStore: store,
		ModelMap: ModelMap{{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4.1-flash-1017", Protocol: "responses"}},
	}
	out, err := Convert([]byte(`{"model":"claude-opus-5-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`), opts)
	if err != nil { t.Fatal(err) }
	if strings.Contains(string(out), `"messages"`) || !strings.Contains(string(out), `"input"`) {
		t.Fatalf("not converted: %s", out)
	}
	sess := store.Get("sid-1")
	if sess == nil || sess.From != ProtocolAnthropic || sess.To != ProtocolOpenAIResponses {
		t.Fatalf("session = %+v", sess)
	}
}

func TestConvertRoutesResponsesReplyToAnthropicClient(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid-1", &Session{ID: "sid-1", From: ProtocolAnthropic, To: ProtocolOpenAIResponses})
	opts := &ConvertOptions{SID: "sid-1", Direction: "response", SessionStore: store, RequestModel: "claude-opus-5-5"}
	out, err := Convert([]byte(`{"id":"resp_1","object":"response","status":"completed","model":"deepseek-v4.1-flash-1017","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`), opts)
	if err != nil { t.Fatal(err) }
	if !strings.Contains(string(out), `"type":"message"`) || !strings.Contains(string(out), `"text":"ok"`) {
		t.Fatalf("not converted: %s", out)
	}
}

func TestConvertResponsesClientStillPassesThrough(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid-1", &Session{ID: "sid-1", From: ProtocolOpenAIResponses, IsResponses: true})
	body := `{"id":"resp_1","object":"response","status":"completed","output":[]}`
	out, _ := Convert([]byte(body), &ConvertOptions{SID: "sid-1", Direction: "response", SessionStore: store})
	if string(out) != body { t.Fatalf("passthrough changed body: %s", out) }
}

func TestConvertUnknownResponsesPairFails(t *testing.T) {
	opts := &ConvertOptions{
		Direction: "request",
		ModelMap:  ModelMap{{SourcePrefix: "gpt", TargetModel: "deepseek-v4.1-flash-1017", Protocol: "responses"}},
	}
	_, err := Convert([]byte(`{"model":"gpt-5","messages":[{"role":"user","content":"hi"}]}`), opts)
	if err == nil || !strings.Contains(err.Error(), "no converter") {
		t.Fatalf("err = %v, want a missing-converter diagnostic", err)
	}
}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestConvertRoutes|TestConvertResponsesClientStillPassesThrough|TestConvertUnknownResponsesPairFails' -v`
Expected: FAIL — request is not converted, the response returns raw Responses JSON, or the missing pair passes through.

- [x] **Step 3: Restructure `Convert()` and set `Session.To`**

In the request branch, store `&Session{ID: opts.SID, From: source, To: downstream}`.

In the response path, replace the early `if source == ProtocolOpenAIResponses { ... return body, nil }` behavior with this order:

1. If a session exists and `sess.IsResponses`, return `convertToResponsesResponse(body, opts)` (unchanged Responses-client behavior).
2. If `source == ProtocolOpenAIResponses` and `dir == DirectionResponse`, resolve `client` from the session or `detectByURI(opts.URI, DirectionRequest)`.
3. If `client == ProtocolOpenAIResponses`, return `passthrough`.
4. If `client != ProtocolUnknown` and `{Responses, client}` is registered, call it.
5. Otherwise passthrough.

Keep the request-side `convertResponsesRequest` path for Responses clients.

If the request's `downstream` is `ProtocolOpenAIResponses` and no converter is
registered for the source protocol, return
`fmt.Errorf("no converter for %s=>%s", source, downstream)` instead of
passthrough. This makes a missing Responses upstream pair loud, while leaving
other unknown-pair passthrough behavior unchanged.

- [x] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -run 'TestConvertRoutes|TestConvertResponsesClientStillPassesThrough|TestConvertUnknownResponsesPairFails' -v`
Expected: PASS.

- [x] **Step 5: Run the existing package tests for regressions**

Run: `go test ./convert -count=1`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add convert/convert.go convert/anthropic_responses_test.go
git commit -m "feat(convert): route Anthropic traffic to Responses upstreams"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `7638c43`. Files: `convert/convert.go`, `convert/anthropic_responses_test.go`. Tests: routing tests → PASS; `go test ./convert -count=1` → PASS. Ruling: the brief's routing bodies were not classifiable and the reply assertion matched raw Responses fields; added a system block/tool_choice marker and asserted the absence of `object:"response"` plus `stop_reason`. Review findings: none. Deferred: Chat/Gemini clients targeting Responses remain unsupported and now fail loudly.

---

### Task 5: Responses stream → Anthropic stream, text path

**Files:**
- Create: `convert/stream_responses_to_anthropic.go`
- Test: `convert/stream_responses_to_anthropic_test.go`

**Interfaces:**
- Consumes: `responsesStreamHandler`, `ResponsesStreamEvent`, `ConvertOptions`.
- Produces:
  - `func NewResponsesToAnthropicStreamConverter(model string, opts *ConvertOptions) *ResponsesToAnthropicStreamConverter`
  - methods `HandleStreamStart() []byte`, `HandleChunk(data []byte) ([]byte, error)`, `HandleStreamEnd() []byte`, `EmitError(message string) []byte`
  - `func anthropicSSE(event string, payload any) []byte`
  - `type responsesAnthropicBlock struct { index int; kind string; itemID string; args strings.Builder; argsSent bool; closed bool }`

```go
type ResponsesToAnthropicStreamConverter struct {
	model     string
	opts      *ConvertOptions
	started   bool
	finished  bool
	messageID string
	blockIndex int
	open      *responsesAnthropicBlock
	blocks    map[string]*responsesAnthropicBlock
	usage     AnthropicUsage
	stopReason string
	sawTool   bool
}
```

- [x] **Step 1: Write the failing test**

```go
func TestResponsesToAnthropicStreamText(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("claude-opus-5-5", &ConvertOptions{})
	var out strings.Builder
	out.Write(c.HandleStreamStart())
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"deepseek-v4.1-flash-1017"}}`,
		`{"type":"response.output_item.added","item":{"id":"msg_1","type":"message"},"output_index":0}`,
		`{"type":"response.content_part.added","item_id":"msg_1","part":{"type":"output_text","text":""}}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","delta":"Hello"}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","delta":" world"}`,
		`{"type":"response.output_text.done","item_id":"msg_1","text":"Hello world"}`,
		`{"type":"response.content_part.done","item_id":"msg_1"}`,
		`{"type":"response.output_item.done","item":{"id":"msg_1","type":"message"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"message"}],"usage":{"input_tokens":10,"output_tokens":2}}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil { t.Fatal(err) }
		out.Write(b)
	}
	out.Write(c.HandleStreamEnd())
	s := out.String()
	for _, want := range []string{"event: message_start", "event: content_block_start", `"type":"text"`, `"text_delta"`, `"text":"Hello"`, `"text":" world"`, "event: content_block_stop", `"stop_reason":"end_turn"`, `"output_tokens":2`, "event: message_stop"} {
		if !strings.Contains(s, want) { t.Fatalf("missing %s in %s", want, s) }
	}
}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `go test ./convert -run TestResponsesToAnthropicStreamText -v`
Expected: FAIL — undefined converter.

- [x] **Step 3: Implement the text path**

State: `started`, `finished`, `messageID`, `model`, `blockIndex`, `open *responsesAnthropicBlock`, `usage AnthropicUsage`, `stopReason string`.

`HandleChunk` unmarshals `ResponsesStreamEvent` and switches on `Type`:

- `response.created`: emit `message_start` with id/model, empty content, `usage.input_tokens` when present; mark started.
- `response.output_item.added` with item type `message`: no output.
- `response.content_part.added` with part type `output_text`: open a text block (`index`, `{"type":"text","text":""}`).
- `response.output_text.delta`: ensure a text block, emit `content_block_delta` with `text_delta`.
- `response.output_text.done` / `response.content_part.done` / `response.output_item.done` for the text item: close the block once.
- `response.completed`: close any open block, set usage and `end_turn`, emit `message_delta` + `message_stop`, mark finished.
- Other events: ignore for this task.

`HandleStreamEnd` closes an open block and, if not finished, emits `message_delta` + `message_stop` with the best-known values. `EmitError` emits `event: error` with `{"type":"error","error":{"type":"api_error","message":...}}`.

Use `anthropicSSE` to append `\n\n` between events.

- [x] **Step 4: Run the test to verify it passes**

Run: `go test ./convert -run TestResponsesToAnthropicStreamText -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add convert/stream_responses_to_anthropic.go convert/stream_responses_to_anthropic_test.go
git commit -m "feat(convert): stream Responses text as Anthropic events"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `8ac69de`. Files: `convert/stream_responses_to_anthropic.go` (new), `convert/stream_responses_to_anthropic_test.go` (new). Tests: `TestResponsesToAnthropicStreamText` → PASS; `go test ./convert -count=1` → PASS. Review findings: none. Deferred: tools/reasoning/errors are Task 6.

---

### Task 6: Stream tools, reasoning, parallel calls and failures

**Files:**
- Modify: `convert/stream_responses_to_anthropic.go`
- Test: `convert/stream_responses_to_anthropic_test.go`
- Test data: `convert/testdata/responses/*.sse`

**Interfaces:**
- Consumes: Task 5 converter.
- Produces: handling for `function_call`, `reasoning_text`/`reasoning_summary_text`, parallel items, `response.incomplete`/`response.failed`/`error`, and truncated arguments.

- [x] **Step 1: Write the failing tests**

```go
func TestResponsesToAnthropicStreamToolCall(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("claude-opus-5-5", &ConvertOptions{})
	var out strings.Builder
	out.Write(c.HandleStreamStart())
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"msg_1","type":"function_call","call_id":"toolu_1","name":"get_weather","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"msg_1","delta":"{\"city\":\"Beijing\"}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"msg_1","arguments":"{\"city\":\"Beijing\"}"}`,
		`{"type":"response.output_item.done","item":{"id":"msg_1","type":"function_call","call_id":"toolu_1","name":"get_weather","arguments":"{\"city\":\"Beijing\"}"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call"}]}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil { t.Fatal(err) }
		out.Write(b)
	}
	s := out.String()
	for _, want := range []string{`"type":"tool_use"`, `"id":"toolu_1"`, `"input_json_delta"`, `"partial_json":"{\"city\":\"Beijing\"}"`, `"stop_reason":"tool_use"`} {
		if !strings.Contains(s, want) { t.Fatalf("missing %s in %s", want, s) }
	}
}

func TestResponsesToAnthropicStreamReasoningText(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("claude-opus-5-5", &ConvertOptions{})
	var out strings.Builder
	out.Write(c.HandleStreamStart())
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"msg_r","type":"reasoning"}}`,
		`{"type":"response.reasoning_text.delta","item_id":"msg_r","delta":"think"}`,
		`{"type":"response.reasoning_text.done","item_id":"msg_r","text":"think"}`,
		`{"type":"response.output_item.done","item":{"id":"msg_r","type":"reasoning"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil { t.Fatal(err) }
		out.Write(b)
	}
	s := out.String()
	if !strings.Contains(s, `"type":"thinking"`) || !strings.Contains(s, `"signature":""`) || !strings.Contains(s, `"thinking_delta"`) || !strings.Contains(s, `"thinking":"think"`) {
		t.Fatalf("unexpected reasoning stream: %s", s)
	}
	if strings.Contains(s, "signature_delta") { t.Fatalf("unexpected signature_delta: %s", s) }
}

func TestResponsesToAnthropicStreamReasoningSummaryVariant(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.output_item.added","item":{"id":"r","type":"reasoning"}}`,
		`{"type":"response.reasoning_summary_text.delta","item_id":"r","delta":"summary"}`,
		`{"type":"response.reasoning_summary_text.done","item_id":"r","text":"summary"}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	if !strings.Contains(out.String(), `"thinking":"summary"`) { t.Fatalf("out = %s", out.String()) }
}

func TestResponsesToAnthropicStreamReasoningPartAdded(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.output_item.added","item":{"id":"r","type":"reasoning"}}`,
		`{"type":"response.content_part.added","item_id":"r","part":{"type":"reasoning_text","text":""}}`,
		`{"type":"response.reasoning_text.delta","item_id":"r","delta":"think"}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	s := out.String()
	if strings.Contains(s, `"type":"text"`) {
		t.Fatalf("reasoning content part opened a text block: %s", s)
	}
	if !strings.Contains(s, `"type":"thinking"`) || !strings.Contains(s, `"thinking":"think"`) {
		t.Fatalf("unexpected stream: %s", s)
	}
}

func TestResponsesToAnthropicStreamParallelCalls(t *testing.T) {
	// Two function_call items, each with its own arguments. Assert two
	// tool_use blocks in output order and both arguments complete. Use the
	// event sequence from convert/testdata/responses/parallel.sse.
}

func TestResponsesToAnthropicStreamInterleavedCalls(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a"}}`,
		`{"type":"response.output_item.added","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"a","delta":"{\"x\":1}"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"b","delta":"{\"y\":2}"}`,
		`{"type":"response.output_item.done","item":{"id":"a","type":"function_call","call_id":"call_a","name":"a","arguments":"{\"x\":1}"}}`,
		`{"type":"response.output_item.done","item":{"id":"b","type":"function_call","call_id":"call_b","name":"b","arguments":"{\"y\":2}"}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"function_call"}]}}`,
	} {
		b, err := c.HandleChunk([]byte(ev))
		if err != nil { t.Fatal(err) }
		out.Write(b)
	}
	s := out.String()
	if strings.Count(s, `"type":"tool_use"`) != 2 {
		t.Fatalf("want two tool blocks: %s", s)
	}
	if strings.Index(s, `"id":"call_a"`) > strings.Index(s, `"id":"call_b"`) {
		t.Fatalf("calls out of order: %s", s)
	}
	for _, want := range []string{`"partial_json":"{\"x\":1}"`, `"partial_json":"{\"y\":2}"`} {
		if !strings.Contains(s, want) { t.Fatalf("missing %s in %s", want, s) }
	}
}

func TestResponsesToAnthropicStreamTruncatedToolIsError(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{ErrorMsg: "unexpected EOF"})
	var out strings.Builder
	for _, ev := range []string{
		`{"type":"response.output_item.added","item":{"id":"t","type":"function_call","call_id":"toolu_1","name":"f"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"t","delta":"{\"city\":\"Bei"}`,
	} {
		b, _ := c.HandleChunk([]byte(ev))
		out.Write(b)
	}
	out.Write(c.HandleStreamEnd())
	s := out.String()
	if !strings.Contains(s, "event: error") || strings.Contains(s, "event: message_stop") {
		t.Fatalf("out = %s", s)
	}
}

func TestResponsesToAnthropicStreamFailedEvent(t *testing.T) {
	c := NewResponsesToAnthropicStreamConverter("m", &ConvertOptions{})
	_ = c.HandleStreamStart()
	out, _ := c.HandleChunk([]byte(`{"type":"response.failed","response":{"status":"failed","error":{"message":"upstream failed"}}}`))
	if !strings.Contains(string(out), "event: error") || !strings.Contains(string(out), "upstream failed") {
		t.Fatalf("out = %s", out)
	}
}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestResponsesToAnthropicStream(Tool|Reasoning|Parallel|Interleaved|Truncated|Failed)' -v`
Expected: FAIL — events are ignored or emitted as text.

- [x] **Step 3: Implement the remaining event handlers**

- `response.output_item.added` with `item.type=="function_call"`: close the open block, open a tool block with `id = call_id` (fallback item id), `name`, `input:{}`; record it by item ID.
- `response.function_call_arguments.delta`: find the item's block (or the open call); append to its buffer; emit `input_json_delta` when the delta is non-empty; mark `argsSent`.
- `response.function_call_arguments.done` / `response.output_item.done` for a function call: if no delta was sent, emit the whole `arguments` as one `input_json_delta`; close the block; set `sawTool`.
- `response.output_item.added` with `item.type=="reasoning"`: close the open block, open a thinking block with `{"type":"thinking","thinking":"","signature":""}`.
- `response.reasoning_text.delta`, `response.reasoning_summary_text.delta`, `response.reasoning_summary_part.added`: ensure the thinking block; emit `thinking_delta` for non-empty text.
- `response.reasoning_text.done`, `response.reasoning_summary_text.done`, `response.output_item.done` for reasoning: close the thinking block.
- Parallel calls: keep a `map[string]*responsesAnthropicBlock` by item ID and a queue of pending calls. When a new call starts while another is open, hold its events; release it after the open call closes, in output order.
- `response.completed`: close any open block; stop reason `tool_use` when `sawTool`, else `end_turn`; emit `message_delta` + `message_stop`.
- `response.incomplete`: close open block; stop reason `max_tokens` for `max_output_tokens`, `refusal` for `content_filter`, else `tool_use` when `sawTool`, else `end_turn`; emit `message_delta` + `message_stop`.
- `response.failed` / `error`: close open block; emit Anthropic `error`; mark finished.
- At end, if an open tool block's accumulated arguments are not valid JSON, call `EmitError("truncated tool arguments")` instead of closing normally.

Copy the captured streams into `convert/testdata/responses/`:

- `text.sse` from the Bailian text probe
- `tool.sse` from the Bailian tool probe
- `reasoning.sse` from the Bailian reasoning probe
- `parallel.sse` from the Bailian two-tool probe
- `deepseek-text.sse` from the DeepSeek official text probe (it emits
  `reasoning_text` even without a reasoning request and ends `response.incomplete`
  when the token cap is small)
- `deepseek-reasoning.sse` from the DeepSeek official reasoning probe
  (`deepseek-reasoner`, reasoning item followed by a message item)

If the captured files are unavailable, build the fixture from the same event
sequence the test feeds inline; do not block on a live upstream. Use the
fixtures in at least the tool, reasoning, parallel and DeepSeek tests.

DeepSeek's `response.content_part.added` may carry `part.type == "reasoning_text"`.
The converter must ignore that event for text purposes and keep the thinking
block it opened from `output_item.added`. Add a test that asserts no `text`
block is opened for a `reasoning_text` content part.

- [x] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -run 'TestResponsesToAnthropicStream' -v`
Expected: PASS (including the interleaved-call test).

- [x] **Step 5: Commit**

```bash
git add convert/stream_responses_to_anthropic.go convert/stream_responses_to_anthropic_test.go convert/testdata/responses
git commit -m "feat(convert): stream Responses tools and reasoning to Anthropic"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `fe84e0f`. Files: `convert/stream_responses_to_anthropic.go`, `convert/stream_responses_to_anthropic_test.go`, `convert/responses.go`, `convert/testdata/responses/*.sse`. Tests: all `TestResponsesToAnthropicStream*` → PASS, including interleaved calls, truncated args, failed events and the seven captured Bailian/DeepSeek fixtures; `go test ./convert -count=1` → PASS. Ruling: added internal-only `block.sent`, `pendingResponsesCall`, `ResponsesStreamEvent.Message` and reasoning-text constants to support interleaving, done-only text/reasoning and error messages. Review findings: none. Deferred: encrypted reasoning replay, web search/custom tools.

---

### Task 7: Wire the stream converter into the lifecycle

**Files:**
- Modify: `convert/convert.go`
- Test: `convert/anthropic_responses_test.go` (lifecycle tests)

**Interfaces:**
- Consumes: `NewResponsesToAnthropicStreamConverter`, `Session{From,To}`, `detectByURI`.
- Produces: `HandleSSEEvent` routing and end-phase error handling.

- [x] **Step 1: Write the failing tests**

```go
func TestHandleSSEEventResponsesToAnthropic(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid-1", &Session{ID:"sid-1", From: ProtocolAnthropic, To: ProtocolOpenAIResponses})
	opts := &ConvertOptions{SID:"sid-1", SessionStore: store, URI:"/v1/messages", Direction:"response"}
	start, err := HandleSSEEvent("sid-1", "start", 0, []byte(`data: {"type":"response.created","response":{"id":"resp_1"}}`), opts)
	if err != nil { t.Fatal(err) }
	if !strings.Contains(string(start), "message_start") { t.Fatalf("start = %s", start) }
	ev, _ := HandleSSEEvent("sid-1", "event", 1, []byte(`data: {"type":"response.output_text.delta","delta":"hi"}`), opts)
	if !strings.Contains(string(ev), "text_delta") { t.Fatalf("event = %s", ev) }
	end, _ := HandleSSEEvent("sid-1", "end", 0, nil, opts)
	if !strings.Contains(string(end), "message_stop") { t.Fatalf("end = %s", end) }
}

func TestHandleSSEEventMissingSessionUsesURI(t *testing.T) {
	opts := &ConvertOptions{SID:"missing", SessionStore: NewSessionStore(), URI:"/v1/messages", Direction:"response"}
	out, err := HandleSSEEvent("missing", "start", 0, []byte(`data: {"type":"response.created","response":{"id":"resp_1"}}`), opts)
	if err != nil { t.Fatal(err) }
	if !strings.Contains(string(out), "message_start") { t.Fatalf("out = %s", out) }
}

func TestHandleSSEEventEndWithErrorMsg(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid-1", &Session{ID:"sid-1", From: ProtocolAnthropic, To: ProtocolOpenAIResponses})
	opts := &ConvertOptions{SID:"sid-1", SessionStore: store, URI:"/v1/messages", Direction:"response", ErrorMsg:"unexpected EOF"}
	_, _ = HandleSSEEvent("sid-1", "start", 0, []byte(`data: {"type":"response.created","response":{"id":"resp_1"}}`), opts)
	out, _ := HandleSSEEvent("sid-1", "end", 0, nil, opts)
	if !strings.Contains(string(out), "event: error") || strings.Contains(string(out), "message_stop") {
		t.Fatalf("out = %s", out)
	}
}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestHandleSSEEvent(Responses|MissingSession|EndWithErrorMsg)' -v`
Expected: FAIL — raw Responses events are passed through or the converter is not created.

- [x] **Step 3: Add the lifecycle branch**

In `HandleSSEEvent` start:

- Resolve `target` from `Session.From`.
- If the session is absent or `From` is unknown, use `detectByURI(opts.URI, DirectionRequest)`.
- If `Session.To == ProtocolOpenAIResponses` and `target == ProtocolAnthropic`, force `from = ProtocolOpenAIResponses` before the `from == target` passthrough check. This also covers a first event that is only `{"type":"error"}`.
- Add a branch before the generic OpenAI converter:

```go
if from == ProtocolOpenAIResponses && target == ProtocolAnthropic {
	handler = NewResponsesToAnthropicStreamConverter(model, opts)
}
```

- Store the handler in `Session{From: target, To: from, StreamHandler: handler}`.
- Add `*ResponsesToAnthropicStreamConverter` to the end-phase `opts.ErrorMsg` type switch.
- On `event`, the existing handler lookup works unchanged.

- [x] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -run 'TestHandleSSEEvent(Responses|MissingSession|EndWithErrorMsg)' -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add convert/convert.go convert/anthropic_responses_test.go
git commit -m "feat(convert): route Responses streams to Anthropic clients"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `419a3dd`. Files: `convert/convert.go`, `convert/anthropic_responses_test.go`. Tests: lifecycle routing, missing-session URI fallback and error-phase tests → PASS; `go test ./convert -count=1` → PASS. Review findings: none. Deferred: a missing session plus a bare error event remains ambiguous; response.created is covered.

---

### Task 8: CLI parser, contract allowance and fixtures

**Files:**
- Modify: `engine/engine.go`
- Test: `engine/engine_test.go`
- Modify: `convert/contract_test.go`
- Test: `convert/anthropic_responses_test.go`
- Test data: `convert/testdata/responses/` (already created in Task 6)

**Interfaces:**
- Consumes: `ParseModelMap`, `validateAnthropicResponse`.
- Produces: `responses` accepted by `ParseModelMap`; `validateAnthropicResponseOpts(body []byte, allowEmptySignature bool) []string`.

- [x] **Step 1: Write the failing tests**

```go
func TestParseModelMapAcceptsResponses(t *testing.T) {
	mm := ParseModelMap("claude-opus=deepseek-v4.1-flash-1017:responses")
	if len(mm) != 1 || mm[0].Protocol != "responses" {
		t.Fatalf("model map = %+v", mm)
	}
}

func TestValidateAnthropicResponseTranslatedThinking(t *testing.T) {
	body := []byte(`{"type":"message","role":"assistant","model":"deepseek-v4.1-flash-1017","content":[{"type":"thinking","thinking":"x"}],"stop_reason":"end_turn"}`)
	if errs := validateAnthropicResponseOpts(body, true); len(errs) != 0 {
		t.Fatalf("translated response rejected: %v", errs)
	}
	if errs := validateAnthropicResponseOpts(body, false); len(errs) == 0 {
		t.Fatal("native validation accepted an empty signature")
	}
}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `go test ./engine -run TestParseModelMapAcceptsResponses -v && go test ./convert -run TestValidateAnthropicResponseTranslatedThinking -v`
Expected: FAIL — `responses` is cleared or the validator function is undefined.

- [x] **Step 3: Implement the parser case and validator option**

In `engine.ParseModelMap`, accept `"responses"` alongside `"openai"`, `"anthropic"` and `"gemini"`.

In `convert/contract_test.go`, keep `validateAnthropicResponse` as the strict wrapper and add:

```go
func validateAnthropicResponseOpts(body []byte, allowEmptySignature bool) []string
```

Only the thinking-signature check changes: skip it when `allowEmptySignature` is true.

- [x] **Step 4: Run the tests to verify they pass**

Run: `go test ./engine -run TestParseModelMapAcceptsResponses -v && go test ./convert -run TestValidateAnthropicResponseTranslatedThinking -v`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add engine/engine.go engine/engine_test.go convert/contract_test.go convert/anthropic_responses_test.go
git commit -m "test(convert): accept responses in the CLI map and translated thinking"
```

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `6fe797d`. Files: `engine/engine.go`, `engine/engine_test.go`, `convert/contract_test.go`, `convert/anthropic_responses_test.go`. Tests: `TestParseModelMapAcceptsResponses` → PASS; `TestValidateAnthropicResponseTranslatedThinking` → PASS; `go test ./convert ./engine -count=1` → PASS. Ruling: the brief's validator fixture used a DeepSeek model, which the native validator already allows; switched to `claude-opus-5-5` so the strict path rejects and the new allowance is exercised. Review findings: none. Deferred: fixtures were added in Task 6.

---

### Review fix pass (2026-11-09)

**Files:**
- Modify: `convert/stream_responses_to_anthropic.go`, `convert/convert.go`, `convert/anthropic_responses.go`, `convert/responses.go`
- Add: `convert/anthropic_responses_review_test.go`

**Interfaces:**
- Consumes: Tasks 1-8.
- Produces: fixes for the final whole-branch review findings.

- [x] **Step 1: Fix the review findings**

Critical:
1. Out-of-order interleaved calls hung in `flushPending` because the deferred
   call's `output_item.done` carried no top-level `item_id`, and the replay
   re-entered the pending gate while a tool block was open. The pending gate now
   resolves the item via `eventItemID` (top-level `item_id` or nested
   `item.id`), and `flushPending` refuses to replay while a block is open.
   `finish` closes the open block, drains deferred calls one at a time, and
   repeats until the queue is empty.
2. Zero-argument `tool_use` blocks omitted the required `input` object.
   `anthropicOutBlock.Input` is now a pointer, set only for tool blocks, so an
   empty map marshals as `{}`. `closeOpenForTerminal` emits the same `{}` for a
   stream tool block that never saw an arguments event.
3. Truncated tool arguments were validated only for the open block in the
   no-terminal-event path. `finish` now validates every tracked tool block
   before emitting `message_delta`/`message_stop`, using `toolArgsComplete`
   (JSON object) rather than `json.Valid`.
4. Events after a terminal event were not ignored. `HandleChunk` returns early
   when `c.finished` is set.
5. A bare `{"type":"error"}` first event with no session was classified as
   Anthropic and passed through raw. `isStreamErrorPayload` detects it and
   routes an Anthropic client to the Responses→Anthropic converter, which emits
   an Anthropic `error` event. The nested `error.message` is now read by
   `nestedErrorMessage`.

Tests added in `convert/anthropic_responses_review_test.go`:
- request-side parallel tool pairing and thinking+tool pairing;
- non-stream zero-argument tool `input`;
- stream done-only arguments, out-of-order interleaved completion, truncated
  tool arguments on `response.completed`, events after terminal, missing-session
  bare error;
- fixture contract assertions (block shapes, ids, stop reasons) plus
  `validateAnthropicSSE` on every fixture.

- [x] **Step 2: Run the full gate**

```bash
GOWORK=/tmp/opencode/ws/go.work go test -count=1 ./...
GOWORK=/tmp/opencode/ws/go.work go vet ./...
CGO_ENABLED=1 GOWORK=/tmp/opencode/ws/go.work go test -race -count=1 ./convert
```
All pass: `convert`, `engine`, `rewriter`, `tests/e2e`; `vet` clean; race clean.

**Implementation record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `675d3ad`. Files: `convert/stream_responses_to_anthropic.go`, `convert/convert.go`, `convert/anthropic_responses.go`, `convert/responses.go`, `convert/anthropic_responses_review_test.go`. Tests: the full gate above → PASS; the new review tests → PASS. Review findings: fixed the Critical hang, the zero-argument `input` omission, truncation validation for all tool blocks, post-terminal events, the missing-session bare error, and the missing review-focus tests. Deferred: `message_start.model` uses the engine default rather than the client model, README/comment updates for the new pair, empty `message_start.id` when `response.created` is absent, and backfilling a tool name from `function_call_arguments.done`.

**Follow-up record:** 2026-11-09 — complete. Repo `llm-api-converter`; code commit `0b8a0b2`. Files: `convert/convert.go`, `convert/stream_responses_to_anthropic.go`, `convert/anthropic_responses_review_test.go`. The re-grade found three small regressions from the first fix pass: the bare-error special case also rewrote legitimate Anthropic passthrough error types; `finish` closed an open tool block before truncation validation, emitting `content_block_stop` with partial JSON; and `ResponseFailed`/`"error"` no longer emitted `message_start`, which the project's own `validateAnthropicSSE` rejects. Fixes: the special case now requires `sess == nil || sess.To == ProtocolOpenAIResponses`; `finish` validates before closing and after each deferred call; the error paths emit `message_start` and read `ev.Response.Error` as a fallback. Added `TestHandleSSEEventAnthropicPassthroughError`, `TestResponsesToAnthropicStreamFirstEventFailed`, and `validateAnthropicSSE` assertions on the truncated and failed streams. Tests: the full gate (`go test -count=1 ./...`, `go vet ./...`, `CGO_ENABLED=1 go test -race -count=1 ./convert`) → PASS. Deferred: the four minor items listed above remain.

---

### Task 9: Full gate, release and cuttle verification

**Files:**
- Modify: `go.mod` / `go.sum` in `cuttle` (separate repository)
- No converter code changes unless the gate finds a defect.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: a release tag and a verified cuttle deployment.

- [x] **Step 1: Run the converter gate**

```bash
cd /root/code/go-gost/llm-api-converter
go test -count=1 ./...
go vet ./...
CGO_ENABLED=1 go test -race -count=1 ./convert
```
Expected: all pass. Fix and commit any failures before continuing.

- [x] **Step 2: Create the release candidate tag (human-gated push)**

```bash
git tag -a v0.6.0 -m "v0.6.0: Anthropic to Responses upstream"
```
Do not push the tag until the human approves. Record the tag and commit SHA in the implementation record.

- [x] **Step 3: Bump cuttle to the published tag**

After the tag is pushed and `v0.6.0` is available:

```bash
cd /root/code/go-gost/cuttle
GOWORK=off go get github.com/ginuerzh/llm-api-converter@v0.6.0
GOWORK=off go mod tidy
GOWORK=off go test ./... -count=1
```
Expected: tests pass. Commit:

```bash
git add go.mod go.sum
git commit -m "chore(deps): bump llm-api-converter to v0.6.0"
```

- [x] **Step 4: Verify against the live Bailian node (human-gated deploy)**

- Push cuttle `main` only with approval; wait for CI to publish the image.
- `kubectl -n llm rollout restart deploy/llm-gw` and wait for rollout.
- Back up `/data/cuttle.yaml`, pin `bailian-dsf` to `api: responses`, send `SIGHUP`.
- Send a Claude Code-shaped `/v1/messages` request with a tool call and a reasoning round.
- Confirm HTTP 200, `stats.protocol=responses`, a stream ending in `message_stop`, and the tool call executing.
- Repeat against DeepSeek official Responses (`https://api.deepseek.com/v1/responses`):
  - `deepseek-flash` text with a non-trivial `max_output_tokens` (the model
    may emit a reasoning item even without a reasoning request).
  - `deepseek-flash` tool call.
  - `deepseek-reasoner` reasoning followed by a text answer.
  - Confirm the Anthropic output has `thinking` blocks for reasoning, text
    blocks for the answer, `tool_use` for the call, and ends `message_stop`.
- Roll back by removing the pin and sending `SIGHUP` if anything fails.

- [x] **Step 5: Commit the cuttle dependency change and record both repositories**

```bash
cd /root/code/go-gost/cuttle
git log --oneline -1
```
Record the cuttle commit in the plan's implementation record. Do not push without approval.

**Implementation record:** 2026-11-09 — Steps 1-5 complete. Converter gate: `GOWORK=/tmp/opencode/ws/go.work go test -count=1 ./...` PASS (`convert`, `engine`, `rewriter`, `tests/e2e`), `go vet ./...` PASS, `CGO_ENABLED=1 GOWORK=/tmp/opencode/ws/go.work go test -race -count=1 ./convert` PASS. Release tag: `v0.6.0` at commit `e31a433`, pushed to `origin`. Cuttle dependency bump: `github.com/ginuerzh/llm-api-converter v0.6.0` in `go.mod`/`go.sum`; `GOWORK=off go test ./... -count=1` PASS; cuttle commit `d840f147` on `main`, pushed as part of `1ff93fe5`. CI: `docker` run `42722509502` succeeded; `ghcr.io/ginuerzh/cuttle:latest` published. Live verification: `/data/cuttle.yaml` backed up to `/data/cuttle.yaml.bak-20261109-pre-responses`; `bailian-dsf` pinned to `api: responses` and SIGHUP reloaded. Bailian (`claude-opus-4-5` → `deepseek-v4.1-flash-1017`): HTTP 200, `stats.protocol=responses`, upstream `token-plan.cn-beijing.maas.aliyuncs.com:443/compatible-mode/v1/responses`, SSE ended in `message_stop`, tool_use present, and a reasoning prompt produced a thinking block plus text. DeepSeek official Responses (`deepseek-dadax` → `api.deepseek.com/v1/responses`): `deepseek-flash` text produced thinking + text and `message_stop`; `deepseek-flash` tool call produced tool_use + text and `message_stop`; `deepseek-reasoner` produced thinking + text and `message_stop`; every row recorded `stats.protocol=responses`, HTTP 200. The temporary `deepseek-test` entrypoint and `Deepseek-dadax` responses pin/model change were removed after verification and the config was restored to the post-Bailian-pin state; SIGHUP reloaded. The `bailian-dsf` pin remains live. Two cuttle `provider_capabilities_test.go` fallback tests were changed to exercise the supported Anthropic→Responses path because v0.6.0 now fails loudly on Chat/Gemini→Responses; see cuttle commit `d840f147`.

---

## Implementation records

Each task above has an inline record. When the whole plan is complete, add a short summary here: converter tag, cuttle commit, gate output, live result, and any deferred work (Chat→Responses upstream, sealed reasoning, cache breakpoints).

**Summary:** Converter `v0.6.0` (`e31a433`, pushed); cuttle `d840f147` (pushed in `1ff93fe5`); gate green (`go test`, `go vet`, `go test -race`); live Bailian and DeepSeek official Responses verified with HTTP 200, `stats.protocol=responses`, streams ending in `message_stop`, and reasoning/tool-call blocks present. Deferred: Chat→Responses upstream, sealed reasoning, cache breakpoints.
