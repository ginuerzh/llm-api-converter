# Chat stream terminal markers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every passthrough LLM stream end with a protocol-correct terminal — `data: [DONE]` for a healthy OpenAI-chat stream, an in-band error chunk for a truncated one — so clients such as opencode stop failing turns with `OpenAI Chat stream ended without finish_reason`.

**Architecture:** The `PassthroughStreamHandler` in `llm-api-converter` gains per-stream terminal state (observed `choices[].finish_reason`, anthropic `message_stop`, or the upstream `[DONE]` sentinel) and one end-phase method that emits `[DONE]` when a terminal was seen and a protocol-shaped error when it was not. The SSE dispatcher stops swallowing the upstream sentinel silently (it marks state instead) and routes the end phase through that method for every passthrough stream. cuttle keeps its rewriter untouched: it only bumps the converter dependency and ships the change.

**Tech Stack:** Go 1.2x, `github.com/ginuerzh/llm-api-converter` (module tag `v0.7.2`), `github.com/ginuerzh/cuttle` (k8s deployment `llm/llm-gw`, image `ghcr.io/ginuerzh/cuttle:latest`).

**Spec:** `docs/superpowers/specs/2026-11-22-chat-stream-terminal-design.md` (this repository)

## Global Constraints

- A healthy OpenAI-chat passthrough stream ends with **exactly one** `data: [DONE]`, whether or not the upstream sent one.
- A truncated OpenAI-chat stream ends with the error chunk `{"error":{"message":"upstream stream ended without a terminal event","type":"stream_truncated","code":"stream_truncated"}}` and **no** `[DONE]`. With a transport error the message is `upstream stream ended without a terminal event: <error>`, capped at 512 characters.
- Anthropic-target streams: `event: error` on truncation, existing JSON shape `{"type":"error","error":{"type":"stream_error","message":…}}`, capacity for `[DONE]` is zero. Never synthesize `message_stop`, never synthesize `finish_reason`.
- Every emitted payload is `json.Marshal`-built (no string concatenation of user/error text).
- Untouched: `StreamConverter`, `AnthropicToOpenAIStreamConverter`, `GeminiTo…`/`…ToGemini` converters, `Responses*` handlers, `handleResponsesSSEEvent`, request-side conversion, cuttle's rewriter/observability.
- Converter release tag: `v0.7.2`. cuttle pins it in `go.mod`; the deployed image stays `ghcr.io/ginuerzh/cuttle:latest` and no `/data/cuttle.yaml` change ships.
- `git push` to `llm-api-converter` (main + tag) and `cuttle` (main) is in scope: the production deploy depends on CI building the image.
- Record the running pod's `imageID` digest before rollout; rollback is `kubectl -n llm set image deploy/llm-gw cuttle=ghcr.io/ginuerzh/cuttle@sha256:<digest>`.

## Review Focus

- Upstream ends with `finish_reason` + usage chunk but no `[DONE]` (the aliyun-bailian production shape): the client must still receive exactly one `[DONE]`.
- Anthropic-target passthrough must never emit a `[DONE]` line, and a healthy one must stay byte-identical.
- An upstream in-band `{"error":…}` chunk is already terminal: it must not be followed by `[DONE]`, nor be re-reported as a truncation.
- A tool-call turn (`finish_reason: "tool_calls"`) ends with exactly one `[DONE]` and untouched tool arguments.
- End phase with no session (upstream died before any chunk): stay silent, emit nothing.

---

### Task 1: Terminal state, protocol-aware errors, and end-phase output in `PassthroughStreamHandler`

**Files:**
- Modify: `convert/session.go` (`PassthroughStreamHandler` struct and methods, ~line 150-195)
- Test: `convert/session_terminal_test.go` (new)

**Interfaces:**
- Produces:
  - `func (p *PassthroughStreamHandler) markTerminal()`
  - `func (p *PassthroughStreamHandler) observeTerminal(data []byte)`
  - `func (p *PassthroughStreamHandler) EndStream(streamErr string) []byte`
  - `func (p *PassthroughStreamHandler) EmitError(message string) []byte` (now protocol-aware; anthropic shape unchanged)
  - struct fields: `terminal bool`

- [ ] **Step 1: Write the failing tests**

```go
package convert

import (
	"strings"
	"testing"
)

func chatHandler() *PassthroughStreamHandler {
	return NewPassthroughStreamHandler("m", ProtocolOpenAIChat, false)
}

const (
	chatFinishStop   = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	chatFinishTool   = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"
	chatContentNull  = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
	chatErrorChunk   = "data: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n"
	anthropicStop    = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	anthropicContent = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
)

func TestPassthroughChatEmitsDoneAfterFinishReason(t *testing.T) {
	h := chatHandler()
	if _, err := h.HandleChunk([]byte(chatFinishStop)); err != nil {
		t.Fatal(err)
	}
	if got := string(h.EndStream("")); got != "data: [DONE]" {
		t.Fatalf("end = %q, want \"data: [DONE]\"", got)
	}
}

func TestPassthroughChatTruncatedEmitsErrorOnly(t *testing.T) {
	h := chatHandler()
	if _, err := h.HandleChunk([]byte(chatContentNull)); err != nil {
		t.Fatal(err)
	}
	got := string(h.EndStream(""))
	if !strings.Contains(got, `"type":"stream_truncated"`) {
		t.Fatalf("end = %q, want a stream_truncated error", got)
	}
	if strings.Contains(got, "[DONE]") {
		t.Fatalf("end = %q, must not carry [DONE] after an error", got)
	}
	if !strings.HasPrefix(got, "data: {") {
		t.Fatalf("end = %q, want OpenAI chat data framing", got)
	}
}

func TestPassthroughChatTransportErrorIsCarried(t *testing.T) {
	h := chatHandler()
	got := string(h.EndStream("read tcp: i/o timeout"))
	if !strings.Contains(got, "i/o timeout") {
		t.Fatalf("end = %q, want the transport error in the message", got)
	}
}

func TestPassthroughChatSentinelCountsAsTerminal(t *testing.T) {
	h := chatHandler()
	h.markTerminal()
	if got := string(h.EndStream("")); got != "data: [DONE]" {
		t.Fatalf("end = %q, want \"data: [DONE]\"", got)
	}
}

func TestPassthroughChatErrorChunkIsTerminal(t *testing.T) {
	h := chatHandler()
	if _, err := h.HandleChunk([]byte(chatErrorChunk)); err != nil {
		t.Fatal(err)
	}
	if got := h.EndStream(""); got != nil {
		t.Fatalf("end = %q, want nil (in-band error already terminated the stream)", got)
	}
}

func TestPassthroughAnthropicHealthyNoDone(t *testing.T) {
	h := NewPassthroughStreamHandler("m", ProtocolAnthropic, false)
	if _, err := h.HandleChunk([]byte(anthropicStop)); err != nil {
		t.Fatal(err)
	}
	if got := h.EndStream(""); got != nil {
		t.Fatalf("end = %q, want nil for a healthy anthropic passthrough", got)
	}
}

func TestPassthroughAnthropicTruncatedEmitsEventError(t *testing.T) {
	h := NewPassthroughStreamHandler("m", ProtocolAnthropic, false)
	if _, err := h.HandleChunk([]byte(anthropicContent)); err != nil {
		t.Fatal(err)
	}
	got := string(h.EndStream(""))
	if !strings.HasPrefix(got, "event: error") || !strings.Contains(got, `"type":"stream_error"`) {
		t.Fatalf("end = %q, want an anthropic error event", got)
	}
	if strings.Contains(got, "[DONE]") || strings.Contains(got, "message_stop") {
		t.Fatalf("end = %q, must not fake a normal anthropic close", got)
	}
}

func TestPassthroughEmitErrorShapeFollowsProtocol(t *testing.T) {
	if got := string(chatHandler().EmitError(`a"b`)); !strings.HasPrefix(got, "data: {") || !strings.Contains(got, `a\"b`) {
		t.Fatalf("chat error = %q", got)
	}
	anthropic := NewPassthroughStreamHandler("m", ProtocolAnthropic, false)
	if got := string(anthropic.EmitError(`a"b`)); !strings.HasPrefix(got, "event: error") || !strings.Contains(got, `a\"b`) {
		t.Fatalf("anthropic error = %q", got)
	}
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestPassthrough' -count=1`
Expected: FAIL — `h.EndStream undefined`, `h.markTerminal undefined`.

- [ ] **Step 3: Implement in `convert/session.go`**

Add `terminal bool` to `PassthroughStreamHandler`. Detection:

- `markTerminal()` sets `terminal = true`.
- `observeTerminal(data []byte)`: extract the payload with `extractSSEPayload`; a `nil` payload means the upstream `[DONE]` sentinel → `markTerminal()`. For `ProtocolOpenAIChat` unmarshal `{Choices []struct{ FinishReason *string }}` and mark on a non-blank `finish_reason`; for `ProtocolAnthropic` unmarshal `{Type string}` and mark on `type == "message_stop"`. Persisted tool-call chunks (`finish_reason: "tool_calls"`) count like any other reason.
- Call `p.observeTerminal(data)` at the top of `HandleChunk` before dispatching to `openaiPassthrough`/`anthropicPassthrough`.
- Make `EmitError` protocol-aware: `ProtocolOpenAIChat` returns `data: ` + `json.Marshal(map[string]any{"error": map[string]any{"message": message, "type": "stream_truncated", "code": "stream_truncated"}})`. Any other protocol keeps today's `event: error` + `{"type":"error","error":{"type":"stream_error","message":…}}` shape.
- `EndStream(streamErr string) []byte`: if `terminal` → for `ProtocolOpenAIChat` return `[]byte("data: [DONE]")`, otherwise `nil`. If not terminal → build the message (`"upstream stream ended without a terminal event"`, plus `": "+streamErr` when non-empty, truncated to 512 bytes), return `EmitError(msg)` and nothing else.
- An in-band upstream error chunk (`{"error":…}`, `type: "error"` for anthropic) marks the stream terminal so the truncation policy does not double-report it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./convert -run 'TestPassthrough' -count=1`
Expected: PASS, no other test regressions (`go test ./... -count=1`).

- [ ] **Step 5: Commit**

```bash
git add convert/session.go convert/session_terminal_test.go
git commit -m "fix(convert): track passthrough terminal state and emit protocol-shaped ends"
```

**Implementation record:** 2026-11-22 · **complete** · `llm-api-converter` `08cf393` · files: `convert/session.go`, `convert/session_terminal_test.go` · tests: `go test ./convert -run TestPassthrough -count=1` RED (`EndStream`/`markTerminal` undefined) → GREEN 8/8; `go vet ./...` clean; `go test $(go list ./... | grep -v /tests/e2e) -count=1` ok (convert/engine/rewriter) · review findings and fixes: the in-band-error test caught `EndStream` emitting `[DONE]` after an error → added `errored bool`; `HandleStreamEnd` kept for the `responsesStreamHandler` contract, `EndStream` is the new end-phase entry point · deferred: `tests/e2e` `TestRewriterE2E_Streaming` fails identically at BASE (pre-existing environment failure, excluded from CI).

---

### Task 2: Wire the end phase and the sentinel through `HandleSSEEvent`

**Files:**
- Modify: `convert/convert.go` (`StreamPhaseEvent` sentinel branch ~line 1082; `StreamPhaseEnd` ~line 1150)
- Modify: `convert/anthropic_openai_p1_test.go:901` (`TestP1_PassthroughEndPhaseIgnoresError` — the behaviour it pins is what this task changes)
- Test: `convert/sse_terminal_test.go` (new)

**Interfaces:**
- Consumes: `(*PassthroughStreamHandler).markTerminal()`, `.EndStream(streamErr string) []byte` from Task 1.
- Produces: no new exported API; `HandleSSEEvent(sid, phase, eventIndex string/int, data []byte, opts *ConvertOptions)` end phase now returns `[DONE]`/error bytes for passthrough sessions.

- [ ] **Step 1: Write the failing tests**

```go
package convert

import (
	"strings"
	"testing"
)

func chatOpts(store *SessionStore) *ConvertOptions {
	return &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/chat/completions", SID: "s1"}
}

// The production shape: finish_reason + a usage-only chunk, upstream [DONE] absent.
func TestHandleSSEEvent_ChatPassthroughNoUpstreamDoneStillEndsWithDone(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	mustEvent(t, "s1", "event", chatFinishStop, opts)
	mustEvent(t, "s1", "event", "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}", opts)
	end, err := HandleSSEEvent("s1", "end", 0, nil, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(end); got != "data: [DONE]" {
		t.Fatalf("end = %q, want \"data: [DONE]\"", got)
	}
}

func TestHandleSSEEvent_ChatPassthroughUpstreamDoneEmitsOnce(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	mustEvent(t, "s1", "event", chatFinishTool, opts)
	if mid, err := HandleSSEEvent("s1", "event", 2, []byte("data: [DONE]"), opts); err != nil || len(mid) != 0 {
		t.Fatalf("sentinel event = %q, %v; want swallowed", mid, err)
	}
	end, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	if got := string(end); got != "data: [DONE]" {
		t.Fatalf("end = %q, want exactly one \"data: [DONE]\"", got)
	}
}

func TestHandleSSEEvent_ChatPassthroughTruncatedReportsError(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	mustEvent(t, "s1", "event", chatContentNull, opts)
	end, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	got := string(end)
	if !strings.Contains(got, `"type":"stream_truncated"`) || strings.Contains(got, "[DONE]") {
		t.Fatalf("end = %q, want a stream_truncated error and no [DONE]", got)
	}
}

func TestHandleSSEEvent_ChatPassthroughTransportErrorCarried(t *testing.T) {
	store := NewSessionStore()
	opts := chatOpts(store)
	mustEvent(t, "s1", "start", "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}", opts)
	opts.ErrorMsg = "read tcp: i/o timeout"
	end, _ := HandleSSEEvent("s1", "end", 0, nil, opts)
	if !strings.Contains(string(end), "i/o timeout") {
		t.Fatalf("end = %q, want the transport error carried", end)
	}
}

func TestHandleSSEEvent_AnthropicPassthroughNeverGetsDone(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/messages", SID: "a1"}
	mustEvent(t, "a1", "start", "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude\"}}", opts)
	mustEvent(t, "a1", "event", anthropicStop, opts)
	end, _ := HandleSSEEvent("a1", "end", 0, nil, opts)
	if len(end) != 0 {
		t.Fatalf("end = %q, want nothing for a healthy anthropic passthrough", end)
	}
}

func TestHandleSSEEvent_AnthropicPassthroughTruncatedEmitsError(t *testing.T) {
	store := NewSessionStore()
	opts := &ConvertOptions{SessionStore: store, Direction: "response", URI: "/v1/messages", SID: "a2"}
	mustEvent(t, "a2", "start", "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude\"}}", opts)
	mustEvent(t, "a2", "event", anthropicContent, opts)
	end, _ := HandleSSEEvent("a2", "end", 0, nil, opts)
	if !strings.HasPrefix(string(end), "event: error") {
		t.Fatalf("end = %q, want an anthropic error event", end)
	}
}

func TestHandleSSEEvent_EndWithoutSessionStaysSilent(t *testing.T) {
	opts := chatOpts(NewSessionStore())
	end, err := HandleSSEEvent("nope", "end", 0, nil, opts)
	if err != nil || len(end) != 0 {
		t.Fatalf("end = %q, %v; want silence", end, err)
	}
}

func mustEvent(t *testing.T, sid, phase, data string, opts *ConvertOptions) []byte {
	t.Helper()
	out, err := HandleSSEEvent(sid, phase, 0, []byte(data), opts)
	if err != nil {
		t.Fatalf("%s %s: %v", sid, phase, err)
	}
	return out
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./convert -run 'TestHandleSSEEvent_' -count=1`
Expected: FAIL — the truncation and `[DONE]` cases return empty output.

- [ ] **Step 3: Implement the two dispatch changes in `convert/convert.go`**

- `StreamPhaseEvent`: in the `payload == nil` branch (currently `return []byte{}, nil // [DONE] marker, swallow.`), first look up `store.Get(sid)` and, when the handler is a `*PassthroughStreamHandler`, call `markTerminal()`. Keep returning empty bytes: the sentinel is re-emitted by the end phase so every chat stream ends with exactly one.
- `StreamPhaseEnd`: handle passthrough **before** the `opts.ErrorMsg` switch — `if ph, ok := handler.(*PassthroughStreamHandler); ok { return ph.EndStream(opts.ErrorMsg), nil }`. Delete the trailing `if _, ok := handler.(*PassthroughStreamHandler); ok { return nil, nil }` branch and its comment. Leave the `opts.ErrorMsg` `EmitError` switch for the stateful converters exactly as is.
- Do not touch the phase-less fallback at `engine/engine.go:148`: without `sse_phase` metadata there is no end phase to emit from, so that path keeps today's behaviour. x's SSE sniffer always sets the phases, which is the path production uses.

- [ ] **Step 4: Update the pinned test and run the suite**

`convert/anthropic_openai_p1_test.go:901` (`TestP1_PassthroughEndPhaseIgnoresError`) asserts the old silence: rename it to `TestP1_PassthroughEndPhaseReportsStreamError` and assert the end phase returns an `event: error` carrying `upstream reset`.

Run: `CGO_ENABLED=1 go test ./... -count=1 -race && go vet ./...`
Expected: PASS. `convert/audit_test.go:512` and `convert/contract_test.go` must stay green: a healthy stream still carries no `[DONE]` for anthropic clients.

- [ ] **Step 5: Commit**

```bash
git add convert/convert.go convert/anthropic_openai_p1_test.go convert/sse_terminal_test.go
git commit -m "fix(convert): emit [DONE] and report truncation for passthrough streams"
```

**Implementation record:** 2026-11-22 · **complete** · `llm-api-converter` `4bf0608` · files: `convert/convert.go` (sentinel marks the passthrough handler terminal; end phase delegates to `EndStream`), `convert/sse_terminal_test.go` (new, 8 tests), `convert/anthropic_openai_p1_test.go` (`TestP1_PassthroughEndPhaseIgnoresError` → `TestP1_PassthroughEndPhaseReportsStreamError`), `convert/convert_test.go` (`TestHandleSSEEvent_AnthropicPassthroughStreaming` now sends `message_stop` so it keeps pinning the healthy passthrough) · tests: new tests RED (5 of 8 failing: end phase returned empty) → GREEN; `go vet ./...` clean; `CGO_ENABLED=1 go test $(go list ./... | grep -v /tests/e2e) -race` ok · review findings and fixes: a second pre-existing test pinned the removed silence and had to be updated (`convert_test.go:2413`) · deferred: none.

---

### Task 3: Release `llm-api-converter` v0.7.2

**Files:**
- Modify: `docs/superpowers/specs/2026-11-22-chat-stream-terminal-design.md` (Status: approved)

**Interfaces:**
- Produces: module tag `v0.7.2` that Task 4 pins.

- [ ] **Step 1: Full verification before tagging**

Run: `CGO_ENABLED=1 go test ./... -count=1 -race && go vet ./...`
Expected: PASS. Note the output in the record.

- [ ] **Step 2: Commit the approved spec, tag, and push**

```bash
git add docs/superpowers/specs/2026-11-22-chat-stream-terminal-design.md
git commit -m "docs: approve chat stream terminal marker design"
git tag -a v0.7.2 -m "v0.7.2: chat stream terminal markers"
git push origin main && git push origin v0.7.2
git describe --tags    # v0.7.2
```

Expected: tag resolves; `buildx.yml` runs green for the tag.

**Implementation record:** 2026-11-22 · **complete** · `llm-api-converter` `3fa6159` + tag `v0.7.2` · files: `docs/superpowers/specs/2026-11-22-chat-stream-terminal-design.md` (Status → approved) · tests: `CGO_ENABLED=1 go test $(go list ./... | grep -v /tests/e2e) -race` → convert/engine/rewriter ok · verification: `git push origin main` (`7f82597..3fa6159`) and `git push origin v0.7.2` accepted; `go list -m -versions` shows `v0.7.2` and `go mod download` fetches it through goproxy.cn; `buildx.yml` runs `44978600212` (tag) and `44978591284` (main) started · review findings: none · deferred: none.

---

### Task 4: Bump cuttle and pin the behaviour with gateway tests

**Files:**
- Modify: `cuttle/go.mod`, `cuttle/go.sum` (`github.com/ginuerzh/llm-api-converter` → `v0.7.2`)
- Test: `cuttle/internal/gateway/stream_terminal_test.go` (new)

**Interfaces:**
- Consumes: converter `v0.7.2` behaviour from Tasks 1–2.
- Produces: no new cuttle API.

- [ ] **Step 1: Bump the dependency**

```bash
cd /root/code/go-gost/cuttle
GOFLAGS=-mod=mod go get github.com/ginuerzh/llm-api-converter@v0.7.2 && go mod tidy
go build ./... && go test ./internal/gateway -count=1
```

If the proxy has not seen the tag yet: retry after a minute, then `GOPROXY=direct` as the fallback. Expected: builds and the gateway package passes.

- [ ] **Step 2: Write the failing gateway tests**

Follow the existing helpers: `testConfig(up.URL)`, `startGateway(t, cfg)`, `testUpstream(...)`, and the `openAIChatStream` constant in `internal/gateway/provider_capabilities_test.go`.

- `TestChatStreamEndsWithDone`: upstream streams role → content → `finish_reason:"stop"` + usage and returns **without** `data: [DONE]` (the bailian shape). POST `{"model":"claude-sonnet-4-5","stream":true,"messages":[{"role":"user","content":"hi"}]}` to `/v1/chat/completions`, read the whole body. Assert `strings.Count(body, "data: [DONE]") == 1`, `strings.Contains(body, `"finish_reason":"stop"`)`, and that `"type":"stream_truncated"` is absent.
- `TestChatTruncatedStreamReportsError`: upstream streams role → content chunk with `finish_reason:null` and returns. Assert the body contains `"type":"stream_truncated"`, contains the phrase `upstream stream ended without a terminal event`, and contains no `[DONE]`.
- Both tests set `Accept-Encoding: gzip` on the client request so the gateway's strip-and-forward behaviour is exercised with the new end-phase output.

- [ ] **Step 3: Run the tests to verify they fail, then pass**

Run: `go test ./internal/gateway -run 'TestChatStream' -count=1 -v`
Expected before the bump is picked up: FAIL (`[DONE]` count 0). After the bump: PASS. Then `go vet ./... && go test ./... -count=1`.

- [ ] **Step 4: Review the accumulated deploy diff**

Run: `git log --oneline 2f6fcb24..HEAD` and `git diff --stat 2f6fcb24..HEAD | tail -3`.
Expected: the deployed pod runs `2f6fcb24` (converter v0.5.2); this deploy ships every cuttle commit since then. Read the list for anything needing a config or rollout note, and record it.

- [ ] **Step 5: Commit**

```bash
git add go.mod go.sum internal/gateway/stream_terminal_test.go
git commit -m "fix(gateway): pin converter v0.7.2 and cover chat stream terminals"
```

**Implementation record:** as in Task 1, with the cuttle commit SHA and the accumulated-diff note.

---

### Task 5: Deploy to production and verify

**Files:** none (deploy + records only)

- [ ] **Step 1: Record the rollback point**

Run: `kubectl -n llm get pod -l app=llm-gw -o jsonpath='{.items[0].status.containerStatuses[0].imageID}'`
Expected: `...@sha256:...`; write the digest into this task's record before continuing.

- [ ] **Step 2: Push and build the image**

```bash
git push origin main
gh run list --workflow=docker.yml --limit 1   # wait for the push run
gh run watch <id>                              # or poll until success
```

Expected: `ghcr.io/ginuerzh/cuttle:latest` rebuilt from the new commit.

- [ ] **Step 3: Roll out**

```bash
kubectl -n llm rollout restart deploy/llm-gw
kubectl -n llm rollout status deploy/llm-gw --timeout=180s
kubectl -n llm get pod -l app=llm-gw -o jsonpath='{.items[0].status.containerStatuses[0].imageID}'
```

Expected: new pod `Running`, digest differs from Step 1.

- [ ] **Step 4: Verify the wire**

```bash
KEY=$(cat /root/.config/opencode/cuttle.key)
curl -sS -N --max-time 60 http://llm-api.home.pi/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"claude-opus-5-5","stream":true,"stream_options":{"include_usage":true},"max_tokens":24,"messages":[{"role":"user","content":"count 1 to 5"}]}' > /tmp/opencode/after.txt
grep -c 'DONE' /tmp/opencode/after.txt          # 1
tail -c 200 /tmp/opencode/after.txt             # ends with data: [DONE]
```

Repeat once with a tool-bearing request (any real opencode turn) and confirm the tail holds `finish_reason":"tool_calls"` **and** a single `data: [DONE]`.

- [ ] **Step 5: Verify no regression in stats**

After ~30 minutes, query the production stats DB the way the investigation did (temporary `sqlite-probe` pod on the `pvc-llm-gw-data` claim, `sqlite3 -readonly /data/stats.sqlite`, delete the pod afterwards):

```sql
select node, client_model, count(*) n, sum(truncated) t,
       sum(case when client_status>=400 then 1 else 0 end) errs
from requests where ts > (strftime('%s','now')-1800)*1000
group by node, client_model order by n desc;
```

Expected: no new `error_class`, `finish_reason`/`saw_done` distribution unchanged, `truncated` still recorded when an upstream cuts.

- [ ] **Step 6: Record and report**

Write the record (digests before/after, CI run id, curl evidence, stats snapshot) and tell the human partner that the win10 opencode client can confirm on their next long turn.

Rollback, if needed: `kubectl -n llm set image deploy/llm-gw cuttle=ghcr.io/ginuerzh/cuttle@sha256:<Step 1 digest>`.

**Implementation record:** as in Task 1, plus the deploy evidence above.

---

## Self-review notes

- Spec coverage: A (truncation reporting) = Tasks 1–2; B (`[DONE]`) = Tasks 1–2; release = Task 3; cuttle = Task 4; deploy + rollback = Task 5. Deferred by spec and not in this plan: truncation retry/failover, cuttle stats reclassification, Gemini-target passthrough.
- Type consistency: `markTerminal`, `observeTerminal`, `EndStream(streamErr string) []byte` are defined once in Task 1 and used verbatim in Task 2.
- Review Focus lines 1–5 each map to a test: line 1 → Task 2 Step 1 first test + Task 4; line 2 → Task 2 `AnthropicPassthroughNeverGetsDone`; line 3 → Task 1 `ErrorChunkIsTerminal`; line 4 → Task 2 `UpstreamDoneEmitsOnce` + Task 5 Step 4; line 5 → Task 2 `EndWithoutSessionStaysSilent`.