# Chat stream terminal markers: emit `[DONE]`, report truncation

Date: 2026-11-22
Status: approved (2026-11-22)
Repositories: `llm-api-converter` (v0.7.2), `cuttle` (dependency bump + production deploy)
Driven by: production report — opencode v2 fails turns with
`OpenAI Chat stream ended without finish_reason` when talking to the cuttle
gateway.

## Intent

Two defects in the OpenAI-chat passthrough path make a truncated upstream
stream indistinguishable from a complete one, and make even healthy streams
non-compliant:

1. `PassthroughStreamHandler` never reports a missing terminal marker. Its
   `HandleStreamEnd()` returns `nil`, and `StreamPhaseEnd`'s `EmitError`
   switch omits `*PassthroughStreamHandler`, so neither a clean EOF without
   `finish_reason` (no `stream_error` metadata) nor a transport-level cut
   (metadata present) produces any client-visible event.
2. The upstream `data: [DONE]` sentinel is swallowed
   (`extractSSEPayload` → `nil`), and the passthrough end phase emits nothing,
   so chat clients never receive `[DONE]` — verified on the wire: aliyun-bailian
   sends `data: [DONE]`, the response through cuttle ends at the usage chunk.

Clients that check for a terminal marker therefore only see a bare EOF.
opencode v2 (openai-compatible provider, `requireFinishReason` defaults to
`true`) reports `InvalidProviderOutput / incomplete-stream` with the message in
the title, and the whole turn is lost.

Production evidence (cuttle `stats.sqlite`, 7-day window):

| when (UTC) | node | client model | shape |
|---|---|---|---|
| 11-21 12:46 | cc-dsf (CommandCode) | claude-sonnet-5-5 | upstream read timeout after 72 s, `error_class=upstream_error`, nothing sent to the client |
| 11-21 14:56 | cc-dsf | claude-sonnet-5-5 | clean EOF, 67.7 s / 391 KB, no `finish_reason`, no `[DONE]`, no transport error |
| 11-18/19 | oc-go-glm (OpenCode Go) | claude-opus-5-5 | 120 of 142 streams truncated while `claude-opus` was routed to the `glm` route |
| other days | — | — | 0–1 per day |

## Goals

- An OpenAI-chat response stream that reaches a terminal marker always ends
  with exactly one `data: [DONE]`, whether or not the upstream sent the
  sentinel.
- An OpenAI-chat response stream that ends without a terminal marker — clean
  EOF or transport error — ends with an in-band error event and no `[DONE]`
  (an OpenAI stream terminates with either `[DONE]` or an error, never both),
  so the client reports a precise upstream-truncation error instead of
  "no finish_reason".
- Anthropic-target passthrough streams follow the same policy with their own
  shape: `event: error` on truncation, and never a synthesized `message_stop`.
- The transport error, when present, is carried into the emitted message.

## Non-goals

- Synthesizing `finish_reason: "stop"` (a truncation must not look complete).
- Failover/retry of a truncated stream (separate idea; only useful before the
  first byte reaches the client).
- Reclassifying these rows in cuttle's stats as `upstream_truncated` so the
  existing alert fires — real gap, deliberately deferred as a follow-up.
- Gemini-target passthrough (`GeminiStreamHandler`), Responses conversions, and
  the stateful converters (`StreamConverter`,
  `AnthropicToOpenAIStreamConverter`, …) that already synthesize terminals or
  emit errors.
- Any change to request-side conversion.

Anti-goal: hiding truncation. A half-written tool call must stay an error.

## Constraints

- x's SSE sniffer calls the rewriter with `sse_phase=end` on clean EOF (no
  `stream_error`) and on scanner error with `stream_error` set; the converter's
  end phase is the only hook for both.
- `[DONE]` is consumed before the handler sees it, so terminal tracking needs an
  explicit signal from the event phase.
- Response bodies are also observed by cuttle's stats sink from the raw upstream
  bytes, so `saw_done`/`finish_reason` in stats stay correct regardless of what
  we now emit to clients.
- Deployed cuttle (`2f6fcb24`) pins converter v0.5.2, while cuttle HEAD already
  pins v0.7.1: shipping this change deploys v0.5.2 → v0.7.2 plus every cuttle
  change since `2f6fcb24`. That accumulated diff needs its own pass
  (`go build`/`go test`, and a read of `git log 2f6fcb24..HEAD`) before rollout.
- opencode maps an in-band chat `error` object to a provider error and shows its
  `message`, which is why the envelope is `{"error":{...}}` with a string
  `code`.

## Approaches considered

1. **Synthesize a safe terminal** (`finish_reason: "stop"` + `[DONE]` on
   truncation). Rejected: it reports partial output — including half-written
   tool arguments — as a successful turn.
2. **Report truncation, always terminate chat streams with `[DONE]`** (chosen):
   protocol-honest, keeps the partial text on the wire, and gives the client a
   precise error.
3. **Fix it in cuttle's rewriter instead of the converter.** Rejected: cuttle's
   rewriter owns observation and model rewriting, has no per-stream protocol
   state, and duplicating stream policy there would drift from the converter.
4. **Retry the truncated stream on another node.** Deferred (non-goal).

## Decisions

Mine, unless corrected:

- The end phase is the single emission point for `[DONE]`; the upstream sentinel
  stays swallowed in the event phase (no de-duplication needed, ordering is
  preserved). Lower risk than changing the event path every protocol shares.
- Truncation is detected per stream in `PassthroughStreamHandler`:
  - chat target: a chunk with a non-empty `choices[].finish_reason`, or the
    upstream `[DONE]`;
  - anthropic target: a `message_stop` event, or the upstream `[DONE]`.
- Chat truncation emits only
  `data: {"error":{"message":"upstream stream ended without a terminal event","type":"stream_truncated","code":"stream_truncated"}}`
  — no `[DONE]` after it: `validateOpenAISSE` in `contract_test.go` treats an
  in-band error chunk as terminal and flags any event after it, and OpenAI
  clients follow the same rule. With `stream_error` present the message reads
  `upstream stream ended without a terminal event: <sanitized error>`.
- Anthropic truncation emits the existing `event: error` +
  `{"type":"error","error":{"type":"stream_error","message":…}}` and no
  `[DONE]`. `EmitError` becomes protocol-aware (it is anthropic-only today).
- Message text is JSON-encoded (never string-concatenated) and capped, matching
  the existing `EmitError` style.
- `PassthroughStreamHandler` is added to `StreamPhaseEnd`'s `EmitError` type
  switch, so a transport cut is reported on both targets.
- Release as converter v0.7.2; cuttle bumps `go.mod` and redeploys `:latest`.

Left to the builder: helper placement and naming, whether sentinel tracking is a
`MarkSentinel()` method on the handler or a `sentinel` field set from the event
phase, the exact test table, and whether a `docs/plans` entry is shared or split
between the two repositories.

## Verification

Converter (local):

- table-driven tests: healthy chat → single `[DONE]`; chat EOF without
  `finish_reason` → error event + `[DONE]`; chat with `stream_error` → message
  carries it; upstream `[DONE]` only → still exactly one `[DONE]` emitted;
  healthy anthropic → bytes unchanged and no `[DONE]`; truncated anthropic →
  `event: error`; existing conversion suites green;
  `CGO_ENABLED=1 go test -race ./...` and `go vet ./...`.

cuttle (local):

- bump, `go build ./...`, `go test ./...`;
- gateway-level test with a fake upstream that cuts mid-stream: client body ends
  with the error event; healthy upstream: body ends with exactly one
  `data: [DONE]`.

Production (after rollout):

- `curl -N` the `claude-opus` entrypoint: tail must contain `data: [DONE]`;
  with `max_tokens` low enough to hit `finish_reason: length`, and on a
  tool-call turn, `finish_reason: tool_calls`.
- `requests` for the following hour: no new `error_class`, `finish_reason` /
  `saw_done` distributions unchanged, `truncated` still recorded when an
  upstream cuts.
- opencode on the win10 client: a normal long turn completes; the truncation
  error, if it happens, now reads as an upstream truncation.

## Rollout and rollback

- Record the running pod's `imageID` digest before touching the deployment.
- cuttle: push to `main` → `docker` workflow publishes
  `ghcr.io/ginuerzh/cuttle:latest` → `kubectl -n llm rollout restart
  deploy/llm-gw` → `kubectl -n llm rollout status`.
- Rollback: `kubectl -n llm set image deploy/llm-gw
  cuttle=ghcr.io/ginuerzh/cuttle@sha256:<recorded digest>` (the `latest` +
  `Always` policy makes `rollout undo` insufficient on its own). Rollback copies
  of `/data/cuttle.yaml` are untouched: this change ships no config change.