# Fix: non-streaming chunked LLM responses bypass the Rewriter plugin

## Context

Claude Code's auto-mode safety classifier fires `claude-opus-4-8[1m] is temporarily unavailable`. Root-caused to the GOST Rewriter plugin path:

- The classifier makes a **non-streaming** Anthropic API call through the GOST proxy (`llm.home.pi`) backed by the `llm-api-converter` plugin.
- The upstream (DeepSeek) returns the response as **HTTP chunked** — `Transfer-Encoding: chunked`, no `Content-Length`. Go's HTTP client de-chunks the body stream, but sets `resp.ContentLength = -1` to signal "unknown length".
- In [`newRewriteBody`](x/internal/util/forwarder/sniffer_sse.go#L82) the guard at [line 92](x/internal/util/forwarder/sniffer_sse.go#L92) `if !streaming && contentLength < 0 { return nil, nil }` **skips rewriting** for any non-streaming body without a known length → the OpenAI-format response passes through to Claude Code untouched → SDK/mismatch error → "temporarily unavailable".

The skip is **intentional** (avoid unbounded buffering of unknown-length bodies) and locked in by `TestRewriteRespBody_NegativeContentLength`. So the fix is **opt-in with a size bound**, not an unconditional buffer.

Intended outcome: when a `rewriteResponseBody` rule sets `maxChunkSize: N`, GOST reads up to N+1 bytes from the body. If the body fits within N bytes, it rewrites the body through the Rewriter plugin and writes the response with `Content-Length` set (so Go's `resp.Write()` sends it as identity, not chunked). If it exceeds N, it passes through the original body unchanged — `Content-Length` stays -1 so `resp.Write()` re-chunk-encodes naturally. No OOM risk.

## Approach

Add a `MaxChunkSize int` field to body-rewrite rules. In `newRewriteBody`, for `!streaming && contentLength < 0`:

1. If no rule sets `MaxChunkSize > 0`, skip (return nil) — original behavior.
2. Read up to `MaxChunkSize + 1` bytes from the source via `io.LimitReader`.
3. If read exceeds `MaxChunkSize`: replay the buffered prefix + remaining source through a passthrough reader (`io.MultiReader`); `Content-Length` stays -1 so Go re-chunk-encodes on write.
4. If read fits:
   - **Compressed** (contentEncoding != ""): decompress → rewrite → recompress → set `Content-Length` to recompressed size, clear `Transfer-Encoding`. Must check `hasTypeMatch` first (same as existing compressed path).
   - **Uncompressed**: rewrite → set `Content-Length` to rewritten size, clear `Transfer-Encoding`.

## Changes

### 1. `core/chain/node.go` — add field to `HTTPBodyRewriteSettings`
```go
MaxChunkSize int
```

### 2. `x/config/config.go` — add field to `HTTPBodyRewriteConfig`
```go
MaxChunkSize int `yaml:"maxChunkSize,omitempty" json:"maxChunkSize,omitempty"`
```

### 3. `x/config/parsing/node/parse.go` — copy the flag in `parseBodyRewrites`
```go
rw := chain.HTTPBodyRewriteSettings{
    Type:         v.Type,
    Pattern:      pattern,
    Replacement:  []byte(v.Replacement),
    MaxChunkSize: v.MaxChunkSize,  // add this line
}
```

### 4. `x/internal/util/forwarder/sniffer_sse.go` — bounded-buffer logic

Replace the unconditional skip at lines 91-94 with a `maxChunkSize` computation; then after `mdOpts` (line 105), insert bounded-buffer logic that:
- Determines the effective maxChunkSize (max across all rules)
- If <= 0 → return nil, nil (original skip; compressed bodies continue to the existing path)
- Reads up to maxChunkSize+1 bytes via io.LimitReader
- If overflow → returns a passthrough `rewriteBody` with `rewrites: nil`, `streaming: false`, `contentLength: -1`, and a `multiReadCloser` wrapping `io.MultiReader(bytes.NewReader(buf), src)` that replays buffered+remaining bytes
- If fits, **compressed** (contentEncoding != ""): check `hasTypeMatch`, decompress → rewrite → recompress → set `contentLength` on recompressed result
- If fits, **uncompressed**: apply the rewrite chain → set `contentLength` on rewritten result

The existing compressed path (lines 107-154, only runs when `contentLength >= 0`) stays untouched — the new block handles the chunked variant of the same logic.

Add a `multiReadCloser` helper struct:
```go
type multiReadCloser struct {
    r io.Reader
    c io.Closer
}
func (m *multiReadCloser) Read(p []byte) (int, error) { return m.r.Read(p) }
func (m *multiReadCloser) Close() error               { return m.c.Close() }
```

### 5. `x/internal/util/forwarder/sniffer_rewrite.go` — conditional Content-Length + clear Transfer-Encoding
Change the `resp.ContentLength` assignment in BOTH `rewriteRespBody` and `rewriteReqBody` to only apply when the body was actually rewritten (and clear stale chunked headers — this also fixes a latent bug where compressed rewritten bodies with a known Content-Length would still carry `Transfer-Encoding: chunked` from the original response):
```go
if !rb.streaming && rb.contentLength >= 0 {
    resp.ContentLength = rb.contentLength
    resp.TransferEncoding = nil
    resp.Header.Del("Transfer-Encoding")
}
```
When `contentLength` stays `-1` (passthrough/overflow), `resp.Write` re-chunk-encodes naturally. Same pattern applies to `req.ContentLength` in `rewriteReqBody`.

### 6. `x/internal/util/forwarder/sniffer_test.go` — tests
- `TestRewriteRespBody_MaxChunkSize_Fits`: `ContentLength: -1`, rule with `MaxChunkSize: 1024`, body "original" (8 bytes < 1024) → rewritten to "rewritten", `ContentLength: 9`, `TransferEncoding` nil.
- `TestRewriteRespBody_MaxChunkSize_Overflow`: `ContentLength: -1`, rule with `MaxChunkSize: 4`, body "original" (8 bytes > 4) → NOT rewritten (body unchanged), `ContentLength` still -1.
- `TestRewriteRespBody_MaxChunkSize_Compressed_Fits`: `ContentLength: -1`, `Content-Encoding: gzip`, rule with `MaxChunkSize: 1024`, gzip("original") → decompress→rewrite→recompress, `ContentLength` = len(recompressed), `TransferEncoding` nil.
- Existing `TestRewriteRespBody_NegativeContentLength` (no MaxChunkSize set) still passes unchanged.

### 7. `llm-api-converter/CLAUDE.md` — document the requirement
Note: for **non-streaming** LLM responses, the GOST node's `rewriteResponseBody` rule must set `maxChunkSize: <bytes>` (e.g., 2097152 for 2MB), otherwise chunked responses bypass the plugin.

## Verification

1. `cd core && go build ./... && go vet ./...`
2. `cd x && go build ./... && go vet ./...`
3. `cd x && go test ./internal/util/forwarder/ -run TestRewriteRespBody -v`
4. Rebuild GOST binary + llm-api-converter plugin; restart with `maxChunkSize` on the response rewrite rule.
5. Non-streaming curl against `llm.home.pi`:
   ```
   curl -sS -D- -X POST http://llm.home.pi/v1/messages \
     -H 'content-type: application/json' -H 'x-api-key: sk-test' \
     -H 'anthropic-version: 2023-06-01' \
     -d '{"model":"claude-sonnet-4-6","max_tokens":30,"messages":[{"role":"user","content":"Say hi"}]}'
   ```
   Expect: Anthropic format (`"type":"message"`, `content[]`) with `"model":"claude-sonnet-4-6"`, `Content-Length` header, no `Transfer-Encoding: chunked`.
6. Confirm Claude Code's safety classifier no longer fails.
