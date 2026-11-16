# Model quirks: a capability-profile layer for `llm-api-converter`

Date: 2026-11-16
Status: draft
Repository: `llm-api-converter`
Driven by: the Gemini thought-signature replay issue (cuttle `fable` route →
Google OpenAI-compatible endpoint returned `400 Function call is missing a
thought_signature in functionCall parts` for Claude Code agentic sessions)
and the growing inventory of model-family conditionals scattered across
`convert/`.
Reference: magpie (`/root/code/magpie`), read as a behaviour reference only.

## Intent

Stop scattering `isDeepSeekModel(model)` / `isGLMModel(model)` /
`isOpenAIReasoningModel(model)` checks across the pairwise converters. Move
every model-family-dependent behavior behind one typed `ModelProfile`
computed once per request from the target model name, and attach each quirk
to a fixed **seam** in the conversion path. Adding a new provider family
(gemini did, kimi may) becomes a single new file plus tests, not a scatter of
ad-hoc conditionals.

Goal: future quirks are reviewable, testable, and honest — every quirk
records why it exists (incident link), where it applies, and is deleted when
it no longer applies.

## Goals

- A single `modelProfile`/`ModelProfile` struct replaces the current
  `classifyModel` ad-hoc `modelProfile{model, isDeepSeek, isOpenAI}`. Fields
  are named for **capabilities/behaviors**, not vendors — e.g.
  `ForceToolChoice: allow|strip`, `JsonSchemaMode: pass|json_object|reject`,
  `ReasoningReplay: none|deepseek|glm`, `ReasoningFallback: none|placeholder`,
  `ContentFlatten: bool`, `StripTopK: bool`,
  `ReasoningMaxTokens: none|max_completion_tokens`,
  `ThinkingStyle: none|deepseek|glm`,
  `SignatureScheme: none|openai_toolcall_gemini`,
  `StreamTailSynthesize: bool`. Rows that involve Gemini thinking config
  are NOT model-family quirks — they are Gemini-native adapter behavior and
  stay where they are.
  This is the authoritative field set; if Goals text below sketches anything
  else, this list wins.
- All current ad-hoc conditionals are rewritten to read profile fields. After
  the refactor, `grep -E "isDeepSeekModel|isGLMModel|isOpenAIStyleModel"` on
  all files returns only the profile definition itself (test files may call
  pure profile functions directly instead of the model-name helpers; the
  helpers disappear).
- Every quirk carried by an incident link (e.g. `deepseek-dadax 400
  "messages unavailable for thinking; choose thinking instead"`,
  `gemini-4-hydrogen thought_signature 400`) is a named field with the incident
  in a doc comment.
- A behavior-identical Part A lands first (no request/response bytes change);
  new code paths enter only in the (Part B) Gemini fix.
- docs gains an authoritative **Model profiles and quirks** table:
  quirk → models → why (incident) → deleted when.

## Non-goals

- **No IR migration.** Pairwise stays final (2026-11-11 decision); no step
  pipeline / middleware layer like magpie's translation registry.
- **No user-configurable quirks** — for this work, detection is static by
  target model name only (option (a)). No cuttle-side `quirks:` config key.
  A later change can add overrides if real custom gateways need them.
- **No coverage of every Gemini edge el**: Part A is a refactor, Part B is
  the one fix. The broader magpie capability set (sigstore `__tsk16__`
  indirection, stream-tail synthesis for lost `[DONE]`, `response_format`
  fallback, `pinned=false`) stays out of this design and is captured as
  documented follow-on candidates below.
- **No cuttle repo changes** in Part A. Whether Part B ships through a
  converter release + cuttle dependency bump + redeploy is Open decision #3;
  it is a scope choice for this plan, not a code change.
- No behavior changes to OpenAI/Responses/Gemini-native conversions beyond
  the listed quirk move.

## Constraint: pairwise is final

Every seam in this design has exactly two owners — the conversion direction
and the quirk — and nothing cross-cuts. The profile does not encode
cross-protocol behavior.

## The quirk inventory (Part A scope)

Each row below is a current ad-hoc conditional that moves behind the profile:

| # | Site (file:line today) | Ad-hoc | Profile field | Why (incident/reason) |
|---|---|---|---|---|
| 1 | `anthropic_to_openai.go:88` tool_choice `any`/`tool` stripped for DeepSeek | `isDeepSeekModel` | `ForceToolChoice = strip` | DeepSeek rejects forced `tool_choice`; Anthropic "any"/"tool" becomes a system instruction |
| 2 | `anthropic_to_openai.go:93` tuple | same | same | same row |
| 3 | `anthropic_to_openai.go:109` `toolChoiceInstruction` deepseek-only | `isDeepSeekModel` | `ForceToolChoice = strip` | the system-instruction soft copy of #1 |
| 4 | `anthropic_to_openai.go:239-245` `top_k` + `isOpenAIReasoningModel(stripAllProviderPrefixes(...))` | `isOpenAIReasoningModel` | `StripTopK = true` | OpenAI reasoning models reject `top_k` |
| 5 | `anthropic_to_openai.go:218,242` `reasoningModel := isOpenAIReasoningModel` — `max_completion_tokens` vs `max_tokens` | same | same (profile field `ReasoningMaxTokens = max_completion_tokens`) | OpenAI reasoning models reject `max_tokens` |
| 6 | `anthropic_to_openai.go:262-269` `isDeepSeek` → `{thinking,reasoning_effort}` vs `isGLMModel` → `{type,budget_tokens}` | `isDeepSeek` / `isGLMModel` | `ThinkingStyle = deepseek | glm | none` | DeepSeek uses `reasoning_effort`; GLM uses `thinking.budget_tokens` |
| 7 | `anthropic_to_openai.go:283-289` `json_schema` unsupported → `json_object` + system instruction | `profile.isDeepSeek` | `JsonSchemaMode = json_object` | DeepSeek rejects `json_schema` |
| 8 | `anthropic_to_openai.go:342-346` GLM flattens multi-part content | `isGLMModel` | `ContentFlatten = true` | GLM accepts only string content |
| 9 | `anthropic_to_openai.go:620-641` reasoning replay (thinking → re-inject `reasoning_content`) / cache restore / placeholder | `replaysReasoning` + `isDeepSeekModel` | `ReasoningReplay = none | deepseek | glm` + `ReasoningFallback = placeholder | none` | DeepSeek V4 requires reasoning_content round trip |
| 10 | `anthropic_to_openai.go:15` `replaysReasoning` def | `isDeepSeekModel \|\| isGLMModel` | `ReasoningReplay` | helper inlining of #9 |
| 11 | `openai_to_anthropic.go:576` replay thinking blocks into Anthropic | `isDeepSeekModel` | `ReasoningReplay` | symmetric leg of #9 |
| 12 | `types.go:308-336` `modelProfile{model,isDeepSeek,isOpenAI}` + `classifyModel` cache | n/a (definition) | profile definition itself | the struct to replace |
| 13 | `types.go:344-352` `isOpenAIReasoningModel` o1/o3/o4/gpt-5 prefix | prefix list | `ReasoningMaxTokens = max_completion_tokens` | OpenAI reasoning family |
| 14 | `convert/anthropic_to_gemini.go` + `openai_to_gemini.go`: `Anthropic→Gemini` thinking config maps directly to `thinkingConfig` (today correct, stays) | n/a | not a quirk — editorial note; only the replay leg needs new fields | Gemini-native adapter reads `thinking.budget` |
| 15 | `stream.go` / `stream_anthropic_to_openai.go` — `xhigh → max` effort cap | `xhigh` literal (no model gate) | leave as-is; not model-family specific | behavior identical across providers; not a quirk |

After Part A, these leave production code paths; they re-enter only via
`opts.Profile` (or their own unit-tested helpers).

## Part B — the Gemini thought-signature fix, delivered through the profile

The driver incident: post-tool-use replay requests to Google's
OpenAI-compatible endpoint return `400 Function call is missing a
thought_signature in functionCall parts. … position 3`. magpie reports the
same class of 400 in `gemini_signature_test.go:106`.

**Scope (as decided in the planning thread — see "Open decisions" below):**
Sentinel-only**: every replayed tool call for a gemini-family target carries
`extra_content:{google:{thought_signature:"skip_thought_signature_validator"}}`.
The `SessionStore` branch stays legal but is NOT wired in Part B; see item 1.

Seam:

1. New profile field `SignatureScheme = openai_toolcall_gemini` for target
   models matching `gemini-*` (and aliases that carry `gemini` as a model
   prefix segment; match by
   `strings.Contains(strings.ToLower(model), "gemini")` gated after
   `StripProviderPrefix`).
2. A new seam function `geminiToolReplay()` mutates the outgoing OpenAI
   assistant message for every tool call before dispatch:
   - ALWAYS emit `extra_content:{google:{thought_signature:"skip_thought_signature_validator"}}`
     (Google's documented bypass, verified live 2026-11-16 with 200) — unless
     a real signature is supplied by a future phase;
   - do NOT read `SessionStore` in Phase 1 (per-connection SID makes it
     unreliable across agentic turns; see the X/Y headers note in the follow
     up section).
3. Sentinel-on-every-call is never *worse* for Google than today (which
   fails 400), and correct enough: real signatures are an optimization, not
   a correctness requirement.
4. `assert assistantMessagesOnly` applied: the mutation touches nothing else.
5. Real signature round-trip is a **named follow-on**:
   - SID was verified 2026-11-16 to be per-GOST-connection (`x/service/service.go:306`
     assigns `xctx.Sid(xid.New().String())`), so keying signatures by SID
     breaks across agentic turns whenever the client does not reuse the
     connection (and cuttle pins `keepalive:false`, making every request its
     own connection).
   - The only SID-independent key that round-trips inside the body is the
     tool_call id — same reasoning that powers `ReasoningCache.GetBest(ids,…)`
     today. A follow-on phase wires capture of `extra_content.google.thought_signature`
     from the OpenAI→Anthropic leg (both non-stream and SSE) into a
     ReasoningCache-style ID-keyed store. That phase costs ~1 day and is NOT
     part of this plan.

Part B lands behind the same seam layer: it does not add any new inline
conditionals; it is one profile field + one seam function + a replay
fixture that reproduces the 400→200 behavior without a live endpoint.

## Maintainer decisions (2026-11-16)

1. **Sentinel-only.** Replay emits `skip_thought_signature_validator` always;
   no capture code in this plan. Real-signature round-tripping, if wanted,
   is the follow-on design documented below (ID-keyed `ReasoningCache`-style
   store, not SID-keyed).
2. **Seam hook surface: (a)** inside `convertAnthropicRequestToOpenAI` only.
   Passthrough bodies stay untouched; gemini→openai-outbound clients are a
   separate matter if they ever appear.
3. **Release: not in this plan.** The plan ends at a merged converter
   library change (spec is approved for merge only). Release tag, cuttle
   `go.mod` bump and redeploy are separate steps the maintainer decides later.

## Known follow-on candidates (NOT in this design)

- Stream-tail synthesis: a Gemini stream that never emits `finish_reason` or
  `[DONE]` (magpie synthesizes `finishChunk`). Currently breaks silently as a
  cut reply in `stream_*`; detection + synthesis is its own design.
- `__tsk16__`-style signature-in-tool_call-id indirection for non-Gemini-aware
  clients (magpie's approach) — requires client-side id stability guarantees
  we do not have; our simpler round-trip via `SessionStore` suffices.
- `reasoning_details` unification for openrouter-style reasoning (native
  collapse into `reasoning_content` — today implemented, partially; not a quirk).
- `response_format` JSON-schema→`json_object` fallback for Gemini (today: 400
  forbidden schema type — needs its own seam).
- finish-reason/tool finish parity across the three Gemini directions.
- cuttle-side provider `quirks:` override for custom gateways.

## The record

Part A: one structural change plus unit-tested behavior-preserving moves.
Part B: two tests (round-trip + sentinel fallback), one seam function.
Total expected size: ~1–2 working days, ~600 changed lines including tests.

## Test plan

- Part A: every existing test passes after mechanical updates that delete
  the model-name helpers from test call sites (`isDeepSeekModel`,
  `isOpenAIReasoningModel`, `replaysReasoning`, `classifyModel`); the
  assertion content stays the same. No test logic changes.
- Part B: new fixture `gemini_tool_replay_extra_content` (assistant message with
  tool call → outgoing body carries `extra_content` with sentinel); a second
  fixture with `SessionStore` pre-populated asserts the real signature and
  **absent** sentinel; a third asserts no mutation when the target is not
  gemini-family.
- No test invokes a live endpoint.

## Risks

- Mis-attributed profile (e.g. an `o3`-authored-but-`gemini`-served proxy)
  applies the wrong quirk; bounded by `option (a)` scope: this library
  detects by model name only, as it does today.
- Part A is mechanically wide (every call site) and runs on `go vet` + full
  gate; a single-day review is enough.
- Part B's sentinel masks missing real signatures; acceptable by design
  (verified, documented, opt-in to real sig by filling `SessionStore`).

## Verification bar

- `go test ./... -count=1` green before Part B is merged.
- After Part A, behavior is identical: run the full suite unchanged and
  diff-capture a representative baseline (temp tool_strip, thinking, replay,
  gemini path) — only structural diffs allowed.
- Part B verified by test fixtures + maintainer live session (the original
  agentic session through the `fable` route: Claude Code tool-use turn must
  end with `message_stop` and non-empty input/output tokens in stats).
