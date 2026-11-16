# Model profile quirks implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace scattering of `isDeepSeekModel(model)` / `isGLMModel(model)` / `isOpenAIReasoningModel(model)` checks with one typed `modelProfile` computed once per request, and land the Gemini thought-signature replay seam through the new layer.

**Architecture:** Pure refactor first (Part A, behavior identical): one `modelProfile` struct + `classifyModel` replace the current ad-hoc struct and the three model-name helpers; every quirk site reads profile fields. Then the Gemini fix (Part B) as one new profile field and one seam function.

**Tech Stack:** Go 1.27, llm-api-converter convert package, stdlib testing.

**Spec:** `/root/code/go-gost/llm-api-converter/docs/superpowers/specs/2026-11-16-model-profile-quirks-design.md` (the spec argues; this plan lists tasks).

## Context (Why Part B)

Claude Code `/v1/messages` traffic → cuttle → Google OpenAI-compatible Gemini endpoint (model e.g. `gemini-4-hydrogen`) returned `HTTP 400` on every replayed tool call: `message "Function call is missing a thought_signature in functionCall parts …"`. The tool_call is deterministically replayable by the client (Anthropic echoes the ids), but our Anthropic→OpenAI path emits `tool_calls` with no `extra_content.google.thought_signature` at all, so Google rejects the assistant turn.

Maintainer decision (2026-11-16): **sentinel-only** for this plan. Send `skip_thought_signature_validator` always, which Google accepts as a bypass (verified live 2026-11-16: replay of the exact same request returns 200 with an answer). Real signatures per `tool_call id` are a follow-on design (keyed by `tool_call id` like `ReasoningCache`, not SID — see spec).

Scope fixed by the maintainer: Part A = profile refactor only (no behavior change); Part B = the one Gemini fix; **no** OpenAI passthrough hook post-pass; **no** release in this plan (merge only, maintainer decides on tag separately); **no** cuttle changes; no user-configurable quirks (static model-name match); pairwise architecture stays final.

## Global Constraints

- English only in code, comments, tests and docs.
- No new dependencies, no `go.mod` bump, no sibling repo edits.
- `go test -p 1 ./... -count=1` must be green after every task; `go vet ./...` clean.
- Part A adds zero behavior changes (byte-compatible request/response bodies).
- After Task 5, `grep -rn "isDeepSeekModel\|isGLMModel\|isOpenAIStyleModel\|isOpenAIReasoningModel" --include=*.go .` returns **nothing**.
- `StripProviderPrefix` (single) vs `stripAllProviderPrefixes` (loop) — keep both, never unify them; the profile computes both spots.
- Wire-field shapes of OpenAI and Anthropic stay byte-compatible for non-quirk cases (`omitempty` preserved).
- Test edits allowed in Part A: deleting model-name helper calls from test call sites (mechanical), no assertion changes.
- No `SessionStore` wiring for signatures in this plan (see spec).
- Do not tackle OpenRouter `reasoning_details`, no `stream_tail` synthesis, no cache for the real signature. Not this plan.

## Review Focus

Five input classes / failure modes most likely to bite, and the task that pins each:

1. **Provider- and multi-segment prefixed model ids** (`ollama/deepseek-v4-flash`, `foo/bar/glm-4.7`, `models/o3-mini`) — quirk mis-attribution. Pinned in Task 1 by `TestModelProfileMatchesExpectedMatrix`.
2. **DeepSeek `json_schema → json_object` fallback** (incident a9a7347/dσdadax "response_format json_schema unavailable") — confused with generic instruction. Pinned in Task 2 by re-asserting `TestAnthropicOpenAI_DeepSeekJsonSchemaInstruction` still passes byte-for-byte.
3. **Reasoning replay placeholder fallback on DeepSeek when cache is empty** (placeholder `"..."`) — regression re-introduces 400s. Pinned in Task 2 by re-asserting `TestReasoningContentFallback…` placeholder path.
4. **Reasoning-model detection on prefixed ids** (`zhipu/glm-4.7`, `openai/o3-mini`) — strip-order ambiguity (single vs loop). Pinned in Task 1 (matrix rows 4/5).
5. **Client SID assumption for signatures** — any `GetThoughtSigFor` call keyed by SID silently fails across turns. Pinned in Task 4 by explicitly asserting no `SessionStore` call in the replay seam + maintainer smoke (Task 6).
6. **Assistant-message replay without the sentinel** — regression to Google 400 for every agentic turn. Pinned in Task 4 by the new fixtures `TestGeminiToolReplayExtraContent…`.

---

## File map

| File | Task | Responsibility |
|------|------|----------------|
| `convert/types.go` | 1, 5 | `modelProfile` struct + `classifyModel`; later delete `isDeepSeekModel`/`isGLMModel`/`isOpenAIStyleModel`/`isOpenAIReasoningModel` |
| `convert/quirks_equivalence_test.go` | 1 (new) | Expected-matrix equivalence test pinning model→profile |
| `convert/anthropic_to_openai.go` | 2 | Rows 1-12 call-site migration (tool_choice, top_k/max_completion_tokens, thinking style, json_schema, flatten, replay/cache/placeholder) |
| `convert/openai_to_anthropic.go` | 3 | Row 11 (replay thinking gating on DeepSeek replay mode) |
| ~~`convert/stream.go`~~ | — | no edits (no quirk lives here) |
| `convert/anthropic_to_gemini.go`, `openai_to_gemini.go`, `stream_gemini*.go` | — | untouched; Gemini-native behavior is not profiled |
| `convert/anthropic_to_openai.go` (`geminiToolReplay`) | 4 | `SignatureScheme = openai_toolcall_gemini` seam + sentinel fixture (Part B) |
| `convert/gemini_tool_replay_test.go` | 4 (new) | Gemini replay fixtures: sentinel path, non-gemini isolation |
| `convert/anthropic_to_openai.go` + `convert/openai_to_anthropic.go` | 5 | delete repo-wide `isXModel` helpers once migrated; grep must print only the profile line |
| `docs/QUIRKS.md` | 6 (new) | Model profiles and quirks — quirk → models → why (incident) → deleted when |
| `docs/superpowers/plans/2026-11-16-model-profile-quirks.md` | every | this record |

## Execution order

Task 1 → Task 2 → Task 3 (Part A core) → Task 4 (Part B) → Task 5 (final sweep, deletes helpers) → Task 6 (docs + records + maintainer verification).
Dependencies: 2 needs 1, 3 needs 2 (shared file), 4 needs 1+2 (types + Anthropic→OpenAI path), 5 needs 4 (delete helpers only when nothing references them).

---

### Task 1: Profile definition + equivalence pin

**Files:**
- Create: `convert/quirks_equivalence_test.go`
- Modify: `convert/types.go` (`classifyModel`, `modelProfile`, `isOpenAIReasoningModel` as needed)
- Delete (this task only): nothing

**Interfaces:**
- Consumes: current `isDeepSeekModel`, `isGLMModel`, `isOpenAIStyleModel`, `isOpenAIReasoningModel` — keep them unchanged in this task; `classifyModel` reads them and the equivalence pin compares the assembled profile against expected values only.
- Produces: `modelProfile` with new fields; consumed by Tasks 2-5.

- [ ] **Step 1: Write the equivalence-pin test matrix**

Add `convert/quirks_equivalence_test.go` with `TestModelProfileMatchesExpectedMatrix` over this table (values hand-derived from the spec's inventory):

| model | StripTopK | ReasoningMaxTokens | ThinkingStyle | ForceToolChoice | JsonSchemaMode | ContentFlatten | ReasoningReplay |
|---|---|---|---|---|---|---|---|
| `deepseek-v4-pro` | false | none | deepseek | strip | json_object | false | deepseek |
| `deepseek-chat` | false | none | deepseek | strip | json_object | false | deepseek |
| `ollama/deepseek-v4-flash` | false | none | none | allow | pass | false | none |
| `zhipu/deepseek-v3` | false | none | none | allow | pass | false | none |
| `glm-4.7` | false | none | glm | allow | pass | true | glm |
| `zhipu/glm-51` | false | none | none | allow | pass | false | none |
| `foo/bar/glm-4.7` | false | none | none | allow | pass | false | none |
| `o1-mini` | true | max_completion_tokens | none | allow | pass | false | none |
| `o3-mini` | true | max_completion_tokens | none | allow | pass | false | none |
| `o4-mini` | true | max_completion_tokens | none | allow | pass | false | none |
| `gpt-5.5` | true | max_completion_tokens | none | allow | pass | false | none |
| `models/o3-mini` | true | max_completion_tokens | none | allow | pass | false | none |
| `foo/bar/o3-mini` | **true** | **none** | none | allow | pass | false | none |
| `gemini-4-hydrogen` | false | none | none | allow | pass | false | none |
| `models/gemini-2.5-pro` | false | none | none | allow | pass | false | none |
| `kimi-k3` | false | none | none | allow | pass | false | none |
| `gpt-4o` | false | none | none | allow | pass | false | none |

**How to derive these rows** (do this exactly; do not infer): read
`types.go: isDeepSeekModel/isGLMModel/isOpenAIReasoningModel` and
`anthropic_to_openai.go: supportsTopKPassthrough, stripAllProviderPrefixes`.
`isDeepSeekModel` and `isGLMModel` do **no** prefix stripping at all (they only
check the verbatim prefix), so every *prefixed* deepseek/glm id classifies as
`none` — the provider-prefix class is not a deepseek/glm today by mistake, and
Part A must not change that (a provider-served deepseek goes down the generic
path identically).
`StripTopK` = `isOpenAIReasoningModel(stripAllProviderPrefixes(model))` — the
multi-segment ids matter (`foo/bar/o3-mini` is `true` even though the single-
strip path inside `isOpenAIReasoningModel` leaves `bar/o3-mini`, which does
NOT match `o3`). `ReasoningMaxTokens = max_completion_tokens` iff
`isOpenAIReasoningModel(model)` — single-strip only (`foo/bar/o3-mini` is
`none`, `models/o3-mini` IS `max_completion_tokens` because one strip leaves
`o3-mini`). `zhipu/deepseek-v3` and `foo/bar/glm-4.7` have NO prefix stripping
and classify as `none`.

Write every row's expected value by re-reading the current code path in `types.go:320-350` and `anthropic_to_openai.go` — do NOT compute by inference.

- [ ] **Step 2: Run the matrix test — expect FAIL**

Run: `GOWORK=off go test ./convert/ -run TestModelProfileMatchesExpectedMatrix -count=1 -v`
Expected: FAIL — `modelProfile` has no fields yet.

- [ ] **Step 3: Extend `modelProfile` in `convert/types.go:304-318`**

Replace `modelProfile{model, isDeepSeek, isOpenAI}` with a struct carrying (all unexported, all zero-valued = off):

```go
type modelProfile struct {
	model             string
	stripTopK         bool
	reasoningMaxTokens string // "none" | "max_completion_tokens"
	thinkingStyle      string // "none" | "deepseek" | "glm"
	forceToolChoice    string // "allow" | "strip"
	jsonSchemaMode     string // "pass" | "json_object" | "reject"
	contentFlatten     bool
	reasoningReplay     string // "none" | "deepseek" | "glm"
	reasoningFallback  string // "none" | "placeholder"
	signatureScheme    string // "none" | "openai_toolcall_gemini"
	streamTailSynthesize bool // reserved; not read in this plan
}
```

`classifyModel(model)` builds it from today's three helpers (so it is behavior-identical): `isDeepSeekModel` → Thinking/replay/`forceToolChoice=strip`/`jsonSchemaMode=json_object`; `isGLMModel` → glm twins; `isOpenAIReasoningModel` → `reasoningMaxTokens=max_completion_tokens` and `stripTopK=true`. The matrix rows above are the oracle.

Keep the old helpers `isDeepSeekModel`, `isGLMModel`, `isOpenAIStyleModel`, `isOpenAIReasoningModel` in place for this task (used only by the equivalence test's cross-check inside `classifyModel`).

- [ ] **Step 4: Run the equivalence matrix — expect PASS**

Run: `GOWORK=off go test ./convert/ -run TestModelProfileMatchesExpectedMatrix -count=1 -v`
Expected: PASS.

- [ ] **Step 5: Full suite — expect PASS (behavior identical)**

Run: `GOWORK=off go vet ./... && GOWORK=off go test -p 1 ./... -count=1`
Expected: PASS, with the same total test count as before this task (new file adds one test only).

- [ ] **Step 6: Commit**

Append the Task 1 implementation record to this plan, then:
```bash
git add convert/types.go convert/quirks_equivalence_test.go
git commit -m "refactor(convert): typed model profile with equivalence pin"
```

---

### Task 2: migrate request-side quirks in `anthropic_to_openai.go`

**Files:**
- Modify: `convert/anthropic_to_openai.go` (rows 1-12)
- Test: existing `anthropic_openai_p*_test.go`, `anthropic_to_openai_test.go`, `convert_test.go` must pass unchanged

**Interfaces:**
- Consumes: `classifyModel` / `modelProfile` from Task 1.
- Produces: the same converted bodies as today, via profile fields.

Rewrite this seat-map, in order, keeping byte-identical output:

- [ ] **Step 1: `anthropicToolChoiceToOpenAi`** (rows 1-2): parameter changes from `model string` to accepting profile, or pass `profile.forceToolChoice`; the DeepSeek forced-choice strip path becomes profile-driven (`"any"` / `"tool"` → `nil` + system instruction; `"auto"` / `"none"` stay).
- [ ] **Step 2: `toolChoiceInstruction`** (row 3): same `strip` profile field.
- [ ] **Step 3: `reasoningModel` / `stripTopK`** (rows 4-5): replace `isOpenAIReasoningModel(outputModel)` with `profile.reasoningMaxTokens == "max_completion_tokens"` **at the two literal call sites**, including the prefixed-id site (`stripAllProviderPrefixes`) by resolving the profile ONCE at the top from the same input (the profile must agree with the stripped/un-stripped distinction it replaces).
- [ ] **Step 4: thinking style** (row 6): `if profile.thinkingStyle == "deepseek" → {thinking, reasoning_effort}`; `else if == "glm" → {type, budget_tokens}`; unchanged branch body for non-quirk models.
- [ ] **Step 5: `jsonSchemaMode`** (row 7): DeepSeek branch applies `json_object` + instruction; other models keep `json_schema`.
- [ ] **Step 6: flatten GLM** (row 8): `if profile.contentFlatten { flattenContentForGLM(...) }` — identical behavior.
- [ ] **Step 7: Replay / cache restore / placeholder** (rows 9-10): `replaysReasoning(model)` call sites become `profile.reasoningReplay != "none"`; the DeepSeek placeholder fallback becomes `profile.reasoningFallback == "placeholder"`; the `GetBest` path is unchanged.
- [ ] **Step 8: Re-run whole suite; fix only mechanical fallout**

Run: `GOWORK=off go vet ./... && GOWORK=off go test -p 1 ./... -count=1`
Expected: PASS. Any failure => the migration changed behavior => STOP, revert, re-read the current code side-by-side with the changed lines until it no longer fails; do NOT edit assertions.

- [ ] **Step 9: Commit**

Append the Task 2 implementation record, then commit:
```bash
git add convert/anthropic_to_openai.go
git commit -m "refactor(convert): migrate quirks to profile in anthropic-to-openai"
```

---

### Task 3: migrate replay thinking in `openai_to_anthropic.go`

**Files:**
- Modify: `convert/openai_to_anthropic.go` (`openAI → anthropic` path; row 11)
- Test: existing `stream_test.go`, `stream_anthropic_test.go`, `openai_to_anthropic_test.go` (and the review suites) must pass unchanged

**Interfaces:**
- Consumes: `classifyModel(profile)` — note this is a **different entry function**; the profile must come from the same call site (no new helper).
- Produces: unchanged behavior.

- [ ] **Step 1: `replayThinking` gate**

Change `replayThinking := msg.ReasoningContent != "" && isDeepSeekModel(model)` to `profile.reasoningReplay == "deepseek"` by resolving the profile at the top of the function (or threading it through from the caller, whichever is the minimal honest change — pick the smaller diff, no public signature changes).

- [ ] **Step 2: GLM sibling gate**

Row number 10 already handled `reasoningReplay == "glm"` on the Anthropic→OpenAI side; GLM here stays as today (`isDeepSeekModel` only) — verify there is no DeepSeek-GLM divergence to fix, and **do not** extend it.

- [ ] **Step 3: Full suite — PASS**

Run: `GOWORK=off go test -p 1 ./... -count=1`
Expected: PASS.

- [ ] **Step 4: Commit**

Append record; commit:
```bash
git add convert/openai_to_anthropic.go
git commit -m "refactor(convert): gate anthropic replay thinking on profile"
```

---

### Task 4: Part B — Gemini replay sentinel seam

**Files:**
- Modify: `convert/types.go` (`signatureScheme` field exists since Task 1 — just wire detection), `convert/anthropic_to_openai.go` (new `geminiToolReplay` seam applied to replay-visible assistant messages)
- Create: `convert/gemini_tool_replay_test.go`

**Interfaces:**
- Consumes: `convertAnthropicRequestToOpenAI`'s assembled `OpenAIChatRequest.Messages` structure; target profile.
- Produces: `signatureScheme = openai_toolcall_gemini` profile field live, consumed inside the replay seam; no other outbound path touches it.

- [ ] **Step 1: Profile detection**

In `classifyModel`, set `signatureScheme` to `openai_toolcall_gemini` iff `strings.Contains(strings.ToLower(model), "gemini")` (this is `gemini-4-hydrogen` / `models/gemini-2.5-pro` — i.e. `StripProviderPrefix` not needed for the detection, just literal presence).

- [ ] **Step 2: Write the new fixtures (expect FAIL)**

Add `convert/gemini_tool_replay_test.go`:

1. `TestGeminiToolReplayAddsExtraContentWithSentinel` — input an Anthropic request with an assistant turn containing `tool_use` blocks followed by `tool_result` blocks (replay shape) and `model: "gemini-4-hydrogen"`. Expect the outgoing OpenAI body: each `tool_calls` entry carries `extra_content:{google:{thought_signature:"skip_thought_signature_validator"}}`.
2. `TestGeminiToolReplayNonGeminiTargetIsUntouched` — same request with `model: "deepseek-v4-pro"`, expect NO `extra_content` on tool_calls (byte-identical marshal or explicit absent key).
3. `TestGeminiToolReplayOnlyAssistantTurns` — a user message containing the substring `"tool_use"` (fake): expect zero mutation.
4. `TestGeminiToolReplaySentinelNeverReadsSessionStore` — passes a `ReasoningCache`/SessionStore double that `t.Fatal`s on any Get/Set call; convert completes (proves Part B does not lean on SID). (If the fixture shape cannot host the double, skip this test with an explicit comment — it is the assertion that makes the SID trap loud for the reviewer.)

- [ ] **Step 3: Run fixtures — FAIL**

Run: `GOWORK=off go test ./convert/ -run ExtraContent -count=1 -v`
Expected: FAIL.

- [ ] **Step 4: Implement `geminiToolReplay`**

In `anthropic_to_openai.go`, inside `convertAnthropicRequestToOpenAI`, after the full `oai.Messages` assembly and before returning the marshaled bytes, for every `assistant` message with `tool_calls` and `profile.signatureScheme == "openai_toolcall_gemini"`: for every `tool_calls` entry, set its `ExtraContent = map[string]any{"google": map[string]any{"thought_signature": "skip_thought_signature_validator"}}`.

Wire format decision (pinned): emit `extra_content` as a **top-level key on the tool_call object**, exactly where Google's live API accepts it (validated 2026-11-16 with the curl replay); case `extra_content` (snake_case), nested `google` object, nested `thought_signature` key.

Reuse the existing `OpenAIToolCall` type only if it can carry the field; otherwise add `ExtraContent json:"extra_content,omitempty"` to it. **NO** new outbound type, NO SessionStore read.

- [ ] **Step 5: Run all fixtures — PASS; suite — PASS**

Run: `GOWORK=off go vet ./... && GOWORK=off go test -p 1 ./... -count=1`
Expected: fixtures PASS; total suite PASS.

- [ ] **Step 6: Commit**

Append record; commit:
```bash
git add convert/types.go convert/anthropic_to_openai.go convert/gemini_tool_replay_test.go
git commit -m "fix(convert): emit signature sentinel replay for gemini tool calls"
```

---

### Task 5: delete legacy model-name helpers repo-wide

**Files:**
- Modify: every file in `convert/*.go` that still references `isDeepSeekModel`, `isGLMModel`, `isOpenAIStyleModel`, `isOpenAIReasoningModel` (incl. `_test.go`)

- [ ] **Step 1: Enumerate remaining references**

Run: `grep -rn "isDeepSeekModel\|isGLMModel\|isOpenAIStyleModel\|isOpenAIReasoningModel" --include=*.go .`

- [ ] **Step 2: Migrate every hit**

Replace with profile-field reads (or pure matrix, task-1-style, in tests if the test asserts on a model-name variant directly); tests that still need the values of `isDeepSeekModel(...)` etc. should assert on the equivalence matrix instead.

- [ ] **Step 3: Equivalence matrix now pins behavior against the profile only**

`TestModelProfileMatchesExpectedMatrix` must not reference the legacy helpers at all; it keeps the expected-value table as the oracle. (It already does after Task 1 — confirm here and delete any stray references.)

- [ ] **Step 4: Grep bar — expect zero hits**

Run: `grep -rn "isDeepSeekModel\|isGLMModel\|isOpenAIStyleModel\|isOpenAIReasoningModel" --include=*.go .` → **no output**.

- [ ] **Step 5: Full suite — PASS**

Run: `GOWORK=off go vet ./... && GOWORK=off go test -p 1 ./... -count=1`
Expected: PASS.

- [ ] **Step 6: Commit**

Append record; commit:
```bash
git add convert/
git commit -m "refactor(convert): remove legacy model-name helpers"
```

---

### Task 6: docs + records + verification bar

**Files:**
- Create: `docs/QUIRKS.md`
- Modify: `docs/superpowers/plans/2026-11-16-model-profile-quirks.md` (records)

- [ ] **Step 1: Write `docs/QUIRKS.md`** — table of every quirk: name, model families, reason (link the incident), file:line, deleted when. Model after the spec's inventory table; add the new `SignatureScheme`.

- [ ] **Step 2: Commit docs**

```bash
git add docs/
git commit -m "docs: model profiles and quirks guide"
```

- [ ] **Step 3: Maintainer verification (not part of this plan)**

Hand the maintainer exactly this script:
1. Merge to `main` (or hold in `work/model-profile` — maintainer's call; the plan touches only this repo).
2. Tag `llm-api-converter` per their release flow, bump cuttle `go.mod`, redeploy. (Open decision #3 — explicitly outside this plan.)
3. Live Claude Code session through the `fable` route with at least one agentic tool-use turn; expected: `message_stop` + non-trivial token count in stats.

- [ ] **Step 4: Final record**

Append Task 6 record to this plan; commit (this plan's records are the audit trail).

---

## Implementation record
(Each task appends its record here: date/status, repo+SHA, files actually changed, tests/commands and results, review findings and fixes, deferred items.)

### Task 1: Profile definition + equivalence pin — 2026-11-16 — BLOCKED (oracle contradicts current helper semantics; maintainer ruling required)
- Repo: `llm-api-converter` (branch `main`, base HEAD `2770b49`; **nothing committed — task gate (Step 4 matrix PASS) unmet and a red suite must not land on main**; changes left in working tree).
- Files actually changed: `convert/types.go` (extended `modelProfile` with the new unexported fields per the brief's struct sketch, `classifyModel` rewired to build the profile from today's helpers, `stripTopK = isOpenAIReasoningModel(stripAllProviderPrefixes(model))` computed inside `classifyModel`; `isDeepSeek` kept as a deprecated legacy field for the two unmigrated `profile.isDeepSeek` call sites in `anthropic_to_openai.go`; `isOpenAI` dropped — zero call sites) and `convert/quirks_equivalence_test.go` (new, `TestModelProfileMatchesExpectedMatrix`, 17 rows verbatim from the brief). Not committed.
- What passes: `go vet` clean (convert/engine/rewriter/root); all pre-existing convert tests green; `engine`, `rewriter` ok; e2e fails as before this task (pre-existing: builds sibling `gost` binary, `github.com/go-gost/x/dashboard` unresolvable under GOWORK=off — unrelated).
- What fails (expected per the subagent context note, "report instead of editing the row"): `TestModelProfileMatchesExpectedMatrix` fails on exactly the four provider-prefixed deepseek/glm rows — `ollama/deepseek-v4-flash`, `zhipu/deepseek-v3`, `zhipu/glm-51`, `foo/bar/glm-4.7` — got `thinkingStyle="none"/allow/pass/none`, want `deepseek|glm` per the matrix. All other 13 rows pass.
- Root cause (verified in code, not inference): current `isDeepSeekModel` (`types.go:320`) and `isGLMModel` (`types.go:325`) do **no** prefix stripping: `strings.HasPrefix(strings.ToLower(model), "deepseek")` / `"glm-"` on the verbatim model. The matrix requires prefixed ids to classify as deepseek/glm; row `foo/bar/glm-4.7` even requires removing *all* leading `segment/` pairs (single strip leaves `bar/glm-4.7`, which matches nothing). The brief's derivation text ("single-strip path in `isDeepSeekModel`/`isGLMModel`") describes a code path that does not exist in this repo and could not satisfy row 7 anyway.
- The two mandates are mutually exclusive as written: (a) "classifyModel builds it from today's three helpers (so it is behavior-identical)" + hard constraint "Part A adds zero behavior changes" (making deepseek/glm detection strip-aware would newly apply DeepSeek/GLM quirks — forced tool_choice strip, json_object, flatten, reasoning replay — to prefixed ids that get none today), vs (b) the matrix as the "oracle". The spec's own wording says the matrix maps each id "to the exact profile field values the current code would produce", which the four rows do not.
- Decision needed (maintainer/plan author): (1) amend the 4 rows to current semantics (behavior-identical Part A preserved; matrix stays an honest pin of "what the code does"); or (2) amend the brief/spec/plan to declare strip-all detection for deepseek/glm intended, accept the Part A behavior change explicitly (it then propagates to all Task 2 call sites), and implement `stripAllProviderPrefixes`-based detection inside `classifyModel`; or (3) another ruling the author prefers. Not done unilaterally: editing the rows is forbidden by the brief/context; silently changing helper semantics violates the global Part A constraint.
- Review findings: none beyond the above; no rows were altered, no helpers were changed, no call sites migrated (Task 2 scope) except the two `profile.isDeepSeek` readers left bit-identical via the legacy field.
- Deferred/limitations: `isDeepSeek` legacy field to be deleted in Task 2 when the two `anthropic_to_openai.go` call sites migrate to `thinkingStyle == "deepseek"`; `signatureScheme`/`streamTailSynthesize` are set to `"none"`/zero and unread (per brief; Task 4 wires detection); e2e suite remains red for the pre-existing env reason above; test count delta is exactly +1 (the matrix, as the brief requires).
