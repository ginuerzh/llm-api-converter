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

### Task 1: Profile definition + equivalence pin — 2026-11-16 — DONE (maintainer ruling: option 1 — amend the 4 rows to current helper semantics)
- Repo: `llm-api-converter` (branch `main`, base HEAD `883b0a0`).
- Ruling: the BLOCKED record's decision point was resolved by the plan author as **option (1)** — amend the 4 provider-prefixed deepseek/glm matrix rows to current helper semantics instead of changing detection behavior. Plan/brief/spec were corrected in `b394f1c` (matrix rows reflect current prefix behavior: deepseek/glm do not strip) and `883b0a0` (drop stale single-strip derivation paragraph). Part A remains behavior-identical.
- Files actually changed (vs base): `convert/quirks_equivalence_test.go` (new; the 4 rows `ollama/deepseek-v4-flash`, `zhipu/deepseek-v3`, `zhipu/glm-51`, `foo/bar/glm-4.7` updated from `deepseek/glm` to `none | allow | pass | false | none`; other 13 rows verbatim from the corrected brief), `convert/types.go` (extended `modelProfile` with the new unexported fields per the brief's struct sketch; `classifyModel` rewired to build the profile from today's helpers; `stripTopK = isOpenAIReasoningModel(stripAllProviderPrefixes(model))`; `isDeepSeek` kept as a deprecated legacy field for the two unmigrated `anthropic_to_openai.go` call sites; `isOpenAI` dropped — zero call sites), and this plan file (this record).
- Commands and results:
  - `GOWORK=off go test ./convert/ -run TestModelProfileMatchesExpectedMatrix -count=1 -v` → PASS (17/17 subtests)
  - `GOWORK=off go vet ./...` → clean (no output; convert/engine/rewriter/root)
  - `GOWORK=off go test -p 1 ./... -count=1` → `convert`, `engine`, `rewriter` ok; `tests/e2e` FAIL — pre-existing environment failure (cmd/gost builds against sibling `github.com/go-gost/x/dashboard`, unresolvable under GOWORK=off); unrelated to this task; not fixed (per plan constraint).
- Test count delta: +1 (the matrix test only), as the brief requires.
- Review findings: none — no helper semantics changed, no call sites migrated (Task 2 scope), the two `profile.isDeepSeek` readers left bit-identical via the legacy field.
- Deferred/limitations: `isDeepSeek` legacy field to be deleted in Task 2 when the two `anthropic_to_openai.go` call sites migrate to `thinkingStyle == "deepseek"`; `signatureScheme`/`streamTailSynthesize` set to `"none"`/zero and unread (per brief; Task 4 wires detection); e2e suite remains red for the pre-existing env reason; behavior-identical Part A confirmed by the unchanged suite results.
- SHA: 9e9702b (this commit)

### Task 2: migrate request-side quirks in `anthropic_to_openai.go` — 2026-11-16 — DONE
- Repo: `llm-api-converter` (branch `main`, base HEAD `da05024`).
- Files actually changed (vs base): `convert/anthropic_to_openai.go` (all request-side quirk call sites migrated to profile fields; `anthropicToolChoiceToOpenAi` and `toolChoiceInstruction` now take `forceToolChoice string` instead of `model string`; `convertAnthropicMessage` / `convertAnthropicAssistantMessage` now take `profile modelProfile`, threaded from the single `classifyModel(outputModel)` resolution at the top of `convertAnthropicRequestToOpenAI`; `reasoningModel` is `profile.reasoningMaxTokens == "max_completion_tokens"`; top_k gate is `profile.stripTopK` (the strip-all site); thinking gate is `profile.thinkingStyle == "deepseek" | "glm"`; json_schema→json_object branch is `profile.jsonSchemaMode == "json_object"`; GLM flatten gate is `profile.contentFlatten`; `replaysReasoning(model)` call sites are `profile.reasoningReplay != "none"`; the DeepSeek placeholder fallback is `profile.reasoningFallback == "placeholder"`; `profile.isDeepSeek` readers: 2 → 0), and this plan file. No test edits, no assertion changes.
- Commands and results:
  - `GOWORK=off go vet ./...` → clean (convert/engine/rewriter/root)
  - `GOWORK=off go test -p 1 ./... -count=1` → `convert`, `engine`, `rewriter` ok; `tests/e2e` FAIL — pre-existing environment failure (cmd/gost builds against sibling `github.com/go-gost/x/dashboard`, unresolvable under GOWORK=off); unrelated to this task; unchanged before/after; not fixed (per plan constraint).
  - `GOWORK=off go test ./convert/ -run 'TestModelProfileMatchesExpectedMatrix|DeepSeek|GLM|ReasoningCache|Placeholder' -count=1 -v` → all PASS (17/17 matrix rows; DeepSeek `json_object`+instruction pin, DeepSeek forced-tool_choice strip pins, DeepSeek placeholder-only-for-DeepSeek pin, GLM replay/no-placeholder pin).
- Behavior: byte-identical by construction — every guard expression swapped 1:1 to the profile field `classifyModel` derives from the same helpers; profile resolved once from the resolved output model and threaded down; `reasoningMaxTokens` (single-strip) and `stripTopK` (strip-all) keep their respective distinctions exactly as pinned in Task 1.
- Decisions/notes:
  - `replaysReasoning` helper kept in this file: `TestP1_ReplaysReasoningTargets` (`anthropic_openai_p1_test.go`) still calls it, and the commit constraint allows only `convert/anthropic_to_openai.go` in the commit (the mechanical test call-site migration belongs to Task 5's repo-wide sweep). Zero call sites remain in non-test code.
  - `isDeepSeek` legacy field: zero readers remain after this task (this file's two readers migrated). Per the commit-file-set constraint (`convert/anthropic_to_openai.go` + plan record only), the field definition + `classifyModel` setter + comment in `types.go` were left in place as zero-reader dead code; deletion is handed to Task 5's repo-wide sweep rather than silently widening this task's file set.
  - `stripAllProviderPrefixes` retained (used by `classifyModel`); `supportsTopKPassthrough`, `thinkingToOpenAi`, `thinkingToOpenAiGLM`, and `reasoningEffortToOpenAi` (pass-through, unchanged) untouched.
  - Test count delta: 0 (no test edits).
- Review findings: none — no branch added/removed, no logic moved, comments updated only where they cited the old guard expression; `go vet` clean.
- Deferred/limitations: `isDeepSeek` field definition in `types.go` (see above); `signatureScheme`/`streamTailSynthesize` still unread (Task 4 wires `signatureScheme`); `replaysReasoning` kept for the p1 test; e2e suite remains red for the pre-existing env reason.
- SHA: 64bfc14 (this commit)

### Task 3: migrate replay thinking in `openai_to_anthropic.go` — 2026-11-16 — DONE
- Repo: `llm-api-converter` (branch `main`, base HEAD `21820fc`).
- Files actually changed (vs base): `convert/openai_to_anthropic.go` (`convertAssistantContent` resolves `profile := classifyModel(model)` once at the top; the `replayThinking` gate is now `profile.reasoningReplay == "deepseek"`; no signature changes, no new helpers, `classifyModel` called exactly once per call; the other call site (`convertMessageContent`, and `responses.go`'s direct call) unchanged — profile resolved inside the owning function), and this plan file. No test edits, no assertion changes.
- Step 2 verification (GLM sibling gate): no DeepSeek-GLM divergence to fix. The gate derives from `isDeepSeekModel` via `classifyModel`, so GLM targets (`reasoningReplay == "glm"`) still do NOT replay thinking here — identical to before; the site was not extended to GLM, per the brief.
- Commands and results:
  - `GOWORK=off go test -p 1 ./... -count=1` → `convert`, `engine`, `rewriter` ok; `tests/e2e` FAIL — pre-existing environment failure (cmd/gost builds against sibling `github.com/go-gost/x/dashboard`, unresolvable under GOWORK=off). Verified pre-existing by stashing this change and re-running `go test ./tests/e2e/ -count=1` on the clean tree: identical failure. Unrelated to this task; not fixed (per plan constraint).
  - `GOWORK=off go vet ./...` → clean (no output; convert/engine/rewriter/root/tests).
- Behavior: byte-identical by construction — the guard expression swapped 1:1 to the profile field `classifyModel` derives from the same `isDeepSeekModel` helper; profile resolved once from the same `model` argument the old gate used.
- Decisions/notes:
  - Minimal-diff choice: resolving the profile inside `convertAssistantContent` (the owning function) instead of threading `modelProfile` through `convertMessageContent` / the `responses.go` call site — smaller honest diff, no signature changes anywhere.
  - Zero references to `isDeepSeekModel` remain in this file after the swap; other files' references (`replaysReasoning` in `anthropic_to_openai.go`, `contract_test.go`) are Task 5 scope and untouched.
  - Test count delta: 0 (no test edits).
- Review findings: none — one guard expression swapped, one `classifyModel` resolution added, comments left accurate (they cite the replay rule, not the old guard); `go vet` clean.
- Deferred/limitations: e2e suite remains red for the pre-existing env reason; `isDeepSeek` legacy field and `replaysReasoning` test-file call sites remain for Task 5's sweep; `signatureScheme`/`streamTailSynthesize` still unread (Task 4 wires `signatureScheme`).
- SHA: 69452fe (this commit)

### Task 4: Part B — Gemini replay sentinel seam — 2026-11-16 — DONE
- Repo: `llm-api-converter` (branch `main`, base HEAD `26207fe`).
- Files actually changed (vs base): `convert/types.go` (`OpenAIToolCall` gains `ExtraContent map[string]any \`json:"extra_content,omitempty"\``; `classifyModel` sets `signatureScheme = openai_toolcall_gemini` iff `strings.Contains(strings.ToLower(model), "gemini")` — covers `gemini-4-hydrogen` and `models/gemini-2.5-pro` without prefix stripping, per the brief), `convert/anthropic_to_openai.go` (new `geminiToolReplay` seam: no-ops unless `profile.signatureScheme == "openai_toolcall_gemini"`, then for every assistant message with `tool_calls` sets each tool call's `ExtraContent` to `{"google":{"thought_signature":"skip_thought_signature_validator"}}`; called just before `json.Marshal(oai)` in `convertAnthropicRequestToOpenAI`, after sanitize/flatten so coalesced tool calls are covered), `convert/gemini_tool_replay_test.go` (new, 4 fixtures), and this plan file (this record). No other file touched; no outbound schema changes beyond the one `omitempty` field.
- Fixture 4 (`TestGeminiToolReplaySentinelNeverReadsSessionStore`) was hosted, not dropped: the convert path takes a `*ReasoningCache`, so the test builds one over a `fatalReasoningStore` double whose `Get`/`Set` `t.Fatalf` on any call; a real `SessionStore` with `SID: "sid-trap"` and `Direction: "request"` is passed alongside and convert completes. `SessionStore` is a concrete struct (not an interface) and cannot be doubled without changing the seam — the fatal `ReasoningStore` is the loud assertion. All 4 tests PASS.
- Commands and results:
  - `GOWORK=off go test ./convert/ -run 'ExtraContent|GeminiToolReplay' -count=1 -v` → 4/4 PASS (after the expected FAIL before implementation).
  - `GOWORK=off go vet ./...` → clean (no output).
  - `GOWORK=off go test -p 1 ./... -count=1` → `convert`, `engine`, `rewriter` ok; `tests/e2e` FAIL — pre-existing environment failure (cmd/gost builds against sibling `github.com/go-gost/x/dashboard`, unresolvable under GOWORK=off); unchanged before/after; not fixed (per plan constraint).
- Behavior: non-gemini targets are byte-compatible (`extra_content` absent, asserted by absence in raw bytes); only assistant turns with `tool_calls` are mutated (user prose containing the substring `tool_use` is asserted untouched); the seam reads/writes nothing besides `tool_calls` (no caches, no `SessionStore`, no SID, no real thought_signature capture).
- Decisions/notes:
  - Wire format pinned and asserted on raw bytes: `extra_content` (snake_case) top-level on the tool_call object, nested `google`, nested `thought_signature` — exactly where Google's live API accepts it.
  - The sentinel bypasses Google's validator; it is NOT a captured signature. Real per-tool_call signatures remain a follow-on design (keyed by tool_call id, not SID).
  - `isOpenAIStyleModel` still lists `gemini-` (pre-existing); no other outbound path reads `signatureScheme`; `streamTailSynthesize` remains reserved/unread.
  - Test count delta: +4.
- Review findings: none — seam is a single pure mutation of assistant `tool_calls`; no new dependencies; English only.
- Deferred/limitations: `tests/e2e` remains red for the pre-existing sibling-repo reason; maintainer smoke (live Claude Code through the fable route, Task 6) remains for the plan's verification step; amendment to git history avoided (see commit note).
- SHA: 031f9ea

### Task 5: delete legacy model-name helpers repo-wide — 2026-11-16 — DONE
- Repo: `llm-api-converter` (branch `main`, base HEAD `031f9ea`).
- Files actually changed (vs base): `convert/types.go` (`modelProfile` gains `topKPassthrough bool`; `isDeepSeek` legacy field deleted; `classifyModel` now computes `reasoningMaxTokens`, `stripTopK`, `topKPassthrough` inline via the new unexported `reasoningModelVerdict` helper (o1/o3/o4/gpt-5 prefixes, case-insensitive, single- vs strip-all prefix handling preserved); `isDeepSeekModel`, `isGLMModel`, `isOpenAIStyleModel`, `isOpenAIReasoningModel` deleted; stale comments citing the legacy helpers rewritten), `convert/anthropic_to_openai.go` (`replaysReasoning` deleted — its only caller was the p1 test, migrated to the profile; `supportsTopKPassthrough` kept as the profile's only reader via `classifyModel` (top-k gate now reads `profile.topKPassthrough`); doc comment updated), `convert/contract_test.go` (2 call sites: `!isDeepSeekModel(m)` → `classifyModel(m).reasoningReplay != "deepseek"`, assertions unchanged), `convert/anthropic_openai_p1_test.go` (`TestP1_ReplaysReasoningTargets`: `replaysReasoning(m)` → `classifyModel(m).reasoningReplay != "none"`; `TestP1_ReasoningModelDetection`: `isOpenAIReasoningModel(m)` → `classifyModel(m).reasoningMaxTokens == "max_completion_tokens"`; assertions and truth tables unchanged), `convert/quirks_equivalence_test.go` (doc comment drops `isOpenAIReasoningModel` reference; table oracle unchanged — test already referenced zero legacy helpers in code), and this plan file. No behavior change; `geminiToolReplay` untouched.
- Commands and results:
  - `grep -rn "isDeepSeekModel\|isGLMModel\|isOpenAIStyleModel\|isOpenAIReasoningModel" --include=*.go .` → no output (exit 1).
  - `GOWORK=off go vet ./...` → clean (no output).
  - `GOWORK=off go test -p 1 ./... -count=1` → `convert`, `engine`, `rewriter` ok; `tests/e2e` FAIL — pre-existing environment failure (cmd/gost builds against sibling `github.com/go-gost/x/dashboard`, unresolvable under GOWORK=off); unchanged before/after; not fixed (per plan constraint).
  - `GOWORK=off go test ./convert/ -run 'TestModelProfileMatchesExpectedMatrix|TestP1_ReplaysReasoningTargets|TestP1_ReasoningModelDetection|DeepSeek|GLM|GeminiToolReplay|ExtraContent' -count=1 -v` → all PASS.
- Behavior: byte-identical by construction — every guard expression swapped 1:1 to the profile field `classifyModel` derives from the same predicate; `reasoningModelVerdict` is the exact body of the deleted `isOpenAIReasoningModel` minus the single StripProviderPrefix (callers apply it), preserving the single-strip (`reasoningMaxTokens`) vs strip-all (`stripTopK`) distinction pinned by `TestModelProfileMatchesExpectedMatrix` (17/17 rows PASS).
- Decisions/notes:
  - `supportsTopKPassthrough` kept under the profile per the task carryover: its sole reader is now `classifyModel` (via the new `topKPassthrough` field); the top-k call site reads `profile.topKPassthrough` instead of the free function. Grep-verified before/after.
  - `replaysReasoning` deleted per carryover; its only caller (`TestP1_ReplaysReasoningTargets`) migrated to `classifyModel(m).reasoningReplay != "none"`; test name and assertion truth table unchanged.
  - `goToolChoiceToOpenAi` carryover: no such symbol exists anywhere in the tree (checked current tree across all file types, full `git log -S`, and a bounded search of the sibling `cuttle` module) and no stale "see" doc comment referencing tool-choice helpers exists. Doc-comment verification referenced the surviving "see toolChoiceInstruction" style sentence; no assertion edit was made. One stale tool-choice comment did exist — `convert/anthropic_to_openai.go`'s "Tool choice instruction for DeepSeek" comment, since the forced-tool-choice strip is profile-driven, not DeepSeek-specific — and was corrected in the final-review fix wave (see the fix-wave record below). Adjacent stale references that DID exist (`quirks_equivalence_test.go` doc citing `isOpenAIReasoningModel`; `types.go` doc citing `isDeepSeekModel`/`isOpenAIStyleModel`) were rewritten as part of the sweep.
  - Equivalence matrix: code had zero legacy-helper references before and after; doc comment now describes the derivation without naming legacy helpers.
  - Test count delta: 0 (migrations only, no assertions changed).
- Review findings: none — mechanical 1:1 guard swaps; assertions untouched; `go vet` clean; `gofmt -l` flags only pre-existing unformatted files (unchanged by this task).
- Deferred/limitations: `tests/e2e` remains red for the pre-existing sibling-repo reason; `streamTailSynthesize` remains reserved/unread; real per-tool_call thought_signatures remain a follow-on design.
- SHA: 312aad9

### Task 6: docs + records + verification bar — 2026-11-16 — DONE (maintainer verification handed off; not performed)
- Repo: `llm-api-converter` (branch `main`, base HEAD `312aad9`).
- Files actually changed (vs base): `docs/QUIRKS.md` (new; the Model profiles and quirks guide — preface on why the layer exists, "How to read the tables" prefix-matching rules, the full quirk inventory table (12 delete-condition rows), the `SignatureScheme` section (replay-layer, not a quirk — `gemini-thought-signature-sentinel`), "Deliberately not quirks" (Gemini-native thinking config, effort-cap mappings, `topKPassthrough` allowlist, reserved `streamTailSynthesize`), and a "How to add a new quirk" / "Deleting a quirk" checklist; spec doc path cited at the top as the source of incidents and constraint reasons), and this plan file (this record). No `.go` file touched; no other docs touched.
- Commands and results: docs-only — **not run** (`go vet`/`go test` not invoked; no code changed, so suite state is identical to Task 5's green baseline for `convert`/`engine`/`rewriter`; `tests/e2e` remains red for the pre-existing sibling-repo reason, unchanged and unrelated). Line references in `docs/QUIRKS.md` were verified against the tree by grep/sed at write time (e.g. `classifyModel` at `convert/types.go:332-363`, `geminiToolReplay` at `convert/anthropic_to_openai.go:392-406`, `OpenAIToolCall.ExtraContent` at `convert/types.go:105-115`, `replayThinking` at `convert/openai_to_anthropic.go:578`).
- **Maintainer verification script (handoff — NOT performed by this task; maintainer's call, per the plan's Step 3):**
  1. Merge to `main` (or hold in `work/model-profile` — maintainer's call; the plan touches only this repo).
  2. Tag `llm-api-converter` per their release flow, bump cuttle `go.mod`, redeploy. (Open decision #3 — explicitly outside this plan.)
  3. Live Claude Code session through the `fable` route with at least one agentic tool-use turn; expected: `message_stop` + non-trivial token count in stats (`/api/stats`, not just a 200).
- Acceptance criteria for the maintainer gate: (a) the agentic tool-use turn ends with `message_stop` in the SSE stream; (b) non-trivial input/output token usage visible in `/api/stats` for the session; (c) no repeat of the 2026-11-16 `Function call is missing a thought_signature in functionCall parts` 400 on the replayed tool call.
- What the verification covers: only Part B's replay seam (`geminiToolReplay`, `SignatureScheme = openai_toolcall_gemini`). Part A is byte-identical by construction (Tasks 1-5 equivalence pins) and needs no live check.
- Decisions/notes:
  - The spec doc path is cited at the top of `docs/QUIRKS.md` as the source of incidents and constraint reasons, per the task brief.
  - `docs/QUIRKS.md` names the sentinel scheme `gemini-thought-signature-sentinel` and explicitly scopes it as a replay-layer behavior, not a quirk, per the brief.
  - Real per-tool_call signature round-trip remains a named follow-on (ID-keyed `ReasoningCache`-style store, NOT SID-keyed); nothing in this task wires it.
  - Commit flow per the task constraint: `docs/QUIRKS.md` committed as `docs: model profiles and quirks guide`, then this record was folded into the same commit via `git commit --amend` — no separate record commit; no other files added. Nothing pushed (branch ahead of `origin/main`; push remains the maintainer's call with the merge).
- Review findings: none — docs-only; every "Lives at" reference was grep-verified; the two inaccuracies caught in self-review (effort-cap bullet conflated the Responses/Gemini effort mappings; the `x/service/service.go:306` SID reference was ambiguous between repos) were fixed before the record was written.
- Deferred/limitations: maintainer verification (above) is explicitly NOT performed and NOT claimed; `tests/e2e` red (pre-existing); `streamTailSynthesize` reserved/unread; real-signature follow-on and the cuttle release/deploy steps remain out of this plan (Open decision #3).
- SHA: bd47944 — the `docs: model profiles and quirks guide` commit amended to fold in this record

### Fix wave: final-review comment and record fixes — 2026-11-16 — DONE
- Repo: `llm-api-converter` (branch `main`, base HEAD `bd47944`). One commit on `main`; nothing pushed.
- Files actually changed (vs base): `convert/anthropic_to_openai.go` (finding 1 — the stale `// Tool choice instruction for DeepSeek (softened to system instruction).` comment reworded to `// Tool choice instruction for models whose profile strips forced tool choice (softened to system instruction).`), `docs/superpowers/specs/2026-11-16-model-profile-quirks-design.md` (finding 2 — Part B §2 wording: the Gemini substring match is now described as `match by substring on the lowercased id (literal presence; do not prefix-strip)`, replacing the incorrect "gated after `StripProviderPrefix`" claim; no code changed), `docs/QUIRKS.md` (finding 3 — preface softened from "computed once per request" to the real contract: computed once per request on the Anthropic→OpenAI path, per assistant message on OpenAI→Anthropic, per thinking block in tests only; rest of preface unchanged), and this plan file (finding 4 — Task 5's "all tool-choice doc comments were verified accurate" claim reworded to honest wording: verification referenced the surviving "see toolChoiceInstruction" style sentence, no assertion edit was made, one stale comment did exist and was corrected in this fix wave, cross-referenced to finding 1; Tasks 4/5/6 `SHA: (this commit)` self-references replaced with the concrete SHAs `031f9ea` (Task 4), `312aad9` (Task 5), `bd47944` (Task 6); Tasks 1-3 historical strings unchanged; this record). No assertions, tests, or Go logic changed.
- Commands and results: docs + one comment only — `GOWORK=off go build ./convert/` → clean (comment-only change; confirms the edit parses). No other code commands run.
- Decisions/notes:
  - Finding 2 is doc-only; the implementation (`convert/types.go:362`, `strings.Contains(strings.ToLower(model), "gemini")` on the raw id) stays as-is and the spec now matches it.
  - Finding 4 rewording cross-references finding 1 rather than restating it, so the plan tells the truth about what was verified at the time vs fixed here.
  - Explicitly out of scope and untouched, per the final whole-branch review triage: matrix ride-along columns, the unconditional `ExtraContent` overwrite (follow-on), spec/test renames, maintainer smoke-check status (maintainer outstanding by design).
- Review findings: none beyond the listed findings — comment reworded only; no logic, assertion, or file-set widening.
- Deferred/limitations: unchanged from Task 6 (maintainer verification handoff, `tests/e2e` red for the pre-existing sibling-repo reason, `streamTailSynthesize` reserved/unread, real-signature follow-on).
- SHA: `d243905` — the fixes and this record landed together in `docs(plans): final-review comment and record fixes`.

## Hotfix record — 2026-11-17 (fable OpenAI-Chat passthrough 400)

- Status: implemented, pending release.
- Root cause (production evidence, request id `dbtq3bqu9fis73diem80`, 00:37 UTC): OpenCode CLI (`opencode/latest/2.3.0/cli`) speaks `/v1/chat/completions` → `inbound_protocol=openai-chat`; auto routes gemini-4-hydrogen via chat→chat `passthrough()`, which only rewrote the model field. The Part B sentinel seam lived exclusively in `convertAnthropicRequestToOpenAI`, so the 1.4MB replay body carried 0 `extra_content` and Google rejected 583 tool calls with `HTTP 400 Function call is missing a thought_signature`.
- Fix: `geminiToolReplayPassthrough()` in `convert/convert.go` injects the same sentinel into assistant `tool_calls` on both request-passthrough branches when `classifyModel(targetModel).signatureScheme == openai_toolcall_gemini`; unknown fields preserved via map mutation; non-gemini and no-tool-call bodies stay byte-identical. Production const `thoughtSignatureSentinel` shared with the existing Anthropic seam.
- Tests (TDD — watched RED then GREEN): `TestGeminiToolReplayOpenAIChatPassthrough` (2 tool calls → 2 sentinels), `TestGeminiToolReplayPassthroughLeavesNonGeminiAndUnknownFieldsAlone`. Full `convert/engine/rewriter` green; `go vet ./...` clean. `tests/e2e` remains pre-existing red (sibling `x/dashboard` under GOWORK=off).
- Deferred: real signature capture (ID-keyed, follow-on as before); release + cuttle bump + deploy + live fable smoke are the remaining rollout steps.
- SHA: `601fdf4` (hotfix code, tests and record). Released as converter `v0.7.1`; cuttle bump `9ba62ee2` deployed at 00:51 UTC; live OpenAI-Chat-inbound replay 200 (non-stream + SSE) — see memory note [[llm-gw-gemini-thought-signature]].
