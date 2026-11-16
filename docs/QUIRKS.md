# Model profiles and quirks

Source of incidents and constraint reasons: [`docs/superpowers/specs/2026-11-16-model-profile-quirks-design.md`](superpowers/specs/2026-11-16-model-profile-quirks-design.md). Implementation record: [`docs/superpowers/plans/2026-11-16-model-profile-quirks.md`](superpowers/plans/2026-11-16-model-profile-quirks.md).

## Why this layer exists

This library used to scatter `isDeepSeekModel` / `isGLMModel` / `isOpenAIReasoningModel`-style conditionals across the pairwise converters. Every model-family-dependent behavior now lives behind one typed `modelProfile` computed once per request from the target model name (`classifyModel`), and attaches to a fixed seam in the conversion path — so a new provider family is one field plus one `classifyModel` arm plus one equivalence-pin row, not a scatter of ad-hoc conditionals. The layer is honest by construction: every quirk records why it exists, where it applies, and is deleted when it no longer applies.

Detection is static, model-name only (option (a)); pairwise conversion is final; there is no user-configurable quirks key. The profile never encodes cross-protocol behavior — each seam has exactly two owners (the conversion direction and the quirk).

## How to read the tables

- **Delete-condition names.** Quirk names are deletion conditions (`deepseek-tool-choice-strip`, not `deepseek-workaround-2`). When the upstream API accepts the rejected shape, delete the row.
- **Prefix behavior matters.** `deepseek` / `glm-` match the verbatim (lowercased) prefix with **no** provider-prefix stripping — `ollama/deepseek-v4-flash` classifies as `none`, deliberately. OpenAI reasoning ids use a single `StripProviderPrefix` for `max_completion_tokens` but `stripAllProviderPrefixes` for `top_k` (`foo/bar/o3-mini` strips fully). Gemini uses a case-insensitive substring anywhere in the id. All of this is pinned byte-for-byte by `TestModelProfileMatchesExpectedMatrix` (`convert/quirks_equivalence_test.go`).
- "Lives at" references are current as of 2026-11-16 (post-refactor); grep the field name to re-locate.

## Quirk inventory

| Quirk (delete-condition) | Applies to | Why (driver incident, date) | Lives at | Deleted when (upstream link) |
|---|---|---|---|---|
| `deepseek-tool-choice-strip` | `deepseek*` prefix | DeepSeek rejects forced `tool_choice`; the driver incident is the deepseek-dadax 400 (`messages unavailable for thinking; choose thinking instead`) cited in the spec. Anthropic `any`/`tool` are dropped and softened to a system instruction. | `convert/types.go:317` (`forceToolChoice`); gate `convert/anthropic_to_openai.go:83,88` (`anthropicToolChoiceToOpenAi`, field passed from `:367`) | DeepSeek accepts forced `tool_choice` on Chat Completions — <https://api-docs.deepseek.com/api/create-chat-completion> |
| `deepseek-tool-choice-instruction` | `deepseek*` | The system-instruction soft copy of the row above (same incident). | `convert/anthropic_to_openai.go:104-108` (`toolChoiceInstruction`), wired at `:298` | With `deepseek-tool-choice-strip` |
| `deepseek-json-schema-to-json-object` | `deepseek*` | DeepSeek rejects `response_format:{"type":"json_schema"}` on `/chat/completions` (400 `This response_format type is unavailable now`; accepted only on `/responses`). Driver incident **2026-11-11**: cuttle route `dadax` / node `deepseek-dadax` returned 400 on every Claude Code session-title call (commit `a9a7347`). Schema moves into a system instruction ("json" must appear in the prompt anyway). | `convert/types.go:318` (`jsonSchemaMode`); `convert/types.go:154` (`jsonSchemaInstruction`); branch `convert/anthropic_to_openai.go:273-291` | DeepSeek accepts `json_schema` on Chat Completions — <https://api-docs.deepseek.com/api/create-chat-completion> |
| `deepseek-thinking-reasoning-effort` | `deepseek*` | DeepSeek thinking mode uses `reasoning_effort` (`thinking` + effort), not GLM's `thinking.budget_tokens` shape. Driver: 2026-06-28 reference-bridge alignment (`31b22da`); spec §The quirk inventory row 6. | `convert/types.go:316` (`thinkingStyle`); `convert/anthropic_to_openai.go:149` (`thinkingToOpenAi`), `:185` (`reasoningEffortToOpenAi`), gate `:259-265` | DeepSeek accepts a standard reasoning/thinking field — <https://api-docs.deepseek.com/guides/thinking_mode> |
| `glm-thinking-budget-tokens` | `glm-*` | GLM thinking mode is structured `thinking:{type,budget_tokens}`. Driver incident: **2026-06-29** GLM-5.2 compatibility (`d9e781f`); spec row 6. | `convert/types.go:316`; `convert/anthropic_to_openai.go:161` (`thinkingToOpenAiGLM`), gate `:266-268` | GLM aligns with a standard reasoning field — <https://docs.z.ai/guides/capabilities/thinking-mode> |
| `glm-content-flatten` | `glm-*` | GLM API accepts only string content, not multi-part content arrays. Driver incident: **2026-06-29** GLM-5.2 (`d9e781f`); spec row 8. | `convert/types.go:319` (`contentFlatten`); gate `convert/anthropic_to_openai.go:340`; impl `convert/openai_to_anthropic.go:421` (`flattenContentForGLM`) | GLM accepts multi-part content arrays — <https://open.bigmodel.cn/dev/api> |
| `openai-reasoning-max-completion-tokens` | `o1`/`o3`/`o4`/`gpt-5` prefixes (single-strip) | OpenAI reasoning models reject `max_tokens` and require `max_completion_tokens`; sampling params (`temperature`/`top_p`/`top_k`) are dropped for the same reason. Spec rows 5, 13. | `convert/types.go:315` (`reasoningMaxTokens`), derived in `convert/types.go:356` + `:368` (`reasoningModelVerdict`); used `convert/anthropic_to_openai.go:215`, `:217-245` | OpenAI reasoning models accept `max_tokens` — <https://platform.openai.com/docs/guides/reasoning> |
| `openai-reasoning-strip-top-k` | `o1`/`o3`/`o4`/`gpt-5` prefixes (strip-**all** path) | OpenAI reasoning models reject `top_k`. Note the strip-all path is intentional: `foo/bar/o3-mini` must still match (`models/o3-mini`-style ids are common). Spec row 4. | `convert/types.go:313` (`stripTopK`), derived `convert/types.go:357`; gate `convert/anthropic_to_openai.go:239-243` | OpenAI reasoning models accept `top_k` — <https://platform.openai.com/docs/guides/reasoning> |
| `deepseek-reasoning-replay` | `deepseek*` | DeepSeek V4 requires a `reasoning_content` round trip on replayed tool calls (else 400). Anthropic→OpenAI re-injects reasoning verbatim (or from `ReasoningCache.GetBest`); OpenAI→Anthropic replays `reasoning_content` as Anthropic thinking blocks. Driver incident: **2026-06-28** ReasoningCache for DeepSeek V4 (`7475c7b`); spec rows 9-11. | `convert/types.go:320` (`reasoningReplay`); `convert/anthropic_to_openai.go:630` (verbatim re-inject), `:649-655` (cache restore); `convert/openai_to_anthropic.go:565,578` (thinking-block gate) | DeepSeek drops the `reasoning_content` round-trip requirement — <https://api-docs.deepseek.com/guides/thinking_mode> |
| `glm-reasoning-replay` | `glm-*` | Same round-trip requirement on the Anthropic→OpenAI leg (`reasoningReplay == "glm"` passes the `!= "none"` gates). Deliberately **not** mirrored on the OpenAI→Anthropic leg, which stays DeepSeek-only (`convert/openai_to_anthropic.go:578`) — do not extend. Spec rows 9-10. | `convert/types.go:320`; gates `convert/anthropic_to_openai.go:630,649` | GLM drops the `reasoning_content` round-trip requirement — <https://docs.z.ai/guides/capabilities/thinking-mode> |
| `deepseek-reasoning-placeholder` | `deepseek*` | When the cache is empty but tool_calls are replayed, DeepSeek still requires `reasoning_content`; a fixed placeholder is sent instead of nothing. Driver incident: **2026-06-28** (`7475c7b`, placeholder fallback); spec rows 9-10. | `convert/types.go:321` (`reasoningFallback`); const `convert/anthropic_to_openai.go:11` (`placeholderReasoning`); branch `:657-662` | DeepSeek tolerates absent `reasoning_content` on tool-call replay — <https://api-docs.deepseek.com/guides/thinking_mode> |

## Signature scheme (replay-layer, not a quirk)

`gemini-thought-signature-sentinel` — `SignatureScheme = openai_toolcall_gemini`.

This is a **replay-layer behavior**, carried by the profile for symmetry with the quirks, not a model-family quirk: it runs only inside `convertAnthropicRequestToOpenAI`, touches only assistant `tool_calls`, and reads/writes nothing else (no caches, no `SessionStore`, no SID).

| Aspect | Detail |
|---|---|
| Applies to | Any target whose model id contains `gemini` (case-insensitive substring, no prefix strip) — `gemini-4-hydrogen`, `models/gemini-2.5-pro`, … |
| Driver incident | **2026-11-16**: the cuttle `fable` route (gemini-4-hydrogen via Google's OpenAI-compatible endpoint) returned HTTP 400 `Function call is missing a thought_signature in functionCall parts` on every agentic tool-use turn from Claude Code. The sentinel `skip_thought_signature_validator` was verified live 2026-11-16 (same replay → HTTP 200). |
| Behavior | Every replayed assistant `tool_call` gets top-level `extra_content` `{"google":{"thought_signature":"skip_thought_signature_validator"}}` — Google's documented validator bypass. Never a captured signature. |
| Lives at | `convert/types.go:322` (`signatureScheme`, detection `:359-361`); seam `convert/anthropic_to_openai.go:382` (call), `:387-406` (`geminiToolReplay`); wire field `convert/types.go:105-115` (`OpenAIToolCall.ExtraContent`, snake_case `extra_content`, `omitempty`); fixtures `convert/gemini_tool_replay_test.go` (sentinel path, non-gemini isolation, assistant-turns-only, no-SessionStore) |
| Deleted when | Google returns real per-part signatures for replayed `functionCall` parts and accepts them without a validator bypass — i.e. the compat endpoint no longer needs the sentinel (<https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures>) |

Real per-tool_call signature round-trip is a **named follow-on** (ID-keyed `ReasoningCache`-style store, *not* SID-keyed — SID is per-connection, see `x/service/service.go:306` in the go-gost `x/` module); see the spec's Part B and maintainer decision #1. Sentinel-on-every-call is deliberately not worse than today (today = 400).

## Deliberately not quirks

- **Gemini-native thinking config** maps directly to `thinkingConfig` in the Gemini adapters (`anthropic_to_gemini.go`, `openai_to_gemini.go`) — native adapter behavior, not profiled.
- **Effort-cap mappings** (e.g. `xhigh → max` in `reasoningEffortToOpenAi`, `convert/anthropic_to_openai.go:185-197`) — identical across providers, not model-family specific.
- **`topKPassthrough` allowlist** (`convert/types.go:314,358`, `convert/anthropic_to_openai.go:133`) — provider allowlist (groq/openrouter/together/fireworks/vllm/ollama/lmstudio) for `top_k`, not a model-family quirk.
- **`streamTailSynthesize`** — reserved field, unread; stream-tail synthesis is a follow-on design (see spec "Known follow-on candidates").

## How to add a new quirk

1. Decide it is a capability (a profile field), not a vendor conditional at a call site. Fields are unexported; zero value = off.
2. Add the field to `modelProfile` in `convert/types.go` with a doc comment naming the incident date and the delete condition.
3. Add the `classifyModel` arm. Pick and document the prefix class (verbatim / single-strip / strip-all / substring) — do not mix them.
4. Add one row to `TestModelProfileMatchesExpectedMatrix` (`convert/quirks_equivalence_test.go`), pinning both prefix-strip variants for OpenAI-reasoning-style ids.
5. Wire the seam(s) that read it; add or extend fixtures.
6. Add a row to the inventory above.
7. **Deleting a quirk:** remove the field + the `classifyModel` arm + the seam branch + the matrix row + this row together. The profile is the definition — a stale entry has nowhere to hide.
