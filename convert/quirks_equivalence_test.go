package convert

import "testing"

// TestModelProfileMatchesExpectedMatrix pins the model -> profile mapping to
// hand-derived expected values. The table is the oracle for the profile
// refactor: if a row fails, the codebase (not the row) is telling us the
// profile was mis-assembled.
//
//	StripTopK         = isOpenAIReasoningModel(stripAllProviderPrefixes(model))
//	ReasoningMaxTokens = max_completion_tokens iff isOpenAIReasoningModel(model)
func TestModelProfileMatchesExpectedMatrix(t *testing.T) {
	type expected struct {
		stripTopK          bool
		reasoningMaxTokens string
		thinkingStyle      string
		forceToolChoice    string
		jsonSchemaMode     string
		contentFlatten     bool
		reasoningReplay    string
	}
	cases := []struct {
		model string
		want  expected
	}{
		{"deepseek-v4-pro", expected{false, "none", "deepseek", "strip", "json_object", false, "deepseek"}},
		{"deepseek-chat", expected{false, "none", "deepseek", "strip", "json_object", false, "deepseek"}},
		{"ollama/deepseek-v4-flash", expected{false, "none", "none", "allow", "pass", false, "none"}},
		{"zhipu/deepseek-v3", expected{false, "none", "none", "allow", "pass", false, "none"}},
		{"glm-4.7", expected{false, "none", "glm", "allow", "pass", true, "glm"}},
		{"zhipu/glm-51", expected{false, "none", "none", "allow", "pass", false, "none"}},
		{"foo/bar/glm-4.7", expected{false, "none", "none", "allow", "pass", false, "none"}},
		{"o1-mini", expected{true, "max_completion_tokens", "none", "allow", "pass", false, "none"}},
		{"o3-mini", expected{true, "max_completion_tokens", "none", "allow", "pass", false, "none"}},
		{"o4-mini", expected{true, "max_completion_tokens", "none", "allow", "pass", false, "none"}},
		{"gpt-5.5", expected{true, "max_completion_tokens", "none", "allow", "pass", false, "none"}},
		{"models/o3-mini", expected{true, "max_completion_tokens", "none", "allow", "pass", false, "none"}},
		{"foo/bar/o3-mini", expected{true, "none", "none", "allow", "pass", false, "none"}},
		{"gemini-4-hydrogen", expected{false, "none", "none", "allow", "pass", false, "none"}},
		{"models/gemini-2.5-pro", expected{false, "none", "none", "allow", "pass", false, "none"}},
		{"kimi-k3", expected{false, "none", "none", "allow", "pass", false, "none"}},
		{"gpt-4o", expected{false, "none", "none", "allow", "pass", false, "none"}},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			got := classifyModel(tc.model)
			if got.stripTopK != tc.want.stripTopK {
				t.Errorf("stripTopK = %v, want %v", got.stripTopK, tc.want.stripTopK)
			}
			if got.reasoningMaxTokens != tc.want.reasoningMaxTokens {
				t.Errorf("reasoningMaxTokens = %q, want %q", got.reasoningMaxTokens, tc.want.reasoningMaxTokens)
			}
			if got.thinkingStyle != tc.want.thinkingStyle {
				t.Errorf("thinkingStyle = %q, want %q", got.thinkingStyle, tc.want.thinkingStyle)
			}
			if got.forceToolChoice != tc.want.forceToolChoice {
				t.Errorf("forceToolChoice = %q, want %q", got.forceToolChoice, tc.want.forceToolChoice)
			}
			if got.jsonSchemaMode != tc.want.jsonSchemaMode {
				t.Errorf("jsonSchemaMode = %q, want %q", got.jsonSchemaMode, tc.want.jsonSchemaMode)
			}
			if got.contentFlatten != tc.want.contentFlatten {
				t.Errorf("contentFlatten = %v, want %v", got.contentFlatten, tc.want.contentFlatten)
			}
			if got.reasoningReplay != tc.want.reasoningReplay {
				t.Errorf("reasoningReplay = %q, want %q", got.reasoningReplay, tc.want.reasoningReplay)
			}
		})
	}
}
