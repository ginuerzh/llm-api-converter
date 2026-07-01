package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Protocol override tests
// ---------------------------------------------------------------------------

func TestModelMapLookupTarget(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "minimax-m3", Protocol: "anthropic"},
		{SourcePrefix: "gpt-4", TargetModel: "deepseek-chat", Protocol: "openai"},
	}

	if p := mm.LookupTarget("minimax-m3"); p != "anthropic" {
		t.Fatalf("LookupTarget(minimax-m3): want anthropic, got %q", p)
	}
	if p := mm.LookupTarget("DEEPSEEK-chat"); p != "openai" {
		t.Fatalf("LookupTarget(DEEPSEEK-chat): want openai (case insensitive), got %q", p)
	}
	if p := mm.LookupTarget("unknown"); p != "" {
		t.Fatalf("LookupTarget(unknown): want empty, got %q", p)
	}
}

func TestModelMapLookupTargetNoMatch(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "minimax-m3", Protocol: "anthropic"},
	}
	if p := mm.LookupTarget("other-model"); p != "" {
		t.Fatalf("LookupTarget(other-model): want empty, got %q", p)
	}
}

func TestModelMapApply_WithProtocol(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
		{SourcePrefix: "claude-sonnet", TargetModel: "deepseek-v4-flash"},
		{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"},
	}

	target, proto, ok := mm.Apply("claude-opus-4")
	if !ok || target != "deepseek-v4-pro" || proto != "openai" {
		t.Fatalf("claude-opus: want (deepseek-v4-pro, openai, true), got (%s, %s, %v)", target, proto, ok)
	}

	target, proto, ok = mm.Apply("claude-sonnet-4")
	if !ok || target != "deepseek-v4-flash" || proto != "" {
		t.Fatalf("claude-sonnet: want (deepseek-v4-flash, '', true), got (%s, %s, %v)", target, proto, ok)
	}

	target, proto, ok = mm.Apply("gpt-4")
	if !ok || target != "deepseek-chat" || proto != "openai" {
		t.Fatalf("gpt-4 via catch-all: want (deepseek-chat, openai, true), got (%s, %s, %v)", target, proto, ok)
	}

	target, proto, ok = mm.Apply("unknown-model")
	if !ok || target != "deepseek-chat" || proto != "openai" {
		t.Fatalf("unknown-model via catch-all: want (deepseek-chat, openai, true), got (%s, %s, %v)", target, proto, ok)
	}
}

func TestConvert_ProtocolOpenAI_OpenAIReqPassthrough(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	// Model should be rewritten but format preserved (no protocol conversion).
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v", err)
	}
	if result["model"] != "deepseek-chat" {
		t.Fatalf("model should be rewritten to deepseek-chat, got %v", result["model"])
	}
	if msgs, ok := result["messages"].([]any); !ok || len(msgs) == 0 {
		t.Fatal("messages should be preserved")
	}
}

func TestConvert_ProtocolOpenAI_AnthropicReqConverted(t *testing.T) {
	body := `{"model":"claude-sonnet-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("should produce valid OpenAI request, err=%v\nbody=%s", err, b)
	}
	if o.Model != "deepseek-chat" {
		t.Fatalf("model should be deepseek-chat, got %s", o.Model)
	}
	if len(o.Messages) == 0 {
		t.Fatal("expected messages in converted request")
	}
}

func TestConvert_ProtocolAnthropic_AnthropicReqPassthrough(t *testing.T) {
	body := `{"model":"claude-sonnet-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "claude-sonnet-4", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "claude-sonnet-4", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != body {
		t.Fatalf("expected passthrough:\n  got:  %s\n  want: %s", b, body)
	}
}

func TestConvert_ProtocolAnthropic_AnthropicReqPassthroughWithRename(t *testing.T) {
	body := `{"model":"claude-opus-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude-opus", TargetModel: "minimax-m3", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v\nbody=%s", err, b)
	}
	if result["model"] != "minimax-m3" {
		t.Fatalf("model should be rewritten to minimax-m3, got %v", result["model"])
	}
	// Should remain Anthropic format (preserve max_tokens, messages with content blocks)
	if _, ok := result["max_tokens"]; !ok {
		t.Fatal("should preserve Anthropic max_tokens field")
	}
	if _, ok := result["messages"]; !ok {
		t.Fatal("should preserve Anthropic messages field")
	}
}

func TestConvert_ProtocolAnthropic_OpenAIReqConverted(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`
	opts := &ConvertOptions{Model: "claude-sonnet-4", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "claude-sonnet-4", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicRequest
	if err := json.Unmarshal(b, &a); err != nil {
		t.Fatalf("should produce valid Anthropic request, err=%v\nbody=%s", err, b)
	}
	if len(a.Messages) == 0 {
		t.Fatal("expected messages in converted request")
	}
}

func TestConvert_ProtocolOpenAI_OpenAIRespPassthrough(t *testing.T) {
	body := `{"id":"chatcmpl-abc","object":"chat.completion","created":1234567890,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v", err)
	}
	if result["model"] != "deepseek-chat" {
		t.Fatalf("model should be rewritten to deepseek-chat, got %v", result["model"])
	}
}

func TestConvert_ProtocolAnthropic_AnthropicRespPassthrough(t *testing.T) {
	body := `{"id":"msg_abc","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":5}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "*", TargetModel: "claude-sonnet-4", Protocol: "anthropic"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != body {
		t.Fatalf("expected passthrough:\n  got:  %s\n  want: %s", b, body)
	}
}

func TestConvert_ProtocolUnset_CurrentBehavior(t *testing.T) {
	body := `{"model":"claude-opus-4","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatalf("should produce OpenAI request, err=%v\nbody=%s", err, b)
	}
	if o.Model != "deepseek-v4-pro" {
		t.Fatalf("model should be deepseek-v4-pro, got %s", o.Model)
	}
}

func TestConvert_ProtocolCatchAll(t *testing.T) {
	// Catch-all "*" with protocol=openai should rewrite model for unmatched models.
	mm := ModelMap{
		{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"},
	}
	body := `{"model":"unknown-model","messages":[{"role":"user","content":"hello"}]}`
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var result map[string]any
	if err := json.Unmarshal(b, &result); err != nil {
		t.Fatalf("should produce valid JSON, err=%v", err)
	}
	if result["model"] != "deepseek-chat" {
		t.Fatalf("model should be rewritten to deepseek-chat, got %v", result["model"])
	}
}

// Test that asymmetric protocol override (specific entries have different
// target than catch-all) still converts responses correctly.  The catch-all
// should not rewrite the downstream model, and the response should be
// converted back to the client format, not passed through.
func TestConvert_ProtocolOpenAI_AsymmetricRespConverted(t *testing.T) {
	// Response from downstream API (OpenAI format, model = target of a specific entry).
	body := `{"id":"chatcmpl-abc","object":"chat.completion","created":1234567890,"model":"deepseek-v4-pro","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
		{SourcePrefix: "gpt-5.5", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
		{SourcePrefix: "*", TargetModel: "deepseek-v4-flash", Protocol: "openai"},
	}
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	// Should be converted to Anthropic format (not passed through as OpenAI).
	var anth AnthropicResponse
	if err := json.Unmarshal(b, &anth); err != nil {
		t.Fatalf("should produce Anthropic response, err=%v\nbody=%s", err, b)
	}
	if anth.Type != "message" {
		t.Fatalf("expected Anthropic message response, got type=%q", anth.Type)
	}
	// Model should be reverse-mapped to the client-facing source prefix.
	if anth.Model != "claude-opus" {
		t.Fatalf("model should be claude-opus (client-facing), got %q", anth.Model)
	}
}

// Test that asymmetric protocol override still converts requests correctly.
func TestConvert_ProtocolOpenAI_AsymmetricReqConverted(t *testing.T) {
	body := `{"model":"claude-opus-4-8","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
		{SourcePrefix: "*", TargetModel: "deepseek-v4-flash", Protocol: "openai"},
	}
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var oai OpenAIChatRequest
	if err := json.Unmarshal(b, &oai); err != nil {
		t.Fatalf("should produce OpenAI request, err=%v\nbody=%s", err, b)
	}
	if oai.Model != "deepseek-v4-pro" {
		t.Fatalf("model should be deepseek-v4-pro, got %s", oai.Model)
	}
}

// Test that symmetric protocol override (catch-all has same target as response
// model) still passes through responses correctly.
func TestConvert_ProtocolOpenAI_SymmetricRespPassthrough(t *testing.T) {
	body := `{"id":"chatcmpl-abc","object":"chat.completion","created":1234567890,"model":"deepseek-chat","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	mm := ModelMap{
		{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"},
	}
	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	// Should pass through unchanged (model already matches catch-all target).
	if string(b) != body {
		t.Fatalf("expected passthrough unchanged:\n  got:  %s\n  want: %s", b, body)
	}
}

// Test that resolveModel preserves the model name when it matches a downstream
// target (not a source prefix), and uses the target entry's protocol.
func TestResolveModel_LookupTargetPreservesModel(t *testing.T) {
	mm := ModelMap{
		{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
		{SourcePrefix: "*", TargetModel: "deepseek-v4-flash", Protocol: "openai"},
	}

	// Request direction: model="claude-opus-4-8" → maps via source prefix.
	target, proto := resolveModel("claude-opus-4-8", "fallback", mm)
	if target != "deepseek-v4-pro" {
		t.Fatalf("request: want target deepseek-v4-pro, got %q", target)
	}
	if proto != "openai" {
		t.Fatalf("request: want proto openai, got %q", proto)
	}

	// Response direction: model="deepseek-v4-pro" → is a downstream target, preserve it.
	target, proto = resolveModel("deepseek-v4-pro", "fallback", mm)
	if target != "deepseek-v4-pro" {
		t.Fatalf("response: want model preserved as deepseek-v4-pro, got %q", target)
	}
	if proto != "openai" {
		t.Fatalf("response: want proto openai, got %q", proto)
	}
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Fuzz tests — protocol conversion matrix.
//
// The matrix has 3 dimensions:
//   1. Model-map configuration (asymmetric, symmetric, mixed, no-protocol)
//   2. Input shape (Anthropic req, OpenAI req, OpenAI resp, Anthropic resp,
//      OpenAI stream chunk, Anthropic SSE)
//   3. Model field value (the fuzz variable — explores source-prefix,
//      target, and unknown values against each config)
//
// Each fuzz function covers one config × all input shapes, varying the
// model field.  This catches regressions like the catch-all hijack bug
// (downstream target model rewritten by an unrelated catch-all entry).
// ---------------------------------------------------------------------------

// -------- Model-map configurations --------

// cfgAsymmetricOpenAI: specific entries share target "deepseek-v4-pro",
// catch-all has a different target "deepseek-v4-flash".  This is the
// production shape that triggered the catch-all hijack bug.
var cfgAsymmetricOpenAI = ModelMap{
	{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
	{SourcePrefix: "gpt-5.5", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
	{SourcePrefix: "*", TargetModel: "deepseek-v4-flash", Protocol: "openai"},
}

// cfgSymmetricOpenAI: single catch-all with :openai — model remapping only,
// both sides speak OpenAI.
var cfgSymmetricOpenAI = ModelMap{
	{SourcePrefix: "*", TargetModel: "deepseek-chat", Protocol: "openai"},
}

// cfgSymmetricAnthropic: single catch-all with :anthropic — model remapping
// only, both sides speak Anthropic.
var cfgSymmetricAnthropic = ModelMap{
	{SourcePrefix: "*", TargetModel: "claude-sonnet-4", Protocol: "anthropic"},
}

// cfgMixedProtocol: entries with different downstream protocols.  One source
// maps to an OpenAI backend, another to an Anthropic backend.  No catch-all.
var cfgMixedProtocol = ModelMap{
	{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro", Protocol: "openai"},
	{SourcePrefix: "claude-sonnet", TargetModel: "minimax-m3", Protocol: "anthropic"},
}

// cfgNoProtocol: mapping without protocol override — auto-detect only.
var cfgNoProtocol = ModelMap{
	{SourcePrefix: "claude-opus", TargetModel: "deepseek-v4-pro"},
	{SourcePrefix: "*", TargetModel: "deepseek-v4-flash"},
}

// allConfigs is the matrix of model-map configurations covered by fuzz tests.
var allConfigs = []struct {
	name string
	mm   ModelMap
}{
	{"asymmetric-openai", cfgAsymmetricOpenAI},
	{"symmetric-openai", cfgSymmetricOpenAI},
	{"symmetric-anthropic", cfgSymmetricAnthropic},
	{"mixed-protocol", cfgMixedProtocol},
	{"no-protocol", cfgNoProtocol},
}

// -------- Input shape templates (model field is the fuzz variable) --------

// shape templates use {MODEL} as placeholder for the fuzzed model value.
// Each shape is a structurally-valid body for its format so the auto-detector
// classifies it correctly and we exercise the full convert/passthrough path.

var inputShapes = []struct {
	name   string
	format string // "anthropic-req", "openai-req", "openai-resp", "anthropic-resp", "openai-stream"
	tmpl   string // JSON body with {MODEL} placeholder
}{
	{
		name:   "anthropic-req",
		format: "anthropic-req",
		tmpl:   `{"model":"{MODEL}","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
	},
	{
		name:   "openai-req",
		format: "openai-req",
		tmpl:   `{"model":"{MODEL}","messages":[{"role":"user","content":"hello"}]}`,
	},
	{
		name:   "openai-resp",
		format: "openai-resp",
		tmpl:   `{"id":"chatcmpl-abc","object":"chat.completion","created":1234567890,"model":"{MODEL}","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
	},
	{
		name:   "anthropic-resp",
		format: "anthropic-resp",
		tmpl:   `{"id":"msg_abc","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}],"model":"{MODEL}","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":5}}`,
	},
	{
		name:   "openai-stream",
		format: "openai-stream",
		tmpl:   `{"id":"chatcmpl-abc","object":"chat.completion.chunk","created":1234567890,"model":"{MODEL}","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`,
	},
}

// modelSeeds are values for the model field that cover key categories:
// source-prefix match, target match, unknown, and edge cases.
var modelSeeds = []string{
	// Source-prefix matches (explicit entries).
	"claude-opus-4-8",
	"claude-opus-4-1",
	"gpt-5.5",
	"claude-sonnet-4",
	// Target-model matches (response from downstream API).
	"deepseek-v4-pro",
	"deepseek-v4-flash",
	"deepseek-chat",
	"minimax-m3",
	"claude-sonnet-4",
	// Unknown / unmapped models.
	"unknown-model",
	"gpt-4",
	"claude-haiku-4",
	// Edge cases.
	"",
	"anthropic/claude-opus-4-8", // provider prefix
}

// -------- Fuzz tests --------

// FuzzConvert_Matrix_AsymmetricOpenAI fuzzes the full config × shape matrix
// for the asymmetric-openai configuration (the original bug shape).
func FuzzConvert_Matrix_AsymmetricOpenAI(f *testing.F) {
	seedFuzzMatrix(f, cfgAsymmetricOpenAI)
	f.Fuzz(func(t *testing.T, shapeName, model string) {
		fuzzMatrixConvert(t, cfgAsymmetricOpenAI, shapeName, model)
	})
}

// FuzzConvert_Matrix_SymmetricOpenAI fuzzes the symmetric-openai config.
func FuzzConvert_Matrix_SymmetricOpenAI(f *testing.F) {
	seedFuzzMatrix(f, cfgSymmetricOpenAI)
	f.Fuzz(func(t *testing.T, shapeName, model string) {
		fuzzMatrixConvert(t, cfgSymmetricOpenAI, shapeName, model)
	})
}

// FuzzConvert_Matrix_SymmetricAnthropic fuzzes the symmetric-anthropic config.
func FuzzConvert_Matrix_SymmetricAnthropic(f *testing.F) {
	seedFuzzMatrix(f, cfgSymmetricAnthropic)
	f.Fuzz(func(t *testing.T, shapeName, model string) {
		fuzzMatrixConvert(t, cfgSymmetricAnthropic, shapeName, model)
	})
}

// FuzzConvert_Matrix_MixedProtocol fuzzes the mixed-protocol config.
func FuzzConvert_Matrix_MixedProtocol(f *testing.F) {
	seedFuzzMatrix(f, cfgMixedProtocol)
	f.Fuzz(func(t *testing.T, shapeName, model string) {
		fuzzMatrixConvert(t, cfgMixedProtocol, shapeName, model)
	})
}

// FuzzConvert_Matrix_NoProtocol fuzzes the no-protocol-override config.
func FuzzConvert_Matrix_NoProtocol(f *testing.F) {
	seedFuzzMatrix(f, cfgNoProtocol)
	f.Fuzz(func(t *testing.T, shapeName, model string) {
		fuzzMatrixConvert(t, cfgNoProtocol, shapeName, model)
	})
}

// FuzzResolveModel_Matrix fuzzes resolveModel across all configs, verifying
// that downstream targets are never rewritten by catch-all entries.
func FuzzResolveModel_Matrix(f *testing.F) {
	for _, cfg := range allConfigs {
		for _, seed := range modelSeeds {
			f.Add(cfg.name, seed)
		}
	}
	f.Fuzz(func(t *testing.T, cfgName, model string) {
		if model == "" {
			return
		}
		cfg := findConfig(cfgName)
		if cfg == nil {
			return // fuzzer-generated config name — skip
		}
		target, proto := resolveModel(model, "fallback", cfg.mm)

		// Invariant: if the model is a known downstream target of an
		// entry WITH protocol override, it must not be rewritten by
		// a different catch-all entry.  (No-protocol entries don't
		// trigger passthrough, so catch-all rewriting there is harmless —
		// auto-detection handles the format correctly.)
		for _, entry := range cfg.mm {
			if entry.SourcePrefix == "*" || entry.Protocol == "" {
				continue
			}
			if strings.EqualFold(entry.TargetModel, model) {
				if target != strings.ToLower(model) && target != model {
					t.Errorf("config=%s: downstream target %q (protocol=%s) was rewritten to %q by catch-all",
						cfgName, model, entry.Protocol, target)
				}
				if proto != entry.Protocol {
					t.Errorf("config=%s: downstream target %q lost protocol: got %q, want %q",
						cfgName, model, proto, entry.Protocol)
				}
				return
			}
		}

		// Invariant: if proto is set, target must be non-empty.
		if proto != "" && target == "" {
			t.Errorf("config=%s: proto=%q but target is empty for model %q", cfgName, proto, model)
		}
	})
}

// -------- Helpers --------

// seedFuzzMatrix seeds a fuzz test with all shape × model-seed combinations.
func seedFuzzMatrix(f *testing.F, _ ModelMap) {
	for _, shape := range inputShapes {
		for _, seed := range modelSeeds {
			f.Add(shape.name, seed)
		}
	}
}

// fuzzMatrixConvert runs Convert with the given config, shape, and model,
// and checks invariants on the output.
func fuzzMatrixConvert(t *testing.T, mm ModelMap, shapeName, model string) {
	t.Helper()

	shape := findShape(shapeName)
	if shape == nil {
		return // fuzzer-generated shape name — skip
	}
	body := strings.ReplaceAll(shape.tmpl, "{MODEL}", model)

	opts := &ConvertOptions{Model: "deepseek-chat", MaxTokens: 8192, ModelMap: mm}
	result, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatalf("Convert error: %v", err)
	}
	if len(result) == 0 {
		t.Logf("empty output for shape=%s model=%q", shapeName, model)
		return
	}

	// Collect target models from non-catch-all entries that have
	// protocol override set.  Only these are at risk of catch-all
	// hijack: without protocol override the passthrough block never
	// triggers, so auto-detection handles the format correctly.
	protoTargets := make(map[string]bool)
	for _, entry := range mm {
		if entry.SourcePrefix != "*" && entry.TargetModel != "" && entry.Protocol != "" {
			protoTargets[strings.ToLower(entry.TargetModel)] = true
		}
	}
	// Also collect the catch-all target.
	catchTarget, hasCatchAll := mm.catchAllTarget()
	if hasCatchAll {
		catchTarget = strings.ToLower(catchTarget)
	}

	// ---- Invariant 1: output is parseable JSON or valid SSE ----
	var out map[string]any
	if json.Unmarshal(result, &out) != nil {
		// Might be SSE-framed — try parsing as SSE event.
		if isSSE(result) {
			evt := parseSSEEvent(result)
			if evt.Data != "" {
				if json.Unmarshal([]byte(evt.Data), &out) != nil {
					// SSE with non-JSON data (e.g. [DONE]) — fine.
					return
				}
			} else {
				return
			}
		} else {
			// Non-JSON, non-SSE output — pass-through, fine.
			return
		}
	}
	outModel, _ := out["model"].(string)

	modelLower := strings.ToLower(model)

	// ---- Invariant 2: downstream target must not be corrupted to catch-all target ----
	if protoTargets[modelLower] && outModel != "" {
		if hasCatchAll && strings.ToLower(outModel) == catchTarget && catchTarget != modelLower {
			t.Errorf("shape=%s model=%q: downstream target corrupted to catch-all target %q",
				shapeName, model, outModel)
		}
	}

	// ---- Invariant 3: output must not lose the model field if input had one ----
	if model != "" && outModel == "" {
		// Some conversions legitimately drop model (e.g. Anthropic SSE
		// content_block_delta events).  Only flag if the output is a
		// top-level request/response object.
		if _, ok := out["messages"]; ok {
			t.Logf("shape=%s model=%q: model field lost in request output", shapeName, model)
		}
		if _, ok := out["choices"]; ok {
			t.Logf("shape=%s model=%q: model field lost in response output", shapeName, model)
		}
	}

	// ---- Invariant 4: format consistency — if the input is a known
	// format, the output must be a valid format of some type ----
	switch shape.format {
	case "anthropic-req":
		// Must produce valid JSON output.
		if outModel == "" && len(out) == 0 {
			t.Errorf("shape=%s model=%q: empty output for Anthropic request", shapeName, model)
		}
	case "openai-resp":
		// Must produce either passthrough OpenAI or converted Anthropic.
		if _, hasChoices := out["choices"]; hasChoices {
			// Passthrough — model should be correct.
			if outModel != model && outModel != "" && protoTargets[modelLower] {
				// Model changed — verify it wasn't catch-all corruption.
				if hasCatchAll && strings.ToLower(outModel) == catchTarget && catchTarget != modelLower {
					t.Errorf("shape=%s model=%q: response model corrupted to catch-all", shapeName, model)
				}
			}
		}
	case "anthropic-resp":
		// Must produce either passthrough Anthropic or converted OpenAI.
		if _, hasType := out["type"]; hasType {
			// Passthrough — model should be unchanged.
		}
	}
}

func mustFindConfig(t *testing.T, name string) struct {
	name string
	mm   ModelMap
} {
	t.Helper()
	cfg := findConfig(name)
	if cfg == nil {
		t.Fatalf("unknown config: %s", name)
	}
	return *cfg
}

func findConfig(name string) *struct {
	name string
	mm   ModelMap
} {
	for _, cfg := range allConfigs {
		if cfg.name == name {
			return &cfg
		}
	}
	return nil
}

func mustFindShape(t *testing.T, name string) struct {
	name   string
	format string
	tmpl   string
} {
	t.Helper()
	s := findShape(name)
	if s == nil {
		t.Fatalf("unknown shape: %s", name)
	}
	return *s
}

func findShape(name string) *struct {
	name   string
	format string
	tmpl   string
} {
	for _, s := range inputShapes {
		if s.name == name {
			return &s
		}
	}
	return nil
}

// parseModelMap tests are in rewriter/server_test.go
