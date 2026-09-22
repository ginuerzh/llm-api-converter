package convert

import (
	"encoding/json"
	"strings"
	"testing"
)

// Claude Code sends thinking:{"type":"adaptive"} (adaptive reasoning mode).
// There is no OpenAI-format equivalent, and reasoning-only models like
// ox-alpha-free reject any non-{enabled,disabled} thinking directive. The
// thinking field must not leak verbatim into the OpenAI request.
func TestConvert_DeepSeekAdaptiveThinkingDropped(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Thinking != nil {
		t.Fatalf("thinking should be dropped for non-OpenAI thinking.type, got %v", o.Thinking)
	}
	if o.ReasoningEffort == nil {
		t.Fatal("reasoning_effort should still be set from output_config.effort")
	}
	if e, _ := o.ReasoningEffort.(string); e != "high" {
		t.Fatalf("reasoning_effort: want 'high', got %v", o.ReasoningEffort)
	}
}

// enabled/disabled remain valid OpenAI-format values and must still pass through.
func TestConvert_DeepSeekThinkingEnabledDisabledPassthrough(t *testing.T) {
	for _, tt := range []struct {
		thinking string
		wantType string
	}{
		{`{"type":"enabled","budget_tokens":4096}`, "enabled"},
		{`{"type":"disabled"}`, "disabled"},
	} {
		body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":` + tt.thinking + `}`
		opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
		b, err := Convert([]byte(body), opts)
		if err != nil {
			t.Fatal(err)
		}
		var o OpenAIChatRequest
		if err := json.Unmarshal(b, &o); err != nil {
			t.Fatal(err)
		}
		if o.Thinking == nil {
			t.Fatalf("thinking should be forwarded for type %s", tt.wantType)
		}
		if m, ok := o.Thinking.(map[string]any); !ok || m["type"] != tt.wantType {
			t.Fatalf("thinking: want type %q, got %v", tt.wantType, o.Thinking)
		}
	}
}

// reasoning_effort values are preserved verbatim (OpenAI-compatible models such
// as DeepSeek/GLM/ox-alpha accept "max"); only xhigh is capped to "max", and
// low/medium must not be collapsed to high.
func TestReasoningEffortToOpenAi(t *testing.T) {
	for _, tt := range []struct {
		effort string
		want   any
	}{
		{"low", "low"},
		{"medium", "medium"},
		{"high", "high"},
		{"max", "max"},
		{"xhigh", "max"},
		{"", nil},
	} {
		got := reasoningEffortToOpenAi(&AnthropicOutputConfig{Effort: tt.effort})
		if got != tt.want {
			t.Fatalf("effort %q: want %v, got %v", tt.effort, tt.want, got)
		}
	}
	if reasoningEffortToOpenAi(nil) != nil {
		t.Fatal("nil config should return nil")
	}
}

// GLM path has the same problem: adaptive must not leak.
func TestConvert_GLMAdaptiveThinkingDropped(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}],"thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-20250514", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "glm-4.6", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Thinking != nil {
		t.Fatalf("GLM thinking should be dropped for non-OpenAI thinking.type, got %v", o.Thinking)
	}
}

// Anthropic metadata (user_id etc.) must not leak into the OpenAI request —
// strict endpoints reject unknown params ("Invalid API parameter").
func TestConvert_AnthropicMetadataNotForwarded(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"metadata":{"user_id":"u_abc123"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"output_config":{"effort":"high"}}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-5", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatRequest
	if err := json.Unmarshal(b, &o); err != nil {
		t.Fatal(err)
	}
	if o.Metadata != nil {
		t.Fatalf("Anthropic metadata should not be forwarded, got %v", o.Metadata)
	}
}

// A single stop_sequences entry must serialize as an array, not a bare string.
// Jackson-backed endpoints type stop as a list and reject a string value
// (the reported 'Cannot construct instance of java.util.ArrayList' error).
func TestConvert_StopSequencesAlwaysArray(t *testing.T) {
	for _, seqs := range [][]string{
		{`</block>`},
		{`</block>`, `</tool>`},
	} {
		raw, _ := json.Marshal(seqs)
		body := `{"model":"claude","max_tokens":8192,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stop_sequences":` + string(raw) + `}`
		opts := &ConvertOptions{Model: "claude-sonnet-4-5", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
		b, err := Convert([]byte(body), opts)
		if err != nil {
			t.Fatal(err)
		}
		var o OpenAIChatRequest
		if err := json.Unmarshal(b, &o); err != nil {
			t.Fatal(err)
		}
		arr, ok := o.Stop.([]any)
		if !ok {
			t.Fatalf("stop should be an array, got %T (%v)", o.Stop, o.Stop)
		}
		if len(arr) != len(seqs) {
			t.Fatalf("stop length: want %d, got %d", len(seqs), len(arr))
		}
		for i, want := range seqs {
			if s, _ := arr[i].(string); s != want {
				t.Fatalf("stop[%d]: want %q, got %q", i, want, s)
			}
		}
	}
}
// Claude Code sends tool_result content as a block array when the tool output
// contains an image (Read on a PNG, screenshots). The OpenAI tool role requires
// string content; strict upstreams reject arrays with
// "messages.N.tool.content: Input should be a valid string". Images move to a
// trailing user message as image_url parts.
func TestConvert_ToolResultContentAlwaysString(t *testing.T) {
	const img = `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}`
	const imgURL = "data:image/png;base64,iVBORw0KGgo="
	for _, tt := range []struct {
		name     string
		content  string
		want     string
		wantImgs int
	}{
		{"string content", `"plain output"`, "plain output", 0},
		{"text and image", `[{"type":"text","text":"screenshot below"},` + img + `]`, "screenshot below", 1},
		{"image only", `[` + img + `]`, "", 1},
		{"empty array", `[]`, "", 0},
		{"non-base64 image", `[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]`, "[image omitted]", 0},
		{"document block", `[{"type":"text","text":"pdf attached"},{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]`, "pdf attached\n[document omitted]", 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"model":"claude","max_tokens":8192,"system":"sys","messages":[` +
				`{"role":"user","content":[{"type":"text","text":"read it"}]},` +
				`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{}}]},` +
				`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":` + tt.content + `}]}]}`
			opts := &ConvertOptions{Model: "claude-sonnet-4-5", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
			b, err := Convert([]byte(body), opts)
			if err != nil {
				t.Fatal(err)
			}
			var o struct {
				Messages []struct {
					Role    string `json:"role"`
					Content any    `json:"content"`
				} `json:"messages"`
			}
			if err := json.Unmarshal(b, &o); err != nil {
				t.Fatal(err)
			}
			toolIdx := -1
			for i, m := range o.Messages {
				if m.Role != "tool" {
					continue
				}
				s, ok := m.Content.(string)
				if !ok {
					t.Fatalf("tool content is %T, want string: %s", m.Content, b)
				}
				if s != tt.want {
					t.Fatalf("tool content: want %q, got %q", tt.want, s)
				}
				toolIdx = i
				break
			}
			if toolIdx < 0 {
				t.Fatalf("no tool message in output: %s", b)
			}
			var gotImgs int
			for _, m := range o.Messages[toolIdx+1:] {
				parts, ok := m.Content.([]any)
				if !ok {
					continue
				}
				for _, p := range parts {
					pm, _ := p.(map[string]any)
					if pm["type"] != "image_url" {
						continue
					}
					gotImgs++
					iu, _ := pm["image_url"].(map[string]any)
					if u, _ := iu["url"].(string); u != imgURL {
						t.Fatalf("image url: want %q, got %q", imgURL, u)
					}
				}
			}
			if gotImgs != tt.wantImgs {
				t.Fatalf("forwarded images: want %d, got %d: %s", tt.wantImgs, gotImgs, b)
			}
		})
	}
}

// Claude Code sends mid-conversation role:"system" messages (system-reminder
// blocks, mid-conversation-system beta). Their content must convert to text,
// not a Go struct dump of []AnthropicContent ("[{text ...} <nil> ... false}]").
func TestConvert_MidConversationSystemMessageContent(t *testing.T) {
	body := `{"model":"claude","max_tokens":8192,"system":"sys","messages":[` +
		`{"role":"user","content":[{"type":"text","text":"hi"}]},` +
		`{"role":"system","content":[{"type":"text","text":"<system-reminder>be concise</system-reminder>"}]},` +
		`{"role":"user","content":[{"type":"text","text":"go"}]}]}`
	opts := &ConvertOptions{Model: "claude-sonnet-4-5", MaxTokens: 8192, ModelMap: ModelMap{{SourcePrefix: "claude", TargetModel: "deepseek-chat", Protocol: "openai"}}}
	b, err := Convert([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if strings.Contains(s, "<nil>") || strings.Contains(s, "[{text") {
		t.Fatalf("system message content leaked Go struct formatting: %s", s)
	}
	if !strings.Contains(s, "be concise") {
		t.Fatalf("system message text lost: %s", s)
	}
}
