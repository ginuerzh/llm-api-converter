package convert

import (
	"encoding/json"
	"regexp"
	"strings"
	"testing"
)

// TestGeminiSignature_KeptPerCallID_Lifecycle covers the path the rewriter
// actually uses for a Gemini SSE response: HandleSSEEvent -> stateful
// converter. The signature must survive into the next request.
func TestGeminiSignature_KeptPerCallID_Lifecycle(t *testing.T) {
	store := NewSessionStore()
	store.Set("sig-sid", &Session{ID: "sig-sid", From: ProtocolAnthropic})
	opts := &ConvertOptions{
		MaxTokens:    100,
		SessionStore: store,
		SID:          "sig-sid",
		Direction:    "response",
		ModelMap:     ModelMap{{SourcePrefix: "*", TargetModel: "gemini-pro", Protocol: "gemini"}},
	}

	chunk := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"p":"a"}},"thoughtSignature":"sigA"}],"role":"model"},"finishReason":"STOP"}]}`
	out, err := HandleSSEEvent("sig-sid", string(StreamPhaseStart), 0, []byte(chunk), opts)
	if err != nil {
		t.Fatalf("handle start: %v", err)
	}
	id := regexp.MustCompile(`toolu_[0-9a-f]+`).FindString(string(out))
	if id == "" {
		t.Fatalf("no tool_use id in: %s", out)
	}
	if got := thoughtSigForCall(t, store, id, "read"); got != "sigA" {
		t.Fatalf("signature: want sigA, got %q", got)
	}
}

func TestGeminiThought_NonStreamOpenAI(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolOpenAIChat})
	body := `{"candidates":[{"content":{"parts":[{"text":"pondering","thought":true},{"text":"answer"}],"role":"model"},"finishReason":"STOP"}]}`
	out, err := Convert([]byte(body), &ConvertOptions{
		MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response",
	})
	if err != nil {
		t.Fatal(err)
	}
	var o OpenAIChatResponse
	if err := json.Unmarshal(out, &o); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if len(o.Choices) != 1 {
		t.Fatalf("choices: %s", out)
	}
	if o.Choices[0].Message.ReasoningContent != "pondering" {
		t.Fatalf("reasoning_content: want pondering, got %q", o.Choices[0].Message.ReasoningContent)
	}
	if o.Choices[0].Message.Content != "answer" {
		t.Fatalf("content: want answer, got %v", o.Choices[0].Message.Content)
	}
}

func TestGeminiThought_NonStreamAnthropic(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolAnthropic})
	body := `{"candidates":[{"content":{"parts":[{"text":"pondering","thought":true},{"text":"answer"}],"role":"model"},"finishReason":"STOP"}]}`
	out, err := Convert([]byte(body), &ConvertOptions{
		MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response",
	})
	if err != nil {
		t.Fatal(err)
	}
	var a AnthropicResponse
	if err := json.Unmarshal(out, &a); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	var thinking, text string
	for _, c := range a.Content {
		switch c.Type {
		case "thinking":
			thinking = c.Thinking
		case "text":
			text = c.Text
		}
	}
	if thinking != "pondering" || text != "answer" {
		t.Fatalf("want thinking=pondering text=answer, got %q / %q\n%s", thinking, text, out)
	}
}

func TestGeminiThought_StreamAnthropic(t *testing.T) {
	c := NewGeminiToAnthropicStreamConverter("claude-sonnet-4-20250514")
	c.HandleStreamStart()
	out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"text":"pondering","thought":true}],"role":"model"}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	if !strings.Contains(s, `"type":"thinking"`) || !strings.Contains(s, `"thinking_delta"`) {
		t.Fatalf("thought not mapped to a thinking block:\n%s", s)
	}
}

func TestGeminiRequest_AnthropicThinkingConfig(t *testing.T) {
	body := `{"model":"claude-sonnet-4-20250514","max_tokens":100,
		"thinking":{"type":"enabled","budget_tokens":4096},
		"messages":[{"role":"user","content":"hi"}]}`
	out, err := convertAnthropicRequestToGemini([]byte(body), &ConvertOptions{MaxTokens: 100})
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(out, &g); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if g.GenerationConfig == nil || g.GenerationConfig.ThinkingConfig == nil {
		t.Fatalf("thinkingConfig missing: %s", out)
	}
	tc := g.GenerationConfig.ThinkingConfig
	if tc.ThinkingBudget == nil || *tc.ThinkingBudget != 4096 {
		t.Fatalf("thinkingBudget: want 4096, got %v", tc.ThinkingBudget)
	}
	if tc.IncludeThoughts == nil || !*tc.IncludeThoughts {
		t.Fatalf("includeThoughts: want true, got %v", tc.IncludeThoughts)
	}
	if g.GenerationConfig.MaxOutputTokens == nil || *g.GenerationConfig.MaxOutputTokens != 100 {
		t.Fatalf("maxOutputTokens must stay the request's: %v", g.GenerationConfig.MaxOutputTokens)
	}
}

func TestGeminiRequest_OpenAIReasoningEffort(t *testing.T) {
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`
	out, err := convertOpenAIRequestToGemini([]byte(body), &ConvertOptions{MaxTokens: 100})
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatRequest
	if err := json.Unmarshal(out, &g); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if g.GenerationConfig == nil || g.GenerationConfig.ThinkingConfig == nil ||
		g.GenerationConfig.ThinkingConfig.ThinkingBudget == nil {
		t.Fatalf("thinkingConfig missing: %s", out)
	}
	if got := *g.GenerationConfig.ThinkingConfig.ThinkingBudget; got != 24000 {
		t.Fatalf("thinkingBudget: want 24000 for high, got %d", got)
	}
}

func TestSessionStore_ThoughtSignaturesBounded(t *testing.T) {
	store := NewSessionStore()
	store.maxSize = 2
	store.SetThoughtSigFor("a", "id-a", "read", "sig-a")
	store.SetThoughtSigFor("b", "id-b", "read", "sig-b")
	store.SetThoughtSigFor("c", "id-c", "read", "sig-c")

	if got := store.GetThoughtSig("a", "id-a"); got != "" {
		t.Fatalf("oldest session's signature kept: %q", got)
	}
	if got := store.GetThoughtSigFor("c", "id-c", "read"); got != "sig-c" {
		t.Fatalf("newest session's signature lost: %q", got)
	}
}

func TestConvertSSE_PassthroughTidiesOpenAIChunk(t *testing.T) {
	opts := &ConvertOptions{
		Model:     "gpt-4",
		MaxTokens: 100,
		ModelMap:  ModelMap{{SourcePrefix: "*", TargetModel: "gpt-4", Protocol: "openai"}},
	}
	body := `data: {"model":"gpt-4","choices":[{"index":0,"delta":{"content":"hi","reasoning_content":""},"finish_reason":null}]}`
	out, err := ConvertSSE([]byte(body), opts)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "reasoning_content") {
		t.Fatalf("empty reasoning_content not dropped: %s", out)
	}
}

func TestDetectSource_AnthropicStreamEvent(t *testing.T) {
	for _, typ := range []string{"message_start", "content_block_start", "content_block_delta", "message_delta", "message_stop"} {
		if got := detectSource(map[string]any{"type": typ}); got != ProtocolAnthropic {
			t.Fatalf("%s: want Anthropic, got %v", typ, got)
		}
	}
}

func TestGeminiSignature_PrefersCallID(t *testing.T) {
	store := NewSessionStore()
	store.SetThoughtSig("sid", "call_1", "sig-id")
	store.SetThoughtSig("sid", "read", "sig-name")
	body := `{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,
		"message":{"role":"assistant","content":"","tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}
		]},"finish_reason":"tool_calls"}]}`
	out, err := convertOpenAIResponseToGemini([]byte(body), &ConvertOptions{SessionStore: store, SID: "sid"})
	if err != nil {
		t.Fatal(err)
	}
	var g GeminiChatResponse
	if err := json.Unmarshal(out, &g); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if len(g.Candidates) == 0 || len(g.Candidates[0].Content.Parts) == 0 {
		t.Fatalf("no parts: %s", out)
	}
	part := g.Candidates[0].Content.Parts[0]
	if part.FunctionCall == nil {
		t.Fatalf("no functionCall: %s", out)
	}
	if got := part.ThoughtSignature; got != "sig-id" {
		t.Fatalf("signature: want sig-id, got %q", got)
	}
}

func TestAnthropicSSEToOpenAIStream_NoMessageStart(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolOpenAIChat})
	opts := &ConvertOptions{MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response", URI: "/v1/messages"}

	out, err := HandleSSEEvent("sid", string(StreamPhaseStart), 0, []byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`), opts)
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	if !strings.Contains(s, `"role":"assistant"`) {
		t.Fatalf("role chunk must precede content:\n%s", s)
	}
	if !strings.Contains(s, `"content":"Hi"`) {
		t.Fatalf("content missing:\n%s", s)
	}
}

func TestGeminiStream_MergesSplitFunctionCall_OpenAI(t *testing.T) {
	c := NewGeminiToOpenAIStreamConverter("gpt-4")
	c.HandleStreamStart()
	if out, _ := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"path":"a"}}}],"role":"model"}}]}`)); len(out) != 0 {
		t.Fatalf("first fragment must wait for the rest: %s", out)
	}
	out, _ := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"line":1}}}],"role":"model"},"finishReason":"STOP"}]}`))
	s := string(out) + string(c.HandleStreamEnd())
	if got := strings.Count(s, `"tool_calls"`); got != 1 {
		t.Fatalf("want one tool call, got %d:\n%s", got, s)
	}
	if !strings.Contains(s, "path") || !strings.Contains(s, "line") {
		t.Fatalf("arguments not merged:\n%s", s)
	}
}

func TestGeminiStream_MergesSplitFunctionCall_Anthropic(t *testing.T) {
	c := NewGeminiToAnthropicStreamConverter("claude-sonnet-4-20250514")
	c.HandleStreamStart()
	if out, _ := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"path":"a"}}}],"role":"model"}}]}`)); strings.Contains(string(out), "tool_use") {
		t.Fatalf("first fragment emitted a tool block: %s", out)
	}
	out, _ := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"line":1}}}],"role":"model"},"finishReason":"STOP"}]}`))
	s := string(out) + string(c.HandleStreamEnd())
	if got := strings.Count(s, `"type":"tool_use"`); got != 1 {
		t.Fatalf("want one tool_use block, got %d:\n%s", got, s)
	}
	if !strings.Contains(s, "path") || !strings.Contains(s, "line") {
		t.Fatalf("arguments not merged:\n%s", s)
	}
}

type geminiTestChunk struct {
	cand  GeminiCandidate
	usage *GeminiUsage
}

func geminiChunksFromSSE(t *testing.T, s string) []geminiTestChunk {
	t.Helper()
	var out []geminiTestChunk
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" || data == "" {
			continue
		}
		var g GeminiChatResponse
		if err := json.Unmarshal([]byte(data), &g); err != nil {
			t.Fatalf("unmarshal Gemini chunk: %v\n%s", err, data)
		}
		chunk := geminiTestChunk{usage: g.UsageMetadata}
		if len(g.Candidates) > 0 {
			chunk.cand = g.Candidates[0]
		}
		out = append(out, chunk)
	}
	return out
}

func TestAnthropicSSEToGeminiStream(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolGemini})
	opts := &ConvertOptions{MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response", URI: "/v1beta/models/gemini-pro:streamGenerateContent"}

	var out []byte
	add := func(phase StreamPhase, data string) {
		t.Helper()
		b, err := HandleSSEEvent("sid", string(phase), 0, []byte(data), opts)
		if err != nil {
			t.Fatalf("%s: %v", phase, err)
		}
		out = append(out, b...)
	}
	add(StreamPhaseStart, `data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":10}}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`)
	add(StreamPhaseEvent, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)
	b, _ := HandleSSEEvent("sid", string(StreamPhaseEnd), 0, nil, opts)
	out = append(out, b...)

	chunks := geminiChunksFromSSE(t, string(out))
	var text string
	var finish string
	for _, c := range chunks {
		for _, p := range c.cand.Content.Parts {
			text += p.Text
		}
		if c.cand.FinishReason != "" {
			finish = c.cand.FinishReason
		}
	}
	if text != "Hello" {
		t.Fatalf("text: want Hello, got %q\n%s", text, out)
	}
	if finish != "STOP" {
		t.Fatalf("finishReason: want STOP, got %q", finish)
	}
}

func TestAnthropicSSEToGeminiStreamToolUse(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolGemini})
	opts := &ConvertOptions{MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response", URI: "/v1beta/models/gemini-pro:streamGenerateContent"}

	var out []byte
	add := func(phase StreamPhase, data string) {
		t.Helper()
		b, err := HandleSSEEvent("sid", string(phase), 0, []byte(data), opts)
		if err != nil {
			t.Fatalf("%s: %v", phase, err)
		}
		out = append(out, b...)
	}
	add(StreamPhaseStart, `data: {"type":"message_start","message":{"id":"msg_1","model":"claude"}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read"}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"p\":1}"}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_stop","index":0}`)
	add(StreamPhaseEvent, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`)
	b, _ := HandleSSEEvent("sid", string(StreamPhaseEnd), 0, nil, opts)
	out = append(out, b...)

	chunks := geminiChunksFromSSE(t, string(out))
	var call *GeminiFunctionCall
	for _, c := range chunks {
		for _, p := range c.cand.Content.Parts {
			if p.FunctionCall != nil {
				call = p.FunctionCall
			}
		}
	}
	if call == nil || call.Name != "read" {
		t.Fatalf("functionCall missing:\n%s", out)
	}
	args, _ := json.Marshal(call.Args)
	if !strings.Contains(string(args), `"p":1`) {
		t.Fatalf("args: want p=1, got %s", args)
	}
}

func TestOpenAISSEToGeminiStream(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolGemini})
	opts := &ConvertOptions{MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response", URI: "/v1beta/models/gemini-pro:streamGenerateContent"}

	var out []byte
	add := func(phase StreamPhase, data string) {
		t.Helper()
		b, err := HandleSSEEvent("sid", string(phase), 0, []byte(data), opts)
		if err != nil {
			t.Fatalf("%s: %v", phase, err)
		}
		out = append(out, b...)
	}
	add(StreamPhaseStart, `data: {"id":"chatcmpl-1","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`)
	add(StreamPhaseEvent, `data: {"id":"chatcmpl-1","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	b, _ := HandleSSEEvent("sid", string(StreamPhaseEnd), 0, nil, opts)
	out = append(out, b...)

	chunks := geminiChunksFromSSE(t, string(out))
	var text, finish string
	var usage *GeminiUsage
	for _, c := range chunks {
		for _, p := range c.cand.Content.Parts {
			text += p.Text
		}
		if c.cand.FinishReason != "" {
			finish = c.cand.FinishReason
		}
		usage = c.usage
	}
	if text != "Hi" || finish != "STOP" {
		t.Fatalf("text/finish: want Hi/STOP, got %q/%q\n%s", text, finish, out)
	}
	if usage == nil || usage.PromptTokenCount != 10 {
		t.Fatalf("usageMetadata missing: %+v", usage)
	}
}

func TestOpenAISSEToGeminiStreamToolUse(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolGemini})
	opts := &ConvertOptions{MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response", URI: "/v1beta/models/gemini-pro:streamGenerateContent"}

	var out []byte
	add := func(phase StreamPhase, data string) {
		t.Helper()
		b, err := HandleSSEEvent("sid", string(phase), 0, []byte(data), opts)
		if err != nil {
			t.Fatalf("%s: %v", phase, err)
		}
		out = append(out, b...)
	}
	add(StreamPhaseStart, `data: {"id":"chatcmpl-1","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"p\":1}"}}]},"finish_reason":null}]}`)
	add(StreamPhaseEvent, `data: {"id":"chatcmpl-1","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
	b, _ := HandleSSEEvent("sid", string(StreamPhaseEnd), 0, nil, opts)
	out = append(out, b...)

	chunks := geminiChunksFromSSE(t, string(out))
	var call *GeminiFunctionCall
	for _, c := range chunks {
		for _, p := range c.cand.Content.Parts {
			if p.FunctionCall != nil {
				call = p.FunctionCall
			}
		}
	}
	if call == nil || call.Name != "read" {
		t.Fatalf("functionCall missing:\n%s", out)
	}
	args, _ := json.Marshal(call.Args)
	if !strings.Contains(string(args), `"p":1`) {
		t.Fatalf("args: want p=1, got %s", args)
	}
}

func TestGeminiChunkToAnthropic_Thought(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolAnthropic})
	evt := &SSEEvent{Data: `{"candidates":[{"content":{"parts":[{"text":"pondering","thought":true,"thoughtSignature":"sig-1"}],"role":"model"}}]}`}
	out, err := convertGeminiStreamChunk(evt, &ConvertOptions{SessionStore: store, SID: "sid"})
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	if !strings.Contains(s, `"type":"thinking"`) || !strings.Contains(s, `"thinking_delta"`) {
		t.Fatalf("thought not mapped to thinking block:\n%s", s)
	}
	if !strings.Contains(s, `"signature_delta"`) || !strings.Contains(s, `"sig-1"`) {
		t.Fatalf("thought signature missing:\n%s", s)
	}
}

func TestAnthropicSSEToOpenAIStream(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolOpenAIChat})
	opts := &ConvertOptions{
		MaxTokens:    100,
		SessionStore: store,
		SID:          "sid",
		Direction:    "response",
		URI:          "/v1/messages",
	}

	var out []byte
	add := func(phase StreamPhase, data string) {
		t.Helper()
		b, err := HandleSSEEvent("sid", string(phase), 0, []byte(data), opts)
		if err != nil {
			t.Fatalf("%s: %v", phase, err)
		}
		out = append(out, b...)
		out = append(out, '\n')
	}

	add(StreamPhaseStart, `data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"output_tokens":0}}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`)
	add(StreamPhaseEvent, `data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`)
	b, err := HandleSSEEvent("sid", string(StreamPhaseEnd), 0, nil, opts)
	if err != nil {
		t.Fatalf("end: %v", err)
	}
	out = append(out, b...)

	s := string(out)
	if !strings.Contains(s, `"role":"assistant"`) {
		t.Fatalf("no OpenAI role chunk:\n%s", s)
	}
	if !strings.Contains(s, `"content":"Hello"`) {
		t.Fatalf("no content delta:\n%s", s)
	}
	if !strings.Contains(s, `"finish_reason":"stop"`) {
		t.Fatalf("no stop finish_reason:\n%s", s)
	}
	if !strings.Contains(s, "data: [DONE]") {
		t.Fatalf("no [DONE]:\n%s", s)
	}
	if strings.Contains(s, "event: message_start") {
		t.Fatalf("Anthropic event leaked to OpenAI client:\n%s", s)
	}
}

func TestAnthropicSSEToOpenAIStreamToolUse(t *testing.T) {
	store := NewSessionStore()
	store.Set("sid", &Session{ID: "sid", From: ProtocolOpenAIChat})
	opts := &ConvertOptions{MaxTokens: 100, SessionStore: store, SID: "sid", Direction: "response", URI: "/v1/messages"}

	var out []byte
	add := func(phase StreamPhase, data string) {
		t.Helper()
		b, err := HandleSSEEvent("sid", string(phase), 0, []byte(data), opts)
		if err != nil {
			t.Fatalf("%s: %v", phase, err)
		}
		out = append(out, b...)
	}
	add(StreamPhaseStart, `data: {"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":10}}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read"}}`)
	add(StreamPhaseEvent, `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"p\":1}"}}`)
	add(StreamPhaseEvent, `data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`)
	b, _ := HandleSSEEvent("sid", string(StreamPhaseEnd), 0, nil, opts)
	out = append(out, b...)

	s := string(out)
	if !strings.Contains(s, `"tool_calls"`) || !strings.Contains(s, `"name":"read"`) {
		t.Fatalf("no tool call chunk:\n%s", s)
	}
	if !strings.Contains(s, `"finish_reason":"tool_calls"`) {
		t.Fatalf("finish_reason not tool_calls:\n%s", s)
	}
}

func TestGeminiLifecycle_AnthropicStreamParallelCalls(t *testing.T) {
	c := NewGeminiToAnthropicStreamConverter("claude-sonnet-4-20250514")
	c.HandleStreamStart()
	out, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[
		{"functionCall":{"name":"alpha","args":{"x":1}}},
		{"functionCall":{"name":"beta","args":{"y":2}}}
	],"role":"model"},"finishReason":"STOP"}]}`))
	if err != nil {
		t.Fatalf("chunk: %v", err)
	}
	if got := strings.Count(string(out), "event: content_block_start"); got != 2 {
		t.Fatalf("want 2 tool_use blocks, got %d:\n%s", got, out)
	}
	if !strings.Contains(string(out), `"name":"alpha"`) || !strings.Contains(string(out), `"name":"beta"`) {
		t.Fatalf("both calls must reach the client:\n%s", out)
	}
}

func TestGeminiLifecycle_AnthropicStreamCacheRead(t *testing.T) {
	c := NewGeminiToAnthropicStreamConverter("claude-sonnet-4-20250514")
	c.HandleStreamStart()
	_, err := c.HandleChunk([]byte(`{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":50,"totalTokenCount":1050,"cachedContentTokenCount":800}}`))
	if err != nil {
		t.Fatalf("chunk: %v", err)
	}
	end := string(c.HandleStreamEnd())
	if !strings.Contains(end, `"input_tokens":200`) {
		t.Fatalf("input_tokens: want 200 in: %s", end)
	}
	if !strings.Contains(end, `"cache_read_input_tokens":800`) {
		t.Fatalf("cache_read_input_tokens: want 800 in: %s", end)
	}
}

func TestResponses_AnthropicCacheUsage(t *testing.T) {
	body := `{
		"id":"msg_1","type":"message","role":"assistant",
		"content":[{"type":"text","text":"hi"}],
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"usage":{"input_tokens":200,"output_tokens":50,
			"cache_read_input_tokens":800,"cache_creation_input_tokens":100}
	}`
	out, err := ConvertAnthropicToResponses([]byte(body), &ConvertOptions{})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	var r ResponsesResponse
	if err := json.Unmarshal(out, &r); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if r.Usage == nil {
		t.Fatalf("no usage: %s", out)
	}
	if r.Usage.InputTokens != 1100 {
		t.Fatalf("input_tokens: want 1100, got %d", r.Usage.InputTokens)
	}
	if r.Usage.TotalTokens != 1150 {
		t.Fatalf("total_tokens: want 1150, got %d", r.Usage.TotalTokens)
	}
	if r.Usage.InputTokensDetails == nil || r.Usage.InputTokensDetails.CachedTokens != 800 {
		t.Fatalf("cached_tokens: want 800, got %+v", r.Usage.InputTokensDetails)
	}
}

func TestResponses_AnthropicStreamCacheUsage(t *testing.T) {
	c := NewAnthropicStreamConverter("claude-sonnet-4-20250514")
	_, err := c.HandleChunk([]byte(`{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":200,"output_tokens":50,"cache_read_input_tokens":800,"cache_creation_input_tokens":100}}}`))
	if err != nil {
		t.Fatalf("chunk: %v", err)
	}
	if c.response.Usage == nil {
		t.Fatal("no usage")
	}
	if got := c.response.Usage.InputTokens; got != 1100 {
		t.Fatalf("input_tokens: want 1100, got %d", got)
	}
	if got := c.response.Usage.TotalTokens; got != 1150 {
		t.Fatalf("total_tokens: want 1150, got %d", got)
	}
	if c.response.Usage.InputTokensDetails == nil || c.response.Usage.InputTokensDetails.CachedTokens != 800 {
		t.Fatalf("cached_tokens: want 800, got %+v", c.response.Usage.InputTokensDetails)
	}
}

func TestResponses_ChatStreamCacheUsage(t *testing.T) {
	c := NewResponsesStreamConverter("gpt-4")
	c.setUsage(&OpenAIUsage{
		PromptTokens:     1000,
		CompletionTokens: 50,
		TotalTokens:      1050,
		PromptTokensDetails: &OpenAIPromptTokensDetails{
			CachedTokens: 800,
		},
	})
	if c.response.Usage == nil {
		t.Fatal("no usage")
	}
	if c.response.Usage.InputTokensDetails == nil || c.response.Usage.InputTokensDetails.CachedTokens != 800 {
		t.Fatalf("cached_tokens: want 800, got %+v", c.response.Usage.InputTokensDetails)
	}
}

func TestStream_RestrictedToolFragmentsStaySkipped(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil, []string{"allowed"})
	sc.HandleStreamStart()

	// First fragment names an undeclared tool: suppressed.
	out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_bad","function":{"name":"bad","arguments":"{}"}}]},"finish_reason":null}]}`))
	if strings.Contains(string(out), "tool_use") {
		t.Fatalf("undeclared tool was emitted: %s", out)
	}

	// A later fragment of the same call has no name: it must stay
	// suppressed instead of opening an empty-named tool_use block.
	out, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]},"finish_reason":null}]}`))
	if strings.Contains(string(out), "tool_use") || strings.Contains(string(out), "input_json_delta") {
		t.Fatalf("suppressed tool leaked through later fragments: %s", out)
	}
}

// TestStream_MissingToolIndexStartsNewCall covers providers (Gemini's
// OpenAI-compatible endpoint, some relays) that omit tool_calls[].index on
// every fragment. Two calls in one delta must not collapse into one.
func TestStream_MissingToolIndexStartsNewCall(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil, nil)
	sc.HandleStreamStart()

	out, err := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[
		{"id":"call_a","type":"function","function":{"name":"alpha","arguments":"{}"}},
		{"id":"call_b","type":"function","function":{"name":"beta","arguments":"{}"}}
	]},"finish_reason":null}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(out), "event: content_block_start"); got != 2 {
		t.Fatalf("want 2 tool_use blocks, got %d:\n%s", got, out)
	}
	if !strings.Contains(string(out), `"name":"alpha"`) || !strings.Contains(string(out), `"name":"beta"`) {
		t.Fatalf("both tool names must reach the client:\n%s", out)
	}
}

// TestStream_MissingToolIndexKeepsFragmentsTogether covers a continuation
// fragment with no index and no id: it belongs to the call opened last.
func TestStream_MissingToolIndexKeepsFragmentsTogether(t *testing.T) {
	sc := NewStreamConverter("claude-sonnet-4-20250514", nil, nil)
	sc.HandleStreamStart()

	_, _ = sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_a","type":"function","function":{"name":"alpha","arguments":"{\"x\":"}}]},"finish_reason":null}]}`))
	out, _ := sc.HandleChunk([]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"type":"function","function":{"arguments":"1}"}}]},"finish_reason":null}]}`))
	if strings.Contains(string(out), "content_block_start") {
		t.Fatalf("continuation opened a new block: %s", out)
	}
	if !strings.Contains(string(out), `"partial_json":"1}"`) {
		t.Fatalf("continuation arguments missing: %s", out)
	}
}
