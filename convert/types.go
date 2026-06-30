package convert

import (
	"encoding/json"
	"strings"
)

// -------- OpenAI Chat Completions (Request) --------

// OpenAIChatRequest is an OpenAI /v1/chat/completions request body.
type OpenAIChatRequest struct {
	Model    string          `json:"model"`
	Messages []OpenAIMessage `json:"messages"`

	MaxTokens      *int       `json:"max_tokens,omitempty"`
	Temperature    *float64   `json:"temperature,omitempty"`
	TopP           *float64   `json:"top_p,omitempty"`
	TopK           *int       `json:"top_k,omitempty"` // x Groq
	Stop           any        `json:"stop,omitempty"`  // string or []string
	Stream         *bool      `json:"stream,omitempty"`
	StreamOptions  any        `json:"stream_options,omitempty"`
	Tools          []OpenAITool `json:"tools,omitempty"`
	ToolChoice     any        `json:"tool_choice,omitempty"`
	Thinking       any        `json:"thinking,omitempty"`        // DeepSeek extended thinking
	ReasoningEffort any       `json:"reasoning_effort,omitempty"` // DeepSeek reasoning effort
	ThinkingBudget *int       `json:"thinking_budget,omitempty"` // DeepSeek thinking budget
	OpenAIPromptCacheKey string `json:"prompt_cache_key,omitempty"` // upstream prompt cache key
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"` // ignored
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`  // ignored
	N              *int       `json:"n,omitempty"`               // ignored
	LogitBias      any        `json:"logit_bias,omitempty"`      // ignored
	User           string     `json:"user,omitempty"`            // ignored
	Metadata       any        `json:"metadata,omitempty"`
}

type OpenAIMessage struct {
	Role             string              `json:"role"`
	Content          any                 `json:"content"`                       // string or []ContentPart or null
	ToolCalls        []OpenAIToolCall    `json:"tool_calls,omitempty"`
	FunctionCall     *OpenAIFunctionCall `json:"function_call,omitempty"`        // legacy (name+arguments at top level)
	ToolCallID       string              `json:"tool_call_id,omitempty"`         // tool role
	Name             string              `json:"name,omitempty"`                  // function role
	ReasoningContent string              `json:"reasoning_content,omitempty"`   // DeepSeek/GLM thinking
}

// OpenAIContentPart is an element in a multi-part user message.
type OpenAIContentPart struct {
	Type     string           `json:"type"`               // "text" or "image_url"
	Text     string           `json:"text,omitempty"`
	ImageURL *OpenAIImageURL  `json:"image_url,omitempty"`
}

type OpenAIImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // ignored
}

type OpenAIToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function OpenAIFunctionCall  `json:"function"`
}

type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // raw JSON string, re-marshaled later
}

type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters"`
}

// -------- OpenAI Chat Completions (Response) --------

type OpenAIChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage    `json:"usage"`
}

type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason *string       `json:"finish_reason"`
	LogProbs     any           `json:"logprobs"` // always null in our case
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// -------- Anthropic Messages (Request) --------

// AnthropicThinking is the extended thinking configuration.
type AnthropicThinking struct {
	Type         string `json:"type"`                    // "enabled" or "disabled"
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// AnthropicOutputConfig is the output configuration block (e.g. effort).
type AnthropicOutputConfig struct {
	Effort string `json:"effort,omitempty"` // "low"|"medium"|"high"|"max"
}

type AnthropicRequest struct {
	Model         string                 `json:"model"`
	Messages      []AnthropicMessage     `json:"messages"`
	System        []AnthropicTextBlock   `json:"system,omitempty"`
	MaxTokens     int                    `json:"max_tokens"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	TopK          *int                   `json:"top_k,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Stream        *bool                  `json:"stream,omitempty"`
	Thinking      *AnthropicThinking     `json:"thinking,omitempty"`
	OutputConfig  *AnthropicOutputConfig `json:"output_config,omitempty"`
	Tools         []AnthropicTool        `json:"tools,omitempty"`
	ToolChoice    *AnthropicToolChoice   `json:"tool_choice,omitempty"`
	Metadata      any                    `json:"metadata,omitempty"`
}

// UnmarshalJSON implements json.Unmarshaler and normalizes the system field
// which can be either a plain string (legacy) or an array of text blocks (current).
func (r *AnthropicRequest) UnmarshalJSON(data []byte) error {
	// Shadow type to break recursion.
	type shadow struct {
		Model         string                 `json:"model"`
		Messages      []AnthropicMessage     `json:"messages"`
		System        json.RawMessage        `json:"system,omitempty"`
		MaxTokens     int                    `json:"max_tokens"`
		Temperature   *float64               `json:"temperature,omitempty"`
		TopP          *float64               `json:"top_p,omitempty"`
		TopK          *int                   `json:"top_k,omitempty"`
		StopSequences []string               `json:"stop_sequences,omitempty"`
		Stream        *bool                  `json:"stream,omitempty"`
		Thinking      *AnthropicThinking     `json:"thinking,omitempty"`
		OutputConfig  *AnthropicOutputConfig `json:"output_config,omitempty"`
		Tools         []AnthropicTool        `json:"tools,omitempty"`
		ToolChoice    *AnthropicToolChoice   `json:"tool_choice,omitempty"`
		Metadata      any                    `json:"metadata,omitempty"`
	}
	var s shadow
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	r.Model = s.Model
	r.Messages = s.Messages
	r.MaxTokens = s.MaxTokens
	r.Temperature = s.Temperature
	r.TopP = s.TopP
	r.TopK = s.TopK
	r.StopSequences = s.StopSequences
	r.Stream = s.Stream
	r.Thinking = s.Thinking
	r.OutputConfig = s.OutputConfig
	r.Tools = s.Tools
	r.ToolChoice = s.ToolChoice
	r.Metadata = s.Metadata

	if len(s.System) == 0 || string(s.System) == "null" {
		r.System = nil
		return nil
	}

	// If it's a string, wrap it into a single text block.
	if s.System[0] == '"' {
		var text string
		if err := json.Unmarshal(s.System, &text); err != nil {
			return err
		}
		r.System = []AnthropicTextBlock{{Type: "text", Text: text}}
		return nil
	}

	return json.Unmarshal(s.System, &r.System)
}

// modelProfile caches the detection result for a single request so that
// repeated isDeepSeekModel/isOpenAIStyleModel calls don't re-parse the string.
type modelProfile struct {
	model    string
	isDeepSeek bool
	isOpenAI   bool
}

func classifyModel(model string) modelProfile {
	return modelProfile{
		model:      model,
		isDeepSeek: isDeepSeekModel(model),
		isOpenAI:   isOpenAIStyleModel(model),
	}
}

func isDeepSeekModel(model string) bool {
	ml := strings.ToLower(model)
	return strings.HasPrefix(ml, "deepseek")
}

func isGLMModel(model string) bool {
	ml := strings.ToLower(model)
	return strings.HasPrefix(ml, "glm-")
}

func isOpenAIStyleModel(model string) bool {
	ml := strings.ToLower(model)
	prefixes := []string{"gpt-", "o1", "o3", "deepseek", "gemini-", "glm-"}
	for _, p := range prefixes {
		if strings.HasPrefix(ml, p) {
			return true
		}
	}
	return false
}

type AnthropicMessage struct {
	Role    string             `json:"role"`
	Content []AnthropicContent `json:"content"`
}

// UnmarshalJSON implements json.Unmarshaler and normalizes the content field
// which can be either a plain string (deprecated Anthropic API format) or
// an array of typed content blocks (current format).
func (m *AnthropicMessage) UnmarshalJSON(data []byte) error {
	// Shadow type to break recursion.
	type shadow struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	var s shadow
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	m.Role = s.Role

	if len(s.Content) == 0 || string(s.Content) == "null" {
		m.Content = nil
		return nil
	}

	if s.Content[0] == '"' {
		var text string
		if err := json.Unmarshal(s.Content, &text); err != nil {
			return err
		}
		m.Content = []AnthropicContent{{Type: "text", Text: text}}
		return nil
	}

	return json.Unmarshal(s.Content, &m.Content)
}

type AnthropicContent struct {
	Type    string `json:"type"`
	Text    string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"` // extended thinking

	// image
	Source    *AnthropicImageSource `json:"source,omitempty"`

	// tool_use
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     any    `json:"input,omitempty"`

	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"` // string or []AnthropicContent
	IsError   bool   `json:"is_error,omitempty"`
}

type AnthropicTextBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type AnthropicImageSource struct {
	Type      string `json:"type"`       // "base64"
	MediaType string `json:"media_type"` // e.g. "image/jpeg"
	Data      string `json:"data"`
}

type AnthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type AnthropicTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema"`
}

// -------- Anthropic Messages (Response) --------

type AnthropicResponse struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Role         string              `json:"role"`
	Content      []AnthropicContent  `json:"content"`
	Model        string              `json:"model"`
	StopReason   *string             `json:"stop_reason"`
	StopSequence *string             `json:"stop_sequence"`
	Usage        AnthropicUsage      `json:"usage"`
}

type AnthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// -------- SSE Event --------

// SSEEvent represents a single Server-Sent Event as defined by the SSE spec.
// Fields correspond to the event stream format:
//
//	[event: <type>]
//	data: <payload>
//	[id: <id>]
//	[retry: <ms>]
type SSEEvent struct {
	Event string // event type (e.g. "message_start", "content_block_delta")
	Data  string // JSON payload (the data: value)
	ID    string // optional last-event-id
	Retry int    // optional reconnection time in ms
}

// -------- ModelMap --------

// ModelMapEntry maps a source model prefix to a target model ID.
type ModelMapEntry struct {
	SourcePrefix string // lowercase prefix (e.g. "claude-opus"), or "*" for catch-all
	TargetModel  string // target model (e.g. "deepseek-v4-pro")
	Protocol     string // downstream protocol: "" (auto-detect), "openai", "anthropic"
}

// ModelMap is an ordered list of prefix-based model mapping rules, checked in order.
type ModelMap []ModelMapEntry

// Apply checks sourceModel against all entries and returns the target, protocol override,
// and true on the first match. Specific prefixes are checked first in declaration order;
// a catch-all entry ("*") is checked last only if no specific prefix matched.
func (mm ModelMap) Apply(sourceModel string) (string, string, bool) {
	sourceModel = strings.ToLower(sourceModel)
	var catchAll ModelMapEntry
	for _, entry := range mm {
		if entry.SourcePrefix == "*" {
			catchAll = entry
			continue
		}
		if strings.HasPrefix(sourceModel, entry.SourcePrefix) {
			return entry.TargetModel, entry.Protocol, true
		}
	}
	if catchAll.SourcePrefix != "" {
		return catchAll.TargetModel, catchAll.Protocol, true
	}
	return "", "", false
}

// SourcePrefix does a case-insensitive reverse lookup: returns the source
// prefix of the first entry whose TargetModel is a prefix of targetModel.
// Used to recover the original client-facing model name from an upstream
// response so that Claude Code's safety classifier sees the expected model.
func (mm ModelMap) SourcePrefix(targetModel string) string {
	targetModel = strings.ToLower(targetModel)
	for _, entry := range mm {
		t := strings.ToLower(entry.TargetModel)
		if entry.SourcePrefix != "*" && strings.HasPrefix(targetModel, t) {
			return entry.SourcePrefix
		}
	}
	return ""
}

// LookupTarget does a case-insensitive reverse lookup: returns the protocol
// of the entry whose TargetModel matches the given model name. Used when a
// request or response model is already the mapped target (e.g. downstream
// API response) and the source-prefix path did not match.
func (mm ModelMap) LookupTarget(targetModel string) string {
	targetModel = strings.ToLower(targetModel)
	for _, entry := range mm {
		if strings.ToLower(entry.TargetModel) == targetModel && entry.Protocol != "" {
			return entry.Protocol
		}
	}
	return ""
}

// -------- ConvertOptions --------

// ConvertOptions controls the conversion behavior.
type ConvertOptions struct {
	// Model is the fallback model ID for both request directions.
	// Used only when mapping misses and the input model is empty.
	Model string
	// MaxTokens is the default max_tokens value for Anthropic requests.
	MaxTokens int
	// ModelMap is an optional prefix-based model mapping table for both
	// request directions. Checked before passthrough. Empty = passthrough only.
	ModelMap ModelMap
	// SSEPhase is the stream lifecycle phase: "start", "event", or "end".
	// Empty means non-streaming conversion.
	SSEPhase string
	// SID is the GOST session ID for correlating stream lifecycle calls.
	SID string
	// EventIndex is the 0-based event index within a stream (only set when SSEPhase is "event").
	EventIndex int
	// ReasoningCache is an optional cache for storing/replaying reasoning_content
	// across requests. Used to handle DeepSeek V4's requirement that
	// reasoning_content must be replayed when tool_calls are present.
	ReasoningCache *ReasoningCache
	// DeclaredTools lists the tool names declared in the original Anthropic request.
	// When non-nil, tool_use blocks not in this list are filtered from the response
	// to prevent upstream tool hallucination.
	DeclaredTools []string
	// StreamErrorMsg, when non-empty, causes HandleSSEEvent to emit an error event
	// instead of normal stream conversion.
	StreamErrorMsg string
	// RequestModel is the original (unmapped) model name from the client request.
	// When set, Convert uses it for the model field in Anthropic responses so
	// that Claude Code's safety classifier sees the expected model name.
	// ponytail: safety classifier requires original model name in responses
	RequestModel string
}

// -------- OpenAI Streaming Types --------

// OpenAIStreamChunk is an OpenAI /v1/chat/completions streaming response chunk.
type OpenAIStreamChunk struct {
	ID      string              `json:"id,omitempty"`
	Object  string              `json:"object,omitempty"`
	Model   string              `json:"model,omitempty"`
	Choices []OpenAIStreamChoice `json:"choices"`
	Usage   *OpenAIUsage        `json:"usage,omitempty"`
}

// OpenAIStreamChoice is a single choice in a streaming chunk (delta instead of message).
type OpenAIStreamChoice struct {
	Index        int         `json:"index"`
	Delta        OpenAIDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// OpenAIDelta is the delta payload inside a streaming chunk.
type OpenAIDelta struct {
	Role             string               `json:"role,omitempty"`
	Content          string               `json:"content,omitempty"`
	ReasoningContent string               `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIDeltaToolCall `json:"tool_calls,omitempty"`
	FunctionCall     *OpenAIFunctionCall   `json:"function_call,omitempty"`
}

// OpenAIDeltaToolCall is a tool call entry inside a streaming delta.
// Unlike OpenAIToolCall (which has no array index since non-streaming
// tool_calls are plain arrays), streaming tool calls carry an .index
// field identifying their position in the array.
type OpenAIDeltaToolCall struct {
	Index    int               `json:"index"`
	ID       string            `json:"id,omitempty"`
	Type     string            `json:"type,omitempty"`
	Function OpenAIFunctionCall `json:"function"`
}

// streamToolState tracks partial tool call state across deltas.
type streamToolState struct {
	ID        string
	Name      string
	Arguments string
}

// StreamPhase describes the SSE stream lifecycle phase.
type StreamPhase string

const (
	StreamPhaseStart StreamPhase = "start"
	StreamPhaseEvent StreamPhase = "event"
	StreamPhaseEnd   StreamPhase = "end"
	StreamPhaseError StreamPhase = "error"
)
