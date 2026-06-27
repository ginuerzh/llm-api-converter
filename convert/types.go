package convert

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
	StreamOptions  any        `json:"stream_options,omitempty"` // ignored
	Tools          []OpenAITool `json:"tools,omitempty"`
	ToolChoice     any        `json:"tool_choice,omitempty"`     // ignored
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"` // ignored
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`  // ignored
	N              *int       `json:"n,omitempty"`               // ignored
	LogitBias      any        `json:"logit_bias,omitempty"`      // ignored
	User           string     `json:"user,omitempty"`            // ignored
	Metadata       any        `json:"metadata,omitempty"`
}

type OpenAIMessage struct {
	Role      string           `json:"role"`
	Content   any              `json:"content"`    // string or []ContentPart or null
	ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"` // tool role
	Name      string           `json:"name,omitempty"`          // function role
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

type AnthropicRequest struct {
	Model         string               `json:"model"`
	Messages      []AnthropicMessage   `json:"messages"`
	System        []AnthropicTextBlock `json:"system,omitempty"`
	MaxTokens     int                  `json:"max_tokens"`
	Temperature   *float64             `json:"temperature,omitempty"`
	TopP          *float64             `json:"top_p,omitempty"`
	TopK          *int                 `json:"top_k,omitempty"`
	StopSequences []string             `json:"stop_sequences,omitempty"`
	Stream        *bool                `json:"stream,omitempty"`
	Tools         []AnthropicTool      `json:"tools,omitempty"`
	Metadata      any                  `json:"metadata,omitempty"`
}

type AnthropicMessage struct {
	Role    string              `json:"role"`
	Content []AnthropicContent  `json:"content"`
}

type AnthropicContent struct {
	Type      string                `json:"type"`
	Text      string                `json:"text,omitempty"`

	// image
	Source    *AnthropicImageSource `json:"source,omitempty"`

	// tool_use
	ID        string                `json:"id,omitempty"`
	Name      string                `json:"name,omitempty"`
	Input     any                   `json:"input,omitempty"`

	// tool_result
	ToolUseID string                `json:"tool_use_id,omitempty"`
	Content   any                   `json:"content,omitempty"` // string or []AnthropicContent
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

// -------- ConvertOptions --------

type ConvertOptions struct {
	Model     string
	MaxTokens int
}
