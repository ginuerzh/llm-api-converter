package convert

// -------- Gemini generateContent (Request) --------

// GeminiChatRequest is a Google Gemini generateContent request body.
type GeminiChatRequest struct {
	Contents         []GeminiContent        `json:"contents"`
	SystemInstruction *GeminiContent        `json:"systemInstruction,omitempty"`
	Tools            []GeminiTool           `json:"tools,omitempty"`
	ToolConfig       *GeminiToolConfig      `json:"toolConfig,omitempty"`
	SafetySettings   []GeminiSafetySetting  `json:"safetySettings,omitempty"`
	GenerationConfig *GeminiGenerationConfig `json:"generationConfig,omitempty"`
}

type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"` // "user" or "model"
}

type GeminiPart struct {
	Text                string                    `json:"text,omitempty"`
	InlineData          *GeminiInlineData         `json:"inlineData,omitempty"`
	FunctionCall        *GeminiFunctionCall       `json:"functionCall,omitempty"`
	FunctionResponse    *GeminiFunctionResponse   `json:"functionResponse,omitempty"`
	FileData            *GeminiFileData           `json:"fileData,omitempty"`
	ExecutableCode      *GeminiExecutableCode     `json:"executableCode,omitempty"`
	CodeExecutionResult *GeminiCodeExecutionResult `json:"codeExecutionResult,omitempty"`
	ThoughtSignature    string                    `json:"thoughtSignature,omitempty"`
}

// GeminiFunctionCall represents a function call part (like tool_use).
type GeminiFunctionCall struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}

// GeminiFunctionResponse represents a function result part (like tool_result).
type GeminiFunctionResponse struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
}

type GeminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type GeminiFileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type GeminiExecutableCode struct {
	Language string `json:"language"`
	Code     string `json:"code"`
}

type GeminiCodeExecutionResult struct {
	Outcome  string `json:"outcome"`
	Output   string `json:"output"`
}

type GeminiGenerationConfig struct {
	Temperature      *float64            `json:"temperature,omitempty"`
	TopP             *float64            `json:"topP,omitempty"`
	TopK             *float64            `json:"topK,omitempty"`
	MaxOutputTokens  *int                `json:"maxOutputTokens,omitempty"`
	StopSequences    []string            `json:"stopSequences,omitempty"`
	ResponseMimeType string              `json:"responseMimeType,omitempty"`
	ResponseSchema   any                 `json:"responseSchema,omitempty"`
	CandidateCount   *int                `json:"candidateCount,omitempty"`
	Seed             *int                `json:"seed,omitempty"`
	PresencePenalty  *float64            `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64            `json:"frequencyPenalty,omitempty"`
}

// GeminiTool wraps function declarations for function calling.
type GeminiTool struct {
	FunctionDeclarations []GeminiFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

type GeminiFunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// GeminiToolConfig controls function calling behavior.
type GeminiToolConfig struct {
	FunctionCallingConfig *GeminiFunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type GeminiFunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"` // "AUTO", "ANY", "NONE"
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type GeminiSafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// -------- Gemini generateContent (Response) --------

type GeminiChatResponse struct {
	Candidates     []GeminiCandidate  `json:"candidates"`
	UsageMetadata  *GeminiUsage       `json:"usageMetadata,omitempty"`
	PromptFeedback *GeminiPromptFeedback `json:"promptFeedback,omitempty"`
}

type GeminiCandidate struct {
	Content       GeminiContent       `json:"content"`
	FinishReason  string              `json:"finishReason"`
	SafetyRatings []GeminiSafetyRating `json:"safetyRatings,omitempty"`
	Index         int                 `json:"index"`
}

type GeminiSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
}

type GeminiUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type GeminiPromptFeedback struct {
	BlockReason   string                `json:"blockReason,omitempty"`
	SafetyRatings []GeminiSafetyRating  `json:"safetyRatings,omitempty"`
}
