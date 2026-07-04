package convert

import "encoding/json"

// ConversionKey identifies a conversion direction: from → to.
type ConversionKey struct {
	From Protocol
	To   Protocol
}

// conversions maps conversion keys to their non-streaming converter functions.
// Passthrough (from == to) is handled by the caller, not registered here.
var conversions = map[ConversionKey]func([]byte, *ConvertOptions) ([]byte, error){}

func init() {
	// Anthropic ↔ OpenAI Chat (request and response).
	conversions[ConversionKey{ProtocolAnthropic, ProtocolOpenAIChat}] = convertAnthropicBodyToOpenAI
	conversions[ConversionKey{ProtocolOpenAIChat, ProtocolAnthropic}] = convertOpenAIBodyToAnthropic

	// OpenAI Chat ↔ Gemini (request and response).
	conversions[ConversionKey{ProtocolOpenAIChat, ProtocolGemini}] = convertOpenAIBodyToGemini
	conversions[ConversionKey{ProtocolGemini, ProtocolOpenAIChat}] = convertGeminiBodyToOpenAI

	// Anthropic ↔ Gemini (request and response).
	conversions[ConversionKey{ProtocolAnthropic, ProtocolGemini}] = convertAnthropicBodyToGemini
	conversions[ConversionKey{ProtocolGemini, ProtocolAnthropic}] = convertGeminiBodyToAnthropic
}

// convertAnthropicBodyToOpenAI dispatches an Anthropic body (request or response)
// to the correct converter.
func convertAnthropicBodyToOpenAI(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	if _, ok := raw["messages"]; ok {
		return convertAnthropicRequestToOpenAI(body, opts)
	}
	return convertAnthropicResponseToOpenAI(body)
}

// convertOpenAIBodyToAnthropic dispatches an OpenAI body (request or response)
// to the correct converter.
func convertOpenAIBodyToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	if _, ok := raw["choices"]; ok {
		return convertOpenAIResponseToAnthropic(body, opts)
	}
	return convertOpenAIRequestToAnthropic(body, opts)
}

// convertOpenAIBodyToGemini dispatches OpenAI body → Gemini.
func convertOpenAIBodyToGemini(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	if _, ok := raw["choices"]; ok {
		return convertOpenAIResponseToGemini(body, opts)
	}
	return convertOpenAIRequestToGemini(body, opts)
}

// convertGeminiBodyToOpenAI dispatches Gemini body → OpenAI Chat.
func convertGeminiBodyToOpenAI(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	if _, ok := raw["candidates"]; ok {
		return convertGeminiResponseToOpenAI(body, opts)
	}
	return convertGeminiRequestToOpenAI(body, opts)
}

// convertAnthropicBodyToGemini dispatches Anthropic body → Gemini.
func convertAnthropicBodyToGemini(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	if _, ok := raw["messages"]; ok {
		return convertAnthropicRequestToGemini(body, opts)
	}
	return convertAnthropicResponseToGemini(body, opts)
}

// convertGeminiBodyToAnthropic dispatches Gemini body → Anthropic.
func convertGeminiBodyToAnthropic(body []byte, opts *ConvertOptions) ([]byte, error) {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return body, nil
	}
	if _, ok := raw["candidates"]; ok {
		return convertGeminiResponseToAnthropic(body, opts)
	}
	return convertGeminiRequestToAnthropic(body, opts)
}
