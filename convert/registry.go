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
	// Anthropic → OpenAI Chat (request and response).
	conversions[ConversionKey{ProtocolAnthropic, ProtocolOpenAIChat}] = convertAnthropicBodyToOpenAI
	// OpenAI Chat → Anthropic (request and response).
	conversions[ConversionKey{ProtocolOpenAIChat, ProtocolAnthropic}] = convertOpenAIBodyToAnthropic
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
