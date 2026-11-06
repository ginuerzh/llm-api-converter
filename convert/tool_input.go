package convert

import (
	"encoding/json"
	"strings"
)

// anthropicToolInput turns OpenAI tool arguments into an Anthropic tool_use
// input object. Empty, null, invalid, or non-object JSON is preserved as an
// object so the tool_use block stays schema-valid.
func anthropicToolInput(arguments string) any {
	s := strings.TrimSpace(arguments)
	if s == "" {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return map[string]any{"input": s}
	}
	if v == nil {
		return map[string]any{}
	}
	if _, ok := v.(map[string]any); ok {
		return v
	}
	return map[string]any{"input": v}
}

// openAIToolArguments turns an Anthropic tool_use input into the JSON string
// OpenAI expects. Nil or null becomes {}; non-object input is wrapped so the
// result is always a JSON object.
func openAIToolArguments(input any) string {
	if input == nil {
		return "{}"
	}
	if _, ok := input.(map[string]any); !ok {
		input = map[string]any{"input": input}
	}
	b, err := json.Marshal(input)
	if err != nil || len(b) == 0 || string(b) == "null" {
		return "{}"
	}
	return string(b)
}
