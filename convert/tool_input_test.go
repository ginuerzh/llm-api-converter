package convert

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestAnthropicToolInputAlwaysObject(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want map[string]any
	}{
		{"empty", "", map[string]any{}},
		{"whitespace", " \n\t ", map[string]any{}},
		{"null", "null", map[string]any{}},
		{"invalid JSON", "not json", map[string]any{"input": "not json"}},
		{"invalid JSON with padding", "  not json  ", map[string]any{"input": "not json"}},
		{"array", `[1,2]`, map[string]any{"input": []any{float64(1), float64(2)}}},
		{"string", `"hello"`, map[string]any{"input": "hello"}},
		{"number", "42", map[string]any{"input": float64(42)}},
		{"valid object", `{"a":1}`, map[string]any{"a": float64(1)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := anthropicToolInput(tc.in)
			obj, ok := got.(map[string]any)
			if !ok {
				t.Fatalf("anthropicToolInput(%q) = %T, want object", tc.in, got)
			}
			if !reflect.DeepEqual(obj, tc.want) {
				t.Fatalf("anthropicToolInput(%q) = %#v, want %#v", tc.in, obj, tc.want)
			}
			b, err := json.Marshal(obj)
			if err != nil {
				t.Fatalf("marshal result: %v", err)
			}
			var round map[string]any
			if err := json.Unmarshal(b, &round); err != nil {
				t.Fatalf("result is not a JSON object: %s", b)
			}
		})
	}
}

func TestOpenAIToolArgumentsAlwaysObject(t *testing.T) {
	var nilMap map[string]any
	cases := []struct {
		name string
		in   any
		want string
	}{
		{"nil", nil, "{}"},
		{"typed nil map", nilMap, "{}"},
		{"empty object", map[string]any{}, "{}"},
		{"valid object", map[string]any{"a": float64(1)}, `{"a":1}`},
		{"array", []any{float64(1), float64(2)}, `{"input":[1,2]}`},
		{"string", "hello", `{"input":"hello"}`},
		{"number", float64(42), `{"input":42}`},
		{"bool", true, `{"input":true}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := openAIToolArguments(tc.in)
			if got != tc.want {
				t.Fatalf("openAIToolArguments(%#v) = %q, want %q", tc.in, got, tc.want)
			}
			var obj map[string]any
			if err := json.Unmarshal([]byte(got), &obj); err != nil {
				t.Fatalf("result %q is not a JSON object: %v", got, err)
			}
		})
	}
}
