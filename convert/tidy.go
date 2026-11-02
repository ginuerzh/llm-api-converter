package convert

// tidyOpenAIChunk repairs the Chat Completions stream shapes that strict
// clients reject. It reports whether anything changed.
//
// The cases come from real upstreams and their clients:
//
//   - "reasoning_content": "" after the thinking is over: some clients
//     (Qoder) read it as the thinking starting again.
//   - "tool_calls": [] on text and thinking deltas: some clients close the
//     current block on the field being present.
//   - "finish_reason": "" instead of null while a chunk is unfinished: the
//     AI SDK rejects the chunk.
//   - a tool call fragment with no "index": the AI SDK's OpenAI provider
//     rejects the chunk ("expected number, received undefined").
//   - an empty "function.name" on a later fragment: a client that merges
//     by overwriting fields assembles a call named "".
//   - "finish_reason": "stop" on a chunk that made tool calls: an agent
//     ends its turn before running them.
func tidyOpenAIChunk(raw map[string]any) bool {
	choices, _ := raw["choices"].([]any)
	changed := false
	for _, ch := range choices {
		c, ok := ch.(map[string]any)
		if !ok {
			continue
		}

		if fr, ok := c["finish_reason"].(string); ok && fr == "" {
			c["finish_reason"] = nil
			changed = true
		}

		for _, key := range []string{"delta", "message"} {
			m, ok := c[key].(map[string]any)
			if !ok {
				continue
			}
			if rc, ok := m["reasoning_content"].(string); ok && rc == "" {
				delete(m, "reasoning_content")
				changed = true
			}
			calls, ok := m["tool_calls"].([]any)
			if !ok {
				continue
			}
			if len(calls) == 0 {
				delete(m, "tool_calls")
				changed = true
				continue
			}
			for i, t := range calls {
				call, ok := t.(map[string]any)
				if !ok {
					continue
				}
				if _, ok := call["index"]; !ok {
					call["index"] = i
					changed = true
				}
				fn, ok := call["function"].(map[string]any)
				if !ok {
					continue
				}
				if name, ok := fn["name"].(string); ok && name == "" {
					delete(fn, "name")
					changed = true
				}
			}
		}

		if fr, _ := c["finish_reason"].(string); fr == "stop" && choiceMadeCalls(c) {
			c["finish_reason"] = "tool_calls"
			changed = true
		}
	}
	return changed
}

// choiceMadeCalls reports whether a Chat choice carries tool calls in its
// delta or message.
func choiceMadeCalls(c map[string]any) bool {
	for _, key := range []string{"delta", "message"} {
		m, _ := c[key].(map[string]any)
		calls, _ := m["tool_calls"].([]any)
		if len(calls) > 0 {
			return true
		}
	}
	return false
}
