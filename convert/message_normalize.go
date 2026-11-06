package convert

import "strings"

const (
	// missingToolResultText is the content of a synthetic tool_result that
	// answers a tool_use whose result never appeared in the history.
	missingToolResultText = "Tool result missing."
	// orphanToolResultText replaces an orphan tool_result whose content is
	// empty. Anthropic rejects empty text blocks.
	orphanToolResultText = "Tool result received."
	// stubToolDescription marks synthetic declarations added for tool calls
	// that appear in the history but are no longer declared by the client.
	stubToolDescription = "Tool from earlier conversation. Do not call."
)

// normalizeAnthropicMessages repairs tool_use/tool_result pairing and makes
// the message sequence acceptable to Anthropic:
//
//   - tool_result blocks of a user turn are placed before text/image blocks;
//   - a tool_use without a matching result gets a synthetic error result;
//   - an orphan or duplicate tool_result becomes a text block;
//   - an assistant turn with unanswered calls is closed with a synthetic
//     user message, including at the end of the history;
//   - consecutive messages with the same role are merged.
func normalizeAnthropicMessages(msgs []AnthropicMessage) []AnthropicMessage {
	if len(msgs) == 0 {
		return msgs
	}

	var out []AnthropicMessage
	var pending []string
	open := make(map[string]bool)
	answered := make(map[string]bool)

	// closeExchange answers every still-open call with a synthetic result and
	// appends it as a user message. Called when an assistant message starts a
	// new exchange or when the history ends with open calls.
	closeExchange := func() {
		var results []AnthropicContent
		for _, id := range pending {
			if answered[id] {
				continue
			}
			results = append(results, missingToolResult(id))
		}
		if len(results) > 0 {
			out = append(out, AnthropicMessage{Role: "user", Content: results})
		}
		pending = nil
		open = make(map[string]bool)
		answered = make(map[string]bool)
	}

	for i := 0; i < len(msgs); {
		msg := msgs[i]

		switch msg.Role {
		case "assistant":
			closeExchange()
			for _, block := range msg.Content {
				if block.Type == "tool_use" && block.ID != "" {
					pending = append(pending, block.ID)
					open[block.ID] = true
				}
			}
			out = append(out, msg)
			i++

		case "user":
			// A run of consecutive user messages is one turn. Parallel tool
			// results arrive as separate messages and must be answered
			// together before deciding which calls are still unanswered.
			var turn []AnthropicContent
			j := i
			for ; j < len(msgs) && msgs[j].Role == "user"; j++ {
				turn = append(turn, msgs[j].Content...)
			}

			var results, rest []AnthropicContent
			for _, block := range turn {
				if block.Type != "tool_result" {
					rest = append(rest, block)
					continue
				}
				if open[block.ToolUseID] && !answered[block.ToolUseID] {
					answered[block.ToolUseID] = true
					results = append(results, block)
					continue
				}
				// Orphan or duplicate result: keep its text, but not as a
				// tool_result that has no matching tool_use.
				rest = append(rest, toolResultAsText(block))
			}
			for _, id := range pending {
				if answered[id] {
					continue
				}
				results = append(results, missingToolResult(id))
			}
			pending = nil
			open = make(map[string]bool)
			answered = make(map[string]bool)

			content := append(results, rest...)
			if len(content) > 0 {
				out = append(out, AnthropicMessage{Role: "user", Content: content})
			}
			i = j

		default:
			out = append(out, msg)
			i++
		}
	}

	// The history may end with an assistant turn whose calls were never
	// answered (interrupted generation).
	closeExchange()

	return mergeConsecutiveAnthropicMessages(out)
}

// missingToolResult builds the synthetic result for an unanswered tool_use.
func missingToolResult(id string) AnthropicContent {
	return AnthropicContent{
		Type:      "tool_result",
		ToolUseID: id,
		Content:   missingToolResultText,
		IsError:   true,
	}
}

// toolResultAsText converts an orphan or duplicate tool_result into a text
// block so the information is not lost and the history stays valid.
func toolResultAsText(block AnthropicContent) AnthropicContent {
	text := extractTextContent(block.Content)
	if strings.TrimSpace(text) == "" {
		text = orphanToolResultText
	}
	return AnthropicContent{Type: "text", Text: text}
}

// mergeConsecutiveAnthropicMessages appends the content of each message to
// the previous one when both have the same role. Anthropic requires
// alternating user and assistant turns.
func mergeConsecutiveAnthropicMessages(msgs []AnthropicMessage) []AnthropicMessage {
	out := make([]AnthropicMessage, 0, len(msgs))
	for _, msg := range msgs {
		if len(msg.Content) == 0 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].Role == msg.Role {
			merged := make([]AnthropicContent, 0, len(out[n-1].Content)+len(msg.Content))
			merged = append(merged, out[n-1].Content...)
			merged = append(merged, msg.Content...)
			out[n-1].Content = merged
			continue
		}
		out = append(out, msg)
	}
	return out
}

// stubHistoryTools declares a stub Anthropic tool for every distinct tool_use
// name found in the history when the request declares no tools. This keeps
// tool_use blocks in the history valid for Anthropic. tool_choice is set to
// "none" so the model does not call a tool that the client no longer offers.
func stubHistoryTools(req *AnthropicRequest, msgs []AnthropicMessage) {
	if req == nil || len(req.Tools) > 0 {
		return
	}

	seen := make(map[string]bool)
	for _, msg := range msgs {
		for _, block := range msg.Content {
			if block.Type != "tool_use" || block.Name == "" || seen[block.Name] {
				continue
			}
			seen[block.Name] = true
			req.Tools = append(req.Tools, AnthropicTool{
				Name:        block.Name,
				Description: stubToolDescription,
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			})
		}
	}
	if len(req.Tools) == 0 {
		return
	}
	req.ToolChoice = &AnthropicToolChoice{Type: "none"}
}
