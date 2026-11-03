package engine

import (
	"bytes"
	"encoding/json"
)

type usageProbe struct {
	Usage struct {
		InputTokens          int `json:"input_tokens"`
		OutputTokens         int `json:"output_tokens"`
		PromptTokens         int `json:"prompt_tokens"`
		CompletionTokens     int `json:"completion_tokens"`
		CacheReadInputTokens int `json:"cache_read_input_tokens"`
		PromptTokensDetails  struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
	} `json:"usageMetadata"`
}

// extractUsage reads token counts from a JSON body or from the data: lines of
// an SSE chunk, taking the maximum seen for each field.
func extractUsage(data []byte) (input, output, cached int) {
	for _, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		payload := bytes.TrimSpace(line[len("data:"):])
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		i, o, c := usageFromJSON(payload)
		input = max(input, i)
		output = max(output, o)
		cached = max(cached, c)
	}
	if input == 0 && output == 0 && cached == 0 {
		return usageFromJSON(data)
	}
	return input, output, cached
}

func usageFromJSON(b []byte) (input, output, cached int) {
	var p usageProbe
	if err := json.Unmarshal(b, &p); err != nil {
		return 0, 0, 0
	}
	input = max(p.Usage.InputTokens, p.Usage.PromptTokens)
	input = max(input, p.UsageMetadata.PromptTokenCount)
	output = max(p.Usage.OutputTokens, p.Usage.CompletionTokens)
	output = max(output, p.UsageMetadata.CandidatesTokenCount)
	cached = max(p.Usage.CacheReadInputTokens, p.Usage.PromptTokensDetails.CachedTokens)
	cached = max(cached, p.UsageMetadata.CachedContentTokenCount)
	return input, output, cached
}
