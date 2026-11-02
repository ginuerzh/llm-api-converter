package convert

// The three APIs count the same cache reads and writes in different places.
//
// OpenAI's prompt_tokens includes everything the prompt was made of, cache
// reads and writes included, and breaks the cached part out under
// prompt_tokens_details. Anthropic's input_tokens excludes what was read
// from and written to the cache; those are cache_read_input_tokens and
// cache_creation_input_tokens. Gemini's promptTokenCount includes cached
// content, and cachedContentTokenCount is the part that came from it.
//
// These helpers keep the totals right in every direction, including the
// vendor spellings that relays use for the same numbers (DeepSeek's
// prompt_cache_hit_tokens, Moonshot's cached_tokens, OpenRouter's
// cache_write_tokens).

// openAICacheCounts returns the prompt tokens an OpenAI-shaped usage reports
// as read from and written to the provider's cache.
func openAICacheCounts(u *OpenAIUsage) (read, write int) {
	if u == nil {
		return 0, 0
	}
	if d := u.PromptTokensDetails; d != nil {
		read, write = d.CachedTokens, d.CacheWriteTokens
	}
	if read == 0 {
		for _, n := range []int{u.PromptCacheHitTokens, u.CachedTokens, u.CacheReadInputTokens} {
			if n > 0 {
				read = n
				break
			}
		}
	}
	if write == 0 {
		write = u.CacheCreationInputTokens
	}
	return read, write
}

// anthropicUsageFromOpenAI maps an OpenAI-shaped usage to Anthropic's
// counting, where input_tokens leaves out cache reads and writes.
func anthropicUsageFromOpenAI(u *OpenAIUsage) AnthropicUsage {
	if u == nil {
		return AnthropicUsage{}
	}
	read, write := openAICacheCounts(u)
	input := u.PromptTokens
	if read+write > 0 {
		// OpenAI's prompt includes the cached part; Anthropic's does not.
		// A body that already carries Anthropic's counts in OpenAI's field
		// names is left alone unless the prompt is large enough to hold
		// them (a relay in front of Claude).
		openAIStyle := u.PromptTokensDetails != nil || u.PromptCacheHitTokens > 0 || u.CachedTokens > 0
		if openAIStyle || u.PromptTokens >= read+write {
			input = u.PromptTokens - read - write
		}
	}
	if input < 0 {
		input = 0
	}
	return AnthropicUsage{
		InputTokens:              input,
		OutputTokens:             u.CompletionTokens,
		CacheReadInputTokens:     read,
		CacheCreationInputTokens: write,
	}
}

// openAIUsageFromAnthropic maps Anthropic's counting to OpenAI's, where the
// prompt includes the cached part.
func openAIUsageFromAnthropic(u AnthropicUsage) OpenAIUsage {
	prompt := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	out := OpenAIUsage{
		PromptTokens:     prompt,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      prompt + u.OutputTokens,
	}
	if u.CacheReadInputTokens > 0 || u.CacheCreationInputTokens > 0 {
		out.PromptTokensDetails = &OpenAIPromptTokensDetails{
			CachedTokens:     u.CacheReadInputTokens,
			CacheWriteTokens: u.CacheCreationInputTokens,
		}
	}
	return out
}

// anthropicUsageFromGemini maps Gemini's counting to Anthropic's.
func anthropicUsageFromGemini(g *GeminiUsage) AnthropicUsage {
	if g == nil {
		return AnthropicUsage{}
	}
	input := g.PromptTokenCount - g.CachedContentTokenCount
	if input < 0 {
		input = 0
	}
	return AnthropicUsage{
		InputTokens:          input,
		OutputTokens:         g.CandidatesTokenCount,
		CacheReadInputTokens: g.CachedContentTokenCount,
	}
}

// openAIUsageFromGemini maps Gemini's counting to OpenAI's.
func openAIUsageFromGemini(g *GeminiUsage) OpenAIUsage {
	if g == nil {
		return OpenAIUsage{}
	}
	out := OpenAIUsage{
		PromptTokens:     g.PromptTokenCount,
		CompletionTokens: g.CandidatesTokenCount,
		TotalTokens:      g.TotalTokenCount,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = g.PromptTokenCount + g.CandidatesTokenCount
	}
	if g.CachedContentTokenCount > 0 {
		out.PromptTokensDetails = &OpenAIPromptTokensDetails{CachedTokens: g.CachedContentTokenCount}
	}
	return out
}

// geminiAnthropicUsage is Gemini's usage as the usage object of an Anthropic
// message_delta.
func geminiAnthropicUsage(g *GeminiUsage) map[string]any {
	usage := map[string]any{"output_tokens": 0}
	if g == nil {
		return usage
	}
	u := anthropicUsageFromGemini(g)
	usage["input_tokens"] = u.InputTokens
	usage["output_tokens"] = u.OutputTokens
	if u.CacheReadInputTokens > 0 {
		usage["cache_read_input_tokens"] = u.CacheReadInputTokens
	}
	return usage
}

// geminiUsageFromOpenAI maps OpenAI's counting to Gemini's.
func geminiUsageFromOpenAI(u *OpenAIUsage) *GeminiUsage {
	if u == nil {
		return nil
	}
	read, _ := openAICacheCounts(u)
	return &GeminiUsage{
		PromptTokenCount:        u.PromptTokens,
		CandidatesTokenCount:    u.CompletionTokens,
		TotalTokenCount:         u.TotalTokens,
		CachedContentTokenCount: read,
	}
}

// geminiUsageFromAnthropic maps Anthropic's counting to Gemini's.
func geminiUsageFromAnthropic(u AnthropicUsage) *GeminiUsage {
	prompt := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	return &GeminiUsage{
		PromptTokenCount:        prompt,
		CandidatesTokenCount:    u.OutputTokens,
		TotalTokenCount:         prompt + u.OutputTokens,
		CachedContentTokenCount: u.CacheReadInputTokens,
	}
}
