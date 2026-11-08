package engine

import (
	"encoding/json"
	"strings"
)

// Effort thresholds for Anthropic thinking budgets and Gemini thinking budgets.
const (
	lowThreshold    = 1024
	mediumThreshold = 4096
	highThreshold   = 16384
)

type requestProbe struct {
	Model string `json:"model"`

	ReasoningEffort string `json:"reasoning_effort"`
	Reasoning       struct {
		Effort string `json:"effort"`
	} `json:"reasoning"`
	Thinking struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	} `json:"thinking"`
	// Anthropic's newer clients send thinking.type="adaptive" with the band in
	// output_config.effort instead of a thinking budget.
	OutputConfig struct {
		Effort string `json:"effort"`
	} `json:"output_config"`
	GenerationConfig struct {
		ThinkingConfig struct {
			ThinkingLevel  string `json:"thinkingLevel"`
			ThinkingBudget int    `json:"thinkingBudget"`
		} `json:"thinkingConfig"`
	} `json:"generationConfig"`
}

// inspectRequest extracts the client model and the normalized effort band.
// Effort is one of none, low, medium, high.
func inspectRequest(data []byte) (model, effort string) {
	if len(data) == 0 {
		return "", "none"
	}
	var p requestProbe
	if err := json.Unmarshal(data, &p); err != nil {
		return "", "none"
	}
	return p.Model, normalizeEffort(p)
}

func normalizeEffort(p requestProbe) string {
	if v := strings.ToLower(strings.TrimSpace(p.ReasoningEffort)); v != "" {
		return effortFromString(v)
	}
	if v := strings.ToLower(strings.TrimSpace(p.Reasoning.Effort)); v != "" {
		return effortFromString(v)
	}
	if v := strings.ToLower(strings.TrimSpace(p.OutputConfig.Effort)); v != "" {
		return effortFromString(v)
	}
	if strings.EqualFold(p.Thinking.Type, "enabled") && p.Thinking.BudgetTokens > 0 {
		return effortFromBudget(p.Thinking.BudgetTokens)
	}
	if v := strings.ToLower(strings.TrimSpace(p.GenerationConfig.ThinkingConfig.ThinkingLevel)); v != "" {
		return effortFromString(v)
	}
	if b := p.GenerationConfig.ThinkingConfig.ThinkingBudget; b > 0 {
		return effortFromBudget(b)
	}
	return "none"
}

func effortFromString(v string) string {
	switch v {
	case "minimal", "low":
		return "low"
	case "medium":
		return "medium"
	case "high", "xhigh", "max":
		return "high"
	}
	return "none"
}

func effortFromBudget(b int) string {
	switch {
	case b >= highThreshold:
		return "high"
	case b >= mediumThreshold:
		return "medium"
	case b >= lowThreshold:
		return "low"
	}
	return "none"
}
