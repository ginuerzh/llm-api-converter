package cmd

import (
	"log/slog"
	"os"
	"strings"

	"llm-api-converter/rewriter"

	"github.com/spf13/cobra"
)

var (
	addr         string
	model        string
	maxTokens    int
	modelMap     string
	logLevel     string
	logFormat    string
	cacheBackend string
)

var rootCmd = &cobra.Command{
	Use:   "llm-api-converter",
	Short: "LLM API protocol converter — GOST Rewriter plugin",
	Long:  "Converts between OpenAI Chat Completions and Anthropic Messages API formats.\nRuns as a GOST rewriter HTTP plugin on POST /rewrite.",
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		llevel := slog.LevelInfo
		switch strings.ToLower(logLevel) {
		case "debug":
			llevel = slog.LevelDebug
		case "info":
			llevel = slog.LevelInfo
		case "warn":
			llevel = slog.LevelWarn
		case "error":
			llevel = slog.LevelError
		}

		if logFormat == "json" {
			slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
				Level: llevel,
			})))
		} else {
			slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
				Level: llevel,
			})))
		}
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		return rewriter.ListenAndServe(addr, &rewriter.Options{
			Model:      model,
			MaxTokens:  maxTokens,
			Cache:     cacheBackend,
		})
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.PersistentFlags().StringVar(&logLevel, "log.level", "info", "log level: debug, info, warn or error")
	rootCmd.PersistentFlags().StringVar(&logFormat, "log.format", "json", "log format: text or json")
	rootCmd.PersistentFlags().StringVar(&addr, "addr", ":8000", "listening address")
	rootCmd.PersistentFlags().StringVar(&model, "model", "deepseek-chat", "default fallback model ID")
	rootCmd.PersistentFlags().IntVar(&maxTokens, "max-tokens", 8192, "default max_tokens")
	rootCmd.PersistentFlags().StringVar(&modelMap, "model-map", "", "model mapping: prefix=target[:protocol],... (* for catch-all, protocol: openai|anthropic)")
	rootCmd.PersistentFlags().StringVar(&cacheBackend, "cache", "memory", "reasoning cache backend: memory (default) or file:<path>")

	rootCmd.SilenceErrors = true
	rootCmd.SilenceUsage = true
}
