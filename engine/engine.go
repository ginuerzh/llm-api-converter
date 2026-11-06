// Package engine wraps the llm-api-converter conversion pipeline so it can
// be used in-process (llm-gw) or behind the rewriter HTTP plugin.
package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"

	"llm-api-converter/convert"
)

// Options configures an Engine.
type Options struct {
	Model                  string
	MaxTokens              int
	ModelMap               convert.ModelMap
	Cache                  string // "memory" (default) or "file:<path>"
	FilterRedactedThinking bool

	// ResolveModel maps the client-facing model (and normalized effort) to the
	// provider model and wire. Returning an empty target means "no mapping".
	ResolveModel func(clientModel, effort string) (target, protocol, logical string)
}

// Meta carries the per-call metadata supplied by the proxy.
type Meta struct {
	SID         string
	Direction   string // "request" or "response"
	URI         string
	SSEPhase    string
	EventIndex  int
	StreamError string // upstream stream error reported by the proxy
}

// Result is the converted payload plus the model/effort facts observed on the
// request path.
type Result struct {
	Data         []byte
	ClientModel  string
	TargetModel  string
	LogicalModel string
	Effort       string
	Direction    string

	InputTokens  int
	OutputTokens int
	CachedTokens int
	Stream       bool
}

// Engine holds the session store and reasoning cache shared by all calls.
type Engine struct {
	opts           Options
	reasoningCache *convert.ReasoningCache
	sessionStore   *convert.SessionStore
}

// New creates an Engine.
func New(opts Options) *Engine {
	if opts.Model == "" {
		opts.Model = "deepseek-chat"
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 8192
	}
	return &Engine{
		opts:           opts,
		reasoningCache: newReasoningCache(opts.Cache, 1000),
		sessionStore:   convert.NewSessionStore(),
	}
}

// Rewrite converts one body chunk. Non-chat paths are returned unchanged.
func (e *Engine) Rewrite(ctx context.Context, data []byte, meta Meta) (Result, error) {
	_ = ctx

	res := Result{Direction: meta.Direction}
	if meta.Direction == "response" {
		res.Stream = isSSEFramed(data) || meta.SSEPhase != ""
		res.InputTokens, res.OutputTokens, res.CachedTokens = extractUsage(data)
	}
	// An empty URI means the caller (the HTTP plugin) did not supply routing
	// metadata; keep the historical behaviour and convert.
	if meta.URI != "" && !chatPath(meta.URI) {
		res.Data = data
		return res, nil
	}

	if meta.Direction == "request" {
		res.ClientModel, res.Effort = inspectRequest(data)
	}

	opts := &convert.ConvertOptions{
		Model:                  e.opts.Model,
		MaxTokens:              e.opts.MaxTokens,
		ModelMap:               e.opts.ModelMap,
		ReasoningCache:         e.reasoningCache,
		SessionStore:           e.sessionStore,
		FilterRedactedThinking: e.opts.FilterRedactedThinking,
		SID:                    meta.SID,
		EventIndex:             meta.EventIndex,
		SSEPhase:               meta.SSEPhase,
		Direction:              meta.Direction,
		URI:                    meta.URI,
		ErrorMsg:               meta.StreamError,
	}

	// Resolve the original model before mapping for the safety classifier.
	opts.RequestModel = resolveRequestModel(data, e.opts.ModelMap)

	// Extract declared tool names from Anthropic requests for tool restriction.
	if names := extractAnthropicToolNames(data); len(names) > 0 {
		opts.DeclaredTools = names
	}

	if meta.Direction == "request" && e.opts.ResolveModel != nil && res.ClientModel != "" {
		target, protocol, logical := e.opts.ResolveModel(res.ClientModel, res.Effort)
		res.LogicalModel = logical
		if target != "" {
			res.TargetModel = target
			if !strings.EqualFold(target, res.ClientModel) {
				mm := make(convert.ModelMap, 0, len(opts.ModelMap)+1)
				mm = append(mm, convert.ModelMapEntry{
					SourcePrefix: strings.ToLower(res.ClientModel),
					TargetModel:  target,
					Protocol:     protocol,
				})
				opts.ModelMap = append(mm, opts.ModelMap...)
			}
		}
	}

	// SSE lifecycle events (start/event/end phases).
	if meta.SSEPhase != "" {
		out, err := convert.HandleSSEEvent(meta.SID, meta.SSEPhase, meta.EventIndex, data, opts)
		res.Data = out
		return res, err
	}

	if len(data) == 0 {
		return res, nil
	}

	// Session-aware dispatch: when an active session's SSE event arrives
	// without SSEPhase metadata, route through HandleSSEEvent.
	if opts.SID != "" {
		if sess := e.sessionStore.Get(opts.SID); sess != nil && sess.StreamHandler != nil {
			if isSSEFramed(data) {
				out, err := convert.HandleSSEEvent(opts.SID, string(convert.StreamPhaseEvent), 0, data, opts)
				if err != nil {
					slog.Warn("session event", "sid", opts.SID, "err", err)
				}
				if out != nil {
					res.Data = out
					return res, nil
				}
				res.Data = []byte{}
				return res, nil
			}
		}
	}

	out, err := convert.Convert(data, opts)
	if err != nil {
		return res, err
	}

	// The passthrough branch keeps the client model; replace it so the
	// provider never sees a model name it does not serve.
	if meta.Direction == "request" && res.TargetModel != "" {
		out = replaceModel(out, res.TargetModel)
	}

	res.Data = out
	return res, nil
}

// ParseModelMap parses the --model-map flag syntax.
func ParseModelMap(s string) convert.ModelMap {
	if s == "" {
		return nil
	}
	var mm convert.ModelMap
	for _, pair := range strings.Split(s, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		prefix, rest, ok := strings.Cut(pair, "=")
		if !ok || prefix == "" || rest == "" {
			slog.Warn("model-map: skipping malformed entry", "entry", pair)
			continue
		}
		target, protocol, _ := strings.Cut(rest, ":")
		target = strings.TrimSpace(target)
		protocol = strings.ToLower(strings.TrimSpace(protocol))
		if target == "" {
			slog.Warn("model-map: skipping entry with empty target", "entry", pair)
			continue
		}
		if protocol != "" && protocol != "openai" && protocol != "anthropic" && protocol != "gemini" {
			slog.Warn("model-map: unknown protocol, ignoring", "protocol", protocol, "entry", pair)
			protocol = ""
		}
		mm = append(mm, convert.ModelMapEntry{
			SourcePrefix: strings.ToLower(strings.TrimSpace(prefix)),
			TargetModel:  target,
			Protocol:     protocol,
		})
	}
	return mm
}

func chatPath(uri string) bool {
	path, _, _ := strings.Cut(uri, "?")
	switch path {
	case "/v1/messages", "/v1/chat/completions", "/v1/responses":
		return true
	}
	return false
}

func replaceModel(body []byte, model string) []byte {
	if len(body) == 0 || model == "" {
		return body
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return body
	}
	if _, ok := raw["model"]; !ok {
		return body
	}
	v, err := json.Marshal(model)
	if err != nil {
		return body
	}
	raw["model"] = v
	out, err := json.Marshal(raw)
	if err != nil {
		return body
	}
	return out
}

// resolveRequestModel returns the original client-facing model name from the
// request body, for mapping into the safety classifier on the response path.
func resolveRequestModel(data []byte, mm convert.ModelMap) string {
	if model := extractModelFromPayload(data); model != "" {
		return convert.StripProviderPrefix(model)
	}
	return ""
}

func isSSEFramed(data []byte) bool {
	return bytes.HasPrefix(data, []byte("data:")) ||
		bytes.HasPrefix(data, []byte("event:")) ||
		bytes.HasPrefix(data, []byte("id:"))
}

func extractModelFromPayload(data []byte) string {
	var probe struct {
		Model     string          `json:"model"`
		MaxTokens int             `json:"max_tokens"`
		Messages  json.RawMessage `json:"messages"`
		Choices   json.RawMessage `json:"choices"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return ""
	}
	if probe.Model != "" && probe.MaxTokens > 0 &&
		len(probe.Messages) > 0 && len(probe.Choices) == 0 {
		return probe.Model
	}
	var respProbe struct {
		Model string          `json:"model"`
		Input json.RawMessage `json:"input"`
	}
	if json.Unmarshal(data, &respProbe) == nil &&
		respProbe.Model != "" && len(respProbe.Input) > 0 {
		return respProbe.Model
	}
	return ""
}

func extractAnthropicToolNames(data []byte) []string {
	var probe struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(data, &probe); err != nil || len(probe.Tools) == 0 {
		return nil
	}
	names := make([]string, 0, len(probe.Tools))
	for _, t := range probe.Tools {
		if t.Name != "" {
			names = append(names, t.Name)
		}
	}
	return names
}

func newReasoningCache(spec string, maxSize int) *convert.ReasoningCache {
	typ, option, _ := strings.Cut(spec, ":")
	switch strings.ToLower(strings.TrimSpace(typ)) {
	case "memory", "":
		return convert.NewReasoningCache(maxSize)
	case "file":
		if option == "" {
			slog.Warn("cache: file backend requires a path, falling back to memory")
			return convert.NewReasoningCache(maxSize)
		}
		return convert.NewReasoningCacheWithFile(option, maxSize)
	default:
		slog.Warn("cache: unknown backend, falling back to memory", "type", typ)
		return convert.NewReasoningCache(maxSize)
	}
}
