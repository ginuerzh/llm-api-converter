package rewriter

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync/atomic"

	"llm-api-converter/convert"
)

var reqIDSeq atomic.Int64

// Options holds the configuration for the rewriter plugin server.
type Options struct {
	Model     string
	MaxTokens int
	ModelMap  string // raw --model-map flag value
	Cache     string // reasoning cache backend: "memory" (default) or "file:<path>"
}

type rewriteRequest struct {
	Data     []byte `json:"data"`
	Metadata []byte `json:"metadata"`
}

type rewriteResponse struct {
	OK   bool   `json:"ok"`
	Data []byte `json:"data"`
}

// ListenAndServe starts the HTTP server on the given address.
func ListenAndServe(addr string, opts *Options) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		slog.Error(fmt.Sprintf("server listen: %v", err))
		return err
	}
	slog.Info(fmt.Sprintf("server listening on %v", ln.Addr()))

	return http.Serve(ln, newServer(opts))
}

func newServer(opts *Options) http.Handler {
	mux := http.NewServeMux()
	rc := newReasoningCache(opts.Cache, 1000)
	mux.Handle("/rewrite", &rewriteHandler{
		opts:           opts,
		reasoningCache: rc,
		modelMap:       parseModelMap(opts.ModelMap),
		sessionStore:   convert.NewSessionStore(),
	})
	return mux
}

type rewriteHandler struct {
	opts           *Options
	reasoningCache *convert.ReasoningCache
	modelMap       convert.ModelMap
	sessionStore   *convert.SessionStore
}

func (h *rewriteHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		slog.Error("read body", "err", err)
		writeJSON(w, rewriteResponse{OK: false})
		return
	}
	defer r.Body.Close()

	var req rewriteRequest
	if err := json.Unmarshal(body, &req); err != nil {
		slog.Warn("unmarshal request", "err", err)
		writeJSON(w, rewriteResponse{OK: false})
		return
	}

	reqID := reqIDSeq.Add(1)

	opts := &convert.ConvertOptions{
		Model:          h.opts.Model,
		MaxTokens:      h.opts.MaxTokens,
		ModelMap:       h.modelMap,
		ReasoningCache: h.reasoningCache,
		SessionStore:   h.sessionStore,
	}

	// Resolve original model name from model map (safety classifier needs it).
	opts.RequestModel = resolveRequestModel(req.Data, h.modelMap)

	// Extract declared tool names from Anthropic requests for tool restriction.
	if names := extractAnthropicToolNames(req.Data); len(names) > 0 {
		opts.DeclaredTools = names
	}

	// Parse metadata: SSE lifecycle phases, URI, direction.
	var meta struct {
		SID        string `json:"sid,omitempty"`
		EventIndex int    `json:"event_index,omitempty"`
		SSEPhase   string `json:"sse_phase,omitempty"`
		Direction  string `json:"direction,omitempty"`
		URI        string `json:"uri,omitempty"`
	}
	if len(req.Metadata) > 0 {
		if err := json.Unmarshal(req.Metadata, &meta); err == nil {
			opts.SID = meta.SID
			opts.EventIndex = meta.EventIndex
			opts.SSEPhase = meta.SSEPhase
			opts.Direction = meta.Direction
			opts.URI = meta.URI
		}
	}

	// SSE lifecycle events (start/event/end phases).
	if meta.SSEPhase != "" {
		out, err := convert.HandleSSEEvent(meta.SID, meta.SSEPhase, meta.EventIndex, req.Data, opts)
		if err != nil {
			slog.Error("stream event", "req_id", reqID, "phase", meta.SSEPhase, "err", err)
			writeJSON(w, rewriteResponse{OK: false})
			return
		}
		slog.Debug("stream response", "req_id", reqID, "phase", meta.SSEPhase,
			"received", truncate(req.Data), "data", string(out))
		writeJSON(w, rewriteResponse{OK: true, Data: out})
		return
	}

	if len(req.Data) == 0 {
		slog.Debug("empty data in rewrite request (non-SSE) — pass through", "req_id", reqID)
		writeJSON(w, rewriteResponse{OK: true})
		return
	}

	// Session-aware dispatch: when an active session's SSE event arrives
	// without SSEPhase metadata, route through HandleSSEEvent.
	if opts.SID != "" {
		if sess := h.sessionStore.Get(opts.SID); sess != nil && sess.StreamHandler != nil {
			if isSSEFramed(req.Data) {
				out, err := convert.HandleSSEEvent(opts.SID, string(convert.StreamPhaseEvent), 0, req.Data, opts)
				if err != nil {
					slog.Warn("session event", "req_id", reqID, "err", err)
				}
				if out != nil {
					slog.Debug("session event output", "req_id", reqID,
						"received", truncate(req.Data), "data", string(out))
					writeJSON(w, rewriteResponse{OK: true, Data: out})
					return
				}
				writeJSON(w, rewriteResponse{OK: true, Data: []byte{}})
				return
			}
			slog.Debug("active session non-SSE data → falling through to Convert()", "req_id", reqID)
		}
	}

	// Non-streaming conversion.
	out, err := convert.Convert(req.Data, opts)
	if err != nil {
		slog.Error("convert", "req_id", reqID, "err", err)
		writeJSON(w, rewriteResponse{OK: false})
		return
	}

	slog.Debug("conversion response", "req_id", reqID, "received", truncate(req.Data), "data", string(out))
	writeJSON(w, rewriteResponse{OK: true, Data: out})
}

// resolveRequestModel returns the original client-facing model name from the
// request body, for mapping into the safety classifier on the response path.
func resolveRequestModel(data []byte, mm convert.ModelMap) string {
	if model := extractModelFromPayload(data); model != "" {
		return convert.StripProviderPrefix(model)
	}
	return ""
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(v)
}

func truncate(b []byte) string {
	if len(b) > 1024*1024 {
		return string(b[:200]) + "..."
	}
	return string(b)
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

func parseModelMap(s string) convert.ModelMap {
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
		if protocol != "" && protocol != "openai" && protocol != "anthropic" {
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
