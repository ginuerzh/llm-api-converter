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
	"sync"
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
		return err
	}
	slog.Info(fmt.Sprintf("server listening on %v", ln.Addr()))

	return http.Serve(ln, newServer(opts))
}

func newServer(opts *Options) http.Handler {
	mux := http.NewServeMux()
	rc := newReasoningCache(opts.Cache, 1000)
	mux.Handle("/rewrite", &rewriteHandler{opts: opts, reasoningCache: rc, modelMap: parseModelMap(opts.ModelMap)})
	return mux
}

type rewriteHandler struct {
	opts           *Options
	reasoningCache *convert.ReasoningCache
	modelMap       convert.ModelMap
	requestModels  sync.Map // targetModel (string) → originalModel (string)
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
	}

	// Cache the original model keyed by target so response/SSE phases can
	// recover it. The safety classifier needs the model the client asked for.
	if originalModel := extractModelFromPayload(req.Data); originalModel != "" {
		bare := convert.StripProviderPrefix(originalModel)
		if target, _, ok := h.modelMap.Apply(bare); ok && target != "" {
			h.requestModels.Store(target, bare)
		}
	}

	// Probe the model from data (handles both raw JSON and SSE "data:" prefix).
	if dataModel := extractModelField(req.Data); dataModel != "" {
		if cached, ok := h.requestModels.Load(dataModel); ok {
			opts.RequestModel = cached.(string)
		}
	}

	// Extract declared tool names from Anthropic requests for tool restriction.
	if names := extractAnthropicToolNames(req.Data); len(names) > 0 {
		opts.DeclaredTools = names
	}

	// SSE lifecycle phases are metadata-only signals with nil body.
	// Check BEFORE the empty-data guard so start/end phases reach HandleSSEEvent.
	if len(req.Metadata) > 0 {
		var meta struct {
			Sid        string `json:"sid,omitempty"`
			EventIndex int    `json:"event_index,omitempty"`
			SSEPhase   string `json:"sse_phase,omitempty"`
		}
		if err := json.Unmarshal(req.Metadata, &meta); err == nil {
			opts.SID = meta.Sid
			opts.EventIndex = meta.EventIndex
			opts.SSEPhase = meta.SSEPhase
		}
		if meta.SSEPhase != "" {
			out, err := convert.HandleSSEEvent(meta.Sid, meta.SSEPhase, meta.EventIndex, req.Data, opts)
			if err != nil {
				slog.Error("stream event", "req_id", reqID, "phase", meta.SSEPhase, "err", err)
				writeJSON(w, rewriteResponse{OK: false})
				return
			}
			slog.Debug("stream response", "req_id", reqID, "phase", meta.SSEPhase, "received", truncate(req.Data), "data", string(out))
			writeJSON(w, rewriteResponse{OK: true, Data: out})
			return
		}
	}

	if len(req.Data) == 0 {
		slog.Debug("empty data in rewrite request (non-SSE) — pass through", "req_id", reqID)
		writeJSON(w, rewriteResponse{OK: true})
		return
	}

	// Session-aware dispatch: when a Responses API session's SSE event
	// arrives without SSEPhase metadata, route streaming chunks through
	// HandleSSEEvent. Non-streaming (complete JSON) responses fall through
	// to Convert() which handles Responses→Chat round-trip conversion.
	if opts.SID != "" && convert.IsResponsesSession(opts.SID) {
		if isSSEFramed(req.Data) {
			out, err := convert.HandleSSEEvent(opts.SID, string(convert.StreamPhaseEvent), 0, req.Data, opts)
			if err != nil {
				slog.Warn("responses session event", "req_id", reqID, "err", err)
			}
			if out != nil {
				slog.Debug("responses session event output", "req_id", reqID, "received", truncate(req.Data), "data", string(out))
				writeJSON(w, rewriteResponse{OK: true, Data: out})
				return
			}
			// Swallow: consumed by stream handler with no output to emit.
			writeJSON(w, rewriteResponse{OK: true, Data: []byte{}})
			return
		}
		slog.Debug("responses session non-SSE data → falling through to Convert()", "req_id", reqID)
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

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(v)
}

// truncate shortens long byte slices for debug logging.
func truncate(b []byte) string {
	if len(b) > 1024*1024 {
		return string(b[:200]) + "..."
	}
	return string(b)
}

// isSSEFramed returns true if data starts with an SSE prefix (data:, event:, id:).
func isSSEFramed(data []byte) bool {
	return bytes.HasPrefix(data, []byte("data:")) || bytes.HasPrefix(data, []byte("event:")) || bytes.HasPrefix(data, []byte("id:"))
}

// extractModelFromPayload returns the model name from an Anthropic request body.
// It differentiates Anthropic requests (model + max_tokens + messages, no choices)
// from OpenAI responses (choices array) to avoid false-positive caching.
func extractModelFromPayload(data []byte) string {
	// Anthropic request: model + max_tokens + messages, no choices.
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
	// Responses API request: model + input, no messages/choices/max_tokens.
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

// extractModelField returns the "model" field from data, handling both raw JSON
// and SSE-framed data ("data: {...}" prefix).
func extractModelField(data []byte) string {
	var probe struct{ Model string `json:"model"` }
	if json.Unmarshal(data, &probe) == nil && probe.Model != "" {
		return probe.Model
	}
	// Strip SSE "data:" prefix for stream chunk data.
	if stripped, ok := strings.CutPrefix(string(data), "data:"); ok {
		payload := []byte(strings.TrimLeft(stripped, " \t"))
		if json.Unmarshal(payload, &probe) == nil {
			return probe.Model
		}
	}
	return ""
}

// extractAnthropicToolNames parses the request body as an Anthropic request and
// returns the declared tool names. Returns nil if parsing fails or no tools.
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

// newReasoningCache creates a ReasoningCache based on the cache spec string.
// Format: "memory" (default) or "file:<path>".
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

// parseModelMap parses a comma-separated model map string into a ModelMap.
// Format: "prefix1=target1[:protocol1],prefix2=target2[:protocol2],..." — * prefix is catch-all.
// Protocol is optional (openai|anthropic); when unset, auto-detect is used.
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
		// Validate protocol values: only "" (unset), "openai", or "anthropic".
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
