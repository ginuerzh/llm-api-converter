package rewriter

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"

	"llm-api-converter/convert"
)

// Options holds the configuration for the rewriter plugin server.
type Options struct {
	Model     string
	MaxTokens int
	ModelMap  string // raw --model-map flag value
}

type rewriteRequest struct {
	Data     []byte `json:"data"`
	Metadata []byte `json:"metadata"`
}

type rewriteResponse struct {
	OK   bool   `json:"ok"`
	Data []byte `json:"data,omitempty"`
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
	rc := convert.NewReasoningCache(1000)
	mux.Handle("/rewrite", &rewriteHandler{opts: opts, reasoningCache: rc, modelMap: parseModelMap(opts.ModelMap)})
	return mux
}

type rewriteHandler struct {
	opts           *Options
	reasoningCache *convert.ReasoningCache
	modelMap       convert.ModelMap
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

	slog.Debug("received request", "data", string(req.Data), "metadata", string(req.Metadata))

	opts := &convert.ConvertOptions{
		Model:          h.opts.Model,
		MaxTokens:      h.opts.MaxTokens,
		ModelMap:       h.modelMap,
		ReasoningCache: h.reasoningCache,
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
		if err := json.Unmarshal(req.Metadata, &meta); err == nil && meta.SSEPhase != "" {
			out, err := convert.HandleSSEEvent(meta.Sid, meta.SSEPhase, meta.EventIndex, req.Data, opts)
			if err != nil {
				slog.Error("stream event", "phase", meta.SSEPhase, "err", err)
				writeJSON(w, rewriteResponse{OK: false})
				return
			}
			slog.Debug("stream response", "phase", meta.SSEPhase, "data", string(out))
			writeJSON(w, rewriteResponse{OK: true, Data: out})
			return
		}
	}

	if len(req.Data) == 0 {
		slog.Debug("empty data in rewrite request (non-SSE) — pass through")
		writeJSON(w, rewriteResponse{OK: true})
		return
	}

	// Non-streaming conversion.
	out, err := convert.Convert(req.Data, opts)
	if err != nil {
		slog.Error("convert", "err", err)
		writeJSON(w, rewriteResponse{OK: false})
		return
	}

	slog.Debug("conversion response", "data", string(out))
	writeJSON(w, rewriteResponse{OK: true, Data: out})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(v)
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

// parseModelMap parses a comma-separated model map string into a ModelMap.
// Format: "prefix1=target1,prefix2=target2,..." — * prefix is catch-all.
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
		prefix, target, ok := strings.Cut(pair, "=")
		if !ok || prefix == "" || target == "" {
			slog.Warn("model-map: skipping malformed entry", "entry", pair)
			continue
		}
		mm = append(mm, convert.ModelMapEntry{
			SourcePrefix: strings.ToLower(strings.TrimSpace(prefix)),
			TargetModel:  strings.TrimSpace(target),
		})
	}
	return mm
}
