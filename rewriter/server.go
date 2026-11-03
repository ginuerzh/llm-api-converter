package rewriter

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"

	"llm-api-converter/convert"
	"llm-api-converter/engine"
)

// Options holds the configuration for the rewriter plugin server.
type Options struct {
	Model                  string
	MaxTokens              int
	ModelMap               string // raw --model-map flag value
	Cache                  string // reasoning cache backend: "memory" (default) or "file:<path>"
	FilterRedactedThinking bool
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
	mux.Handle("/rewrite", &rewriteHandler{
		eng: engine.New(engine.Options{
			Model:                  opts.Model,
			MaxTokens:              opts.MaxTokens,
			ModelMap:               parseModelMap(opts.ModelMap),
			Cache:                  opts.Cache,
			FilterRedactedThinking: opts.FilterRedactedThinking,
		}),
	})
	return mux
}

// rewriteHandler is the thin HTTP adapter around engine.Engine.
type rewriteHandler struct {
	eng *engine.Engine
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

	res, err := h.eng.Rewrite(r.Context(), req.Data, metaFrom(req.Metadata))
	if err != nil {
		slog.Error("rewrite", "err", err)
		writeJSON(w, rewriteResponse{OK: false})
		return
	}
	writeJSON(w, rewriteResponse{OK: true, Data: res.Data})
}

func metaFrom(b []byte) engine.Meta {
	var m struct {
		SID        string `json:"sid,omitempty"`
		EventIndex int    `json:"event_index,omitempty"`
		SSEPhase   string `json:"sse_phase,omitempty"`
		Direction  string `json:"direction,omitempty"`
		URI        string `json:"uri,omitempty"`
	}
	if len(b) > 0 {
		_ = json.Unmarshal(b, &m)
	}
	return engine.Meta{
		SID:        m.SID,
		Direction:  m.Direction,
		URI:        m.URI,
		SSEPhase:   m.SSEPhase,
		EventIndex: m.EventIndex,
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(v)
}

// parseModelMap keeps the historical helper used by tests.
func parseModelMap(s string) convert.ModelMap {
	return engine.ParseModelMap(s)
}
