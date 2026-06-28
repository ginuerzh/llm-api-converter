// Package e2e_test contains end-to-end integration tests for the GOST rewriter
// plugin chain. It orchestrates three components:
//
//  1. Mock OpenAI server (in-process httptest.Server)
//  2. llm-api-converter plugin (subprocess)
//  3. GOST proxy (subprocess)
//
// The test verifies that an Anthropic API request sent to GOST is correctly
// converted to OpenAI format, forwarded to the mock server, and the response
// is converted back to Anthropic format.
package e2e_test

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"text/template"
	"time"
)

// sseEvent represents a single Server-Sent Event parsed from the response stream.
type sseEvent struct {
	Event string
	Data  string
}

// Configuration flags set by TestMain or test flags.
var (
	gostBin   = "" // path to pre-built GOST binary (set via -gost-bin)
	pluginBin = "" // path to pre-built llm-api-converter binary (set via -plugin-bin)
	gostDir   = "" // path to gost module directory (set via -gost-dir)
	pluginDir = "" // path to llm-api-converter module directory (set via -plugin-dir)
)

const (
	gostPort       = 18080
	pluginPort     = 18000
	startupTimeout = 30 * time.Second
	pollInterval   = 200 * time.Millisecond
)

func TestMain(m *testing.M) {
	// Determine workspace root from the test file location.
	testDir, err := filepath.Abs(".")
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "failed to get test dir: %v\n", err)
		os.Exit(1)
	}
	workspaceRoot, err := findWorkspaceRoot(testDir)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "failed to find workspace root: %v\n", err)
		os.Exit(1)
	}

	if gostBin == "" {
		gostBin = filepath.Join(workspaceRoot, "gost", ".build", "gost-e2e")
	}
	if pluginBin == "" {
		pluginBin = filepath.Join(workspaceRoot, "llm-api-converter", ".build", "llm-api-converter-e2e")
	}
	if gostDir == "" {
		gostDir = filepath.Join(workspaceRoot, "gost")
	}
	if pluginDir == "" {
		pluginDir = filepath.Join(workspaceRoot, "llm-api-converter")
	}

	// Build binaries.
	if err := buildBinary(gostDir, gostBin, "./cmd/gost/..."); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "failed to build gost: %v\n", err)
		os.Exit(1)
	}
	if err := buildBinary(pluginDir, pluginBin, "."); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "failed to build plugin: %v\n", err)
		os.Exit(1)
	}

	os.Exit(m.Run())
}

// findWorkspaceRoot walks up from dir looking for the go.work file.
func findWorkspaceRoot(dir string) (string, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(abs, "go.work")); err == nil {
			return abs, nil
		}
		parent := filepath.Dir(abs)
		if parent == abs {
			return "", errors.New("reached filesystem root without finding go.work")
		}
		abs = parent
	}
}

// buildBinary builds a Go binary and writes it to outputPath.
func buildBinary(moduleDir, outputPath, pkg string) error {
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("mkdir build dir: %w", err)
	}
	cmd := exec.Command("go", "build", "-o", outputPath, pkg)
	cmd.Dir = moduleDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// waitForPort polls a TCP port until it is reachable or the context expires.
func waitForPort(ctx context.Context, addr string) error {
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for %s: %w", addr, ctx.Err())
		default:
		}
		conn, err := net.DialTimeout("tcp", addr, pollInterval)
		if err == nil {
			conn.Close()
			return nil
		}
		time.Sleep(pollInterval)
	}
}

// GOST config template. UpstreamAddr is populated at runtime with the mock
// server's address (e.g., "127.0.0.1:54321").
const gostConfigTemplate = `
services:
- name: anthropic-proxy
  addr: :{{.GostPort}}
  handler:
    type: forward
    metadata:
      sniffing: true
  listener:
    type: tcp
  forwarder:
    nodes:
    - name: upstream
      addr: {{.UpstreamAddr}}
      http:
        rewriteRequestBody:
        - rewriter: llm-converter
          type: application/json
        rewriteResponseBody:
        - rewriter: llm-converter
          type: "*"
rewriters:
- name: llm-converter
  plugin:
    type: http
    addr: http://127.0.0.1:{{.PluginPort}}/rewrite
`

type gostConfigData struct {
	GostPort     int
	PluginPort   int
	UpstreamAddr string
}

// startPlugin starts the llm-api-converter plugin as a subprocess.
func startPlugin(t *testing.T) {
	t.Helper()

	cmd := exec.Command(pluginBin,
		"--addr", fmt.Sprintf(":%d", pluginPort),
		"--model", "claude-sonnet-4-20250514",
		"--max-tokens", "8192",
		"--downstream", "deepseek-chat",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start plugin: %v", err)
	}

	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
	})

	ctx, cancel := context.WithTimeout(context.Background(), startupTimeout)
	defer cancel()
	if err := waitForPort(ctx, fmt.Sprintf("127.0.0.1:%d", pluginPort)); err != nil {
		t.Fatalf("plugin did not start in time: %v", err)
	}
	t.Logf("plugin ready on :%d", pluginPort)
}

// startGost starts the GOST proxy as a subprocess with a generated config
// pointing to the given upstream address.
func startGost(t *testing.T, upstreamAddr string) {
	t.Helper()

	// Render config.
	var cfgBuf bytes.Buffer
	if err := template.Must(template.New("gost").Parse(gostConfigTemplate)).Execute(&cfgBuf, gostConfigData{
		GostPort:     gostPort,
		PluginPort:   pluginPort,
		UpstreamAddr: upstreamAddr,
	}); err != nil {
		t.Fatalf("failed to render gost config: %v", err)
	}

	// Write config to a temp file.
	cfgFile, err := os.CreateTemp("", "gost-e2e-*.yaml")
	if err != nil {
		t.Fatalf("failed to create temp config: %v", err)
	}
	if _, err := cfgFile.Write(cfgBuf.Bytes()); err != nil {
		cfgFile.Close()
		t.Fatalf("failed to write config: %v", err)
	}
	cfgFile.Close()

	ctx, cancel := context.WithTimeout(context.Background(), startupTimeout)
	defer cancel()

	cmd := exec.Command(gostBin, "-C", cfgFile.Name(), "-D")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start gost: %v", err)
	}

	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
		os.Remove(cfgFile.Name())
	})

	if err := waitForPort(ctx, fmt.Sprintf("127.0.0.1:%d", gostPort)); err != nil {
		t.Fatalf("gost did not start in time: %v", err)
	}
	t.Logf("gost ready on :%d", gostPort)
}

// startMockOpenAI creates an httptest.Server that acts as a mock OpenAI
// Chat Completions API server. requestLog captures the raw request bodies
// received by the mock server for test assertions.
func startMockOpenAI(t *testing.T, requestLog *[]string) *httptest.Server {
	t.Helper()

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		r.Body.Close()

		if requestLog != nil {
			*requestLog = append(*requestLog, string(body))
		}

		// Check if the request is OpenAI format (has model field).
		var req struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream,omitempty"`
		}
		json.Unmarshal(body, &req)
		if req.Model == "" {
			// Not OpenAI format — return generic passthrough response.
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"received":true}`))
			return
		}

		// Streaming response.
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(http.StatusOK)
			flusher, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, "streaming not supported", http.StatusInternalServerError)
				return
			}

			// Event 1: role announcement (empty content).
			event1 := fmt.Sprintf(`data: {"id":"chatcmpl-e2e-stream","object":"chat.completion.chunk","created":%d,"model":"%s","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`+"\n\n", time.Now().Unix(), req.Model)
			fmt.Fprint(w, event1)
			flusher.Flush()

			// Event 2: content delta.
			event2 := `data: {"choices":[{"index":0,"delta":{"content":"Hello from mock SSE streaming!"},"finish_reason":null}]}` + "\n\n"
			fmt.Fprint(w, event2)
			flusher.Flush()

			// Event 3: finish with stop reason (no [DONE] marker — EOF triggers stream end).
			event3 := `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
			fmt.Fprint(w, event3)
			flusher.Flush()
			return
		}

		// Return OpenAI Chat Completions response.
		resp := map[string]any{
			"id":      "chatcmpl-e2e-test",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   req.Model,
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Hello! I'm the mock OpenAI server responding to your converted request.",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 15,
				"total_tokens":      25,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server
}

// postToGost sends an HTTP POST to the GOST proxy and returns the response.
func postToGost(t *testing.T, path, body string) *http.Response {
	t.Helper()

	gostURL := fmt.Sprintf("http://127.0.0.1:%d%s", gostPort, path)
	req, err := http.NewRequest(http.MethodPost, gostURL, strings.NewReader(body))
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request to gost failed: %v", err)
	}
	return resp
}

// readBody reads and returns the response body.
func readBody(t *testing.T, resp *http.Response) []byte {
	t.Helper()
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("failed to read response body: %v", err)
	}
	return data
}

// TestRewriterE2E_NonStreaming verifies the full round-trip:
// Anthropic Request → GOST → Plugin (Anthropic→OpenAI) → Mock
// → Plugin (OpenAI→Anthropic) → Anthropic Response.
func TestRewriterE2E_NonStreaming(t *testing.T) {
	mock := startMockOpenAI(t, nil)
	startPlugin(t)
	startGost(t, mock.Listener.Addr().String())

	anthropicReq := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
		]
	}`

	resp := postToGost(t, "/v1/messages", anthropicReq)
	body := readBody(t, resp)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d: %s", resp.StatusCode, string(body))
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "application/json") {
		t.Errorf("expected application/json content-type, got %q", ct)
	}

	// Validate Anthropic response format.
	var anthResp struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason   string         `json:"stop_reason"`
		StopSequence *string        `json:"stop_sequence"`
		Usage        map[string]int `json:"usage"`
	}
	if err := json.Unmarshal(body, &anthResp); err != nil {
		t.Fatalf("failed to parse anthropic response: %v\nbody: %s", err, string(body))
	}

	if anthResp.Type != "message" {
		t.Errorf("expected type=message, got %q", anthResp.Type)
	}
	if anthResp.Role != "assistant" {
		t.Errorf("expected role=assistant, got %q", anthResp.Role)
	}
	if len(anthResp.Content) == 0 {
		t.Fatal("expected non-empty content array")
	}
	if anthResp.Content[0].Type != "text" {
		t.Errorf("expected content[0].type=text, got %q", anthResp.Content[0].Type)
	}
	if anthResp.Content[0].Text == "" {
		t.Error("expected non-empty content[0].text")
	}
	if anthResp.StopReason != "end_turn" {
		t.Errorf("expected stop_reason=end_turn, got %q", anthResp.StopReason)
	}
	if anthResp.Usage == nil || anthResp.Usage["input_tokens"] == 0 {
		t.Errorf("expected usage.input_tokens > 0, got %v", anthResp.Usage)
	}
	if anthResp.Usage == nil || anthResp.Usage["output_tokens"] == 0 {
		t.Errorf("expected usage.output_tokens > 0, got %v", anthResp.Usage)
	}
	if !strings.HasPrefix(anthResp.ID, "msg_") {
		t.Errorf("expected id starting with msg_, got %q", anthResp.ID)
	}

	t.Logf("anthropic response: id=%s type=%s stop_reason=%s",
		anthResp.ID, anthResp.Type, anthResp.StopReason)
}

// TestRewriterE2E_Passthrough verifies that unknown JSON formats pass through
// GOST without modification.
func TestRewriterE2E_Passthrough(t *testing.T) {
	mock := startMockOpenAI(t, nil)
	startPlugin(t)
	startGost(t, mock.Listener.Addr().String())

	unknownReq := `{"foo": "bar", "data": [1, 2, 3]}`

	resp := postToGost(t, "/v1/messages", unknownReq)
	body := readBody(t, resp)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d: %s", resp.StatusCode, string(body))
	}

	// Mock returns {"received":true} for non-OpenAI requests.
	var result struct {
		Received bool `json:"received"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatalf("failed to parse response: %v\nbody: %s", err, string(body))
	}
	if !result.Received {
		t.Errorf("expected received=true from mock, got %+v", result)
	}
}

// TestRewriterE2E_MockServerVerification verifies that the mock OpenAI server
// receives an OpenAI-format request (after Anthropic→OpenAI conversion).
func TestRewriterE2E_MockServerVerification(t *testing.T) {
	var requestLog []string
	mock := startMockOpenAI(t, &requestLog)
	startPlugin(t)
	startGost(t, mock.Listener.Addr().String())

	anthropicReq := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Hello"}]}
		]
	}`

	resp := postToGost(t, "/v1/messages", anthropicReq)
	body := readBody(t, resp)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d: %s", resp.StatusCode, string(body))
	}

	if len(requestLog) == 0 {
		t.Fatal("mock server received no requests")
	}

	// Parse the last request the mock server received.
	var openAIReq struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
		MaxTokens int `json:"max_tokens"`
	}
	lastReq := requestLog[len(requestLog)-1]
	if err := json.Unmarshal([]byte(lastReq), &openAIReq); err != nil {
		t.Fatalf("mock received non-JSON request: %s (%v)", lastReq, err)
	}

	if openAIReq.Model != "deepseek-chat" {
		t.Errorf("expected model=deepseek-chat, got %q", openAIReq.Model)
	}
	if len(openAIReq.Messages) == 0 {
		t.Fatal("expected non-empty messages array")
	}
	if openAIReq.Messages[0].Role != "user" {
		t.Errorf("expected messages[0].role=user, got %q", openAIReq.Messages[0].Role)
	}
	if openAIReq.MaxTokens != 100 {
		t.Errorf("expected max_tokens=100, got %d", openAIReq.MaxTokens)
	}

	t.Logf("mock received OpenAI request: model=%q messages=%d max_tokens=%d",
		openAIReq.Model, len(openAIReq.Messages), openAIReq.MaxTokens)
}

// TestRewriterE2E_Streaming verifies SSE streaming mode through the
// rewriter plugin chain. The test sends an Anthropic request with
// stream:true, and validates the returned Anthropic SSE event sequence.
func TestRewriterE2E_Streaming(t *testing.T) {
	mock := startMockOpenAI(t, nil)
	startPlugin(t)
	startGost(t, mock.Listener.Addr().String())

	anthropicReq := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 100,
		"stream": true,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Hello, SSE streaming!"}]}
		]
	}`

	resp := postToGost(t, "/v1/messages", anthropicReq)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected HTTP 200, got %d: %s", resp.StatusCode, string(body))
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("expected text/event-stream content-type, got %q", ct)
	}

	// Read all SSE events from the response stream.
	scanner := bufio.NewScanner(resp.Body)
	scanner.Split(scanSSEEvents)

	var events []sseEvent
	for scanner.Scan() {
		evt := parseSSEEvent(scanner.Bytes())
			events = append(events, evt)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("SSE scanner error: %v", err)
	}

		// Validate the Anthropic SSE event sequence by walking through events
	// and matching expected types in order (robust against extra delimiter events).
	type eventCheck struct {
		event string
		data  []string
	}
	checks := []eventCheck{
		{event: "message_start", data: []string{`"type":"message_start"`, `"role":"assistant"`, `"msg_`}},
		{event: "ping", data: []string{`"type":"ping"`}},
		{event: "content_block_start", data: []string{`"type":"content_block_start"`, `"content_block":{"type":"text"`}},
		{event: "content_block_delta", data: []string{`"type":"content_block_delta"`, `"type":"text_delta"`, "Hello from mock SSE streaming!"}},
		{event: "content_block_stop", data: []string{`"type":"content_block_stop"`}},
		{event: "message_delta", data: []string{`"type":"message_delta"`, `"stop_reason":"end_turn"`}},
		{event: "message_stop", data: []string{`"type":"message_stop"`}},
	}

	ei := 0
	for _, check := range checks {
		for ei < len(events) {
			if events[ei].Event == check.event {
				break
			}
			t.Logf("  skipping event[%d]: event=%q data=%.100s", ei, events[ei].Event, events[ei].Data)
			ei++
		}
		if ei >= len(events) {
			t.Errorf("expected event %q but not found in %d events", check.event, len(events))
			continue
		}
		for _, substr := range check.data {
			if !strings.Contains(events[ei].Data, substr) {
				t.Errorf("events[%d] (%s): expected %q in data", ei, check.event, substr)
			}
		}
		t.Logf("event[%d]: %s \u2713", ei, check.event)
		ei++
	}

	if ei > 0 {
		t.Logf("streaming test: matched %d expected SSE events out of %d total through GOST rewriter chain",
			len(checks), len(events))
	}
}

// scanSSEEvents is a bufio.SplitFunc that splits SSE data on \n\n delimiters.
func scanSSEEvents(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for i := 0; i < len(data)-1; i++ {
		if data[i] == '\n' && data[i+1] == '\n' {
			return i + 2, data[:i], nil
		}
	}
	if atEOF && len(data) > 0 {
		return len(data), data, bufio.ErrFinalToken
	}
	return 0, nil, nil
}

// parseSSEEvent parses raw SSE event bytes into an sseEvent struct.
func parseSSEEvent(raw []byte) sseEvent {
	evt := sseEvent{}
	for _, line := range bytes.Split(raw, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		key, value, found := bytes.Cut(line, []byte(":"))
		if !found {
			continue
		}
		// Strip one leading space per SSE convention ("field: value").
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}
		switch string(key) {
		case "event":
			evt.Event = string(value)
		case "data":
			if evt.Data != "" {
				evt.Data += "\n"
			}
			evt.Data += string(value)
		}
	}
	return evt
}

// extractEventField extracts a JSON string field value from the data: payload.
// This is a simple helper for test assertions that doesn't require full JSON parsing.
func extractEventField(data, field string) string {
	needle := fmt.Sprintf(`"%s":"`, field)
	idx := strings.Index(data, needle)
	if idx < 0 {
		return ""
	}
	start := idx + len(needle)
	end := strings.IndexByte(data[start:], '"')
	if end < 0 {
		return ""
	}
	return data[start : start+end]
}
