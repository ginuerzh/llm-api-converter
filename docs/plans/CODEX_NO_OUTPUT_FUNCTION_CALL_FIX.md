# Fix: Codex "no output, exits" — function_call `output_item.done` missing `arguments`

## Context

Codex exits with no output when the converter proxies a Responses-API turn whose model output is a **tool call**. `llm-api-converter/debug.log` (req_id 2–72) shows a complete-looking SSE stream — reasoning deltas, argument deltas, and a final `response.completed` at phase=end (line 76, seq 75) — yet codex displays nothing and exits. The reasoning summary displays (those events are handled), but the actual tool call never runs.

## Root cause (evidence chain)

1. **Codex ignores argument-streaming events.** `codex/codex-rs/codex-api/src/sse/responses.rs:315-326` (`process_responses_event`) has **no arm** for `response.function_call_arguments.delta` / `.done` — the only reference is a test (line 865). Codex relies **entirely** on `response.output_item.done` to deliver the final `function_call` item. (Confirmed: `codex-rs/core/tests/common/responses.rs:844` `ev_function_call` puts the full `arguments` inside the `output_item.done` item and emits no argument-delta events.)

2. **Codex silently drops unparseable items.** For `response.output_item.done` it runs `serde_json::from_value::<ResponseItem>(item_val)`; on failure it logs `debug!("failed to parse ResponseItem from output_item.done")` and returns `Ok(None)` — no error, no item delivered. Same for `.added` (line 428-433).

3. **`arguments` is required.** `codex/codex-rs/protocol/src/models.rs:996-1012` defines `ResponseItem::FunctionCall { … arguments: String, … }` with **no `#[serde(default)]`**. A `function_call` item missing `arguments` fails to deserialize.

4. **The converter omits it.** `llm-api-converter/convert/stream_responses.go:239` emits the function-call `output_item.done` with only `ItemID` + `OutputIndex` — **no `Item` at all** on current HEAD (the debug-log binary had a synthesized item that still omitted `arguments`). Either way codex receives no deserializable `function_call` → the tool call is silently dropped → no tool executes → `response.completed` ends the turn → codex shows nothing and exits.

OpenAI's streaming spec confirms `response.output_item.done` is the carrier of the final, complete function-call item including `arguments`.

## Fix (single root-cause change)

`convert/stream_responses.go`, `HandleStreamEnd` function-call loop (~line 234-241): replace the itemless `output_item.done` with one carrying a complete `function_call` item, populating `Arguments` from the accumulated `fc.Arguments`. Mirror what `completedFunctionCallItem` (`responses.go:941-951`) already does for the non-streaming path.

```go
for _, fc := range sc.toolCallByIndex {
    if fc.Arguments != "" {
        evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseFunctionCallArgumentsDone, ItemID: fc.ID, OutputIndex: &fc.ItemIx, Arguments: canonicalJSONString(fc.Arguments)})
        events = append(events, sc.makeEvent(ResponseFunctionCallArgumentsDone, string(evt)))
    }
    item := ResponsesOutputItem{
        ID:        fc.ID,
        Type:      "function_call",
        Status:    "completed",
        CallID:    fc.ID,
        Name:      fc.Name,
        Arguments: canonicalJSONString(fc.Arguments),
    }
    evt, _ := json.Marshal(ResponsesStreamEvent{Type: ResponseOutputItemDone, OutputIndex: &fc.ItemIx, Item: &item})
    events = append(events, sc.makeEvent(ResponseOutputItemDone, string(evt)))
}
```

`fc.Name` is already captured in `handleToolCall` (line 334). `ResponsesOutputItem` has all needed fields (`responses.go:47-57`).

### Known ceiling (do not fix now)
`ResponsesOutputItem.Arguments` is `omitempty` (`responses.go:56`), so a tool call with genuinely-empty args would still omit the field and be dropped. Real codex tool calls (`exec_command`, `apply_patch`) always carry args, so this is out of scope. `// ponytail: omitempty drops empty-args function_call; codex requires the field. Real codex tools always carry args.`

### Why not touch `output_item.added` (line 335-340)
It also omits `arguments` and is silently dropped by codex, but `.added` is **not load-bearing** for codex tool execution (`.done` is — `handle_output_item_done` drives `ToolRouter::build_tool_call`). Restructuring `omitempty` for `.added` isn't justified. Leave it.

## Verification

1. **Unit test** — add to `convert/stream_responses_test.go`: feed a tool-call delta sequence through `ResponsesStreamConverter`; assert the emitted `response.output_item.done`'s `item` is a `function_call` whose `arguments` equals the concatenated argument deltas (non-empty, valid JSON).
2. `cd llm-api-converter && go build ./... && go vet ./... && go test ./... -count=1 -race`
3. **E2E with codex** (per CLAUDE.md): run the plugin on `:8000`, point codex at it, issue a prompt that triggers a tool call (e.g. "check if README.md needs improvement"), confirm the tool executes and codex produces output instead of exiting blank.
4. **Secondary check**: the parallel `custom_tool_call` path (`toolTypeFromSpec`, line 256-258 → uses `input` not `arguments`) — verify `ResponseItem::CustomToolCall`'s required `input` field is present in its `.done`; file follow-up if broken.
