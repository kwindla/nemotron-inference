# Interactive Multi-Turn Testing Utility

## Architecture

Two-process design: a **Python CLI** (`tools/interactive_forward/nemotron_interactive_forward.py`) drives a **C++ server** (`tools/interactive_forward/nemotron_interactive_forward_server.cpp`). They communicate over stdin/stdout using a line-oriented JSON protocol.

The Python side handles tokenization, chat template rendering, user interaction, and performance reporting. The C++ side handles model loading, GPU execution, prefix caching, and the full prefill/decode loop.

## Files

| File | Lines | Role |
|---|---|---|
| `tools/interactive_forward/nemotron_interactive_forward.py` | 491 | Python CLI: tokenizer, chat templates, REPL, metrics display |
| `tools/interactive_forward/nemotron_interactive_forward_server.cpp` | 906 | C++ server: model init, prefill, decode, prefix cache, JSON IPC |
| `tools/interactive_forward/CMakeLists.txt` | 7 | Build target, links `nemotron_runtime_api`, C++17 |

## IPC Protocol

Python spawns the C++ binary as a subprocess with piped stdin/stdout. Every exchange is: Python writes one line to stdin, reads one JSON line from stdout.

**Startup**: C++ prints a ready JSON line immediately after model load:
```json
{"ok":true,"status":"ready","model_id":"...","conversation_id":"interactive-0","prefix_cache_enabled":true,"target_context_tokens":8192}
```

**Commands** (Python → C++, tab-delimited):
- `EXIT` — graceful shutdown, server responds with status JSON and exits
- `RESET` — clears prefix cache, increments `conversation_serial`, responds with status JSON
- `SET_PREFIX_CACHE\t<on|off>` — toggles prefix cache at runtime
- `TURN\t<max_new_tokens>\t<eos_csv>\t<prompt_csv>` — execute one conversation turn. EOS and prompt tokens are comma-separated int lists.

**Responses**: Always a single JSON line with `"ok":true|false`. TURN responses include ~40 fields of metrics (see `TurnResponse` struct, cpp:68-109).

## C++ Server Core Logic

### Startup (cpp:768-824)

1. Parse args (`--manifest`, `--max-new-tokens`, `--target-context-tokens`, `--prefix-cache`, `--tenant-namespace`, `--serializer-revision`)
2. Load manifest JSON → `PackedModelManifest`
3. Select model config via `ConfigForManifest()` (maps model_id to a known config)
4. Build `RuntimeEnvironment` from manifest + `RuntimeBootstrapOptions`
5. Create `SingleTokenForwardModel` from environment + config
6. Print ready JSON, enter stdin command loop

### ExecuteTurn (cpp:520-764)

The heart of the server:

1. Create `RequestExecutionContext` from the model
2. Reset all global execution counters (linear, expert staging, attention, mamba, expert)
3. Build `SerializedPromptIdentity` from prompt tokens + namespace/revision metadata
4. **Prefix cache lookup** (cpp:565-605): if enabled, look up the prompt identity. Three outcomes:
   - **Partial match** (`resume`): restore cached state, prefill only the new suffix tokens via `ContinuePrefill()`
   - **Exact match** (`exact_hit`): restore cached boundary logits, skip prefill entirely
   - **Miss** (`cold`): full prefill via `RunPrefill()`
5. **Prefill** (cpp:610-646): allocate prompt logits tensor `[prefill_tokens × vocab_size]`, run prefill, copy logits to host
6. **First token selection** (cpp:648-659): argmax on the last row of prompt logits
7. **Greedy decode loop** (cpp:661-710): for each step up to `max_new_tokens`:
   - Push token_id to generated list
   - Call `ContinueSingleToken(token_id, context, step_logits)`
   - Copy step logits to host
   - Check EOS → if hit, break; otherwise argmax for next token
   - Record per-step latency
8. **Commit to prefix cache** (cpp:731-743): publish `prompt + generated` tokens as a conversation head snapshot, with boundary logits, so the next turn can resume
9. Collect all metrics and counters

### Command Loop (cpp:826-903)

`std::getline(std::cin, line)` → dispatch on command string → JSON response → flush.

## Python CLI Core Logic

### Tokenization (py:158-206)

- Uses HuggingFace `AutoTokenizer` with `apply_chat_template()`
- `build_turn_prompt_token_ids()` is the key function: it renders the full message history through the chat template, then splices `exact_history_token_ids` (the actual committed token sequence from prior turns) with only the new suffix. This ensures the prefix cache can match the committed prefix exactly even if the tokenizer is nondeterministic across re-renderings.

### Conversation State (py:392-431)

- `messages: list[dict]` — the chat history as role/content dicts
- `committed_token_ids: list[int]` — the exact token sequence of all prior turns (prompt + generated). After each turn, this is extended with the new prompt tokens + generated tokens.
- On `/reset`: both are cleared (system prompt re-added to messages if set)

### REPL (py:438-473)

- `input("user> ")` loop
- Commands: `/help`, `/quit`, `/reset`, `/cache on|off`
- Anything else → `run_turn(user_text)`
- `--once <text>` mode: single turn then exit (useful for scripted testing)

### run_turn() (py:397-431)

1. Build `pending_messages = messages + [{"role":"user","content":user_text}]`
2. Call `build_turn_prompt_token_ids()` to get the full token sequence
3. Format `TURN\t{max_new_tokens}\t{eos_csv}\t{prompt_csv}` and `send_command()`
4. Decode response `generated_token_ids` via tokenizer
5. Update `messages` (append user + assistant) and `committed_token_ids` (prompt + generated)
6. Print performance summary

### Performance Display (py:251-319)

Prints timing (tokenize, cache lookup/restore, prefill, decode), token counts, cache state, decode rate, kernel execution counters, and optional per-step latencies.

## Key Design Decisions

1. **Greedy decoding only** — argmax token selection, no sampling. This is a forward-pass correctness/performance testing tool, not a production serving endpoint.

2. **Prefix cache enables efficient multi-turn** — after each turn, the full `prompt+generated` sequence is committed to the cache. The next turn's prompt (which is a strict prefix extension) gets a cache hit, skipping re-prefill of the entire history.

3. **Token-level history tracking** — the Python side tracks `committed_token_ids` (the real token sequence) separately from `messages` (the logical chat history). The `build_turn_prompt_token_ids` function splices these together so the tokenizer only re-renders the new turn's suffix while preserving the exact byte-level prefix for cache matching.

4. **Process isolation** — keeping the GPU model in a separate C++ process means the Python tokenizer/REPL can crash or be restarted without tearing down the model. The tab-delimited protocol is deliberately simple.

5. **Counter reset per turn** — all global execution counters are reset at the start of each `ExecuteTurn`, so the metrics in each response reflect exactly that turn's work.
