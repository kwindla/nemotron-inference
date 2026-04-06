# Interactive Forward Tool

## Scope

This directory hosts a two-process interactive forward-pass tool for the Nemotron runtime:

- `nemotron_interactive_forward.py`
- `nemotron_interactive_forward_server`

It follows the broad structure from the `khk/wip-nano` branch, but it is intentionally narrower and more honest about current runtime limits.

## Current Contract

This first Nemotron version uses:

- direct `SingleTokenForwardModel::Create(...)`
- full prompt rerender each turn
- full prompt prefill each turn
- greedy decode only
- line-oriented stdin/stdout IPC

This version does **not** use:

- `CreateFromCache(...)`
- prefix-cache restore/commit across turns
- request-context snapshot/restore
- sampling

Those paths are left out on purpose because the direct parity path is now the working baseline, while the cache-backed model-build path is still unqualified.

## Why Prefix Cache Is Not Exposed Yet

The nano design assumes the runtime can restore a reusable request state and continue prefill from a matched prefix. This Nemotron runtime is not there yet for the interactive serving path.

Today, the safe contract is:

- Python owns conversation history and chat-template rendering
- Python sends the complete prompt token sequence for the current turn
- C++ creates a fresh `RequestExecutionContext`
- C++ runs full prefill and then greedy decode

That avoids pretending that multi-turn cache reuse works when it does not.

## Protocol

The Python CLI spawns the C++ server and talks over line-oriented stdin/stdout.

Startup:

```json
{"ok":true,"status":"ready","model_id":"...","conversation_id":"interactive-0","prefix_cache_supported":false}
```

Commands:

- `EXIT`
- `RESET`
- `TURN\t<max_new_tokens>\t<eos_csv>\t<prompt_csv>`

Responses are one JSON line each. `TURN` responses include:

- prompt token count
- generated token ids
- prefill timing
- first-token selection timing
- per-step decode timings
- aggregate throughput
- runtime execution stats snapshot

## Important Limitations

1. Prompt logits are still materialized as `[prompt_tokens, vocab_size]` because that is the current `RunPrefill(...)` API contract.
2. Large prompts therefore allocate a large logits tensor, even though the tool only copies the last prefill row back to host.
3. No automatic context truncation is implemented in the Python CLI yet.
4. `/cache` is intentionally unsupported.

## Intended Use

Use this tool to:

- smoke-test interactive chat turns against the direct path
- inspect prefill and decode latency by turn
- exercise graph-on / graph-off runtime configurations
- validate future forward-path changes without rebuilding bespoke test harnesses
