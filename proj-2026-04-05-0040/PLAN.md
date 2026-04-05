# Plan: Prefill Throughput Optimization

Project directory: `./proj-2026-04-05-0040`

## Goal

Optimize production prefill latency, not just the benchmark. The desired end state is that production prefill uses the largest MoE prefill batch that fits the runtime's VRAM policy by default, so expert layers stop paying small-window iteration overhead on long prompts.

## Context

Cold TTFT for a 1024-token prompt is 16.6 seconds on RTX 5090. The visible bottleneck is MoE prefill chunking: the runtime slices multi-token prefill into windows before entering each expert layer, so a 1024-token prefill can turn one expert-layer call into hundreds of expert-layer calls.

There are two separate problems today:

- The TTFT benchmark forces `moe_prefill_window_tokens` to `tail_token_count` when the flag is omitted, so the benchmark is not exercising the runtime's natural `0 => use max_tokens` behavior.
- In the runtime, expert multi-token capacity is effectively tied to per-layer pre-allocated scratch. A production fix cannot just raise the window, because large per-layer scratch would multiply across all expert layers and waste VRAM.

## Design Constraints

- Preserve `SingleTokenForwardConfig::max_tokens` as request-context capacity. Do not repurpose it as a MoE scratch knob.
- Keep decode scratch and decode performance intact.
- Use a request-scoped MoE prefill workspace owned by `RequestExecutionContext`; do not use a shared leased workspace for the first production implementation.
- The request-scoped workspace must cover all large multi-token expert temporaries, not just fused-prefill scratch. That includes the multi-token expert input/normalization/router/output temporaries as well as top-k, routing, and fused-prefill buffers.
- Keep MoE prefill capacity explicit and bounded. Do not introduce an unbounded "allocate whatever the prompt needs" default.
- Eliminate per-layer MoE scratch amplification. Multi-token MoE workspace must be reusable across expert layers within a forward pass, not permanently replicated in every expert slice.
- Production default should prefer full-sequence MoE prefill when it fits the configured VRAM policy, and fall back to chunking only when capacity requires it.
- The resolved MoE prefill capacity must be derived from live request-admission / request-context sizing inputs, not from a stale one-time model-create VRAM snapshot.
- Benchmark behavior should mirror the production runtime default rather than override it.

## Reference Implementation

vLLM processes all prompt tokens through each layer as a single `(num_tokens, hidden_dim)` tensor. The MoE layer (`fused_moe`) receives the full batch, runs grouped top-k, then dispatches all tokens to experts in one fused kernel.

Local reference: `third_party/vllm/vllm/model_executor/layers/fused_moe/`

## Current State

- `runtime/src/api/single_token_forward_model.cpp`: expert layers chunk prefill when `token_count > effective_moe_window_tokens`.
- `runtime/src/api/single_token_forward_model.cpp`: `EffectiveMoePrefillWindowTokens()` currently uses `config.max_tokens` when `moe_prefill_window_tokens == 0`.
- `runtime/src/backend/expert_layer.cpp`: multi-token expert fast paths size and guard against `ExpertLayerConfig::max_token_count`, and they currently own their own top-k/routing/scratch allocations.
- `runtime/src/backend/expert_layer.cpp`: multi-token expert execution also allocates large per-call temporaries (`normalized_bf16`, `input_fp32`, `normalized`, `router_logits`, `output_fp32`) outside the fused-prefill scratch path.
- `runtime/src/backend/request_context.cpp` and `runtime/src/api/single_token_forward_model.cpp`: request-context VRAM sizing already exists for hidden/residual/KV/state capacity, but it does not yet budget MoE prefill workspace.
- `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`: omitted `moe_prefill_window_tokens` is rewritten to `tail_token_count`, which prevents the benchmark from reflecting a production full-prefill default.
- `runtime/src/backend/mamba_layer.cpp`: the multi-token Mamba path still allocates per-call temporaries, but that is a secondary optimization until MoE traces say otherwise.

## Steps

- [x] **1. Add prefill tracing and capture a production-relevant baseline**
  Add lightweight timing around each layer `Run()` call inside `RunTokens()` in `runtime/src/api/single_token_forward_model.cpp`, gated by `NEMOTRON_FORWARD_PREFILL_TRACE=1`. When enabled, print layer index, layer kind, token count, effective MoE window, derived `window_count`, and wall time.

  Reuse the benchmark's existing CUDA-event timing for total `RunPrefill()` latency. For representative baseline runs, also enable `NEMOTRON_FORWARD_MOE_PROFILE=1` so we know which MoE backend is active and whether time is dominated by routing, workspace churn, or actual compute. Record the result as `proj-2026-04-05-0040/baseline_profile.txt`.

  Key files: `runtime/src/api/single_token_forward_model.cpp`, `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`

- [x] **2. Introduce request-scoped full multi-token expert workspace**
  Decouple multi-token MoE capacity from per-layer scratch ownership. Keep the existing per-slice decode scratch for `token_count == 1`, but add a request-scoped workspace owned by `RequestExecutionContext` for expert prefill.

  This workspace must include all large multi-token expert temporaries needed by the hot path:

  - `normalized_bf16`, `input_fp32`, `normalized`, `router_logits`, `output_fp32`
  - device top-k IDs and weights
  - routing tables / routing scratch
  - fused-prefill routed-output, gather, expert-up, and shared-up scratch

  Expert fast paths should validate against request-workspace capacity rather than per-slice static scratch size. If the requested token count exceeds workspace capacity, the runtime must fall back to chunking instead of failing.

  Key files: `runtime/include/nemotron/request_context.h`, `runtime/src/backend/request_context.cpp`, `runtime/include/nemotron/expert_layer.h`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/api/single_token_forward_model.cpp`

- [ ] **3. Add an explicit production MoE prefill capacity policy and budget it with request creation**
  Add a MoE-prefill-capacity setting that is separate from `max_tokens`. This may be expressed as capacity-in-tokens, workspace-budget-bytes, or an equivalent bounded policy, but it must remain distinct from request-context capacity.

  Resolve the effective MoE prefill capacity when creating / admitting a request context using live VRAM budget inputs and reserve policy. Extend the request-budget sizing path so MoE prefill workspace bytes are budgeted together with hidden/residual/KV/state bytes, rather than being an untracked extra allocation.

  Default behavior should be production-oriented:

  - if the user does not force a smaller `moe_prefill_window_tokens`, the runtime should use the largest MoE prefill batch allowed by the bounded workspace capacity
  - on hardware/configurations where the full prompt fits the bounded workspace policy, expert prefill should run as one window per expert layer
  - on tighter-memory configurations, chunking remains the fallback behavior

  Key files: `runtime/include/nemotron/single_token_forward_model.h`, `runtime/src/api/single_token_forward_model.cpp`, `runtime/include/nemotron/request_context.h`, `runtime/src/backend/request_context.cpp`, `runtime/include/nemotron/expert_layer.h`, `runtime/src/backend/expert_layer.cpp`

- [ ] **4. Make the benchmark follow the production runtime default**
  Change `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp` so `--moe-prefill-window-tokens` accepts `0` via `ParseNonNegativeSizeT`, and stop rewriting omitted `0` to `tail_token_count`.

  The benchmark should log both the explicit CLI value and the resolved runtime MoE prefill capacity/window so that benchmark output matches production behavior rather than masking it.

  Record the optimized result as `proj-2026-04-05-0040/optimized_profile.txt`.

  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`

- [ ] **5. Validate both full-batch and bounded-fallback production behavior, then revisit secondary bottlenecks**
  Validate the runtime with the optimized default path on the same 1024-token cold-prefill and cached 1024 + 4 tail cases used for baseline measurement. Success means:

  - expert-layer `window_count` collapses to `1` on the target 5090 production case
  - cold TTFT improves materially versus baseline
  - correctness and decode performance remain intact

  Also add at least one forced reduced-capacity validation where the resolved MoE prefill capacity is smaller than the prompt, so the runtime must chunk by policy and still produce correct results.

  Only after that, revisit Mamba prefill scratch reuse if the optimized traces show Mamba as a material share of remaining latency.

  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `runtime/src/backend/mamba_layer.cpp`, `runtime/include/nemotron/mamba_layer.h`

## Verification Policy

### Tier 1: every code-change step
- `cmake --build build -j$(nproc)`
- `ctest --test-dir build --output-on-failure`

### Tier 2: after step 4 and step 5
- `proj-2026-04-03-0318/verify_correctness.sh`
- `benchmarks/nano_fused_decode/nano_fused_decode_bench` or the existing decode-regression wrapper
- `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench`
- one reduced-capacity TTFT / correctness run that exercises the chunking fallback path

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Add prefill tracing and capture a production-relevant baseline | done | dfbf62a | |
| 2 | Introduce request-scoped full multi-token expert workspace | done | PENDING | |
| 3 | Add an explicit production MoE prefill capacity policy and budget it with request creation | pending | — | |
| 4 | Make the benchmark follow the production runtime default | pending | — | |
| 5 | Validate both full-batch and bounded-fallback production behavior, then revisit secondary bottlenecks | pending | — | |
