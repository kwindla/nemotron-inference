# Plan: Unified NVFP4 MoE Alignment vs vLLM

Project directory: `./proj-2026-04-03-0318`

## Goal

Aim for the fastest practical MoE execution on RTX 5090 for both:

- single-token decode
- short and medium prefill (`32`, `64`, `128`, `256` tokens)

This plan is no longer prefill-only. The architectural target is a unified routed MoE execution surface that can serve both decode and prefill, with specialization retained only when benchmarking proves it is worth the complexity.

## External Baseline

The external comparison target for Nemotron Nano NVFP4 on RTX 5090 is:

- vLLM `v0.19.0`
- model path: `vllm/model_executor/models/nemotron_h.py`
- MoE backend: `flashinfer_cutlass`

Do **not** treat `flashinfer_trtllm` as the primary latency baseline for this model. vLLM explicitly skips that path for Nemotron Nano NVFP4 because Nano's hidden size is `2688`, and `2688 % 512 != 0`.

The relevant vLLM architecture is:

- GPU top-k routing
- load-time NVFP4 backend-native weight preparation
- one fused expert call per chunk
- the same fused MoE stack for chunked prefill and non-chunked execution
- a runner-level chunk/window size separate from overall request capacity

Internal optimization work counts only if it is measured against this external baseline, not just against our current runtime.

## Current Runtime Snapshot

The current repo is structurally behind that baseline in three important ways:

1. Prefill fused execution is still plumbing-only. `RunFusedMoePrefill()` currently returns fallback immediately.
2. The active multi-token MoE path is still a host-routed, per-expert loop with gather/GEMM/activation/GEMM/scatter behavior repeated for each active expert.
3. The only real fused path today is decode-only, and it is a scalar/shared-memory dot-product kernel rather than a unified fused experts backend.

Known measurement result:

- `32`-token tail prefill is currently far slower than the original stretch target.
- The per-expert GEMM/pack/launch loop still dominates the measured cost.
- The existing routing histogram data is still incomplete for plan-locking purposes:
  - the current histogram is coarse
  - the current `32`-token artifact needs provenance verification
  - representative `64`-token and `128`-token histograms still need to be captured

## Plan Pivots

This is the concrete change in direction relative to the earlier prefill-centric plan.

### 1. Optimize against a unified MoE surface, not a prefill-only kernel

The primary architecture should be one MoE execution surface for `token_count >= 1`. Decode specialization is allowed, but only as a backend choice under that shared surface.

### 2. Treat vLLM's routing contract as the reference shape

The first-class routing product should be GPU `topk_ids` and `topk_weights`, not an expert-major routing table.

Expert-major compaction may still exist, but only as:

- an adapter for the legacy fallback path
- an adapter for a future custom backend that truly requires it

It should not remain the center of the design.

### 3. Move fast-path weight work to load time

The fast path should be allowed to use a backend-native prepared NVFP4 weight layout. Raw monolithic weights can remain for:

- fallback
- validation
- debug comparisons

But "no new weight layout" should not constrain the fast path.

### 4. Separate MoE execution window from request capacity

MoE execution window size must be independent from:

- `max_tokens`
- KV reservation
- request bookkeeping
- persistent request activation capacity

Long-prompt support should come from internal chunk orchestration, not from sizing MoE scratch or persistent row buffers to full prompt context.

### 5. Keep the current decode kernel only if it wins by benchmark

The existing decode-only fused kernel should not be preserved by policy. It should remain only if it is measurably faster than the unified fused backend once that backend exists.

### 6. Delay a custom SM120 backend until the unified fused path is real

An owned SM120-specific routed-expert kernel may still be the right end state, but it should be a follow-on backend, not the first milestone. The custom kernel should target the residual gap after:

- unified backend surface
- GPU routing contract
- load-time weight preparation
- runner-level chunking

are already in place.

## Constraints

- The external performance bar is vLLM `flashinfer_cutlass`, not just our current runtime.
- The fast path must support both decode and prefill, even if different internal backends are selected by token count.
- Unsupported expert counts, unsupported dimensions, missing residency, or failed fast-path scratch allocation must disable the fast path non-fatally and preserve baseline execution.
- Consumer Blackwell is `SM120 / compute capability 12.0`; do not assume SM100-only features such as `tcgen05` or TMEM portability.
- For future custom kernels on SM120, the important limits remain:
  - `128 KB` shared memory per SM
  - `99 KB` shared memory per block
  - `48` resident warps per SM
  - `1536` resident threads per SM
  - `64K` 32-bit registers per SM
- For Nano (`hidden=2688`, routed intermediate `1856`), keeping fp32 input and fp32 intermediate state live across both routed matmuls costs `M * (2688 + 1856) * 4` bytes. That is already near the per-block shared-memory limit at `M=5`, so any custom backend must treat `M>=5` as a streamed or tiled regime.
- Do not spend the next milestone on more cleanup inside the host-routed cuBLASLt fallback except where required for correctness or comparison.

## RTX 5090 Sizing Policy

The `64`-token region remains important, but it is now a tuning point inside a unified plan rather than the architectural center of a prefill-only design.

### Separate capacities

- **Request context capacity**: total prompt + decode tokens the request may hold.
- **MoE execution window**: the token chunk size processed by one routed MoE call.
- **Persistent request row buffers**: reusable activation buffers on `RequestExecutionContext`.

These must be treated as different capacities.

### Recommended initial policy

- Keep full request capacity in KV/Mamba state and request bookkeeping.
- Add a dedicated MoE execution window knob, for example `moe_max_tokens_per_call` or `prefill_window_tokens`.
- Make `64` tokens the first-class tuning point for early fused benchmarking.
- Treat `128` tokens as the next extension point.
- Treat `256` tokens as a later streamed-kernel regime.
- Shrink persistent request row buffers to decode-sized or another small fixed cap rather than coupling them to full request capacity by default.

### Why this still matters

For Nano (`128` routed experts, `top_k=6`):

- `32` tokens -> `192` routed selections
- `64` tokens -> `384` routed selections
- `128` tokens -> `768` routed selections
- `256` tokens -> `1536` routed selections

Increasing chunk size pushes more experts into `M>=5`, where SM120 shared-memory limits make simple tiny-`M` kernels much less attractive. That is why `64` remains a useful operating point, but it should not force a separate prefill-only architecture.

## Execution Phases

- [x] **0. Lock the external benchmark and measurement discipline**
  Benchmark vLLM locally for Nemotron Nano NVFP4 on RTX 5090 and make it the explicit bar for both decode and prefill. Capture:
  - end-to-end latency for `1`, `32`, `64`, `128`, and `256` tokens
  - MoE-layer time where available
  - representative per-layer routing histograms for at least `32`, `64`, and `128` tokens
  - exact provenance for the current `32`-token baseline artifact
  Internal profiling does not replace this external comparison.
  Key files: `third_party/vllm/`, `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-03-0318/`

- [x] **1. Introduce a unified MoE backend surface**
  Add an internal backend surface for routed MoE execution with at least:
  - `PrepareWeights(...)`
  - `Run(...)` for `token_count >= 1`
  - `Supports(config, token_count, device)`

  Planned backends:
  - current batched cuBLASLt fallback
  - current decode-only scalar kernel
  - unified fused backend
  - optional future custom SM120 backend

  The layer and model code should route both decode and prefill through this shared selection surface.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/include/nemotron/expert_layer.h`, `runtime/include/nemotron/fused_moe_decode.h`, `runtime/include/nemotron/fused_moe_prefill.h`

- [x] **2. Make GPU top-k the primary routing contract**
  Keep router top-k selection on device and make GPU `topk_ids` / `topk_weights` the first-class interface between routing and expert execution. Keep expert-major compaction only as an adapter when the selected backend requires it.

  This changes the role of the current device routing helper:
  - useful for the legacy path
  - possibly useful for a later custom backend
  - not the primary contract for the vLLM-aligned fast path
  Key files: `runtime/src/backend/fused_moe_decode.cu`, `runtime/src/backend/expert_routing_device.cu`, `runtime/include/nemotron/expert_routing_device.h`

- [x] **3. Add load-time NVFP4 backend-native weight preparation**
  Prepare routed and shared expert weights once after loading into the format required by the chosen fused backend. Keep the raw monolithic views only for fallback, validation, and debugging.

  This is a major plan change. The fast path should not be constrained to consume only the raw monolithic layout if backend-native preparation produces materially better execution behavior.
  Key files: `runtime/src/backend/monolithic_expert_weights.cu`, `runtime/src/backend/expert_layer.cpp`

- [x] **4. Separate MoE chunking from request capacity**
  Add a dedicated MoE execution window knob and move long-prefill chunk orchestration to the model runner. Prompts larger than the MoE execution window should be driven as repeated internal chunks using the existing `RunPrefill(...)` / `ContinuePrefill(...)` semantics, rather than forcing expert-layer fallback to the monolithic path.

  This phase also owns the request-buffer cleanup:
  - stop coupling persistent request row buffers to full request capacity by default
  - keep `max_tokens` for request/KV limits
  - size MoE scratch to the bounded execution window, not to the full request context
  Key files: `runtime/src/api/single_token_forward_model.cpp`, `runtime/src/backend/request_context.cpp`, `runtime/src/backend/expert_layer.cpp`

- [x] **5. Land a unified fused backend before a custom kernel**
  Implement one fused backend that follows the vLLM-class contract:
  - GPU top-k inputs
  - prepared NVFP4 weights
  - one fused routed-expert execution path used by both decode and prefill
  - shared experts integrated under the same execution surface

  Preferred order:
  1. Build the backend surface and integration path.
  2. Make the backend correct for decode and bounded-window prefill.
  3. Benchmark it against the current decode kernel and the current batched prefill path.

  If direct FlashInfer/CUTLASS integration is acceptable in this codebase, that is the fastest route to vLLM-class alignment. If it is not acceptable, the internal backend should still mirror that contract closely.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/src/backend/fused_moe_decode.cu`, `runtime/src/backend/expert_layer.cpp`

- [x] **6. Decide the fate of the current decode kernel by measurement**
  Once the unified fused backend exists, benchmark decode at `token_count == 1`:
  - keep the current decode-only kernel if it still wins
  - otherwise demote or remove the special case

  The repository should not carry a separate decode architecture unless it earns its keep on this model and hardware.
  Key files: `runtime/src/backend/fused_moe_decode.cu`, `runtime/src/backend/expert_layer.cpp`

- [x] **7. Benchmark the unified path against vLLM and baseline**
  Re-run the full matrix against:
  - current baseline runtime
  - unified fused backend
  - vLLM `flashinfer_cutlass`

  Required coverage:
  - correctness against the current runtime
  - decode latency
  - `32`-token tail prefill
  - `64`-token and `128`-token chunked prefill
  - fallback behavior
  - non-fatal disable behavior
  - long-prompt chunked equivalence
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-03-0318/`

- [x] **8. Only then decide whether to build a custom SM120 backend**
  Pursue an owned SM120-specific routed-expert backend only if the unified fused path still leaves a clear measured gap on RTX 5090.

  If that happens, the custom backend should be scoped narrowly:
  - target the residual performance gap
  - use the real measured `M` histogram
  - specialize for the important buckets
  - handle `M>=5` as a streamed or tiled regime
  - plug into the unified backend surface instead of creating a third architecture
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/src/backend/fused_moe_decode.cu`, `runtime/src/backend/expert_routing_device.cu`

## What Not To Optimize Next

Do not spend the next milestone on:

- more host-path cleanup in the cuBLASLt fallback
- more expert-major routing machinery unless the selected backend requires it
- preserving the current decode kernel by policy
- locking the design around a prefill-only persistent kernel before the unified backend surface and weight-prep surface exist

## Progress

| # | Phase | Status | Commit | Notes |
|---|-------|--------|--------|-------|
| 0 | External benchmark and measurement discipline | done | 02d5ada | Benchmark script, wrapper, measurement methodology, and baseline provenance analysis landed |
| 1 | Unified MoE backend surface | done | 2537abc | MoeBackend interface + 4 concrete backends, unified dispatch loop in Run() |
| 2 | GPU top-k as primary routing contract | done | 4dc5eb4 | topk_ids/topk_weights in MoeBackend::Run(), single RunDeviceExpertSelection call site |
| 3 | Load-time NVFP4 backend-native weight preparation | done | fcb3f00 | MoeBackendPrepareContext + PreparedMoeWeights on all 4 backends |
| 4 | Separate MoE chunking from request capacity | done | acd7abf | moe_prefill_window_tokens knob + runner-level chunking loop for expert layers |
| 5 | Unified fused backend | done | a6f7582 | RunFusedMoePrefill implemented, UnifiedFusedBackend for token_count >= 1 |
| 6 | Decode specialization decision by benchmark | done | ea3df20 | Benchmark harness, BACKEND_SELECTION.md, NEMOTRON_FORWARD_UNIFIED_FUSED env var |
| 7 | External comparison vs vLLM and current baseline | done | 5cba945 | Full comparison harness, correctness verification, updated MEASUREMENT.md |
| 8 | Optional custom SM120 backend | done | — | Decision framework in SM120_BACKEND_DECISION.md, GO/NO-GO pending benchmark data |
