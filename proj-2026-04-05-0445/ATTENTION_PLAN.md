# Plan: Attention-First Prefill / Extend Backend

Project directory: `./proj-2026-04-05-0445`

## Status

This is the active execution plan for performance work in this project. The
older fused-MoE plan in `PLAN.md` is explicitly deferred until attention is no
longer the dominant bottleneck.

## Goal

Replace the current multi-token `PagedAttentionDeviceFallback` path with an
end-state attention backend that is actually aligned with the architecture we
want for local Nano serving on RTX 5090. The immediate target is to remove the
current root cause shown by the fresh `SM120` profile: fallback attention is
`98.7%` of GPU time, while routed MoE GEMMs are only `0.2%`.

## Scope Guardrails

- **Target GPU only**: GeForce RTX 5090, consumer Blackwell, `SM120`, CUDA
  13.0. Do not assume datacenter Blackwell (`SM100/103`) support, cubins,
  shared-memory budgets, NVLink behavior, or TRT-LLM-gen kernels transfer
  cleanly.
- **Target model only**: Nemotron 3 Nano NVFP4 with attention geometry
  `q_heads=32`, `kv_heads=2`, `head_dim=128`, `tokens_per_page=16`, causal
  paged KV cache, maximum retained context `64k`.
- **Target workload only**: single-user, long multi-turn conversations, `<=4`
  concurrent requests, low scheduler complexity, no DCP, no EP/TP, no sink
  attention, no multimodal path, no obligation to generalize the fast path to
  arbitrary shapes.
- **Optimization objective**: optimize prefill and extend on this exact local
  serving path. Do not spend engineering time on measurement-only detours that
  are not candidates for the final architecture.

## Why The Plan Changed

- Fresh Nsight Systems data in `baseline_root_cause.md` shows:
  - `PagedAttentionDeviceFallback`: `98.7%` of GPU time
  - Mamba SSD prefill: `0.6%`
  - NVFP4 routed MoE GEMMs: `0.2%`
- The current 1024-token cold TTFT is about `5971 ms`, and the 4096-token cold
  TTFT is about `133038 ms`.
- Continuing to optimize routed MoE first would attack a rounding error on the
  current baseline.

## External Oracle Policy

- **Primary oracle: vLLM**
  - Use vLLM to establish attention and MoE operation floors on this exact
    machine.
  - Use vLLM's attention benchmark harness and, when needed, full-model traces.
  - Record the backend that actually activates on this RTX 5090 build. Do not
    assume FlashInfer, FlashAttention, or TRTLLM attention is active without
    proof.
- **Secondary oracle: TRT-LLM**
  - Use TRT-LLM layer-wise and NVTX tooling to cross-check full-layer
    composition and attention cost.
  - Do **not** treat TRT-LLM as the MoE oracle for Nano NVFP4 on this machine:
    its relevant FP4/TRTLLM-gen MoE paths are not a clean `SM120` match.
- **Non-goal**
  - Do not enable cuDNN in this runtime merely to get a quick number if cuDNN
    is not a serious end-state candidate for the architecture we want.

## Parameters To Lock Down Before Coding

- Attention geometry: `32` query heads, `2` KV heads, `128` head dim,
  `16` tokens per page.
- Data types: BF16 Q/K/V path, BF16 KV cache, causal attention only.
- Page table contract: `int32` page tables, current paged-KV layout only.
- Runtime query buckets to optimize first:
  - cold prefill: `256`, `1024`, `4096`
  - long-turn extend: `128`, `256`, `512`, `1024` with live context
    `8k`, `32k`, `64k`
- Concurrency policy:
  - benchmark and optimize `1` request first
  - verify scheduler behavior up to `4` requests
  - if needed, serialize long prefills rather than degrade decode
- Maximum supported attention fast path:
  - exact Nano attention shape only
  - `64k` retained context, but bounded query-chunk sizes are allowed
- Backend policy:
  - one production multi-token backend
  - one decode-specialized backend if it still wins
  - device fallback only as a correctness backstop

## Architecture Decision Gate

Before writing runtime attention code, choose one end-state direction and reject
the others for this project:

1. **Adopt an existing backend as the production path**
   A backend such as cuDNN or a FlashInfer-like path is acceptable only if it
   is available on consumer `SM120`, fits the runtime's paged-KV ABI, and is
   close enough to the external oracle that writing a custom kernel is unlikely
   to pay back.
2. **Implement a custom specialized attention backend**
   This is the likely direction if the best external numbers depend on
   unavailable/private cubins, incompatible scheduler assumptions, or a backend
   stack that does not map cleanly onto this runtime.

The project should not proceed with ad hoc integration work until this gate is
closed.

## Steps

- [x] **0. Freeze the old MoE plan and make revisit criteria explicit**
  Add a project-local TODO entry stating that `PLAN.md` must be revisited only
  after a new attention backend lands and a fresh end-to-end profile is
  captured. The revisit question is not "can we still optimize MoE?" but
  "does MoE remain important enough to deserve implementation time after
  attention is fixed?"
  Key files: `proj-2026-04-05-0445/TODO.md`, `proj-2026-04-05-0445/PLAN.md`

- [x] **1. Define the oracle benchmark contract**
  Freeze the exact shapes and workload contract used for all external
  comparisons: Nano attention geometry, BF16 KV cache, cold prefill buckets
  (`256`, `1024`, `4096`), long-turn extend buckets (`128`, `256`, `512`,
  `1024`) against live context (`8k`, `32k`, `64k`), and concurrency points
  (`1`, then `4`). If a benchmark cannot match that contract closely, it does
  not count as a decision-grade oracle.
  Key files: `runtime/src/api/single_token_forward_model.cpp`,
  `proj-2026-04-05-0445/ATTENTION_PLAN.md`,
  `proj-2026-04-05-0445/ORACLE_BENCHMARK_CONTRACT.md`,
  `proj-2026-04-05-0445/vllm_attention_oracle_single.yaml`,
  `proj-2026-04-05-0445/vllm_attention_oracle_concurrency4.yaml`

- [x] **2. Capture external oracle floors on this exact machine**
  Run the vLLM attention benchmark suite and any needed full-model traces on
  this RTX 5090, recording the actual backend activated for each case. Use
  TRT-LLM only as a secondary cross-check for attention/layer-level cost
  composition. Save the results under this project directory so the rest of the
  work is grounded in measured floors rather than assumptions.
  Key files: `third_party/vllm/benchmarks/attention_benchmarks/README.md`,
  `third_party/vllm/benchmarks/kernels/benchmark_paged_attention.py`,
  `third_party/TensorRT-LLM/tensorrt_llm/tools/layer_wise_benchmarks/calibrator.py`,
  `proj-2026-04-05-0445/oracle_runs/`,
  `proj-2026-04-05-0445/oracle_initial_findings.md`

- [x] **3. Choose the end-state attention architecture**
  Based on the oracle data and support matrix, explicitly choose whether the
  production direction is:
  - an adopted backend that we can actually ship on consumer `SM120`, or
  - a custom specialized multi-token paged-attention backend.
  Reject measurement-only detours here. If cuDNN is not selected as the final
  architecture, do not spend implementation time integrating cuDNN just to get
  off the fallback path.
  Key files: `proj-2026-04-05-0445/ATTENTION_PLAN.md`,
  `runtime/src/backend/attention_layer.cpp`,
  `proj-2026-04-05-0445/attention_architecture_decision.md`

- [x] **4. Lock down the runtime attention contract**
  Freeze the ABI and policy surface before coding: page-table layout, page size,
  dtype policy, chunking policy for long prefills, max query tokens per launch,
  concurrency behavior, prefix-cache interaction, and fallback rules. Decide
  now whether the production path uses one chunk size or a small fixed set of
  chunk sizes; do not leave this as an open-ended autotuning problem.
  Key files: `runtime/include/nemotron/attention_layer.h`,
  `runtime/include/nemotron/paged_attention_plan.h`,
  `runtime/src/backend/request_context.cpp`,
  `proj-2026-04-05-0445/attention_runtime_contract.md`

- [x] **5. Build an attention-only correctness and profiling harness**
  Add or extend harnesses so the chosen backend can be validated independently
  of the rest of the model. The harness must cover:
  - pure cold prefill
  - extend with restored prefix state
  - exact Nano geometry only
  - comparison against a trustworthy reference for selected shapes
  The same harness should emit profiler-friendly markers so kernel-level and
  layer-level regressions are easy to spot.
  Key files: `testing/backend`, `benchmarks`, `runtime/src/backend/attention_layer.cpp`

- [x] **6. Implement the first production-form multi-token attention backend**
  Write the first real backend implementation for multi-token prefill / extend,
  using the architecture selected in step 3. It should target the locked Nano
  geometry only and be integrated behind an explicit backend policy, not an env
  var maze. Keep the current device fallback only as a correctness backstop.
  Key files: `runtime/src/backend/attention_layer.cpp`,
  `runtime/src/backend`,
  `runtime/include/nemotron`

- [x] **7. Optimize for the real long-conversation workload**
  Tune the new backend for the actual local workload rather than synthetic peak
  only: long prompts, restored prefixes, bounded chunking, and low concurrency.
  Measure:
  - 1024-token cold TTFT
  - 4096-token cold TTFT
  - extend latency at `8k`, `32k`, `64k` live context
  - behavior at concurrency `4`
  Use Nsight Compute and Nsight Systems only after the production-form backend
  exists.
  Key files: `proj-2026-04-05-0445`,
  `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`

- [x] **8. Make the new backend the default and shrink policy leakage**
  Once correctness and performance are acceptable, make the chosen backend the
  default multi-token attention path for supported Nano shapes. Keep the old
  fallback reachable only for unsupported cases and debugging. Remove any
  policy branches that exist solely because the fallback path used to be the
  default.
  Key files: `runtime/src/backend/attention_layer.cpp`

- [x] **9. Re-profile the whole model and re-assess the old MoE plan**
  Capture a fresh full-model TTFT profile after the new attention path is in
  place. Then revisit `PLAN.md` and decide whether grouped MoE fusion is still
  worth doing, or whether the next priorities are instead input packing,
  scheduler policy, prefix-cache policy, or Mamba work.
  Key files: `proj-2026-04-05-0445/PLAN.md`,
  `proj-2026-04-05-0445/TODO.md`,
  `proj-2026-04-05-0445`

## Verification Policy

### Tier 1: after every implementation step
- `cmake --build build-sm120-relwithdebinfo -j$(nproc)`
- `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure`

### Tier 2: before switching defaults
- attention-only correctness harness for locked Nano shapes
- `nano_prefix_cache_ttft_bench`
- Nsight Systems trace on 1024-token cold prefill
- one long-turn extend trace at `32k` or `64k` live context

### Tier 3: after step 9
- refreshed full-model kernel-time breakdown
- explicit decision note on whether `PLAN.md` is resumed, revised, or dropped

## Progress Notes

- `2026-04-05`: step 1 completed by freezing the external oracle contract in
  `ORACLE_BENCHMARK_CONTRACT.md` and project-local vLLM configs/scripts.
- `2026-04-05`: step 2 completed for the primary vLLM oracle path. Artifacts
  are saved under `oracle_runs/`. Initial outcome: on this RTX 5090 setup,
  `supports_trtllm_attention=false`, the vLLM benchmark resolves to BF16, and
  FlashInfer is the best backend across the long-context extend cases while
  FlashAttention remains competitive on the smallest prefills.
- `2026-04-05`: step 3 completed by explicitly choosing a custom specialized
  multi-token paged-attention backend as the project direction, using
  FlashInfer-like long-context behavior as prior art rather than treating
  cuDNN or TRTLLM attention as the end-state architecture.
- `2026-04-05`: step 4 completed by freezing the runtime contract around the
  existing BF16 paged-KV ABI, `tokens_per_page=16`, `max_multi_token_query_tokens=1024`,
  exact-prefix restore semantics, and a backend split of `token_count==1`
  decode-specialized versus `2..1024` multi-token attention.
- `2026-04-05`: step 5 completed by adding a Nano-only attention harness in
  `benchmarks/nano_paged_attention/` plus a correctness test in
  `testing/backend/nano_paged_attention_test.cpp`. The harness covers cold
  prefill and restored-prefix extend with the locked `32q/2kv/128d/page16`
  geometry, uses a CPU paged-attention reference on selected shapes, and emits
  NVTX ranges for `query_pack`, `scatter_kv`, `attention`, and
  `output_unpack`. Verified with
  `ctest --test-dir build-sm120-relwithdebinfo -R '^nano_paged_attention_test$'`
  plus smoke benchmark runs for `prefill_q256` and `extend_q128_live8k`.
- `2026-04-05`: step 6 completed by adding `AttentionBackend::kNanoMultiToken`,
  wiring it ahead of cuDNN/generic fallback for exact Nano multi-token shapes,
  implementing `RunPagedAttentionNanoMultiToken` in
  `attention_device_fallback.cu`, and adding the fixed `1024`-token chunking
  path in `attention_layer.cpp` for larger prefills. Runtime integration is
  covered by `testing/backend/nano_attention_layer_integration_test.cpp`.
- `2026-04-05`: step 7 completed after fixing the remaining long-context
  measurement blockers outside attention itself:
  - reusable-state snapshot storage now allocates exact bytes instead of
    reserving the full shared-cache budget up front, which restored healthy
    8k/32k cached-prefix iteration behavior
  - boundary-only logits are now supported for multi-token prefill, removing
    the benchmark-only `[tokens, vocab]` allocation cliff for 32k/64k prompts
  - Mamba BF16 multi-token prefill now uses bounded chunking, which made the
    64k cold/cached cases measurable on this runtime
  - committed-head snapshot publish now retries after dropping the unique old
    conversation head under a tight shared-cache budget and clears the handled
    CUDA allocation error before returning, which made full 64k cached-prefix
    publication stable across repeated iterations
  Representative measured results:
  - `cold_prefill_prefix1024`: median cold TTFT `439.427 ms`
  - `cold_prefill_prefix4096`: median cold TTFT `1391.921 ms`
  - `cached_committed_head_prefix8192_tail32`: median hot-prefix TTFT
    `242.595 ms`
  - `cached_committed_head_prefix32768_tail32`: median hot-prefix TTFT
    `499.605 ms`
  - `cached_committed_head_prefix65536_tail32`: median hot-prefix TTFT
    `844.574 ms`, median commit/snapshot latency `18.999 ms`
  - `target_active_requests=4` remained effectively neutral on the
    representative 8k and 32k cached-prefix cases
- `2026-04-05`: step 8 confirmed complete. The runtime backend policy already
  selects `kNanoMultiToken` ahead of cuDNN and generic fallback for exact Nano
  multi-token shapes, and the focused validation set passed:
  `attention_backend_policy_test`,
  `nano_attention_layer_integration_test`,
  `nano_paged_attention_test`.
  Verification:
  - `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -R '^(attention_backend_policy_test|nano_paged_attention_test|nano_attention_layer_integration_test)$'`
  - `cmake --build build-sm120-relwithdebinfo -j$(nproc)`
  - attention-only benchmark artifacts saved under
    `attention_step6_benchmarks/20260405T094405Z/`
  Key harness deltas from those artifacts:
  - `prefill_q256` hot mean: `66.66 ms` fallback -> `0.204 ms` native backend
  - `extend_q128_live8k` hot mean: `4792.60 ms` fallback -> `5.80 ms` native backend
  - `prefill_q4096` chunked native backend hot mean: `25.22 ms`
- `2026-04-05`: step 9 completed with a fresh post-attention end-to-end
  re-profile saved under `step9_runs/20260405T/` and summarized in
  `post_attention_root_cause.md`.
  The new full-model TTFT numbers are:
  - `cold_prefill_prefix1024`: median `440.312 ms` vs old baseline `5971.231 ms`
    (`13.56x` faster)
  - `cold_prefill_prefix4096`: median `1369.694 ms` vs old baseline
    `133038.284 ms` (`97.13x` faster)
  The fresh `nsys` kernel-time breakdown for `cold_prefill_prefix1024` shows
  that attention is no longer the dominant cost:
  - Mamba prefill kernels: about `31%` of GPU kernel time
  - routed MoE GEMMs + runtime NVFP4 input packing/scaling: about `43%`
  - additional routed MoE dispatch/finalize kernels: about `9%`
  - attention kernels (multi-token prefill + first-token decode): about `6%`
  Decision: the old fused-MoE plan should not be executed as written, but
  grouped routed-expert work is again worth doing after rewriting that plan
  around the new post-attention profile. The updated order is:
  1. revise `PLAN.md` around the new baseline and include runtime packing
     elimination explicitly
  2. treat grouped routed-expert work as a real next-tier optimization
  3. keep Mamba prefill optimization in scope because it is now a co-primary
     bottleneck
