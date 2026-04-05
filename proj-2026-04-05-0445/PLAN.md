# Plan: Post-Attention Routed MoE / Mamba Optimization

Project directory: `./proj-2026-04-05-0445`

## Status

This plan supersedes the earlier "fused grouped MoE prefill kernel" plan that
was written before the attention rewrite landed. The attention-first plan in
`ATTENTION_PLAN.md` is now complete. The fresh post-attention profile is saved
in `post_attention_root_cause.md`.

## Goal

Optimize the remaining dominant prefill work for Nemotron 3 Nano NVFP4 on the
local RTX 5090 runtime. The current post-attention baseline is already much
better:

- `cold_prefill_prefix1024`: `5971.231 ms` -> `440.312 ms`
- `cold_prefill_prefix4096`: `133038.284 ms` -> `1369.694 ms`

The next goal is not "fix attention". It is to cut the remaining routed-MoE
and Mamba cost on this exact stack without broadening scope to generic models
or datacenter-only kernels.

## Scope Guardrails

- **Target GPU only**: GeForce RTX 5090, consumer Blackwell, `SM120`, CUDA
  13.0.
- **Target model only**: Nemotron 3 Nano NVFP4 with `hidden_size=2688`,
  `routed_expert_intermediate_size=1856`,
  `shared_expert_intermediate_size=3712`, `n_routed_experts=128`, `top_k=6`,
  `activation=ReLU²`, `46` expert layers, causal paged KV cache, `64k`
  retained context.
- **Target workload only**: single-user, long multi-turn conversations,
  `<=4` concurrent requests, low scheduler complexity, no EP/TP, no LoRA, no
  multimodal path, no obligation to generalize the fast path.
- **Optimization objective**: improve the real local serving path. Do not spend
  time on detours that are not candidates for the final architecture.

## Current Root Cause

The fresh `1024`-token cold-prefill `nsys` profile shows:

- Mamba prefill kernels: about `31%` of GPU kernel time
- routed MoE GEMMs plus runtime NVFP4 input packing/scaling: about `43%`
- additional routed MoE dispatch/finalize kernels: about `9%`
- attention kernels: about `6%`

That means:

1. Attention is no longer the gating issue.
2. Grouped routed-expert work is worth revisiting.
3. The old MoE-only plan was incomplete because it treated runtime
   FP32→NVFP4 input packing as secondary, but that packing/scaling path alone
   is now a large cost.
4. Mamba prefill is now a co-primary bottleneck and must stay in scope.

Reference artifact: `post_attention_root_cause.md`

## Prior Art

- **vLLM / FlashInfer CUTLASS MoE**
  Uses grouped routed-expert execution with explicit routing and finalize
  structure. This remains the closest external prior art for reducing per-expert
  launch count on the local workload.
- **TRT-LLM grouped GEMM / fused MoE**
  Useful for algorithmic structure and grouped execution patterns, but much of
  the strongest path is shaped around datacenter Blackwell assumptions and is
  not a direct template for consumer `SM120`.
- **Our current runtime**
  Already has native SM120 NVFP4 GEMMs that are individually fast. The problem
  is now the whole routed path: repeated input packing/scaling, routing/finalize
  traffic, and the remaining launch surface.

## Locked Contract

- Exact Nano routed-MoE dimensions only.
- `SM120` only for the production fast path.
- BF16 live activations and BF16 KV cache remain the runtime default.
- `top_k=6`, `ReLU²`, FP32 accumulation/output on the routed path unless a
  later step proves another choice is both faster and safe.
- `<=4` concurrent requests, optimize `1` first.
- `64k` retained context is required, but the main optimization target remains
  the routed prefill path, not arbitrary full-context all-at-once operation.

## Architecture Decision

Do **not** resume the old plan as "write one giant custom grouped kernel
immediately". The new order should be:

1. decide what happens to runtime FP32→NVFP4 input packing
2. reduce routed-expert launch count and dispatch overhead
3. optimize Mamba prefill in parallel or immediately after the routed path,
   depending on the next measured split

If grouped routed-expert execution is built without addressing runtime packing,
the result will leave a large known cost untouched.

## Steps

- [x] **0. Freeze the post-attention baseline and root-cause profile**
  Save the current TTFT and fresh `nsys` kernel-time breakdown as the new
  baseline for all remaining work.
  Key files:
  `post_attention_root_cause.md`,
  `step9_runs/20260405T/post_attention_ttft.txt`,
  `step9_runs/20260405T/post_attention_1024_stats.txt`

- [x] **1. Lock the revised optimization contract**
  Rewrite the implementation contract around the post-attention profile:
  routed-MoE/input-packing work plus Mamba, not attention. Freeze the target
  metric set for future comparisons:
  - `cold_prefill_prefix1024`
  - `cold_prefill_prefix4096`
  - `cached_committed_head_prefix8192_tail32`
  - `cached_committed_head_prefix32768_tail32`
  - `cached_committed_head_prefix65536_tail32`
  Key files:
  `PLAN.md`,
  `ATTENTION_PLAN.md`,
  `post_attention_root_cause.md`

- [ ] **2. Decide the routed-expert input format and packing strategy**
  The current runtime spends a large share of GPU time in:
  - `PackRowMajorFp32ToNvfp4Kernel`
  - `ComputeGlobalMaxAbsKernel`
  This step must decide whether the production path should:
  - eliminate runtime FP32→NVFP4 packing,
  - fuse it into a reduced-launch routed path,
  - or replace it with a different activation/input contract.
  Do not start grouped routed-expert work until this is explicit.
  Key files:
  `runtime/src/backend/fused_moe_prefill.cu`,
  `runtime/src/backend/expert_layer.cpp`,
  `runtime/src/backend/device_nvfp4_matrix.cu`,
  `runtime/src/api/single_token_forward_model.cpp`

- [ ] **3. Prototype reduced-launch routed-expert execution**
  Revisit grouped routed-expert execution with the new profile in mind. The
  candidate design is still "routing separate, grouped expert execution,
  finalize explicit", informed by vLLM and TRT-LLM prior art. But the first
  production-form prototype should be judged against the post-attention
  baseline, not the old 6-second TTFT world.
  Exit criterion:
  - lower routed-path GPU time than the current per-expert path
  - no regression in correctness on real Nano weights
  - a clear story for how input packing is handled
  Key files:
  `runtime/src/backend/expert_layer.cpp`,
  `runtime/src/backend/fused_moe_prefill.cu`,
  `runtime/src/backend/expert_routing_device.cu`

- [ ] **4. Reduce routed dispatch/finalize and allocator churn**
  The current post-attention profile still shows meaningful routed non-GEMM
  work plus noticeable `cudaMalloc` / `cudaFree` traffic in CUDA API time.
  After step 3 exists, reduce the residual routing/finalize overhead and remove
  avoidable allocation churn on the hot path.
  Key files:
  `runtime/src/backend/expert_routing_device.cu`,
  `runtime/src/backend/request_context.cpp`,
  `runtime/src/backend/expert_layer.cpp`

- [ ] **5. Optimize Mamba prefill**
  Mamba is now a co-primary bottleneck. Continue from the already-landed
  bounded-chunking fix and profile the actual hot Mamba kernels rather than
  assuming the routed-MoE path will dominate forever.
  Key files:
  `runtime/src/backend/mamba_layer.cpp`,
  `runtime/src/backend/mamba_ssd_prefill.cu`,
  `runtime/src/backend/mamba_conv_prefill.cu`

- [ ] **6. Re-profile and choose the next default workstream**
  After steps 2 through 5 materially change the runtime, capture a fresh
  end-to-end profile and decide which of these becomes the next default
  production focus:
  - routed MoE/input packing
  - Mamba prefill
  - decode-path cleanup
  - scheduler/prefix-cache policy
  Key files:
  `post_attention_root_cause.md`,
  `PLAN.md`,
  `TODO.md`

## Frozen Baseline (post-attention, SM120, 2026-04-05)

### Cold Prefill TTFT

| Prompt | Median | p95 |
|--------|--------|-----|
| 256 tokens | 293 ms | 396 ms |
| 1024 tokens | 440 ms | 444 ms |
| 4096 tokens | 1,370 ms | 1,374 ms |

### Cached Prefix + 32 Tail Tokens

| Prefix | Cold TTFT | Hot-prefix TTFT | Tail Prefill | Speedup |
|--------|-----------|-----------------|--------------|---------|
| 4096 + 32 | 1,551 ms | 210 ms | 172 ms | 7.4x |
| 8192 + 32 | 3,241 ms | 243 ms | 184 ms | 13.4x |
| 32768 + 32 | 20,200 ms | 500 ms | 308 ms | 40.4x |
| 65536 + 32 | 63,395 ms | 845 ms | 477 ms | 75.1x |

### Kernel Time Split (1024-token cold, nsys)

| Component | Share |
|-----------|-------|
| MambaSsdPrefill | 28.7% |
| Mamba other (conv + group norm) | 2.6% |
| Routed NVFP4 GEMM | 26.3% |
| PackRowMajorFp32ToNvfp4 | 10.7% |
| ComputeGlobalMaxAbs | 6.2% |
| Routed dispatch/finalize | 8.4% |
| Attention (prefill + decode) | 6.0% |
| **Mamba total** | **31.3%** |
| **Routed MoE total** | **51.6%** |

### Decode (unchanged)

| Metric | Value |
|--------|-------|
| Steady-state | 64.6 tok/s |

### Optimization Contract

All future steps in this plan are measured against the baselines above.
Success means materially reducing routed-MoE and/or Mamba kernel time
without regressing correctness (62/62 ctest), decode throughput, or
attention prefill latency.

## Verification Policy

### Tier 1: after each implementation step

- `cmake --build build-sm120-relwithdebinfo -j$(nproc)`
- `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure`

### Tier 2: after routed-path or Mamba changes

- `nano_prefix_cache_ttft_bench`
- one fresh `nsys` trace on `cold_prefill_prefix1024`
- one long-context cached-prefix run at `32k` or `64k`

### Tier 3: before changing priorities again

- refreshed root-cause note in `post_attention_root_cause.md`
- explicit decision on whether grouped routed-MoE work remains ahead of Mamba

## Progress

| # | Step | Status | Notes |
|---|------|--------|-------|
| 0 | Freeze the post-attention baseline and root-cause profile | done | `1024` cold TTFT `440.312 ms`, `4096` cold TTFT `1369.694 ms`, attention now about `6%` of GPU kernel time |
| 1 | Lock the revised optimization contract | done | Baseline frozen below |
| 2 | Decide routed-expert input format and packing strategy | pending | |
| 3 | Prototype reduced-launch routed-expert execution | pending | |
| 4 | Reduce routed dispatch/finalize and allocator churn | pending | |
| 5 | Optimize Mamba prefill | pending | |
| 6 | Re-profile and choose the next default workstream | pending | |
