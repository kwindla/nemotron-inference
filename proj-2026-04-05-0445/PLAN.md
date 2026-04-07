# Plan: Post-Attention Routed MoE / Mamba Optimization

Project directory: `./proj-2026-04-05-0445`

## Status

This plan supersedes the earlier "fused grouped MoE prefill kernel" plan that
was written before the attention rewrite landed. The attention-first plan in
`ATTENTION_PLAN.md` is now complete. The fresh post-attention profile is saved
in `post_attention_root_cause.md`.

As of `2026-04-06`, this file is again the canonical optimization plan.
The later project directory `proj-2026-04-05-1704` now serves as the execution
log / artifact bundle for the recovery work:

- `proj-2026-04-05-1704/STEP3_RECOVERY_PLAN.md`
- `proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md`

## Canonical Update (2026-04-06)

### Current Objective

The target is no longer just "recover the old native baseline." The target is:

1. beat local `vLLM` cold prefill
2. beat local `TRT-LLM` PyTorch cold prefill
3. do it on one active runtime path, with device-only execution and behavioral
   reuse equivalence preserved

### Competition Surface

Use the direct exact-token prefill profile surface from
`proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md` as the race metric.

Current measured times on the local Nano checkpoint / RTX 5090:

| runtime | prefix4 | prefix128 | prefix4096 |
|---|---:|---:|---:|
| native runtime | `54.016 ms` | `127.698 ms` | `2503.116 ms` |
| vLLM | `48.011 ms` | `40.396 ms` | `82.938 ms` |
| TRT-LLM | `35.922 ms` | `38.407 ms` | `66.789 ms` |

That means the current external targets to beat are:

- `prefix4 < 35.922 ms`
- `prefix128 < 38.407 ms`
- `prefix4096 < 66.789 ms`

### Current Root Cause

The current cross-codebase profiles now make the work order much clearer than
the older internal-only `1024` trace:

- `prefix4`: native is still split between routed expert custom math and
  shared expert WMMA
- `prefix128`: native is still dominated by routed expert custom math
- `prefix4096`: native is dominated by routed expert custom math and then by
  Mamba prefill

By contrast, both `vLLM` and `TRT-LLM` spend these regions mostly in grouped
`CUTLASS` / `FP4` GEMM kernels with much smaller glue overhead, and their
long-prefix Mamba cost is much lower.

### Locked Contract Addendum

In addition to the original contract below, the active fast path is now locked
to these constraints:

- one active runtime path
- device-only runtime contract
- no hot-path `DtoH`
- behavioral reuse equivalence is the correctness bar, not bitwise identity

### Current Execution Order

This is the active order of operations for the optimization work:

1. finish the routed MoE transition to grouped `FP4xFP4` tensor-core math
2. move shared experts onto the same stronger kernel family
3. then attack long-prefix Mamba
4. only then spend time on attention polish or smaller cleanup

The immediate implementation gap is now very specific:

- the runtime already has device launch-plan infrastructure
- the runtime already has preallocated `DeviceNvfp4Matrix` work buffers
- but the active fused prefill path still uses a BF16/FP32-oriented routed
  kernel shape instead of the intended grouped `FP4xFP4` contract from step `3`

So the next code changes should start from the packed-activation / grouped-GEMM
boundary, not from more row-kernel scheduling tweaks.

### Immediate Prototype Finding

The first benchmark-only packed-activation prototype is now in
`benchmarks/nano_moe_prefill/nano_routed_up_bench.cpp`.

At the design-center `prefix128` case:

- `launch_plan_wmma_bf16_transposed_m16n32k16`: about `1.509 ms`
- `launch_plan_wmma_packed_input_transposed_m16n32k16`: about `2.193 ms`
- max diff vs baseline: about `2.318`

At `prefix4096`:

- `launch_plan_wmma_bf16_transposed_m16n32k16`: about `27.216 ms`
- `launch_plan_wmma_packed_input_transposed_m16n32k16`: about `42.512 ms`
- max diff vs baseline: about `2.318`

Interpretation:

- the existing `DeviceNvfp4Matrix` activation contract is not a direct
  substitute for the current BF16 routed-up kernel
- simply feeding packed activations into the current WMMA tile is slower and
  materially changes numerics
- the next grouped `FP4xFP4` step therefore needs a better packed-activation
  contract and a math core designed for it, not just a storage-format swap

### Contract Progress Update

That next packed-activation contract is now partially landed on the active
runtime side:

- routed prefill launch-plan rows are expert-major and padded to the routed
  token tile (`16`) rather than separated by standalone `128`-row expert gaps
- routed reduction now correctly maps `selection_to_sorted` through
  `sorted_to_permuted_indices`, which restored behavioral reuse equivalence
  after the padded-layout switch
- the static MoE workspace now carries a prefill `normalized_pack` source
  buffer plus grouped NVFP4 pack storage
- FC1 now packs normalized activations once and then permutes packed rows into
  grouped padded order, instead of gathering FP32 rows and repacking them
- the grouped packed-row helper preserves the source tensor scale and zeroes
  padded rows
- the active packed path now no longer spends time computing the dead
  per-expert activation-scale metadata that the current consumer does not use

What is still missing is the remaining TRT output-side contract:

- FC2 still starts from FP32 scratch plus a neutral grouped repack, not from a
  TRT-like `gemm1_output_scale` / activation-output-scale contract
- the next step is to move routed `gemm1` / `gemm2` onto the same grouped FP4
  activation contract end to end, not just on the FC1 source side

Focused validation is currently green:

- `device_nvfp4_matrix_test`
- `moe_launch_plan_device_test`
- `fused_moe_prefill_test`
- `multi_turn_prefix_reuse_test`

Focused TTFT on the design-center `prefix128 / tail4` case after moving FC1 to
per-expert input scales is:

- `cold_prefill_prefix128 = 125.352 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.328 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.612 ms`

The next FC2/output-scale step is now also landed in the working tree:

- grouped per-expert packing again honors expert tensor scales
- routed-down packed input now consumes those FC2 expert scales instead of
  treating FC2 input as a neutral matrix contract
- focused correctness remains green

Focused TTFT on the same gate with that FC2 expert-scale contract is:

- `cold_prefill_prefix128 = 126.679 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.423 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.766 ms`

The next TRT-style contract step is now landed too:

- the active packed prefill path no longer keeps `routed_up_scratch` as the
  FC1->FC2 boundary
- it now computes Gemm1/activation output scales directly from the packed FC1
  output and packs activated rows straight into the grouped FC2 input matrix
- `routed_up_scratch` remains only as a fallback-only path, not the active
  packed contract
- packed-path focused tests now run with `routed_up_scratch = nullptr`

Focused TTFT on the same gate after removing the FP32 boundary is:

- `cold_prefill_prefix128 = 126.959 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.447 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.684 ms`

Interpretation:

- the direct Gemm1-output contract is now correct and green
- this step is structurally important, but it is still roughly
  performance-neutral on the `prefix128 / tail4` gate
- the next remaining win has to come from stronger grouped math, not from more
  boundary cleanups

The FC1 contract then moved one more step toward TRT-LLM:

- the active packed routed FC1/FC2 path now executes from the planned CTA grid
  instead of the exact-task queue
- the active planned packed kernels derive token work from `cta_m_limits`
  with implicit `tileTokensDim` row starts, instead of consuming a separate
  `row_start + valid_rows` task contract
- the runtime file no longer carries a second packed routed consumer

Focused TTFT on the same gate after this FC1 alignment work is:

- `cold_prefill_prefix128 = 125.592 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.557 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.158 ms`

### Next Jump: Match TRT-LLM For The Routed Math Stage

Yes, this is the point where we should jump to the TRT-LLM execution shape for
this stage in one deliberate rewrite rather than keep iterating on the old row
kernel family.

The target stage shape is:

1. exact routed CTA metadata
2. grouped `PermuteGemm1`
3. TRT-style activation/output-scale production
4. grouped `Gemm2`
5. finalize

What "jump in one shot" means here:

- replace the active routed-up and routed-down math kernels together
- keep the existing device-only routing and launch-plan foundation
- keep one runtime path only
- preserve the new packed FC1/FC2 contracts we just landed
- stop carrying forward the old row-kernel family as an equal implementation

What still should not be "one shot":

- do not rewrite routing, shared experts, and Mamba in the same patch
- do not merge the new grouped math core before the focused correctness gates
  and `prefix128 / tail4` TTFT gate pass on the new stage

The implementation plan for this jump is:

1. Replace the routed FC1 consumer with a true grouped GEMM tile kernel that
   consumes the current planned CTA metadata directly, rather than one CTA
   externalizing output-row tiles.
2. Match TRT-LLM's handoff exactly at the stage boundary:
   `gemm1_output`, `gemm1_output_scale`, activation, packed FC2 input,
   `gemm2_output`.
3. Replace routed FC2 with the same grouped kernel family and CTA map, rather
   than keeping a mixed FC1-new / FC2-old implementation.
4. Keep finalize unchanged initially so the only moving part is the routed math
   stage.
5. After the routed stage wins, move shared experts onto the same stronger math
   family.

The acceptance criteria for this jump are:

- `device_nvfp4_matrix_test`
- `moe_launch_plan_device_test`
- `fused_moe_prefill_test`
- `multi_turn_prefix_reuse_test`
- `cold_prefill_prefix128` must improve versus the current `125.592 ms`
- no regression in behavioral reuse equivalence

The reason this is safe now is that the contract work is no longer the blocker.
The remaining gap versus TRT-LLM is the grouped math and CTA scheduling itself,
and that gap will not close through more incremental cleanup.

Progress on this jump:

- The active packed FC1/FC2 routed path now executes from the planned CTA grid
  rather than the old exact-task queue.
- The launch plan now pads expert batches to `tileTokensDim=16`, which makes
  `cta_m_limits` sufficient to describe active token rows the TRT way.
- The active planned packed kernels now consume `cta_expert_ids + cta_m_limits`
  as their token-batch contract instead of a second task-local
  `row_start + valid_rows` map.
- The active routed stage now follows TRT-LLM's BF16 x NVFP4 contract more
  closely:
  - launch-plan aliases now expose
    `permuted_idx_to_token_idx`,
    `total_num_padded_tokens`,
    `num_non_exiting_ctas`,
    `cta_idx_xy_to_batch_idx`, and
    `cta_idx_xy_to_mn_limit`
  - routed FC1 no longer pre-packs BF16 activations into an NVFP4 input matrix
    on the active path
  - the active path now does `BF16 grouped Gemm1 -> gemm1_output_scale ->
    packed Gemm2 input`, which matches TRT's BF16 routed entry much better than
    the earlier packed-FC1 experiment
- Focused correctness remains green:
  `device_nvfp4_matrix_test`, `moe_launch_plan_device_test`,
  `fused_moe_prefill_test`, `multi_turn_prefix_reuse_test`.
- Focused TTFT moved only slightly:
  `cold_prefill_prefix128 = 126.012 ms`,
  `cached_committed_head_prefix128_tail4 hot-prefix = 54.447 ms`,
  `cached_global_root_prefix128_tail4 hot-prefix = 54.616 ms`.

Interpretation:

- The routed launch contract is now close enough to TRT that it is unlikely to
  be the dominant remaining blocker.
- The BF16 routed FC1 entry is now also structurally aligned with TRT's BF16
  runner, so further FC1-input packing experiments should not remain on the
  active path.
- The next remaining jump is the grouped math core itself: replace the current
  task-driven WMMA row-tile consumer with a fuller TRT-like grouped GEMM
  consumer for routed FC1 and FC2.

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
- behavioral reuse equivalence is required for prefix-cache correctness
- no second runtime path should be introduced just to chase these wins

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

- [x] **2. Decide the routed-expert input format and packing strategy**
  The current runtime spends a large share of GPU time in:
  - `PackRowMajorFp32ToNvfp4Kernel`
  - `ComputeGlobalMaxAbsKernel`
  This step must decide whether the production path should:
  - eliminate runtime FP32→NVFP4 packing,
  - fuse it into a reduced-launch routed path,
  - or replace it with a different activation/input contract.
  Do not start grouped routed-expert work until this is explicit.

  **Decision (revised after TRT-LLM investigation)**: Keep FP4×FP4 on the
  tensor cores — that is what cuBLASLt, vLLM, and TRT-LLM all do. The
  problem is not that activations are quantized to NVFP4; the problem is
  that the current path quantizes **per expert** inside the per-expert host
  loop. The fix is:
  1. Quantize activations to NVFP4 **once per layer** (one kernel launch)
  2. Grouped FP4×FP4 GEMM for all experts in one launch using SM120
     tensor core MMA instructions
  TRT-LLM does exactly this: separate `fp4_quantize()` kernel, then
  grouped CUTLASS FP4×FP4 GEMM. An earlier scaffold kernel using scalar
  `Nvfp4RowMajorDot()` was removed because it bypassed tensor cores
  entirely — the production kernel must use FP4×FP4 MMA.

  Key files:
  `runtime/src/backend/fused_moe_prefill.cu`,
  `runtime/src/backend/expert_layer.cpp`,
  `runtime/src/backend/device_nvfp4_matrix.cu`,
  `runtime/src/api/single_token_forward_model.cpp`

- [ ] **3. Grouped FP4×FP4 tensor core MoE kernel**
  Write a single CUDA kernel that processes all routed experts in one launch
  using SM120 FP4×FP4→FP32 tensor core MMA instructions. The architecture:
  1. **Quantize activations to NVFP4 once** — one kernel converts the full
     BF16 normalized input to block-scaled FP4 + scales. (Matches TRT-LLM's
     `fp4_quantize()` step.)
  2. **Grouped GEMM** — one kernel launch for all experts. Each block
     handles one (expert, tile) pair. Both activations and weights are FP4.
     CUTLASS-style tiled GEMM with warp-level MMA instructions. Fuse ReLU²
     between up and down projections. Weighted scatter to output.
  3. Wire into `expert_layer.cpp` behind an env-var gate, compare against
     the cuBLASLt path for correctness.
  Exit criterion:
  - lower routed-path GPU time than the current per-expert cuBLASLt path
  - no regression in correctness on real Nano weights
  - FP4×FP4 on tensor cores, not scalar dot products
  Key files:
  `runtime/src/backend/fused_moe_grouped.cu` (new),
  `runtime/src/backend/expert_layer.cpp`,
  `runtime/src/backend/expert_routing_device.cu`
  Reference:
  `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/`,
  `third_party/TensorRT-LLM/cpp/tensorrt_llm/thop/fp4Quantize.cpp`

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
without regressing correctness (ctest green), decode throughput, or
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
| 0 | Freeze the post-attention baseline and root-cause profile | done | Historical baseline frozen; newer external race targets now live above and in `proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md` |
| 1 | Lock the revised optimization contract | done | One runtime path, device-only execution, behavioral reuse equivalence |
| 2 | Decide routed-expert input format and packing strategy | done | `FP4xFP4` on tensor cores; quantize activations once per layer; grouped GEMM for all experts |
| 3 | Grouped `FP4xFP4` tensor core MoE kernel | in progress | Device launch plan, padded row layout, and partial BF16 WMMA path exist; the remaining gap is moving the active routed path onto the intended packed-activation grouped-`FP4xFP4` contract |
| 4 | Shared-expert alignment | pending | Shared experts still distort short-prefix cold prefill and need the same stronger math family |
| 5 | Optimize Mamba prefill | pending | Still the second blocker at `prefix4096` after routed MoE |
| 6 | Re-profile and choose the next default workstream | pending | Final race phase only after routed MoE, shared, and long-prefix Mamba move materially |
