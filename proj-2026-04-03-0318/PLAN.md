# Plan: Prefill Kernel Launch Optimization

Project directory: `./proj-2026-04-03-0318`

## Context
GPU batched prefill works but 32-token tail prefill costs 241ms instead of the earlier ~50ms stretch target. The bottleneck is kernel launch overhead, allocation overhead, and host synchronization in expert/MoE layers. Expert layers are responsible for ~97% of GPU dispatches per forward pass.

Working target: test whether 32-token tail prefill can move from 241ms toward ~80-100ms by reducing expert-layer overhead. Treat this as a hypothesis, not an expected floor: RTX 5090 / SM120 still has to execute a large number of tiny-M NVFP4 GEMMs, which may leave the post-cleanup floor materially higher.

## Root Cause Analysis

Nano config (`single_token_forward_model.cpp:434-439`): **128 routed experts, 6 experts per token** (`experts_per_token=6`), `expert_n_group=1`, `expert_topk_group=1`.

With 32 tokens and experts_per_token=6:
- Total routed selections: 32 × 6 = **192 token-expert pairs**
- Active experts: with 128 experts and 192 selections, uniform occupancy gives **~100 active experts** in expectation — actual count depends on router skew, likely ~70-110
- Per-expert current cost in `RunBatchedDirectMoeViaCublaslt` (`expert_layer.cpp:1024`):
  - `GatherRowsFp32()` → 1 kernel
  - `RunNvfp4RowMajorFp32SourceToDevice()` for up_proj → **multiple kernels** (DeviceNvfp4Matrix::Create + cudaMalloc, pack kernels for nibbles/block-scales/matmul-scales/tensor-scale, cuBLASLt call) — ~5 dispatches total
  - `Relu2InPlaceFp32()` → 1 kernel
  - `RunNvfp4RowMajorFp32SourceToDevice()` for down_proj → ~5 dispatches
  - `ScatterAddWeightedRowsFp32()` → 1 kernel
  - **Per-expert total: ~13 GPU dispatches**
- Per expert layer: ~100 active experts × 13 = **~1,300 dispatches** + routing + shared expert + residual
- 23 expert layers: **~30,000+ GPU dispatches** per forward pass

## Host Synchronization Overhead

Each expert layer does **two device-host round-trips** during the batched path (`expert_layer.cpp:965-1007`):

1. **GPU → CPU** (`expert_layer.cpp:985-990`): Expert selection kernel writes `selected_indices[192]` and `selected_weights[192]` to device memory. These are copied to host via `CopyToHost()` — an implicit `cudaDeviceSynchronize()` + `cudaMemcpy D→H`. GPU goes idle while waiting.

2. **CPU routing table construction** (`expert_layer.cpp:992-1004`): `BuildExpertRoutingTable()` runs on CPU — a counting sort that histograms expert indices, prefix-sums to get offsets, then scatters token indices and weights into expert-sorted order. This is O(n_experts + selection_count) = O(128 + 192) = ~320 operations. Fast on CPU (~microseconds) but the **sync cost** of stopping the GPU is the real overhead.

3. **CPU → GPU** (`expert_layer.cpp:1006-1007`): Sorted `expert_token_indices[192]` and `expert_token_weights[192]` are uploaded back to device via `CopyFromHost()`.

4. **Memory allocation** (`expert_layer.cpp:1008-1022`): `DeviceTensorFp32::Create()` calls for `routed_output`, `row_scratch`, `expert_up_output`, `shared_up_output` — each does a `cudaMalloc`.

With 23 expert layers: **46 device-host sync points** + **~92 cudaMalloc calls** per forward pass.

### GPU replacement for BuildExpertRoutingTable

`BuildExpertRoutingTable` (`expert_layer.cpp:823-870`) is a counting sort:
1. **Histogram**: count selections per expert → `expert_counts[128]`
2. **Prefix-sum**: exclusive scan of counts → `expert_offsets[129]`
3. **Scatter**: for each selection, write `token_index` and `weight` to `expert_offsets[expert] + count++`

This maps directly to standard GPU primitives:
1. **GPU histogram**: one kernel, each thread atomically increments `expert_counts[selected_indices[i]]`
2. **GPU exclusive prefix-sum**: one kernel (or CUB `DeviceScan::ExclusiveSum` over 128 elements)
3. **GPU scatter**: one kernel, each thread writes its token index and weight to the sorted position

Total: 3 kernel launches instead of 2 sync points + CPU work. The `expert_offsets` array stays on device and is read directly by the per-expert GEMM loop (which only needs the offset and count per expert, read via `cudaMemcpy D→H` of the 129-element array once — or better, iterate on GPU).

## What to optimize (in priority order)

1. **Measure the real routed shape distribution**: collect per-layer timing and actual expert-row histograms for Nano 32-token tail prefill. The owned kernel depends on the true `M` distribution, not the uniform expectation.

2. **Build device-side routing for the fused path**: move routing compaction fully onto the GPU and produce the metadata the persistent kernel actually needs: `expert_offsets`, `active_expert_ids`, routed token indices/weights, and `M`-bucket metadata. Do not optimize the host cuBLASLt loop as an intermediate destination.

3. **Add runtime plumbing and fallback**: wire a new prefill-only fused path into `ExpertLayerSlice::Run()` with runtime capability checks, Nano-shape guards, and a safe fallback to the existing batched path.

4. **Implement the SM120-specific persistent routed-expert kernel family**: make this the mainline optimization target. The kernel must be expert-major, persistent, and bucketed for Nano's tiny-`M` expert populations.

5. **Benchmark and tune the fused path**: only after the fused path is correct should we spend time tuning bucket boundaries, CTA shapes, shared-memory staging, and accumulation strategy.

Optional side probe: `cuDNN FE MoE grouped matmul` is still worth checking, but it is no longer on the critical path if we are explicitly owning the kernel.

## Constraints

- Experimental cuBLASLt grouped GEMM arrived in CUDA 13.1+, but the documented support does **not** cover CC 12.0 block-scaled NVFP4. Pointer-array grouped mode is also tensorwide-scaling oriented, so it is not a drop-in replacement for this path.
- Expert weights are contiguous per-expert in monolithic storage (`monolithic_expert_weights.cu:188`). The owned fused path should consume those monolithic views directly rather than introducing a new weight layout.
- The `token_count == 1` fused decode path must remain unchanged.
- Shared expert should start as a companion fused kernel, not part of the first routed-kernel milestone.
- Weight residency matters: if `monolithic_resident` / `full_residency_enabled` is false, per-expert uploads still happen inside the loop. The first fused prefill milestone should target resident monolithic weights only.
- Pack-buffer reuse, global gather/scatter, and scratch pre-allocation are no longer mainline milestones. They remain fallback ideas for the existing cuBLASLt path, but the owned-kernel branch should not depend on them.
- The fused prefill path must inherit the same high-level compatibility envelope as `fused_direct_moe_supported` in `expert_layer.cpp`: direct-MLP topology, no latent projection, NVFP4 routed up/down experts, and NVFP4 shared up/down experts. Do not describe the runtime gate as "Nano shape + SM120" only.
- The owned-kernel branch still needs reusable device scratch. Routing metadata, active-expert lists, bucket offsets, queue state, and any per-layer fused scratch should live on `ExpertLayerSlice::Impl` and be reused across runs rather than being allocated per prefill.

## RTX 5090 / SM120-specific constraints

- GeForce RTX 5090 is **SM120 / compute capability 12.0**, i.e. consumer Blackwell, not datacenter SM100 Blackwell.
- The SM120 narrow-precision Tensor Core path is extended `mma.sync` FP4/FP6 support. Do **not** assume `tcgen05`, TMEM, or SM100-specific CUTLASS kernels are portable.
- Compute capability 12.x limits relevant to any future custom kernel: **128 KB shared memory per SM**, **99 KB shared memory per block**, **48 resident warps per SM**, **1536 resident threads per SM**, **32 resident blocks per SM**, **64K 32-bit registers per SM**, **255 registers per thread**.
- For Blackwell GeForce FP4/NVFP4 via cuBLASLt, keep the documented contract: **TN layout**, `CUBLAS_COMPUTE_32F`, `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, Tensor Core-friendly **16-byte-aligned** pointers/dimensions, and **16-byte-aligned** scaling-factor bases with tiled/swizzled scaling layouts.
- CUTLASS SM120 NVFP4 kernels are documented around large TN tiles (for example `128x128x128`), so this workload's many tiny-`M` routed experts are an inherently awkward fit even after host-side cleanup.
- For Nano's `hidden=2688` and routed intermediate `1856`, keeping fp32 input rows and fp32 intermediate rows live across both routed matmuls costs `M * (2688 + 1856) * 4` bytes. That is ~72.7 KB at `M=4`, ~90.9 KB at `M=5`, and ~109.1 KB at `M=6` before extra staging. So `M>=5` cannot assume a single-CTA "keep everything in shared memory" design under the 99 KB per-block limit; larger buckets need tiled or streamed staging.

## Steps

- [x] **1. Profile and measure baseline**
  Instrument the expert layer batched path (`expert_layer.cpp:934-1148`) with CUDA events to measure: (a) device expert selection time, (b) D→H copy + CPU routing table build + H→D copy time, (c) per-expert GEMM loop total time, (d) shared expert time, (e) residual add time, (f) allocation time. Record the full `expert_token_count` histogram per layer, not just the average active-expert count, because the fused kernel design depends on the actual `M` buckets that occur in practice. Use dedicated telemetry or a debug dump mode that does **not** route through `ExpertLayerRunTrace`, since the current GPU fast paths are gated on `trace == nullptr`. Save at least one representative 32-token tail-prefill routing histogram and timing breakdown for Nano.
  Key files: `runtime/src/backend/expert_layer.cpp`

- [ ] **2. Build device-side routing compaction for the fused path**
  Add a GPU routing stage that replaces the CPU `BuildExpertRoutingTable` path for prefill and emits the metadata the owned kernel actually consumes: `expert_offsets[n_experts+1]`, `expert_token_indices[selection_count]`, `expert_token_weights[selection_count]`, `active_expert_ids[active_expert_count]`, and `bucket_offsets` or equivalent metadata for the chosen `M` buckets. This should stay entirely on device; do **not** optimize around copying `expert_offsets` back to the host for the current cuBLASLt loop. Allocate these routing outputs and the persistent work-queue buffers once on `ExpertLayerSlice::Impl` and reuse them across runs. Create `runtime/src/backend/expert_routing_device.cu` and `runtime/include/nemotron/expert_routing_device.h`.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/expert_routing_device.cu` (new)

- [ ] **3. Add prefill fused-path plumbing and runtime gating**
  Introduce a new prefill-only entry point, for example `RunFusedMoePrefill()` in `runtime/src/backend/fused_moe_prefill.cu` with a header in `runtime/include/nemotron/fused_moe_prefill.h`. Integrate it in `ExpertLayerSlice::Run()` next to the decode fast path in `expert_layer.cpp`, but gate it with **runtime** checks: `token_count > 1`, direct-MLP topology, no latent projection, NVFP4 routed up/down experts, NVFP4 shared up/down experts, resident monolithic routed weights, Nano-compatible dimensions, and device capability sufficient for SM120 NVFP4 kernels. Do not use `__CUDA_ARCH__` in the host dispatch path. Fall back to `RunBatchedDirectMoeViaCublaslt()` when the fused path is unavailable.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/fused_moe_prefill.cu` (new), `runtime/include/nemotron/fused_moe_prefill.h` (new)

- [ ] **4. Implement the routed persistent kernel family**
  Make the owned routed path the mainline optimization target. The kernel should be expert-major, persistent, and fed by the routing metadata from step 2. Start with correctness-first bucket specializations for the real Nano `M` distribution:

  **Integration boundary**
  - Mirror `FusedMoeDirectLayerParams` from `fused_moe_decode.h`, but extend it with device routing buffers, active-expert lists, bucket metadata, and monolithic routed-weight views for all experts.
  - Reuse the existing `RunDeviceExpertSelection()` path for router-side work; do not reimplement selection unless profiling proves it necessary.
  - Keep debug and trace modes on the old path until the fused path is numerically stable.

  **Kernel decomposition**
  - Phase A: keep router softmax / top-k selection on device using the existing `RunDeviceExpertSelection()` path.
  - Phase B: build an expert-major routing table fully on device: `expert_offsets[129]`, `expert_token_indices[192]`, `expert_token_weights[192]`, `active_expert_ids[<=128]`, and bucket metadata for the chosen `M` ranges.
  - Phase C: launch a persistent routed-expert kernel over the active experts. Each CTA or warpgroup repeatedly pops work from a global queue, stages an expert tile from `normalized`, runs the routed `up_proj`, applies `Relu2`, runs `down_proj`, multiplies by route weights, and accumulates into `routed_output[token, hidden]`.
  - Phase D: run the shared expert in a companion fused kernel first; only fuse it into the routed kernel after the routed path is stable. The shared expert has no routing conflicts and is easier to validate independently.
  - Phase E: finish with residual add in-kernel if register/shared-memory pressure allows; otherwise keep the existing residual epilogue.

  **Scheduling strategy**
  - Schedule work expert-major, not token-major. Nano has only `192` routed rows per layer and usually `~70-110` active experts, so the average expert has `M ~= 1-3`. The kernel must therefore optimize for many tiny expert batches, not for large batched GEMMs.
  - Start with `M=1`, `M=2`, and `M=3-4` buckets. Larger `M` values should use a streamed/tiled variant instead of assuming all fp32 row state fits in shared memory.
  - Keep a global work queue of expert tiles so CTAs that finish a 1-row expert immediately claim more work. Do not bind one CTA permanently to one expert.
  - Start with a simple ownership rule for output accumulation: one CTA owns one `(expert, token-row tile)` at a time and uses FP32 `atomicAdd` into `routed_output`. If the atomics become a visible bottleneck, upgrade to a token-major segmented reduction epilogue after the fused path is numerically stable.
  - The fused path must define behavior for every observed `M`, not just the common buckets. The minimum acceptable first milestone is: native fused handling for `M=1`, `M=2`, and `M=3-4`, plus either (a) a streamed/tiled fused fallback for `M>=5`, or (b) explicit per-expert fallback of those rare larger-`M` experts to the existing batched path. Do not ship a path that silently assumes `M<=4`.

  **SM120 kernel design**
  - Do not port the scalar `Nvfp4RowMajorDot` loop from `fused_moe_decode.cu` to prefill. That is structurally correct but not the performance target. The new routed kernel should use SM120 NVFP4 Tensor Core microkernels or CUTLASS/CuTe building blocks specialized for consumer Blackwell `mma.sync` FP4/NVFP4 instructions.
  - Match the documented GeForce NVFP4 contract: TN-layout weights, FP32 accumulation, 16-byte-aligned data/scale pointers, and the existing swizzled/block-scaled weight layouts already produced by `monolithic_expert_weights.cu`.
  - Avoid a design that materializes a global `[192, hidden]` gather buffer or global routed-intermediate buffer as the steady-state path.
  - Expect multiple kernel variants. A dedicated `M=1` or `M=2` kernel can be materially different from an `M=3-4` kernel because register pressure, staging strategy, and accumulation ownership are different at Nano's routing scale.

  **Implementation order**
  1. Land the API surface and feature flag, reusing the existing device expert selection and monolithic weight views.
  2. Allocate reusable fused-path scratch on `ExpertLayerSlice::Impl`: routing outputs, active-expert arrays, bucket metadata, work queues, and any per-layer fused scratch needed by the new kernels.
  3. Implement Phase B routing compaction with explicit debug dumps so expert ordering and weights can be validated against the current CPU-sorted path.
  4. Implement a correctness-first fused routed kernel for `M=1`.
  5. Extend to `M=2` and `M=3-4`, and add either a streamed/tiled `M>=5` kernel or explicit fallback of those experts to the existing batched path.
  6. Replace scalar dot-product code with SM120 Tensor Core microkernels for the stabilized buckets.
  7. Add the companion shared-expert kernel and integrate residual handling.

  **Success criteria**
  - Reduce routed expert dispatches from `~1,300` per layer to a small fixed count, ideally: routing build + routed persistent launch(es) + shared expert + residual.
  - Preserve numerical agreement with the current batched path to within explicit FP32 accumulation tolerances on full 32-token Nano prefills.
  - Demonstrate that the fused path covers the full observed runtime `M` histogram for Nano, including the tail of larger-`M` experts.
  - Demonstrate that the fused path beats the current batched cuBLASLt baseline on RTX 5090 for the real Nano routed shape distribution, not just on synthetic large-`M` cases.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/fused_moe_decode.cu`, `runtime/src/backend/monolithic_expert_weights.cu`, `runtime/src/backend/fused_moe_prefill.cu` (new), `runtime/include/nemotron/fused_moe_prefill.h` (new)

- [ ] **5. Benchmark and tune the fused path**
  Re-run TTFT and the expert-layer micro-breakdown against the new fused path. Tune bucket boundaries, CTA sizes, work-queue granularity, shared-memory staging, and accumulation strategy only after the first fused implementation is correct. If the fused path still misses the working target, decide based on measurement whether to: (a) improve larger-`M` streamed variants, (b) reduce atomic pressure with a reduction epilogue, or (c) revisit cuDNN FE as a side experiment.
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-03-0318/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Profile and measure baseline | done | — | GEMM loop 92-97% of cost; M=4+ dominates histogram; D→H sync negligible |
| 2 | Build device-side routing compaction for the fused path | pending | — | |
| 3 | Add prefill fused-path plumbing and runtime gating | pending | — | |
| 4 | Implement the routed persistent kernel family | pending | — | |
| 5 | Benchmark and tune the fused path | pending | — | |
