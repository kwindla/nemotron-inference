# Plan: Fused Grouped MoE Prefill Kernel

Project directory: `./proj-2026-04-05-0445`

## Goal

Replace the per-expert host-loop cuBLASLt MoE prefill path with a single hand-tuned CUDA kernel that processes all routed experts in one launch, fusing gather → NVFP4 GEMM → activation → GEMM → weighted scatter for the full token batch. Target: collapse the ~30 kernel launches per expert layer down to 1, bringing 1024-token cold TTFT from 6s toward 1-2s.

## Scope Guardrails

- **Target architecture**: GeForce RTX 5090, consumer Blackwell, `SM120`, CUDA 13.0. Treat this as a distinct target from datacenter Blackwell (`SM100/103`) and do not assume GB200/B200 kernel behavior, NVLink topology, TMA behavior, shared-memory budgets, or TRT-LLM-gen support transfer directly.
- **Target model/config only**: Nemotron 3 Nano NVFP4 routed MoE with `hidden_size=2688`, `routed_expert_intermediate_size=1856`, `shared_expert_intermediate_size=3712`, `n_routed_experts=128`, `top_k=6`, `activation=ReLU²`, `routed_scaling_factor=2.5`, `46` expert layers.
- **Target runtime only**: single-GPU, single-user, long multi-turn conversations, `<=4` concurrent requests, `64k` maximum context, no EP/TP, no expert remapping, no LoRA, no alternative activations, no support obligation for non-Nano shapes in the fast path.
- **Performance objective**: optimize the common local path, not the generalized backend. Fallbacks may remain for unsupported configs, but the specialized path should assume the dimensions above unless a later step proves broader support is essentially free.

## Context

After the prefill window optimization (proj-2026-04-05-0040), cold TTFT for 1024 tokens improved from 16.6s to 6.0s by eliminating window iteration overhead. The remaining 6s is dominated by per-expert kernel launch overhead: each expert layer currently launches 4 routing kernels + 5 kernels per active expert (gather, GEMM up, activation, GEMM down, scatter) = ~34 launches per layer × 46 layers ≈ 1,564 kernel launches for a single forward pass.

Both vLLM (FlashInfer CUTLASS) and TRT-LLM (CUTLASS grouped GEMM) solve this by processing all experts in a single fused kernel. We will build our own hand-tuned kernel following the same architecture, optimized for Nano's exact shapes.

## Nano MoE Dimensions

```
hidden_size:                      2688
routed_expert_intermediate_size:  1856
shared_expert_intermediate_size:  3712
n_routed_experts:                 128
experts_per_token (top_k):        6
activation:                       ReLU² (x > 0 ? x² : 0)
weight format:                    NVFP4 block-scaled
routed_scaling_factor:            2.5
expert_layers:                    46 out of 52
```

## Reference Implementations

- **Our decode kernel** (`fused_moe_decode.cu:159-286`): Hand-tuned single-token kernel that fuses all experts in one launch. Uses `Nvfp4RowMajorDot()` from `fused_decode_common.cuh` for inline NVFP4 GEMM. One block per token, all experts processed sequentially in shared memory. This is the pattern we extend to multi-token.

- **TRT-LLM** (`third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/`): Uses CUTLASS grouped GEMM with TMA warp specialization. Fuses activation in epilogue. Token-to-expert permutation is explicit. SM100+ path uses TMA for async data movement.

- **vLLM FlashInfer** (`third_party/vllm/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`): Single `flashinfer_cutlass_fused_moe()` call handles all experts. Routing is pre-computed, expert dispatch fused inside kernel.

All three converge on: routing separate → single grouped kernel for expert GEMM + activation + scatter.

For this repo specifically, the current prefill path is not just suffering from per-expert kernel launches. It also performs host-side control flow and device-to-host copies before the routed expert loop. Those costs must be re-measured on the fresh `SM120` build before the grouped kernel is treated as the sole remaining bottleneck.

## Current Kernel Launch Sequence (per expert layer, multi-token)

```
Phase 1: Routing (4 kernels)
  K1: ExpertHistogramKernel        — count tokens per expert
  K2: ExpertPrefixSumKernel        — compute expert offsets
  K3: ExpertScatterKernel           — reorder tokens by expert
  K4: ExpertCompactKernel           — build active expert list

Phase 2: Per-expert loop (5 × N_active kernels, N_active ≈ ~60-80 for 1024 tokens)
  For each active expert:
    K5: GatherRowsFp32              — gather routed tokens
    K6: cublasLtMatmul (up_proj)    — NVFP4 GEMM: [M, 2688] × [2688, 1856]
    K7: Relu2InPlaceFp32            — activation
    K8: cublasLtMatmul (down_proj)  — NVFP4 GEMM: [M, 1856] × [1856, 2688]
    K9: ScatterAddWeightedRowsFp32  — weighted scatter to output

Phase 3: Shared expert (4 kernels)
  K10: cublasLtMatmul (shared_up)   — [token_count, 2688] × [2688, 3712]
  K11: Relu2InPlaceFp32
  K12: cublasLtMatmul (shared_down)  — [token_count, 3712] × [3712, 2688]
  K13: ResidualAdd

Total: ~4 + 5×N_active + 4 ≈ 300-400+ kernel launches
```

## Target Kernel Launch Sequence

```
Phase 1: Routing (4 kernels — unchanged)
  K1-K4: same as today

Phase 2: Fused grouped routed expert (1 kernel)
  K5: FusedGroupedMoeKernel
    - Input: normalized[token_count, 2688], expert_offsets, sorted_token_indices/weights
    - Weights: all routed expert up/down NVFP4 views
    - Fuses: gather → up_proj GEMM → ReLU² → down_proj GEMM → weighted scatter
    - Output: routed_output[token_count, 2688]

Phase 3: Shared expert (keep cuBLASLt for now — only 3 launches, large batch)
  K6-K9: same as today (shared expert is always full-batch, cuBLASLt is efficient)

Total: ~4 + 1 + 4 = 9 kernel launches (down from 300-400+)
```

## Design

### Block/thread mapping

Each CUDA block processes one (expert, tile) pair. The grid is `(n_tiles_per_expert, n_active_experts)`. Each expert's token count comes from `expert_offsets`. Blocks for empty experts exit immediately.

A tile covers a contiguous range of tokens assigned to one expert. Tile size is a tuning parameter — start with 32 tokens per tile (matching warp count). Each block:
1. Loads the tile's token indices from `sorted_token_indices`
2. Gathers input rows from `normalized` using those indices
3. Computes up-projection GEMM against expert's NVFP4 up weights → intermediate
4. Applies ReLU² activation in-place
5. Computes down-projection GEMM against expert's NVFP4 down weights → hidden
6. Weighted-scatter-adds results to `routed_output` using `sorted_token_weights`

### GEMM strategy

For the NVFP4 GEMMs, adapt the `Nvfp4RowMajorDot()` pattern from `fused_decode_common.cuh` to handle multiple rows (tiles of tokens). Each thread computes partial dot products, using shared memory for intermediate accumulation. The up-projection produces intermediate values that stay in registers/shared memory — never written to global memory between up and down projections.

### Memory layout

- **Input**: `normalized[token_count, hidden_size]` in global memory (FP32)
- **Weights**: NVFP4 packed with block scales, indexed by expert_id
- **Intermediate**: stays in shared memory between up-proj and down-proj
- **Output**: `routed_output[token_count, hidden_size]` in global memory (FP32), accumulated via atomic adds (tokens from different experts may map to the same output row)

### Shared memory budget

Per block: `tile_size × max(intermediate_size, hidden_size) × sizeof(float)`
= 32 × 2688 × 4 = 344 KB — too large for shared memory.

On the local RTX 5090 environment, the practical opt-in shared-memory budget exposed to CUDA user code is about `101 KB` per block, not the much larger datacenter-class budgets discussed in some Blackwell materials. This means the implementation must assume aggressive register blocking / K-tiling and should not rely on a large per-block activation staging buffer.

Fallback: tile the K dimension of the GEMM and accumulate in registers, using shared memory only for the weight tile and partial results. Alternatively, reduce tile_size to 8 tokens: 8 × 2688 × 4 = 86 KB — feasible.

This is a tuning decision that will be resolved during implementation with measurement.

## Locked Decisions Before Coding

- `SM120` consumer Blackwell is the only optimization target for this project. `SM100/103` TRT-LLM-gen kernels are prior art, not implementation templates.
- Build and benchmark using dedicated `SM120` build directories, not a reused generic `build/` tree.
- The fast path is specialized to the exact Nano routed-expert dimensions listed above.
- Accumulation/output remains FP32 in the first implementation. Any lower-precision combine path requires separate measurement and correctness sign-off.
- Routing remains separate initially, but routing/finalize must be profiled as first-class costs. If grouped routed experts succeed, routing is the next optimization target.
- Any `atomicAdd`-based weighted scatter is acceptable only as a scaffold for correctness. The performance path should converge toward explicit permute/finalize reduction without atomics.
- `64k` refers to maximum retained conversation context, not necessarily a `64k` MoE prefill window. The runtime must be allowed to prefill in smaller windows if the full window would create unacceptable scratch or latency behavior.

## Workspace Constraint

The current request workspace model scales with `token_capacity * top_k` for both gather scratch and expert-up scratch. For Nano dimensions, the existing MoE prefill workspace is roughly:

- `1024` tokens: `~0.16 GiB`
- `4096` tokens: `~0.66 GiB`
- `8192` tokens: `~1.32 GiB`
- `16384` tokens: `~2.64 GiB`
- `32768` tokens: `~5.28 GiB`
- `65536` tokens: `~10.55 GiB`

This is too large to treat “full `64k` MoE prefill window” as the default optimization target on a `32 GiB` RTX 5090 when `<=4` requests may coexist. The implementation plan must therefore optimize the common long-conversation path with a bounded MoE prefill window, not assume that a full `64k` routed-expert window is the primary operating mode.

## Steps

- [x] **0. Reconfigure the build for RTX 5090 / SM120 and refresh the baseline**
  The current source tree defaults `CMAKE_CUDA_ARCHITECTURES` to `120`, but the existing `build/CMakeCache.txt` is still pinned to `75`. Before doing any MoE kernel work, wipe or move aside the stale build directory, create dedicated build dirs such as `build-sm120-relwithdebinfo` and `build-sm120-release`, reconfigure from scratch with `-DCMAKE_CUDA_ARCHITECTURES=120`, rebuild, and rerun the current correctness + TTFT baseline on the fresh `SM120` binaries. Capture the exact toolchain tuple in the project notes: GPU = RTX 5090 (consumer Blackwell, SM120), driver = 580.65.06, CUDA toolkit = 13.0, build arch = 120. Record whether the refreshed baseline still matches the claimed 6.0s / 16.6s TTFT figures; if it does not, update the targets in this plan before proceeding.
  Key files: `CMakeLists.txt`, `build/CMakeCache.txt`, `proj-2026-04-05-0445/final_profile.txt`

- [x] **1. Capture a fresh root-cause profile on the SM120 baseline**
  On the fresh `SM120` build, collect one Nsight Systems trace and one Nsight Compute profile for the existing prefill path on the representative workload (`1024`-token cold TTFT, real Nano weights). The purpose is to validate how much time is actually spent in: routing kernels, host-side device-to-host synchronization/copies, per-expert cuBLASLt launch overhead, shared expert kernels, and residual/finalize work. Save the findings in `proj-2026-04-05-0445/baseline_root_cause.md` and update this plan if grouped routed experts are not the dominant remaining bottleneck.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/expert_routing_device.cu`, `proj-2026-04-05-0445/baseline_root_cause.md`

- [ ] **2. Lock down the specialized implementation contract and runtime policy**
  Write down the exact fast-path contract before coding: Nano dimensions only, `SM120` only, `top_k=6`, `ReLU²`, FP32 accumulation/output, no EP/TP, no expert map, no LoRA, no alternate activation, no attempt to generalize to arbitrary hidden sizes. In the same step, lock down the runtime policy for long context on RTX 5090: decide the intended `moe_prefill_capacity_tokens` / effective prefill window for production, document why a full `64k` routed-expert window is not the primary optimization target on this GPU, and state the concurrency assumptions explicitly (`<=4` requests, but long-prefill requests may need serialization or bounded windows).
  Key files: `proj-2026-04-05-0445/PLAN.md`, `runtime/src/api/single_token_forward_model.cpp`, `runtime/src/backend/request_context.cpp`, `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`

- [ ] **3. Define the combine/finalize strategy before writing the kernel**
  Decide explicitly whether step 4's scaffold is permitted to use `atomicAdd` in the weighted scatter path, and define the exit criterion for removing it. Prior art in vLLM and TRT-LLM relies on explicit permute/finalize structures to avoid atomics in the optimized path. The plan should therefore treat atomics only as a temporary correctness scaffold and define the intended steady-state design: explicit permute/unpermute/finalize reduction, or an equally strong alternative proven by measurement on `SM120`.
  Key files: `proj-2026-04-05-0445/PLAN.md`, `runtime/src/backend/expert_routing_device.cu`, `runtime/src/backend/primitive_ops.cu`

- [ ] **4. Write the FusedGroupedMoeKernel scaffold with naive global-memory GEMM**
  Create `runtime/src/backend/fused_moe_grouped.cu` and `runtime/include/nemotron/fused_moe_grouped.h`. Implement the kernel with the block-per-(expert,tile) mapping described above. Start with a naive implementation that does global memory reads for both input gathering and weight access (no shared memory tiling). Use `Nvfp4RowMajorDot()` from `fused_decode_common.cuh` for the NVFP4 dot products, adapted for multi-row tiles. Fuse ReLU² between up and down projections. Include the weighted scatter-add to output. Add a `RunFusedGroupedMoe()` host function that takes routing outputs + weight views + scratch and launches the kernel. Add a standalone correctness test that compares the kernel output against the existing `RunFusedMoePrefill()` for a small synthetic case (e.g., 32 tokens, 4 experts, top-2). Register in CMakeLists.
  Key files: `runtime/src/backend/fused_moe_grouped.cu`, `runtime/include/nemotron/fused_moe_grouped.h`, `runtime/src/backend/fused_decode_common.cuh`, `testing/backend/fused_moe_grouped_test.cpp`

- [ ] **5. Validate against the existing prefill path on real Nano weights**
  Wire `RunFusedGroupedMoe()` as an alternative code path in `expert_layer.cpp`, selectable via `NEMOTRON_FORWARD_GROUPED_MOE_PREFILL=1`. Run it alongside the existing `RunFusedMoePrefill()` and compare outputs for real Nano expert layers with real prompts. The comparison should use the oracle test infrastructure — run `nano_save_prompt_oracle` with both paths and verify token-level agreement. Fix any precision or correctness issues. This step does NOT make it the default — it's gated behind the env var.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/fused_moe_grouped.cu`

- [ ] **6. Optimize the kernel: shared memory tiling and register blocking**
  Profile the naive kernel with Nsight Compute to identify the bottleneck (memory bandwidth, compute, launch overhead). Add shared memory tiling for weight access — load weight tiles into shared memory, compute partial GEMM products, accumulate in registers. Tune tile sizes for Nano's dimensions on SM 120. Keep the intermediate values (between up-proj and down-proj) in registers or shared memory to avoid the global memory round-trip. Measure the improvement and compare against the cuBLASLt baseline.
  Key files: `runtime/src/backend/fused_moe_grouped.cu`

- [ ] **7. Make the grouped kernel the default prefill path**
  Remove the env-var gate from step 5. Make `RunFusedGroupedMoe()` the default multi-token expert path in `ExpertLayerSlice::Run()` when the kernel supports the layer's configuration. Keep `RunFusedMoePrefill()` (the cuBLASLt path) as an explicit fallback for unsupported configurations or when gated by env var. Run the full verification suite: ctest 60/60, `verify_correctness.sh`, vLLM parity, decode regression.
  Key files: `runtime/src/backend/expert_layer.cpp`

- [ ] **8. Measure end-to-end TTFT improvement and capture final profile**
  Run the TTFT benchmark with default production settings. Compare against the 6.0s baseline (post window optimization) and the original 16.6s baseline. Run with `NEMOTRON_FORWARD_PREFILL_TRACE=1` to capture per-layer timing. Record profile as `proj-2026-04-05-0445/final_profile.txt`. Target: 1024-token cold TTFT under 2s.
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`

## Verification Policy

### Tier 1: every step
- `cmake --build build-sm120-relwithdebinfo -j$(nproc)`
- `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure`

### Tier 2: after steps 7 and 8
- `verify_correctness.sh` — local correctness gate
- `nano_fused_decode_bench` — decode must not regress
- `nano_prefix_cache_ttft_bench` — the primary metric
- vLLM parity if VRAM allows

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 0 | Reconfigure for RTX 5090 / SM120 and refresh the baseline | done | 3c22632 | GPU=RTX 5090 SM120, driver=580.65.06, CUDA=13.0, arch=120. Cold 1024 TTFT=5971ms (confirmed, matches SM75 baseline) |
| 1 | Capture a fresh root-cause profile on the SM120 baseline | done | PENDING | PagedAttentionDeviceFallback is 98.7% of GPU time. MoE GEMMs are only 0.2%. Plan needs revision. |
| 2 | Lock down the specialized implementation contract and runtime policy | pending | — | |
| 3 | Define the combine/finalize strategy before writing the kernel | pending | — | |
| 4 | Fused grouped kernel scaffold with naive GEMM | pending | — | |
| 5 | Validate against existing prefill on real Nano weights | pending | — | |
| 6 | Optimize: shared memory tiling and register blocking | pending | — | |
| 7 | Make grouped kernel the default prefill path | pending | — | |
| 8 | Measure end-to-end TTFT and capture final profile | pending | — | |
