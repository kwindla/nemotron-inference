# Plan: Prefill Kernel Launch Optimization

Project directory: `./proj-2026-04-03-0318`

## Context
GPU batched prefill works but 32-token tail prefill costs 241ms instead of the ~50ms target. The bottleneck is kernel launch and allocation overhead in expert/MoE layers. Expert layers are responsible for ~97% of GPU dispatches per forward pass.

Target: reduce 32-token tail prefill from 241ms toward ~80ms by reducing expert layer overhead.

## Root Cause Analysis (corrected after review)

With 32 tokens and top_k=6 (Nano config: 128 experts, n_group=1, topk_group=1):
- Total routed selections: 32 × 6 = 192 token-expert pairs
- **Active experts: ~100** (not ~45 — birthday paradox with 128 experts and 192 independent selections)
- Per-expert current cost in `RunBatchedDirectMoeViaCublaslt` (`expert_layer.cpp:1024`):
  - `GatherRowsFp32()` → 1 kernel
  - `RunNvfp4RowMajorFp32SourceToDevice()` for up_proj → **multiple kernels** (DeviceNvfp4Matrix::Create + cudaMalloc, pack kernels for nibbles/block-scales/matmul-scales/tensor-scale, cuBLASLt call) — ~5 dispatches total
  - `Relu2InPlaceFp32()` → 1 kernel
  - `RunNvfp4RowMajorFp32SourceToDevice()` for down_proj → ~5 dispatches
  - `ScatterAddWeightedRowsFp32()` → 1 kernel
  - **Per-expert total: ~13 GPU dispatches** (not 5)
- Per expert layer: ~100 experts × 13 = **~1,300 dispatches** + routing + shared expert + residual
- 23 expert layers: **~30,000+ GPU dispatches** per forward pass

Additionally, the current path has **host synchronization overhead**:
- `expert_layer.cpp:985`: copies expert selection results D→H
- `BuildExpertRoutingTable` (`expert_layer.cpp:823`): builds offsets on CPU
- Copies indices/weights H→D for gather/scatter

## What to optimize (in priority order)

1. **NVFP4 pack buffer allocation** (~200 cudaMalloc+cudaFree per layer): Each `RunNvfp4RowMajorFp32SourceToDevice` allocates a fresh `DeviceNvfp4Matrix` via cudaMalloc. The decode path already reuses pre-allocated buffers (`expert_layer.cpp:1165`). Pre-allocating reusable pack buffers for the prefill path eliminates ~200 allocation cycles per layer.

2. **Global gather + contiguous per-expert GEMMs**: Replace per-expert gather kernels with one global gather into selection-order buffer. Per-expert GEMMs then read contiguous rows without additional gather. Final scatter uses a single pass with atomic adds (current scatter is non-atomic and only safe because experts are processed serially).

3. **Device-side routing**: Move routing table construction to GPU. Current path does D→H copy, CPU sorting, H→D copy. A GPU histogram + prefix-sum eliminates the host sync point.

## Constraints

- `cublasLtMatmulGrouped` is **not available** on CUDA 13.0 (this system). Batched cuBLASLt with pointer-array mode doesn't support NVFP4 block scaling. Per-expert cuBLASLt calls are unavoidable for now.
- NVFP4 activation-scale layout is swizzled per tile — expert subranges of a single packed buffer are not sliceable. Packing must remain per-expert, but the pack **buffer** can be reused.
- Expert weights are contiguous per-expert in monolithic storage (`monolithic_expert_weights.cu:188`), which is compatible with sorted-buffer addressing.
- The `token_count == 1` fused decode path must remain unchanged.
- Shared expert (shared_up_proj, shared_down_proj) remains outside routed optimization — adds ~5 dispatches per layer regardless.
- Weight residency: if `monolithic_resident` / `full_residency_enabled` is false, `ResolveRoutedExpertWeightViews` uploads weights per expert. This is a separate concern not addressed here.

## Steps

- [ ] **1. Profile and measure baseline**
  Instrument the expert layer batched path (`expert_layer.cpp:934-1148`) with CUDA events to measure: (a) device expert selection time, (b) D→H copy + CPU routing table build time, (c) per-expert GEMM loop total time (broken into pack + matmul + activation + scatter), (d) shared expert time, (e) residual add time. Run on a 32-token prefill and report the breakdown per expert layer. Also count actual active experts from the routing table to verify the ~100 estimate. This determines which optimization has the highest impact.
  Key files: `runtime/src/backend/expert_layer.cpp`

- [ ] **2. Pre-allocate reusable NVFP4 pack buffers**
  Create reusable `DeviceNvfp4Matrix` activation-pack buffers in the expert layer for the prefill path, sized to `max_expert_token_count` (which is at most `token_count` = 32 for 32-token prefill). Switch from `RunNvfp4RowMajorFp32SourceToDevice` (which allocates fresh each call) to `RunNvfp4RowMajorFp32AccumToDevice` (which reuses a pre-packed buffer, matching the decode fastpath pattern at `expert_layer.cpp:1165`). This eliminates ~200 cudaMalloc/cudaFree cycles per expert layer. The pack buffer needs `PackInto()` called per-expert to repack the activation, but the underlying device memory is reused.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/device_nvfp4_matrix.cu`

- [ ] **3. Global gather + atomic scatter**
  Replace per-expert `GatherRowsFp32` + `ScatterAddWeightedRowsFp32` with: (a) one global gather sorting all 192 token-expert pairs into a contiguous `[192, hidden_size]` buffer ordered by expert, (b) per-expert GEMMs read contiguous slices of this buffer (no per-expert gather kernel), (c) one global `ScatterAddWeightedRowsFp32` using atomicAdd to handle multiple experts writing to the same token output row. This reduces ~200 gather+scatter kernels to 2 total per layer. Requires pre-allocating scratch for `selection_count = token_count × top_k` rows.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/primitive_ops.cu`

- [ ] **4. Benchmark and iterate**
  Re-run TTFT benchmark. Target: 32-token tail prefill drops from ~241ms toward ~100ms. Profile again if above target — the remaining cost will be cuBLASLt per-expert matmul calls (~100 experts × 2 matmuls = 200 cuBLASLt calls per layer, irreducible without grouped GEMM). Save results.
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-03-0318/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Profile and measure baseline | pending | — | |
| 2 | Pre-allocate reusable NVFP4 pack buffers | pending | — | |
| 3 | Global gather + atomic scatter | pending | — | |
| 4 | Benchmark and iterate | pending | — | |
