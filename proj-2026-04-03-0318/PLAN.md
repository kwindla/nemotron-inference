# Plan: Prefill Kernel Launch Optimization

Project directory: `./proj-2026-04-03-0318`

## Context
GPU batched prefill works but 32-token tail prefill costs 241ms instead of the ~50ms target. The bottleneck is kernel launch overhead, allocation overhead, and host synchronization in expert/MoE layers. Expert layers are responsible for ~97% of GPU dispatches per forward pass.

Target: reduce 32-token tail prefill from 241ms toward ~80ms by reducing expert layer overhead.

## Root Cause Analysis

Nano config (`single_token_forward_model.cpp:400-403`): **512 routed experts, 22 experts per token** (`experts_per_token=22`), `expert_n_group=1`, `expert_topk_group=1`.

With 32 tokens and experts_per_token=22:
- Total routed selections: 32 × 22 = **704 token-expert pairs**
- Active experts: with 512 experts and 704 selections, birthday paradox gives ~374 active experts (uniform) — actual count depends on router skew, likely 200-400
- Per-expert current cost in `RunBatchedDirectMoeViaCublaslt` (`expert_layer.cpp:1024`):
  - `GatherRowsFp32()` → 1 kernel
  - `RunNvfp4RowMajorFp32SourceToDevice()` for up_proj → **multiple kernels** (DeviceNvfp4Matrix::Create + cudaMalloc, pack kernels for nibbles/block-scales/matmul-scales/tensor-scale, cuBLASLt call) — ~5 dispatches total
  - `Relu2InPlaceFp32()` → 1 kernel
  - `RunNvfp4RowMajorFp32SourceToDevice()` for down_proj → ~5 dispatches
  - `ScatterAddWeightedRowsFp32()` → 1 kernel
  - **Per-expert total: ~13 GPU dispatches**
- Per expert layer: ~300 active experts × 13 = **~3,900 dispatches** + routing + shared expert + residual
- 23 expert layers: **~90,000+ GPU dispatches** per forward pass

## Host Synchronization Overhead

Each expert layer does **two device-host round-trips** during the batched path (`expert_layer.cpp:965-1007`):

1. **GPU → CPU** (`expert_layer.cpp:985-990`): Expert selection kernel writes `selected_indices[704]` and `selected_weights[704]` to device memory. These are copied to host via `CopyToHost()` — an implicit `cudaDeviceSynchronize()` + `cudaMemcpy D→H`. GPU goes idle while waiting.

2. **CPU routing table construction** (`expert_layer.cpp:992-1004`): `BuildExpertRoutingTable()` runs on CPU — a counting sort that histograms expert indices, prefix-sums to get offsets, then scatters token indices and weights into expert-sorted order. This is O(n_experts + selection_count) = O(512 + 704) = ~1,200 operations. Fast on CPU (~microseconds) but the **sync cost** of stopping the GPU is the real overhead.

3. **CPU → GPU** (`expert_layer.cpp:1006-1007`): Sorted `expert_token_indices[704]` and `expert_token_weights[704]` are uploaded back to device via `CopyFromHost()`.

4. **Memory allocation** (`expert_layer.cpp:1008-1022`): `DeviceTensorFp32::Create()` calls for `routed_output`, `row_scratch`, `expert_up_output`, `shared_up_output` — each does a `cudaMalloc`.

With 23 expert layers: **46 device-host sync points** + **~92 cudaMalloc calls** per forward pass.

### GPU replacement for BuildExpertRoutingTable

`BuildExpertRoutingTable` (`expert_layer.cpp:823-870`) is a counting sort:
1. **Histogram**: count selections per expert → `expert_counts[512]`
2. **Prefix-sum**: exclusive scan of counts → `expert_offsets[513]`
3. **Scatter**: for each selection, write `token_index` and `weight` to `expert_offsets[expert] + count++`

This maps directly to standard GPU primitives:
1. **GPU histogram**: one kernel, each thread atomically increments `expert_counts[selected_indices[i]]`
2. **GPU exclusive prefix-sum**: one kernel (or CUB `DeviceScan::ExclusiveSum` over 512 elements)
3. **GPU scatter**: one kernel, each thread writes its token index and weight to the sorted position

Total: 3 kernel launches instead of 2 sync points + CPU work. The `expert_offsets` array stays on device and is read directly by the per-expert GEMM loop (which only needs the offset and count per expert, read via `cudaMemcpy D→H` of the 513-element array once — or better, iterate on GPU).

## What to optimize (in priority order)

1. **Device-side routing** (eliminates 46 sync points): Move `BuildExpertRoutingTable` to GPU. Eliminates the D→H copy of selected_indices/weights, CPU sorting, and H→D copy of sorted results. The GPU kernel outputs `expert_offsets[513]` + sorted `expert_token_indices[704]` + sorted `expert_token_weights[704]` directly on device.

2. **NVFP4 pack buffer reuse** (eliminates ~600 cudaMalloc+cudaFree per layer): Pre-allocate reusable `DeviceNvfp4Matrix` activation-pack buffers. Switch to `RunNvfp4RowMajorFp32AccumToDevice` which reuses pre-packed buffers.

3. **Global gather + atomic scatter** (eliminates ~600 gather+scatter kernels per layer): One global gather into expert-sorted buffer, per-expert GEMMs on contiguous slices, one atomic scatter-add back.

4. **Scratch pre-allocation** (eliminates ~92 cudaMalloc per forward pass): Pre-allocate `routed_output`, `row_scratch`, `expert_up_output`, `shared_up_output` once at model creation, reuse per layer.

## Constraints

- `cublasLtMatmulGrouped` is **not available** on CUDA 13.0. Per-expert cuBLASLt calls are unavoidable for now.
- NVFP4 activation-scale layout is swizzled per tile — packing must remain per-expert, but the buffer can be reused.
- Expert weights are contiguous per-expert in monolithic storage (`monolithic_expert_weights.cu:188`).
- The `token_count == 1` fused decode path must remain unchanged.
- Shared expert remains outside routed optimization — adds ~5 dispatches per layer.
- Weight residency: if `monolithic_resident` / `full_residency_enabled` is false, per-expert weight uploads happen inside the loop. Not addressed here.

## Steps

- [ ] **1. Profile and measure baseline**
  Instrument the expert layer batched path (`expert_layer.cpp:934-1148`) with CUDA events to measure: (a) device expert selection time, (b) D→H copy + CPU routing table build + H→D copy time, (c) per-expert GEMM loop total time (broken into pack + matmul + activation + scatter), (d) shared expert time, (e) residual add time, (f) cudaMalloc time for scratch tensors. Run on a 32-token prefill and report the breakdown per expert layer. Also count actual active experts from the routing table. This determines which optimization has the highest impact.
  Key files: `runtime/src/backend/expert_layer.cpp`

- [ ] **2. Device-side routing table construction**
  Add a GPU kernel `BuildExpertRoutingTableDevice()` that replaces the CPU `BuildExpertRoutingTable` + the D→H copy of selected_indices/weights + H→D copy of sorted results. The kernel takes `selected_indices[selection_count]` and `selected_weights[selection_count]` already on device (from `RunDeviceExpertSelection`) and outputs `expert_offsets[n_experts+1]`, `sorted_token_indices[selection_count]`, `sorted_token_weights[selection_count]` on device. Implementation: (a) histogram kernel — atomicAdd per expert count, (b) exclusive prefix-sum — CUB `DeviceScan::ExclusiveSum` or a simple 512-element scan kernel, (c) scatter kernel — each thread writes its token index and weight to the sorted position using atomicAdd on a per-expert write counter. **Note:** the host-side per-expert GEMM loop still needs `expert_token_count` per expert to build tensor views and GEMM plans (`expert_layer.cpp:1025-1064`). This requires one blocking D→H copy of the `expert_offsets[513]` array (513 × 4 bytes = ~2 KB). This is much smaller than the current round-trip (704 ints + 704 floats = ~5.6 KB both ways + CPU sort), and eliminates the selected_indices/weights D→H + CPU sort + H→D copy. Moving the GEMM dispatch loop itself to GPU would require further work (not in this plan). Create `runtime/src/backend/expert_routing_device.cu` and `runtime/include/nemotron/expert_routing_device.h`.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/expert_routing_device.cu` (new)

- [ ] **3. Pre-allocate scratch and NVFP4 pack buffers**
  Pre-allocate reusable tensors in the expert layer at model creation time: `routed_output[max_tokens, hidden_size]`, `row_scratch[selection_count, hidden_size]`, `expert_up_output[selection_count, intermediate_size]`, `shared_up_output[max_tokens, shared_intermediate_size]`. This eliminates ~92 cudaMalloc per forward pass. For NVFP4 pack buffer reuse: `DeviceNvfp4Matrix::PackInto` rejects row mismatches (`device_nvfp4_matrix.cu:375`), and the GEMM runner requires `activations.rows == plan.m == output.rows` (`nvfp4_gemm_runner.cpp:503`). A single max-sized buffer cannot serve varying `expert_token_count`. Instead, cache a small set of `DeviceNvfp4Matrix` objects by M (the distinct expert_token_counts are typically 1-4 for 32 tokens across 512 experts), or add a `PackIntoPrefix(rows)` API that packs only the first `rows` of a larger buffer. Either approach eliminates the ~600 cudaMalloc/cudaFree per layer while respecting the runner's dimension checks.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/include/nemotron/expert_layer.h`, `runtime/src/backend/device_nvfp4_matrix.cu`

- [ ] **4. Global gather + atomic scatter**
  Replace per-expert `GatherRowsFp32` + `ScatterAddWeightedRowsFp32` with: (a) one global gather of all 704 token-expert pairs into a contiguous `[704, hidden_size]` buffer ordered by expert (using the device-side sorted indices from step 2), (b) per-expert GEMMs read contiguous slices via `DeviceTensorFp32::CreateView`, (c) one global weighted scatter-add back to `routed_output[token_count, hidden_size]`. **Scatter correctness:** the current per-expert scatter uses non-atomic `+=` (`primitive_ops.cu:72,92`), safe because experts run serially. A single global scatter has cross-expert write conflicts on the same token row (a token routed to 22 experts has 22 concurrent writes). Use FP32 `atomicAdd` which is functionally correct but non-deterministic in accumulation order — document this and test with tolerance for small numerical drift vs the per-expert sequential baseline. Alternatively, a token-wise segmented reduction kernel avoids atomics entirely but is more complex. Start with atomicAdd for simplicity. This reduces ~600 gather+scatter kernels to 2 per layer.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/primitive_ops.cu`

- [ ] **5. Benchmark and iterate**
  Re-run TTFT benchmark. Target: 32-token tail prefill drops from ~241ms toward ~80-100ms. Profile again if above target — the remaining irreducible cost is ~300 active experts × 2 cuBLASLt matmul calls = ~600 cuBLASLt dispatches per layer × 23 layers = ~13,800 cuBLASLt calls. At ~0.01ms per cuBLASLt call (lower overhead than full kernel launch), that's ~138ms floor. If we hit this floor, the next step would be expert pruning, GEMM fusion, or a custom fused MoE kernel. Save results.
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-03-0318/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Profile and measure baseline | pending | — | |
| 2 | Device-side routing table construction | pending | — | |
| 3 | Pre-allocate scratch and NVFP4 pack buffers | pending | — | |
| 4 | Global gather + atomic scatter | pending | — | |
| 5 | Benchmark and iterate | pending | — | |
