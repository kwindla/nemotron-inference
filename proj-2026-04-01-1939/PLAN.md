# Plan: Monolithic Expert Tensor Refactor — Match vLLM Memory Layout

Project directory: `./proj-2026-04-01-1939`

## Goal

Eliminate per-expert device allocations and match vLLM's proven memory strategy: all expert weights in contiguous monolithic tensors, fully GPU-resident. This should make all 23 expert layers resident on the 32 GB RTX 5090 by replacing ~23,552 separate `cudaMalloc` calls (128 experts × 2 projections × 4 allocs × 23 layers) with 138 total (3 per projection × 2 projections × 23 layers).

Current: 21/23 expert layers resident using 128 separate `DeviceNvfp4Weight` objects per layer (1,024 cudaMalloc calls each). 2 layers fall back to per-token selected-only upload. Decode at 2,035 ms/token.

Target: 23/23 expert layers resident using 6 contiguous allocations per layer (packed + block_scales + tensor_scales for each of up_proj and down_proj). Zero per-token expert uploads.

## vLLM Reference (confirmed from source review)

What vLLM does that we should match:
- Expert weights stored as monolithic 3D tensors `[E, dim_out, dim_in/2]` (packed FP4) — one contiguous allocation per projection per layer
- All experts fully GPU-resident from model load; zero per-token weight movement
- Memory budget: weights first, KV cache sized to remaining VRAM

What vLLM does that we should NOT copy:
- FlashInfer/TRT-LLM layout transforms (block-scale interleave, w1/w3 reorder, alignment padding) — our fused decode kernel expects raw row-major layout, not the shuffled FlashInfer format
- Stride-based kernel indexing — our kernel uses a `FusedNvfp4WeightView` device array, which is fine; we just need the views to point into monolithic storage instead of separate objects

## Per-Expert Memory Math (Nano direct-MoE, from code review)

Per expert pair in the fused decode layout (raw row-major, no matmul scales):
- up_proj: packed `1856×2688/2 = 2,494,464`, block_scales `1856×(2688/16) = 311,808`, tensor_scale `4`
- down_proj: packed `2688×1856/2 = 2,494,464`, block_scales `2688×(1856/16) = 311,808`, tensor_scale `4`
- Total per pair: `5,612,552` bytes (5.35 MiB)

Per expert layer (128 experts): `718,406,656` bytes (685 MiB)
All 23 expert layers: `16,523,353,088` bytes (15.4 GiB)

Current per-expert path adds matmul_block_scales (~634 KB/pair), making each pair 5.96 MiB and each layer ~762 MiB. Monolithic layout saves ~10% by not allocating the cuBLASLt-specific matmul scales (the fused decode kernel doesn't use them).

## Steps

- [x] **1. Add MonolithicNvfp4ExpertWeights data structure**
  Create `runtime/include/nemotron/monolithic_expert_weights.h` and `runtime/src/backend/monolithic_expert_weights.cu` with a class that holds one projection's worth of expert weights for all experts in 3 contiguous device buffers:
  - `packed_data`: single cudaMalloc of `E × (output_rows × input_cols / 2)` bytes
  - `block_scales`: single cudaMalloc of `E × (output_rows × (input_cols / 16))` bytes
  - `tensor_scales`: single cudaMalloc of `E × sizeof(float)` bytes
  Provide:
  - `static Create(num_experts, output_rows, input_cols)` — 3 cudaMalloc calls total
  - `UploadExpert(expert_index, packed_src, packed_nbytes, block_scales_src, block_scales_nbytes, tensor_scale_src)` — cudaMemcpy into the correct offset
  - `GetView(expert_index) -> FusedNvfp4WeightView` — returns a view with pointers into the monolithic buffers at the correct expert offset
  - `BuildAllViews() -> std::vector<FusedNvfp4WeightView>` — builds views for all experts
  - `total_bytes()` and `num_experts()` accessors
  Important: views use `block_scales_data` (raw row-major), NOT `matmul_block_scales_data` (swizzled). The fused decode kernel reads raw block scales via `Nvfp4RowMajorDot` in `fused_decode_common.cuh`.
  Key files: `runtime/include/nemotron/monolithic_expert_weights.h` (new), `runtime/src/backend/monolithic_expert_weights.cu` (new), `runtime/CMakeLists.txt`

- [x] **2. Wire monolithic weights into ExpertLayerSlice::Create()**
  In `expert_layer.cpp`, when `NEMOTRON_EXPERT_MONOLITHIC=1` (default enabled):
  - Allocate one `MonolithicNvfp4ExpertWeights` for up_proj (E=128, N=1856, K=2688)
  - Allocate one `MonolithicNvfp4ExpertWeights` for down_proj (E=128, N=2688, K=1856)
  - For each routed expert descriptor, extract the raw packed data, block scales, and tensor scale from the `GemmDescriptor` and upload via `UploadExpert()`
  - Build the `FusedNvfp4WeightView` arrays and upload them to a device array (same `DeviceArray<FusedNvfp4WeightView>` pattern already used)
  - Store the monolithic objects + prebuilt device view arrays in `Impl`
  - This replaces the current per-expert `DeviceNvfp4Weight::Upload()` loop (1,024 cudaMalloc) with 6 cudaMalloc + 128 cudaMemcpy
  - Remove the VRAM-check skip logic — with monolithic allocation, either the whole layer fits or it doesn't
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/include/nemotron/expert_layer.h`

- [x] **3. Use prebuilt views in Run() — zero per-token uploads**
  When monolithic residency is active in `Run()`:
  - Skip the entire routed expert staging loop
  - Pass the prebuilt device-side `FusedNvfp4WeightView` arrays directly to `RunFusedMoeDirectDecode`
  - Set staging counters: `total_bytes_uploaded += 0`, `staging_elapsed_us += 0`, `total_staging_calls += 1`
  - Add `monolithic_layers` counter to `ExpertStagingCounters`
  When disabled, fall back to the current per-expert or selected-only upload path.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/include/nemotron/expert_staging_counters.h`, `runtime/src/backend/expert_staging_counters.cpp`

- [x] **4. Verify full residency, correctness, and benchmark**
  Pre-test: `nvidia-smi` clean GPU check, kill stale processes.
  Run:
  - Smoke test with `NEMOTRON_FORWARD_BUILD_MODEL=1 NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_FUSED_MAMBA_DECODE=1 NEMOTRON_FORWARD_FUSED_MOE_DECODE=1 NEMOTRON_FORWARD_DEBUG=1` — verify all 23 expert layers report `monolithic` residency
  - Benchmark `--mode=steady-state --decode-tokens 16` — verify `expert_staging_counters.total_bytes_uploaded=0` and measure decode ms/token
  - Compare artifact against 2,035 ms baseline
  Key files: benchmark scripts, `compare_artifacts.py`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | MonolithicNvfp4ExpertWeights data structure | done | 5531d00 | 3 cudaMalloc per projection vs 512 current |
| 2 | Wire into ExpertLayerSlice::Create() | done | c13f438 | 6 allocs per layer vs 1,024 current |
| 3 | Zero per-token uploads in Run() | done | — | monolithic path skips staging loop entirely |
| 4 | Full residency verification + benchmark | done | — | 23/23 monolithic, 0 bytes uploaded, 1805 ms/token (12.7x from baseline) |
