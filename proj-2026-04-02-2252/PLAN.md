# Plan: NVFP4 Batched GEMM for Prefill (M > 1)

Project directory: `./proj-2026-04-02-2252`

## Context
The GPU Mamba2 prefill kernels from `proj-2026-04-02-2055` are implemented and wired in, but batched prefill can't run at speed because every NVFP4 linear op falls back to CPU reference for M > 1. The `UploadedLinearOp::Create()` at `linear_op.cpp:357-362` pre-allocates the activation pack buffer for M=1 only. The M > 1 path exists (`RunNvfp4RowMajorFp32SourceToDevice`, `nvfp4_gemm_runner.cpp:776`) and the packing infrastructure supports arbitrary M, but the end-to-end path is not working — 0% GPU utilization with 100% CPU for minutes on a 256-token prefill.

Target: NVFP4 GEMMs work correctly for M=1 (decode) AND M > 1 (prefill), so the full GPU prefill path delivers ~50ms for 32-token tail prefill.

## Diagnosis (completed)

**The NVFP4 linear ops work fine for M > 1.** With `LINEAR_DEVICE_FASTPATH=1`, Mamba layer 0's in_proj and out_proj GEMMs complete successfully at M=256 on GPU.

**The real bottleneck is the expert layer's batched path** at `expert_layer.cpp:2263`. When `token_count > 1`, the expert layer loops per-token on CPU:
- Line 2263: `for (token_index = 0; token_index < token_count; ++token_index)`
- Line 2300: `RunNvfp4LinearHost(normalized_row_vec, 1, *pair.up_proj)` — CPU NVFP4 dequant+matmul per expert per token
- Line 2309: `RunNvfp4LinearHost(activated_up, 1, *pair.down_proj)` — same for down projection

With 256 tokens × top_k experts × 23 expert layers, this is tens of thousands of CPU-side matmuls. The fused MoE decode path (GPU) only works for `token_count == 1`.

## Steps

- [x] **1. Diagnose the exact failure point**
  Diagnosed via debug logging. Finding: NVFP4 linear ops work fine for M > 1 (Mamba layer 0 completes at M=256). The bottleneck is `expert_layer.cpp:2263-2309` — the batched expert path loops per-token on CPU, calling `RunNvfp4LinearHost` for every expert × every token. The fused GPU MoE path only works for `token_count == 1`.
  Key files: `runtime/src/backend/expert_layer.cpp`

- [x] **2. GPU expert layer batched prefill path**
  Add a GPU-dispatched path for expert layers when `token_count > 1`. The current per-token CPU loop (`expert_layer.cpp:2263-2340`) routes each token to top-k experts, then calls `RunNvfp4LinearHost` per expert. Replace with: (a) batch the router logits computation on GPU (existing), (b) run expert selection on GPU (existing `DeviceExpertSelectionKernel`), (c) for each selected expert, run the up_proj and down_proj NVFP4 GEMMs on GPU using the existing `UploadedLinearOp::Run` path which already works for M > 1 (verified in step 1). The key challenge is that different tokens route to different experts, so the GEMMs need per-expert grouping. The simplest approach: for each expert that has at least one token routed to it, gather the relevant token rows, run the expert GEMM, scatter results back. The M=1 fused decode path must remain unchanged.
  Key files: `runtime/src/backend/expert_layer.cpp`, `testing/backend/`

- [x] **3. End-to-end benchmark**
  Re-run the TTFT benchmark with GPU prefill + GPU expert dispatch. Measure cold prefill at 32/256/1K/4K tokens and cached tail prefill at 32 tokens. Compare against the sequential baseline (from `proj-2026-04-02-1029/ttft_benchmark_results.log`). Target: 32-token tail prefill ~40-60ms (down from 1,175ms), 256-token cold ~70-140ms (down from 9,248ms). Save results.
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-02-2252/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Diagnose exact failure point | done | baf4be6 | NVFP4 GEMMs work; expert layer CPU loop is bottleneck |
| 2 | GPU expert layer batched prefill path | done | 636c46e | |
| 3 | End-to-end benchmark | done | 4f8bc85 | 256-tok cold 13.6x, 256-tail TTFT 4.7x; 4K tail regresses — needs profiling |
