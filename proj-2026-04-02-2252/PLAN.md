# Plan: NVFP4 Batched GEMM for Prefill (M > 1)

Project directory: `./proj-2026-04-02-2252`

## Context
The GPU Mamba2 prefill kernels from `proj-2026-04-02-2055` are implemented and wired in, but batched prefill can't run at speed because every NVFP4 linear op falls back to CPU reference for M > 1. The `UploadedLinearOp::Create()` at `linear_op.cpp:357-362` pre-allocates the activation pack buffer for M=1 only. The M > 1 path exists (`RunNvfp4RowMajorFp32SourceToDevice`, `nvfp4_gemm_runner.cpp:776`) and the packing infrastructure supports arbitrary M, but the end-to-end path is not working — 0% GPU utilization with 100% CPU for minutes on a 256-token prefill.

Target: NVFP4 GEMMs work correctly for M=1 (decode) AND M > 1 (prefill), so the full GPU prefill path delivers ~50ms for 32-token tail prefill.

## Current Failure Chain

When `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1` and M > 1:
1. `linear_op.cpp:484`: `BuildRuntimeGemmPlan(handle, cache, rows=256, ...)` is called with actual M
2. If plan builds: `linear_op.cpp:533`: `RunNvfp4RowMajorFp32SourceToDevice(handle, plan, activations, weight, output, pack_options)` is called
3. `nvfp4_gemm_runner.cpp:783`: `PackDeviceRowMajorFp32ToNvfp4(activations, pack_options)` packs M=256 rows on GPU
4. `nvfp4_gemm_runner.cpp:506-512`: `PrepareNvfp4MatmulCall` validates M/N/K dimensions against the plan
5. Algo caching at `nvfp4_gemm_runner.cpp:436-470` is per-(M,N,K) tuple — should find or build the right algo
6. If any step fails: `linear_op.cpp:581-589` falls back to `RunNvfp4RowMajorReferenceToDevice` (CPU)

The exact failure point needs diagnosis. Possible causes:
- `BuildRuntimeGemmPlan` returns nullopt for M=256 (cuBLASLt can't find an algo)
- Scale layout mismatch: `RuntimeNvfp4PackOptions` forces `kSwizzled128x4` (line 138), but the scale layout used during packing may not match what cuBLASLt expects for M > 1
- The plan succeeds but `cublasLtMatmulAlgoGetHeuristic` returns 0 results for NVFP4 with M=256

## Steps

- [ ] **1. Diagnose the exact failure point**
  Add targeted diagnostic output to `linear_op.cpp` Run() for the M > 1 NVFP4 path. For each GEMM where M > 1: log whether BuildRuntimeGemmPlan succeeded, whether RunNvfp4RowMajorFp32SourceToDevice succeeded, and if not, at which sub-step it failed (packing, PrepareNvfp4MatmulCall, ExecutePreparedNvfp4Matmul, or algo heuristic). Run a short probe: 16-token prefill of the first 2 layers with `LINEAR_DEVICE_FASTPATH=1` and `NEMOTRON_FORWARD_DEBUG=1`. Capture the output to identify whether the failure is in plan building, algo selection, dimension validation, or execution. This step produces a diagnosis, not a fix.
  Key files: `runtime/src/backend/linear_op.cpp`, `runtime/src/backend/nvfp4_gemm_runner.cpp`

- [ ] **2. Fix the M > 1 NVFP4 GEMM path**
  Based on the diagnosis from step 1, fix the failure. Likely changes: (a) ensure `BuildRuntimeGemmPlan` correctly handles M > 1 for NVFP4 kernel family, (b) ensure the scale layout in `RuntimeNvfp4PackOptions` is compatible with what cuBLASLt expects for the actual M, (c) if `cublasLtMatmulAlgoGetHeuristic` can't find an algo for NVFP4 at M=256, investigate whether the matmul descriptor or layout is misconfigured. The M=1 decode fastpath must remain unchanged and working at current 13ms/token speed. Add a focused test: run a single NVFP4 GEMM with M=32 and M=256, verify GPU execution succeeds and produces correct output vs CPU reference.
  Key files: `runtime/src/backend/linear_op.cpp`, `runtime/src/backend/nvfp4_gemm_runner.cpp`, `testing/backend/`

- [ ] **3. End-to-end benchmark**
  Re-run the TTFT benchmark with GPU prefill + NVFP4 batched GEMMs. Measure cold prefill at 32/256/1K/4K tokens and cached tail prefill at 32 tokens. Compare against the sequential baseline (from `proj-2026-04-02-1029/ttft_benchmark_results.log`). Target: 32-token tail prefill ~40-60ms (down from 1,175ms), 256-token cold ~70-140ms (down from 9,248ms). Save results.
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-02-2252/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Diagnose exact failure point | pending | — | |
| 2 | Fix M > 1 NVFP4 GEMM path | pending | — | |
| 3 | End-to-end benchmark | pending | — | |
