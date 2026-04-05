# Baseline Root Cause Analysis (SM120)

## Key Finding

**The MoE expert kernel fusion is not the primary bottleneck.** PagedAttentionDeviceFallback dominates at 98.7% of total GPU time.

## Nsight Systems GPU Kernel Time Breakdown

| Kernel | Time % | Total (s) | Calls | Median | Notes |
|--------|--------|-----------|-------|--------|-------|
| PagedAttentionDeviceFallback | 98.7% | 1,017s | 288 | 71ms | max 22s (!), multi-token prefill |
| MambaSsdPrefill | 0.6% | 5.8s | 1,955 | 22μs | |
| NVFP4 GEMM (cutlass3x SM120) | 0.2% | 2.5s | 218,504 | 11.5μs | expert up/down proj |
| PackRowMajorFp32ToNvfp4 | 0.1% | 1.0s | 216,281 | 4.5μs | input quantization |
| ComputeGlobalMaxAbs | 0.1% | 0.7s | 216,281 | 1μs | scale computation |
| ExpertScatter | <0.1% | 0.4s | 1,127 | 95μs | routing |
| MambaConvPrefill | <0.1% | 0.4s | 1,955 | 2.4μs | |
| GatherRows | <0.1% | 0.2s | 102,164 | 1.5μs | per-expert gather |
| ScatterAddWeightedRows | <0.1% | 0.2s | 102,164 | 1.3μs | per-expert scatter |
| Relu2InPlace | <0.1% | 0.1s | 109,248 | 0.8μs | activation |
| FusedAddRmsNormBf16 | <0.1% | 55ms | 4,529 | 7μs | layer norm |

## Host-side overhead

| API | Total (s) | Calls | Avg | Notes |
|-----|-----------|-------|-----|-------|
| cudaLaunchKernel | 3.0s | 1,442,648 | 2.1μs | launch overhead |
| cuLaunchKernelEx | 0.5s | 221,387 | 2.2μs | cuBLASLt launches |

## Implications

1. **PagedAttentionDeviceFallback is the root cause.** It's a naive device-fallback attention kernel for multi-token prefill that takes 22 seconds for a single 4096-token attention call. The kernel is not designed for efficient multi-token prefill.

2. **The MoE expert GEMMs are already fast.** The SM120 native cutlass3x NVFP4 GEMM takes only 11.5μs median per expert GEMM call. Even with 218,504 total calls, the aggregate is only 2.5s — a fraction of the attention cost.

3. **Kernel launch overhead is real but secondary.** 3.5s total for ~1.7M launches. Fusing expert kernels would recover some of this, but it's dwarfed by the attention problem.

4. **The FP32→NVFP4 packing overhead is notable.** 1.0s for PackRowMajorFp32ToNvfp4 + 0.7s for ComputeGlobalMaxAbs = 1.7s spent on runtime input quantization. Keeping inputs in BF16/NVFP4 format would eliminate this.

## Revised Priority

1. **Fix attention prefill** — replace PagedAttentionDeviceFallback with cuDNN or FlashAttention for multi-token prefill
2. **Eliminate FP32→NVFP4 runtime packing** (1.7s) — the expert layer converts FP32 normalized input to NVFP4 on every call
3. **Fuse expert kernels** — still valuable for launch overhead reduction, but now a tertiary optimization
