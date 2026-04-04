# Batched Expert GEMMs — Draft

## Problem

The cuBLASLt MoE path runs 12 sequential GEMM calls per MoE layer (6 up_proj + 6 down_proj for top_k=6 selected experts). With 23 MoE layers, that's 276 individual cuBLASLt launches per token. At ~2μs launch overhead each, that's ~0.55ms just in GEMM launch overhead, plus lost GPU utilization from sequential M=1 GEMMs that each use only a fraction of the GPU.

## Solution

Replace the 6 sequential up_proj GEMMs with 1 pointer-array batched cuBLASLt call, and the 6 sequential down_proj GEMMs with 1 pointer-array batched call. Also batch the shared expert GEMMs alongside if shapes match.

cuBLASLt supports `CUBLASLT_BATCH_MODE_POINTER_ARRAY` which allows different A/B/C pointers per batch element while sharing the same M/N/K/layout. This is exactly our case — all 6 up_proj GEMMs have the same shapes (M=1, N=1856, K=2688) but different B (weight) pointers and different C (output) pointers. The A (activation) pointer is the same for all 6 (same normalized input packed to NVFP4).

## Per MoE layer, current vs batched

Current (12 launches):
- Pack normalized → NVFP4 (1 launch)
- For each of 6 selected experts:
  - cuBLASLt up_proj GEMM (1 launch)
  - relu2 (1 launch)
  - Pack activated → NVFP4 (1 launch)
  - cuBLASLt down_proj GEMM (1 launch)
  - AccumulateScaled (1 launch)
- Shared up GEMM + relu2 + pack + shared down GEMM (4 launches)
- ResidualAdd (1 launch)
Total: ~40 launches per MoE layer

Batched (reduced launches):
- Pack normalized → NVFP4 (1 launch)
- Batched cuBLASLt up_proj ×6 (1 launch)
- relu2 ×6 (6 launches, or fuse into 1 batched relu2)
- Pack activated ×6 → NVFP4 (6 launches)
- Batched cuBLASLt down_proj ×6 (1 launch)
- AccumulateScaled ×6 (6 launches)
- Shared expert path (4 launches)
- ResidualAdd (1 launch)
Total: ~22 launches per MoE layer

Savings: ~18 launches per MoE layer × 23 layers = ~414 fewer launches per token.

## Steps

1. Add batched NVFP4 GEMM support to the GEMM runner
2. Wire batched GEMMs into the cuBLASLt MoE path
3. Benchmark
