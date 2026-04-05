# Post-Attention Root Cause Analysis (SM120)

Artifacts:
- `step9_runs/20260405T/post_attention_ttft.txt`
- `step9_runs/20260405T/post_attention_1024.nsys-rep`
- `step9_runs/20260405T/post_attention_1024_stats.txt`
- `sm120_baseline_ttft.txt`
- `baseline_root_cause.md`

## Key Finding

**Attention is no longer the bottleneck.** After the Nano multi-token attention
backend landed, the current `1024`-token cold-prefill profile shows attention
at only about `6%` of GPU kernel time. The dominant costs are now:

- Mamba prefill kernels: about `31%`
- routed MoE GEMMs plus runtime NVFP4 input packing/scaling: about `43%`
- additional routed MoE dispatch/finalize kernels: about `9%`

That means the old "fix attention first" conclusion is now closed out. The
next performance decisions should center on routed MoE/input packing and Mamba.

## TTFT Comparison

| Case | Old baseline | Current | Improvement |
|------|--------------|---------|-------------|
| `cold_prefill_prefix1024` | `5971.231 ms` | `440.312 ms` | `13.56x` faster |
| `cold_prefill_prefix4096` | `133038.284 ms` | `1369.694 ms` | `97.13x` faster |

Source artifacts:
- baseline: `sm120_baseline_ttft.txt`
- current: `step9_runs/20260405T/post_attention_ttft.txt`

## Fresh 1024-Token Kernel-Time Breakdown

The profile below is from `cold_prefill_prefix1024` on the current runtime.
Because the benchmark includes first-token decode, the decode-specialized
attention kernel appears as a small additional attention cost.

| Kernel group | Share | Notes |
|-------------|-------|-------|
| `MambaSsdPrefillFixedKernel` | `28.7%` | dominant single kernel |
| `MambaConvPrefillFixedKernel` + `MambaGatedGroupNormKernel` | `2.6%` | additional Mamba work |
| routed NVFP4 GEMM | `26.3%` | current expert up/down GEMMs |
| `PackRowMajorFp32ToNvfp4Kernel` | `10.7%` | runtime expert-input repacking |
| `ComputeGlobalMaxAbsKernel` | `6.2%` | runtime scaling pass for the same packing flow |
| routed dispatch/finalize (`ExpertScatter`, `GatherRows`, `ScatterAddWeightedRows`, `Relu2InPlace`) | `8.4%` | non-GEMM routed-expert work |
| `PagedAttentionNanoMultiTokenKernel` | `3.9%` | multi-token prefill attention |
| `PagedAttentionDecodeProductionKernel` | `2.1%` | first-token decode |

Approximate grouped shares:
- Mamba total: `31.3%`
- routed MoE plus runtime packing/scaling: `51.6%`
- attention total: `6.0%`

## Host/API Signals

The fresh `nsys` trace still shows substantial host-side launch and allocation
traffic:

- `cudaLaunchKernel`: `11.0%` of CUDA API time
- `cuLaunchKernelEx`: `1.7%`
- `cudaMalloc`: `6.2%`
- `cudaFree`: `7.1%`

This does not by itself prove the next optimization, but it does support the
case for reducing routed-expert launch count and avoiding unnecessary runtime
repacking/allocation churn.

## Decision

The old fused grouped MoE plan should **not** be resumed unchanged, because it
still assumes the pre-attention-fix world. But grouped routed-expert work is
again justified after the attention rewrite.

The revised order should be:

1. Rewrite `PLAN.md` around the post-attention profile rather than the old
   `6s -> 1-2s` TTFT target.
2. Make runtime FP32→NVFP4 input packing elimination or restructuring a
   first-class objective, not an afterthought.
3. Revisit grouped routed-MoE fusion as a genuine next-tier optimization.
4. Keep Mamba prefill optimization in scope, because it is now a co-primary
   bottleneck rather than background noise.

## Implication For The Old Plan

`PLAN.md` is no longer blocked on attention. It is blocked on being out of
date. The next useful action is to rewrite that plan against this new profile.
