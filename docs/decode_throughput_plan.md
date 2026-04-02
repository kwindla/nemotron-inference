# Decode Throughput Plan

## The Problem

Community implementations achieve ~14 tok/s (71ms/token) on DGX Spark with Nemotron 3 Super NVFP4. We achieve ~1.8 tok/s (548ms/token). The gap is 7.7x.

## Root Cause: Kernel Launch Count, Not GEMM Speed

Per-layer CUDA event profiling showed we are 24x above the memory bandwidth floor. We initially attributed this to kernel launch overhead between layers. But a detailed comparison with vLLM revealed the real issue is **within** each MoE layer:

| Operation | Our runtime | vLLM |
|---|---:|---:|
| Kernel launches per MoE layer | 10-12 | 1 |
| cudaMalloc per MoE layer | 4+ | 0 |
| cudaMemset per MoE layer | 2-6 | 0 |
| Host→Device copies per MoE layer | 1-2 | 0 |

Our MoE layer does this per token:
1. GatherExpertSelectionLookupsChecked — 1 kernel
2. PackDeviceRowMajorFp32ToNvfp4 — **4 kernels** (global max, tensor scale, pack, swizzle) + **4 cudaMalloc**
3. FusedRoutedUpProjPackedNvfp4SingleToken — 1 kernel
4. ComputeRowTensorScales — 1 kernel
5. PackScaledRelu2RowsToNvfp4 — 1 kernel + cudaMalloc
6. SwizzlePerRowBlockScales — 1 kernel
7. GatherExpertSelectionLookupsChecked (down) — 1 kernel
8. FusedRoutedDownProjWeightedPackedNvfp4SingleToken — 1 kernel

vLLM does: **1 fused kernel** (flashinfer.fused_moe.trtllm_fp4_block_scale_moe).

Across 40 MoE layers: we launch **400-480 kernels** with **160+ cudaMalloc calls**. vLLM launches **40 kernels** with **0 allocations**.

### CUTLASS Tensor Cores Don't Help At M=1

We validated this experimentally: replacing our custom scalar GEMM with CUTLASS SM121 tensor-core NVFP4 grouped GEMM produced **identical throughput** (548ms vs 551ms per token). At M=1, the GEMM is memory-bandwidth-bound. Both kernels read the same weight data at the same bandwidth. The arithmetic is negligible.

Tensor cores help at M>1 (batched execution) where the compute-to-memory ratio improves.

## The Fix: Reduce Kernel Launches Per MoE Layer

In priority order:

### 1. Eliminate Per-Call Allocation (biggest quick win)

Pre-allocate ALL intermediate buffers during model construction:
- NVFP4 packing buffers (packed data, block scales, tensor scale) — one set per layer in request context
- Swizzled scale buffers — pre-allocated alongside packing buffers
- Down-proj aligned activation buffers — pre-allocated at 256-byte aligned stride
- Expert gather output buffers — already done (DeviceBuffer in Impl)

This eliminates **160+ cudaMalloc calls** per token. Each cudaMalloc is ~10-100µs of host-GPU synchronization.

### 2. Fuse NVFP4 Activation Packing (4 kernels → 1)

Combine `ComputeGlobalMaxAbs` + `WriteTensorScale` + `PackRowMajorFp32ToNvfp4` + `SwizzleBlockScales` into a single kernel that:
- Computes the global max in shared memory
- Derives the tensor scale
- Packs to FP4 and swizzles scales in one pass

This eliminates 3 kernel launches per packing operation × 2 packing operations per MoE layer = **6 fewer launches per layer**, **240 fewer across 40 layers**.

### 3. Fuse relu2 + repack (3 kernels → 1)

Combine `ComputeRowTensorScales` + `PackScaledRelu2RowsToNvfp4` + `SwizzlePerRowBlockScales` into a single kernel that applies relu2, computes per-row scales, packs to FP4, and swizzles — all in one pass.

This eliminates **2 more launches per MoE layer**, **80 fewer across 40 layers**.

### 4. Pre-compute weight gather into persistent lookup tables

The `GatherExpertSelectionLookupsChecked` is called twice per MoE layer (once for up, once for down). If we store up and down expert pointers in a combined lookup table, one gather produces both sets of pointers.

This eliminates **1 launch per MoE layer**, **40 fewer across 40 layers**.

### Expected Progression

| Step | Kernels per MoE | Estimate |
|---|---:|---:|
| Current | 10-12 | 548ms/tok |
| + Pre-alloc (no cudaMalloc) | 10-12 | ~400ms/tok |
| + Fused packing (1 kernel) | 4-6 | ~300ms/tok |
| + Fused relu2+repack | 2-4 | ~200ms/tok |
| + Combined gather | 1-3 | ~150ms/tok |
| + CUDA graph capture | 1-3 (replayed) | ~80ms/tok |

### 5. CUDA Graph Capture (after kernel count is reduced)

Once the per-layer kernel count is down to 2-4, CUDA graph capture becomes effective — it eliminates the remaining host launch overhead for the entire 88-layer forward pass. This is the step that gets us to ~80ms (12+ tok/s).

Graph capture requires all per-token logic to be device-resident (already achieved with the clean CUTLASS/custom kernel paths) and a fixed kernel launch pattern (achievable once per-call allocations are eliminated).

### 6. MTP (Multi-Token Prediction)

Doubles effective throughput by speculating multiple tokens per forward pass. Independent of the kernel optimization work. Gets us from ~12 tok/s to ~24 tok/s.

## What NOT To Do

- **Don't switch GEMM backends.** CUTLASS tensor cores = custom scalar kernel at M=1. The GEMM is at the memory bandwidth floor.
- **Don't focus on CUDA graph capture first.** Graph capture hides launch overhead but doesn't reduce it. With 400+ kernels, the graph itself becomes large and complex. Reduce kernel count first, then capture.
- **Don't build a fully monolithic kernel yet.** vLLM's single-kernel approach (flashinfer.fused_moe.trtllm_fp4_block_scale_moe) is the end state, but it requires fusing routing + packing + GEMM + activation + GEMM + merge into one kernel. That's a large custom CUDA kernel. The incremental approach (eliminate allocations, fuse packing, fuse repack) gets most of the benefit with much less risk.

## Immediate Next Step

Eliminate per-call cudaMalloc in the NVFP4 packing path. This is the highest-impact change with the lowest risk — pre-allocate the packing buffers in the request context during model construction, reuse them every token. No kernel code changes needed, just buffer management.
