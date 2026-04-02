# Host Overhead Elimination Plan — Draft

## Problem

28.2 ms/token measured, kernel compute floor ~11.5ms. The remaining ~16.5ms is host overhead dominated by 36,769 cudaMemcpy calls (87.6% of CUDA API time).

Profile breakdown of H2D copies:
- 6,063 copies >1MB (18.3 GB total, 1.38s) — ~357/token, maps to ~15.5 per MoE layer
- 11,925 copies 64KB-1MB (3.6 GB total, 94ms) — ~701/token
- 6,495 copies <64B (26 KB total, 1.8ms) — tiny scalar copies

The 6,063 large H2D copies are the primary target. They correspond to NVFP4 activation packing in the cuBLASLt MoE path — the activation data should already be on device but is being copied host-to-device during packing.

## Root Cause Analysis

The cuBLASLt MoE path in `RunMoeDirectDecodeViaCublaslt()` does for each expert:
1. Pack FP32 activation → NVFP4 via `PackDeviceRowMajorFp32ToNvfp4()`
2. Build `Nvfp4PackedMatrixDeviceView` from the packed result
3. Call cuBLASLt GEMM

`PackDeviceRowMajorFp32ToNvfp4()` creates a new `DeviceNvfp4Matrix` each call. The `DeviceNvfp4Matrix::Create()` allocates device memory and uploads. But the activation is already on device — the packing kernel runs on device. The H2D copies may come from:
- The tensor scale computation (global max → host → device tensor scale)
- The block scale swizzling for matmul format
- Or from `DeviceNvfp4Matrix` constructor doing unnecessary host staging

In vLLM, activation quantization is a single device kernel (`scaled_fp4_quant`) that writes packed FP4 + scales directly to device buffers — no H2D copies involved.

## Proposed Steps

1. **Audit the NVFP4 activation packing path** — trace exactly which cudaMemcpy calls come from `PackDeviceRowMajorFp32ToNvfp4` and the cuBLASLt expert path
2. **Make activation packing fully device-side** — the FP32→FP4 quantization, tensor scale computation, and block scale swizzling should all stay on device with no H2D copies
3. **Cache packed activation buffers** — pack the normalized input once, reuse for all 6 selected experts (currently repacks per expert)
4. **Eliminate small scalar H2D copies** — routing indices, weights, tensor scales may be doing unnecessary device→host→device round-trips
5. **Re-profile and measure**
