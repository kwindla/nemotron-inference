# SM120 FP4 MoE Implementation Research (Corrected)

This document summarizes the research findings for implementing the Grouped FP4×FP4 Tensor Core MoE kernel for NVIDIA Blackwell (SM120 / RTX 5090).

## 1. SM120 FP4 MMA Interface
The Blackwell consumer architecture (SM120) uses warp-level `mma.sync` instructions for dense block-scaled FP4 matrix multiplication.
- **Instruction Shape**: **`m16n8k64`** for dense e2m1 x e2m1 operations. (Note: `k128` is reserved for sparse paths).
- **Data Type**: NVFP4 (`e2m1`) for both activations and weights.
- **Accumulation**: FP32 (`.f32`) for numerical stability.
- **Microscaling (MX)**: Block-scaled operations are handled directly within the MMA instruction. Supported scale formats include `ue4m3` and `ue8m0`.

## 2. Block Scale Layout & Choice
The layout of scale factors (SF) is a software-defined choice that must be consistent with the GEMM kernel's expectations.
- **Fastpath Choice**: For our optimized MoE prefill path, we choose the **128x4 swizzled layout** (128 rows, 4 scale columns).
- **Small-M Handling**: The runtime and reference implementations also support an **8x4** layout for smaller M dimensions.
- **Padding**: When using the 128x4 fastpath choice, the execution scale buffer pads rows to **128**, while logical matrix rows remain unchanged.
- **Index Math (`get_sf_out_offset_128x4`)**:
    - `innerKIdx`: `kIdx % 4` (stride 1)
    - `innerMIdx`: `(mIdx % 128) / 32` (stride 4)
    - `outerMIdx`: `mIdx % 32` (stride 16)
    - `kTileIdx`: `kIdx / 4` (stride 512)
    - `mTileIdx`: `mIdx / 128` (stride `numKTiles * 512`)

## 3. Quantization Strategy
The quantization boundaries are determined by the MoE architecture:
- **Layer Input**: The normalized hidden state can be quantized **once per layer** before being dispatched to experts.
- **Expert Intermediates**: The `down_proj` inputs (produced after `up_proj` + `ReLU²`) are expert-specific. In the current per-expert loop, these are quantized individually. A grouped MoE kernel would need to handle this intermediate quantization, potentially by fusing it or using a grouped quantization launch.
- **Fusing**: To reduce TTFT, the quantization kernels should fuse max-abs computation, packing, and swizzling.

## 4. Grouped GEMM for MoE
- **Library**: Use **CUTLASS 3.x (Cute)** with `GroupProblemShape`.
- **Scheduler**: Prior art shows both `KernelScheduleAuto` (vLLM) and `KernelTmaWarpSpecializedCooperative` (TRT-LLM) are viable. The choice should be based on measured performance for our specific expert dimensions.
- **Layout**: Standard Row-Major A and Column-Major B are well-supported. Avoid speculative "layout tricks" unless measured.

## 5. References and Prior Art
- **TRT-LLM**: `nvfp4_nvfp4_gemm_template_sm120.h`, `fp4Quantize.cpp`.
- **vLLM**: `nvfp4_blockwise_moe_kernel.cu`.
- **CUTLASS Headers**: `vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/cutlass/include/cute/arch/mma_sm120.hpp`.
