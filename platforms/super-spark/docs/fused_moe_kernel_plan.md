# Fused MoE Kernel Plan

Build a single CUDA kernel that executes the entire routed-expert MoE path for one layer in one launch, using CUTLASS NVFP4 grouped GEMM primitives compiled for SM121 (GB10). No external library dependency.

## Hard Constraint: Zero Host Involvement On The Hot Path

Every step of the fused MoE execution must run entirely on the GPU. No D→H copies, no host-side dispatch, no CPU fallbacks, no host-driven pointer array construction. The entire point of this kernel is to eliminate the ~660ms of overhead that comes from host-GPU coordination.

This means:
- Expert weight pointer arrays must be pre-built on device during model construction, not assembled per-token
- The CUTLASS pointer-pair format (`tuple<ElementA, ElementSF>`) must be materialized on device at construction time
- Router selection, expert dispatch, activation packing, GEMMs, activation function, weighted merge — all device-resident
- The only host involvement is the initial kernel launch (one `cudaLaunchKernel` call per MoE layer, eventually captured in a CUDA graph)

## What The Kernel Does

Input: hidden states, router logits, expert weights (NVFP4), config.
Output: weighted sum of selected expert outputs.

In one kernel launch:
1. Compute top-k expert selection from router logits (sigmoid + grouped top-k)
2. Permute tokens by expert assignment
3. GEMM1: permuted tokens × selected expert up_proj weights (NVFP4 block-scaled)
4. ReLU² activation
5. GEMM2: activated intermediates × selected expert down_proj weights (NVFP4 block-scaled)
6. Unpermute, apply expert routing weights, reduce to output

## Why From Scratch

- No external library dependency (FlashInfer build system, version pinning, ABI stability)
- Full control over SM121 tuning (99 KiB shared memory, no TMEM, no WGMMA)
- We already have the complete reference from vLLM's FlashInfer integration — the kernel interface, weight layout, shuffle functions, and CUTLASS tile configs
- Weight prep utilities (`shuffle_matrix_a`, `shuffle_matrix_sf_a`, 128x4 interleave) are already implemented locally

## Reference Architecture

From the vLLM/FlashInfer analysis, the community kernel has this structure:

```
// Pseudocode for the fused MoE kernel
__global__ void fused_moe_nvfp4(
    // Routing
    const bfloat16* router_logits,      // [num_tokens, num_experts]
    const bfloat16* routing_bias,       // [num_experts]
    // Activations
    const bfloat16* hidden_states,      // [num_tokens, hidden_size]
    // Expert weights (all experts stacked contiguously)
    const uint8_t* w1_packed,           // [num_experts, intermediate_size, hidden_size/2]
    const fp8_e4m3* w1_scales,          // [num_experts, intermediate_size/B, hidden_size/B]
    const uint8_t* w2_packed,           // [num_experts, hidden_size, intermediate_size/2]
    const fp8_e4m3* w2_scales,          // [num_experts, hidden_size/B, intermediate_size/B]
    // Scale factors
    float output1_scale,
    float output2_scale,
    // Config
    int num_experts, int top_k, int n_group, int topk_group,
    int hidden_size, int intermediate_size,
    float routed_scaling_factor,
    // Output
    bfloat16* output                    // [num_tokens, hidden_size]
) {
    // Phase 1: Router + top-k (threadblock-level)
    //   - Each token computes sigmoid(logits) + bias
    //   - Grouped top-k selection
    //   - Write (expert_id, weight) pairs to shared memory
    
    // Phase 2: Token-expert permutation
    //   - Sort/group tokens by assigned expert
    //   - Build expert_offsets array
    
    // Phase 3: Grouped GEMM1 (CUTLASS)
    //   - For each active expert: tokens_for_expert × w1[expert_id]
    //   - NVFP4 block-scale dequant inline
    //   - ReLU² activation fused into epilogue
    
    // Phase 4: Grouped GEMM2 (CUTLASS)
    //   - For each active expert: activated_tokens × w2[expert_id]
    //   - NVFP4 block-scale dequant inline
    
    // Phase 5: Unpermute + weighted reduce
    //   - Scatter expert outputs back to token positions
    //   - Apply routing weights
    //   - Sum across top-k experts per token
}
```

## Implementation Plan

### Step 1: CUTLASS SM121 NVFP4 GEMM Primitive

Build and validate a standalone CUTLASS NVFP4 grouped GEMM for our exact shapes:
- M=1 (single-token decode), K=1024 (moe_latent_size), N=2688 (intermediate_size)
- M=1, K=2688, N=1024 (down_proj)
- NVFP4 block-scaled weights with FP8 E4M3 block scales
- FP32 accumulation, BF16 output
- SM121 constraints: 99 KiB shared memory, no TMEM

Use CUTLASS 4.4.2+ (the version validated by the community). The key template parameters:
- `cutlass::gemm::collective::CollectiveMma` with FP4 A and FP4 B
- Tile shape must fit in 99 KiB shared memory
- Check the SM121-specific tile configs from CUTLASS examples

Validation: compare output against our existing per-expert cublasLt path on the same expert weights and activations. Should match to FP32 accumulation tolerance.

### Step 2: Token-Expert Permutation Kernel

Build a small kernel that:
- Takes router logits + config → produces (expert_ids, expert_weights, token_permutation)
- Implements sigmoid scoring + grouped top-k (matching our existing `SelectTopExpertsFp32`)
- Outputs token indices sorted by expert, plus expert offset array
- All on device, no host involvement

This is a lightweight kernel — mostly comparisons and sorting for top_k=6 across 512 experts with n_group/topk_group constraints. For single-token decode (M=1), this is trivial: just one token selecting 6 experts.

Validation: compare expert_ids and expert_weights against our existing `SelectTopExpertsFp32` output.

### Step 3: Fused Kernel Assembly

Combine steps 1 and 2 into a single kernel launch:
- Shared memory layout: routing metadata + GEMM tiles
- Phase 1 (routing) runs in a few warps, writes to shared memory
- Phase 2 (permutation) reorganizes in shared memory
- Phase 3 (GEMM1 + activation) runs the CUTLASS collective
- Phase 4 (GEMM2) runs the CUTLASS collective
- Phase 5 (unpermute + reduce) writes final output

For single-token decode (M=1), the "permutation" is trivial — there's one token, it goes to 6 experts. The kernel simplifies to: select 6 experts, run 6 up_proj GEMMs, relu2, run 6 down_proj GEMMs, weighted sum. The CUTLASS grouped GEMM handles the "6 different B matrices" by indexing into the contiguous expert weight block.

### Step 4: Weight Layout Preparation

During model construction, prepare expert weights in the kernel's expected layout:
- Stack all 512 experts contiguously: `[512, N, K]` packed NVFP4
- Apply the CUTLASS-required scale shuffling (our existing local implementations)
- Store as one contiguous device allocation per MoE layer (we already have the pooled allocation infrastructure)

This replaces the current per-expert `UploadedLinearOp` approach with a single contiguous weight block per layer.

### Step 5: Integration

Replace `ExpertLayerSlice::RunWithRequestContext()` routed path with a single call to the fused kernel. Keep the existing per-expert path for oracle tests (gated by trace != nullptr).

The shared expert path stays separate — it's one shared up_proj + down_proj, not worth fusing.

### Step 6: Validate and Measure

- Expert oracle tests (functional equivalence, not bitwise)
- Full decode oracle (top-k token agreement)
- 16-token generation bench (steady-state tok/s)
- Memory telemetry (UMA pressure check)
- Per-layer CUDA event profiling (confirm expert layer time dropped from ~8.6ms to <1ms)

## SM121 Constraints

GB10 reports 48 KiB shared memory per block (verified via `cudaDeviceProp.sharedMemPerBlock`). This is even less than the earlier 99 KiB estimate. Despite this, the 128x128x128 CUTLASS tile compiled and the `can_implement` check should verify it fits.

CUTLASS 4.4.2 has SM121-aware tile selection via `KernelScheduleAuto` and `StageCountAutoCarveout`.

## Expected Result

| Metric | Before | After fused kernel |
|---|---:|---:|
| Expert layer time (per layer) | ~8.6ms | <1ms |
| Expert total (40 layers) | ~344ms | <40ms |
| Kernel launches per MoE layer | ~8+ | 1 |
| Host involvement per MoE layer | D→H sync + dispatch loop | none |

This unblocks CUDA graph capture for the full forward pass, since the MoE path will be entirely device-resident.

## Files To Create

- `runtime/include/nemotron/fused_moe_kernel.h` — kernel interface
- `runtime/src/backend/fused_moe_kernel.cu` — CUTLASS-based fused kernel
- `runtime/src/backend/moe_permutation.cu` — token-expert permutation (can be inline in fused kernel)
- `testing/backend/fused_moe_kernel_test.cpp` — standalone validation against existing oracle

## Dependencies

- CUTLASS 4.4.2+ headers (add as a build dependency, header-only)
- CUDA 13.2 with SM121 support
- Our existing NVFP4 weight pool infrastructure
- Our existing `SelectTopExpertsFp32` as the reference for routing validation

## Implementation Progress

### Step 1: CUTLASS NVFP4 Grouped GEMM Primitive — DONE

- CUTLASS 4.4.2 added as a FetchContent dependency in `plugins/fused_moe/CMakeLists.txt`
- NVFP4 grouped GEMM template instantiated for SM120/SM121 with:
  - A: `nv_float4_t<float_e2m1_t>`, RowMajor (activations)
  - B: `nv_float4_t<float_e2m1_t>`, ColumnMajor (weights)
  - Output: FP32 (matches runtime hidden-state format)
  - FP32 accumulation
  - 128x128x128 tile, 1x1x1 cluster
  - `OpClassBlockScaledTensorOp` with `Sm120` arch tag
- Scale factor layouts computed via `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`
- Compiled and linked for SM121 on CUDA 13.2
- Verified running on GB10 hardware (SM 12.1, 48 KiB per-block shared memory)
- **Critical finding**: must compile with `sm_121a` (not `sm_121`) to enable conditional MMA arch features. Without the `a` suffix, the CUTLASS kernel hits a runtime assertion: "Arch conditional MMA instruction used without targeting appropriate compute capability."
- Execution test with zero-filled buffers at exact MoE shapes passed:
  - M=1, N=2688, K=1024, 6 groups (top_k experts)
  - `can_implement`: OK
  - workspace: 43,008 bytes
  - kernel launched and produced correct zero output
  - no CUDA errors
- Files:
  - `plugins/fused_moe/CMakeLists.txt`
  - `plugins/fused_moe/src/nvfp4_grouped_gemm.cu`
  - `plugins/fused_moe/src/plugin_entry.cu`
  - `plugins/fused_moe/test/nvfp4_grouped_gemm_test.cu`

### Step 2: Weight Format Conversion — NOT NEEDED

**Critical finding**: our existing NVFP4 weight format is already compatible with CUTLASS.

- Packed weight data: both cublasLt and CUTLASS use `nv_float4_t<float_e2m1_t>` in ColumnMajor layout for B (weights). Same byte-level packing.
- Block scale layout: numerically verified that our `SwizzleRowMajorNvfp4ScalesForExecution` produces the **identical** byte layout as FlashInfer's `InterleaveBlockScales128x4` and CUTLASS's `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB` at the exact expert dimensions (2688×64 = 172,032 elements, zero differences).
- The existing pooled expert weight buffers on device already contain data in the format CUTLASS expects. No reformat needed.
- The only difference is how the tensor-level scale is applied: CUTLASS uses `alpha = activation_tensor_scale * weight_tensor_scale` as the GEMM alpha parameter, while cublasLt reads it from a device pointer. This is handled at the call site.

### Step 3: Device-Resident Pointer Pairing — IN PROGRESS

The CUTLASS grouped GEMM Arguments struct expects A/B pointer arrays in a paired tuple format: `tuple<ElementA, ElementSF> const**`. Our existing infrastructure stores data pointers and scale pointers in separate device arrays. These must be pre-paired into the tuple format during model construction — not per-token.

Implementation plan:
1. During `ExpertLayerSlice::Create()`, build device-resident arrays of paired `(packed_data_ptr, scale_ptr)` tuples for all 512 experts, for both up_proj and down_proj
2. At dispatch time, a small device kernel gathers the 6 selected experts' paired pointers from the lookup table using the top-k indices — fully on device, zero host involvement
3. The gathered paired pointer array feeds directly into `RunCutlassNvfp4GroupedGemm`
4. Similarly, build paired activation pointer arrays on device (all groups point to the same packed activation)

This replaces the current `GatherExpertSelectionLookupsChecked` → `D→H copy` → `host pointer array` → `CUTLASS re-upload` pipeline with a single device-side gather.

### Alignment Issue: Down-Proj Activation Rows

The CUTLASS NVFP4 GEMM requires 256-byte aligned A pointers. The up_proj works because `PackDeviceRowMajorFp32ToNvfp4` allocates a fresh buffer (cudaMalloc guarantees alignment). But the down_proj activation rows are packed contiguously by `ScaleRelu2PackRowsToNvfp4` at 1344-byte row stride (2688/2), which is NOT 256-byte aligned. The GEMM launches but hits `misaligned address` at runtime.

Fix: pad the packed row stride to the next 256-byte multiple (1344 → 1536 bytes, or 3072 elements). This requires modifying `ScaleRelu2PackRowsToNvfp4` to accept a padded stride, or allocating per-expert aligned buffers.

### Step 4: Full Fused Expert Dispatch — NOT YET STARTED

Once Step 3 is done, wire the full routed expert path:
1. Pack FP32 latent activations to NVFP4 on device
2. Device-side gather of selected expert paired pointers (from Step 3)
3. `RunCutlassNvfp4GroupedGemm` for up_proj — one kernel launch for all 6 experts
4. relu2 activation on device
5. Pack intermediate to NVFP4 on device
6. `RunCutlassNvfp4GroupedGemm` for down_proj — one kernel launch for all 6 experts
7. Weighted expert merge and reduce on device

All steps device-resident. No host involvement. No D→H copies. No CPU fallbacks.

### Steps 5-6: Not yet started
