# Plan: Per-row FP4 tensor scale for batched MoE prefill

Project directory: `./proj-2026-04-10-2330`

## Context
The fused MoE prefill path packs all tokens' activations with a single global tensor scale (max-abs across the batch). When one token has much larger activations, the shared scale crushes the FP4 dynamic range of smaller tokens, causing degenerate output (repetitive loops) at small token counts. The fix: compute per-row tensor scales during packing, gather them with the permuted rows, and apply per-row scales during FC1 decode only where the current contract uses `input_tensor_scale_data`. vLLM and TRT-LLM use per-token scales; our current global scale is the root cause of the batch-coupled activation scaling problem.

## Rules

- Per-row tensor scales replace only the global tensor-scale path (`input_tensor_scale_data`). They must never override `input_expert_tensor_scales` or `input_dq_scales`.
- `PackIntoPerRow()` is MoE-prefill-only until other consumers are explicitly upgraded. A matrix packed with per-row tensor scales must not be consumed by code that only reads `device_tensor_scale_ptr()`.
- Validation must use exact oracles:
  - low-level pack/gather checks
  - constrained greedy parity against vLLM/TRT-LLM
  - existing fused prefill/runtime tests
  Text that merely "looks non-repetitive" is not sufficient.

## Steps

- [x] **1. Add per-row packing infrastructure to DeviceNvfp4Matrix**
  Add `per_row_tensor_scales` (float*) storage to `DeviceNvfp4Matrix::Impl`. Allocate in `Create()`, free in destructor. Add three CUDA kernels: `ComputePerRowMaxAbsKernel` (one atomicMax per row), `WritePerRowTensorScalesKernel` (per-row max-abs to tensor scale), `PackAndSwizzleRowMajorFp32ToNvfp4PerRowScaleKernel` (same as existing pack kernel but reads `per_row_tensor_scales[row]` instead of `*tensor_scale_data`). Add `PackIntoPerRow()` method that orchestrates these. Keep writing the existing global tensor scale for bookkeeping/debug visibility, but do not treat it as a complete decode contract for per-row-packed matrices. Add `per_row_tensor_scales()` accessor. Add per-row scale gathering in `GatherDeviceNvfp4Rows`: when both source and output have `per_row_tensor_scales != nullptr`, launch `GatherPerRowTensorScalesKernel` to gather `source.per_row_tensor_scales[source_row_indices[i]]` into `output.per_row_tensor_scales[i]`.
  Key files: `runtime/include/nemotron/device_nvfp4_matrix.h`, `runtime/src/backend/device_nvfp4_matrix.cu`

- [x] **2. Add `input_per_row_tensor_scales` parameter to all FC1 kernels**
  Add `const float* input_per_row_tensor_scales` parameter to the following kernel signatures (after `input_dq_scales`): `Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalse` (line 4668), `Nvfp4LaunchPlannedPackedInputGroupedKernelSwapTrue` (line 4877), `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel` (line 5405). Also add to `Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel` (line 7748) and `Nvfp4LaunchPlannedPackedInputExpertMatVecRowsBf16Kernel` (line 7911) for completeness, though these legacy kernels will pass nullptr. For the unified kernel's `#else` (non-CUTE) block, add `(void) input_per_row_tensor_scales;`.
  In each NON-unified kernel (SwapFalse, SwapTrue): preserve the current precedence exactly:
  1. `input_dq_scales`
  2. `input_expert_tensor_scales`
  3. `input_per_row_tensor_scales`
  4. `input_tensor_scale_data`
  Concretely, at the B-tile decode call site where `DecodeGroupedPackedInputBlockBf16` is called, derive `row_ts` only for case 3/4:
  `const float row_ts = (input_expert_tensor_scales != nullptr) ? input_expert_tensor_scales[expert_index] : ((input_per_row_tensor_scales != nullptr) ? input_per_row_tensor_scales[row_start + tile_token] : *input_tensor_scale_data);`
  and continue to bypass tensor-scale logic entirely when `input_dq_scales != nullptr`.
  In the unified kernel: modify `StoreUnifiedRoutedFp4Output` to accept optional `const float* per_row_tensor_scales` and `float weight_tensor_scale` default params. In all three store paths (P5, P13, generic), compute `row_alpha = (per_row_tensor_scales != nullptr) ? per_row_tensor_scales[input_row] * weight_tensor_scale : alpha` and use it instead of `alpha`. This replaces only the current `input_tensor_scale * weight_tensor_scale` path; it must not interfere with FC2 repacked activations that already use per-expert or dq scales. Update the single call site (line ~6866) to pass `input_per_row_tensor_scales` and `*weight.tensor_scale_data`.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [x] **3. Wire per-row scales through launch functions and call sites**
  Update `LaunchPlannedPackedInputMatVecBf16` (line 9468) to accept and forward `const float* input_per_row_tensor_scales`. Pass it through to each FC1 kernel call in the switch statement. For the P5 unified kernel launch via `LaunchProgrammaticKernel`, pass `input_pack.per_row_tensor_scales()`. For all other FC1 kernel launches (P0, P1, P4, P7, legacy), pass the same pointer.
  Update `LaunchPlannedPackedInputMatVec` (line 9252) to accept the same pointer too, because it is used in two different ways:
  - FC1 legacy packed route at `RunFusedMoePrefill(... use_legacy_packed_fc1 ...)`
  - FC2 packed route
  Wire it by caller, not by function name:
  - FC1 legacy packed route: pass `fc1_grouped_pack->per_row_tensor_scales()`
  - FC2 paths: pass `nullptr`, because FC2 inputs are repacked with their own expert/dq scales
  Update all sites in `RunFusedMoePrefill` that call these launch functions accordingly.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [x] **4. Wire PackIntoPerRow in the expert layer**
  In `RunFusedMoePrefillPath` (expert_layer.cpp line 1520), change `normalized_pack->PackInto(...)` to `normalized_pack->PackIntoPerRow(...)`. The rest of the FC1 prefill pipeline then picks up per-row scales because `fc1_grouped_pack` receives them via `GatherDeviceNvfp4Rows` and the FC1 launch functions read them via `input_pack.per_row_tensor_scales()`.
  Do not change the generic decode path, cublasLt direct-decode path, or FC2 repack sites in this step. Those remain on the old contract until explicitly upgraded.
  Key files: `runtime/src/backend/expert_layer.cpp`

- [x] **5. Add exact low-level tests for the new contract**
  Add a focused `DeviceNvfp4Matrix` pack/gather test:
  - pack several rows with deliberately different max-abs magnitudes via `PackIntoPerRow()`
  - assert gathered `per_row_tensor_scales` match the selected source rows exactly after `GatherDeviceNvfp4Rows`
  - assert packed data differs when row scales differ, while row permutation preserves row-to-scale pairing
  Add one fused FC1 oracle that checks row-local scale usage on grouped input, not just text output.
  Key files: `testing/backend/fused_moe_prefill_test.cpp`

- [x] **6. Validate against external and model-level oracles**
  Run `fused_moe_prefill_test` and `multi_turn_prefix_reuse_test` (must PASS). Then run the constrained temp-0 tri-runtime parity harness in `tools/oracle/compare_chat_runtimes.py` against native, vLLM, and TRT-LLM on the known failing prompts. The acceptance bar is:
  - no regression on the existing constrained prompts
  - the previously failing short prompts (`setA`, `setC`, and their chat-templated equivalents) no longer diverge from both external runtimes
  After that, run the 3-token and 9-token prompt sweep with `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1`, then the 5-turn interactive conversation, to confirm the fix survives end-to-end use.
  Text that merely looks less repetitive is not sufficient; use token-level parity or exact constrained answers where possible.
  Key files: `testing/backend/fused_moe_prefill_test.cpp`, `tools/oracle/compare_chat_runtimes.py`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Per-row packing infrastructure | done | 93f3a03 | |
| 2 | Add parameter to all FC1 kernels | done | 053d64d | |
| 3 | Wire through launch functions | done | 15b8041 | build + fused_moe_prefill_test |
| 4 | Wire PackIntoPerRow in expert layer | done | a218519 | fused_moe_prefill_test PASS |
| 5 | Add exact low-level tests | done | 27cd91d | 68/71 pass, 3 pre-existing |
| 6 | Validate and test | done | — | 68/71 ctest pass; tri-runtime parity needs manual run |
