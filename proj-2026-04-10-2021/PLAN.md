# Plan: Native FP4 MMA for shared expert GEMMs

Project directory: `./proj-2026-04-10-2021`

## Context
The shared expert path in `RunFusedMoePrefill` uses a lossy pipeline: `QuantizeDequantizeRows` (FP4 round-trip simulation) → `Nvfp4ContiguousWmmaMatVecRowsKernel` (FP4→BF16 decode, BF16×BF16 WMMA). This loses precision at every layer because of the BF16 intermediate truncation. The shared expert runs on every expert layer for all tokens, making it the primary source of accumulated error that causes degenerate output at short prompt lengths (confirmed by per-layer diagnostic showing error cascading from ~0.004 at layer 1 to ~42 at layer 51). The fix: pack activations to FP4 and use the same native SM120 block-scaled FP4 MMA as the unified routed kernels (P5/P13/P15), eliminating the BF16 intermediate entirely. The weight format (`FusedNvfp4WeightView`) is identical; the difference is contiguous rows with no routing.

## Rules

- The shared expert kernel must use native FP4 block-scaled MMA (SM120 `mma.nvfp4`), not decode-to-BF16 WMMA.
- Activations must be packed to FP4 via `PackIntoPerRow` (per-row tensor scales), not the fake `QuantizeDequantizeRows` round-trip.
- The kernel must support arbitrary token counts (1–512+), handling partial tiles correctly.
- The shared expert pipeline change must not affect the routed expert path.
- TMA loads for the weight operand should follow the pattern established by the unified routed P5 kernel.
- Validation must use exact oracles:
  - per-layer max_abs_diff diagnostic (comparing fused prefill vs legacy decode) must show dramatically reduced error
  - prompt-length sweep (`tools/oracle/prompt_length_sweep.py`) must produce coherent output at all lengths, judged by Claude API
  - constrained greedy parity against vLLM/TRT-LLM via `tools/oracle/compare_chat_runtimes.py`
  - existing fused prefill/runtime tests (full ctest suite, 68/71 baseline)
  Text that merely "looks non-repetitive" is not sufficient.

## Steps

- [ ] **1. Add packed FP4 workspace for shared expert activations**
  Add two new `DeviceNvfp4Matrix` fields to `FusedMoePrefillParams`: `shared_input_pack` (for packing normalized input before shared_up GEMM) and `shared_activated_pack` (for packing activated intermediate before shared_down GEMM). Allocate these in `ExpertLayerSlice::Impl` alongside the existing `fused_prefill_shared_up_pack` (which can be repurposed or kept). Wire through from `RunFusedMoePrefillPath` to `RunFusedMoePrefill`. Both matrices have `Nvfp4ScaleLayout::kSwizzled128x4` to match the unified kernel's TMA requirements.
  Key files: `runtime/include/nemotron/fused_moe_prefill.h`, `runtime/src/backend/expert_layer.cpp`

- [ ] **2. Write contiguous FP4 MMA GEMM kernel**
  Write `LaunchContiguousFp4MatVec` — a new function that replaces `LaunchContiguousMatVec` for the shared expert. It takes a packed `DeviceNvfp4Matrix` (activation input) and a `FusedNvfp4WeightView` (weight) and produces FP32 output. Internally, launch a CUDA kernel that uses CUTE `SM120_MMA_Nvfp4` atoms (same as the unified routed P5 kernel). The grid is `(output_tile_count, input_row_tile_count)` — no routing, no launch plan. Weight tiles are loaded via TMA descriptors built at launch time (one descriptor for the weight matrix, one for its block scales). Activation tiles are loaded from the packed FP4 matrix with block scales applied during the MMA. Start with the P5 tile shape (128×128×128) which is proven. The kernel body is modeled on the unified routed kernel but stripped of all routing indirection (no `cta_batch_indices`, no `cta_row_starts`, no per-expert weight selection — just contiguous row tiles × output tiles).
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [ ] **3. Wire shared expert pipeline to use FP4 MMA**
  In `RunFusedMoePrefill` (lines 10266–10298), replace the shared expert pipeline:
  - **Before**: `cudaMemcpy(normalized→scratch)` → `QuantizeDequantizeRows` → `LaunchContiguousMatVec(shared_up)` → `Relu2` → `QuantizeDequantizeRows` → `LaunchContiguousMatVec(shared_down)` → `AccumulateSharedOutput`
  - **After**: `PackIntoPerRow(normalized→shared_input_pack)` → `LaunchContiguousFp4MatVec(shared_input_pack, shared_up)` → `Relu2` → `PackIntoPerRow(relu2_output→shared_activated_pack)` → `LaunchContiguousFp4MatVec(shared_activated_pack, shared_down)` → `AccumulateSharedOutput`
  The `shared_up_scratch` FP32 buffer is still used for the Relu2 output (the FP4 MMA kernel writes FP32). The `routed_gather_scratch` is still used for the final shared_down output.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/src/backend/expert_layer.cpp`

- [ ] **4. Validate precision and run full test suite**
  Rebuild and run `fused_moe_prefill_test` (must PASS). Run the per-layer diagnostic (`NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`) on the len=9 prompt and compare max_abs_diff per layer against the pre-fix baseline. Expected: layer 1 error should drop from ~0.004 to near-zero, and pass 2 errors should not cascade. Run the full prompt-length sweep (`tools/oracle/prompt_length_sweep.py`) with `--runtimes native --native-env NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1` and verify all 16 lengths produce coherent output. Run the full ctest suite (68/71 baseline).
  Key files: `tools/oracle/prompt_length_sweep.py`, `testing/backend/fused_moe_prefill_test.cpp`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Add packed FP4 workspace | pending | — | |
| 2 | Write contiguous FP4 MMA kernel | pending | — | |
| 3 | Wire shared expert pipeline | pending | — | |
| 4 | Validate precision and test | pending | — | |
