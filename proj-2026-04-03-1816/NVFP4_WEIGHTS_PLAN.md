# Plan: NVFP4 Expert-Weight Preparation Alignment

Project directory: `./proj-2026-04-03-1816`

## Goal

Close all accidental divergences in the NVFP4 expert-weight path and move this branch toward a polished prepared-weight implementation that matches the vLLM reference shape unless there is a concrete reason not to.

This is now a **gap-closing** project, not a feature-prototyping project. The critical priority is to remove accidental complexity from the expert-weight path and keep only the divergence that is required for our custom prefix-caching goals or for clearly superior measured behavior.

## Scope

This review is limited to NVFP4 expert weights:

- monolithic expert storage
- backend-native prepared views
- scale swizzling and padding
- runtime vs load-time transforms
- fast-path vs fallback-path separation

It does not cover the rest of the runtime except where that runtime consumes prepared NVFP4 expert weights.

## Reference Shape

The vLLM reference model for this work is:

- `third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `third_party/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`
- `third_party/vllm/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`
- `third_party/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`

The key vLLM properties are:

- weights are converted after load, not during hot execution
- prepared tensors are rewritten into backend-native kernel format
- block scales are swizzled once into execution layout
- the MoE kernel consumes the prepared form directly
- the kernel object owns the prepared contract, not a runtime re-upload path

## Current Branch State

Current code already has real prepared-weight infrastructure:

- `MonolithicNvfp4ExpertWeights` stores all experts in one GPU-resident monolithic block
- `DeviceNvfp4Weight` uploads one descriptor into a device-resident kernel-ready form
- `FusedNvfp4WeightView` exposes the prepared layout as a narrow execution view
- `SwizzleRowMajorNvfp4ScalesForExecution()` produces execution-layout block scales
- `ExpertLayerSlice` can route into unified fused or legacy backends using prepared views

That is the good news.

The bad news is that the branch still contains multiple overlapping weight representations and at least one hot-path re-upload escape hatch. That is the accidental divergence we need to close.

## Alignment With vLLM

The following are broadly aligned with the reference implementation:

- load-time conversion of packed weight bytes into a kernel-ready form
- backend-specific scale swizzling
- monolithic residency for the fast path when the layer is fully resident
- a prepared-view contract that lets the kernel consume the weight without reinterpreting the original descriptor shape

Concrete aligned pieces in this branch:

- `runtime/src/backend/monolithic_expert_weights.cu`
- `runtime/src/backend/nvfp4_weight.cpp`
- `runtime/src/backend/nvfp4_scale_layout.cpp`
- `runtime/src/backend/expert_layer.cpp`

Concrete vLLM alignment points:

- `prepare_nvfp4_moe_layer_for_fi_or_cutlass()` in [flashinfer_fp4_moe.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py:192)
- `swizzle_blockscale()` in [nvfp4_utils.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:272)
- `process_weights_after_loading()` in [compressed_tensors_moe.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py:508)

## Accidental Divergences

These are the mismatches that should be treated as bugs until proven otherwise.

### 1. Hot-path re-upload still exists

`ResolveRoutedExpertWeightViews()` in [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:1260) will fall back to `TryUploadNvfp4Weight()` if the layer is not already resident or prepared.

That is acceptable as a fallback path. It is not acceptable as the polished fast-path story.

vLLM’s prepared-weight model does not rely on per-call expert uploads. If the expert weights are not resident at prepare time, the backend should refuse the fast path instead of synthesizing prepared views during execution.

### 2. We have multiple prepared-weight owners

The current branch has at least three overlapping owners of NVFP4 expert state:

- `DeviceNvfp4Weight`
- `MonolithicNvfp4ExpertWeights`
- `FusedNvfp4WeightView`

That split may be tolerable for staged bring-up, but it is too fragmented for a polished implementation unless each type has a hard, non-overlapping responsibility.

The current risk is that weight prep is only mostly centralized, while the runtime still has enough hooks to rebuild or re-upload weight state from descriptors.

### 3. Scale preparation is duplicated across code paths

Scale swizzling exists in both:

- `SwizzleRowMajorNvfp4ScalesForExecution()` in C++
- `swizzle_blockscale()` in vLLM

The intended layout appears to match, but that has not yet been proven by a tensor-by-tensor golden test on all relevant shapes, padding cases, and expert layouts.

### 4. Host fallbacks still leak into the GEMM bridge

`RunNvfp4RowMajorFp32AccumToDevice()` in [nvfp4_gemm_runner.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/nvfp4_gemm_runner.cpp:698) still has a host fallback for reading tensor scales when device pointer mode is not available.

That is a defensible safety net, but it is also an accidental performance risk if it survives in any hot path that is supposed to be fully prepared.

### 5. Manifest assumptions are not yet fully proven

`DeviceNvfp4Weight::Upload()` in [nvfp4_weight.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/nvfp4_weight.cpp:56) trusts that the descriptor already has:

- the correct packed layout
- the correct tensor/compute/layout tags
- the correct block-scale shape

If those assumptions are wrong, the bug is not in the upload code. The bug is earlier, in manifest generation or kernel-tensor conversion.

That assumption needs a golden test, not another layer of runtime defense.

## Intentional Divergences

These should remain unless the evidence changes.

- The branch uses custom C++/CUDA prepared-weight types instead of vLLM’s Python-side tensor mutation path.
- The branch keeps a monolithic expert-storage option because it serves the residency and prefix-cache goals.
- The branch may keep a fallback upload path for nonresident or debug-only cases, but that path must not define the fast path.

## Evidence We Still Need

1. A tensor-by-tensor equivalence test between `SwizzleRowMajorNvfp4ScalesForExecution()` and vLLM `swizzle_blockscale()` for representative expert shapes and padding cases.
2. A golden test proving that our descriptor-to-prepared-view conversion matches vLLM’s `prepare_nvfp4_moe_layer_for_fi_or_cutlass()` output for the relevant backend(s).
3. A hard answer on whether `TryUploadNvfp4Weight()` is ever reachable in the intended fast path on RTX 5090.
4. A clear ownership model for the expert-weight state:
   - what is stored once at load time
   - what is cached per backend
   - what is permitted to exist only as fallback
5. Verification that the manifest emitted by this branch already contains kernel-native packed bytes and scale bytes, rather than relying on runtime reshaping.

## Sequencing Advice

1. Lock the prepared-view contract first.
   Make the hot path consume only resident/prepared weights. If a weight is not prepared, fail over to a slower backend explicitly.

2. Prove layout equivalence next.
   Add golden tests against vLLM for scale swizzling and weight-prep output before refactoring the ownership model.

3. Collapse the ownership split after that.
   Once the layout is proven, reduce the number of overlapping prepared-weight wrappers or make the boundaries between them explicit and non-overlapping.

4. Remove runtime work from the fast path last.
   Any remaining host fallback, descriptor reinterpretation, or on-demand upload should be quarantined to fallback/debug paths only.

## Implementation Tasks

- [ ] Add a golden test for C++ scale swizzling versus vLLM `swizzle_blockscale()`
- [ ] Add a golden test for prepared MoE weight output versus vLLM `prepare_nvfp4_moe_layer_for_fi_or_cutlass()`
- [ ] Remove on-demand routed-weight uploads from the intended fast path in `ResolveRoutedExpertWeightViews()`
- [ ] Make fast-path support depend on prepared/resident expert weights, not on just-in-time `DeviceNvfp4Weight::Upload()`
- [ ] Decide whether `DeviceNvfp4Weight` is a fallback-only adapter or a first-class prepared-weight owner
- [ ] Decide whether `MonolithicNvfp4ExpertWeights` should become the single source of truth for resident expert views
- [ ] Audit `RunNvfp4RowMajorFp32AccumToDevice()` for host fallbacks that should be eliminated from the polished path
- [ ] Confirm that manifest generation emits kernel-native packed bytes and block-scale bytes, not raw shapes that still need runtime reshaping

