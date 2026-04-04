# Plan: Non-MoE Linear and Quantization Alignment Review

Project directory: `./proj-2026-04-03-1816`

## Priority

Closing accidental divergences is the critical priority now. The non-MoE linear and quantization path should converge toward one polished, vLLM-shaped implementation, with fallback/reference behavior kept only when it is still justified by correctness or lack of a native backend.

## Current State

- `UploadedLinearOp` is the general runtime wrapper for dense and NVFP4 GEMMs in [runtime/src/backend/linear_op.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/linear_op.cpp), with weight uploads handled by [runtime/src/backend/dense_weight.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/dense_weight.cpp) and [runtime/src/backend/nvfp4_weight.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/nvfp4_weight.cpp).
- `ScaledFp8LinearOp` is a special-purpose bridge in [runtime/src/backend/scaled_fp8_linear.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/scaled_fp8_linear.cu): it dequantizes FP8 weights to host fp32 at creation time, uploads the fp32 weights to device memory, quantizes activations on device, and then runs a dense fp32 GEMM.
- `linear_op.cpp` uses an environment-gated device fastpath plus device reference fallback paths instead of a more explicit quantization-method abstraction.
- The non-MoE users are real and important: attention projections and LM head use `UploadedLinearOp` in [runtime/src/backend/attention_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/attention_layer.cpp), and Mamba uses `ScaledFp8LinearOp` / `UploadedLinearOp` in [runtime/src/backend/mamba_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/mamba_layer.cpp).

## vLLM Reference Shape

The closest vLLM references are:

- [third_party/vllm/vllm/model_executor/layers/linear.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/linear.py)
- [third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)
- [third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py)
- [third_party/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py)
- [third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/transform/linear.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/transform/linear.py)
- [third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/transform/schemes/linear_qutlass_nvfp4.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/quantization/compressed_tensors/transform/schemes/linear_qutlass_nvfp4.py)

The important properties of the vLLM shape are:

- quantization is a first-class layer contract, not just a kernel-family tag
- weight preparation happens after loading and is scheme-specific
- execution selects among supported kernels or backends without collapsing the quantization contract to fp32
- transforms and quantization methods are layered, not interleaved with runtime policy

## Alignment

- Load-time preparation is already present in spirit. `DeviceDenseWeightFp32::Upload()` and `DeviceNvfp4Weight::Upload()` do explicit upload/preparation work before runtime execution.
- NVFP4 handling is relatively close to the vLLM model: weights are made device-resident and scales are prepared up front, which matches the broad direction of vLLM's post-load processing.
- The current code already has useful plan/fallback separation for observability, which is a legitimate transitional feature while the native path is still being tightened.

## Accidental Divergences

- `ScaledFp8LinearOp` is not a native FP8 linear implementation. It converts the packed FP8 weight tensor into host fp32 during `Create()`, stores a host fp32 copy, and then executes as a dense fp32 GEMM after quantizing activations on device. That is a major semantic drift from the vLLM FP8 contract.
- `linear_op.cpp` encodes backend selection as a runtime policy shim with env flags and fallback ordering, rather than as an explicit quantization-method / capability object. That makes the code harder to reason about and easier to fragment.
- The current design still treats dense, NVFP4, and scaled-FP8 linear as separate execution shims instead of one coherent non-MoE linear surface with backend-specific preparation.
- The fallback/reference paths are still part of the main execution story. They are useful as safety rails, but they are too prominent for a polished implementation.
- `DeviceDenseWeightFp32::Upload()` normalizes bf16 and fp8 source weights into fp32 immediately. That is acceptable for truly unquantized dense execution, but it also erases source dtype information and makes it too easy for quantized paths to silently devolve into fp32 bridges.

## Intentional Divergences

- A C++ runtime does not need to mirror vLLM's Python class structure exactly. The requirement is semantic alignment, not a copy of the class hierarchy.
- Reference fallback is acceptable while the native path is still being validated on hardware.
- Dense fp32 upload is fine for actual dense layers. The problem is not that dense weights are fp32 internally; the problem is using the same pattern to stand in for quantized execution.
- NVFP4 device preparation can remain runtime-specific if it stays backend-native and load-time oriented.

## Concrete Implementation Tasks

1. Define a single non-MoE linear-method abstraction for quantized and unquantized layers, so scheme selection is explicit instead of encoded in `GemmKernelFamily` and env flags.
2. Replace `ScaledFp8LinearOp`'s fp32 bridge with a native FP8 execution path that keeps FP8 weights and scale tensors native end-to-end.
3. Move host dequantization of FP8 weights out of the default path. If a compatibility path must remain, fence it off clearly as fallback only.
4. Make backend selection for dense, NVFP4, and FP8 layers capability-driven rather than policy-driven through global env switches.
5. Keep `DeviceDenseWeightFp32` and `DeviceNvfp4Weight` as load-time preparation helpers, but stop letting them substitute for a layer-level quantization abstraction.
6. Unify plan logging and fallback reporting so execution failures read as compatibility exceptions, not as alternate normal paths.
7. Add explicit tests for scheme preservation, backend selection, and numerical parity for every intended non-MoE linear mode.

## Missing Evidence

- No direct vLLM-vs-runtime comparison exists for non-MoE FP8 linear or generic NVFP4 linear on the current branch.
- No benchmark sweep exists for batch-size thresholds or shape thresholds in the non-MoE linear paths.
- No measurement exists for the startup and allocation cost of host dequantization in `ScaledFp8LinearOp`.
- No evidence shows that the current FP32 bridge is faster or more stable than a native FP8 path on RTX 5090.
- No regression matrix covers mixed dense, NVFP4, and FP8 non-MoE layers together.
- The current tests validate local correctness and planning behavior, but they do not prove vLLM-class parity or long-run stability.

## Ordering Advice

1. Freeze the semantics first: decide which linear modes are truly native, which are compatibility bridges, and which should be removed from the default path.
2. Reintroduce a native FP8 linear backend before polishing fallback behavior, so the implementation target is concrete.
3. Tighten scheme metadata and capability checks next, because later refactors depend on those being explicit.
4. Unify dense, NVFP4, and FP8 preparation and validation once the execution contract is settled.
5. Only then spend time on cleanup of logging, fallback presentation, and benchmark polish.

## Success Criteria

- FP8 linear no longer relies on host dequantization as its default execution model
- non-MoE linear mode is selected by explicit scheme/capability, not by incidental fallback order
- dense and quantized execution paths are easy to distinguish in code and logs
- reference fallbacks exist only as compatibility rails
- the implementation is close enough to vLLM that parity bugs are about kernel quality, not about missing abstractions
