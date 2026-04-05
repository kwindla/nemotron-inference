# Plan: MoE Execution-Surface Alignment

Project directory: `./proj-2026-04-03-1816`

**Superseded status (2026-04-04):** Background analysis only.
Follow `proj-2026-04-04-0133/PLAN.md` for active execution sequencing and
verification requirements. This file is not the authoritative source for the
current multi-token BF16 closure order.

## Priority

Closing accidental divergences is the critical priority now. The MoE surface should converge toward one polished, vLLM-shaped execution path, with legacy paths kept only when they are still justified by correctness, unsupported shapes, or measured performance.

## Current State

- The branch already has a real backend abstraction in `runtime/include/nemotron/moe_backend.h`: `Supports(...)`, `PrepareWeights(...)`, and `Run(...)`.
- `ExpertLayerSlice::Impl` currently registers four backends in order: `unified_fused`, `decode_cublaslt`, `fused_decode`, `batched_cublaslt`.
- Routing is device-side first. `RunBackendExpertSelection()` produces GPU `topk_ids` and `topk_weights`, and the layer iterates the backend list until one runs successfully.
- The hot path still has a host-routed direct-MoE fallback, including per-expert staging and GEMM loops in `RunMoeDirectDecodeViaCublaslt()`.
- The unified fused path exists, but it is still gated by residency mode and opt-in env state.

## vLLM Reference Shape

- vLLM treats MoE as one layer abstraction with a quant-method / experts split, not as several sibling execution surfaces. See `third_party/vllm/vllm/model_executor/layers/fused_moe/layer.py` and `fused_moe_method_base.py`.
- vLLM’s unquantized NVFP4 path selects a backend once, prepares weights at load time, and then runs through a single method object. See `third_party/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` and `unquantized_fused_moe_method.py`.
- vLLM’s grouped top-k router returns tensor-native `topk_weights` and `topk_ids` directly. See `third_party/vllm/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`.
- vLLM’s NVFP4 preparation is load-time work, not a runtime side path. See `third_party/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py` and the NVFP4 MoE oracle files.

## Alignment

- The branch is aligned with vLLM on the core routing contract: device-side expert selection feeds the MoE executor.
- The branch is aligned with vLLM on load-time weight preparation in spirit: `PrepareWeights()` is the right hook, and monolithic / resident views are being cached ahead of execution.
- The branch is aligned with vLLM on the idea of backend selection under one layer abstraction, even though the implementation is still more fragmented than vLLM’s method object model.

## Divergences That Look Intentional

- The custom prefix-cache and reusable-state goal justifies a stronger runtime control plane than vLLM’s standard MoE stack.
- Residency gating for the unified fused backend is defensible as a temporary safety boundary while memory behavior is being stabilized.
- Keeping a correctness-oriented fallback path is justified until every fast path has been proven on hardware and against the vLLM baseline.

## Divergences That Look Accidental

- `DecodeCublasLtBackend` and `FusedDecodeBackend` overlap heavily. Both target `token_count == 1`, but they encode two separate implementation strategies for the same logical path.
- `FusedDecodeBackend` is especially suspect because the internal benchmark results already show it losing badly to the unified / cuBLASLt path. That makes it look like leftover experimentation, not a polished production surface.
- The repeated `PrepareWeights()` logic across the four backends is duplication drift. vLLM keeps weight preparation closer to a single method object; this branch is repeating the same readiness logic in each backend class.
- The host-routed direct-MoE fallback still looks like a mainstream implementation path in the code, not just a last-resort safety valve.
- `SupportsUnifiedFusedBackend()` is more conservative than a pure capability check because it requires residency state. That may be correct for now, but it also risks hiding a backend that should be viable if the prepared-weight model is cleaned up.

## Judgment On Legacy Backends

- `UnifiedFusedBackend` should remain the primary target. It is the closest match to the intended end state.
- `BatchedCublasLtBackend` is still defensible as a fallback for unsupported shapes or residency gaps, but it should stop being the mental center of the design.
- `DecodeCublasLtBackend` is acceptable only as a transitional compatibility path until the unified backend fully subsumes `token_count == 1`.
- `FusedDecodeBackend` looks like the strongest candidate for removal or demotion after the unified backend is validated across the relevant token-count matrix.

## Concrete Implementation Tasks

- Collapse the selection story so there is one canonical MoE execution surface for `token_count >= 1`, with explicit fallback ordering rather than multiple competing “real” backends.
- Extract shared `PrepareWeights()` logic into a common helper or base path so backend classes only own the parts that are actually backend-specific.
- Make the unified fused backend the first-class implementation, not an opt-in special case behind residency flags.
- Decide whether `FusedDecodeBackend` still earns its keep. If it does not beat the unified path, remove it from the default surface.
- Decide whether `DecodeCublasLtBackend` is a needed transitional reference path or just duplicate maintenance burden.
- Reduce the prominence of host-routed direct-MoE fallback so it is clearly a fallback, not an alternate route that the code naturally drifts into.
- Tighten support predicates so they reflect actual backend capability, not incidental load-state or historical implementation boundaries.
- Add assertions or tests that prove the selected backend for each token-count regime is the intended one, especially for `1`, `32`, `64`, and `128` tokens.

## Missing Evidence

- We do not yet have a final proof that the unified fused backend is the best or only production path for `token_count == 1`.
- We do not yet have a strong reason to keep `FusedDecodeBackend` if the unified backend matches or beats it on both correctness and latency.
- We do not yet have an explicit benchmark matrix showing which backend should own each token-count range after the cleanup.
- We do not yet have evidence that removing or demoting the host-routed direct-MoE fallback would break any supported configuration.
- We do not yet have a crisp answer on whether residency should remain a gating condition for backend selection or should be moved fully into prepared-weight handling.

## Sequencing

- First, validate backend ownership on hardware: unified fused versus decode-only versus batched fallback.
- Second, remove the clearly redundant single-token path if the data supports that move.
- Third, factor shared backend-preparation logic so the remaining backends do not drift independently.
- Fourth, simplify the selection policy so the runtime always converges on the same intended backend for the same token-count regime.
- Fifth, re-run parity and performance checks after the surface has been collapsed.

## Review Question

If a backend does not improve correctness, latency, or compatibility, it should not survive this cleanup just because it exists today.
