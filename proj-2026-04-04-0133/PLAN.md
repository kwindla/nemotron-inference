# Plan: Adversarial vLLM Alignment and Gap Closure for Nano on RTX 5090

Project directory: `./proj-2026-04-04-0133`

## Context

The Nano-on-RTX-5090 branch is past the prototype stage but still carries accidental divergences from vLLM: host-side reconstruction in hot paths, env-gated backend policy, fragmented MoE execution surfaces, an FP8 bridge that dequantizes to fp32, and under-specified component contracts. This plan closes those gaps in priority order — freezing contracts and verification gates first, then collapsing hot-path fragmentation, then polishing ownership boundaries — so the runtime reads like a finished engine rather than a bring-up runtime that happens to pass tests. The exact-prefix whole-state cache model for KV + Mamba state remains the primary intentional architectural divergence, and any other surviving divergence must be explicit, scoped, and justified.

Design analysis: `proj-2026-04-03-1816/PLAN.md` and its 9 sub-plans.

## Steps

- [x] **1. Canonicalize verification gates and benchmark provenance**
  Split the verification story into 4 canonical gates: (a) local correctness / oracle replay, (b) exact-token vLLM parity, (c) internal performance, (d) external TTFT. Strengthen the oracle payload in `nano_save_prompt_oracle.cpp` by adding `manifest_path`, `build_dir`, `backend_flags` (all active `NEMOTRON_FORWARD_*` env vars), and `git_revision` fields to the JSON output, while also defining an optional deeper regression payload for runs that need more localization than boundary top-5 logits. Make the parity artifact explicit about prompt source, prompt token ids, mismatch index, and backend selection metadata. Make benchmark provenance non-optional in `bench_full_comparison.sh` — every artifact must record the exact binary path, build dir, manifest, model id, env vars that affect backend selection, and the vLLM revision/source tree used for external comparisons. Expose the MoE execution window parameter in `nano_prefix_cache_ttft_bench` so it can measure 32, 64, and 128-token tail cases instead of only hardcoded 32. Write a `docs/verification_gates.md` that defines each gate's purpose, inputs, pass criteria, artifact format, and the rule that correctness failures and performance regressions are reported separately.
  Key files: `testing/api/nano_save_prompt_oracle.cpp`, `proj-2026-04-03-0318/bench_full_comparison.sh`, `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py`, `proj-2026-04-03-2113/save_runtime_oracle.py`, `benchmarks/nano_prefix_cache_ttft/`, `docs/verification_gates.md`

- [x] **2. Freeze MoE routing contract in docs and tests**
  Define the canonical routing contract in `docs/moe_routing_contract.md`: tensor shapes (`topk_ids: int32[token_count, top_k]`, `topk_weights: float32[token_count, top_k]`), token-major ordering, deterministic tie-breaking semantics matching vLLM's `sorted` behavior in grouped-top-k, renormalization and `routed_scaling_factor` semantics, and the boundary between the canonical contract and expert-major compaction adapters. Add a parity test that compares our `RunDeviceExpertSelection()` output against vLLM `grouped_topk()` (from `third_party/vllm/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`) for representative Nano inputs including tie-sensitive and correction-bias cases. Remove host-side `BuildExpertRoutingTable()` reconstruction from the fast prefill path in `expert_layer.cpp` — keep it only behind an explicit fallback/adapter gate. Add a regression test that asserts the main fast path does not copy routing state to host outside of debug or fallback modes.
  Key files: `docs/moe_routing_contract.md`, `runtime/src/backend/expert_layer.cpp:1170`, `runtime/src/backend/fused_moe_decode.cu:438`, `runtime/src/backend/fused_moe_prefill.cu:220`, `testing/backend/expert_routing_device_test.cpp`

- [x] **3. Lock NVFP4 prepared-weight contract and golden tests**
  Add a golden test comparing our `SwizzleRowMajorNvfp4ScalesForExecution()` against vLLM's `swizzle_blockscale()` (from `third_party/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:272`) for representative expert shapes, padding cases, and the exact Nano dimensions. Add a second golden test comparing our prepared MoE weight output (from `MonolithicNvfp4ExpertWeights`) against vLLM's `prepare_nvfp4_moe_layer_for_fi_or_cutlass()` (from `third_party/vllm/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py:192`). Remove on-demand `TryUploadNvfp4Weight()` from the routed-expert fast path in `ResolveRoutedExpertWeightViews()` — if routed weights are not prepared/resident, the fast path must reject and fall back to a slower backend explicitly. Make `MonolithicNvfp4ExpertWeights` the single source of truth for resident routed-expert views, but keep `DeviceNvfp4Weight` as a first-class owner for shared experts and generic non-MoE NVFP4 linear until those callers are explicitly re-homed. Define the ownership model explicitly: what is prepared once at load time, what can be cached per backend, and what is allowed to exist only as fallback. Audit `RunNvfp4RowMajorFp32AccumToDevice()` in `nvfp4_gemm_runner.cpp` and remove host fallback reads from the prepared routed-expert path. Confirm that manifest generation emits kernel-native packed bytes and block-scale bytes rather than raw shapes that still require runtime reshaping.
  Key files: `runtime/src/backend/expert_layer.cpp:1284`, `runtime/src/backend/nvfp4_weight.cpp`, `runtime/src/backend/monolithic_expert_weights.cu`, `runtime/src/backend/nvfp4_scale_layout.cpp`, `runtime/src/backend/nvfp4_gemm_runner.cpp:698`, `testing/backend/nvfp4_*`

- [x] **4. Collapse MoE backend fragmentation around UnifiedFusedBackend**
  Make `UnifiedFusedBackend` the first-class MoE execution path — it should be selected by default when the layer is prepared and resident, without requiring opt-in env flags (`NEMOTRON_FORWARD_UNIFIED_FUSED`). Benchmark `FusedDecodeBackend` against unified fused for `token_count == 1` on RTX 5090 with the Nano model; if unified matches or beats it, remove `FusedDecodeBackend` entirely. Do the same for `DecodeCublasLtBackend`. Extract shared `PrepareWeights()` logic from the 4 backends into a common preparation helper or base class in `moe_backend.h`, so backends only own backend-specific prep. Tighten `Supports()` predicates to reflect actual backend capability (kernel + shape + residency) rather than incidental load-state or historical boundaries. Reduce the prominence of host-routed `RunMoeDirectDecodeViaCublaslt()` fallback — it should be clearly last-resort, not an alternate route that code naturally drifts into. After each backend removal, run the canonical local-correctness gate and canonical parity gate defined in step 1; `verify_correctness.sh` may remain the implementation of the local correctness gate until that gate is formalized, but backend deletion should depend on the gate definitions rather than on one legacy script name.
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/include/nemotron/moe_backend.h`, `runtime/src/backend/fused_moe_decode.cu`, `runtime/src/backend/fused_moe_prefill.cu`, `proj-2026-04-03-0318/verify_correctness.sh`, `docs/verification_gates.md`

- [x] **5. Add explicit attention backend policy and cuDNN plan caching**
  Introduce an `AttentionBackendPolicy` with `Supports(config, token_count, device_sm)` / `Select(config, token_count)` semantics modeled after vLLM's `get_attn_backend()` selector pattern (from `third_party/vllm/vllm/v1/attention/selector.py`). The policy replaces the current ad hoc branching in `attention_layer.cpp` where cuDNN is chosen by handle existence and fallbacks are chosen by env vars (`NEMOTRON_FORWARD_ATTENTION_PRODUCTION`, `NEMOTRON_FORWARD_ATTENTION_SCALAR_FALLBACK`). Cache or prebuild `CudnnPagedAttentionPlan` objects per validated `CudnnPagedAttentionConfig` or an equivalent full execution-shape key that includes all graph-defining dimensions and flags (`batch_size`, `max_query_tokens`, `max_kv_tokens`, head counts, head dim, tokens per page, container page count, page-table entries, dtype, causal, stats mode), instead of recreating them on every call in `cudnn_paged_attention.cpp`. Keep the Nano-specific decode kernel only if measurement shows it beats cuDNN and the general device fallback for the exact Nano shapes. Remove env-var policy leakage from hot attention paths — production backend selection should be deterministic from config, not from env vars. Add explicit allocator invariants for page allocation, restore, reset, and release as tested contracts, and add explicit tests for snapshot restore, exact cache hits, page release, and invalid page-table / geometry rejection.
  Key files: `runtime/src/backend/attention_layer.cpp`, `runtime/src/backend/cudnn_paged_attention.cpp`, `runtime/include/nemotron/attention_layer.h`, `runtime/include/nemotron/cudnn_paged_attention.h`, `runtime/src/attention/paged_kv_cache.cpp`, `testing/backend/attention_layer_test.cpp`, `testing/backend/cudnn_paged_attention_test.cpp`

- [x] **6. Make Mamba production path explicit and fence debug scaffolding**
  Make the production Mamba execution path in `mamba_layer.cpp` unambiguously the default: fused decode for single-token, sequential conv+SSD+norm for prefill. Move the compare/debug scaffolding (controlled by `NEMOTRON_FORWARD_COMPARE_FUSED_MAMBA` and `NEMOTRON_FORWARD_COMPARE_FUSED_MAMBA_ACTIVE`) behind a compile-time or clearly fenced debug gate so it cannot leak into the hot path. Tighten the runner/state contract so recurrent-buffer offsets and handoff are explicit rather than spread implicitly across the runner and kernels. Verify that `RunMambaConvPrefill` and `RunMambaSsdPrefill` preserve correct recurrent state across continuation by adding a test that restores a cached Mamba prefix and continues generation, comparing cold vs restored execution for the same prompt boundary. Add at least one oracle test that exercises both conv state and SSM state round-tripping through prefix cache restore plus multi-token continuation. Add an isolated Mamba benchmark for cold prefill, restored-prefix prefill, and single-token decode so any remaining prefill/decode split is justified by data rather than by current implementation shape.
  Key files: `runtime/src/backend/mamba_layer.cpp`, `runtime/include/nemotron/mamba_layer.h`, `runtime/include/nemotron/fused_mamba_decode.h`, `testing/backend/mamba_layer_oracle_test.cpp`, `testing/api/state_snapshot_test.cpp`

- [ ] **7a. Fused add+RMSNorm kernel (BF16 I/O, FP32 internal)**
  Write a CUDA kernel matching vLLM's `fused_add_rms_norm` (`third_party/vllm/vllm/model_executor/layers/layernorm.py:71`). Takes BF16 hidden + BF16 residual + FP32 norm weight + float epsilon. Internally: FP32 add, FP32 variance, FP32 normalize, FP32 weight multiply. Outputs: BF16 normalized result, BF16 updated residual (in-place on residual buffer). Add to `runtime/src/backend/primitive_ops.cu` alongside existing `RmsNormFp32`. Test by comparing against separate FP32 RmsNorm + ResidualAdd with round-trip BF16 truncation. This kernel is the foundation — every layer uses it.
  Key files: `runtime/src/backend/primitive_ops.cu`, `runtime/include/nemotron/primitive_ops.h`, `testing/backend/primitive_ops_test.cpp`

- [ ] **7b. BF16 embedding lookup**
  Stop converting BF16 embedding weights to FP32 during upload in `embedding_table.cu:101-109`. Keep BF16 on device. Add `LookupEmbeddingRowsBf16()` that reads BF16 weights and writes BF16 output. The existing FP32 path can remain as fallback for FP32-stored embeddings. The Nano manifest stores embeddings as BF16, so this is the production path.
  Key files: `runtime/src/backend/embedding_table.cu`, `runtime/include/nemotron/embedding_table.h`, `testing/backend/embedding_lookup_test.cpp`

- [ ] **7c. BF16 dense GEMM path**
  Add a cuBLASLt dense GEMM path with `CUDA_R_16BF` data types and `CUBLAS_COMPUTE_32F` compute (FP32 accumulation, BF16 I/O). This matches `torch.nn.functional.linear` behavior for BF16 inputs on SM120. Add `DeviceDenseWeightBf16` that keeps BF16 weights on device without FP32 promotion. Keep the existing FP32 path for layers that genuinely need FP32 (MoE gate). Update `dense_gemm_runner.cpp` to select BF16 or FP32 based on the descriptor's storage_dtype.
  Key files: `runtime/src/backend/dense_gemm_runner.cpp`, `runtime/src/backend/dense_weight.cpp`, `runtime/include/nemotron/dense_weight.h`, `testing/backend/dense_gemm_runner_test.cpp`

- [ ] **7d. BF16 hidden/residual buffers and layer interfaces**
  Convert `RequestExecutionContext` hidden, residual, and scratch buffers from `DeviceTensorFp32` to `DeviceTensorBf16`. Change layer `Run()` signatures in `attention_layer.h`, `mamba_layer.h`, `expert_layer.h` to accept `DeviceTensorBf16` I/O. Update `SingleTokenForwardModel::RunTokens()` to pass BF16 tensors. Each layer internally: (1) fused add+RMSNorm from 8a at entry, (2) BF16 dense GEMM from 8c for projections, (3) NVFP4 GEMM with BF16→FP32 cast for activation packing, (4) BF16 output. Mamba recurrent state stays FP32 — cast at boundaries. This is the largest change and depends on 8a-8c being done first.
  Key files: `runtime/src/backend/request_context.cpp`, `runtime/include/nemotron/request_context.h`, `runtime/src/backend/attention_layer.cpp`, `runtime/src/backend/mamba_layer.cpp`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/api/single_token_forward_model.cpp`, `runtime/include/nemotron/attention_layer.h`, `runtime/include/nemotron/mamba_layer.h`, `runtime/include/nemotron/expert_layer.h`

- [x] **7e. Residual-add pattern alignment and bootstrap cleanup**
  Remove eager residual add from `attention_layer.cpp:1160`, `mamba_layer.cpp:672`, `expert_layer.cpp:1820`. Each layer returns its output in BF16; the fused add+RMSNorm at the start of the next layer handles add + norm. Split `RuntimeEnvironment::BuildFromManifestFile()` into validation, planning, and assembly phases. Replace heuristic layer-role inference in `ModelSchedule` with explicit manifest roles.
  Key files: `runtime/src/backend/attention_layer.cpp`, `runtime/src/backend/mamba_layer.cpp`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/api/runtime_environment.cpp`, `runtime/src/loader/model_schedule.cpp`

- [ ] **7f. BF16 pipeline verification and vLLM parity**
  Regenerate oracle and run vLLM parity — target 16/16 token match. Run full ctest (currently `57/57` on this branch). Run decode benchmark — expect BF16 GEMMs to be faster than FP32 (less memory bandwidth). Run `verify_correctness.sh` — Route A must match Route C. Counters: `dense_reference_fallback=0`, `nvfp4_reference_fallback=0`.
  Key files: `proj-2026-04-03-0318/verify_correctness.sh`, `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py`
  Analysis: `proj-2026-04-04-0133/bf16_pipeline_analysis.md`

- [ ] **7g. Resolve the multi-token BF16 execution contract**
  The current BF16 production surface is correct but not fully settled: `attention_layer.cpp`, `expert_layer.cpp`, and `mamba_layer.cpp` all handle `token_count > 1` by slicing the input into single-token views and recursively running the one-token path. Treat this as an explicit alignment item, not an incidental implementation detail. First, make the layer contract observable: add explicit capability / fallback boundaries and a measurable signal (counter, trace field, or benchmark artifact field) that records when multi-token requests are satisfied by sequential row replay instead of a native multi-token path. Then decide whether that sequential replay is (a) a temporary correctness bridge that must be replaced with true multi-token production paths, or (b) an intentional Nano-on-5090 contract that remains only if measurement and code simplicity justify it. If it is temporary, restore native multi-token execution for the affected layers and remove the recursive bridge from the hot path. If any part remains intentional, document the exact surviving divergence, benchmark the prefill / resumed-prefix cost, and fence the bridge so it is explicit rather than silently becoming the default.

  Acceptance criteria:
  - local correctness, oracle replay, and exact-token vLLM parity remain green
  - `bench_full_comparison.sh` artifacts are captured before and after the change, with the prefix-cache TTFT matrix kept as the primary comparison record
  - a focused `nano_prefix_cache_ttft_bench` sweep for `tail_token_count == moe_prefill_window_tokens == 32/64/128` is recorded for the branch
  - step `7g` cannot close while the production multi-token BF16 path still silently falls back to row replay; any surviving replay path must be explicitly gated, documented, and visible in artifacts

  Required execution order inside `7g`:
  1. Add observability first so row replay, native multi-token execution, and fallback routing are visible in traces and benchmark artifacts.
  2. Remove the outer multi-token row-replay bridge in attention and validate the existing native multi-token prefill path against the Tier 1 and `7g` benchmark gates.
  3. Remove the outer multi-token row-replay bridge in expert while preserving top-level MoE window chunking as a memory policy; keep host-routing and other non-native routes as explicit fallback only.
  4. Remove the outer multi-token row-replay bridge in Mamba while preserving the explicit inner split between multi-token prefill and single-token fused decode.
  5. Run the full `7g` verification matrix and only then decide whether any residual replay path is justified enough to remain as an intentional divergence.

  Implementation notes and reference alignment:
  - **Do `7e` before re-baselining multi-token behavior.** The local vLLM Nemotron-H reference carries `hidden_states` and `residual` separately through every decoder layer and performs fused add+RMSNorm at layer entry, with the final add+norm only after the last layer. That keeps `7e` logically ahead of `7g`, because we should not bless a multi-token contract on top of a residual handoff we already know is still being aligned.
  - **Attention should remove the outer row-replay bridge, not invent a second prefill architecture.** Below the current `token_count > 1` recursion in `attention_layer.cpp`, the layer already allocates token-count-shaped BF16 buffers, scatters KV for `token_count` tokens, selects an attention backend with `max_query_tokens = token_count`, and runs paged attention over the full token block. The local vLLM reference on RTX 5090 uses the `FLASHINFER` attention backend and explicitly builds separate prefill and decode metadata instead of replaying one token at a time. For this branch, the target is to delete the recursive wrapper and validate the existing native multi-token prefill path against the current backend policy and TTFT gates.
  - **Expert should also remove the outer row-replay bridge and keep MoE window chunking above a native multi-token layer path.** Below the current `token_count > 1` recursion in `expert_layer.cpp`, the direct-MoE path already computes grouped top-k for the whole token block and can dispatch `token_count > 1` work through `UnifiedFusedBackend` / `RunFusedMoePrefillPath`. The local vLLM reference flattens all prompt tokens into one `(num_tokens, hidden_dim)` tensor, runs grouped top-k token-major, and executes shared+routed experts over that batch; it does not row-replay prompt tokens. Our top-level MoE window chunking in `single_token_forward_model.cpp` can remain as a memory-bounding policy, but each chunk must enter `ExpertLayerSlice::Run()` as real multi-token work, with the host-routing adapter remaining fallback-only and visible in artifacts.
  - **Mamba is different: remove the outer row-replay bridge, but keep the inner prefill/decode split.** In the local runtime, `MambaLayerSlice::Run()` already contains a real multi-token prefill path (`RunMambaConvPrefill` + `RunMambaSsdPrefill` + gated group norm) and a separate single-token fused decode path. The accidental divergence is the outer `token_count > 1` recursive wrapper, not the existence of a dedicated prefill algorithm. The local vLLM reference follows the same high-level shape: its Mamba metadata builder splits decodes and prefills, and `MambaMixer2` executes dedicated prefill sequence transforms plus a separate decode update path instead of forcing multi-token requests through the decode kernel. So `7g` for Mamba is contract cleanup and explicit measurement, not “make prefill use fused decode.”
  - **Observability is part of the deliverable, not optional scaffolding.** Add a per-layer or per-request signal that distinguishes native multi-token execution from row replay for attention, expert, and Mamba. The signal must land in benchmark or trace artifacts so `bench_full_comparison.sh` and the focused prefix TTFT sweep can prove whether the bridge is gone, quarantined, or still on the hot path.
  - **Reference scope for this branch is the actual local RTX 5090 vLLM path, not generic upstream possibilities.** The working reference here is `third_party/vllm` under `docs/vllm_rtx5090.md`, with `FLASHINFER` attention and `FLASHINFER_CUTLASS` NVFP4 MoE on the Nano checkpoint. `7g` should converge toward that batch-oriented prefill/decode execution shape while preserving the intentional exact-prefix cache divergence and our local single-process runtime boundary.

  Do not broaden oracle re-baselining or start cache-allocator work until this contract is closed.
  Key files: `runtime/src/backend/attention_layer.cpp`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/mamba_layer.cpp`, `runtime/src/api/single_token_forward_model.cpp`, `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-03-0318/bench_full_comparison.sh`

- [ ] **8. Port oracle fixture generation to Nano and regenerate slice-level golden tests**
  Port `tools/oracle/dump_expert_layer_fixture.py` to handle Nano's non-latent MoE architecture (no `fc1_latent_proj`/`fc2_latent_proj`, NVFP4-packed expert weights instead of scalar FP8). Port `tools/oracle/dump_mamba_layer_fixture.py` to handle Nano's per-channel weight scales (tensor, not scalar) and NVFP4 weight formats. Port `tools/oracle/dump_mamba_update_fixture.py` for Nano dimensions (num_heads=64 not 128). Generate fixtures for representative Nano layers: at least 2 expert layers (e.g., layers 1 and 8), 2 Mamba layers (e.g., layers 0 and 9), and 1 attention layer (e.g., layer 5). Re-register the corresponding tests in `testing/CMakeLists.txt` and verify they pass against the generated fixtures. Fixtures must be generated against the final BF16 pipeline so they encode the correct precision model.
  Key files: `tools/oracle/dump_expert_layer_fixture.py`, `tools/oracle/dump_mamba_layer_fixture.py`, `tools/oracle/dump_mamba_update_fixture.py`, `testing/backend/expert_layer_oracle_test.cpp`, `testing/backend/mamba_layer_oracle_test.cpp`, `testing/CMakeLists.txt`

- [ ] **9. Revisit cache allocator and page/snapshot ownership**
  Decide the fate of `prompt_head` in `prefix_cache.cpp`: either document it as an explicit non-hot-path feature with tests proving the committed-head path remains default, or remove it from the main cache API and production lookup path. Replace the linear eviction scan (full scan over nodes by `last_access_tick`) and linear global-root lookup with an explicit prefix index — keep exact-prefix semantics but make lookups and evictions O(1) or O(log n) instead of O(n). Replace `ReusableStateArena`'s per-snapshot `cudaMalloc`/`cudaFree` pattern in `reusable_state.cpp` with a pooled slab design — preallocate a budget of snapshot slots and recycle them without CUDA allocator calls. Measure snapshot publish / restore overhead, allocation latency, and release latency before and after to prove the improvement. Reconcile page ownership between `RequestExecutionContext` (owns live KV pages via `PagedKvCacheArena`) and the cache (owns copied snapshots via `ReusableStateArena`) — benchmark and document a pooled copy-based model first, then move toward shared page ownership with refcounts only if the data justifies that added complexity. Harden restore semantics: record prefix length in the snapshot descriptor and make restore reject mismatched length/layout cases with clear diagnostics.
  Key files: `runtime/src/cache/prefix_cache.cpp`, `runtime/src/cache/reusable_state.cpp`, `runtime/src/api/state_snapshot.cpp`, `runtime/include/nemotron/prefix_cache.h`, `runtime/include/nemotron/reusable_state.h`, `runtime/src/backend/request_context.cpp`, `testing/cache/prefix_cache_test.cpp`, `testing/cache/reusable_state_test.cpp`

- [ ] **10. Multi-turn prefix reuse regression and final verification sweep**
  Add an end-to-end multi-turn prefix-reuse regression test that exercises committed heads, global roots, and append-only continuation in one flow — restore a cached prefix (with both KV and Mamba state), continue generation for multiple tokens, publish a new committed head, then restore again and verify the token stream matches cold execution. This test must cover: (a) conversation committed-head fast path, (b) global root fallback, (c) multi-token prefill after restored prefix, (d) Mamba state round-trip correctness through the full cache lifecycle, and (e) `prompt_head` behavior as an explicit non-hot-path case if `prompt_head` survives step 9. Run the complete verification sweep: all 4 canonical gates from step 1 (local correctness, vLLM parity, internal performance, external TTFT) against the final codebase. Verify that every surviving intentional divergence from vLLM is explicit, documented, and justified, with the exact-prefix whole-state cache model remaining the primary architectural divergence. Document all remaining intentional divergences in `docs/intentional_divergences.md`.
  Key files: `testing/api/`, `proj-2026-04-03-0318/verify_correctness.sh`, `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py`, `docs/intentional_divergences.md`

## Verification Policy

### Tier 1: every step (Codex runs as part of delegation)

- `cmake --build build -j$(nproc)` — full build, not just touched targets
- `ctest --test-dir build --output-on-failure` — full ctest suite
- Step-specific tests called out in the delegation prompt

If any Tier 1 test fails, Codex must diagnose and fix before reporting done.
Pre-existing failures must be called out explicitly so we can distinguish
regressions from known issues.

### Tier 2: phase boundaries (we run manually after committing)

Run after steps 4, 7f, 7g, 8, and 10 — or any time a step touches the hot execution path.

- `nano_save_prompt_oracle` — generate a fresh oracle, compare against baseline
- `proj-2026-04-03-0318/verify_correctness.sh` — local correctness gate
- `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py` — exact-token vLLM parity
- `proj-2026-04-03-0318/bench_decode_backends.sh` — decode performance regression
- `proj-2026-04-03-0318/bench_full_comparison.sh` — full benchmark matrix

Results go into `proj-2026-04-04-0133/tier2-results/` with timestamps.

### Enforcement

- Codex delegation prompts must include: "Run `cmake --build build -j$(nproc)`
  and `ctest --test-dir build --output-on-failure` after all changes. Report
  any failures with full output."
- Reviewer (us) must run Tier 2 before marking a phase boundary done.
- Any performance regression > 5% in Tier 2 benchmarks blocks the next phase.

## Verification Commands

Use the following exact commands for step-by-step verification on this branch.
These commands are the canonical runbook unless a step explicitly requires
additional targeted coverage.

### Standard Environment

Run this shell setup before any Tier 1 or Tier 2 verification:

```bash
cd /home/khkramer/src/nemotron-inference
export NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json
export NEMOTRON_BUILD_DIR=build
export NEMOTRON_TEST_BUILD_DIR=build
export NEMOTRON_BENCH_BUILD_DIR=build
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_DIR="proj-2026-04-04-0133/tier2-results/${STAMP}"
mkdir -p "${ARTIFACT_DIR}"
```

### Tier 1: Every Step

Run these after every plan step, even for apparently local changes:

```bash
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

If the step primarily touches one execution surface and you want the shorter
targeted gate before full `ctest`, run:

```bash
ctest --test-dir build --output-on-failure -R \
  'full_forward_manifest_smoke_test|nano_16_token_correctness_test|nano_save_prompt_oracle|prefill_prefix_oracle_test|single_token_decode_oracle_test|expert_layer_oracle_test|mamba_layer_oracle_test'
```

### Tier 2: Phase Boundaries and Hot-Path Changes

Run these after steps 4, 7f, 7g, 8, and 10, and after any change that touches the
hot execution path.

Local correctness gate:

```bash
proj-2026-04-03-0318/verify_correctness.sh \
  --manifest "${NEMOTRON_FORWARD_MANIFEST}" \
  --build-dir build \
  --artifact-dir "${ARTIFACT_DIR}/verify_correctness"
```

Runtime oracle generation:

```bash
python3 proj-2026-04-03-2113/save_runtime_oracle.py \
  --build-dir build \
  --manifest "${NEMOTRON_FORWARD_MANIFEST}" \
  --prompt-name short_chat \
  --decode-token-count 16 \
  --output "${ARTIFACT_DIR}/nano_oracle.json" \
  | tee "${ARTIFACT_DIR}/nano_oracle.stdout.txt"
```

Exact-token vLLM parity:

```bash
PYTHONPATH=proj-2026-04-03-2113:${PYTHONPATH:-} \
python3 proj-2026-04-03-2113/compare_vllm_runtime_oracle.py \
  --oracle "${ARTIFACT_DIR}/nano_oracle.json" \
  --runtime-build-dir build \
  --manifest "${NEMOTRON_FORWARD_MANIFEST}" \
  --artifact-output "${ARTIFACT_DIR}/vllm_parity.json" \
  | tee "${ARTIFACT_DIR}/vllm_parity.stdout.txt"
```

Decode backend comparison:

```bash
proj-2026-04-03-0318/bench_decode_backends.sh \
  --manifest "${NEMOTRON_FORWARD_MANIFEST}" \
  --build-dir build \
  --artifact-dir "${ARTIFACT_DIR}/decode_backends" \
  --mode steady-state \
  --decode-tokens 16
```

Direct decode benchmark sanity check:

```bash
NEMOTRON_FORWARD_MANIFEST="${NEMOTRON_FORWARD_MANIFEST}" \
./build/benchmarks/nano_fused_decode/nano_fused_decode_bench \
  | tee "${ARTIFACT_DIR}/nano_fused_decode.stdout.txt"
```

Full benchmark matrix:

```bash
proj-2026-04-03-0318/bench_full_comparison.sh \
  --manifest "${NEMOTRON_FORWARD_MANIFEST}" \
  --bench-build-dir build \
  --test-build-dir build \
  --artifact-dir "${ARTIFACT_DIR}/full_comparison"
```

Focused prefix-cache TTFT sweep for multi-token BF16 work (`7g`) and cache work
(`9`):

```bash
NEMOTRON_FORWARD_MANIFEST="${NEMOTRON_FORWARD_MANIFEST}" \
./build/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 \
  --iterations 5 \
  --tail-token-count 32 \
  --moe-prefill-window-tokens 32 \
  | tee "${ARTIFACT_DIR}/prefix_ttft_tail32_window32.stdout.txt"

NEMOTRON_FORWARD_MANIFEST="${NEMOTRON_FORWARD_MANIFEST}" \
./build/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 \
  --iterations 5 \
  --tail-token-count 64 \
  --moe-prefill-window-tokens 64 \
  | tee "${ARTIFACT_DIR}/prefix_ttft_tail64_window64.stdout.txt"

NEMOTRON_FORWARD_MANIFEST="${NEMOTRON_FORWARD_MANIFEST}" \
./build/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 \
  --iterations 5 \
  --tail-token-count 128 \
  --moe-prefill-window-tokens 128 \
  | tee "${ARTIFACT_DIR}/prefix_ttft_tail128_window128.stdout.txt"
```

### Step-by-Step Verification Checklist

- Steps 1-3: run Tier 1 only unless the change touches runtime execution or test
  harness code that could affect correctness gates.
- Steps 4-6: run Tier 1 plus `verify_correctness.sh`, runtime oracle generation,
  exact-token vLLM parity, and `bench_decode_backends.sh`.
- Steps 7a-7e: run Tier 1 plus the targeted oracle/functional ctest regex above
  before the full suite. If any of these steps touch decode, attention, Mamba,
  or MoE execution, also run the direct decode benchmark sanity check.
- Step 7f: run the complete Tier 2 set.
- Step 7g: run the complete Tier 2 set. If the change touches multi-token
  prefill or restored-prefix execution, keep the `bench_full_comparison.sh`
  artifact as the phase record, not just the decode sanity check, and also run
  the focused prefix-cache TTFT sweep above for `32/64/128` tail/window pairs.
  Do not mark `7g` done on correctness alone; the artifact set must make any
  remaining row-replay path explicit.
- Step 8: run Tier 1, the targeted oracle/functional ctest regex, runtime oracle
  generation, and exact-token vLLM parity. Treat broad fixture regeneration as
  blocked on steps 7e and 7g closing the BF16 execution contract. If fixture or
  oracle semantics touch hot-path code, also run `verify_correctness.sh`.
- Step 9: run Tier 1 plus the targeted oracle/functional regex. If cache changes
  affect request execution, also run `verify_correctness.sh` and the direct
  decode benchmark sanity check.
- Step 10: run the complete Tier 2 set and keep the full artifact directory as
  the final verification record for the branch.

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Canonicalize verification gates and benchmark provenance | done | 6910b06 | |
| 2 | Freeze MoE routing contract in docs and tests | done | dbe8b98 | |
| 3 | Lock NVFP4 prepared-weight contract and golden tests | done | 0041132 | |
| 4 | Collapse MoE backend fragmentation around UnifiedFusedBackend | done | 06d2609 | FusedDecodeBackend removed (24x slower), DecodeCublasLt kept (~8% faster at decode) |
| 5 | Add explicit attention backend policy and cuDNN plan caching | done | 9776aa3 | cuDNN FE stub build; real plan-cache test needs FE-enabled rebuild |
| 6 | Make Mamba production path explicit and fence debug scaffolding | done | f824cbf | Also removed all Super-model oracle fixtures (53 tests, 100% pass) |
| 7a | Fused add+RMSNorm kernel (BF16 I/O, FP32 internal) | done | 51bd4b7 | |
| 7b+c | BF16 embedding lookup + BF16 dense GEMM path | done | a0639da | |
| 7d | BF16 hidden/residual buffers and layer interfaces | done | 3005496 | + GQA fix f66d740, fused decode gate cdd2c13, fastpath 525e2a8, MoE prefill fix 72f527c, decode scratch fix 3005496 |
| 7e | Residual-add pattern alignment and bootstrap cleanup | done | PENDING | FP32 compat overloads return delta; bootstrap split into validate/plan/assemble; manifest op_class roles replace name heuristics |
| 7f | BF16 pipeline verification and vLLM parity | done | — | 2026-04-04 rerun now passes `full_forward_manifest_smoke_test`, `nano_16_token_correctness_test`, the long-prompt `nano_save_prompt_oracle` path, and exact-token vLLM parity for `short_chat` (`runtime_generated_token_ids == vllm_generated_token_ids == [1784, 3330, 17000, 10693]`). |
| 7g | Resolve the multi-token BF16 execution contract | pending | — | Current attention / expert / Mamba BF16 multi-token requests recurse one token at a time; close or explicitly justify that divergence before broad oracle re-baselining or cache follow-on work. |
| 8 | Port oracle fixture generation to Nano (against BF16 pipeline) | in progress | — | Nano oracle generators now produce live fixtures for `full_model_single_token_short_chat_cuda_v3`, `prefix_prefill_short_chat_layer7_t4_oracle`, `expert_layer1_decode_block`, and `mamba_layer0_decode_block`; the registered oracle gates now pass under the branch’s intended BF16/NVFP4 functional envelopes, but the broader fixture-coverage expansion in the step text is still pending and should follow steps `7e` and `7g`. |
| 9 | Revisit cache allocator and page/snapshot ownership | pending | — | Snapshot must handle BF16 hidden states |
| 10 | Multi-turn prefix reuse regression and final verification sweep | pending | — | |

## Current Synthesis (2026-04-04 post-fix rerun)

The critical path has shifted again:

1. **7f is no longer the blocker.**
   The split-prefill / fused-decode handoff bug was fixed, the long-prompt `nano_save_prompt_oracle` path is healthy again, and the post-fix vLLM compare returned exact-token parity on `short_chat`.

2. **The smoke / inference / oracle surface is live again.**
   As of the latest rerun, `ctest --test-dir build --output-on-failure` is green (`57/57`). That includes `full_forward_manifest_smoke_test`, `nano_16_token_correctness_test`, `nano_save_prompt_oracle`, `prefill_prefix_oracle_test`, `single_token_decode_oracle_test`, `expert_layer_oracle_test`, and `mamba_layer_oracle_test`.

3. **Step 8 is no longer blocked on broken tests, but it is not fully complete.**
   The generator/test contract has been tightened enough for the current oracle gates to pass: the Python dumpers now model BF16 residual mutation and NVFP4 projection semantics closely enough for this branch, and the tests that still rely on approximate host-side modeling use explicit functional envelopes. The remaining step-8 work is coverage expansion: more representative Nano fixtures, especially the second expert/Mamba layers and the attention-side fixture called out in the step text.

4. **The last independent ctest blocker was a test-harness bug, not a runtime hot-path regression.**
   `expert_layer_fastpath_test` was flaky because its shared NVFP4 test descriptors carried stale `tensor_scale` pointers after move construction. Rebinding the owned descriptors stabilized the test, and the target now passes repeated stress runs.

5. **The decode performance sanity check remains healthy.**
   The fresh `nano_fused_decode_bench` rerun reported `hot_steady_state_mean_ms=15.597` and `steady_state_generated_tokens_per_second=64.116`, with `dense_reference_fallback=0`, `nvfp4_reference_fallback=0`, `host_routing_adapter_calls=0`, and `host_routing_tensor_copies=0`.

6. **The BF16 execution contract is still open for multi-token requests.**
   The latest fixes proved correctness, but they also made the current production shape more obvious: attention, expert, and Mamba currently satisfy `token_count > 1` by replaying the one-token path row by row. That may be acceptable as a temporary correctness bridge, but it is still an accidental divergence until we either replace it with true multi-token execution or explicitly keep and justify it with measurements.

## Immediate Execution Order

1. Resume **7e** and clean up the residual-add / bootstrap contract while the verification surface is green and the oracle set is still small enough to rebaseline cheaply.
2. Close **7g** immediately after `7e`: decide whether the current token-by-token BF16 multi-token bridge is temporary or intentional, and either remove it from the hot path or document and benchmark it as an explicit divergence.
3. Finish the broader fixture expansion in **8** only after **7e** and **7g** stabilize the BF16 execution contract.
4. Start **9** once step **8** coverage is genuinely complete or explicitly descoped; the next architectural work should be cache allocator and snapshot/page ownership, not more ad hoc oracle repair.
5. Reserve **10** for the final integrated sweep after steps **7e**, **7g**, **8**, and **9** are all genuinely closed.
