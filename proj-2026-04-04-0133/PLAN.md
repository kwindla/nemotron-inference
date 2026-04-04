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

- [ ] **7e. Residual-add pattern alignment and bootstrap cleanup**
  Remove eager residual add from `attention_layer.cpp:1160`, `mamba_layer.cpp:672`, `expert_layer.cpp:1820`. Each layer returns its output in BF16; the fused add+RMSNorm at the start of the next layer handles add + norm. Split `RuntimeEnvironment::BuildFromManifestFile()` into validation, planning, and assembly phases. Replace heuristic layer-role inference in `ModelSchedule` with explicit manifest roles.
  Key files: `runtime/src/backend/attention_layer.cpp`, `runtime/src/backend/mamba_layer.cpp`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/api/runtime_environment.cpp`, `runtime/src/loader/model_schedule.cpp`

- [ ] **7f. BF16 pipeline verification and vLLM parity**
  Regenerate oracle and run vLLM parity — target 16/16 token match. Run full ctest (53/53 pass). Run decode benchmark — expect BF16 GEMMs to be faster than FP32 (less memory bandwidth). Run `verify_correctness.sh` — Route A must match Route C. Counters: `dense_reference_fallback=0`, `nvfp4_reference_fallback=0`.
  Key files: `proj-2026-04-03-0318/verify_correctness.sh`, `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py`
  Analysis: `proj-2026-04-04-0133/bf16_pipeline_analysis.md`

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

Run after steps 4, 8f, and 10 — or any time a step touches the hot execution path.

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

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Canonicalize verification gates and benchmark provenance | done | 6910b06 | |
| 2 | Freeze MoE routing contract in docs and tests | done | dbe8b98 | |
| 3 | Lock NVFP4 prepared-weight contract and golden tests | done | 0041132 | |
| 4 | Collapse MoE backend fragmentation around UnifiedFusedBackend | done | 06d2609 | FusedDecodeBackend removed (24x slower), DecodeCublasLt kept (~8% faster at decode) |
| 5 | Add explicit attention backend policy and cuDNN plan caching | done | 9776aa3 | cuDNN FE stub build; real plan-cache test needs FE-enabled rebuild |
| 6 | Make Mamba production path explicit and fence debug scaffolding | done | f824cbf | Also removed all Super-model oracle fixtures (53 tests, 100% pass) |
| 7a | Fused add+RMSNorm kernel (BF16 I/O, FP32 internal) | pending | — | Foundation for BF16 pipeline |
| 7b | BF16 embedding lookup | pending | — | |
| 7c | BF16 dense GEMM path | pending | — | |
| 7d | BF16 hidden/residual buffers and layer interfaces | pending | — | Largest change, depends on 7a-7c |
| 7e | Residual-add pattern alignment and bootstrap cleanup | pending | — | |
| 7f | BF16 pipeline verification and vLLM parity | pending | — | Target: 16/16 token match |
| 8 | Port oracle fixture generation to Nano (against BF16 pipeline) | pending | — | Must run after 7f |
| 9 | Revisit cache allocator and page/snapshot ownership | pending | — | Snapshot must handle BF16 hidden states |
| 10 | Multi-turn prefix reuse regression and final verification sweep | pending | — | |
