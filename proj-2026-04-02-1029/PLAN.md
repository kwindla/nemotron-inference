# Plan: Nano Prefix Caching for Multi-Turn Conversations

Project directory: `./proj-2026-04-02-1029`

## Context
Minimize time-to-first-token (TTFT) for multi-turn conversations by caching attention KV state and Mamba recurrent state at conversation boundaries. The cache infrastructure (PrefixCache, ReusableStateArena, snapshot/restore, RunGreedyConversationTurn) already exists. This milestone scopes to two node types -- global shared roots and conversation committed heads -- and needs cleanup of out-of-scope prompt-head code, manifest-driven budget validation, integration test coverage for global roots, and a TTFT benchmark.

## Steps

- [x] **1. Remove prompt heads from the multi-turn runtime path**
  Delete the prompt-head publish block from `RunGreedyConversationTurn()` (the function currently publishes committed heads at lines ~1093-1106 in `single_token_forward_model.cpp`; prompt-head code, if any, should be removed from this path). Update the existing manifest-backed smoke test to assert only committed-head behavior. Update `docs/v1_cache_architecture.md` to mark prompt heads as explicitly out of scope for this milestone. Do NOT delete prompt-head support from the cache API itself -- that is a separate follow-up.
  Key files: `runtime/src/api/single_token_forward_model.cpp`, `testing/api/full_forward_manifest_smoke_test.cpp`, `docs/v1_cache_architecture.md`

- [ ] **2. Fix Nano manifest runtime_profile generation and validate the manifest-backed loader budget path**
  The current `generate_forward_manifest.py` has a hardcoded `RUNTIME_PROFILE` (lines 17-21) with values that don't match the Nano-30B-A3B config (kv_bytes_per_token=4096 instead of 6144, wrong mamba state sizes). Fix the generator to derive the correct runtime profile from the Nano config: kv_bytes_per_token=6144, mamba_state_bytes_fp16=25,247,744, mamba_state_bytes_fp32=50,495,488. Update the shell script as needed. Then write `testing/api/nano_loader_budget_test.cpp` that loads the generated Nano manifest, calls `BuildLoaderPlan()` with the RTX 5090 `ServiceMemoryTarget`, and asserts: (a) loader_plan fits, (b) bytes_per_full_context_node matches `BytesForExactPrefixNode(profile, 4096)`, (c) `RuntimeEnvironment::BuildFromManifestFile()` propagates the same shared-cache budget into PrefixCache/ReusableStateArena. Print the final loader-plan summary for human review.
  Key files: `tools/manifest/generate_forward_manifest.py`, `tools/manifest/generate_forward_manifest_rtx5090_nano.sh`, `testing/api/nano_loader_budget_test.cpp` (new), `testing/CMakeLists.txt`

- [ ] **3. Add global-root integration test**
  Keep the existing `full_forward_manifest_smoke_test.cpp` as the committed-head integration test (removing only prompt-head assertions per step 1). Create a new `testing/api/nano_global_root_prefix_cache_test.cpp` that validates global-root reuse across conversations: prefill system tokens, publish a global root, run two different conversations (A and B) that share the system prefix, verify matched_token_count equals system token count for both, verify generated tokens match an uncached reference run. Use argmax equality with max_abs_diff <= 1e-3f for logit comparisons (matching existing smoke test standard). Register in CMakeLists.txt using the `nemotron_add_test()` pattern.
  Key files: `testing/api/nano_global_root_prefix_cache_test.cpp` (new), `testing/api/full_forward_manifest_smoke_test.cpp`, `testing/CMakeLists.txt`

- [ ] **4. TTFT benchmark**
  Create `benchmarks/nano_prefix_cache_ttft/` following the `benchmarks/nano_fused_decode/` pattern (CMakeLists.txt, .cpp, optional .sh). Benchmark cases: (a) cold prefill + first token decode at 256, 1K, 4K tokens, (b) cached committed-head resume with 32-token user tail, (c) cached global-root resume across conversations. Run 5+ iterations per case, report median and p95. Metrics: cold TTFT, restore latency, tail prefill latency, first-token decode latency, commit/snapshot latency, hot-prefix TTFT, speedup factor, snapshot size bytes, matched prefix token count. Register in `benchmarks/CMakeLists.txt`.
  Key files: `benchmarks/nano_prefix_cache_ttft/CMakeLists.txt` (new), `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp` (new), `benchmarks/CMakeLists.txt`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Remove prompt heads from multi-turn runtime path | done | — | |
| 2 | Fix Nano manifest runtime_profile and validate budget | pending | — | |
| 3 | Add global-root integration test | pending | — | |
| 4 | TTFT benchmark | pending | — | |
