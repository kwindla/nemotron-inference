# Plan: Validate Unified Fused Backend on GPU

Project directory: `./proj-2026-04-03-1856`

## Context

The unified NVFP4 MoE alignment plan (proj-2026-04-03-0318) landed all infrastructure: UnifiedFusedBackend, GPU top-k routing, load-time weight preparation, MoE chunking, and benchmark harnesses. Nothing has run on GPU yet. The benchmark and verification scripts were written by Codex without hardware access, so they may contain bugs. This plan validates everything on hardware, starting from the simplest possible check and building toward the full comparison matrix.

## Steps

- [ ] **1. Bare-minimum inference sanity check**
  Run `nano_save_prompt_oracle` directly with `NEMOTRON_FORWARD_UNIFIED_FUSED=1` and inspect the output JSON by hand. The goal is to verify the unified fused backend loads, runs a 16-token prefill, generates 16 decode tokens, and does not crash or produce garbage (e.g., all-zero logits, repeated token IDs, NaN values). Compare the generated tokens visually against a default-backend run. No scripts — just two direct invocations and manual diff of the JSON output.

  Run commands:
  ```
  # Default backend
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 NEMOTRON_FORWARD_FUSED_MOE_DECODE=1 \
    build/testing/nano_save_prompt_oracle > /tmp/oracle_default.json

  # Unified fused backend
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
    build/testing/nano_save_prompt_oracle > /tmp/oracle_unified.json

  # Manual diff
  diff /tmp/oracle_default.json /tmp/oracle_unified.json
  ```

  If the unified backend crashes or produces obviously wrong output, stop and diagnose before proceeding. Save both JSON files as artifacts.
  Key files: `testing/api/nano_save_prompt_oracle.cpp`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/fused_moe_prefill.cu`

- [ ] **2. Audit and fix verify_correctness.sh**
  Read `proj-2026-04-03-0318/verify_correctness.sh` line by line and check:
  - Does it set the right env vars for default vs unified? (It should disable unified for the default run and enable it for the unified run.)
  - Does the Python comparison inline script actually compare the right fields?
  - Does it handle the manifest path correctly?
  - Does the `nano_16_token_correctness_test` invocation pass the right oracle path?
  - Does it capture stderr (where model loading and errors go)?

  Fix any bugs found. Then run it and capture the output. If it fails, diagnose whether the failure is in the script or in the unified backend.
  Key files: `proj-2026-04-03-0318/verify_correctness.sh`, `testing/api/nano_16_token_correctness_test.cpp`

- [ ] **3. Run internal performance benchmarks**
  First audit `proj-2026-04-03-0318/bench_decode_backends.sh` and `proj-2026-04-03-0318/bench_full_comparison.sh` for correctness — check they invoke the right binaries with the right env vars. Fix any issues.

  Then run the decode backend comparison:
  ```
  proj-2026-04-03-0318/bench_decode_backends.sh
  ```

  Then run the full comparison harness:
  ```
  proj-2026-04-03-0318/bench_full_comparison.sh
  ```

  Capture all artifacts. Record decode and prefill latency for: default backend, unified fused backend, and (if the scripts support it) with MoE window chunking.
  Key files: `proj-2026-04-03-0318/bench_decode_backends.sh`, `proj-2026-04-03-0318/bench_full_comparison.sh`, `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`, `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`

- [ ] **4. Run vLLM external baseline**
  Audit `proj-2026-04-03-0318/bench_vllm_nano.py` and `proj-2026-04-03-0318/run_bench_vllm.sh` for correctness. Check:
  - Does the Python script actually import from the right vLLM checkout?
  - Does `moe_backend="flashinfer_cutlass"` work as a constructor parameter?
  - Does `enable_return_routed_experts=True` work without error?
  - Does the profiler path work?

  Run the vLLM benchmark:
  ```
  proj-2026-04-03-0318/run_bench_vllm.sh
  ```

  If it fails, strip features (profiling, routing histograms) until the core TTFT measurement works. The minimum viable output is TTFT for decode and 32/64/128-token prefill.
  Key files: `proj-2026-04-03-0318/bench_vllm_nano.py`, `proj-2026-04-03-0318/run_bench_vllm.sh`, `third_party/vllm/`

- [ ] **5. Compile results and apply SM120 decision framework**
  Collect all benchmark results into a single comparison document at `proj-2026-04-03-1856/RESULTS.md`:
  - Internal decode latency: default vs unified fused (from step 3)
  - Internal prefill latency: default vs unified fused at 32/64/128 tokens (from step 3)
  - External comparison: our unified fused vs vLLM flashinfer_cutlass (from step 4)
  - Routing histograms if captured

  Apply the decision framework from `proj-2026-04-03-0318/SM120_BACKEND_DECISION.md`:
  - Is there a measurable MoE-localized gap vs vLLM?
  - Which M-buckets dominate?
  - GO or NO-GO for custom SM120 kernel work?

  Record the decision with rationale.
  Key files: `proj-2026-04-03-0318/SM120_BACKEND_DECISION.md`, `proj-2026-04-03-0318/MEASUREMENT.md`

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Bare-minimum inference sanity check | pending | — | |
| 2 | Audit and fix verify_correctness.sh | pending | — | |
| 3 | Run internal performance benchmarks | pending | — | |
| 4 | Run vLLM external baseline | pending | — | |
| 5 | Compile results and apply SM120 decision | pending | — | |
