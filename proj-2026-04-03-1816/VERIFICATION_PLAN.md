# Plan: Verification, Parity, and Benchmark Discipline

Project directory: `./proj-2026-04-03-1816`

**Superseded status (2026-04-04):** Background analysis only.
Follow `proj-2026-04-04-0133/PLAN.md` for the current verification runbook,
phase boundaries, and BF16 multi-token acceptance criteria.

## Goal

Close accidental verification gaps in the Nano-on-RTX-5090 branch so the runtime stays honest against both its own oracles and the local vLLM baseline.

The critical priority is now **not** more ad hoc benchmark scripting. The critical priority is:

1. prove correctness at the smallest useful boundaries
2. prove parity against vLLM on the same prompt and model inputs
3. prove performance with stable, attributable measurements
4. keep the three signals separate so a pass in one area does not hide regressions in another

## Scope

This plan covers:

- oracle generation and replay
- local parity comparisons against vLLM
- decode / TTFT benchmark discipline
- regression guardrails and measurement provenance

It does not cover kernel design decisions except where those decisions change what must be verified.

## Current State

The verification surface is already real, but it is fragmented:

- `testing/api/nano_save_prompt_oracle.cpp` writes a compact oracle with prompt tokens, prompt-boundary top-5 logits, boundary token id, generated token ids, route metadata, and timestamp.
- `testing/api/nano_16_token_correctness_test.cpp` is the main correctness gate for the current Nano path and checks oracle alignment plus continuation behavior.
- `testing/api/state_snapshot_test.cpp` verifies round-trip capture and restore of attention KV plus Mamba state.
- `testing/cache/prefix_cache_test.cpp` covers committed heads, prompt heads, global roots, and serializer / prompt-revision matching.
- `testing/api/prefill_prefix_oracle_test.cpp`, `testing/api/expert_layer3_prefix_replay_test.cpp`, `testing/backend/*_oracle_test.cpp`, and `testing/backend/*_compare_test.cpp` provide strong slice-level coverage for attention, Mamba, MoE, routing, NVFP4, and scaled-FP8 paths.
- `proj-2026-04-03-0318/verify_correctness.sh` compares default vs unified-fused oracle output and then runs `nano_16_token_correctness_test` against the baseline oracle.
- `proj-2026-04-03-0318/bench_decode_backends.sh`, `proj-2026-04-03-0318/bench_full_comparison.sh`, and `proj-2026-04-03-0318/run_bench_vllm.sh` provide the main performance comparison path.
- `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py` and `proj-2026-04-03-2113/save_runtime_oracle.py` provide the exact-token parity path between the runtime oracle and local vLLM.

This is a good start, but it is not yet a polished verification system. The main weaknesses are coverage shape, artifact discipline, and benchmark completeness.

## vLLM Reference Shape

The relevant vLLM comparison surface is documented in [docs/vllm_rtx5090.md](/home/khkramer/src/nemotron-inference/docs/vllm_rtx5090.md). The important lessons are:

- vLLM is the external baseline for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` on RTX 5090.
- The correct local setup is `vllm-env-cu128`, `PYTHONPATH=third_party/vllm`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `flashinfer_cutlass`.
- Exact token agreement is the primary parity signal in the oracle compare flow.
- TTFT is the primary external benchmark metric for this branch.

From the current vLLM code and docs, the reference behavior to match is:

- prompt-token-id driven comparisons, not text-only smoke checks
- explicit backend selection and recorded backend identity
- separate correctness, parity, and performance measurement paths
- stable artifact output with machine-readable summaries

## Verification Alignment

### Aligned

- The branch already has real oracle fixtures instead of handwritten expected strings.
- The main oracle path is token-id based, which avoids tokenizer ambiguity.
- Slice-level tests already exist for the important subsystems rather than only for the top-level model.
- The vLLM comparison path already uses the same prompt-token IDs when it is run correctly.
- The benchmark scripts already write timestamped artifacts and record environment metadata.

### Intentional Divergences

- `nano_save_prompt_oracle` stores a compact boundary summary instead of the full dense logits tensor. That is a deliberate tradeoff, but it means the correctness story must be explicit about what is and is not covered.
- `verify_correctness.sh` keeps the baseline-vs-candidate comparison local to the runtime rather than invoking vLLM. That is acceptable as long as the parity path remains a separate gate.
- The benchmark harness is intentionally split between internal runtime measurements and external vLLM measurements. That split is useful if the artifact boundaries stay crisp.

### Accidental Divergences To Close

- The verification matrix is spread across too many one-off entry points. A polished system should have a small number of canonical gates, not a growing list of scripts with overlapping behavior.
- `nano_save_prompt_oracle` only compares prompt-boundary top-5 logits. That can miss drift that does not disturb the top-5 set.
- `verify_correctness.sh` verifies default vs unified-fused output, but it does not yet serve as a broader regression matrix for route selection, prefix reuse, or benchmark provenance.
- `bench_full_comparison.sh` currently emits structured `unavailable` rows for internal `64`/`128` MoE-window rows because `nano_prefix_cache_ttft_bench` hardcodes 32-token tail cases and does not expose `moe_prefill_window_tokens`.
- The vLLM routed-expert capture path is known to be unstable on this setup, so routed-expert profiling is not yet a trustworthy regression signal.
- The runtime-vs-vLLM comparison is currently token-level only. That is the right primary comparison, but it should be paired with enough metadata to explain divergences when they occur.
- Build provenance is easy to blur because scripts auto-detect build directories. That is convenient, but it is risky for repeatability.

## Adversarial Review

### 1. Oracle coverage

Current state:

- prompt token ids, boundary top-5 logits, boundary token id, generated token ids, route metadata
- broad slice oracles for attention, Mamba, MoE, routing, and quantized kernels
- state round-trip coverage for snapshot / restore

Review:

- Coverage is broad enough to catch many regressions, but it is uneven.
- The compact oracle format is a good default, but it is not enough by itself to localize deeper drift.
- The current tests are strong on single-slice correctness and weaker on full conversation behavior under cache reuse.

What to keep:

- compact token-id oracle format
- exact state round-trip tests
- slice oracles for the key hot paths

What to close:

- add at least one stronger full-path parity artifact per major request shape
- make the oracle format explicit about its limits
- keep cached-prefix and continuation cases as first-class test inputs, not special cases

### 2. Parity testing

Current state:

- `compare_vllm_runtime_oracle.py` compares runtime-generated token ids against local vLLM on the same prompt token ids
- `save_runtime_oracle.py` can create runtime oracles from token ids, named prompts, or raw text
- `docs/vllm_rtx5090.md` documents the working local vLLM setup and the exact compare command

Review:

- This is the right parity shape.
- The weak point is not the compare logic itself; it is the surrounding run discipline.
- Parity should be recorded as exact token agreement plus enough context to explain any mismatch.

What to keep:

- exact generated token match as the primary parity criterion
- support for named prompts, raw text, and explicit token-id inputs
- local vLLM baseline pinned to the working RTX 5090 setup

What to close:

- preserve the model / backend / env metadata with every parity artifact
- make the parity gate independent from performance gates
- add a routine path for replaying the same compare against known regression fixtures

### 3. Benchmark discipline

Current state:

- internal decode and prefix-cache benchmarks exist
- external vLLM TTFT sweep exists
- benchmark outputs are saved as JSON and text artifacts
- the current vLLM baseline is recorded in `proj-2026-04-03-0318/vllm_baseline_results.json`

Review:

- The benchmark structure is good enough to compare backends, but not yet polished enough to serve as the only source of truth.
- Internal measurements and external measurements must remain separate because they answer different questions.
- The current internal prefix benchmark is not yet expressive enough to measure the chunk-window space the plans care about.

What to keep:

- timestamped artifact directories
- environment capture
- internal-vs-external benchmark separation
- TTFT as the primary external metric

What to close:

- expose the internal MoE window size in the benchmark executable or wrapper
- stop fabricating coverage gaps as “unavailable” once the runtime can actually measure them
- ensure every benchmark result records the exact binary, build dir, manifest, model, and env vars used
- make the decode benchmark and prefix benchmark report a common machine-readable summary schema

### 4. vLLM comparison harnesses

Current state:

- `run_bench_vllm.sh` and `bench_vllm_nano.py` use the working local vLLM stack
- routed-expert capture is disabled by default because the current vLLM setup can crash on larger prefill cases
- the compare path is already token-exact rather than text-only

Review:

- The local vLLM path is now good enough to be a real baseline, not a placeholder.
- The routed-expert capture bug means the comparison harness still lacks one useful diagnostic mode.
- The benchmark wrapper is valuable only if it stays reproducible across shells and build trees.

What to keep:

- `flashinfer_cutlass` as the reference MoE backend
- exact token-based compare flow
- `gpu_memory_utilization`, `trust_remote_code`, and runtime env settings in the wrapper defaults

What to close:

- retain a disabled-by-default routed-expert capture mode, but do not let it block the core TTFT sweep
- record the vLLM version / source revision in every artifact
- separate smoke-test success from benchmark success
- make sure the benchmark wrapper refuses to silently fall back to the wrong Python or source tree

### 5. Residual blind spots

The current verification surface still misses:

- a full dense-logit oracle for prompt-boundary comparison
- a strong multi-turn prefix-cache reuse regression that exercises conversation heads and global roots in one flow
- an internal benchmark that can sweep the MoE execution window rather than hardcoding 32-token tail cases
- a trustworthy routed-expert profile from vLLM on this exact setup
- a build-provenance standard that prevents accidental cross-build contamination in results

## Concrete Implementation Tasks

1. Split the verification story into canonical gates. Use one gate for local correctness / oracle replay, one for vLLM parity, one for internal performance, and one for external TTFT.

2. Strengthen the oracle payload without making it too heavy. Keep the compact boundary summary, add an optional deeper payload for regression runs that need more localization, and make the record explicit about prompt, model, manifest, backend flags, and decode length.

3. Add a real multi-turn prefix-reuse regression. Cover committed heads, optional prompt-head behavior if it remains, global root fallback, and append-only continuation, then prove that restore plus continuation still matches the expected token stream.

4. Make benchmark provenance non-optional. Record build dir, manifest, binary path, model id, and all env vars that affect backend selection, and do not let auto-detection hide the provenance in the artifact.

5. Expose the MoE execution window in the internal benchmark path. Measure `32`, `64`, and `128` as real runtime cases rather than placeholders, and align the measured rows with the backend-selection plan.

6. Keep the vLLM compare path exact-token focused, but enrich the artifact. Write out mismatch index, prompt source, prompt token ids, and backend selection metadata, and preserve the exact same prompt token ids for runtime and vLLM.

7. Separate correctness failures from performance regressions. A slower pass should still be a pass, and a faster incorrect path should still fail.

## Missing Evidence

- No single regression matrix yet covers default backend, unified fused, fallback disable, and vLLM parity in one repeatable flow.
- No dense-logit compare artifact yet tells us whether a token-exact pass is hiding numerical drift.
- No internal benchmark yet measures the real MoE-window sweep the architecture plans care about.
- No stable routed-expert profile yet exists for the current local vLLM setup on larger prefill cases.
- No build-provenance policy yet guarantees that benchmark artifacts are always tied to the same binary tree.
- No end-to-end multi-turn prefix-reuse regression yet proves the cache-control-plane path under repeated conversation continuation.

## Sequencing Advice

1. Tighten correctness first. Keep `verify_correctness.sh` as the primary local gate, but make it more explicit about what it covers and what it does not.

2. Then tighten parity. Use `save_runtime_oracle.py` and `compare_vllm_runtime_oracle.py` as the canonical exact-token path, and keep the vLLM baseline pinned to the working RTX 5090 stack.

3. Then tighten benchmark provenance. Every performance run should be attributable to one build, one manifest, one model, and one backend configuration.

4. Then expand internal coverage. Unlock the real MoE window sweep so internal benchmarking stops depending on hardcoded 32-token tails.

5. Finally, add residual diagnostics. Routed-expert profiling, denser oracle payloads, and multi-turn reuse regressions should deepen the system after the core gates are stable.

## Success Criteria

- correctness failures are localized by oracle, not guessed from benchmark noise
- parity failures are exact-token mismatches with reproducible prompt provenance
- benchmark results can be traced back to the exact binary, env, manifest, and backend selection
- the internal benchmark matrix covers the same token regimes the architecture cares about
- the verification system is small enough to trust and broad enough to catch real regressions
