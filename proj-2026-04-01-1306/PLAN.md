# Plan: Prompt-Parity Diagnosis & Linear Fastpath Enablement

Project directory: `./proj-2026-04-01-1306`

## Context

The preflight infrastructure (route-separating test, linear/expert counters, benchmark modes, device argmax) is landed. The next priority is using that infrastructure to fix the two primary blockers: prompt-boundary correctness divergence and the disabled linear device fastpath. The design doc at `~/.claude/plans/warm-greeting-church.md` defines checkpoints `nano-prompt-parity` and `nano-linear-fastpath` as the first two wins after preflight.

## Steps

- [x] **1. Add embedding and final-norm trace comparison to the correctness test**
  Extend `testing/api/nano_16_token_correctness_test.cpp` to capture and compare `SingleTokenForwardTrace` outputs at the embedding level and final-norm/logits level for each route pair (A↔B, B↔C). Currently the test only compares the final argmax token and logits row. Add:
  - Run each route's prefill with a trace (pass `capture_layer_indices={}`, `trace=&trace_obj` to `RunPrefill`)
  - Compare `trace.embedding_output` between routes A and B, then B and C — report max-abs-diff
  - Compare `trace.final_hidden_normed` between routes — report max-abs-diff
  - Print a summary like: `embedding A<->B max_abs_diff=X.XX  final_norm A<->B max_abs_diff=X.XX`
  - This answers whether divergence enters at embedding (impossible if same tokens), at some layer, or at the final projection
  - Gate behind `NEMOTRON_NANO_16_TRACE_EMBEDDING=1` env var to avoid the cost on every run
  Key files: `testing/api/nano_16_token_correctness_test.cpp`

- [x] **2. Add GEMM plan-build diagnostic logging to linear ops**
  When `NEMOTRON_FORWARD_DEBUG=1`, add diagnostic output to `linear_op.cpp` showing why GEMM plan builds fail. In `BuildRuntimeGemmPlan()` and `BuildDescriptorGemmPlan()`, when a plan returns `std::nullopt`, log:
  - The tensor name from the descriptor
  - M, N, K dimensions attempted
  - Which sub-step failed (BuildRuntimeLaunchPlan, PrepareGemmExecution, or BuildCublasLtGemmPlan)
  - For NVFP4, whether block_scales_bytes or tensor_scale_bytes were invalid
  Also add the same diagnostic to `ScaledFp8LinearOp::Run()` when its plan build fails. This makes the `--strict-linear` output actionable — instead of just "N fallbacks", we'll know exactly which shapes/tensors failed and why.
  Key files: `runtime/src/backend/linear_op.cpp`, `runtime/src/backend/scaled_fp8_linear.cu`

- [x] **3. Add linear fastpath counter breakdown to correctness test layer probes**
  Extend the layer-probe divergence sweep in `nano_16_token_correctness_test.cpp` to reset and report linear counters around each layer probe. This shows which specific layers trigger reference fallbacks when the fastpath is enabled. After each `RunLayerObservation()` call, if `NEMOTRON_NANO_16_STRICT_LINEAR=1`:
  - Print the linear counter delta for that specific layer probe (which counters incremented)
  - This localizes fastpath failures to specific layers/operators rather than just a global count
  Key files: `testing/api/nano_16_token_correctness_test.cpp`

- [x] **4. Add per-operator linear counter tracking to the benchmark**
  The current `LinearOpCounters` are global — they don't tell you which tensor/operator fell back. Add a lightweight per-operator tracking mode:
  - Create `runtime/include/nemotron/linear_op_trace.h` with a `LinearOpTrace` struct containing a `std::vector<LinearOpTraceEntry>` where each entry has: `tensor_name`, `kernel_family`, `path_taken` (fastpath/reference), `plan_build_ok`, `execute_ok`
  - Add `GetLinearOpTrace()` / `ResetLinearOpTrace()` / `IsLinearOpTraceEnabled()` gated by `NEMOTRON_FORWARD_LINEAR_TRACE=1`
  - Instrument `UploadedLinearOp::Run()` and `ScaledFp8LinearOp::Run()` to append entries when trace is enabled
  - Add `PrintLinearOpTraceSummary(std::ostream&)` that groups by tensor name and shows fastpath vs fallback counts
  - Wire into the benchmark's `--strict-linear` output so it reports exactly which tensors fell back
  Key files: `runtime/include/nemotron/linear_op_trace.h` (new), `runtime/src/backend/linear_op_trace.cpp` (new), `runtime/src/backend/linear_op.cpp`, `runtime/src/backend/scaled_fp8_linear.cu`, `runtime/CMakeLists.txt`

- [ ] **5. Add saved-token oracle for the fixed 16-token prompt**
  Create a small utility that runs the reference path (Route A) on the fixed 16-token prompt and saves the per-token argmax sequence plus the prompt-boundary logits row as a JSON file under `artifacts/oracles/`. This gives a stable reference that doesn't require re-running the expensive reference path every time.
  - Create `testing/api/nano_save_prompt_oracle.cpp` as a standalone binary
  - Run Route A prefill, save: prompt tokens, boundary token ID, top-5 logits with indices, max logit value
  - Also run 16 continuation steps and save each token ID
  - Write JSON to `artifacts/oracles/nano_16_token_oracle_<timestamp>.json`
  - Add to `testing/CMakeLists.txt`
  Key files: `testing/api/nano_save_prompt_oracle.cpp` (new), `testing/CMakeLists.txt`

- [ ] **6. Add a load-and-compare mode to the correctness test**
  Add `NEMOTRON_NANO_16_ORACLE_PATH=/path/to/oracle.json` support to `nano_16_token_correctness_test.cpp`. When set:
  - Load the saved oracle (token sequence + boundary logits)
  - Compare each route's output against the oracle instead of running a fresh Route A
  - This makes correctness checks much cheaper (one route instead of two/three)
  - Still run the full route matrix if the env var is not set (backward compatible)
  Key files: `testing/api/nano_16_token_correctness_test.cpp`

- [ ] **7. Add expert staging counter integration to correctness test**
  Mirror what was done for linear counters: add `#include "nemotron/expert_staging_counters.h"` to the correctness test, reset/print expert staging counters around the test, and add a `NEMOTRON_NANO_16_STRICT_EXPERT_STAGING` warning mode that flags if expert staging cost is unexpectedly high (e.g., bytes_uploaded > threshold per token). This completes the counter coverage across both test surfaces.
  Key files: `testing/api/nano_16_token_correctness_test.cpp`

- [ ] **8. Add benchmark regression comparison script**
  Create `benchmarks/nano_fused_decode/compare_artifacts.py` that:
  - Takes two JSON benchmark artifact paths as arguments
  - Compares key metrics: steady_state_mean_ms, prefill_ms, tokens/sec
  - Compares counter values: linear fastpath vs fallback counts, expert staging bytes
  - Prints a diff table showing metric, baseline, current, delta, percent change
  - Exits with failure if any regression exceeds a configurable threshold (default 10%)
  - This is the foundation for the `nano-roofline-loop` checkpoint
  Key files: `benchmarks/nano_fused_decode/compare_artifacts.py` (new)

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Embedding and final-norm trace comparison | done | 534ba08 | nano-prompt-parity (diagnosis) |
| 2 | GEMM plan-build diagnostic logging | done | 5edd6f9 | nano-linear-fastpath (diagnostics) |
| 3 | Linear counter breakdown in layer probes | done | ac16b68 | nano-linear-fastpath (per-layer) |
| 4 | Per-operator linear counter tracking | done | — | nano-linear-fastpath (per-tensor) |
| 5 | Saved-token oracle for fixed prompt | pending | — | nano-prompt-parity (oracle) |
| 6 | Load-and-compare oracle mode | pending | — | nano-prompt-parity (fast compare) |
| 7 | Expert staging counters in correctness test | pending | — | counter coverage |
| 8 | Benchmark regression comparison script | pending | — | nano-roofline-loop (foundation) |
