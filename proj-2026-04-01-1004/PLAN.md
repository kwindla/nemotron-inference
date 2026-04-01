# Plan: Fused Decode Preflight Checks & First Optimization Passes

Project directory: `./proj-2026-04-01-1004`

## Context

The Nemotron fused decode runtime (Mamba + MoE) is landed and passing smoke tests, but a 16-token correctness surface shows prompt-boundary divergence (first token mismatch, `max_abs_diff=5.53`), the linear device fastpath is disabled, and there is no operator-level measurement infrastructure. This plan implements the five preflight diagnostic checks from the design doc, then attacks the first three throughput checkpoints: prompt-parity fix, linear fastpath enablement, and device-side token selection. The full design doc is at `~/.claude/plans/warm-greeting-church.md`.

## Steps

- [x] **1. Route-separating prompt comparison matrix**
  Extend `testing/api/nano_16_token_correctness_test.cpp` to run three separate prefill configurations and compare pairwise instead of the current fused-vs-reference binary comparison:
  - **Route A**: reference kernels + legacy multi-token prefill (fused OFF, decode-consistent OFF)
  - **Route B**: reference kernels + decode-consistent prefill (fused OFF, decode-consistent ON)
  - **Route C**: fused kernels + decode-consistent prefill (fused ON, decode-consistent ON)
  Then compare A↔B (isolates whether decode-consistent prefill itself moves the boundary) and B↔C (isolates whether fused kernels move the boundary relative to decode-consistent reference). Print a summary table showing selected token IDs and max-abs-diff for each pair. Keep the existing lockstep decode comparison but drive it from Route A vs Route C.
  Also replace the current binary-search `MaybeTraceFirstDivergentPromptLayer()` with a linear sweep across a configurable set of probe layers (e.g. every 4th layer, then refine). The binary search assumes monotonic divergence which the plan doc says is wrong for residual networks. Add an env var `NEMOTRON_NANO_16_PROBE_STRIDE` to control stride (default 4). Report all divergent layers, not just the "first" one.
  Key files: `testing/api/nano_16_token_correctness_test.cpp`

- [x] **2. Linear operator execution counters**
  Add a `LinearOpCounters` struct (header: `runtime/include/nemotron/linear_op_counters.h`) with atomic counters for:
  - `dense_fastpath_plan_success` / `dense_fastpath_plan_fail`
  - `dense_fastpath_execute` / `dense_fastpath_execute_fail`
  - `dense_reference_fallback`
  - `nvfp4_fastpath_plan_success` / `nvfp4_fastpath_plan_fail`
  - `nvfp4_fastpath_execute` / `nvfp4_fastpath_execute_fail`
  - `nvfp4_reference_fallback`
  - `scaled_fp8_fastpath_execute` / `scaled_fp8_reference_fallback`
  Provide a thread-safe global accessor `GetLinearOpCounters()` and a `ResetLinearOpCounters()`. Instrument `UploadedLinearOp::Run()` in `linear_op.cpp` and `ScaledFp8LinearOp::Run()` in `scaled_fp8_linear.cu` to increment the appropriate counters on each code path. Add a `PrintLinearOpCounterSummary(std::ostream&)` helper. No behavior change to existing code—purely additive.
  Key files: `runtime/include/nemotron/linear_op_counters.h` (new), `runtime/src/backend/linear_op_counters.cpp` (new), `runtime/src/backend/linear_op.cpp`, `runtime/src/backend/scaled_fp8_linear.cu`, `runtime/CMakeLists.txt`

- [x] **3. Steady-state decode benchmark mode**
  Add a `--mode=steady-state` option to `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp` that:
  - Runs prompt prefill once (using the same 16-token prompt)
  - Then times N repeated `ContinueSingleToken(...)` calls individually (default N=16, configurable via `--decode-tokens=N`)
  - Reports per-step timings, mean, min, max, and tokens/sec for the decode-only phase
  - Does NOT include prefill cost in the steady-state metrics
  - Writes a JSON artifact with `"mode": "steady_state"` alongside the existing phased results
  Keep the existing default mode (`--mode=phased` or no flag) unchanged. Also wire `PrintLinearOpCounterSummary()` from step 2 into both benchmark modes so the JSON artifact includes actual linear backend usage counts.
  Key files: `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`

- [x] **4. cuDNN FE attention feasibility report**
  The existing `cudnn_handle_stub.cpp` and `cudnn_paged_attention_stub.cpp` always return `valid()=false` and `nullptr`. This means cuDNN FE paged attention is definitively not available in this build.
  Add a small standalone test `testing/api/cudnn_feasibility_test.cpp` that:
  - Creates a `CudnnHandle` and reports `valid()`, `version()`
  - Attempts `BuildCudnnPagedAttentionConfig(...)` with the Nano attention parameters and reports whether it returns a value
  - Prints a clear summary: "cuDNN FE paged attention: AVAILABLE" or "cuDNN FE paged attention: NOT AVAILABLE (stub build)"
  - Returns exit code 0 regardless (informational only)
  Add it to `testing/CMakeLists.txt`. This documents the current state and gives a concrete signal if cuDNN FE is ever linked for real.
  Key files: `testing/api/cudnn_feasibility_test.cpp` (new), `testing/CMakeLists.txt`

- [x] **5. Expert staging cost metrics**
  Add counters and logging for routed-expert upload/staging cost in the fused MoE decode path. In `runtime/src/backend/expert_layer.cpp`, around the section that uploads routed expert NVFP4 weights for the fused direct-MoE call (~lines 1019-1044):
  - Time each `DeviceNvfp4Weight::Create(...)` call for routed experts
  - Track total bytes uploaded per `Run()` call
  - Track number of experts staged
  Add an `ExpertStagingCounters` struct (same pattern as `LinearOpCounters`) in a new header `runtime/include/nemotron/expert_staging_counters.h` with:
  - `total_bytes_uploaded`, `total_experts_staged`, `total_staging_calls`
  - `staging_elapsed_us` (cumulative microseconds)
  Provide global accessor and reset. Add `PrintExpertStagingCounterSummary()`. Wire into the benchmark from step 3.
  Key files: `runtime/include/nemotron/expert_staging_counters.h` (new), `runtime/src/backend/expert_staging_counters.cpp` (new), `runtime/src/backend/expert_layer.cpp`, `runtime/CMakeLists.txt`

- [x] **6. Device-side argmax kernel for token selection**
  Add a device-side argmax kernel that eliminates the per-step host logits download for greedy decode. Create:
  - `runtime/include/nemotron/device_argmax.h` with `bool DeviceArgmax(const DeviceTensorFp32& logits_row, std::int32_t* device_token_id)` (operates on a 1-row logits tensor, writes a single int32 token ID to device memory)
  - `runtime/src/backend/device_argmax.cu` implementing a block-reduction argmax kernel (single block, threads cooperate to find max across vocab_size elements)
  Add to `runtime/CMakeLists.txt`. Add a unit test `testing/backend/device_argmax_test.cpp` that validates the kernel against known inputs (uniform, single-max, tie-breaking, large vocab).
  Key files: `runtime/include/nemotron/device_argmax.h` (new), `runtime/src/backend/device_argmax.cu` (new), `testing/backend/device_argmax_test.cpp` (new), `runtime/CMakeLists.txt`, `testing/CMakeLists.txt`

- [x] **7. Integrate device argmax into hot decode loop**
  Wire the device argmax from step 6 into `SingleTokenForwardModel`'s continuation path so the default greedy decode no longer copies the full logits row to host for token selection.
  In `runtime/src/api/single_token_forward_model.cpp`:
  - Add an env gate `NEMOTRON_FORWARD_DEVICE_TOKEN_SELECT` (default: enabled when not in debug/trace mode)
  - When device token select is active: after `ContinueSingleToken(...)` produces logits, call `DeviceArgmax()` and copy only the single `int32` token ID back to host (4 bytes instead of `vocab_size * 4` bytes)
  - When device token select is disabled (debug mode, trace capture, or explicit env override): keep the existing host-side logits copy and argmax path
  Update the benchmark and correctness test to respect this gate. The correctness test should keep host logits for comparison; the benchmark should use device argmax by default.
  Key files: `runtime/src/api/single_token_forward_model.cpp`, `runtime/include/nemotron/single_token_forward_model.h`, `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`

- [x] **8. Linear fastpath strict mode**
  Add a strict mode to the benchmark and correctness test surfaces that fails if the linear device fastpath is requested but any hot operator unexpectedly falls back to the reference kernel.
  In the benchmark (`nano_fused_decode_bench.cpp`):
  - Add `--strict-linear` flag
  - After the benchmark run completes, check `GetLinearOpCounters()`: if `*_reference_fallback > 0` when `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1`, report which operators fell back and how many times, then exit with failure
  In the correctness test (`nano_16_token_correctness_test.cpp`):
  - Add env gate `NEMOTRON_NANO_16_STRICT_LINEAR=1`
  - After the test completes, check counters and warn (but don't fail the test, since the primary gate is correctness)
  Include the counter summary in both JSON and stdout output.
  Key files: `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`, `testing/api/nano_16_token_correctness_test.cpp`

- [x] **9. Cached-head and profiler-ready benchmark modes**
  Add two more benchmark modes to `nano_fused_decode_bench.cpp`:
  - `--mode=cached-head`: Restore a cached prompt head (via prefix cache if available, or by running prefill and saving the state), then time continued decode from the cached state. This isolates decode cost when prompt is already cached.
  - `--mode=profile-ready`: Run exactly one prefill + one decode step with `cudaDeviceSynchronize()` barriers placed before and after each step. Print markers like `===PREFILL_START===` / `===PREFILL_END===` / `===DECODE_START===` / `===DECODE_END===` so Nsight Systems captures can be trimmed precisely.
  Add a script `benchmarks/nano_fused_decode/run_nsight_capture.sh` that invokes `nsys profile` on the benchmark with `--mode=profile-ready` and saves the output to `artifacts/profiles/`.
  Key files: `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`, `benchmarks/nano_fused_decode/run_nsight_capture.sh` (new)

- [ ] **10. Wire all counters into benchmark JSON artifact**
  Final integration step: ensure the benchmark JSON artifact (for all modes) includes:
  - `linear_op_counters`: all fields from `LinearOpCounters`
  - `expert_staging_counters`: all fields from `ExpertStagingCounters`
  - `device_token_select_enabled`: boolean
  - `cudnn_fe_available`: boolean (from `CudnnHandle::Create()->valid()`)
  - `mode`: which benchmark mode was run
  Update the JSON serialization in the benchmark to include these fields. Update `run_default_bench.sh` to pass through mode flags. Verify the JSON output is valid by adding a simple parse check in the script.
  Key files: `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`, `benchmarks/nano_fused_decode/run_default_bench.sh`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Route-separating prompt comparison matrix | done | 34eefe4 | preflight-prompt-matrix |
| 2 | Linear operator execution counters | done | 9db3589 | preflight-linear-counters |
| 3 | Steady-state decode benchmark mode | done | 80256a1 | preflight-steady-bench |
| 4 | cuDNN FE attention feasibility report | done | b13f0e9 | preflight-attention-feasibility; confirmed NOT AVAILABLE (stub build) |
| 5 | Expert staging cost metrics | done | f76eaeb | preflight-expert-staging-metrics |
| 6 | Device-side argmax kernel | done | 972bdfb | nano-device-token-select (kernel) |
| 7 | Integrate device argmax into decode loop | done | f4c4996 | nano-device-token-select (integration) |
| 8 | Linear fastpath strict mode | done | 447882f | nano-linear-fastpath (strict gate) |
| 9 | Cached-head and profiler-ready benchmark modes | done | — | nano-hotpath-measure |
| 10 | Wire all counters into benchmark JSON artifact | pending | — | nano-hotpath-measure (integration) |
