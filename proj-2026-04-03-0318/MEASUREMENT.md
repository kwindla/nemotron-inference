# Measurement Methodology

## Scope

This project compares our runtime against local vLLM for the exact external baseline named in the plan:

- model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
- vLLM checkout: `third_party/vllm`
- vLLM version: `v0.19.0`
- vLLM commit: `2a69949bdadf0e8942b7a1619b229cb475beef20`
- required MoE backend for this model on Blackwell: `flashinfer_cutlass`

`flashinfer_trtllm` is not the reference backend for Nano because the hidden size is `2688`, and `2688 % 512 != 0`.

## Automation

The step-7 harness lives in:

- `proj-2026-04-03-0318/bench_full_comparison.sh`
- `proj-2026-04-03-0318/verify_correctness.sh`

The benchmark harness writes one timestamped artifact directory containing:

- raw stdout/stderr captures
- JSON artifacts for each decode run
- parsed JSON for each `nano_prefix_cache_ttft_bench` run
- the raw vLLM JSON result
- fallback-disable verification output
- `summary.txt` and `summary.json`

The correctness harness writes:

- baseline and unified saved-oracle JSONs
- a direct oracle-to-oracle comparison summary
- the stdout log from `nano_16_token_correctness_test` in oracle mode

## What We Measure

Primary metric:

- TTFT, measured as wall-clock time from `llm.llm_engine.add_request(...)` to the first `RequestOutput` carrying a generated token

Secondary diagnostic metrics:

- best-effort MoE operator time from vLLM's torch-profiler summary, when the trace exposes MoE or FlashInfer rows
- routed-expert histograms for `32`, `64`, and `128` prompt-token prefill cases
- prefix-cache hit count for the decode case via `RequestOutput.num_cached_tokens`

Internal runtime diagnostics captured by the new harness:

- steady-state single-token decode step latency from `nano_fused_decode_bench`
- cached-tail `32`-token prefill summaries from `nano_prefix_cache_ttft_bench`
- explicit fallback-disable behavior via `nano_save_prompt_oracle`

## vLLM Configuration

The benchmark uses vLLM's offline `LLM` class, not the serving endpoint.

Forced backend configuration:

- `moe_backend="flashinfer_cutlass"` in the `LLM(...)` constructor
- `VLLM_USE_FLASHINFER_MOE_FP4=1`
- `VLLM_FLASHINFER_MOE_BACKEND=throughput`
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`

Compatibility export from the wrapper:

- `VLLM_MOE_PADDING=0`

Note:

`VLLM_MOE_PADDING` is exported defensively by the wrapper, but vLLM `v0.19.0` does not consume that variable directly in the CUDA path. The effective backend steering for this benchmark comes from `moe_backend`, `VLLM_USE_FLASHINFER_MOE_FP4`, `VLLM_FLASHINFER_MOE_BACKEND`, and `VLLM_NVFP4_GEMM_BACKEND`.

## Case Definitions

### Decode: token count `1`

The decode measurement is a cache-hit first-token benchmark, not a `1`-token uncached prompt.

Method:

1. Enable prefix caching in vLLM.
2. Build one fixed prompt of `decode_prefix_tokens` tokens.
3. Run it once to seed the prefix cache.
4. Re-issue the identical prompt and measure TTFT to the first generated token.
5. Record `num_cached_tokens` so the result can be checked for an actual cache hit.

This is the closest offline-`LLM` analogue to our runtime's cached decode path.

### Prefill: token counts `32`, `64`, `128`, `256`

These cases measure uncached prompt TTFT.

Method:

1. Keep prefix caching enabled globally so decode and prefill run on one engine.
2. Generate unique prompt token IDs per iteration, with a different first token each time.
3. Submit prompt length `N` with `max_tokens=1`.
4. Measure TTFT to the first generated token.

This keeps the prefill cases from benefiting from prior cache entries while preserving the same engine/backend configuration used for decode.

## Internal Runtime Configuration

The full step-7 harness uses two existing executables:

- `build/benchmarks/nano_fused_decode/nano_fused_decode_bench` for decode
- `build/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench` for cached-tail prefill

Decode backend configurations exercised by `bench_full_comparison.sh`:

| Label | Env | Expected backend behavior |
| --- | --- | --- |
| `default` | `NEMOTRON_FORWARD_UNIFIED_FUSED=0`, `NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0`, `NEMOTRON_FORWARD_MOE_CUBLASLT=0` | `FusedDecodeBackend` for decode, `BatchedCublasLtBackend` for multi-token prefill |
| `unified_fused` | `NEMOTRON_FORWARD_UNIFIED_FUSED=1`, `NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1` | `UnifiedFusedBackend` for supported decode and prefill traffic |
| `fallback_disable_override` | `NEMOTRON_FORWARD_UNIFIED_FUSED=0`, `NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1`, `NEMOTRON_FORWARD_MOE_CUBLASLT=0` | Explicit disable must override the legacy opt-in and fall back to the baseline path |

Prefill backend configurations exercised by `bench_full_comparison.sh`:

| Label | Env | Covered workloads |
| --- | --- | --- |
| `default` | `NEMOTRON_FORWARD_UNIFIED_FUSED=0`, `NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0` | `32`-token cached tail via committed-head and global-root prefix cases |
| `unified_fused` | `NEMOTRON_FORWARD_UNIFIED_FUSED=1`, `NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1` | same cached-tail cases on the unified path |

Important limitation:

The current `nano_prefix_cache_ttft_bench` executable hardcodes `32`-token tail cases and does not expose `SingleTokenForwardConfig::moe_prefill_window_tokens`. The harness therefore emits explicit `unavailable` JSON rows for the internal `64`- and `128`-token chunk-window matrix entries instead of pretending those numbers can be measured from the current binary.

## Routing Histograms

Routing histograms come from vLLM's routed-expert capture path:

- `enable_return_routed_experts=True`

The final `RequestOutput` contains per-token routed expert IDs for the prompt tokens. The benchmark converts that into:

- per-layer per-expert token counts
- active expert counts
- bucketed counts for `M=1`, `M=2`, `M=3`, and `M=4+`

These histograms are recorded for the `32`, `64`, and `128` prefill cases.

## MoE Timing

MoE timing is best-effort and diagnostic only.

Method:

1. Start vLLM's torch profiler through `LLM.start_profile(...)`.
2. Run one request for the target case.
3. Stop profiling and parse `profiler_out_*.txt`.
4. Extract rows whose operator names contain MoE / FlashInfer markers such as:
   `fused_moe`, `flashinfer`, `cutlass_fused_moe`, `moe_align_block_size`
5. Sum matched `self CUDA` times as the diagnostic MoE total for that case.

Interpretation:

- if a MoE profile value is present, it is useful for "where did the time go?"
- it is not the primary comparison metric
- if no MoE rows are exposed, the field should remain `null` / unavailable rather than guessed

## Comparison Matrix

`bench_full_comparison.sh` publishes the matrix below into `summary.txt` and `summary.json`.

| Family | Config | Workload | Metric | Status in current tree |
| --- | --- | --- | --- | --- |
| Runtime | `default` | decode `1` | steady-state single-token decode step latency | measured |
| Runtime | `unified_fused` | decode `1` | steady-state single-token decode step latency | measured |
| Runtime | `fallback_disable_override` | decode `1` | steady-state single-token decode step latency | measured |
| Runtime | `default` | cached committed-head `32`-token tail (`prefix256/1024/4096`) | tail prefill latency and hot-prefix TTFT | measured |
| Runtime | `unified_fused` | cached committed-head `32`-token tail (`prefix256/1024/4096`) | tail prefill latency and hot-prefix TTFT | measured |
| Runtime | `default` | cached global-root `32`-token tail (`prefix256/1024/4096`) | tail prefill latency and hot-prefix TTFT | measured |
| Runtime | `unified_fused` | cached global-root `32`-token tail (`prefix256/1024/4096`) | tail prefill latency and hot-prefix TTFT | measured |
| Runtime | `unified_fused_moe_window64` | internal `64`-token chunked prefill | hot-prefix TTFT | recorded as `unavailable` |
| Runtime | `unified_fused_moe_window128` | internal `128`-token chunked prefill | hot-prefix TTFT | recorded as `unavailable` |
| vLLM | `flashinfer_cutlass` | decode `1` | cache-hit TTFT | measured |
| vLLM | `flashinfer_cutlass` | prefill `32` | uncached TTFT | measured |
| vLLM | `flashinfer_cutlass` | prefill `64` | uncached TTFT | measured |
| vLLM | `flashinfer_cutlass` | prefill `128` | uncached TTFT | measured |

Why the internal/runtime rows differ from the vLLM rows:

- vLLM uses uncached prompt TTFT for `32/64/128`
- our current internal prefix-cache executable measures cached-tail replay plus first decode for the `32`-token cases it supports
- the harness keeps those shapes separate instead of silently merging unlike workloads

## How To Run The Complete Suite

Full comparison harness:

```bash
proj-2026-04-03-0318/bench_full_comparison.sh \
  --manifest /path/to/forward_runtime_manifest_nano_rtx5090_unverified.json
```

Correctness verification:

```bash
proj-2026-04-03-0318/verify_correctness.sh \
  --manifest /path/to/forward_runtime_manifest_nano_rtx5090_unverified.json
```

Both scripts auto-detect the common build directories. Override them when needed with:

- `--bench-build-dir`
- `--test-build-dir`
- `--build-dir`

## Correctness Verification

`verify_correctness.sh` uses the existing testing surface rather than new runtime code:

1. Run `nano_save_prompt_oracle` with unified fused disabled.
2. Run `nano_save_prompt_oracle` again with unified fused enabled.
3. Compare:
   - generated token sequence equality
   - prompt-boundary token equality
   - prompt-boundary top-5 logit indices
   - prompt-boundary top-5 logit values within the configured absolute tolerance
4. Run `nano_16_token_correctness_test` in oracle mode with the baseline oracle and the unified candidate enabled.

This is intentionally short and deterministic. It is a backend-equivalence guardrail, not a throughput benchmark.

## How To Interpret Results

Use TTFT as the decision metric.

- Lower decode TTFT means a better cached single-token path.
- Lower prefill TTFT means a better path for the exact measured workload shape.
- MoE profile time explains whether the gap is still dominated by expert execution or has moved elsewhere.
- Routing histograms explain whether a given token-count regime is pushing more experts into high-`M` buckets.

Do not compare unlike cases.

- Decode should be compared against decode.
- Cached-tail replay should be compared against cached-tail replay.
- Uncached vLLM prompt TTFT should be compared against the same vLLM prompt-token count.
- Internal MoE-only timings should not be substituted for external TTFT.

Interpret the `unavailable` internal `64`/`128` rows literally.

- They mean the runtime path exists but the current executable surface does not expose the required chunk-window control.
- They do not mean the runtime lacks chunking support.
- They should be treated as a benchmark-surface gap, not a performance result.

Interpret the fallback-disable check separately from latency.

- It verifies that `NEMOTRON_FORWARD_UNIFIED_FUSED=0` overrides the legacy unified opt-in and preserves the baseline output sequence.
- It is a correctness/configuration check, not a speed comparison.

## Existing Internal Baseline Artifact

The current provenance note for `proj-2026-04-03-0318/moe_profile_baseline.log` is in:

- `proj-2026-04-03-0318/BASELINE_PROVENANCE.md`

Short version:

- it is an internal MoE-layer microprofile from our runtime
- it does not preserve enough provenance to prove a `32`-token case
- its histogram lower bounds show it is not a literal `32`-token routed workload
- it is therefore not a direct apples-to-apples baseline against vLLM TTFT

## References

- `proj-2026-04-03-0318/bench_full_comparison.sh`
- `proj-2026-04-03-0318/verify_correctness.sh`
- `proj-2026-04-03-0318/bench_decode_backends.sh`
- `proj-2026-04-03-0318/bench_vllm_nano.py`
- `proj-2026-04-03-0318/run_bench_vllm.sh`
- `proj-2026-04-03-0318/BACKEND_SELECTION.md`
- `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`
