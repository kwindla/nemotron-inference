# Measurement Methodology

## Scope

This project compares our runtime against local vLLM for the exact external baseline named in the plan:

- model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
- vLLM checkout: `third_party/vllm`
- vLLM version: `v0.19.0`
- vLLM commit: `2a69949bdadf0e8942b7a1619b229cb475beef20`
- required MoE backend for this model on Blackwell: `flashinfer_cutlass`

`flashinfer_trtllm` is not the reference backend for Nano because the hidden size is `2688`, and `2688 % 512 != 0`.

## What We Measure

Primary metric:

- TTFT, measured as wall-clock time from `llm.llm_engine.add_request(...)` to the first `RequestOutput` carrying a generated token

Secondary diagnostic metrics:

- best-effort MoE operator time from vLLM's torch-profiler summary, when the trace exposes MoE or FlashInfer rows
- routed-expert histograms for `32`, `64`, and `128` prompt-token prefill cases
- prefix-cache hit count for the decode case via `RequestOutput.num_cached_tokens`

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

| Workload | Our runtime target metric | vLLM metric from this benchmark | Notes |
| --- | --- | --- | --- |
| Decode (`1`) | cached first-token decode latency | cache-hit TTFT with `num_cached_tokens` recorded | Compare decode separately from prefill |
| Prefill (`32`) | `32`-token tail prefill + first decode, or full cold TTFT at `32` tokens | uncached `32`-token TTFT | Use routing histograms to explain expert density |
| Prefill (`64`) | `64`-token tail prefill + first decode, or full cold TTFT at `64` tokens | uncached `64`-token TTFT | Important early chunk-size tuning point |
| Prefill (`128`) | `128`-token chunk + first decode | uncached `128`-token TTFT | Also collect routing histograms |
| Prefill (`256`) | `256`-token chunk + first decode | uncached `256`-token TTFT | Treat as a later streamed regime |

## How To Interpret Results

Use TTFT as the decision metric.

- Lower decode TTFT means a better cached single-token path.
- Lower prefill TTFT means a better uncached prompt path for that token count.
- MoE profile time explains whether the gap is still dominated by expert execution or has moved elsewhere.
- Routing histograms explain whether a given token-count regime is pushing more experts into high-`M` buckets.

Do not compare unlike cases.

- Decode should be compared against decode.
- Prefill should be compared against the same prompt-token count.
- Internal MoE-only timings should not be substituted for external TTFT.

## Existing Internal Baseline Artifact

The current provenance note for `proj-2026-04-03-0318/moe_profile_baseline.log` is in:

- `proj-2026-04-03-0318/BASELINE_PROVENANCE.md`

Short version:

- it is an internal MoE-layer microprofile from our runtime
- it does not preserve enough provenance to prove a `32`-token case
- its histogram lower bounds show it is not a literal `32`-token routed workload
- it is therefore not a direct apples-to-apples baseline against vLLM TTFT
