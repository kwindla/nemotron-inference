# External Baselines Notes

Project directory: `./proj-2026-04-05-1704`

## Scope

This note tracks the external comparison work against:

- the native from-scratch runtime in this repo
- local vLLM on RTX 5090
- local TRT-LLM PyTorch backend serve on RTX 5090

It records:

- exactly how each benchmark was run
- artifact paths for each result
- current caveats
- the active plan for the next comparison steps

## Active Plan

### 1. Document current one-token TTFT baselines

Status: `done`

Required outputs:

- native runtime TTFT artifacts and summary
- vLLM TTFT artifacts and summary
- TRT-LLM serve TTFT artifacts and summary

### 2. Add an apples-to-apples long-response streaming benchmark

Status: `done`

Target shape:

- one request at a time
- same prompt intent across all three runtimes
- high `max_tokens`
- streaming enabled for the server-backed paths
- multiple runs per runtime

Prompt:

`Tell me a complicated and hilarious joke in the form of a story.`

Planned outputs:

- TTFT
- generated token count
- end-to-end generated tokens / second
- post-first-token generated tokens / second
- per-run variability across several runs

### 3. Investigate vLLM prefix caching

Status: `done`

Required work:

- source review in local `third_party/vllm`
- repeated-prompt experiments with prefix caching enabled and disabled
- record whether caching is unsupported, partially supported, or silently bypassed

### 4. Profile prefill on all three codebases

Status: `done`

Required cases:

- prefix `4`
- prefix `128`
- prefix `4096`

Target:

- one generated token
- token IDs where possible to avoid tokenizer noise
- comparable artifact summaries for kernel mix and dominant bottlenecks

### 5. Analyze optimization opportunities for the native runtime

Status: `done`

Required output:

- cross-codebase comparison of dominant prefill work
- concrete opportunities for the native from-scratch implementation
- clear statement of which TRT-LLM / vLLM design choices matter and which do not

## Current One-Token TTFT Baselines

### Native Runtime

Artifacts:

- `artifacts/benchmarks/ttft_20260406_current_tail4_prefix4_128_4096.stdout.txt`

Current cold medians:

- `prefix4 = 53.367 ms`
- `prefix128 = 124.820 ms`
- `prefix4096 = 2499.870 ms`

### vLLM

Artifacts:

- `artifacts/benchmarks/vllm_ttft_20260406_clean_prefix1_4_128_4096.json`

Current medians:

- `decode_1 = 30.885 ms`
- `prefix4 = 34.445 ms`
- `prefix128 = 31.469 ms`
- `prefix4096 = 78.328 ms`

Current caveat:

- `num_cached_tokens = 0` in every recorded iteration, so this artifact is only
  a cold-TTFT reference today, not a valid cache-hit reference.

### TRT-LLM PyTorch Serve

Artifacts:

- `artifacts/benchmarks/trtllm_serve_ttft_prefix4_out1_20260406_current.json`
- `artifacts/benchmarks/trtllm_serve_ttft_prefix128_out1_20260406_current.json`
- `artifacts/benchmarks/trtllm_serve_ttft_prefix4096_out1_20260406_current.json`

Current medians:

- `prefix4 = 92.335 ms`
- `prefix128 = 160.844 ms`
- `prefix4096 = 375.780 ms`

Current caveat:

- this is the supported local NemotronH path through `trtllm-serve` with the
  PyTorch backend, not TensorRT engine build.

## Current Throughput Baselines

### Native Runtime

Artifacts:

- `artifacts/benchmarks/nano_fused_decode_16tok_phased_20260406_current.json`
- `artifacts/benchmarks/nano_fused_decode_16tok_steady_state_20260406_current.json`
- `artifacts/benchmarks/nano_fused_decode_16tok_cached_head_20260406_current.json`

Current values:

- phased full decode: `12.077 tok/s`
- phased steady decode: `79.223 tok/s`
- steady-state decode: `76.574 tok/s`
- cached-head decode: `76.926 tok/s`

### vLLM

Artifacts:

- `artifacts/benchmarks/vllm_generate_throughput_16prompt_16gen_20260406_current.json`

Current value:

- full decode: `40.856 tok/s`

### TRT-LLM PyTorch Serve

Artifacts:

- `artifacts/benchmarks/trtllm_serve_throughput_prompt16_gen16_20260406_current.json`

Current values:

- request throughput: `12.640 req/s`
- output throughput: `179.073 tok/s`
- total token throughput: `381.320 tok/s`
- mean TPOT: `115.851 ms`

Current caveat:

- this artifact is `--non-streaming`, so its TTFT field is invalid and should
  not be used as a TTFT comparison.

## Long-Response Streaming Comparison

Shape:

- one request at a time
- prompt:
  `Tell me a complicated and hilarious joke in the form of a story.`
- `max_tokens = 512`
- `warmup_runs = 1`
- `runs = 3`
- prompt varied across measured runs with a deterministic run marker suffix to
  avoid prompt-cache reuse

Artifacts:

- native:
  `artifacts/benchmarks/native_story_stream_20260406_warm.json`
- vLLM:
  `artifacts/benchmarks/vllm_story_stream_20260406_warm.json`
- TRT-LLM:
  `artifacts/benchmarks/trtllm_story_stream_20260406_warm.json`

Summary:

| runtime | TTFT mean | TTFT min/max | total gen tok/s mean | post-first tok/s mean | generated tokens |
|---|---:|---:|---:|---:|---:|
| native runtime | `388.379 ms` | `375.359 / 412.198 ms` | `66.521 tok/s` | `69.930 tok/s` | `512` |
| vLLM | `59.058 ms` | `57.194 / 61.987 ms` | `41.742 tok/s` | `41.862 tok/s` | `512` |
| TRT-LLM PyTorch serve | `34.936 ms` | `31.803 / 38.043 ms` | `260.331 tok/s` | `264.521 tok/s` | `512` |

Observed shape:

- TRT-LLM has the best end-to-end long-response serving throughput by a large
  margin on this setup.
- vLLM has much lower TTFT than the native runtime on this prompt, but its
  sustained generation rate is materially lower than the native runtime.
- the native runtime is still paying a large first-token cost relative to both
  external baselines, while its steady decode rate remains better than vLLM on
  this single-request story benchmark.

Commands:

### Native Long-Response Streaming

```bash
uv run proj-2026-04-05-1704/native_story_stream_bench.py \
  --max-new-tokens 512 \
  --target-context-tokens 2048 \
  --warmup-runs 1 \
  --runs 3 \
  --vary-prompt \
  --result-path artifacts/benchmarks/native_story_stream_20260406_warm.json
```

### vLLM Long-Response Streaming

Server:

```bash
proj-2026-04-05-1704/run_vllm_nano_serve_local.sh
```

Client:

```bash
uv run proj-2026-04-05-1704/openai_story_stream_bench.py \
  --base-url http://127.0.0.1:8011 \
  --model-name nemotron-nano-vllm \
  --tokenizer artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --max-tokens 512 \
  --warmup-runs 1 \
  --runs 3 \
  --vary-prompt \
  --result-path artifacts/benchmarks/vllm_story_stream_20260406_warm.json
```

### TRT-LLM Long-Response Streaming

```bash
uv run proj-2026-04-05-1704/openai_story_stream_bench.py \
  --server-script proj-2026-04-05-1704/run_trtllm_nano_serve.sh \
  --base-url http://127.0.0.1:8013 \
  --model-name nemotron-nano-trtllm \
  --tokenizer artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --max-tokens 512 \
  --warmup-runs 1 \
  --runs 3 \
  --vary-prompt \
  --result-path artifacts/benchmarks/trtllm_story_stream_20260406_warm.json \
  --port-env TRTLLM_PORT \
  --port 8013
```

## vLLM Prefix Caching Investigation

Key local source references:

- `third_party/vllm/vllm/model_executor/models/nemotron_h.py`
  `NemotronHForCausalLM` inherits `SupportsMambaPrefixCaching`
- `third_party/vllm/tests/models/language/generation/test_hybrid.py`
  explicitly exercises APC on hybrid / mamba models
- `third_party/vllm/vllm/model_executor/models/config.py`
  sets `mamba_cache_mode = "all"` for supported models when APC is enabled

Critical runtime log detail from this model/config:

- `Setting attention block size to 4176 tokens to ensure that attention page size is >= mamba page size.`
- `Padding mamba page size by 0.19% to ensure that mamba page size and attention page size are exactly equal.`

Artifacts:

- `artifacts/benchmarks/vllm_prefix_cache_probe_20260406.json`
- `artifacts/benchmarks/vllm_prefix_cache_probe_4176_20260406.json`
- `artifacts/benchmarks/vllm_prefix_cache_probe_4200_20260406.json`

Probe command family:

```bash
VLLM_USE_FLASHINFER_MOE_FP4=1 \
VLLM_FLASHINFER_MOE_BACKEND=throughput \
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-05-1704/vllm_prefix_cache_probe.py \
  --token-count <128|4096|4176|4200> \
  --repetitions <2|3> \
  --result-path artifacts/benchmarks/<artifact>.json
```

Results:

- `128` tokens:
  - APC off: warm repetitions `31.289 ms`, `29.314 ms`, `num_cached_tokens = 0`
  - APC on: warm repetitions `39.553 ms`, `37.706 ms`, `num_cached_tokens = 0`
- `4096` tokens:
  - APC off: warm repetitions `79.250 ms`, `78.993 ms`, `num_cached_tokens = 0`
  - APC on: warm repetitions `79.979 ms`, `79.576 ms`, `num_cached_tokens = 0`
- `4176` tokens:
  - APC off: warm repetition `83.377 ms`, `num_cached_tokens = 0`
  - APC on: warm repetition `83.366 ms`, `num_cached_tokens = 0`
- `4200` tokens:
  - APC off: warm repetition `82.267 ms`, `num_cached_tokens = 0`
  - APC on: second repetition `395.310 ms`, `num_cached_tokens = 4176`

Interpretation:

- APC is implemented for NemotronH in vLLM on this stack.
- It is not visible at `128`, `4096`, or even exactly `4176` tokens on this
  model because reuse is constrained by the hybrid full-block boundary.
- The most likely explanation for `4176 -> 0 cached` and `4200 -> 4176 cached`
  is that only complete reusable prefix blocks are admitted and the final prompt
  token is not reusable. With a block size of `4176`, the first prompt length
  that can expose one cached block is therefore `> 4176`.
- On this local configuration, APC is therefore real but partial and not
  practically useful for the shorter prompt lengths we used in earlier vLLM
  comparisons.
- The observed `4200` APC-hit repetition was much slower than the APC-off warm
  repetition (`395 ms` vs `82 ms`), so this is not currently a usable hot-prefix
  performance path for this model on this stack.

## Prefill Profiling Matrix

Artifact root:

- `artifacts/profiles/external_prefill_compare_20260406/`

### Native Runtime

Method:

- `nsys` over `nano_prefix_cache_ttft_bench`
- `cold_prefill_prefix4`, `cold_prefill_prefix128`, `cold_prefill_prefix4096`
- higher iteration counts on short cases to dilute model-load noise

Commands:

```bash
export NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json

/usr/local/cuda-13.0/bin/nsys profile --force-overwrite true \
  --trace=cuda,nvtx,osrt --sample=none \
  --output artifacts/profiles/external_prefill_compare_20260406/native/prefix4 \
  build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 --iterations 80 --tail-token-count 4 --prefix-length 4 --case cold_prefill_prefix4

/usr/local/cuda-13.0/bin/nsys profile --force-overwrite true \
  --trace=cuda,nvtx,osrt --sample=none \
  --output artifacts/profiles/external_prefill_compare_20260406/native/prefix128 \
  build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 --iterations 40 --tail-token-count 4 --prefix-length 128 --case cold_prefill_prefix128

/usr/local/cuda-13.0/bin/nsys profile --force-overwrite true \
  --trace=cuda,nvtx,osrt --sample=none \
  --output artifacts/profiles/external_prefill_compare_20260406/native/prefix4096 \
  build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 --iterations 5 --tail-token-count 4 --prefix-length 4096 --case cold_prefill_prefix4096
```

### vLLM

Method:

- direct local `LLM.generate`
- exact prompt token IDs
- one warmup request
- worker-side CUDA profiler enabled through `ProfilerConfig(profiler="cuda")`
- `nsys --capture-range=cudaProfilerApi --trace-fork-before-exec=true`

Artifacts:

- `proj-2026-04-05-1704/vllm_prefill_profile.py`
- `artifacts/benchmarks/vllm_prefill_profile_prefix{4,128,4096}_20260406.json`

Representative command:

```bash
/usr/local/cuda-13.0/bin/nsys profile --force-overwrite true \
  --trace=cuda,nvtx,osrt --sample=none \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output artifacts/profiles/external_prefill_compare_20260406/vllm/prefix128 \
  bash -lc '
    export VLLM_USE_FLASHINFER_MOE_FP4=1
    export VLLM_FLASHINFER_MOE_BACKEND=throughput
    proj-2026-04-03-2113/run-vllm.sh \
      proj-2026-04-05-1704/vllm_prefill_profile.py \
      --token-count 128 \
      --result-path artifacts/benchmarks/vllm_prefill_profile_prefix128_20260406.json
  '
```

### TRT-LLM PyTorch

Method:

- direct local `tensorrt_llm.LLM.generate`
- exact prompt token IDs
- one warmup request
- `cudaProfilerStart/Stop` around the measured request
- `nsys --capture-range=cudaProfilerApi --trace-fork-before-exec=true`

Artifacts:

- `proj-2026-04-05-1704/trtllm_prefill_profile.py`
- `artifacts/benchmarks/trtllm_prefill_profile_prefix{4,128,4096}_20260406.json`

Representative command:

```bash
/usr/local/cuda-13.0/bin/nsys profile --force-overwrite true \
  --trace=cuda,nvtx,osrt --sample=none \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output artifacts/profiles/external_prefill_compare_20260406/trtllm/prefix128 \
  bash -lc '
    source .venv-trtllm/bin/activate
    export PYTHONPATH=third_party/TensorRT-LLM${PYTHONPATH:+:$PYTHONPATH}
    export LD_LIBRARY_PATH=.venv-trtllm/lib:.venv-trtllm/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    python proj-2026-04-05-1704/trtllm_prefill_profile.py \
      --token-count 128 \
      --result-path artifacts/benchmarks/trtllm_prefill_profile_prefix128_20260406.json
  '
```

### Profiled Prefill Elapsed Times

These are the elapsed times from the direct / profiled request harnesses, not
the longer benchmark medians from the earlier serve/TTFT sections.

| runtime | prefix4 | prefix128 | prefix4096 |
|---|---:|---:|---:|
| native runtime cold TTFT median | `54.016 ms` | `127.698 ms` | `2503.116 ms` |
| vLLM direct prefill+decode1 | `48.011 ms` | `40.396 ms` | `82.938 ms` |
| TRT-LLM direct prefill+decode1 | `35.922 ms` | `38.407 ms` | `66.789 ms` |

## Cross-Codebase Profile Readout

### Prefix 4

- native runtime:
  - `35.1%` routed expert custom kernel
  - `34.0%` shared expert contiguous WMMA matvec
  - `8.1%` CUTLASS GEMM
  - `5.6%` explicit expert selection
- vLLM:
  - dominant work is already grouped CUTLASS MoE GEMM plus BF16 WMMA / GEMV
  - no large bespoke routing / scatter / pack kernel dominates the trace
- TRT-LLM:
  - similar grouped CUTLASS dominance
  - notable extra `cuda_core_gemm_nvfp4` kernel already appears at `11.5%`

### Prefix 128

- native runtime:
  - `58.2%` routed expert custom kernel
  - `10.1%` shared expert contiguous WMMA
  - `7.7%` Mamba prefill
  - `7.1%` second shared expert WMMA stage
  - `2.2%` explicit expert selection
- vLLM:
  - the top two grouped CUTLASS kernels alone account for about `61.6%`
  - the rest of MoE permute/activation machinery is low single digits
- TRT-LLM:
  - the top two grouped CUTLASS kernels account for about `64.1%`
  - the next visible routed FP4 GEMM kernel is only `6.0%`

### Prefix 4096

- native runtime:
  - `50.5%` routed expert custom kernel
  - `25.7%` Mamba prefill
  - `7.2%` shared expert WMMA
  - `6.4%` paged attention
- vLLM:
  - MoE / GEMM / activation still dominates, but no single custom row kernel
    takes over the trace
  - flash attention is about `7.6%`
  - Mamba chunk scan is only about `3.0%`
- TRT-LLM:
  - grouped CUTLASS / FP4 GEMM kernels dominate the trace
  - FMHA is about `5.1%`
  - Mamba chunk scan is about `3.8%`

## Optimization Opportunities For The Native Runtime

The current profile comparison strongly suggests this work order:

1. Replace the remaining custom routed-expert row kernel with a more
   TRT-LLM-like grouped GEMM math core.
   Our custom routed kernel still dominates native prefill at every prompt
   length. Both external baselines already spend most of this time inside a
   small set of grouped CUTLASS kernels instead.

2. Reduce explicit MoE staging overhead around routing and row handling.
   Native still exposes standalone selection / scatter / quant-dequant stages as
   visible top kernels, especially on short prompts. External baselines push
   more of this work into grouped-kernel boundaries with lighter surrounding
   glue.

3. Revisit the long-prefix Mamba path after routed MoE.
   At `prefix4096`, native Mamba prefill is still `25.7%` of GPU time, while
   the comparable `_chunk_scan_fwd_kernel` bucket is only `~3-4%` in vLLM and
   TRT-LLM. That is a real second hotspot once routed MoE is brought down.

4. Do not prioritize attention first.
   Native attention is visible at `prefix4096`, but it is not the dominant gap
   relative to the external baselines. Routed MoE and Mamba are larger and
   clearer optimization opportunities.

5. Keep prefix-cache/APC expectations separate from cold-prefill analysis.
   The vLLM APC probes show that cache support exists but only becomes visible
   above the hybrid full-block boundary. That behavior explains earlier cache
   confusion, but it is not the reason native cold prefill is slower.

## Current Commands

### TRT-LLM One-Token TTFT

Server:

```bash
proj-2026-04-05-1704/run_trtllm_nano_serve.sh
```

Client shape:

```bash
source .venv-trtllm/bin/activate
python third_party/TensorRT-LLM/tensorrt_llm/serve/scripts/benchmark_serving.py \
  --backend openai-chat \
  --base-url http://127.0.0.1:8012 \
  --model nemotron-nano-trtllm \
  --tokenizer artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --dataset-name random \
  --random-input-len <4|128|4096> \
  --random-output-len 1 \
  --random-ids \
  --tokenize-on-client \
  --num-prompts 6 \
  --seed 7 \
  --trust-remote-code \
  --disable-tqdm \
  --save-result \
  --result-filename <artifact-name>.json \
  --result-dir artifacts/benchmarks
```

### vLLM One-Token TTFT

Artifacts came from the local vLLM path documented in
`docs/vllm_rtx5090.md` and the benchmark wrapper family under
`proj-2026-04-03-0318` / `proj-2026-04-03-2113`.

## Progress Log

### 2026-04-06

- confirmed TRT-LLM source build is not required for engine build comparison;
  the working local NemotronH path is PyTorch backend serve
- added local TRT-LLM serve config and wrappers in this project directory
- validated local TRT-LLM serve on the local Nano checkpoint
- captured direct one-token TTFT results for TRT-LLM serve at prefix `4`,
  `128`, and `4096`
- recorded the current external baseline matrix and caveats
- added apples-to-apples long-response streaming benchmarks for native runtime,
  vLLM, and TRT-LLM
- confirmed the current warmed long-response comparison on one request / 512
  generated tokens
- built direct exact-token prefill profiling harnesses for vLLM and TRT-LLM
- captured `nsys` prefill profiles for native runtime, vLLM, and TRT-LLM at
  prefix `4`, `128`, and `4096`
- confirmed that vLLM NemotronH APC is present but only becomes visible when
  the prompt exceeds the hybrid full-block boundary (`4200 -> 4176` cached
  tokens), and that this path is not a practical speedup on the local stack
- recorded the profile-based optimization priorities for the native runtime
