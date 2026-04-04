# Local vLLM on RTX 5090

This note documents the working local vLLM path for
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` on RTX 5090 / SM120.

## Summary

The stable local path is:

- local source checkout: `third_party/vllm`
- Python env: `vllm-env-cu128`
- PyTorch: `2.10.0+cu128`
- FlashInfer: `0.6.6`
- MoE backend: `flashinfer_cutlass`

The older `vllm-env` path is not the one to use for this model on this machine.
It resolved to `torch 2.10.0+cu130`, and plain GEMMs failed in that environment.

## Required Runtime Settings

These settings are part of the working local setup:

- `PYTHONPATH=third_party/vllm`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `VLLM_USE_FLASHINFER_MOE_FP4=1`
- `VLLM_FLASHINFER_MOE_BACKEND=throughput`

The helper scripts below already set the required defaults.

## Smoke Test

Run the local smoke test:

```bash
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/smoke_vllm_nano.py
```

Expected behavior:

- the engine loads successfully on RTX 5090
- the selected MoE backend is `FLASHINFER_CUTLASS`
- the model emits text for a simple prompt

## OpenAI-Compatible Serving

Start the local server:

```bash
proj-2026-04-03-2113/vllm-serve.sh
```

Defaults encoded by the wrapper:

- Python: `vllm-env-cu128/bin/python`
- model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
- `--trust-remote-code`
- `--enforce-eager`
- `--moe-backend flashinfer_cutlass`
- `--gpu-memory-utilization 0.8`

You can override the model or memory budget if needed:

```bash
VLLM_MODEL=... \
VLLM_GPU_MEMORY_UTILIZATION=0.75 \
proj-2026-04-03-2113/vllm-serve.sh
```

## Benchmarking

Run the offline TTFT benchmark:

```bash
bash proj-2026-04-03-0318/run_bench_vllm.sh --skip-moe-profile
```

The benchmark writes results to:

- `proj-2026-04-03-0318/vllm_baseline_results.json`

Current recorded baseline:

- `decode_1`: mean `31.660 ms`
- `prefill_32`: mean `30.955 ms`
- `prefill_64`: mean `29.934 ms`
- `prefill_128`: mean `34.195 ms`
- `prefill_256`: mean `33.471 ms`

## Reusable Scripts

These are the local entry points worth reusing instead of open-coding the env:

- `proj-2026-04-03-2113/run-vllm.sh`
  Runs any Python script against the local `third_party/vllm` checkout using the
  working `vllm-env-cu128` environment.
- `proj-2026-04-03-2113/vllm-serve.sh`
  Starts the OpenAI-compatible server for the Nano NVFP4 checkpoint on RTX 5090.
- `proj-2026-04-03-2113/smoke_vllm_nano.py`
  Loads the model, prints the first fused-MoE backend selection, and runs a short
  text generation smoke test.
- `proj-2026-04-03-2113/save_runtime_oracle.py`
  Resolves a prompt from token IDs, a named prompt fixture, or raw text, then runs
  the from-scratch runtime oracle binary and writes an oracle JSON.
- `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py`
  Loads a saved runtime oracle JSON, or generates one on demand from the same prompt
  sources as `save_runtime_oracle.py`, then runs vLLM on the same `prompt_token_ids`
  and compares generated token IDs exactly.
- `proj-2026-04-03-0318/run_bench_vllm.sh`
  Runs the local TTFT benchmark and writes `vllm_baseline_results.json`.

## Runtime Oracle Token Generation

The from-scratch runtime oracle writer is:

- binary: `build/testing/nano_save_prompt_oracle`
- source: `testing/api/nano_save_prompt_oracle.cpp`

If the binary is not built yet:

```bash
cmake --build build --target nano_save_prompt_oracle
```

The C++ binary now accepts arbitrary prompt token IDs directly:

```bash
NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json \
build/testing/nano_save_prompt_oracle \
  --prompt-token-ids "10,25708,1010,4568" \
  --decode-token-count 4
```

It also supports `--prompt-token-ids-file` and `--output`.

Default output behavior:

- if `--output` is omitted, the binary writes
  `artifacts/oracles/nano_<prompt_token_count>_token_oracle_<timestamp>.json`
- the JSON includes:
  - `prompt_token_ids`
  - prompt-boundary `top5` logits
  - `boundary_token_id`
  - `generated_token_ids`
  - route metadata and UTC timestamp

For arbitrary prompts, the reusable entry point is the Python wrapper:

```bash
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/save_runtime_oracle.py \
  --prompt-name short_chat \
  --decode-token-count 4
```

Raw prompt text:

```bash
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/save_runtime_oracle.py \
  --prompt-text "Hello from the runtime oracle path." \
  --decode-token-count 4
```

Direct token IDs:

```bash
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/save_runtime_oracle.py \
  --prompt-token-ids "10,25708,1010,4568" \
  --decode-token-count 4
```

Prompt-source behavior:

- `--prompt-name <name>` loads `testing/oracle/prompts.json` and applies the model chat template
- `--prompt-text` and `--prompt-file` use plain tokenizer encoding without a chat template
- `--prompt-token-ids` and `--prompt-token-ids-file` bypass tokenization entirely
- `--output`, `--manifest`, and `--build-dir` can still be overridden explicitly

## Compare Runtime Oracle Against vLLM

Run the vLLM-side comparison through the local wrapper:

```bash
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/compare_vllm_runtime_oracle.py
```

By default, the script picks the newest matching oracle in `artifacts/oracles/`.
To compare a specific file:

```bash
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/compare_vllm_runtime_oracle.py \
  --oracle artifacts/oracles/manual_oracle.json
```

To generate the runtime oracle on demand from the named `short_chat` prompt and then
compare it to vLLM in one command:

```bash
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/compare_vllm_runtime_oracle.py \
  --prompt-name short_chat \
  --decode-token-count 4
```

To compare a raw prompt string instead:

```bash
proj-2026-04-03-2113/run-vllm.sh \
  proj-2026-04-03-2113/compare_vllm_runtime_oracle.py \
  --prompt-text "Hello from the shared oracle path." \
  --decode-token-count 4
```

Useful behavior for automation:

- exit status `0`: exact generated token match
- exit status `1`: divergence, with `first_mismatch_index` printed
- when a prompt source is provided, the script first writes a fresh runtime oracle JSON
- the script prints:
  - `prompt_token_ids`
  - decoded prompt text
  - runtime generated token IDs and text
  - vLLM generated token IDs and text

Interpretation:

- exact token agreement is the primary oracle
- readable text is a secondary smoke check
- for `--prompt-name`, the prompt text printed by the compare script is the fully rendered chat-template form
- for `--prompt-text`, both stacks see the same plain tokenized text without chat templating

## Known Limitations

- `enable_return_routed_experts=True` currently crashes this local vLLM setup on larger prefill cases with an index error in `routed_experts_capturer.py`. The benchmark disables routed-expert capture by default.
- FlashInfer autotuning may log skipped tactics such as `Failed to initialize cutlass TMA WS grouped gemm`. On this setup those warnings did not prevent successful inference or TTFT benchmarking.
- A Triton import warning for `SparseMatrix` may appear during startup. It has not blocked the working Nemotron Nano NVFP4 path described here.
- In the current one-command `short_chat` compare flow, vLLM still logs a FlashInfer autotuner failure for one tactic on this machine, but the engine continues and the comparison completes.

## Related Files

- `proj-2026-04-03-2113/run-vllm.sh`
- `proj-2026-04-03-2113/vllm-serve.sh`
- `proj-2026-04-03-2113/smoke_vllm_nano.py`
- `proj-2026-04-03-2113/save_runtime_oracle.py`
- `proj-2026-04-03-2113/compare_vllm_runtime_oracle.py`
- `proj-2026-04-03-2113/runtime_oracle_support.py`
- `proj-2026-04-03-0318/run_bench_vllm.sh`
- `proj-2026-04-03-0318/bench_vllm_nano.py`
- `proj-2026-04-03-0318/vllm_baseline_results.json`
- `testing/api/nano_save_prompt_oracle.cpp`
