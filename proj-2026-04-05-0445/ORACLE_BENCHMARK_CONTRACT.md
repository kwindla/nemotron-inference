# Oracle Benchmark Contract

This file defines the external benchmark contract for
`proj-2026-04-05-0445/ATTENTION_PLAN.md`.

If a benchmark cannot satisfy this contract closely enough, its numbers are not
decision-grade for attention architecture work.

## Purpose

Use external stacks to estimate the realistic performance floor for Nano
attention on this exact local machine before writing runtime backend code.

The primary oracle is vLLM. TRT-LLM is a secondary cross-check for layer-level
composition and NVTX profiling, not the main MoE oracle on this machine.

## Environment Contract

- GPU: RTX 5090, consumer Blackwell, `SM120`
- Driver / toolkit: current local production tuple for this project
- Python imports must prefer the vendored vLLM tree:
  - `PYTHONPATH=/home/khkramer/src/nemotron-inference/third_party/vllm`
- Every oracle run must record:
  - `python --version`
  - `torch.__version__`
  - `torch.version.cuda`
  - `vllm.__file__`
  - device name and compute capability
  - whether vLLM reports `supports_trtllm_attention()`

## Nano Geometry Contract

These values come from the runtime's Nano config:

- query heads: `32`
- KV heads: `2`
- head dim: `128`
- page/block size: `16`
- causal only
- retained context target: up to `64k`

Source: `runtime/src/api/single_token_forward_model.cpp`

## Dtype Contract

- Query / KV oracle target: BF16
- KV cache oracle target: BF16
- In the current vendored vLLM benchmark harness, `kv_cache_dtype=auto` is
  acceptable only if the probe artifact confirms the resolved model dtype is
  `torch.bfloat16`.
- Any run where the benchmark resolves to a non-BF16 dtype is invalid for this
  project.

## Layer Count Contract

- Use `num_layers=6` in the vLLM attention benchmark harness.
- Reason: Nano currently has six attention layers in the local runtime, and the
  harness reports seconds per layer after dividing total time by `num_layers`.
- This keeps CUDA-graph and setup behavior closer to the real layer-count mix
  without pretending the benchmark is a full-model trace.

## Primary Oracle Matrix

These runs are required before choosing the production attention architecture.

### Single-request prefill

- `q256`
- `q1k`
- `q4k`

### Single-request extend

- `q128s8k`
- `q256s8k`
- `q512s8k`
- `q1ks8k`
- `q128s32k`
- `q256s32k`
- `q512s32k`
- `q1ks32k`
- `q128s64k`
- `q256s64k`
- `q512s64k`
- `q1ks64k`

## Secondary Concurrency Matrix

These runs are representative scheduler-pressure checks, not the primary
architecture gate:

### Four-request prefill

- `4q256`
- `4q1k`

### Four-request extend

- `4q128s8k`
- `4q256s32k`
- `4q1ks64k`

Long four-way prefills larger than `4q1k` are intentionally excluded from the
default matrix because the project is allowed to serialize pathological long
prefills instead of optimizing them as the main serving mode.

## Backend Set

For vLLM standard-attention oracle runs, measure:

- `FLASH_ATTN`
- `FLASHINFER`
- `TRITON_ATTN`

Do not assume all three are valid on every run. The benchmark outputs and probe
artifact must show what actually executed.

## Output Contract

Every oracle run should save:

- environment / backend probe JSON
- benchmark CSV
- benchmark JSON
- benchmark stdout/stderr log

Store artifacts under `proj-2026-04-05-0445/oracle_runs/<timestamp>/`.

## Interpretation Rules

- Treat vLLM attention numbers as per-layer floors, not as direct end-to-end
  TTFT predictions.
- Use single-request results to choose the end-state backend direction.
- Use four-request results to reject designs that collapse under the actual
  concurrency cap.
- If a backend requires private cubins, unsupported architecture families, or a
  page-table contract that does not map cleanly onto this runtime, count it as
  prior art rather than an integration target.
