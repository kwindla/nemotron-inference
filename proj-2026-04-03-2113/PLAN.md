# Plan: Project-Local vLLM Installation for Oracle Testing and Benchmarking

Project directory: `./proj-2026-04-03-2113`

## Context

We need a local vLLM path on RTX 5090 (SM120) for two reasons:

- oracle testing against our from-scratch runtime
- external TTFT benchmarking on the same hardware

The original diagnosis was wrong. The blocker was not "Nemotron Nano NVFP4 is unsupported on SM120." The real issues we found are:

- `vllm-env` resolved to `torch 2.10.0+cu130`, and plain PyTorch GEMMs fail in that environment on this machine. vLLM then fails higher up in the same stack.
- The stable local path is `vllm-env-cu128`, which uses `torch 2.10.0+cu128` and can run the model.
- The local `third_party/vllm` checkout already contains compiled extension binaries, so `PYTHONPATH=third_party/vllm` is enough to use the local source tree. A fresh rebuild is optional, not required for unblock.
- For Nemotron 3 Nano NVFP4 on Blackwell, the relevant vLLM backend is `flashinfer_cutlass`, not Marlin.
- On this display-attached RTX 5090, stable vLLM startup also needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; without it, KV-cache allocation can OOM during engine init even on the working cu128 stack.
- For local metadata inspection in the benchmark script, `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is required because `llm.apply_model(...)` serializes a Python callable.

The comparison target is therefore:

- vLLM local source checkout on RTX 5090
- `torch 2.10.0+cu128`
- `flashinfer-python==0.6.6`
- `flashinfer-cubin==0.6.6`
- `moe_backend=flashinfer_cutlass`

## Steps

- [x] **1. Isolate the real failure mode**
  Prove whether the failure is vLLM-specific or lower in the CUDA/PyTorch stack.

  Validation that matters:
  - plain GEMMs fail in `vllm-env` (`torch 2.10.0+cu130`)
  - the same GEMMs succeed in the cu128 environment
  - conclusion: the blocker is the local cu130 runtime stack, not a blanket SM120 incompatibility

- [x] **2. Create a stable project-local vLLM runtime path**
  Use `vllm-env-cu128` as the default interpreter and point it at the local `third_party/vllm` checkout with `PYTHONPATH`.

  Wrapper defaults now need to do all of the following:
  - use `vllm-env-cu128/bin/python`
  - export `PYTHONPATH=third_party/vllm`
  - export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - for Nemotron Nano NVFP4 server and benchmark paths, export:
    - `VLLM_USE_FLASHINFER_MOE_FP4=1`
    - `VLLM_FLASHINFER_MOE_BACKEND=throughput`

- [x] **3. Validate minimal local inference on RTX 5090**
  Run a smoke test from the local checkout and confirm that the model actually generates text.

  Success criteria:
  - vLLM engine starts on the 5090
  - selected backend is `FLASHINFER_CUTLASS`
  - `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` produces coherent output

  Working command:

  ```bash
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  proj-2026-04-03-2113/run-vllm.sh \
    proj-2026-04-03-2113/smoke_vllm_nano.py
  ```

- [x] **4. Update oracle and benchmark tooling**
  Make the benchmark path use the same working local stack.

  Required changes:
  - `proj-2026-04-03-0318/run_bench_vllm.sh`
    - default to `vllm-env-cu128/bin/python`
    - export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    - export `VLLM_USE_FLASHINFER_MOE_FP4=1`
    - export `VLLM_FLASHINFER_MOE_BACKEND=throughput`
    - export `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
    - invoke the benchmark with:
      - `--trust-remote-code`
      - `--gpu-memory-utilization 0.8`
      - `--moe-backend flashinfer_cutlass`
  - `proj-2026-04-03-0318/bench_vllm_nano.py`
    - default `--trust-remote-code` to enabled
    - default `--gpu-memory-utilization` to `0.8`

- [x] **5. Capture the external baseline**
  Run the vLLM TTFT sweep and store the result JSON for comparison against our runtime.

  Working command:

  ```bash
  bash proj-2026-04-03-0318/run_bench_vllm.sh --skip-moe-profile
  ```

  Current baseline in `proj-2026-04-03-0318/vllm_baseline_results.json`:
  - selected backend: `FLASHINFER_CUTLASS`
  - `decode_1`: mean `31.660 ms`
  - `prefill_32`: mean `30.955 ms`
  - `prefill_64`: mean `29.934 ms`
  - `prefill_128`: mean `34.195 ms`
  - `prefill_256`: mean `33.471 ms`

  Important limitation:
  - `enable_return_routed_experts=True` currently crashes this vLLM setup on larger prefill cases with an index error in `routed_experts_capturer.py`.
  - The benchmark therefore disables routed-expert capture by default and records TTFT/backends only.

## Progress

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Isolate failure mode | done | `torch+cu130` is broken locally; `torch+cu128` works |
| 2 | Stable local runtime path | done | wrappers now default to `vllm-env-cu128` and local source |
| 3 | Minimal local inference | done | smoke test generates text on RTX 5090 with `FLASHINFER_CUTLASS` |
| 4 | Oracle/benchmark tooling | done | benchmark wrapper and script defaults now match the working stack |
| 5 | External baseline capture | done | full TTFT sweep completed; routed-expert capture disabled due a vLLM bug on larger prefill cases |
