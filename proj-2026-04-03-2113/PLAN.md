# Plan: Project-Local vLLM Installation for Benchmarking

Project directory: `./proj-2026-04-03-2113`

## Context

We need vLLM running on RTX 5090 (SM120) for oracle testing and performance comparison against our unified fused backend. Key constraints discovered during research and Codex review:

- Prebuilt vLLM wheels lack SM120 arch flags — must build from source
- FlashInfer NVFP4 MoE kernels are broken on SM120 (flashinfer#2577, vllm#34452)
- vLLM's "auto-select" does NOT reliably fall through to Marlin on v0.19.0 — FlashInfer CUTLASS still claims SM120 support. Must explicitly force `marlin` via env var or constructor arg.
- FlashAttention 3/4 unsupported on SM120 — but vLLM v0.19.0 already defaults to FA2 on SM120 via `fa_utils.py:74`. No env var needed (VLLM_FLASH_ATTN_VERSION is a no-op in v0.19.0).
- Our `third_party/vllm` checkout is v0.19.0 with SM120-specific CUDA kernel files
- Local environment: CUDA 13.0, PyTorch 2.10.0+cu128, Python 3.12.10, uv 0.11.1
- `pyproject.toml` requires `torch == 2.10.0` (plain); uv with `--torch-backend=auto` may resolve `+cu130` from the CUDA 13.0 driver. This is acceptable — the editable install just needs compatible torch, not the exact system torch.
- `VLLM_USE_PRECOMPILED=1` is wrong for this setup — it skips local CUDA compilation
- vLLM v0.19.0 pins `flashinfer-python==0.6.6` and `flashinfer-cubin==0.6.6` in `requirements/cuda.txt`

The comparison target is "vLLM on RTX 5090 with Marlin MoE backend" — what a user would get on this GPU after working around the FlashInfer bugs. This is the right external bar for SM120.

## Steps

- [x] **1. Create uv-managed venv and build vLLM from source**
  Create a project-local virtualenv at `vllm-env/` using uv. Build vLLM v0.19.0 from `third_party/vllm` with SM120 support.

  ```bash
  cd /home/khkramer/src/nemotron-inference
  uv venv vllm-env --python 3.12 --seed

  # Build vLLM from source with SM120 arch
  TORCH_CUDA_ARCH_LIST="12.0" \
  MAX_JOBS=6 \
  uv pip install --python vllm-env/bin/python \
    -e third_party/vllm --torch-backend=auto
  ```

  Verify: `vllm-env/bin/python -c "import vllm; print(vllm.__version__)"` should print `0.19.0`.
  Verify SM120 compiled: `vllm-env/bin/python -c "import vllm._C; print('C extension OK')"`.

  Add `vllm-env/` to `.gitignore` if not already there.
  Key files: `third_party/vllm/pyproject.toml`, `.gitignore`

- [x] **2. Create wrapper scripts for running vLLM tools**
  Create `proj-2026-04-03-2113/run-vllm.sh` that:
  - Uses `vllm-env/bin/python` directly (no activate needed)
  - Does NOT set `VLLM_FLASH_ATTN_VERSION` (no-op in v0.19.0, FA2 already default on SM120)
  - Does NOT set `VLLM_USE_FLASHINFER_MOE_FP4` (we want Marlin, not FlashInfer)
  - Passes through arguments to the given command

  Create `proj-2026-04-03-2113/vllm-serve.sh` that:
  - Starts vLLM serving Nemotron Nano NVFP4 with correct settings
  - Uses `--dtype auto --trust-remote-code`
  - Sets `--gpu-memory-utilization 0.9`
  - Explicitly forces Marlin MoE backend to avoid FlashInfer CUTLASS bug

  Key files: `proj-2026-04-03-2113/run-vllm.sh`, `proj-2026-04-03-2113/vllm-serve.sh`

- [ ] **3. Validate vLLM loads and runs Nemotron Nano on RTX 5090**
  Run a minimal vLLM inference test via the wrapper:
  ```bash
  proj-2026-04-03-2113/run-vllm.sh python3 -c "
  from vllm import LLM, SamplingParams
  llm = LLM('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4',
             trust_remote_code=True, max_model_len=256,
             gpu_memory_utilization=0.9, enforce_eager=True)
  out = llm.generate(['Hello'], SamplingParams(max_tokens=8, temperature=0))
  print(out[0].outputs[0].text)
  "
  ```

  If FlashInfer CUTLASS attempts to run and crashes, add explicit Marlin forcing to the wrapper and retry.
  Record which MoE backend vLLM actually selected (check stderr).
  Record whether it produces coherent text.
  Key files: `proj-2026-04-03-2113/run-vllm.sh`

- [x] **4. Update benchmark scripts for SM120 vLLM reality**
  Update `proj-2026-04-03-0318/bench_vllm_nano.py`:
  - Remove hard-coded `moe_backend="flashinfer_cutlass"` from `LLM()` constructor (line 626)
  - Remove `flashinfer_cutlass` from CLI description text (line 56)
  - Remove `requested_moe_backend: "flashinfer_cutlass"` from output JSON (line 701)
  - Let the actual selected backend be recorded from worker inspection
  - Add `enforce_eager=True` default
  - Make `--moe-backend` a CLI argument so the user can override (default: let vLLM auto-select or force marlin)

  Update `proj-2026-04-03-0318/run_bench_vllm.sh`:
  - Use `vllm-env/bin/python` instead of PYTHONPATH hacking
  - Remove `VLLM_USE_FLASHINFER_MOE_FP4=1`
  - Remove `VLLM_FLASHINFER_MOE_BACKEND=throughput`
  - Remove `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`
  - Remove `VLLM_MOE_PADDING=0`
  - Keep only necessary env vars

  Key files: `proj-2026-04-03-0318/bench_vllm_nano.py`, `proj-2026-04-03-0318/run_bench_vllm.sh`

- [ ] **5. Run vLLM benchmark and capture external baseline**
  Run the updated benchmark:
  ```bash
  proj-2026-04-03-0318/run_bench_vllm.sh
  ```

  Capture TTFT for decode (1 token) and prefill (32, 64, 128 tokens). Record the selected MoE backend (expected: Marlin). Compare against our unified fused backend numbers (67ms decode from proj-2026-04-03-1856).

  Update `proj-2026-04-03-1856/RESULTS.md` with the vLLM comparison data and SM120 context.
  Key files: `proj-2026-04-03-0318/run_bench_vllm.sh`, `proj-2026-04-03-1856/RESULTS.md`

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Build vLLM from source with uv | done | — | vllm==0.19.0+cu130 with torch==2.10.0+cu130 |
| 2 | Create wrapper scripts | done | — | run-vllm.sh and vllm-serve.sh |
| 3 | Validate vLLM on RTX 5090 | pending | — | |
| 4 | Update benchmark scripts for SM120 | done | — | Removed flashinfer_cutlass hard-coding, added --moe-backend CLI arg |
| 5 | Run vLLM benchmark | pending | — | |
