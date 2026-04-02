# Plan: Final Push — 22.5ms to 20ms

Project directory: `./proj-2026-04-02-0714`

## Goal

Reduce steady-state decode from 22.5 ms/token to ≤20 ms/token by fixing two identified bottlenecks: the single-thread expert selection kernel (1.3ms/token) and the per-pack D2H tensor scale readback (0.6ms/token).

Current: 22.5 ms/token mean, 20.1ms min, 44.4 tok/sec, 1016x from original baseline.
Target: ≤20 ms/token mean, ≥50 tok/sec.

## vLLM Reference

- Expert selection: vLLM uses `torch.topk()` or fused Triton kernel, keeping results as device tensors. Never copies routing results to host. (`fused_moe.py`)
- FP4 tensor scale: vLLM's `nvfp4_quant_kernels.cu` keeps the scale on device — the kernel reads `SFScale[0]` directly. No host round-trip.
- Alpha computation: vLLM pre-folds `input_global_scale * weight_global_scale` into a single `alpha` value at weight-process time, avoiding per-GEMM computation. (`nvfp4_utils.py:231`)

## Steps

- [ ] **1. Fix expert selection kernel — `<<<1,1>>>` → `<<<1,128>>>`**
  The `DeviceExpertSelectionKernel` in `fused_moe_decode.cu` launches with grid=1, block=1. A single thread does sigmoid on 128 values + serial top-k insertion sort. This takes 0.054ms/call × 23 layers = 1.3ms/token.
  Fix: rewrite as `<<<1, 128>>>` (one thread per expert). Each thread computes sigmoid + correction bias for its expert. Then use warp-level parallel reduction for top-k selection. For Nano (n_group=1, topk_group=1), skip the group ranking entirely.
  Expected: 0.054ms → ~0.005ms per call, saving ~1.1ms/token.
  Keep the original as fallback behind `NEMOTRON_FORWARD_EXPERT_SELECT_LEGACY=1`.
  Key files: `runtime/src/backend/fused_moe_decode.cu`

- [ ] **2. Eliminate PackInto D2H tensor scale readback**
  `DeviceNvfp4Matrix::PackInto()` does a `cudaMemcpy(D2H, 4 bytes)` after computing the activation tensor scale on device. This happens 184 times per token (8 packs × 23 MoE layers). Each forces a pipeline stall.
  Fix: keep the tensor scale on device. Modify the NVFP4 GEMM runner to accept a device pointer for alpha instead of a host float. Use cuBLASLt's `CUBLASLT_POINTER_MODE_DEVICE` to pass the alpha as a device pointer. Compute `alpha = act_tensor_scale * weight_tensor_scale` on device via a tiny multiply kernel that writes to a pre-allocated device float.
  Alternative simpler fix: since we already cache the weight tensor scale on host, and the activation tensor scale is computed on device, compute alpha on device as `act_scale_device * weight_scale_host_constant` via a single-element device kernel, then pass the device alpha pointer to cuBLASLt.
  Expected: eliminate 184 D2H copies/token, saving ~0.6ms/token.
  Key files: `runtime/src/backend/device_nvfp4_matrix.cu`, `runtime/src/backend/nvfp4_gemm_runner.cpp`, `runtime/src/backend/expert_layer.cpp`

- [ ] **3. Benchmark and profile**
  Run steady-state benchmark. Target: ≤20ms/token mean.
  Save benchmark artifact and Nsight profile.
  Key files: benchmark scripts

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Fix expert selection <<<1,1>>> → <<<1,128>>> | pending | — | -1.1ms/token expected |
| 2 | Eliminate PackInto D2H tensor scale | pending | — | -0.6ms/token expected |
| 3 | Benchmark and profile | pending | — | target ≤20ms |
