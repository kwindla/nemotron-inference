# Plan: Eliminate Host Overhead — 28ms to 20ms

Project directory: `./proj-2026-04-02-0355`

## Goal

Reduce steady-state decode from 28.2 ms/token to ≤20 ms/token by eliminating the ~16.7ms of host overhead (kernel compute floor is ~11.5ms).

Current: 28.2 ms/token, 35.5 tok/sec. Kernel compute: ~11.5ms. Host overhead: ~16.7ms.
Target: ≤20 ms/token, ≥50 tok/sec.

## Root Cause (from Codex audit + Nsight profile, corrected per review)

The host overhead comes from four categories:

1. **D2H tensor scale reads + cuBLASLt descriptor rebuild**: `ReadDeviceFloat()` in `nvfp4_gemm_runner.cpp:208,210` does 2 D2H cudaMemcpy per NVFP4 GEMM (activation + weight tensor scale). Additionally, `RunNvfp4RowMajorFp32AccumToDevice()` rebuilds cuBLASLt matmul descriptors and runs the heuristic query per GEMM (lines 215-367). Per MoE layer: 14 GEMMs × (2 D2H reads + descriptor rebuild). 23 MoE layers + Mamba projections = significant host cost per token. Weight tensor scales are immutable after upload; activation tensor scales are constant within each pack call.

2. **NVFP4 activation pack alloc churn**: `PackDeviceRowMajorFp32ToNvfp4()` creates a new `DeviceNvfp4Matrix` per call (4 cudaMalloc in Create, 4 cudaFree on destruction). Per MoE layer: 8 packs (1 normalized + 6 post-relu2 + 1 shared post-relu2). The normalized pack is already reused across all up_proj GEMMs — the plan's claim of 6× duplicate packing was wrong. Total: 8 packs/MoE layer × 23 layers + Mamba/attention NVFP4 projections = ~200+ DeviceNvfp4Matrix alloc/free cycles per token.

3. **MoE router logits D2H**: the cuBLASLt MoE path copies router_logits to host for expert selection (`expert_layer.cpp:1800-1809`). This should be done on device.

4. **Remaining small overhead**: attention metadata H2D (5 copies × 6 attention layers), embedding token ID alloc/copy/free, per-GEMM cudaGetLastError.

## Steps

- [x] **1. Cache tensor scales and cuBLASLt GEMM plans for NVFP4**
  Goal: eliminate per-GEMM D2H reads and cuBLASLt descriptor/heuristic rebuild.
  Current: `RunNvfp4RowMajorFp32AccumToDevice()` does 2 `ReadDeviceFloat()` D2H copies + full cuBLASLt descriptor creation + heuristic query per call.
  Fix:
  - Cache weight tensor scales as host floats at weight upload time (they're immutable). Store in `DeviceNvfp4Weight` or `MonolithicNvfp4ExpertWeights`.
  - Cache activation tensor scales: `PackDeviceRowMajorFp32ToNvfp4` already computes and stores them on device. Read the value once after packing and cache it in the `DeviceNvfp4Matrix` as a host float.
  - Cache cuBLASLt matmul descriptors: for `token_count==1` decode, the M/N/K and layout are constant per weight. Build the cuBLASLt descriptor + heuristic once at Create() time or on first use, cache in `UploadedLinearOp::Impl` or `ExpertLayerSlice::Impl`, and reuse on subsequent calls. Only the alpha scalar changes per call.
  Accept when: zero `ReadDeviceFloat` D2H calls in the decode hot path; cuBLASLt descriptors built once.
  Key files: `nvfp4_gemm_runner.cpp`, `nvfp4_weight.cpp`, `device_nvfp4_matrix.cu`, `expert_layer.cpp`, `linear_op.cpp`

- [ ] **2. Pre-allocate NVFP4 activation pack buffers**
  Goal: eliminate per-call DeviceNvfp4Matrix allocation (4 cudaMalloc + 4 cudaFree per pack).
  Fix:
  - In `ExpertLayerSlice::Impl`: pre-allocate DeviceNvfp4Matrix buffers at Create() for the two activation shapes: normalized `[1, hidden_size]` and post-relu2 `[1, routed_expert_intermediate_size]` (plus shared `[1, shared_expert_intermediate_size]`). Pack into these buffers on each call.
  - In `UploadedLinearOp::Impl`: pre-allocate one DeviceNvfp4Matrix for the activation shape at Create() time. Reuse on each `token_count==1` Run().
  - `PackDeviceRowMajorFp32ToNvfp4` needs a variant that packs into an existing buffer instead of creating a new one. Add `PackInto(existing_matrix, input)`.
  Accept when: zero cudaMalloc/cudaFree in the NVFP4 pack path during decode.
  Key files: `device_nvfp4_matrix.cu`, `device_nvfp4_matrix.h`, `expert_layer.cpp`, `linear_op.cpp`

- [ ] **3. Move MoE expert selection to device**
  Goal: eliminate D2H copy of router_logits for host-side expert selection.
  Current: `expert_layer.cpp:1800-1809` copies router_logits to host, runs `SelectTopExperts()` on CPU, copies selected indices/weights back to device.
  Fix: implement device-side top-k selection (the fused kernel already does this on device — `SelectTopExpertsOneToken` in `fused_moe_decode.cu`). Call the device-side selection from the cuBLASLt path.
  Accept when: no D2H copy of router_logits in the cuBLASLt MoE path.
  Key files: `expert_layer.cpp`, `fused_moe_decode.cu`

- [ ] **4. Benchmark and profile**
  Run steady-state benchmark and Nsight profile.
  Target: ≤20ms/token.
  Key files: benchmark scripts

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Cache tensor scales + cuBLASLt plans | done | — | host tensor scales + cuBLASLt cache + FillZero removed; smoke PASS |
| 2 | Pre-allocate NVFP4 pack buffers | pending | — | ~200 alloc/free cycles → 0 |
| 3 | Device-side expert selection | pending | — | eliminate router_logits D2H |
| 4 | Benchmark and profile | pending | — | target ≤20ms |
