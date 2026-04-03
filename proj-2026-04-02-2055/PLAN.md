# Plan: GPU Mamba2 Batched Prefill

Project directory: `./proj-2026-04-02-2055`

## Context
Prefill is stuck at 35ms/token sequential (1,175ms for 32-token tail) because enabling fused decode kernels forces model-wide token-by-token replay. vLLM, SGLang, and TensorRT-LLM all solve this with separate GPU prefill algorithms: batched causal conv1d + chunked SSD scan, then single-token decode kernels. We need the same split. Target: 32-token tail prefill drops from ~1,175ms to ~50ms, making hot-prefix TTFT ~100ms.

## Reference Implementation

vLLM's Mamba2 prefill (commit `32e0c0b`) is the primary reference:
- **Prefill conv:** `causal_conv1d_fn()` — processes entire token span through 1D causal conv on GPU, writes final state to conv cache ([mamba_mixer2.py:682](https://github.com/vllm-project/vllm/blob/32e0c0b/vllm/model_executor/layers/mamba/mamba_mixer2.py#L682))
- **Prefill SSM:** `mamba_chunk_scan_combined_varlen()` — 5-stage chunked SSD: chunk cumsum, chunk state, inter-chunk state passing, chunk BMM, chunk scan+output ([ssd_combined.py:90-149](https://github.com/vllm-project/vllm/blob/32e0c0b/vllm/model_executor/layers/mamba/ops/ssd_combined.py#L90-L149))
- **Decode conv:** `causal_conv1d_update()` — single-token shift-and-update
- **Decode SSM:** `selective_state_update()` — single-token recurrence

Our implementation follows this same split but in C++/CUDA instead of Python/Triton.

**Scope note:** vLLM's prefill kernels support varlen continuous batching, per-chunk seq_idx tracking, and APC-style intermediate cache block writes ([mamba_mixer2.py:664](https://github.com/vllm-project/vllm/blob/32e0c0b/vllm/model_executor/layers/mamba/mamba_mixer2.py#L664)). Our runtime is single-sequence, so we omit `seq_idx`, `cu_seqlens`, and intermediate cache-block writes. We only need single-sequence final-state snapshots.

## Our Current State

- **Fused decode kernels** (`fused_mamba_decode.cu`): `UpdateMambaConvStateKernel` + `FusedMambaDecodeHeadKernel` + `FusedMambaDecodeGroupNormKernel`. These work correctly at 35ms/token. **Keep unchanged.**
- **Batched path** (`mamba_layer.cpp:743-876`): Copies to host, loops per-token on CPU with `std::memmove` and scalar SSM recurrence. **Replace with GPU kernels.**
- **Model-wide gate** (`single_token_forward_model.cpp:52-58`): `DecodeConsistentPrefillEnabled()` serializes ALL layers including stateless MoE. **Remove.**
- **Expert/MoE layers**: Stateless — no inter-token dependency. Fused MoE decode gates on `token_count == 1` at both `expert_layer.cpp:1860` and `expert_layer.cpp:2032`. Fallback paths allocate `token_count`-wide tensors without cross-token state (`expert_layer.cpp:1828`, `:2263`, `:2561`). **Already correct for batched prefill.**

## Nano Mamba2 Dimensions (per layer)

```
intermediate_size  = 4096  (= num_heads * head_dim = 64 * 64)
conv_dim           = 6144  (= intermediate_size + 2 * n_groups * state_size = 4096 + 2*8*128)
conv_kernel_size   = 4
num_heads          = 64
head_dim           = 64
state_size         = 128   (d_state)
n_groups           = 8
projection_size    = intermediate_size + conv_dim + num_heads = 4096 + 6144 + 64 = 10304
```

Projection layout (split at `mamba_layer.cpp:769` host path, `:639` fused-compare path):
`[gate:4096 | conv_input:6144 | dt_pre:64]`
Conv input further splits into: `[x:4096 | B_groups:1024 | C_groups:1024]`

## Mamba2 SSM Math

These transforms must be applied identically in both prefill and decode:

- `A = -exp(A_log[head])` — negative exponentiated log-A, per head (`mamba_layer.cpp:685`, `fused_mamba_decode.cu:228`)
- `dt = max(softplus(dt_pre + dt_bias[head]), time_step_min)` — softplus + clamp (`mamba_layer.cpp:685-688`, `fused_mamba_decode.cu:228-232`). **Local divergence from vLLM:** vLLM passes `dt_limit=(0.0, inf)` and applies softplus with no explicit floor clamp (`mamba_mixer2.py:737`, `ssd_chunk_state.py:99`). Our `time_step_min` clamp (typically 1e-3) is an intentional stability guard inherited from the model config; the prefill kernel must preserve it for decode compatibility.
- `decay = exp(dt * A)` — per-token decay factor
- SSM recurrence: `state_next = state * decay + dt * B * x`, output `y += state_next * C`
- D skip: `y += x * D[head]` — applied before gating/norm (`mamba_layer.cpp:700`, `fused_mamba_decode.cu:260`; vLLM includes D inside chunk scan at `ssd_chunk_scan.py:380`)
- z gate: `y_gated = y * SiLU(z)` where z is the gate slice of the projection — applied before group RMS norm (`mamba_layer.cpp:831`)
- Group RMS norm: per-group over `intermediate_size / n_groups` = 512 elements, with per-element weight (`mamba_layer.cpp:831-845`, `fused_mamba_decode.cu:265-312`)

## Conv State Layout

Our layout: `[conv_dim][conv_kernel_size]` = `[6144][4]`, stored as contiguous FP32 in `request_context.mamba_conv_state()` at layer-specific offset. The decode kernel's `ShiftConvState` (`fused_mamba_decode.cu:42-60`) shifts then writes `state[K-1] = new_value` — a full-width oldest-to-newest history.

vLLM layout: `[dim][width-1]` = `[dim][3]` — stores only `kernel-1` history taps ([mamba_mixer2.py:579](https://github.com/vllm-project/vllm/blob/32e0c0b/vllm/model_executor/layers/mamba/mamba_mixer2.py#L579)).

**Divergence rationale:** We preserve our full-width `[dim][4]` layout to match the existing decode kernel contract. The prefill kernel must write final state as repeated `ShiftConvState` from the initial state, not just "copy the last 4 inputs."

## SSM State Layout

Our layout: `[num_heads * head_dim][state_size]` = `[4096][128]`, indexed as `hidden_index * state_size` (`mamba_layer.cpp:694`).

vLLM layout: `(nheads, headdim, dstate)` — contiguous with `nheads*headdim = 4096`.

**No divergence.** The memory layout is identical.

## Steps

- [x] **1. Split the model-wide decode-consistent gate**
  Remove `DecodeConsistentPrefillEnabled()` and the model-wide token-by-token replay loop at `single_token_forward_model.cpp:1242-1287`. Replace with per-layer dispatch: attention and expert layers always use batched prefill; only `MambaLayer::Run()` (starts at line 479) branches on token count. The fused decode env vars (`NEMOTRON_FORWARD_FUSED_MAMBA_DECODE`, `NEMOTRON_FORWARD_FUSED_MOE_DECODE`) should only affect the `token_count == 1` path inside each layer, not the model-wide dispatch. Expert layers already gate fused decode on `token_count == 1` (`expert_layer.cpp:1860`, `:2032`), so they need no change. After this step, multi-token prefill should work with batched attention + batched expert + the existing CPU-fallback Mamba path. Clean up `NEMOTRON_FORWARD_DECODE_CONSISTENT_PREFILL` references in tests (e.g., `nano_16_token_correctness_test.cpp:381`). Add a correctness test: run a 16-token prefill + 3 decode steps with fused decode enabled, compare output tokens against the sequential baseline. Extend or reuse `mamba_layer_oracle_test.cpp:455` for the Mamba-specific parity check.
  Key files: `runtime/src/api/single_token_forward_model.cpp`, `testing/api/`, `testing/backend/mamba_layer_oracle_test.cpp`

- [x] **2. GPU causal conv1d prefill kernel**
  Add a CUDA kernel `RunMambaConvPrefill()` that replaces the CPU conv loop in `mamba_layer.cpp:774-800`. Following vLLM's `causal_conv1d_fn` pattern: process all N tokens through the 1D depthwise causal convolution on GPU, applying SiLU activation. The kernel produces two outputs: (a) a device `[T, conv_dim]` tensor of conv results for the SSD step, and (b) updated conv state in `request_context.mamba_conv_state()` in our `[conv_dim][conv_kernel_size]` layout. The conv state update is equivalent to repeated `ShiftConvState` from the initial in-context state — for each channel, the final state contains the last `conv_kernel_size` input values in oldest-to-newest order. Accept optional initial conv state for cache-restore scenarios. For Nano dimensions (conv_dim=6144, kernel_size=4, token_count=32-4096), each channel's convolution is independent — a dot product of `conv_kernel_size` taps with the weight vector, slid causally over the token sequence. Add parity test: compare conv outputs and final conv state against the existing CPU reference path.
  Key files: `runtime/src/backend/mamba_conv_prefill.cu` (new), `runtime/include/nemotron/mamba_conv_prefill.h` (new), `runtime/src/backend/mamba_layer.cpp`, `testing/backend/`

- [x] **3. GPU SSD (Mamba2) chunked prefill kernel**
  Add GPU kernels implementing the chunked SSD scan for multi-token prefill, following vLLM's `mamba_chunk_scan_combined` algorithm (`ssd_combined.py:90-149`). The 5-stage algorithm: (1) `_chunk_cumsum_fwd`: compute chunk-local cumulative sum of `dt*A` (`dA_cumsum`), applying softplus + clamp to dt (`ssd_chunk_state.py:91-99`); (2) `_chunk_state_fwd`: build chunk-end state summaries weighted by `exp(dA_last - dA_t) * dt_t * B * x` — not a plain `B*x` product (`ssd_chunk_state.py:109`); (3) `_state_passing_fwd`: inter-chunk sequential carry of SSM state using chunk-boundary decay factors (trivial for single-chunk short sequences, but always runs); (4) `_bmm_chunk_fwd`: build within-chunk `C * B^T` causal block matrix for intra-chunk token interactions (`ssd_bmm.py:149`; always runs, even for single chunk — `ssd_combined.py:123`); (5) `_chunk_scan_fwd`: per-token outputs combining the `CB` block with inter-chunk state, applying causal masking within chunks, incorporating D skip and initial states. Inputs: `x[T, num_heads, head_dim]`, `dt[T, num_heads]`, `B[T, n_groups, state_size]`, `C[T, n_groups, state_size]`. B/C are consumed in grouped form (not expanded to heads) — the kernel handles group-to-head mapping internally, matching vLLM's convention (`ssd_combined.py:49`). Start with a tunable chunk_size; `256` is a reasonable starting point (vLLM's default is `2048` per `model.py:1331`, but smaller chunks trade compute for lower latency on short sequences). Write final SSM state to `request_context.mamba_state()` in our `[4096][128]` layout. Accept optional initial SSM state for cache-restore. Output: raw pre-gate/pre-norm `y[T, intermediate_size]`. Add parity test against the existing CPU reference.
  Key files: `runtime/src/backend/mamba_ssd_prefill.cu` (new), `runtime/include/nemotron/mamba_ssd_prefill.h` (new), `testing/backend/`

- [ ] **4. Wire GPU prefill into MambaLayer::Run and add end-to-end tests**
  Modify `MambaLayer::Run()` (starts at `mamba_layer.cpp:479`; replace the host multi-token fallback at lines 743-876) so that `token_count > 1` dispatches to the new GPU conv prefill (step 2) + GPU SSD prefill (step 3). The `token_count == 1` path stays on the existing fused decode kernels unchanged. The overall flow for multi-token prefill becomes: input norm → in-projection GEMM (batched, existing) → GPU conv prefill (produces `[T, conv_dim]` outputs + final conv state) → GPU SSD prefill (produces `[T, intermediate_size]` outputs + final SSM state) → **batched gated group RMSNorm** (apply `y * SiLU(gate)`, then per-group RMS norm over `intermediate_size/n_groups` = 512 elements with per-element weight; current `FusedMambaDecodeGroupNormKernel` is single-row only per `fused_mamba_decode.cu:324` — a batched variant is required) → out-projection GEMM (batched, existing) → residual add. Add end-to-end correctness tests: (a) multi-turn conversation with global-root restore + tail prefill + decode, comparing cached vs uncached token parity; (b) split-prefill equivalence: prefill N tokens in one shot vs K + (N-K) with cache restore between. These tests validate that the GPU prefill path produces state compatible with the fused decode kernel.
  Key files: `runtime/src/backend/mamba_layer.cpp`, `runtime/src/backend/mamba_gated_group_norm.cu` (new), `testing/api/`, `testing/backend/`

- [ ] **5. Benchmark and tune**
  Re-run the TTFT benchmark (`benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`) with the GPU prefill path. The benchmark should no longer need to set `NEMOTRON_FORWARD_DECODE_CONSISTENT_PREFILL=1`. Measure cold prefill at 256/1K/4K tokens and cached tail prefill at 32 tokens. Profile with Nsight Compute to identify kernel bottlenecks. Tune chunk_size (64/128/256/2048) and thread-block geometry for RTX 5090 (SM 12.0, 128 SMs, 32 GB VRAM). Expected performance: 32-token tail ~40-60ms, 256-token cold ~70-140ms (2-4x single decode step), 4K cold ~1-3s. If 4K cold prefill is still dominated by sequential inter-chunk state passing, add a parallel scan across chunk summaries. Save benchmark results to the project directory.
  Key files: `benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench.cpp`, `proj-2026-04-02-2055/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Split model-wide decode-consistent gate | done | fabb259 | |
| 2 | GPU causal conv1d prefill kernel | done | 9585ad2 | |
| 3 | GPU SSD chunked prefill kernel | done | — | v1 simple per-hidden-thread scan, not chunked |
| 4 | Wire GPU prefill into MambaLayer + e2e tests | pending | — | |
| 5 | Benchmark and tune | pending | — | |
