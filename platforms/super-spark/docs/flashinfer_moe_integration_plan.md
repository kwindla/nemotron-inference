# FlashInfer Fused MoE Integration Plan

Replace our per-expert cublasLt dispatch with FlashInfer's CUTLASS fused routed NVFP4 MoE path. The first integration target should be `cutlass_fused_moe`, not the older TRT split routed runner, because our runtime already computes grouped top-k routing and can hand FlashInfer precomputed top-k ids + weights directly. This is the production-aligned fused-kernel family the community uses to achieve 14 tok/s on DGX Spark.

## Why This Instead of More cublasLt Work

The cublasLt NVFP4 pointer-array batched GEMM does not work on SM121 / CUDA 13.2. We verified this empirically — `grouped_routed_expert_fastpath_uses = 0` across all runs. The incremental optimization path (device lookup tables, repair downloads, global disable) was working around a backend that doesn't exist on this hardware. FlashInfer's kernel is a completely different code path — CUTLASS-based, compiled for SM120/SM121, and validated by the community.

The repo-local TRT split FlashInfer experiment also turned out to be the wrong end-state for Spark:

- it instantiated on real routed expert layers
- it required a second shuffled resident weight layout for routed experts
- it drove startup to roughly `87s`
- it later failed with `NVRM ... NV_ERR_NO_MEMORY` / `Xid MMU Fault`

So the remaining viable FlashInfer direction is the CUTLASS fused backend that can consume the original packed NVFP4 expert tensors plus raw block scales instead of a duplicated TRT-style shuffled residency surface.

## What the Kernel Does

`flashinfer.fused_moe.cutlass_fused_moe()` executes the routed expert half of one MoE layer in one fused backend call:

1. Consume precomputed top-k expert ids and weights
2. Input activation quantization for NVFP4 execution
3. GEMM1: `hidden_states @ w1` (NVFP4, all selected experts)
4. Activation (ReLU² for Nemotron)
5. GEMM2: `intermediate @ w2` (NVFP4, all selected experts)
6. Expert output weighting and reduction
7. Output rescaling

This is a better first target for our runtime than the full routing entrypoint because we do not need to move router GEMM or grouped top-k selection into FlashInfer on day one.

## Weight Layout Requirements

FlashInfer's CUTLASS fused NVFP4 path can consume the original packed expert tensors directly when `use_packed_weights=True`. The important contract is:

- packed routed expert weights remain in their original packed uint8 NVFP4 storage
- block scales stay in raw linear layout
- per-expert global dequant scales are supplied separately
- precomputed top-k ids and weights are still passed in directly

That is materially different from the rejected TRT split path, which wanted a second shuffled `MajorK` resident weight layout with `128x4` interleaved scale tensors.

**Per MoE layer, two weight tensors:**
- `w1` (up_proj): `[num_experts, intermediate_size, hidden_size]` — packed int8 (NVFP4)
- `w2` (down_proj): `[num_experts, hidden_size, intermediate_size]` — packed int8 (NVFP4)

**Per MoE layer, two scale tensors:**
- `w1_scale`: `[num_experts, intermediate_size/block_size, hidden_size/block_size]` — float8_e4m3fn
- `w2_scale`: `[num_experts, hidden_size/block_size, intermediate_size/block_size]` — float8_e4m3fn

Important source-level finding:

- the CUTLASS binding uses `cutlass_fused_moe(...)`
- for NVFP4, the quant-scale bundle is:
  - gemm1 activation global scale
  - gemm1 weight block scales
  - gemm1 dequant scale
  - gemm2 activation global scale
  - gemm2 weight block scales
  - gemm2 dequant scale
- with BF16 input and on-the-fly quantization, the activation global scales can start at `1.0`
- the dequant scale for a routed expert should initially be derived from the checkpoint tensor scale as `1.0 / tensor_scale`

So the first runtime prep surface should be:

- raw packed routed weights
- raw linear block scales
- per-expert tensor-scale and dequant-scale arrays

That matches the CUTLASS fused backend without forcing a second resident routed-weight copy.

## Additional Inputs

Beyond weights, the fused backend takes:

- `hidden_states`: input activations (bf16 or fp16)
- `topk_ids`: precomputed expert ids
- `topk_weights`: precomputed expert weights
- `quant_scales`: the 6 NVFP4 scale tensors described above
- `num_experts`, `top_k`, `intermediate_size`: config
- `activation_type`: ReLU² (non-gated GEMM1 in FlashInfer CUTLASS terms)
- optionally `input_sf` if we later choose prequantized FP4 input rather than BF16 input with on-the-fly quantization

## Implementation Steps

### Step 1: Build FlashInfer for SM121 CUTLASS fused MoE

Clone FlashInfer, checkout the version matching the community (0.6.7+), apply the GDC fix (PR #2913 if not already merged), and build the C++ library with SM121 support.

The key build outputs we need:
- the CUTLASS fused-MoE sources and headers
- generated SM120 grouped GEMM instantiations
- a local adapter library that exposes our routed-MoE plugin ABI but calls the CUTLASS backend

Build configuration: `TORCH_CUDA_ARCH_LIST=12.1a`, CUDA 13.2, CUTLASS 4.4.2+.

### Step 2: Runtime seam conversion to the CUTLASS contract

During `ExpertLayerSlice::Create()`, stop preparing TRT-shuffled routed expert views. Instead expose the original routed expert descriptors to the FlashInfer backend as:

- raw packed weight pointer + byte size
- raw block-scale pointer + byte size
- tensor scale
- dequant scale
- original logical shape

Repo-local progress on the old TRT step remains useful only as reference:

- the runtime now has deterministic local prep utilities for:
  - `shuffle_matrix_a`
  - `shuffle_matrix_sf_a`
  - `128x4` block-scale interleave
- the routed backend seam now supports two explicit surfaces before plugin creation:
  - `legacy_trt_prepared`
  - `cutlass_raw`
- `cutlass_raw` now hands the plugin seam the original packed NVFP4 bytes plus raw block scales and dequant-scale metadata
- this is validated by the new `flashinfer_layout_test`

Those utilities are no longer the primary serving target. They remain useful for comparison and for the rejected TRT path, but the CUTLASS adapter should consume the raw packed surface instead.

### Step 3: Replace expert dispatch with CUTLASS fused MoE

Replace the body of `ExpertLayerSlice::Run()` (for the default serving path) with a single call to FlashInfer's routed fused MoE kernel. The call signature maps directly to our existing config:

| Our config | FlashInfer parameter |
|---|---|
| `config.n_routed_experts` | `num_experts` |
| `config.top_k` | `top_k` |
| `config.n_group` | `n_group` |
| `config.topk_group` | `topk_group` |
| `config.routed_scaling_factor` | `routed_scaling_factor` |
| `config.routed_expert_intermediate_size` | `intermediate_size` |
| selected expert ids | `topk_ids` |
| selected expert weights | `topk_weights` |
The shared expert path stays separate (it's not part of the fused routed kernel).

Keep the existing custom fused routed-NVFP4 path as a fallback for oracle tests and debugging (gated by env var or trace != nullptr).

### Step 4: Validate

- Run all existing expert oracle tests — the fused kernel should produce numerically close results (not bitwise identical due to different GEMM internals, but functionally equivalent)
- Run the full decode oracle — top-k token agreement should hold
- Run the 16-token generation bench — measure steady-state tok/s
- Compare memory telemetry to ensure no UMA pressure regression

### Step 5: CUDA graph capture

With the fused MoE kernel, all per-token MoE logic runs in one device-side kernel. No host involvement, no D→H copies. This makes the forward pass CUDA-graph-capturable.

Add graph capture after the first warmup token (when all experts are resident and the kernel has been called once).

## Expected Result

| Metric | Before | After fused MoE | After + CUDA graph |
|---|---:|---:|---:|
| Expert time (40 layers) | 344ms | ~40ms | ~40ms |
| Inter-layer overhead | 217ms | ~200ms | ~10ms |
| Total per token | ~937ms | ~630ms | ~300ms |
| tok/s | 1.1 | 1.6 | 3.3 |

Adding fused Mamba kernels and attention optimization on top would push toward 7-8 tok/s. MTP roughly doubles that to ~14 tok/s.

## Dependencies

- FlashInfer 0.6.7+ built for SM121 with CUDA 13.2
- CUTLASS 4.4.2+ (bundled with FlashInfer)
- The GDC fix (FlashInfer PR #2913) to prevent `cudaErrorIllegalInstruction` under load

## Risk

The main risk is that FlashInfer's SM121 support may have edge cases or shared-memory constraints specific to GB10 (99 KiB vs 228 KiB on B200). The community has validated the CUTLASS fused path on DGX Spark, but our exact expert shapes may still require tactic or workspace adjustments. The current custom fused routed path provides safety while the CUTLASS backend lands.
