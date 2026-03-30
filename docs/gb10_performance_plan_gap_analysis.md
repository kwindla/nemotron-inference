# GB10 Performance Plan — Gap Analysis

Date: 2026-03-28

This document is a research-backed gap analysis of `gb10_performance_plan.md`. It identifies critical gaps, missing considerations, corrections to existing content, and recommended additions. All findings are sourced from exhaustive web research across NVIDIA documentation, CUDA release notes, arxiv papers, developer forums, GitHub issues, and published benchmarks.

---

## Critical Gaps

### 1. SM121 Is Not Datacenter Blackwell — Hard Architectural Constraint

The plan discusses GB10 as a "unified-memory platform" but never names the most consequential architectural difference: GB10 is **SM121** (consumer Blackwell), not SM100 (datacenter Blackwell). This changes everything about library and kernel selection.

What SM121 lacks compared to SM100:

- **No TMEM (Tensor Memory)**: The 256 KB per-SM scratchpad that datacenter Blackwell uses for tensor core operands is absent on GB10.
- **No `tcgen05` instructions**: Blackwell's new single-thread tensor core ISA is unavailable. GB10 uses Ampere-era `mma.sync` instructions, extended with FP4/FP6 support.
- **No WGMMA**: Warp-group matrix multiply is unavailable.
- **128 KB shared memory per SM** (vs 228 KB on B200): CUTLASS FP4 kernels dispatch to SM120 tile configurations (128x128x256B, 256x128x128B) that require >99 KiB of shared memory and **crash on GB10** (TensorRT-LLM issue #11368). The cuBLASLt FP4 backend (`nvfp4_gemm_cublaslt`) works correctly on GB10 at 99.6 TFLOPS.
- **24 MB GPU L2 cache, 16 MB shared SLC**: Smaller cache hierarchy than datacenter parts.
- **48 SMs, 6144 CUDA cores at up to 2.55 GHz**: Compute capability SM 12.1.

Broken software on SM121:

- FlashAttention-3 and FlashAttention-4 crash.
- FlashInfer crashes.
- FlashMLA crashes.
- Triton compiler treats SM121 as Ampere (no Blackwell optimizations).
- CUTLASS SM100 kernels (including MoE Example 92) use `tcgen05`/TMEM and do not work.

Impact on the plan:

The library choices in sections "Dense GEMMs", "Attention", and "MoE Expert Path" happen to be correct (cuBLASLt, cuDNN FE), but for a reason the plan does not state: **they are the only options that work on SM121**. This should be documented as a hard constraint, not just a preference. The MoE section suggesting CUTLASS as a long-term alternative needs qualification: CUTLASS SM100 grouped GEMM kernels are not portable to SM121 without significant work.

Sources:

- [TensorRT-LLM issue #11368: FP4 CUTLASS GEMM fails on GB10 SM121](https://github.com/NVIDIA/TensorRT-LLM/issues/11368)
- [vLLM issue #33416: NVFP4 MoE kernels fail on RTX Blackwell SM12.0](https://github.com/vllm-project/vllm/issues/33416)
- [FlashInfer issue #2723: SM120 NVFP4 MoE issues](https://github.com/flashinfer-ai/flashinfer/issues/2723)
- [Microbenchmarking NVIDIA's Blackwell Architecture (arxiv 2512.02189)](https://arxiv.org/html/2512.02189v1)
- [Is DGX Spark Actually Blackwell? (Backend.AI Analysis)](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)

---

### 2. Effective Bandwidth Is Significantly Below Peak

The plan uses 273 GB/s throughout. Real-world measurements show substantially lower effective bandwidth:

| Condition | Measured Bandwidth |
|---|---|
| Peak theoretical | 273 GB/s |
| Idle (no CPU load) | 229 GB/s |
| During inference | ~178 GB/s (65% of peak) |
| With co-located CPU workloads | ~87 GB/s |
| CPU-side bandwidth | Capped around 120 GB/s |

Hardware details: 128 GB LPDDR5x at 8533 MT/s across 16 channels with 256-bit interface. Hot Chips presentation showed 9400 MT/s / 301 GB/s capability on paper.

Impact:

Decode throughput estimates should use **~180 GB/s** as the realistic planning figure, not 273. For a 12B-active MoE model in NVFP4 (~6 GB active weights), the realistic decode ceiling is ~30 tok/s, not ~45. Published benchmarks of 45.34 tok/s for gpt-oss-120B (MXFP4) may reflect a smaller active footprint or benchmark conditions that achieved closer to peak bandwidth.

Any roofline analysis or throughput projections in the plan should include both theoretical and realistic bandwidth figures.

---

### 3. GPU OOM Causes System Zombie — Missing From "Known Constraints"

The plan mentions `cudaMemGetInfo` insufficiency but omits the most dangerous operational failure mode on DGX Spark.

The failure mode: When GPU memory is exhausted on DGX Spark, the system becomes **completely unresponsive**. SSH hangs, UI freezes, and a hard reboot is required. The root cause is that `nvidia-modeset` kernel thread enters an uninterruptible D-state for >120 seconds when the driver-level allocation fails under unified memory pressure. The Linux kernel reports large `MemAvailable` (via reclaimable page cache), but actual reclaim latency exceeds driver timeouts. Only ~1 GB may be truly free while ~100 GB is held in file cache.

Additional details:

- Docker `--memory` cgroup limits have **no effect** on CUDA unified memory allocations.
- `nvidia-smi` displays "Memory-Usage: Not Supported" on the iGPU (expected since there is no discrete framebuffer).
- There is **no API to set a residency boundary on GB10** to prevent CUDA from competing with OS services.
- This is an unresolved architectural issue as of March 2026.

Impact on the plan:

The memory budget planner (currently targeting 64k context / 8 requests) must treat this as a hard safety constraint. The startup residency byte budget (section 6 of "Required GB10 Performance Work") must include an explicit safety margin (~20 GB headroom). The exit criteria should include OOM stress-test validation.

Mitigations:

- Parse `/proc/meminfo` fields (`MemAvailable + SwapFree`) instead of `cudaMemGetInfo` for true allocatable memory.
- Community `nvml-unified-shim`: Drop-in replacement for `libnvidia-ml.so.1` that uses CUDA Runtime API + `/proc/meminfo`.
- Flush page cache before memory-intensive operations: `sync; echo 3 > /proc/sys/vm/drop_caches`.
- For vLLM baseline benchmarks: Use `--num-gpu-blocks-override <N>` to bypass the profiler's `cudaMemGetInfo` logic.

Sources:

- [DGX Spark Zombie/OOM Forum Thread](https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752)
- [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)
- [Unexpected Available Memory Reporting on DGX Spark](https://nvidia.custhelp.com/app/answers/detail/a_id/5728/~/unexpected-available-memory-reporting-on-dgx-spark)
- [NVML Unified Memory Shim (Community)](https://forums.developer.nvidia.com/t/nvml-support-for-dgx-spark-grace-blackwell-unified-memory-community-solution/358869)

---

### 4. cuBLAS 13.0 Has a Known Illegal Memory Access Bug on DGX Spark

The plan's toolchain pinning section (required work item 1) recommends CUDA 13.2+. This is correct but the urgency is understated.

The bug: cuBLAS versions 12.8 through 13.1 have a confirmed bug causing **illegal memory access in `cublasLtMatmul` with BF16 and FP16 inputs on DGX Spark**. This was fixed in cuBLAS 13.2 (CUDA 13.2, March 5, 2026).

Since the current preflight shows CUDA 13.0, the runtime is currently exposed to this bug. The existing dense FP32 path uses FP32 inputs (likely safe), but this bug is a **hard blocker** for any non-FP32 dense GEMM work. Adding BF16 operator paths on CUDA 13.0 would hit this immediately.

Additional cuBLAS 13.2 improvements relevant to the plan:

- **Up to 3x performance improvement for NVFP4 and MXFP8 on DGX Spark** for large M/N problem sizes.
- Improved GB200/B200 performance for MXFP8 and NVFP4 when M,N <= 32.
- Extended experimental Grouped GEMM API to support MXFP8 inputs on CC 10.x and 11.0.
- Fixed `cublasLtMatmul` producing incorrect results for NVFP4 on B300/GB300 when m not divisible by 64.

Driver requirement: Upgrading to CUDA 13.2 requires driver >= 595.45.04 (up from 580 for CUDA 13.0/13.1). The current DGX OS 7.4.0 ships driver 580.126.09.

Impact:

The CUDA 13.2 upgrade is the single highest-ROI action for NVFP4 contract validation (required work item 3). It should be treated as item 0, preceding all other measurement work. All subsequent performance data should be collected on the fixed toolchain.

Sources:

- [CUDA 13.2 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [CUDA 13.2 DGX Spark Impact Forum Thread](https://forums.developer.nvidia.com/t/cuda-13-2-dgx-spark-impact/363182)
- [cuBLAS 13.2 Documentation](https://docs.nvidia.com/cuda/cublas/)
- [DGX Spark Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)

---

### 5. NVFP4 Contract Details Are Now Available

The plan correctly identifies that the first NVFP4 probe failed with `CUBLAS_STATUS_NOT_SUPPORTED`, but the specific contract requirements are now documented and can guide the next probe attempt.

NVFP4 format specification:

- **Element format**: E2M1 (1 sign, 2 exponent, 1 mantissa = 4 bits). Representable values: {0, 0.5, 1, 1.5, 2, 3, 4, 5, 6} and negatives.
- **Block size**: 16 contiguous elements share one scale factor (not 32 like MXFP4/MXFP8).
- **Micro-block scale**: FP8 E4M3 unsigned (UE4M3). Supports fractional non-power-of-two scaling (e.g., 1.5x, 2.5x). MSE for scale factors: E4M3 achieves 0.08 vs MXFP4's E8M0 at 0.72.
- **Tensor-wide scale**: Single FP32 scalar per tensor.
- **Effective bits per value**: 4.5 (4 bits data + 8 bits / 16 elements for scale).
- **Packing**: Two E2M1 values packed per byte (`__nv_fp4x2_e2m1` / CUDA type `CUDA_R_4F_E2M1`).

cuBLASLt descriptor requirements:

- **Scale mode enum**: `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` for NVFP4 (vs `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` for MXFP8).
- **Scale buffer shape**: `(M, K//16)` for matrix A, `(N, K//16)` for matrix B.
- **Scale buffer rearrangement**: Required into interleaved blocked format. Shape must be `(32 * ceil_div(H, 128), 16 * ceil_div(W, 4))`.
- **Alignment**: Base addresses must be 32-byte aligned. Leading dimension must be a multiple of 128 elements.
- **Scale and data buffers must be device pointers** (GPU memory).
- cuBLAS can compute D-matrix scale factors automatically when output is FP4 or FP8.
- Reference implementation: `NVIDIA/CUDALibrarySamples` → `LtNvfp4Matmul`.

Known issues:

- `CUBLAS_STATUS_NOT_SUPPORTED` when `m=1` and bias is passed (PyTorch issue #157054).
- K < 32 was broken on CUDA 12.8, fixed in CUDA 12.9.1.
- GB10 shared memory overflow when CUTLASS dispatch routes to wrong tile configs (use cuBLASLt backend only).
- vLLM NVFP4 MoE backend only checks SM9.0 and SM10.x, **missing SM12.0/SM12.1** (vLLM #33416).

Comparison with MXFP8:

| Property | NVFP4 | MXFP8 |
|---|---|---|
| Element format | E2M1 (4 bits) | E4M3 or E5M2 (8 bits) |
| Block size | 16 elements | 32 elements |
| Scale factor format | UE4M3 (fractional) | UE8M0 (power-of-two only) |
| Second-level scale | FP32 per tensor | None |
| Effective bits/value | 4.5 | 8.25 |
| cuBLASLt scale mode | VEC16_UE4M3 | VEC32_UE8M0 |

Validation against current runtime code:

The `gemm_catalog.h` already has `kCublasLtNvfp4BlockScaled` with `vec16_ue4m3` scale mode, which aligns with the correct contract. The alignment checks in `cublaslt_gemm_plan.cpp` should be validated against the 32-byte base address and 128-element leading dimension requirements.

Sources:

- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference (NVIDIA Blog)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVFP4 Trains with Precision of 16-Bit and Speed and Efficiency of 4-Bit (NVIDIA Blog)](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/)
- [Boosting Matrix Multiplication Speed and Flexibility with NVIDIA cuBLAS 12.9 (NVIDIA Blog)](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/)
- [CUTLASS Tutorial: Hardware-supported Block-scaling with NVIDIA Blackwell GPUs (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-hardware-supported-block-scaling-with-nvidia-blackwell-gpus/)
- [CUTLASS Tutorial: Sub-byte GEMM on NVIDIA Blackwell GPUs (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
- [Low Precision Matrix Multiplication with cuBLASLt (DeepWiki)](https://deepwiki.com/NVIDIA/CUDALibrarySamples/2.1.1-low-precision-matrix-multiplication-with-cublaslt)
- [PyTorch issue #157054: NVFp4 cuBLAS Error](https://github.com/pytorch/pytorch/issues/157054)
- [Transformer Engine FP8 Primer with NVFP4 and MXFP8 coverage](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

---

## Missing Considerations

### 6. Speculative Decoding — Not Mentioned Anywhere in the Plan

NVIDIA's own DGX Spark benchmarks show **2.6x speedup** combining NVFP4 with speculative decoding on Qwen-235B across dual DGX Spark systems. On a bandwidth-limited platform where decode runs at <1% compute utilization, speculative decoding is the highest-leverage technique after weight quantization. It trades idle compute (which GB10 has in abundance during decode) for reduced memory bandwidth passes.

How it works on bandwidth-limited systems:

Standard decode reads all active weights once per token. Speculative decoding uses a small draft model to predict N tokens, then verifies all N in a single target model pass. If the draft produces 5 tokens with 80% acceptance rate, you generate ~3.8 tokens per weight-loading pass instead of 1, reducing bandwidth demand by 3.8x.

Measured speedups:

| Method | Configuration | Speedup |
|---|---|---|
| vLLM + EAGLE3 | Llama 3.3-70B, 4xA100, batch=1 | 2.3x |
| vLLM + external draft | Llama 3.1-70B + 1B draft, H100, batch=1 | 2.31x |
| SGLang + SpecForge | Llama 4 Maverick, MT-Bench | 2.18x |
| NVFP4 + spec decode | Qwen-235B, dual DGX Spark | 2.6x over FP8 alone |

Acceptance rate economics:

Expected tokens per verification pass: `tau = (1 - alpha^(gamma+1)) / (1 - alpha)` where alpha is acceptance rate and gamma is speculation length.

| Acceptance Rate | Speculation Length=5 | Speculation Length=8 |
|---|---|---|
| 0.6 | 2.4x | -- |
| 0.7 | 2.9x | -- |
| 0.8 | 3.8x | -- |
| 0.9 | -- | 6.1x |

Below alpha=0.5, speculative decoding becomes counterproductive.

Draft model strategies for the hybrid Mamba-Transformer architecture:

1. **EAGLE-3 (recommended)**: Trains an auxiliary head reusing target model hidden states. Achieves 0.75-0.85 acceptance rates with minimal memory overhead (hundreds of millions of parameters).
2. **Self-speculative / LayerSkip**: Uses early layers of the target model as draft. Single model, no extra memory, but lower acceptance rates.
3. **External draft model**: Pair the 120B model with a corresponding smaller model. Adds memory overhead.
4. **N-gram / prompt lookup**: Zero memory overhead, works well for repetitive outputs (code, structured data).

When NOT to use on DGX Spark:

- Batch sizes above ~8-10 (diminishing returns as compute becomes the constraint).
- Very short outputs (<50 tokens, overhead exceeds gains).
- Highly creative/unpredictable outputs (low acceptance rate).

Impact on the plan:

This should be added as hypothesis H5 and included in the performance work items. Projected combined decode throughput: starting from ~22 tok/s (FP8 baseline), NVFP4 brings it to ~40-45 tok/s, speculative decoding to ~80-110 tok/s.

Sources:

- [New Software and Model Optimizations Supercharge NVIDIA DGX Spark (NVIDIA Blog)](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/)
- [Speculative Decoding 2-3x Faster (PremAI, 2026)](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/)
- [Boosting Local Inference with Speculative Decoding (OpenInfer)](https://openinfer.io/news/2025-08-05-boosting-local-inference-with-speculative-decoding/)
- [MagicDec (ICLR 2025)](https://arxiv.org/pdf/2408.11049)
- [Decoding Speculative Decoding (NAACL 2025)](https://aclanthology.org/2025.naacl-long.328.pdf)

---

### 7. Operator Fusion — Critical for Bandwidth-Limited Decode

On a 273 GB/s system, every unfused kernel round-trips through global memory. The plan's "operator-selection decisions" framing focuses on which library to call, but does not consider fusion as a dimension.

Key fusions with measured impact:

**Fused residual + RMSNorm (highest ROI, lowest effort):**

Default PyTorch RMSNorm achieves only **11% of peak memory bandwidth**. A fused Triton kernel combining residual addition with RMSNorm achieves **88% of peak bandwidth**: an 8x improvement and 6x wall-clock speedup. This is the single lowest-effort highest-impact fusion.

**Fused SwiGLU MLP (highest absolute impact):**

DeepFusionKernel (Feb 2026, arxiv 2602.11808) fuses gate GEMM, up-projection GEMM, SiLU activation, element-wise gating, and down-projection into a single kernel. Eliminates intermediate buffer materialization. Results: up to 13.2% end-to-end speedup on H100, 9.7% on A100. Greatest gains at small batch sizes (B=1-4), where bandwidth dominates.

**Fused QKV + attention + output projection:**

ClusterFusion (arxiv 2508.18850) fuses QKV projection, scaled dot-product attention, and output projection into a single kernel using distributed shared memory. Results on H100: core module speedup 1.61-1.85x, end-to-end latency 1.34-1.51x improvement, 10x reduction in kernel launch overhead vs CUDA Graph baselines.

Framework support on SM121:

Since CUTLASS and FlashAttention are broken on SM121, the primary portable fusion mechanism is **cuDNN FE's runtime fusion engine**. cuDNN FE supports fused Grouped GEMM + GLU, SwiGLU, dSwiGLU, and Quant operations for FC1/FC2 MoE patterns (cuDNN 9.18.0+). This should be part of the library direction section.

Sources:

- [Deep Kernel Fusion for Transformers (arxiv 2602.11808)](https://arxiv.org/html/2602.11808)
- [ClusterFusion (arxiv 2508.18850)](https://arxiv.org/abs/2508.18850)
- [From 11% to 88% Peak Bandwidth: Fused RMSNorm (Subhadip Mitra)](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [LLM Inference Acceleration via Operation Fusion (arxiv 2502.17728)](https://arxiv.org/html/2502.17728)

---

### 8. Mamba State Batch-Scaling Behavior

The plan identifies Mamba state bandwidth as a decode constraint but does not analyze how it scales with batch size. This is critical because Mamba state is **per-request** while weights are shared across the batch.

Concrete Nemotron-3-Super state dimensions:

Per the model config.json:
- `hidden_size` = 4096, `expand` = 2, so `d_inner` = 8192
- `mamba_head_dim` = 64, `mamba_num_heads` = 128
- `ssm_state_size` (d_state) = 128
- `n_groups` = 8, `conv_kernel` = 4

SSM state per layer per request: 128 heads x 64 headdim x 128 d_state = **1,048,576 elements**. At FP32: **4 MiB per Mamba layer per request**. At FP16: **2 MiB per layer per request**.

Conv state per layer: (8192 + 2 x 8 x 128) x 4 = 10,240 x 4 = 40,960 elements = ~164 KB at FP32.

Total across ~44 Mamba layers:

| Precision | SSM state | Conv state | Total per request |
|---|---|---|---|
| FP32 | ~176 MiB | ~7 MiB | ~183 MiB |
| FP16 | ~88 MiB | ~3.5 MiB | ~91.5 MiB |

Batch-scaling analysis (bandwidth traffic per decode step):

Each decode step reads the full SSM state, computes the update, and writes it back. Read + write = 2x state size per step.

| Batch | Weight traffic (NVFP4, 12B active) | Mamba state traffic (FP16) | State as % of total |
|---|---|---|---|
| 1 | ~6 GB | ~0.17 GB | 2.8% |
| 4 | ~6 GB | ~0.68 GB | 10.2% |
| 8 | ~6 GB | ~1.36 GB | 18.5% |
| 16 | ~6 GB | ~2.72 GB | 31.2% |

At batch 16, nearly a third of bandwidth goes to Mamba state. This means Mamba cache format (FP32 vs FP16 vs INT8) becomes increasingly important as concurrency grows, even though it is a small fraction at batch 1.

The plan's hypothesis H3 should be extended to include batch-scaling analysis. The Mamba cache format benchmarks (required work item 5) should measure at batch=1, 4, 8, and 16.

Arithmetic intensity of SSM decode: Approximately **2.5 FLOPs per byte** for SISO Mamba-2, far below the compute-bound regime. This confirms SSM decode is **heavily memory-bandwidth-bound**.

Sources:

- [Nemotron-3-Super config.json (HuggingFace)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json)
- [Mamba-3 arxiv 2603.15569 (arithmetic intensity analysis)](https://arxiv.org/html/2603.15569v1)
- [Mamba-2 State Space Duality Part I (Goomba Lab)](https://goombalab.github.io/blog/2024/mamba2-part1-model/)

---

### 9. Quamba2: 8-Bit Mamba State Quantization

The plan considers FP32, FP16, and FP16+stochastic-rounding for Mamba cache. It should also consider **INT8 per-group quantization**.

Quamba2 (March 2025, arxiv 2503.22879): A post-training quantization framework that quantizes cached SSM states to 8-bit precision:

- Uses per-group scaling with m=4 head groups and n=4 channel groups.
- States are quantized using symmetric uniform quantization with standard rounding.
- W4A8 with 8-bit states: 7.43ms TPOT vs 22.73ms FP16 at batch=1 (**3x generation speedup**).
- At batch=128, FP16 runs OOM while W4A8 handles batch=256 successfully.
- Quality: Natural Questions 14.2 (W4A8) vs 17.2 (FP16), SquadV2 45.9 vs 51.9.
- Key finding: "cached SSM states are redundant" — they tolerate aggressive quantization.

Impact:

The Mamba cache-format benchmarks (required work item 5) should include INT8 as a candidate alongside FP32, FP16, and FP16+stochastic-rounding:

- FP32 cache / FP32 arithmetic (baseline)
- FP16 cache / FP32 arithmetic
- FP16 cache with stochastic rounding on store (NVIDIA's selected recipe)
- **INT8 cache with per-group scaling (Quamba2 approach)**

Source:

- [Quamba2: Robust and Scalable Post-training Quantization for SSMs (arxiv 2503.22879)](https://arxiv.org/html/2503.22879)

---

### 10. KV Cache Dtype for Attention Layers

The plan's numeric-format section does not mention KV cache quantization for the 8 attention layers. The Nemotron-3-Super deployment recipe uses `kv_cache_dtype: fp8`.

Since KV cache reads scale linearly with context length, FP8 KV halves attention bandwidth at long contexts. cuDNN FE paged attention supports FP8 inputs for Q and KV containers (cuDNN 9.11.0+).

However: **FP8 SDPA is not available on SM120/SM121** (only on SM100 datacenter Blackwell). BF16 is the correct target for paged attention on GB10.

KV cache budget with BF16 (Nemotron-3-Super, 8 attention layers, 2 KV heads, head_dim=128):

- Per token per layer: 2 KV heads x 128 dim x 2 (K+V) x 2 bytes (BF16) = 1024 bytes
- Per token across 8 attention layers: 8192 bytes = 8 KiB
- At 64k context: ~512 MiB per request
- At 8 concurrent requests, 64k context: ~4 GiB total KV cache

This should be added to the numeric-format planning section and the memory budget.

Sources:

- [cuDNN Backend Release Notes (v9.11.0+)](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html)
- [Nemotron-3-Super Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html)

---

### 11. Prefix Caching for Mamba Layers Has Fundamental Limitations

The plan's cache architecture (`v1_cache_architecture.md`) stores "full Mamba recurrent snapshot" at prefix cache nodes. NVIDIA's deployment guide explicitly states: `enable_block_reuse: false` for Mamba layers — "Mamba recurrent state is not prefix-cacheable."

The fundamental challenge: Mamba state is updated in-place via sequential recurrence. Unlike KV cache (which appends independent key/value vectors), SSM state at position t is a lossy compression of all tokens 0..t. You cannot split, truncate, or partially restore it at arbitrary token boundaries.

The correct approach (implemented by SGLang's MambaRadixCache):

- Maintain a **radix tree of immutable SSM state checkpoints** indexed by prefix tokens.
- New requests **fork from cached prefixes**: they receive a copy of the checkpoint state and diverge from there.
- No rollback needed: states branch from checkpoints rather than unwinding.
- Each speculative decode draft token receives a **private cache slot** with its own SSM state copy.
- When draft tokens are accepted, the last accepted slot becomes the new main state.

vLLM V1 approach:

- Automatic Prefix Caching for Mamba-2 is implemented (PR #25752).
- Uses selective state checkpointing at prefix boundaries.
- State reuse semantics maintain correct SSM state evolution.

Key operational implications:

1. **Prompt caching is expensive for SSM layers**: Sharing a prefix requires either (a) storing full SSM state checkpoints at prefix boundaries (4 MiB per Mamba layer per checkpoint at FP32 = ~176 MiB per checkpoint across 44 layers), or (b) replaying the prefix tokens through the Mamba layers.
2. **State copying overhead**: "Copying Mamba-2 states takes at least 10% of CPU time on some systems" due to state sizes (multiple MiB per layer).
3. **Prefill-decode disaggregation**: Mamba state must be serialized and transferred between nodes, adding latency proportional to state size.

This is architecturally different from attention KV prefix sharing (which shares physical page blocks via refcounting). The plan should distinguish these two mechanisms and budget for the copy overhead.

Sources:

- [Nemotron-3-Super Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html)
- [Hybrid Models Meet SGLang (PyTorch Blog)](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- [Hybrid Models as First-Class Citizens in vLLM (PyTorch Blog)](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)
- [vLLM RFC: Native SSM support in V1 (GitHub #17140)](https://github.com/vllm-project/vllm/issues/17140)

---

### 12. Model Loading Performance

Not discussed in the performance plan, but `mmap` is known to be slow on DGX Spark. Measurements:

- Default mmap: 68 seconds for large models (kernel 6.14).
- After kernel 6.17 update: ~30 seconds.
- Eager loading (`--no-mmap`): Faster than mmap in all tested configurations.
- Recommendation from NVIDIA and community: Use `--no-mmap` and ensure model files reside on the fast 4 TB NVMe SSD.

Since the runtime already uses POSIX mmap in `artifact_loader.cpp`, the startup path should be benchmarked. For production use, eager loading with `posix_fadvise(POSIX_FADV_SEQUENTIAL)` may be faster than mmap on DGX Spark's unified memory system.

This directly affects TTFT for the first request after cold start.

Sources:

- [llama.cpp DGX Spark Discussion (GitHub)](https://github.com/ggml-org/llama.cpp/discussions/16578)
- [Very Slow mmap on DGX Spark (NVIDIA Forum)](https://forums.developer.nvidia.com/t/very-slow-mmap-on-dgx-spark-that-affects-model-loading-questions-to-nvidia/349886)

---

### 13. /proc/meminfo-Based Memory Budgeting

The plan says `cudaMemGetInfo` is insufficient but does not name the alternative.

NVIDIA's recommended approach: Parse `/proc/meminfo` fields for true allocatable memory. A reference C implementation from NVIDIA calculates `availableMemory` as:

```
availableMemory = MemAvailable (from /proc/meminfo)
maxAllocatableWithSwap = availableMemory + SwapFree
```

Practical guidance:

- Always allocate headroom (~20 GB) to prevent catastrophic swap scenarios.
- Use `swapoff -a && swapon -a` post-capture to force swap reclamation.
- Monitor `free -h` (specifically Swap used column) during inference.
- For vLLM: Use `--num-gpu-blocks-override <N>` to directly specify KV cache block count.

The memory budget planner in `memory_budget.cpp` should integrate `/proc/meminfo` parsing as the primary budget signal.

Sources:

- [Unexpected Available Memory Reporting on DGX Spark (NVIDIA)](https://nvidia.custhelp.com/app/answers/detail/a_id/5728/~/unexpected-available-memory-reporting-on-dgx-spark)
- [DGX Spark Porting Guide: Optimization](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html)

---

## Corrections to Existing Plan Content

### 14. MoE Expert Path: CUTLASS Qualification Needed

Current plan text:

> "The most promising library foundation remains cuDNN Frontend MoE Grouped Matmul or CUTLASS, not cuBLASLt grouped GEMM as the long-term base."

Correction: CUTLASS Example 92 (Blackwell MoE) uses `tcgen05` and TMEM instructions that are absent on SM121. cuBLASLt 13.1+ now has an experimental Grouped GEMM API with CUDA Graph support (up to 4x MoE speedup), and cuBLASLt 13.2 extends this to MXFP8 on CC 10.x and 11.0.

Recommended revision: cuDNN FE as primary, cuBLASLt Grouped GEMM as secondary, CUTLASS deprioritized for SM121.

cuBLASLt 13.1 Grouped GEMM features:

- Device-side shapes via `cublasLtGroupedMatrixLayoutCreate` / `cublasLtGroupedMatrixLayoutInit`.
- CUDA Graph support for host-synchronization-free implementation.
- Supports FP8 and BF16/FP16 on Blackwell.
- Up to 4x speedup over multi-stream GEMM in MoE use cases.

cuDNN FE MoE features (9.18.0+):

- Three modes: None, Gather, Scatter.
- `FirstTokenOffset [B*E,1,1]` for CSR-style expert-to-token mapping.
- Fused Grouped GEMM + GLU, SwiGLU, dSwiGLU, and Quant for FC1/FC2 patterns.

Sources:

- [cuDNN Frontend MoE Grouped Matmul](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/MoeGroupedMatmul.html)
- [CUDA 13.1 Release Notes / cuBLAS](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
- [CUTLASS Example 92 (Blackwell MoE)](https://github.com/NVIDIA/cutlass/blob/main/examples/92_blackwell_moe_gemm/)

---

### 15. Attention Section: cuDNN FE Is the Only Option, Not Just Recommended

Current plan text:

> "cuDNN Frontend already exposes the right paged-cache model for attention on the serving path"

This undersells the constraint. On SM121, cuDNN SDPA is **the only working attention backend**:

- FlashAttention-3: Crashes on SM121.
- FlashAttention-4: Crashes on SM121 (designed for SM100 TMEM).
- FlashInfer: Crashes on SM121.
- FlashMLA: Crashes on SM121.
- Triton: JIT-compiles but without Blackwell optimizations (treats SM121 as Ampere).
- cuDNN FE SDPA: Works natively on SM121.

This is not a preference — it is a hard constraint that should be documented in the plan.

cuDNN FE paged attention contract (for reference):

- K/V containers: Block pools with shape `(num_blocks, H_kv, block_size, D)`.
- Page tables: INT32 tensors of shape `(B, 1, ceil(S_kv/bs), 1)` indexing into the container.
- Sequence-length tensors: `(B,1,1,1)` INT32, mandatory with paged attention.
- Prefill vs decode: Implicit via `seq_len_q` (>1 for prefill, =1 for decode).
- Block sizes: 32, 64, 128 (must be power of 2). On bandwidth-limited DGX Spark, block size 32 is likely optimal due to better L1 cache hit rates and less fragmentation.

Known limitations on SM121:

- FP8 attention is not available (SM100 only).
- Backward pass does not support paged attention (inference only, which is our use case).
- Runtime compilation of LayerNorm/RMSNorm may be slow on CC 12.0.
- Decode-phase SDPA issues reported with causal masking when Q head count != K/V head count.

Sources:

- [cuDNN Frontend Attention](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)
- [cuDNN 9.18.1 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html)
- [vLLM SM121 support issue (GitHub #36821)](https://github.com/vllm-project/vllm/issues/36821)

---

### 16. cuDNN FE Paged Attention: FP8 Not Available on SM121

The plan does not specify KV cache dtypes. As noted in gap 10, FP8 SDPA is unavailable on SM120/SM121. BF16 is the correct target for paged attention on GB10.

The Nemotron-3-Super deployment recipe specifies `kv_cache_dtype: fp8`, but this applies to datacenter Blackwell (SM100). On GB10, the KV cache must use BF16 or FP16.

---

## Recommended Additions to Required Performance Work

### 17. Add: Work Item 0 — Toolchain Upgrade to CUDA 13.2

This should be the first work item, preceding all other measurement work, because it:

- Fixes the BF16/FP16 illegal memory access bug on DGX Spark (existed since cuBLAS 12.8).
- Delivers up to 3x NVFP4/MXFP8 performance improvement on DGX Spark.
- Extends Grouped GEMM to MXFP8 on CC 10.x and 11.0.
- Unifies the CUDA toolkit for Tegra and desktop GPUs (fewer SM121-specific bugs).
- Extends CUDA Tile to SM120/SM121 architectures.

Requirements:

- Driver upgrade: 580 → 595.45.04 (note this is for x86_64; DGX Spark is aarch64 and may require a different build in the 595 family).
- DGX OS update may be required.
- All subsequent benchmarks and performance data should be collected on CUDA 13.2.

---

### 18. Add: Work Item 7 — Speculative Decoding Feasibility

Evaluate speculative decoding for the hybrid Mamba-Transformer architecture on GB10.

Tasks:

- Evaluate EAGLE-3 and self-speculative approaches.
- Measure draft acceptance rates on representative prompts.
- Measure combined throughput (NVFP4 + speculative decode) on GB10.
- Evaluate memory overhead of draft model / auxiliary head.
- Consider Mamba state branching requirements for draft tokens (each draft needs its own SSM state copy).

Expected outcome: 2-2.5x additional decode speedup based on NVIDIA's published DGX Spark results.

---

### 19. Add: Work Item 8 — Operator Fusion Audit

Identify and measure the highest-impact operator fusions achievable on SM121.

Tasks:

- Benchmark unfused vs fused residual + RMSNorm (expected 6-8x for those ops).
- Benchmark fused SwiGLU MLP via cuDNN FE runtime fusion (expected 10-13% end-to-end).
- Evaluate cuDNN FE fused Grouped GEMM + SwiGLU for MoE FC1/FC2 patterns.
- Measure kernel launch overhead reduction from fusion.

Primary mechanism: cuDNN FE runtime fusion engine (the portable option on SM121).

---

### 20. Add to Exit Criteria: OOM Safety Validation

Current exit criteria focus on performance measurements. Add:

> "The runtime's memory budget planner prevents GPU OOM under all supported concurrency configurations, validated by stress testing at maximum context length and maximum request count. The memory budget uses `/proc/meminfo`-based signals (not `cudaMemGetInfo` alone) and maintains a documented safety margin."

---

## Roofline Analysis Reference

For future planning, the DGX Spark roofline parameters:

Ridge points (OI where compute-bound meets memory-bound):

| Precision | Peak Compute | OI Ridge (at 273 GB/s) | OI Ridge (at 180 GB/s effective) |
|---|---|---|---|
| FP16 | ~100 TFLOPS | ~366 FLOPs/Byte | ~556 FLOPs/Byte |
| FP8 | ~500 TFLOPS | ~1832 FLOPs/Byte | ~2778 FLOPs/Byte |
| FP4 | 1000 TFLOPS | ~3663 FLOPs/Byte | ~5556 FLOPs/Byte |

Decode OI (batch=1):

- FP4 weights: OI = 2/0.5 = 4 FLOPs/Byte → **deeply memory-bound** (compute utilization ~0.1%)
- FP8 weights: OI = 2/1.0 = 2 FLOPs/Byte → deeply memory-bound
- BF16 weights: OI = 2/2.0 = 1 FLOP/Byte → deeply memory-bound

Prefill crossover to compute-bound:

For FP4 on DGX Spark: `sequence_length > (bytes_per_param * OI_ridge) / 2 ≈ (0.5 * 3663) / 2 ≈ 915 tokens`. Prompts longer than ~1K tokens are compute-bound during prefill.

MoE advantage: For the 120B/12B-active architecture, only ~10% of total weights are read per token during decode. Benchmarks confirm: Qwen3 30B MoE (3B active) achieves ~89 tok/s while the similarly-sized dense Qwen3 32B hits ~10.7 tok/s — an 8x advantage for MoE on bandwidth-limited hardware.

Sources:

- [RooflineBench (arxiv 2602.11506)](https://arxiv.org/html/2602.11506)
- [LLM Inference Unveiled (arxiv 2402.16363)](https://arxiv.org/pdf/2402.16363)
- [Efficient LLM Inference: Bandwidth, Compute, Synchronization (arxiv 2507.14397)](https://arxiv.org/html/2507.14397v1)
- [NVIDIA DGX Spark Performance Blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/)

---

## Nemotron-3-Super Quantization Recipe Reference

For reference during NVFP4 contract validation, the model's quantization recipe:

**Stage 1: Hybrid PTQ with AutoQuantize**

AutoQuantize formulates layer format assignment as a neural architecture search problem. Uses second-order Taylor approximation (inspired by Optimal Brain Surgeon) to estimate operator sensitivity, models performance cost, and solves for FP4/FP8/BF16 allocation minimizing total sensitivity under deployment constraints. Result: 99.8% median accuracy vs BF16 in under 2 hours on 8x B200 GPUs using 512 calibration samples.

Layer-specific format assignments:

- **NVFP4**: Majority of linear layers (weights and activations).
- **MXFP8**: MoE GEMMs, Mamba GEMMs, Mamba output projections (which suffer up to 40% values flushed to zero under NVFP4).
- **BF16**: Latent projections, MTP layers, QKV/attention projections, embeddings.
- **FP8**: KV cache (datacenter Blackwell only; BF16 on GB10).
- **FP16**: Mamba SSM kernel, Mamba state cache (reduced from FP32 for speedup).

**Stage 2: FP4 QAD (Quantization-Aware Distillation)**

- Teacher: frozen BF16 model. Student: NVFP4-quantized model.
- Loss: KL divergence between teacher and student logits.
- Data blend: 60% SFT + 40% RL on-policy rollouts.
- Budget: 5 billion tokens.

**Stage 3: Weight scaling**

- Per-block MSE minimization for weights (calibrated offline).
- Per-block max-based scaling for activations (computed at runtime).

Mamba state recipe: FP16 cache with stochastic rounding (Philox round count of 5). "Maintains accuracy and verbosity with Philox round count of 5." TensorRT-LLM config: `mamba_ssm_cache_dtype: float16`, `mamba_ssm_stochastic_rounding: true`, `mamba_ssm_philox_rounds: 5`.

Sources:

- [Nemotron-3-Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Enable NVFP4 Inference for Nemotron with QAD](https://research.nvidia.com/labs/nemotron/nemotron-qad/)
- [Nemotron 3 Super Quantization Documentation](https://docs.nvidia.com/nemotron/latest/nemotron/super3/quantization.html)
- [Nemotron-3-Super Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html)
- [NVIDIA Nemotron-3-Super-120B-A12B-NVFP4 (HuggingFace)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)

---

## CUDA 13.x Release History Reference

| Release | Date | Key DGX Spark Features |
|---|---|---|
| CUDA 13.0.0 | Aug 2025 | SM121 support, Blackwell architecture, unified Arm toolkit |
| CUDA 13.0.1 | Sep 2025 | Block-scaled FP4 GEMM on Blackwell/Blackwell Ultra |
| CUDA 13.0.2 | Dec 2025 | DGX Spark FP16/BF16/FP8 GEMM improvements |
| CUDA 13.1.0 | Jan 2026 | CUDA Tile, Grouped GEMM API (FP8/BF16), CUDA Graph support |
| CUDA 13.2.0 | Mar 2026 | **3x NVFP4/MXFP8 on DGX Spark**, BF16/FP16 illegal memory access fix, CUDA Tile for SM121, unified Tegra/desktop toolkit |

cuDNN version history (Blackwell-relevant):

| Version | Key Feature |
|---|---|
| 9.7.0 | SM 10.0 and 12.0 support, MXFP8/NVFP4 matmul |
| 9.8.0 | Improved SDPA inference on Blackwell |
| 9.9.0 | Arbitrary head dimension SDPA, improved paged attention |
| 9.10.1 | Fixed FP8 SDPA deadlock on Blackwell |
| 9.11.0 | FP8 paged attention inputs |
| 9.13.0 | FP8 SDPA forward on Blackwell |
| 9.18.0 | **MoE Grouped Matmul**, fused SwiGLU, deterministic SDPA backward |
| 9.20.0 | Latest release (Mar 2026) |

---

## cuDNN FE Paged Attention Contract Reference

For the attention implementation:

```
K/V containers: (num_blocks, H_kv, block_size, D)
Page tables:    (B, 1, ceil(S_kv / block_size), 1)   INT32
seq_len_q:      (B, 1, 1, 1)                          INT32
seq_len_kv:     (B, 1, 1, 1)                          INT32
```

- Block sizes: 32, 64, 128 (power of 2). Block size 32 recommended for DGX Spark.
- Prefill vs decode: Implicit via `seq_len_q` (>1 for prefill, =1 for decode).
- Packed ragged page tables available since cuDNN 9.10.2.
- Backward pass does NOT support paged attention (inference only).
- FP8 attention unavailable on SM121.
- BF16 recommended for GB10.

---

## What the Plan Gets Right

The plan's core judgment calls are well-supported by the research:

- **Service priorities (low TTFT, decode speed)**: Validated. Decode is the dominant bottleneck on a 273 GB/s system.
- **cuBLASLt as the dense GEMM foundation**: Validated. It is also the only viable dense GEMM option on SM121 (CUTLASS SM100 kernels crash).
- **cuDNN FE for paged attention**: Validated. It is also the only working attention backend on SM121.
- **Mixed precision preservation**: Validated by the Nemotron-3-Super quantization recipe (NVFP4/MXFP8/BF16/FP32 layer-specific assignments determined by AutoQuantize sensitivity analysis).
- **Per-tensor format identity**: Validated. The model card and technical report confirm intentional mixed-format deployment.
- **Device-resident weight upload as immediate next step**: Validated by roofline analysis. Per-call weight copies make any benchmark results meaningless.
- **Not freezing NVFP4 layout until validated**: Correct. Now unblockable with cuBLAS 13.2 + reference code.
- **Mamba state as first-class decode optimization**: Validated with concrete numbers showing 2-31% of decode bandwidth depending on batch size.
- **Hypothesis H1 (device-resident weights mandatory)**: Validated.
- **Hypothesis H2 (KV layout should follow cuDNN paged attention)**: Validated and strengthened — it is the only option on SM121.
- **Hypothesis H3 (Mamba decode is bandwidth-limited)**: Validated with 2.5 FLOPs/byte arithmetic intensity.
- **Hypothesis H4 (CUDA version matters)**: Strongly validated. cuBLAS 13.2 delivers 3x NVFP4/MXFP8 improvement on DGX Spark and fixes a critical bug.
- **The decision to not design around GPUDirect RDMA**: Correct, confirmed unavailable on DGX Spark.
- **The decision to design KV state around paged attention**: Correct, now further justified as the only portable attention mechanism on SM121.

---

## Mamba Cache Plan: Overall Plan vs Research Findings

This section compares the Mamba cache strategy described across three planning documents:

- `nemotron_dgxspark_codex_plan.md` (Milestone 5, Milestone 4 cache data model, Milestone 8 speculative state)
- `docs/v1_cache_architecture.md` (reusable node contents, deferred work)
- `docs/gb10_performance_plan.md` (Mamba cache format benchmarks, hypotheses)

against the external research findings gathered in this gap analysis.

### What the overall plan gets right about Mamba cache

**1. FP32 baseline first, then FP16 + stochastic rounding**

The plan's two-phase approach in Milestone 5 (Phase 5A: FP32 correctness baseline, Phase 5B: FP16+SR with Philox RNG) exactly matches NVIDIA's recommended deployment recipe. The NVIDIA advanced deployment guide uses `mamba_ssm_cache_dtype: float32` as the conservative baseline and `float16` with `mamba_ssm_stochastic_rounding: true` and `mamba_ssm_philox_rounds: 5` as the optimized recipe.

The plan's explicit warning against naive FP16 round-to-nearest ("Do not optimize the Mamba cache with naive FP16 casts" — item 4 in "What Codex should explicitly avoid") is directly validated by the technical report's finding that naive FP16 casting harms verbosity due to accumulated recurrent bias.

**2. Reproducible counter-based RNG contract**

The plan specifies: "define a reproducible counter-based RNG contract for validation, keyed at minimum by seed, layer, token position, and update step so replay and cache-restore tests are stable." This is correct and aligns with the Philox PRNG approach. Blackwell has dedicated PTX instructions for FP16 conversion with stochastic rounding, and cuRAND provides fast Philox on Blackwell, making the implementation path practical on GB10.

**3. (KV + Mamba state) as the reusable unit**

The project thesis — "Do not count KV-only reuse as a completed Nemotron prefix-cache hit; a reusable node is the pair (KV pages + Mamba state)" — is the correct design choice. NVIDIA's TRT-LLM disables block reuse for Mamba because their stack treats it as non-prefix-cacheable. This project's core differentiator is solving that problem, and the plan correctly identifies this as the primary opportunity.

**4. Full state at conversation boundaries, not dense small-block checkpoints**

The plan's v1 decision to snapshot at high-value boundaries (committed conversation heads, shared roots) rather than at small token intervals is the right tradeoff for the target workload. Dense checkpointing at 32/64/128 token boundaries would require ~176 MiB (FP32) or ~88 MiB (FP16) per checkpoint across 44 Mamba layers, making the memory cost prohibitive for v1. The plan correctly defers this to later exploration.

**5. Conv state inclusion**

The cache data model explicitly calls out: "include both conv state and SSM state if that is how the runtime materializes recurrent state." This is correct. The Nemotron-3-Super conv state is ~164 KB per layer at FP32 (~7 MiB total across 44 layers), which is small relative to the SSM state but must be included for correct resume.

**6. Speculative state isolation**

Milestone 8 correctly specifies: "tentative-state handling for both KV and Mamba state during drafting," "commit accepted draft state without copy-back when possible," and "rollback rejected draft state without polluting committed request state or shared prefix cache." This matches SGLang's approach where each speculative draft token gets a private cache slot with its own SSM state copy.

### What the plan is missing or should extend

**7. INT8 Mamba state quantization (Quamba2) — not mentioned**

The plan considers only FP32, FP16, and FP16+stochastic-rounding. Research shows that **INT8 per-group quantization** (Quamba2, March 2025) is a viable fourth option:

- 4 head groups x 4 channel groups, symmetric uniform quantization
- 3x generation speedup at batch=1 (7.43ms vs 22.73ms TPOT)
- Enables 2x larger batch sizes before OOM
- Quality loss is measurable (~3 points on NaturalQuestions, ~6 on SquadV2) but may be acceptable for latency-sensitive scenarios
- Key finding: "cached SSM states are redundant — they tolerate aggressive quantization"

Recommendation: Add INT8 as a candidate in Phase 5B's benchmark sweep, alongside FP32, FP16, and FP16+SR. Even if not selected for v1, the data informs future batch-scaling decisions.

**8. Batch-scaling behavior of Mamba state bandwidth — not analyzed**

The plan identifies Mamba state bandwidth as a decode constraint (H3 in the performance plan) but does not analyze how the cost scales with concurrent requests. Since Mamba state is per-request while weights are shared across the batch, the state fraction grows significantly:

| Batch | Weight traffic (NVFP4) | Mamba state R+W (FP16) | State as % |
|---|---|---|---|
| 1 | ~6 GB | ~0.17 GB | 2.8% |
| 4 | ~6 GB | ~0.68 GB | 10.2% |
| 8 | ~6 GB | ~1.36 GB | 18.5% |

At the v1 target of 8 concurrent requests, nearly 1/5 of decode bandwidth goes to Mamba state. This has two implications:

- The Mamba cache format choice (FP32 vs FP16 vs INT8) becomes increasingly impactful at higher concurrency
- The benchmark matrix in the plan (which sweeps `batch size: 1 / 2 / 4`) should extend to batch=8 to match the service target

Recommendation: Add batch=8 to the Mamba cache benchmark sweep. Document the batch-scaling analysis so the format decision accounts for the full service profile.

**9. Concrete state dimensions should be documented**

The plan references `mamba_state_bytes_fp16: 87162880` and `mamba_state_bytes_fp32: 174325760` in the manifest, but does not break down the per-layer dimensions. For implementation clarity:

- SSM state per layer: 128 heads x 64 headdim x 128 d_state = 1,048,576 elements = **4 MiB (FP32) / 2 MiB (FP16)**
- Conv state per layer: (8192 + 2 x 8 x 128) x 4 = 10,240 x 4 = 40,960 elements = **~164 KB (FP32)**
- Total SSM state across ~44 Mamba layers: **~176 MiB (FP32) / ~88 MiB (FP16)**
- Total conv state across ~44 layers: **~7 MiB (FP32) / ~3.5 MiB (FP16)**
- Total Mamba state per request: **~183 MiB (FP32) / ~91.5 MiB (FP16)**

These match the manifest values (87162880 bytes FP16 = ~83 MiB for SSM only; the conv state is additional).

Recommendation: Document the per-layer breakdown in the performance plan or cache architecture doc so that memory budget calculations are transparent. This also helps when evaluating checkpoint granularity for prefix caching.

**10. State copy overhead for prefix reuse — not quantified**

The plan's cache architecture correctly describes forking from cached prefixes, but does not quantify the copy cost. Research indicates:

- "Copying Mamba-2 states takes at least 10% of CPU time on some systems" due to multi-MiB state sizes per layer
- At FP16, restoring a full Mamba state from a committed conversation head requires copying ~88 MiB across 44 layers
- On DGX Spark's unified memory (273 GB/s), a bulk memcpy of 88 MiB takes ~0.3ms — fast, but not free at scale

For the speculative decoding path (Milestone 8), each draft token needs its own state copy. With draft length 7, that is 7 x 88 MiB = ~616 MiB of state copies per speculation round at FP16, or 7 x 176 MiB = ~1.2 GiB at FP32.

Recommendation: Add state copy bandwidth to the Milestone 5 benchmarks. For Milestone 8, budget the speculative-state memory overhead explicitly, and consider whether copy-on-write or differential state tracking is worth the complexity.

**11. SGLang's MambaRadixCache as closest prior art — not referenced**

The plan's v1 approach (exact-prefix nodes with full Mamba state) is architecturally compatible with SGLang's MambaRadixCache implementation, which:

- Maintains a radix tree of immutable SSM state checkpoints indexed by prefix tokens
- New requests fork from cached prefixes with a copy of the checkpoint state
- Each speculative draft token receives a private cache slot
- When draft tokens are accepted, the last accepted slot becomes the new main state

SGLang also implements `HybridReqToTokenPool` (binds Mamba state to requests) and `HybridLinearKVPool` (maps logical layers to actual cache indices), configurable via `--mamba-full-memory-ratio`.

vLLM V1 also has merged automatic prefix caching for Mamba-2 (PR #25752) using selective state checkpointing.

Neither is referenced in the plan. While the project intentionally builds a custom runtime, these implementations represent tested approaches to the same problem and are valuable as reference architectures.

Recommendation: Add SGLang's MambaRadixCache and vLLM V1's Mamba prefix caching as reference implementations in the plan's cache section, similar to how TRT-LLM KV cache reuse is already referenced.

**12. SM121 stochastic rounding hardware support — needs verification**

The plan specifies Philox-based stochastic rounding for FP16 Mamba cache, and the research confirms that "Blackwell hardware has dedicated PTX instructions for FP16 conversion with stochastic rounding." However, this finding refers to datacenter Blackwell (SM100). Given that SM121 uses Ampere-era `mma.sync` instructions and lacks TMEM and `tcgen05`, it is worth verifying that the stochastic rounding PTX instructions are available on SM121 specifically, and not only on SM100.

If the dedicated PTX path is unavailable on SM121, the fallback is a software stochastic rounding implementation using cuRAND Philox, which is still fast but has different performance characteristics.

Recommendation: Add a preflight probe for stochastic rounding PTX instruction availability on SM121 before implementing Phase 5B. This is a quick check that could avoid a costly mid-implementation discovery.

**13. Mamba state is not prefix-cacheable in the same sense as KV — architectural note**

The plan correctly treats (KV + Mamba state) as the reusable unit and builds the cache around full-state checkpoints at conversation boundaries. However, the plan should explicitly document the fundamental difference:

- **KV cache**: Attention KV is append-only. Sharing a prefix means sharing the same physical page blocks (refcounting). A cache hit avoids both compute and memory allocation.
- **Mamba state**: SSM state is a lossy compression updated in-place. Sharing a prefix requires **copying** the full checkpoint state (not sharing physical memory). A cache hit avoids compute but still requires a memory copy.

This distinction affects:
- Eviction semantics: Evicting a KV prefix frees shared pages. Evicting a Mamba checkpoint frees its dedicated state allocation.
- Memory accounting: KV pages can be shared (refcount > 1). Mamba state at a cached node is always a single copy, but each request that forks from it gets its own copy.
- The cost model for cache capacity: A "cached prefix" costs KV pages (shared, amortized across users) + Mamba state (one copy in the cache, one copy per active request forked from it).

The v1 cache architecture doc correctly specifies "allocator-owned reusable-state descriptors with deterministic retain/release semantics," which accommodates this, but the distinction should be explicit for implementors.

**14. Benchmark matrix should include state-restore latency**

The plan's suggested benchmark matrix sweeps Mamba cache dtype (FP32 vs FP16+SR) and batch size. It should also measure:

- **State-restore latency**: Time to fork a new request from a committed conversation head (memcpy of full Mamba state)
- **State-commit latency**: Time to snapshot the final Mamba state into a new cache node after assistant turn completion
- **Speculative-branch overhead**: Per-draft-token state copy cost

These directly affect hot-prefix TTFT, which is the primary service priority.

**15. Warm-up cost for Mamba state after partial prefix hit**

The plan's cache architecture handles exact-prefix hits well, but does not discuss the cost model for partial prefix matches. If the cache has a committed head at token position 5000 but the request has 5500 tokens (500 new tokens in the latest user turn), the runtime must:

1. Restore the full Mamba state from the committed head (copy ~88 MiB FP16)
2. Prefill the 500-token uncached tail through all 44 Mamba layers (sequential recurrence, memory-bound)

The prefill-tail cost for Mamba layers is fundamentally different from attention layers:
- **Attention**: Prefill tail processes all 500 tokens in parallel (one forward pass per layer)
- **Mamba**: Prefill tail can use the chunked SSD algorithm (parallel within chunks, sequential across chunks), but the state update is still sequential in terms of information flow

This means hot-prefix TTFT for the Mamba layers is: restore time + chunk-parallel prefill time for the tail. The plan should document this cost model so benchmark expectations are calibrated correctly.

### Summary of Mamba cache alignment

| Aspect | Plan status | Research alignment |
|---|---|---|
| FP32 → FP16+SR progression | Correct | Exact match with NVIDIA recipe |
| Philox RNG contract | Correct | Validated |
| No naive FP16 casting | Correct | Validated by technical report |
| Full state at conversation boundaries | Correct | Best tradeoff for v1 workload |
| Conv + SSM state inclusion | Correct | Both required for correct resume |
| Speculative state isolation | Correct | Matches SGLang approach |
| INT8 state quantization | Missing | Quamba2 shows 3x speedup, worth benchmarking |
| Batch-scaling analysis | Missing | State fraction grows from 3% to 19% at batch 8 |
| State copy overhead | Not quantified | ~88 MiB per fork at FP16; ~0.3ms on DGX Spark |
| SGLang/vLLM prior art | Not referenced | Closest implementations of same problem |
| SM121 stochastic rounding HW | Not verified | May need preflight probe |
| KV vs Mamba cache semantics | Implicit | Should be explicit (share vs copy) |
| State-restore benchmarks | Not in sweep | Should be in benchmark matrix |
| Partial-prefix warm-up cost | Not discussed | Mamba prefill tail is sequential |

Sources:

- [Nemotron-3-Super Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html)
- [Nemotron-3-Super Quantization Stage Documentation](https://docs.nvidia.com/nemotron/latest/nemotron/super3/quantization.html)
- [Nemotron-3-Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Quamba2: Robust and Scalable Post-training Quantization for SSMs (arxiv 2503.22879)](https://arxiv.org/html/2503.22879)
- [Hybrid Models Meet SGLang (PyTorch Blog)](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- [Hybrid Models as First-Class Citizens in vLLM (PyTorch Blog)](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)
- [vLLM RFC: Native SSM support in V1 (GitHub #17140)](https://github.com/vllm-project/vllm/issues/17140)
- [Mamba-3 (arxiv 2603.15569)](https://arxiv.org/html/2603.15569v1)
- [Accelerating Mamba2 with Kernel Fusion (PyTorch Blog)](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/)
