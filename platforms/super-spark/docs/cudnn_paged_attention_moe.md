# cuDNN Frontend Paged Attention and MoE Grouped Matmul: Research Brief

**Date:** March 28, 2026
**cuDNN Backend latest:** 9.20.0 (March 2026)
**cuDNN Frontend latest:** v1.21.0 (March 25, 2026, recommended for cuDNN 9.20.0+)
**CUTLASS latest:** 4.4.0

---

## 1. cuDNN Frontend Paged Attention API -- Exact Contract

### Core SDPA Call with Paged Attention

**Python:**
```python
graph.sdpa(
    q,                                  # Query tensor
    k,                                  # K container (block pool, NOT per-batch)
    v,                                  # V container (block pool, NOT per-batch)
    paged_attention_k_table=page_table_k,
    paged_attention_v_table=page_table_v,
    paged_attention_max_seq_len_kv=max_seq_len_kv,
    seq_len_q=seq_len_q,
    seq_len_kv=seq_len_kv,
    is_inference=True,
    use_padding_mask=True,
    ...
)
```

**C++:**
```cpp
auto sdpa_options = graph::SDPA_attributes()
    .set_paged_attention_k_table(page_table_k)
    .set_paged_attention_v_table(page_table_v)
    .set_paged_attention_max_seq_len_kv(max_seq_len_kv)
    .set_name("sdpa_paged");
```

### K/V Container Tensors (the block pool)

These are NOT per-batch tensors. They are a flat pool of blocks shared across all sequences.

| Property      | K Container                        | V Container                        |
|---------------|------------------------------------|------------------------------------|
| **Shape**     | `(num_blocks_k, H_k, bs_k, D_qk)` | `(num_blocks_v, H_v, bs_v, D_v)` |
| **Layout**    | BHSD-like: block, head, seq_within_block, dim | Same |
| **Data type** | FP16, BF16, or FP8 (arch-dependent) | Same |

- `num_blocks_k` / `num_blocks_v`: total blocks in the pool (not per-batch)
- `H_k` / `H_v`: number of KV heads (for GQA, typically < H_q)
- `bs_k` / `bs_v`: block size (tokens per block), **must be power of 2**
- `D_qk` / `D_v`: head dimension

### Page Table Tensors (INT32)

| Property      | K Page Table                               | V Page Table                               |
|---------------|--------------------------------------------|--------------------------------------------|
| **Data type** | INT32                                      | INT32                                      |
| **Shape**     | `(B, 1, ceil(S_kv / bs_k), 1)`            | `(B, 1, ceil(S_kv / bs_v), 1)`            |
| **Stride**    | `(blocks_per_batch, blocks_per_batch, 1, 1)` | Same |
| **Content**   | Block indices into K container              | Block indices into V container              |

The offset calculation is:
```
K_cache[b, h, s, d] = K_container[page_table_k[b, 1, s // bs_k, 1], h, s % bs_k, d]
V_cache[b, h, s, d] = V_container[page_table_v[b, 1, s // bs_v, 1], h, s % bs_v, d]
```

K and V page tables can be **independent** -- they do not need to point to the same blocks, allowing separate paging for K and V. However, in practice most implementations share the same page table for both.

### Sequence Length Tensors (INT32)

| Tensor       | Shape            | Data type | Purpose                               |
|--------------|------------------|-----------|---------------------------------------|
| `seq_len_q`  | `(B, 1, 1, 1)`  | INT32     | Actual query sequence length per batch |
| `seq_len_kv` | `(B, 1, 1, 1)`  | INT32     | Actual KV sequence length per batch    |

These are **mandatory** when paged attention is active (padding mask is required).

### paged_attention_max_seq_len_kv (Optional but Recommended)

A scalar integer that tells cuDNN the maximum KV sequence length across all batches. When provided, cuDNN can size internal workspaces more efficiently. When omitted, cuDNN infers it from the page table dimensions, but the documentation recommends passing it explicitly.

### Prefill vs Decode Specification

cuDNN FE does **not** have an explicit "prefill" vs "decode" flag. The distinction is encoded implicitly in the tensor shapes:

- **Prefill**: `seq_len_q > 1` (multiple query tokens per batch element). Q shape is `(B, H_q, S_q, D_qk)` where `S_q` can be large.
- **Decode**: `seq_len_q = 1` (single token per batch element). Q shape is `(B, H_q, 1, D_qk)`.

The same SDPA graph can serve both cases -- different batch elements can have different `seq_len_q` values, enabling mixed prefill + decode batching as long as the Q tensor is padded to `max(seq_len_q)`.

Sample 52 demonstrates prefill with paged caches; Sample 53 demonstrates decode with paged caches.

### Packed (Ragged) Page Tables (cuDNN 9.10.2+)

Page tables themselves can use ragged offsets to pack only the necessary block indices, eliminating padding waste in the page table. This is useful when batch elements have very different KV sequence lengths. Use `Tensor_attributes.set_ragged_offset()` on the page table tensor.

### Paged + Ragged (THD) Combination (cuDNN 9.7.0+)

The Q/O tensors can use ragged (THD) layout while K/V use paged containers. This enables packed queries without padding while K/V remain in the block pool. THD layout means T = sum(seq_len) is the total number of valid tokens packed contiguously.

---

## 2. Block Size Choices for Paged Attention

### What cuDNN Supports

The cuDNN FE test suite tests block sizes of **32, 64, and 128**. Block sizes must be **power of 2**. The documentation states this as a hard requirement.

Based on available evidence, the supported block sizes are:
- **32** -- smallest tested
- **64** -- middle option
- **128** -- largest tested

Block sizes of 1, 2, 4, 16, 256, and above are either not supported or not tested in the cuDNN FE test harness. (Note: vLLM's own paged attention kernel does not compile block sizes 1/2/4/64/128/256 to reduce compilation time; this is a vLLM limitation, not a cuDNN one.)

### Performance Implications

**On memory-bandwidth-limited platforms (like DGX Spark with 273 GB/s):**

1. **Smaller blocks (32):**
   - Higher gather overhead: more page-table lookups per layer per token (ceil(T/32) vs ceil(T/128) blocks)
   - But better L1 cache hit rates: smaller working set per block fits L1 better
   - Less internal fragmentation: average waste is B/2 = 16 tokens per sequence
   - vLLM's paged decode kernel shows **higher memory bandwidth utilization** with smaller blocks due to L1 cache efficiency

2. **Larger blocks (128):**
   - Less gather overhead: fewer page-table lookups
   - But worse L1 cache efficiency on decode (each block is 128 * D * dtype bytes)
   - More internal fragmentation: average waste is B/2 = 64 tokens per sequence
   - Can waste significant memory when sequences are shorter than block size

3. **Practical guidance:**
   - Block size **16** is the widely-accepted sweet spot in the vLLM ecosystem (large enough for GPU efficiency, small enough to avoid fragmentation). cuDNN does not appear to support 16 natively.
   - Block size **32** is likely the best choice for cuDNN FE on memory-bandwidth-limited platforms
   - For long-context workloads (64k+), block size 64 or 128 may be acceptable since fragmentation overhead is proportionally small (B / (2 * T) ~ 0.1% for 64k with bs=128)
   - The expected relative waste per sequence is approximately `block_size / (2 * avg_seq_len)`

4. **DGX Spark specific consideration:**
   - 128 GB unified LPDDR5X at 273 GB/s -- severely memory-bandwidth-limited
   - Decode is fundamentally memory-bandwidth-bound (each generated token reads entire KV cache)
   - Smaller block sizes (32) may actually perform better due to L1 cache effects
   - The 128 KB shared memory per SM on GB10 (vs 228 KB on B200) further favors smaller blocks

---

## 3. cuDNN Frontend MoE Grouped Matmul

### API Surface

Introduced in **cuDNN 9.18.0** as part of the runtime fusion engine.

**C++ API:**
```cpp
auto [output] = graph.moe_grouped_matmul(
    token,              // Input activation tensor
    weight,             // Expert weight tensor
    first_token_offset, // Expert-to-token mapping
    token_index,        // Token routing indices (Gather/Scatter modes)
    token_ks,           // Token counts per expert (Scatter mode)
    attributes          // Configuration
);
```

**Configuration attributes:**
```cpp
MoeGroupedMatmul_attributes()
    .set_name("moe_fc1")
    .set_mode(MoeGroupedMatmulMode_t::SCATTER)  // or GATHER or NONE
    .set_compute_data_type(DataType_t::FLOAT)
    .set_top_k(top_k);
```

### Three Execution Modes

| Mode       | Computation                                           | Use Case |
|------------|-------------------------------------------------------|----------|
| **None**   | `Output[1, S*topK, N] = Token[1, S*topK, K] * Weight[E, K, N]` | Tokens already routed |
| **Gather** | `Output[1, S*topK, N] = Token[1, S, K] * Weight[E, K, N]`     | Gather tokens during GEMM |
| **Scatter**| `Output[1, S*topK, N] = Token[1, S*topK, K] * Weight[E, K, N]` | Scatter results after GEMM |

### Input Tensor Specifications

| Tensor            | Shape             | Data type | Description |
|-------------------|-------------------|-----------|-------------|
| Token             | `[1, S*topK, K]` or `[1, S, K]` (Gather) | FP16/BF16/FP8 | Input activations |
| Weight            | `[E, K, N]`       | FP16/BF16/FP8 | Expert weights (E = num_experts) |
| FirstTokenOffset  | `[B*E, 1, 1]`     | INT32     | Offset of first token per expert |
| TokenIndex        | `[1, S*topK, 1]`  | INT32     | Routing index (Gather/Scatter only) |
| TokenKs           | `[1, S*topK, 1]`  | INT32     | Per-token expert count (Scatter only) |
| Output            | `[1, S*topK, N]`  | FP16/BF16/FP8 | Result |

### Expert Dispatch Mechanism

The `FirstTokenOffset` tensor encodes the CSR-style offset of the first token assigned to each expert in each batch element. For `B` batch elements and `E` experts, this is a flat array of `B*E` values. The MoE grouped matmul kernel internally groups tokens by expert and dispatches the grouped GEMM accordingly.

### Fused Variants (cuDNN 9.20.0+, FE v1.21.0)

- **Grouped GEMM + GLU**: Fused GEMM with GLU activation for MoE FC1, supporting both dense and discrete weight layouts
- **Grouped GEMM + dGLU**: Backward pass for the above
- **Grouped GEMM + SwiGLU**: Persistent batched dense GEMM fused with SwiGLU epilogue on Blackwell SM100+ (open-sourced via FE-OSS)
- **Grouped GEMM + dSwiGLU**: Contiguous grouped block-scaled GEMM fused with dSwiGLU backward on Blackwell SM100+
- **Grouped GEMM + Quant**: Unified grouped GEMM with per-row gating and quantization for MoE FC2/dFC1

### Memory Layout Requirements

- Weight tensor layout: `[E, K, N]` -- experts stacked along first dimension
- Tokens must be pre-sorted or sorted by the routing logic before passing to cuDNN (unless using Gather/Scatter modes which handle routing internally)
- For discrete MoE weight layouts (per-expert-pointer variants), weight packing is not required -- each expert can have its own pointer

---

## 4. CUTLASS Grouped GEMM as Alternative for MoE

### Current State (CUTLASS 4.3.0+)

CUTLASS has purpose-built MoE grouped GEMM support, particularly optimized for Blackwell:

- **Example 92**: Ragged Contiguous Grouped GEMM kernel designed specifically for MoE low-latency inference on Blackwell SM100
- **MoeProblemShape**: Simplified struct that takes `max_m, max_n, max_k` and a counts vector, deducing problem shapes internally
- **moe_stride_utils**: API to set up strides in the kernel without manual CuTe utility calls
- **Block-scaled variant**: Supports all microscaling types (MXFP8, MXFP4, etc.)
- **TMA 3D load** for weights with tensormap update for activations

### Key Design Difference from cuDNN

CUTLASS MoE grouped GEMM constrains only the M-axis to vary across experts (N and K are fixed across experts). This is the natural pattern for MoE where all experts share the same hidden dimensions but receive different numbers of tokens. cuDNN's MoE grouped matmul follows a similar pattern.

### Performance: CUTLASS vs cuDNN FE

| Aspect | CUTLASS | cuDNN FE |
|--------|---------|----------|
| **Blackwell perf** | Up to 5x faster than Hopper at various precisions | 5-10% improvements in 9.18.0 for attention |
| **MoE-specific** | Purpose-built low-latency MoE kernels (example 92) | General runtime fusion engine |
| **Flexibility** | Template-based, full control over tiling/scheduling | Graph-based, auto-tuned |
| **FP8 support** | SM90 and SM100 | SM90 and SM100 (datacenter only) |
| **Compilation** | Ahead-of-time (long compile) | Runtime fusion (faster iteration) |
| **Custom fusion** | Arbitrary epilogues via CUTLASS | Predefined fusion patterns |

### DeepGEMM (Third-party Alternative)

DeepSeek's open-source DeepGEMM library provides another option:
- **1.4-2.7x speedup** over CUTLASS for standard FP8 GEMMs
- **1.1-1.3x speedup** for MoE contiguous/masked grouped GEMMs
- JIT-compiled at runtime (no ahead-of-time compilation)
- Supports SM90 and SM100
- Design constraint: only M-axis varies across groups (N, K fixed) -- same as MoE workloads

### Advantages of CUTLASS for MoE

1. **Full control**: Can tune tile sizes, pipeline stages, cluster shapes per workload
2. **Blackwell SM100 optimized**: Dedicated MoE kernels with TMA 3D loads
3. **Block-scaled GEMM**: Native support for MXFP4/MXFP8/NVFP4 microscaling
4. **Epilogue fusion**: Can fuse arbitrary operations (SwiGLU, quantization, etc.)
5. **Open source**: Full source available for modification

### Advantages of cuDNN FE for MoE

1. **Runtime fusion**: No recompilation when problem sizes change
2. **Dynamic shape support** (9.18.0+): Reduced runtime compilation overhead
3. **Integrated graph API**: Can fuse MoE with attention and normalization in one graph
4. **Auto-tuning**: cuDNN's heuristic engine selects optimal kernels
5. **Less code**: Graph-based API requires less boilerplate than CUTLASS templates

### SM120 (DGX Spark / Consumer Blackwell) Considerations

**Critical limitation**: CUTLASS SM100 MoE kernels (example 92) use `tcgen05` / WGMMA instructions that are **not available on SM120/SM121**. The GB10 in DGX Spark uses extended `mma.sync` (Ampere-era programming model). CUTLASS SM100 code will fail with `ptxas error: Instruction 'wgmma.fence' not supported` on SM12x.

For DGX Spark, the options are:
- cuDNN FE runtime fusion (which can target SM120 via backend dispatch)
- CUTLASS SM80/SM89 kernels (Ampere/Ada code path)
- DeepGEMM (if SM120 support is added)
- Triton-based grouped GEMM (JIT-compiles for SM121)

---

## 5. Performance Characteristics on Blackwell

### cuDNN FE Attention on Blackwell (B200/B300 -- Datacenter SM100)

- **cuDNN 9.9.0**: "Significantly improved" paged attention performance on Blackwell for both prefill and decode
- **cuDNN 9.15.0**: "Flash Attention v4" techniques adopted for Blackwell, with configurable environment variables
- **cuDNN 9.17.0**: FP8 attention performance "significantly optimized" on SM10.3
- **cuDNN 9.18.0**: 5-10% improvement for forward/backward attention (Llama3, DSv3 patterns)
- **cuDNN 9.19.0**: Deterministic FP8 backward on Blackwell
- **cuDNN 9.20.0**: Independent SDPA stats generation, further forward pass improvements

### FlashAttention-4 vs cuDNN on Blackwell

- FA4 achieves **1,605 TFLOPs/s** on B200 (71% hardware utilization)
- FA4 is **~20% faster** than cuDNN 9.13 attention on Blackwell
- FA4 delivers **1.1-1.3x** improvement on forward passes vs cuDNN
- FA4 uses software-emulated exponential (skips 90% of output rescaling)
- Implemented in CuTeDSL with 20-30x faster compile times than C++ template approaches

### Blackwell Architectural Characteristics

Blackwell exhibits **asymmetric hardware scaling**:
- Tensor core throughput doubles vs Hopper
- Shared memory bandwidth and exponential units scale more slowly
- This means attention kernels must minimize non-matmul operations (softmax, rescaling)
- FA4 and cuDNN 9.15+ both address this by reducing exponential unit pressure

### GB10 (DGX Spark, SM121) Specific

- **128 GB** unified LPDDR5X at **273 GB/s** -- severely memory-bandwidth-limited
- **1 PFLOP** sparse FP4 peak
- **No TMEM** (tensor memory) -- lacks the 256 KB/SM dedicated tensor memory of datacenter Blackwell
- **128 KB** shared memory per SM (vs 228 KB on B200)
- **48 SMs** with 6,144 CUDA cores
- **48 warps/SM** max (vs 64 on SM100)

Performance implications:
- Decode is **entirely memory-bandwidth-bound** at 273 GB/s
- Prefill can be compute-bound for long sequences but bandwidth-limited for short ones
- cuDNN SDPA backend works on SM121 (falls back to appropriate code path)
- FlashAttention-3/4, FlashInfer, FlashMLA **do not work** on SM121 (require SM100 or SM90)
- Triton JIT-compiles for SM121 but treats it as SM80, disabling Blackwell optimizations
- Recommended: cuDNN SDPA for attention on GB10

---

## 6. Known Issues, Limitations, and Version Requirements

### Version Requirements

| Feature | Minimum cuDNN Backend | Minimum FE Version |
|---------|----------------------|-------------------|
| Basic paged attention | 9.5.0 | -- |
| Paged + ragged (THD) combination | 9.7.0 | -- |
| Packed (ragged) page tables | 9.10.2 | -- |
| Paged attention in unified SDPA node | 9.15.0 | -- |
| MoE grouped matmul | 9.18.0 | v1.18.0 |
| Dynamic shape for SDPA/matmul | 9.18.0 | -- |
| Grouped GEMM + SwiGLU (SM100) | 9.20.0 | v1.21.0 |
| Independent SDPA stats (LSE, SE, Max) | 9.20.0 | v1.20.0 |
| Ragged support on Ampere/Ada | 9.18.0 | -- |

### Hard Limitations

1. **Backward pass does NOT support paged attention.** K and V must be contiguous tensors for backward. This is inference-only.

2. **Block sizes must be power of 2.** Tested values: 32, 64, 128. Other sizes may work but are untested.

3. **Padding mask is mandatory** when paged attention is active. Must provide both `seq_len_q` and `seq_len_kv`.

4. **Head dimension constraints:**
   - Multiple of 8 for FP16/BF16
   - Multiple of 16 for FP8
   - Maximum 256 for prefill, 128 for decode (architecture-dependent)

5. **SM120 (consumer Blackwell) lacks FP8 attention** -- only FP16/BF16 supported

6. **SM80 and SM120 do not support deterministic algorithm with ragged input tensor**

7. **Block masking only supported with UNIFIED implementation** (cuDNN 9.13.1+)

8. **FP8 backward does not support dropout**

### Known Bugs Fixed

- **9.18.1**: Fixed numerically incorrect output for FP8 SDPA with ragged offsets and FP16/BF16 output
- **9.19.1**: Fixed application hangs for FP8 SDPA on Blackwell with small sequence lengths
- **9.18.0**: Fixed successive SDPA runs with ragged offsets causing hangs or illegal accesses

### Performance Degradations

- Sink attention fprop with padding mask on Hopper has degraded performance
- Decode-phase attention with causal masking may produce mismatches or NaNs if Q head count is mismatched
- SDPA bprop is not supported when both K/V have sequence length of 1
- Memory allocations for certain convolution workloads persist until process termination

---

## 7. Comparison: cuDNN FE Paged Attention vs FlashAttention / FlashInfer / Others

### Feature Matrix

| Feature | cuDNN FE | FlashAttention-3 | FlashAttention-4 | FlashInfer | Triton |
|---------|----------|-------------------|-------------------|------------|--------|
| **Paged KV cache** | Native (9.5.0+) | Via FlashInfer | N/A (prefill focus) | Native | Native |
| **FP8 attention** | Hopper, Blackwell DC | Hopper | Blackwell DC | Hopper, Blackwell | Limited |
| **SM121 (GB10)** | Yes (via fallback) | No | No | No | Yes (as SM80) |
| **SM100 (B200)** | Yes | Yes | Yes | Yes | Yes |
| **SM90 (H100)** | Yes | Yes | N/A | Yes | Yes |
| **Causal masking** | Yes | Yes | Yes | Yes | Yes |
| **GQA/MQA** | Yes | Yes | Yes | Yes | Yes |
| **Sliding window** | Yes | Yes | N/A | Yes | Yes |
| **Custom score mod** | Limited (block masking) | No | No | JIT-compiled | Full (via Triton) |
| **Backward (training)** | Yes (not paged) | Yes | Yes | Limited | Yes |
| **Open source** | Partial (FE-OSS) | Full | Full | Full | Full |

### Performance Positioning

**FlashAttention-3 (Hopper):**
- Up to 740 TFLOPs/s on H100 (75% utilization)
- 1.2 PFLOPS with FP8
- Surpasses cuDNN for medium-to-long sequences on H100

**FlashAttention-4 (Blackwell):**
- 1,605 TFLOPs/s on B200 (71% utilization)
- ~20% faster than cuDNN 9.13 on Blackwell
- CuTeDSL-based, 20-30x faster compile times
- SM100 only -- no SM120 support

**FlashInfer:**
- Unified abstraction layer routing to multiple backends (FA2/3, cuDNN, TRT-LLM kernels, native)
- Best-in-class for variable-length batched decode with paged KV
- Up to 31x speedup over baseline vLLM PagedAttention for shared-prefix workloads
- MLSys 2025 Best Paper
- No SM121 support currently

**cuDNN FE:**
- Only option with native SM121 (DGX Spark) support for attention
- Integrated graph API enables end-to-end fusion
- Paged attention native since 9.5.0
- Slightly slower than FA3/FA4 on datacenter GPUs but most portable
- Runtime fusion engine avoids recompilation

**Triton:**
- JIT-compiles for any target architecture including SM121
- Persistent kernel variants achieve 100.7% of FA3 performance for long decode on H100
- Most flexible for custom attention patterns
- Treats SM121 as SM80, missing Blackwell-specific optimizations

### For DGX Spark (GB10, SM121)

The only viable attention backends are:
1. **cuDNN FE** -- native support, auto-dispatches appropriate code path
2. **Triton** -- JIT-compiles but without Blackwell optimizations
3. **Custom CUDA** -- using Ampere-era `mma.sync` and `cp.async` instructions

FlashAttention, FlashInfer, and FlashMLA all crash on SM121 with `cudaErrorIllegalInstruction`.

---

## 8. Best Practices for KV Cache Layout Targeting cuDNN FE Paged Attention

### Container Layout

**Recommended layout:** `(num_blocks, H_kv, block_size, D)` (BHSD order within blocks)

- `num_blocks`: Total block pool size. Size this for your maximum concurrent KV tokens: `num_blocks = max_total_kv_tokens / block_size`
- `H_kv`: Number of KV heads (for GQA with 32 Q heads / 2 KV heads, H_kv = 2)
- `block_size`: Tokens per block (32, 64, or 128)
- `D`: Head dimension (must be multiple of 8 for FP16/BF16, multiple of 16 for FP8)

### Data Type Selection

| Data Type | Memory per token per head | Accuracy | Platform Support |
|-----------|--------------------------|----------|------------------|
| FP16      | 2 * D bytes              | Baseline | All (SM80+) |
| BF16      | 2 * D bytes              | ~FP16    | All (SM80+) |
| FP8       | 1 * D bytes              | Minor loss | SM90, SM100 DC only |
| NVFP4     | 0.5 * D bytes + scales   | <1% loss | SM100+ (via TRT-LLM, not cuDNN FE directly) |

**For DGX Spark (SM121):** FP16 or BF16 only. FP8 attention is not supported on consumer Blackwell.

cuDNN FE's FP8 attention computes in FP8 internally but takes FP16/BF16 inputs and produces FP16/BF16 outputs (the quantization/dequantization is fused). For frameworks that store KV cache in FP8 and want to use cuDNN FE on SM121, a dequantization step before attention is required.

### Page Size Recommendations

| Workload | Recommended Block Size | Rationale |
|----------|----------------------|-----------|
| Short context (< 4k), many concurrent requests | 32 | Minimize fragmentation |
| Medium context (4k-32k) | 64 | Balance fragmentation vs overhead |
| Long context (32k+), few concurrent requests | 128 | Amortize page-table overhead |
| DGX Spark (bandwidth-limited, 128 KB smem) | 32 | L1 cache efficiency, reduce smem pressure |

### Memory Alignment

- **Tensor alignment**: cuDNN 9.18.0 relaxed alignment from 128-byte to 16-byte for normalization engines. For attention tensors, 16-byte alignment is the current requirement.
- **Head dimension**: Must be multiple of 8 (FP16/BF16) or 16 (FP8)
- **Block size**: Must be power of 2

### Page Table Design

- Allocate page tables on GPU memory (INT32)
- Size: `(B, 1, max_blocks_per_seq, 1)` where `max_blocks_per_seq = ceil(max_seq_len / block_size)`
- For variable-length batches, use packed page tables with ragged offsets (cuDNN 9.10.2+) to avoid wasting memory on padding in the page table itself
- Pre-allocate page table at max capacity; update indices as blocks are allocated/freed

### Memory Budget Calculation

For Nemotron-3-Super on DGX Spark (2 KV heads, D=128, 88 layers):

```
Per-token KV memory (BF16) = 2 heads * 128 dim * 2 bytes * 2 (K+V) * 88 layers
                            = 88 * 1024 bytes = 88 KB per token

With block_size=32:
  Per-block = 32 * 88 KB = 2.75 MB
  For 8 concurrent 64k-context requests:
    Total tokens = 8 * 65536 = 524,288 tokens
    Total blocks = 524,288 / 32 = 16,384 blocks
    KV cache memory = 524,288 * 88 KB = ~43.5 GB

With NVFP4 (if supported):
  Per-token = 88 * 256 bytes = ~22 KB per token
  Total for 8x64k = ~11 GB
```

Given the 128 GB unified memory budget of DGX Spark (shared with model weights, activations, and OS), BF16 KV cache at 64k context will consume a significant fraction. NVFP4 KV cache (via TRT-LLM quantization, dequantized before cuDNN FE attention) would be highly beneficial.

### Prefix Caching with Paged Attention

Paged attention naturally supports prefix caching:
- Shared prefix tokens occupy the same physical blocks across requests
- Page tables for different requests point to the same container blocks for shared prefixes
- Only divergent suffixes require new block allocations
- Block size affects prefix-sharing granularity: smaller blocks enable sharing at finer granularity

---

## Sources

### Official NVIDIA Documentation
- [cuDNN Frontend Attention Documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)
- [cuDNN Frontend MoE Grouped Matmul](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/MoeGroupedMatmul.html)
- [cuDNN Backend Release Notes (latest)](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html)
- [cuDNN Backend 9.18.1 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html)
- [cuDNN Frontend FE-OSS APIs Overview](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/overview.html)
- [GEMM + SwiGLU (SM100) FE-OSS](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/fe-oss-apis/gemm_fusions/gemm_swiglu.html)
- [cuDNN Frontend GitHub Releases](https://github.com/NVIDIA/cudnn-frontend/releases)
- [cuDNN Frontend Samples README](https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/README.md)
- [cuDNN Frontend test_mhas.py](https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python/test_mhas.py)
- [cuDNN Frontend scaled_dot_product_flash_attention.h](https://github.com/NVIDIA/cudnn-frontend/blob/main/include/cudnn_frontend/node/scaled_dot_product_flash_attention.h)

### CUTLASS
- [CUTLASS 4.4.0 Changelog](https://docs.nvidia.com/cutlass/4.4.0/CHANGELOG.html)
- [CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html)
- [CUTLASS Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
- [CUTLASS Blackwell Grouped GEMM Python Example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/grouped_gemm.py)

### NVIDIA Blogs and Technical Posts
- [Introducing Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/)
- [Optimizing Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [NVIDIA Blackwell 3x Faster Training](https://developer.nvidia.com/blog/nvidia-blackwell-enables-3x-faster-training-and-nearly-2x-training-performance-per-dollar-than-previous-gen-architecture/)
- [FlashInfer High-Performance Kernels via NVIDIA](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)

### FlashAttention and Alternatives
- [FlashAttention-4 Paper (arXiv:2603.05451)](https://arxiv.org/abs/2603.05451)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608)
- [Modal: Reverse-Engineering Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)

### DGX Spark / GB10
- [Is DGX Spark Actually Blackwell? (Backend.AI)](https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell)
- [vLLM Triton Attention Backend Deep Dive](https://blog.vllm.ai/2026/03/04/vllm-triton-backend-deep-dive.html)

### Performance Analysis
- [Paged Attention Performance Analysis](https://martianlantern.github.io/2025/09/paged-attention-performance-analysis/)
- [Grouped GEMMs and MoE](https://ianbarber.blog/2025/02/11/grouped-gemms-and-moe/)
- [PyTorch Triton Persistent Cache-Aware Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
