# Reference Framework Analysis: How vLLM Achieves 14 tok/s on DGX Spark

Deep analysis of vLLM v0.18.1 serving Nemotron-3-Super-120B-A12B-NVFP4 on DGX Spark (SM121/GB10).

## Community Benchmark Numbers

| Config | Decode tok/s | Source |
|---|---:|---|
| vLLM + NVFP4 CUTLASS, single Spark | 14-15.2 | NVIDIA forums |
| vLLM + Marlin dequant to BF16, single Spark | 16.6 | NVIDIA forums |
| vLLM + NVFP4 TP=2, dual Spark | 24 | NVIDIA forums |
| Ollama Q4_K_M GGUF, single Spark | 18-19.5 | Community |
| Our runtime | 1.0-1.5 | Measured, depending on whether the comparison uses the 16-token tail or shorter warmed decode microbench |

## The Three Architectural Differences

### 1. Fused Monolithic MoE Kernel (biggest difference)

vLLM uses `flashinfer.fused_moe.trtllm_fp4_block_scale_moe()` — a **single monolithic CUDA kernel** from FlashInfer/TRT-LLM that executes the ENTIRE MoE layer:

```
ONE kernel launch does ALL of:
  1. Router logit processing (softmax/sigmoid)
  2. Grouped top-k expert selection
  3. Token-to-expert permutation
  4. For each of top-k experts:
     a. NVFP4 weight dequantization
     b. GEMM1 (up_proj) via CUTLASS
     c. Activation (ReLU²)
     d. GEMM2 (down_proj) via CUTLASS
  5. Expert output weighting
  6. Token-expert reduction to final output
```

**Our implementation** does this in **~8+ separate kernel launches** per MoE layer: router GEMM, top-k selection, per-expert pointer gather, grouped up_proj GEMM, relu2+pack, grouped down_proj GEMM, weighted merge, plus the D→H sync for expert indices. Plus the shared expert path is separate.

**Impact**: 40 MoE layers × (8+ launches vs 1 launch) = 320+ extra launches eliminated. At ~0.1-0.5ms per launch overhead, this alone accounts for 32-160ms of our gap.

The kernel source lives in FlashInfer, not vLLM itself. It's a compiled CUTLASS-based kernel that:
- Reads expert weights directly from a contiguous `[num_experts, intermediate_size, hidden_size]` tensor
- Uses expert indices to compute pointer offsets internally
- Never returns control to the host between expert computations
- Handles NVFP4 block-scale dequantization inline during the GEMM

### 2. CUDA Graph Capture for Decode

vLLM captures the **entire forward pass** as a CUDA graph for decode tokens:

```python
# From vllm/v1/worker/gpu_model_runner.py
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
    # ONE graph replay = ALL layers, ALL kernels
```

The graph is captured once during warmup and replayed on every subsequent decode token. This eliminates:
- All Python overhead between layers (~2.5ms per layer × 88 layers = 220ms)
- All kernel launch scheduling overhead
- All cudaMemcpy for control tensors
- All descriptor/plan building

**What's INSIDE the graph**: embedding lookup, all 88 layer forward passes (including the fused MoE kernel, Mamba ops, attention), final norm, LM head projection.

**What's OUTSIDE the graph**: input tensor updates (token IDs, positions), sampling, KV cache management metadata.

The `CudagraphDispatcher` supports two modes:
- **FULL**: entire forward pass captured (used for uniform decode batches)
- **PIECEWISE**: per-layer graphs with attention staying eager (fallback for mixed batches)

Variable expert selection works because the fused MoE kernel reads routing decisions from tensors inside the graph — the router logits and top-k selection are computed inside the graph too, so the expert selection varies per token without breaking the graph.

**Our implementation** launches every kernel individually from the host. No graph capture at all.

### 3. Optimized Mamba Decode Path

vLLM's Mamba decode uses two specialized kernels from the `causal-conv1d` and `mamba-ssm` packages:

```python
# Decode conv update — one fused kernel
hidden_states_B_C_d = causal_conv1d_update(
    hidden_states_B_C_d, conv_state, conv_weights, bias, activation,
    conv_state_indices=..., ...)

# Decode SSM update — one fused kernel
selective_state_update(
    ssm_state, hidden_states_d, dt_d, A_d, B_d, C_d, D_d,
    dt_bias=..., dt_softplus=True,
    state_batch_indices=..., dst_state_batch_indices=...,
    out=..., is_blackwell=True)
```

The `is_blackwell=True` flag enables a Blackwell-optimized path in `selective_state_update`. These are compiled CUDA kernels (not Triton), tuned for the SM120/SM121 architecture.

**Our implementation** uses custom device kernels for conv/SSM update, but they were written for correctness, not throughput. They likely have suboptimal thread/block configurations for SM121 and lack the batched state indexing that enables efficient multi-request decode.

## The vLLM Serving Configuration

From the community forums, the recommended configuration for Nemotron 3 Super NVFP4 on DGX Spark:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 8 \
  --max-model-len 262144 \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --mamba-ssm-cache-dtype float32 \
  --max-cudagraph-capture-size 512 \
  --quantization modelopt_fp4
```

Key flags:
- `--kv-cache-dtype fp8` — FP8 KV cache (halves attention memory traffic)
- `--enable-prefix-caching` — Mamba SSM state caching for prefix reuse
- `--mamba-ssm-cache-dtype float32` — FP32 SSM state for stability
- `--max-cudagraph-capture-size 512` — CUDA graph capture for batches up to 512

## Quantitative Breakdown

| Component | Our time | vLLM estimated | Ratio | Root cause |
|---|---:|---:|---:|---|
| Expert (40 layers) | 344ms | ~25ms | 14x | Monolithic fused kernel vs 8+ launches |
| Mamba (40 layers) | 208ms | ~20ms | 10x | Optimized selective_state_update + graph |
| Attention (8 layers) | 168ms | ~15ms | 11x | FlashInfer + graph vs cuDNN + per-call setup |
| Inter-layer overhead | 217ms | ~0ms | ∞ | CUDA graph eliminates all host overhead |
| Embedding + norm + LM head | ~0ms | ~11ms | - | Similar |
| **Total** | **937ms** | **~71ms** | **13x** | |

## What We Need To Match

In dependency order (not priority order — each step enables the next):

### 1. Fused MoE Kernel (prerequisite for CUDA graphs)

Replace our 8+ launches per MoE layer with a single fused kernel. This is the prerequisite for CUDA graph capture because our current MoE path is host-driven: selected expert indices are copied D→H so the host can dispatch per-expert GEMMs. CUDA graphs cannot capture host-dependent control flow.

vLLM's fused MoE kernel solves this by computing expert selection from live tensor data inside the kernel — the router logits flow from the hidden states (which change every token), so selection is dynamic without host involvement.

Options:
- Use FlashInfer's `trtllm_fp4_block_scale_moe` directly (open source, validated on SM121 with GDC fix from PR #2913)
- Use CUTLASS grouped GEMM with inline NVFP4 dequantization
- Build a custom fused kernel

FlashInfer is the fastest path — it handles the exact Nemotron architecture (LatentMoE with grouped top-k, ReLU² activation, NVFP4 block-scaled weights).

### 2. Optimized Mamba Decode Kernels (prerequisite for CUDA graphs)

Replace our correctness-first conv/SSM kernels with the Blackwell-optimized `causal_conv1d_update` and `selective_state_update` from the mamba-ssm package, or write equivalents tuned for SM121. These must be fully device-resident (no host involvement per token) to be CUDA-graph-compatible.

### 3. CUDA Graph Capture (only possible after steps 1-2)

Capture the entire forward pass as a CUDA graph. This is the largest throughput win, but it requires ALL per-token logic to be device-resident first. With host-driven MoE dispatch, the graph would replay the same expert selections every token — incorrect.

After steps 1-2, all 88 layers run entirely on device from live tensor data. The graph captures the full forward pass and replays it in one call, eliminating ~217ms of inter-layer host overhead plus all per-kernel launch scheduling cost.

### 4. FlashInfer Attention (~2x improvement on attention layers)

Replace cuDNN SDPA with FlashInfer's attention backend, which is designed for the variable-length decode pattern and integrates natively with CUDA graphs. Incremental on top of steps 1-3.

## Key Libraries Used by vLLM

| Library | Version | Purpose |
|---|---|---|
| FlashInfer | 0.6.7+ | Fused MoE NVFP4 kernel, attention backend |
| CUTLASS | 4.4.2+ | Underlying GEMM kernels for NVFP4 |
| causal-conv1d | latest | Mamba conv state update |
| mamba-ssm | latest | Selective state update for Mamba2 |
| PyTorch | nightly cu132 | CUDA graph capture, tensor management |

The FlashInfer fused MoE kernel and the mamba-ssm `selective_state_update` with `is_blackwell=True` are the two most important external kernels. Both are open source.

## Implications for Our Runtime

The gap is not in our model understanding or correctness — it's in the execution infrastructure:

1. **We launch ~1,120 individual kernels per token** where vLLM launches ~88 (one per layer) inside a single CUDA graph replay
2. **Our MoE path does 8+ launches per layer** where vLLM does 1
3. **Our host controls the GPU** at every layer boundary where vLLM's GPU runs autonomously through the graph

The fix is not incremental optimization of individual kernels. It's adopting the same three-layer execution architecture: fused kernels inside CUDA graphs with optimized Mamba primitives.
