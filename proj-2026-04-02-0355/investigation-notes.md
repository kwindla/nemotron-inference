# Investigation Notes: 22.5ms → 20ms Decode Optimization

## Consolidated Findings (from parallel agent + Codex investigations)

### Finding 1: DeviceExpertSelectionKernel — `<<<1,1>>>` launch

**Root cause**: The kernel launches with `grid=1, block=1` — a SINGLE CUDA thread doing all work.
- Sigmoid on 128 values: serial loop
- Group scoring: serial scan (dead code for Nano where n_group=1, topk_group=1)
- Top-k insertion sort: O(k×N) serial
- 9 KB of thread-local arrays backed by local memory, not shared memory
- Cost: 0.054ms/call × 23 layers = **1.3ms/token**

**Fix**: Change to `<<<1, 128>>>` (one thread per expert) with warp-level parallel reduction for top-k. For Nano's n_group=1 case, the group logic is dead — special-case it. Expected improvement: 0.054ms → ~0.005ms/call, saving **~1.1ms/token**.

**vLLM comparison**: vLLM does expert selection via `torch.topk()` or fused Triton kernel, keeping results as device tensors. Never copies routing results to host.

### Finding 2: PackInto D2H tensor scale readback — 184 calls/token

**Root cause**: `DeviceNvfp4Matrix::PackInto()` computes the activation tensor scale on device (global max reduction), then does a `cudaMemcpy(D2H, 4 bytes)` to read it back to host. The host value is needed because `RunNvfp4RowMajorFp32AccumToDevice()` takes `float activation_tensor_scale_host` and computes `alpha = act_scale * weight_scale` on host.

- 8 PackInto calls per MoE layer × 23 layers = **184 D2H copies per token**
- Each is 4 bytes but forces a pipeline stall (~0.003ms each)
- Total: ~0.6ms/token of stall time

**Fix**: Use cuBLASLt's `CUBLASLT_POINTER_MODE_DEVICE` to pass alpha as a device pointer. Compute `alpha = act_tensor_scale * weight_tensor_scale` on device via a tiny kernel. Eliminate ALL D2H reads from PackInto.

**vLLM comparison**: vLLM's `nvfp4_quant_kernels.cu` keeps the FP4 scale on device — the kernel reads `SFScale[0]` directly from device memory. No host round-trip.

### Finding 3: Expert indices D2H — 23 calls/token

**Root cause**: After device expert selection, the cuBLASLt MoE path copies the 6 selected indices (24 bytes) back to host to iterate over expert weight views. This forces a sync point.

- 1 D2H × 23 layers = **23 D2H copies per token**

**Fix**: Build the weight view lookup on device. The host only needs indices to select which monolithic weight views to use — but these views are already on device in the `monolithic_up_views_device`/`monolithic_down_views_device` arrays. The cuBLASLt path could take device pointers directly if we restructure the expert iteration to be device-driven.

### Finding 4: Attention metadata — 40 async H2D/token

**Root cause**: 5 `cudaMemcpyAsync` calls per attention layer × 8 layers = 40 per token. Small data (4-64 bytes each). `page_table_k` and `page_table_v` are identical data uploaded twice.

**Fix**: Deduplicate page_table_k/page_table_v to a single buffer. Pin the host metadata vectors for truly async operation.

### Per-Token Decode Memcpy Summary

| Source | Direction | Count/Token | Eliminable? |
|--------|-----------|-------------|-------------|
| PackInto tensor_scale D2H | D2H | 184 | Yes — device-side alpha |
| Expert indices D2H | D2H | 23 | Yes — device-side iteration |
| Attention metadata H2D | H2D (async) | 40 | Partially — deduplicate |
| Embedding token ID H2D | H2D | 1 | Low priority |
| **Total** | | **248** | **~207 eliminable** |

### Priority Order

1. **Fix `<<<1,1>>>` expert selection** → -1.1ms/token (easiest, biggest single kernel win)
2. **Eliminate PackInto D2H** → -0.6ms/token (use POINTER_MODE_DEVICE for cuBLASLt alpha)
3. **Eliminate expert indices D2H** → minor (24 bytes, but forces sync)
4. **Deduplicate attention page tables** → minor
