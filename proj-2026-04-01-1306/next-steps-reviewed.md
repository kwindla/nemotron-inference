# Next Steps: Prompt Parity through Prefill Fastpath (Reviewed)

## Context

All diagnostic infrastructure is landed (18 commits). The 5 remaining checkpoints from the design doc need implementation. Each builds on the previous. This version incorporates Codex's code-level review.

## Corrected facts from code review

- Nano runtime config: hidden_size=2688, mamba_intermediate_size=4096, mamba_in_proj=10304, moe_latent_size=2688, routed_expert_intermediate_size=1856, shared_expert_intermediate_size=3712
- The 2688 % 512 hypothesis is unsupported — plan-build failures come from zero/invalid buffers and 16-byte pointer alignment, not dimension divisibility
- Shared expert NVFP4 weights are already eagerly resident at Create(); only routed experts are staged per-run
- Current staging counters don't have hit/miss tracking — that needs new counters
- The attention compare hook is device-vs-host, not production-vs-fallback
- Both fused kernels hard-require token_count==1; multi-token prefill needs dispatcher changes in single_token_forward_model.cpp plus new kernel variants
- cuDNN FE can be built with real or stub paths depending on build-time flag NEMOTRON_RUNTIME_HAVE_CUDNN_FE

## Steps (revised per review)

### 1. Prompt-parity diagnosis runner
Add a thin shell script that runs nano_16_token_correctness_test with all diagnostic flags (NEMOTRON_NANO_16_TRACE_DIVERGENCE=1, NEMOTRON_NANO_16_TRACE_EMBEDDING=1, NEMOTRON_NANO_16_STRICT_LINEAR=1) and formats the route-matrix output into a structured summary. Not new infrastructure — a wrapper around what's already landed.

### 2. Fix prompt-boundary divergence based on route matrix
Based on which pair diverges:
- A↔B diverges: decode-consistent prefill bug — investigate state initialization in request_context.cpp, KV cache commit in the token-by-token loop in single_token_forward_model.cpp:1010
- B↔C diverges: fused kernel math difference — investigate mamba_layer.cpp:527, expert_layer.cpp:1022, attention_layer.cpp:697
- Use existing per-layer localization in nano_16_token_correctness_test.cpp
- The fix is code-specific and depends on the diagnosis output

### 3. Enable and stabilize linear device fastpath
- Enable NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1
- Use per-tensor LinearOpTrace to identify which tensors fall back
- Use NEMOTRON_FORWARD_DEBUG=1 for M/N/K diagnostic logging
- Separate plan-build failures from heuristic/execution failures
- Fix both linear_op.cpp and scaled_fp8_linear.cu paths
- Key shapes: hidden 2688, mamba in_proj 10304, routed expert 1856, shared expert 3712

### 4. Memory-budgeted routed expert device cache
- Shared experts are already eagerly resident — routed experts only
- Add layer-local cache in ExpertLayerSlice::Impl (simpler than global cache)
- Add cache hit/miss counters to ExpertStagingCounters
- Budget a configurable VRAM fraction
- LRU eviction by (expert_id) within each layer
- Wire into expert_layer.cpp:1041 where routed experts are currently staged per-run

### 5. Production decode attention kernel
- cuDNN FE is not available in stub build; commit to custom kernel first
- Replace the current scalar fallback in attention_device_fallback.cu (block size 1, threadIdx.x==0 only)
- Multi-warp per query head, tiled softmax, efficient value accumulation
- Integration point: attention_layer.cpp:477 (non-cuDNN paged-attention call)
- Keep existing device-vs-host compare hook at attention_layer.cpp:735 for parity
- Must handle both decode (single query token) and multi-token queries

### 6a. Multi-token fused Mamba prefill
- Current fused_mamba_decode.cu:134 hard-requires token_count==1
- Need a multi-token variant that processes Mamba recurrence sequentially but batches conv/SSM state updates
- Dispatcher change in single_token_forward_model.cpp:1010 to route multi-token prefill through capability-aware path instead of replay
- Integration in mamba_layer.cpp:527

### 6b. Multi-token fused MoE prefill
- Current fused_moe_decode.cu:278 hard-requires token_count==1
- Need batched routing + per-token expert dispatch
- Integration in expert_layer.cpp:1022
- Can share routing logic but needs per-token accumulation
