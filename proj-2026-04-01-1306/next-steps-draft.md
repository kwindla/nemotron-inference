# Next Steps: Prompt Parity through Prefill Fastpath

## Context

All diagnostic infrastructure is landed (18 commits). The 5 remaining checkpoints from the design doc need implementation. Each builds on the previous — prompt parity must be stable before linear fastpath, which must be stable before expert residency, etc.

Key constraint: steps 1-2 require real-hardware diagnosis runs to identify exact root causes. The implementation plan should include both the diagnostic automation AND the most likely code fixes based on what the design doc already tells us about the failure modes.

## What we know from prior analysis

### Prompt parity (nano-prompt-parity)
- First divergence is at prompt-boundary token index 0
- Reference token: 3174, fused token: 1047, max_abs_diff: 5.53
- The route matrix (A/B/C) will isolate whether decode-consistent prefill or fused kernels cause it
- Design doc says: "multi-token prefill now routes token-by-token through the same one-token continuation path when fused runtime is enabled"
- Previous fix was routing prefill token-by-token, which fixed turn-2 split-prefill parity
- Current issue may be in embedding path, early layer state initialization, or the final norm/logits projection

### Linear fastpath (nano-linear-fastpath)
- `linear_device_fastpath_enabled=false` in saved benchmark artifact
- The fastpath is gated by `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1`
- When enabled, `BuildRuntimeGemmPlan()` or `BuildDescriptorGemmPlan()` may fail for certain shapes
- Nano hidden_size=2688, which is not a multiple of 512 — may cause cuBLASLt alignment issues
- Mamba in_proj output is 2*intermediate_size, Mamba out_proj is hidden_size
- The M/N/K diagnostic logging (just landed) will show exactly which shapes fail

### Expert residency (nano-expert-residency)
- Full eager residency caused VRAM allocation failures at model construction
- Current path stages routed experts lazily per-token
- Expert staging counters will show upload cost per token
- Need a budgeted cache with LRU-style eviction

### Attention production (nano-attention-prod)
- cuDNN FE confirmed NOT AVAILABLE (stub build)
- Current fallback: single-thread-per-query device kernel
- Need a production kernel with multi-thread query processing, efficient softmax reduction

### Prefill fastpath (nano-prefill-fastpath)
- Current fused prefill replays 16 one-token decode steps
- Prefill ms ≈ 16 × steady-state decode ms (confirmed by benchmark)
- Need genuine multi-token processing through Mamba conv/SSM and MoE routing

## Proposed steps

### 1. Automate prompt-parity diagnosis run
Write a script that runs the correctness test with all diagnostic flags and formats results into a summary. This captures the diagnosis data needed before fixing.

### 2. Fix prompt-parity based on diagnosis patterns
Based on the route matrix results (which route pair diverges), apply the targeted fix. Most likely candidates:
- If A↔B diverges: decode-consistent prefill itself has a bug (state initialization, KV cache interaction)
- If B↔C diverges: fused kernels have a math difference from reference
- If both: compound issue requiring per-layer localization

### 3. Enable and stabilize linear device fastpath
Enable `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1`, use the per-tensor trace to identify failing shapes, and fix the plan-build failures. Likely fixes: workspace allocation, shape-specific cuBLASLt descriptor configuration, or alignment padding.

### 4. Add memory-budgeted expert device cache
Implement an LRU device cache for routed expert NVFP4 weights. Budget a configurable fraction of VRAM. Cache by (layer_index, expert_id). Track hit/miss rates through the existing staging counters.

### 5. Production decode attention kernel
Replace the current single-thread-per-query fallback with a production kernel. Multi-warp per query head, tiled softmax, efficient value accumulation. Keep the current fallback as a parity reference under the existing compare hook.

### 6. Multi-token fused prefill
Add a fused prefill path for Mamba and MoE that processes multiple tokens without replaying the decode path. Start with 2-token, scale to full prompt length. Use the correctness test to verify parity at each stage.
