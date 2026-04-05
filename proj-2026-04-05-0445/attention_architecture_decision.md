# Attention Architecture Decision

Decision date: `2026-04-05`

## Decision

The project will target a **custom specialized multi-token paged-attention
backend** for Nano on RTX 5090.

This backend should be informed by FlashInfer-like long-context behavior, but it
should be implemented as a native runtime backend that fits the existing C++ /
CUDA paged-KV and exact-prefix-cache architecture.

## Rejected Directions

### Reject: cuDNN as the project end-state

Reasons:

- cuDNN is not currently available in the active local build.
- The user explicitly does not want an incremental detour that is not the path
  to the target architecture.
- The external oracle work already gives us a better architectural signal than a
  "just make cuDNN work first" step.

cuDNN remains prior art and a possible comparison point, but not the chosen
production direction for this project.

### Reject: TRTLLM attention as the project end-state

Reasons:

- The vLLM probe on this machine reports `supports_trtllm_attention = false`.
- The same probe reports `current_platform_family100 = false` on this consumer
  `SM120` setup.
- That means the relevant TRTLLM-attention path is not the active Blackwell
  path available to us in the current environment.

TRT-LLM remains a profiling and prior-art reference, not the implementation
target.

### Reject: direct FlashInfer adoption as the project end-state

Reasons:

- The runtime is a native C++ / CUDA system with its own paged-KV ABI,
  request-context ownership model, and exact-prefix cache restore flow.
- The project needs a backend that fits `attention_layer.cpp`,
  `paged_attention_plan.h`, and `request_context.cpp` directly.
- The FlashInfer oracle numbers are valuable, but a Python-stack integration is
  not the intended shipping architecture here.

FlashInfer is therefore treated as the closest performance prior art for the
multi-token path, not as the literal implementation target.

## Positive Direction

The chosen direction is:

- keep the existing page-table / KV-cache ABI
- keep BF16 KV cache
- keep the Nano decode-specialized path only if it still wins for `token_count=1`
- replace the current multi-token fallback path with a new specialized backend
  for Nano geometry on `SM120`
- optimize that backend around:
  - long-context extend
  - bounded query chunk sizes
  - low concurrency (`<=4`)
  - causal paged attention only

## Evidence Used

### Root-cause profile

The fresh local profile showed:

- `PagedAttentionDeviceFallback`: `98.7%` of GPU time
- routed MoE GEMMs: `0.2%`

That forced the project away from the old MoE-first plan.

### External oracle

The vLLM oracle artifacts showed:

- FlashInfer is best across the long-context extend cases in the frozen matrix
- FlashAttention remains competitive or best on the smallest prefill cases
- Triton is consistently weaker

This is the right pattern for a custom specialized backend decision: copy the
winning shape behavior, not the whole foreign runtime stack.

## Immediate Next Step

Proceed to step 4 in `ATTENTION_PLAN.md`:

- lock the runtime attention contract
- choose the fixed query chunk set
- define the backend policy boundary between decode-specialized and multi-token
  execution
