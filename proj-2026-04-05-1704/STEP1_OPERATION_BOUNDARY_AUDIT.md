# Step 1: Operation Boundary Audit

Status: implementation boundary locked for the current optimization series.

This record answers the blocking Step 1 question from
[`PLAN.md`](./PLAN.md): what is the single production implementation boundary
for each operation on the active `SM120` Nano forward path, and are projection
GEMMs being replaced in this series or not.

## Scope

Target stack only:

- Nemotron 3 Nano NVFP4
- consumer Blackwell `SM120`
- `build-sm120-relwithdebinfo`

Out of scope for this series:

- `SM75`
- generic `build/`
- alternate runtime paths retained for comparison

## Active operation boundaries

### Embedding / final norm / `lm_head`

Single production boundary:

- [single_token_forward_model.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/single_token_forward_model.cpp)

Decision:

- keep the current native forward-model orchestration path
- do not introduce alternate embedding or logits paths in this series

### Attention

Single production boundary:

- [attention_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/attention_layer.cpp)
- [attention_native_kernels.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/attention_native_kernels.cu)

Decision:

- keep the current native attention path
- optimize inside this path only
- do not add a separate long-context or decode-only backend

### Mamba

Single production boundary:

- [mamba_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/mamba_layer.cpp)
- [mamba_conv_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/mamba_conv_prefill.cu)
- [mamba_ssd_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/mamba_ssd_prefill.cu)
- [fused_mamba_decode.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_mamba_decode.cu)

Decision:

- keep the current native Mamba path
- optimize SSD prefill, conv prefill, and surrounding norm/state plumbing
- do not add a separate long-prefill Mamba backend

### MoE routing / execution

Single production boundary:

- [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp)
- [expert_routing_device.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_routing_device.cu)
- [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu)

Decision:

- keep the current direct device-routing + fused prefill / direct decode path
- optimize inside this path only
- do not restore grouped CUTLASS split dispatch, host routing adapters, or
  token-count fallback logic

### Projection GEMMs

Single production boundary for this series:

- [linear_op.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/linear_op.cpp)
- [nvfp4_gemm_runner.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/nvfp4_gemm_runner.cpp)

Decision:

- yes, the current cuBLASLt / CUTLASS-backed projection stack is the permanent
  single implementation boundary for this optimization series
- no, projection GEMMs are not being replaced by a native fused path in this
  series

Rationale:

- the current TTFT bottlenecks are dominated by work around the active
  projection path as much as by the GEMM kernel itself, especially expert-path
  packing/scaling and synchronization
- another partial projection rewrite would violate the single-path rule before
  the current prefill bottlenecks are understood and reduced
- the relevant `SM120` prior art on this machine is also still CUTLASS-style at
  the MoE/GEMM layer, not TRTLLM-Gen

## External alignment conclusion

Relevant references for this target:

- vLLM: FlashInfer CUTLASS-style NVFP4 MoE
- TRT-LLM: `CutlassFusedMoE`

Not the implementation target for this machine:

- vLLM TRT-LLM NVFP4 expert path for Nano shapes
- TRT-LLM `TRTLLMGenFusedMoE`

Important constraint:

- use vLLM and TRT-LLM for kernel/layout/contract ideas only
- do not import their backend-selection or fallback structure into runtime

## Immediate implementation consequence

The first optimization work should target Step 2 inside the existing expert
prefill path:

1. remove redundant hot-path metadata copies and synchronization where possible
2. attribute and reduce hot-path `cudaMemcpy`
3. reduce quantization/packing overhead without creating a second MoE path
