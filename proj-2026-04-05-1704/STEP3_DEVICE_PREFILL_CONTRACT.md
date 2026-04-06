# Step 3 Device Prefill Contract

## Goal

Replace the old host-driven MoE prefill path with a single device-side contract:

- no `expert_offsets` DtoH copy
- no host loop over active experts
- no per-expert cuBLASLt plan selection in the hot path
- no gather/pack/scatter workspace as part of the execution contract

## Producer: `expert_layer.cpp`

The producer side now hands prefill only the data the custom kernel needs:

- `input[token_count, hidden_size]`
- `normalized[token_count, hidden_size]`
- `topk_ids[token_count, top_k]`
- `topk_weights[token_count, top_k]`
- prepared resident NVFP4 routed up/down weight views
- prepared resident NVFP4 shared up/down weight views

Producer invariants:

- shapes match the layer config
- `topk_ids` / `topk_weights` are already computed on device
- routed/shared weight views are resident and valid
- `output[token_count, hidden_size]` is writable

## Consumer: `fused_moe_prefill.cu`

The consumer side owns the full per-token execution sequence:

1. Load one token per block.
2. Load that token's `topk_ids` / `topk_weights`.
3. Quantize-dequantize the normalized row once.
4. For each selected routed expert:
   - routed up projection
   - `Relu2`
   - activation quantize-dequantize
   - routed down projection
   - weighted accumulation
5. Run the shared expert path:
   - shared up projection
   - `Relu2`
   - activation quantize-dequantize
   - shared down projection
6. Write `output = input + routed + shared`.

Optional debug outputs:

- `routed_output[token_count, hidden_size]`
- `shared_output[token_count, hidden_size]`

## Explicit Non-Contract Items

These are no longer part of the prefill execution boundary:

- `DeviceExpertRouting`
- `expert_offsets`
- `sorted_token_indices`
- `sorted_token_weights`
- routed gather scratch
- routed/shared pack buffers
- cuBLASLt handles, descriptors, or heuristic caches

## Test Anchors

Boundary test:

- `testing/backend/fused_moe_prefill_test.cpp`

Integration smoke:

- `testing/backend/expert_layer_fastpath_test.cpp`
- `testing/api/full_forward_manifest_smoke_test.cpp`
