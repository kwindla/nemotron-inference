# Phase-1 Direct FC1 Contract Notes

This note freezes the phase-1 contract for the routed P5 direct-FP4 path before kernel work begins.

## Goal

Replace the grouped routed FC1 boundary

```text
P5 GEMM -> BF16 output -> ReLU2+FP4 pack
```

with

```text
P5 GEMM -> direct FP4 pack
```

without changing routed FC2 semantics.

## Activation Semantics

- `Relu2` means squared ReLU in this repo:
  - `Relu2(x) = x > 0 ? x * x : 0`
- The fused epilogue must preserve the current grouped path ordering:
  - `activated = Relu2(alpha * accum)`
- It must **not** compute `alpha * Relu2(accum)`.

## Scale Contract

For each routed output row and each 16-element FP4 block:

1. Compute:

```text
dequant_scale = ClampNvfp4Scale(block_max_abs / 6.0f)
```

where `block_max_abs` is taken over the post-activation values.

2. Store the encoded FP8 scale byte into:

- `block_scales_data[row * blocks_per_row + block]`
- `matmul_block_scales_data[ExecutionScaleOffset(row, block, ...)]`

3. Materialize the absolute dequant scale into:

- `activation_output_scale[row, block]`

This preserves the current routed FC2 path, which consumes `activation_output_scale` as `input_dq_scales`.

## DeviceNvfp4Matrix Contract

The direct output must be a fully initialized `DeviceNvfp4Matrix`, not a private one-off payload.

Required field initialization:

- `packed_data`: packed FP4 nibbles for the activated values
- `block_scales_data`: encoded FP8 bytes for the same absolute dequant scales
- `matmul_block_scales_data`: same scales in execution-swizzled layout
- `tensor_scale_data[0] = 1.0f`
- `per_row_tensor_scales[row] = 1.0f` for all matrix rows

Rationale for `per_row_tensor_scales = 1.0f`:

- The routed FC2 phase-1 path still uses `activation_output_scale` directly.
- Initializing per-row tensor scales to `1.0f` keeps the matrix coherent for generic readers that reconstruct values from:

```text
DecodeFp4(nibble) * DecodeFp8(block_scale_byte) * per_row_tensor_scale
```

- With absolute block scales and `per_row_tensor_scale = 1.0f`, that generic path reconstructs the same values.

## Padding / Masking Rules

For inactive padded rows or masked-out blocks:

- `packed_data` bytes must be zero
- `block_scales_data` bytes must be zero
- `matmul_block_scales_data` bytes must be zero
- `activation_output_scale` must be zero

`per_row_tensor_scales` may remain `1.0f` on padded rows as long as the block scales are zeroed.

## Consumer Guarantees

Phase 1 must support both of these read styles:

1. Routed FC2 current path:
   - consumes `packed_data` + `activation_output_scale`
   - ignores per-row tensor scales

2. Generic matrix reader path:
   - consumes `packed_data` + `block_scales_data` or `matmul_block_scales_data`
   - multiplies by `tensor_scale_data` and/or `per_row_tensor_scales`

Because `tensor_scale_data = 1.0f` and `per_row_tensor_scales = 1.0f`, both styles should reconstruct the same absolute scale.

## Non-Goals for Phase 1

- No per-expert tensor-scale side channel
- No FC2 contract rewrite
- No BF16 workspace removal before direct-path validation is complete

## Direct-Output Debug Compare Invariants

For the same launch plan and routed tile, the legacy path and the direct path must match on:

- `packed_data`
- `block_scales_data`
- `matmul_block_scales_data`
- `activation_output_scale`

Small FP4 rounding ties are acceptable only if they are explained and bounded. The default expectation is byte-for-byte equality.
