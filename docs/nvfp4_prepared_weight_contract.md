# NVFP4 Prepared-Weight Contract

## Ownership

Load-time NVFP4 state is split by responsibility, not by convenience:

- Manifest-backed `GemmDescriptor` data is the immutable source payload:
  packed weight bytes, row-major block-scale bytes, and tensor-scale bytes.
- `MonolithicNvfp4ExpertWeights` owns resident routed-expert storage for MoE
  backends. Fast routed-expert backends prepare their `FusedNvfp4WeightView`
  arrays once from resident storage and then reuse those views for execution.
- `DeviceNvfp4Weight` remains a first-class owner for shared experts and
  generic non-MoE NVFP4 linear paths. Those callers still consume
  device-resident per-weight uploads directly.

## Fast-Path Boundary

Prepared routed-expert execution is load-time only:

- fast routed-expert backends may consume only resident/prepared routed views
- they must not upload routed experts on demand during `Run()`
- if routed views are not prepared, the backend rejects and the layer falls
  back to the slower host/reference path explicitly

This keeps the production MoE path aligned with the vLLM prepared-weight model:
weight mutation happens before execution, not inside the token loop.

## Backend Caches

The allowed backend-local caches are narrow:

- host-side `std::vector<FusedNvfp4WeightView>` tables derived from resident
  routed weights
- optional device copies of those view tables for fused kernels

Those caches are backend metadata only. They do not own the underlying routed
weight bytes.

## Fallback-Only State

The following are permitted only outside the prepared fast path:

- host-side NVFP4 dequantization through `RunNvfp4LinearHost()`
- fallback execution that reads routed expert descriptors directly
- any routed-expert path reached after a backend explicitly rejects

## Manifest Assumption

Manifest generation must point at packed NVFP4 weight bytes and explicit
auxiliary scale byte ranges. The runtime may swizzle block scales into
execution layout, but the manifest contract itself is already byte-oriented:
packed weights are in `packed_file`, block scales and tensor scales are in
named `auxiliaries`, and fast paths do not reinterpret raw logical fp32
weights at runtime.
