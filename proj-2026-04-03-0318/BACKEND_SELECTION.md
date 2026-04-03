# Backend Selection

This note documents the MoE backend order after step 5 and the decode comparison workflow for step 6.

## Priority Order

`runtime/src/backend/expert_layer.cpp` initializes backends in this order:

1. `UnifiedFusedBackend`
2. `DecodeCublasLtBackend`
3. `FusedDecodeBackend`
4. `BatchedCublasLtBackend`

The selector walks that list in order and uses the first backend whose `Supports(...)` check passes and whose `Run(...)` call succeeds.

## Backend Roles

| Backend | Typical token count | Notes |
| --- | --- | --- |
| `UnifiedFusedBackend` | `>= 1` | Unified routed-expert path for decode and prefill. Preferred fast path when explicitly enabled and residency requirements are met. |
| `DecodeCublasLtBackend` | `== 1` | Decode-only cuBLASLt-per-expert path. Sits ahead of the scalar kernel when unified fused is not selected. |
| `FusedDecodeBackend` | `== 1` | Existing scalar/shared-memory decode kernel. This is the kernel being judged by step 6. |
| `BatchedCublasLtBackend` | `> 1` | Multi-token fallback path. Also the only path with built-in `moe_profile:` timing today. |

## Env Vars

Primary steering vars:

| Env var | Effect |
| --- | --- |
| `NEMOTRON_FORWARD_UNIFIED_FUSED` | Preferred unified-fused opt-in. If set to a non-zero value, `UnifiedFusedBackend` is enabled for all supported token counts. If set to `0`, unified fused is disabled even if legacy opt-ins are present. |
| `NEMOTRON_FORWARD_FUSED_MOE_PREFILL` | Legacy unified-fused opt-in. Only consulted when `NEMOTRON_FORWARD_UNIFIED_FUSED` is unset or empty. Kept for backward compatibility. |
| `NEMOTRON_FORWARD_MOE_CUBLASLT` | Enables `DecodeCublasLtBackend` when unset or non-zero. Set to `0` to force the selector past cuBLASLt and onto the scalar decode kernel. |
| `NEMOTRON_FORWARD_FUSED_MOE_DECODE` | Enables the decode fused-MoE fast-path family. The existing decode benchmarks set this to `1` internally. |

Availability gates that can still prevent unified fused selection:

| Env var | Effect |
| --- | --- |
| `NEMOTRON_EXPERT_MONOLITHIC` | Controls monolithic resident expert weights. Unified fused requires monolithic residency or full residency. |
| `NEMOTRON_EXPERT_FULL_RESIDENCY` | Controls full routed-expert residency. If both this and monolithic residency are unavailable, unified fused cannot run. |
| `NEMOTRON_FORWARD_DEBUG` | Useful when a forced backend unexpectedly falls through to a lower-priority backend. The runtime prints failure/fallback messages on backend run failure. |

Important: unified-fused opt-in is consumed during expert-layer construction, not only during dispatch. Export `NEMOTRON_FORWARD_UNIFIED_FUSED` or `NEMOTRON_FORWARD_FUSED_MOE_PREFILL` before launching the process.

## Force Specific Backends

These commands assume `nano_fused_decode_bench`, which already enables `NEMOTRON_FORWARD_FUSED_MOE_DECODE=1` internally.

Force the scalar decode kernel:

```bash
env \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0 \
  NEMOTRON_FORWARD_MOE_CUBLASLT=0 \
  ./build/benchmarks/nano_fused_decode/nano_fused_decode_bench \
  --manifest /path/to/manifest.json \
  --mode=steady-state \
  --decode-tokens 16
```

Force the unified fused backend with the new env var:

```bash
env \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  NEMOTRON_FORWARD_MOE_CUBLASLT=1 \
  ./build/benchmarks/nano_fused_decode/nano_fused_decode_bench \
  --manifest /path/to/manifest.json \
  --mode=steady-state \
  --decode-tokens 16
```

Force the unified fused backend with the legacy env var spelling:

```bash
env \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1 \
  NEMOTRON_FORWARD_MOE_CUBLASLT=1 \
  ./build/benchmarks/nano_fused_decode/nano_fused_decode_bench \
  --manifest /path/to/manifest.json \
  --mode=steady-state \
  --decode-tokens 16
```

Force the decode cuBLASLt backend:

```bash
env \
  NEMOTRON_FORWARD_UNIFIED_FUSED=0 \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=0 \
  NEMOTRON_FORWARD_MOE_CUBLASLT=1 \
  ./build/benchmarks/nano_fused_decode/nano_fused_decode_bench \
  --manifest /path/to/manifest.json \
  --mode=steady-state \
  --decode-tokens 16
```

`proj-2026-04-03-0318/bench_decode_backends.sh` runs exactly this three-way matrix and writes one JSON artifact plus stdout log per backend configuration.

## What To Measure

Primary step-6 metric:

- decode latency for `token_count == 1`

Recommended measurement method:

- Run `nano_fused_decode_bench` in `steady-state` mode.
- Use repeated `ContinueSingleToken(...)` calls after a fixed prompt prefill.
- Compare `benchmark.hot_steady_state_mean_ms` across the three forced backend configurations.

Why `steady-state` is acceptable for `token_count == 1`:

- each measured decode step is still a single-token decode call
- using multiple steps reduces noise versus a single profile-ready step

Trace-friendly alternative:

- `--mode=profile-ready` gives one prefill plus one measured decode step and is useful for Nsight capture

Expert-layer timing:

- `NEMOTRON_FORWARD_MOE_PROFILE=1` currently emits `moe_profile:` timings only for `BatchedCublasLtBackend`
- there is no equivalent per-layer internal timer for `UnifiedFusedBackend`, `DecodeCublasLtBackend`, or `FusedDecodeBackend`
- for those backends, use decode latency from the benchmark plus Nsight Systems or Nsight Compute when deeper attribution is needed

## Decision Rule

Use the scalar decode kernel only if it earns its complexity.

- If `UnifiedFusedBackend` matches or beats `FusedDecodeBackend` on decode latency, the scalar decode kernel is a removal candidate.
- If `FusedDecodeBackend` still wins, keep it and document the measured gap.
- Keep `DecodeCublasLtBackend` in the comparison matrix as a third reference point, but the step-6 keep/remove decision is specifically unified fused versus the scalar decode kernel.
