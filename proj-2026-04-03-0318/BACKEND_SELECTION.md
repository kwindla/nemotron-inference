# Backend Selection

This note documents the MoE backend order after collapsing the decode surface around unified fused and the remaining decode comparison workflow.

## Priority Order

`runtime/src/backend/expert_layer.cpp` initializes backends in this order:

1. `UnifiedFusedBackend`
2. `DecodeCublasLtBackend`
3. `BatchedCublasLtHostRoutingAdapterBackend`

The selector walks that list in order and uses the first backend whose `Supports(...)` check passes and whose `Run(...)` call succeeds.

## Backend Roles

| Backend | Typical token count | Notes |
| --- | --- | --- |
| `UnifiedFusedBackend` | `>= 1` | Unified routed-expert path for decode and prefill. This is now the default resident path when not explicitly disabled. |
| `DecodeCublasLtBackend` | `== 1` | Decode-only cuBLASLt-per-expert path. Kept as a secondary single-token fallback because it still beats unified fused on the RTX 5090 Nano decode benchmark. |
| `BatchedCublasLtHostRoutingAdapterBackend` | `> 1` | Host-routing adapter fallback for resident batched MoE. This is intentionally last-resort behind the unified path. |

## Env Vars

Primary steering vars:

| Env var | Effect |
| --- | --- |
| `NEMOTRON_FORWARD_UNIFIED_FUSED` | Preferred unified-fused control. If set to a non-zero value, `UnifiedFusedBackend` is enabled. If set to `0`, unified fused is disabled even if legacy env spellings are present. When unset, unified fused now defaults to enabled behavior. |
| `NEMOTRON_FORWARD_FUSED_MOE_PREFILL` | Legacy unified-fused env spelling. Only consulted when `NEMOTRON_FORWARD_UNIFIED_FUSED` is unset or empty. If both vars are unset, unified fused still defaults to enabled. |

Availability gates that can still prevent unified fused selection:

| Env var | Effect |
| --- | --- |
| `NEMOTRON_EXPERT_MONOLITHIC` | Controls monolithic resident expert weights. Unified fused requires monolithic residency or full residency. |
| `NEMOTRON_EXPERT_FULL_RESIDENCY` | Controls full routed-expert residency. If both this and monolithic residency are unavailable, unified fused cannot run. |
| `NEMOTRON_FORWARD_DEBUG` | Useful when a forced backend unexpectedly falls through to a lower-priority backend. The runtime prints failure/fallback messages on backend run failure. |

Important: unified-fused enable/disable is consumed during expert-layer construction, not only during dispatch. If neither env var is set, the resident path still comes up unified by default.

## Force Specific Backends

These commands assume `nano_fused_decode_bench`, which already enables `NEMOTRON_FORWARD_FUSED_MOE_DECODE=1` internally.

Force the unified fused backend with the new env var:

```bash
env \
  NEMOTRON_FORWARD_UNIFIED_FUSED=1 \
  ./build/benchmarks/nano_fused_decode/nano_fused_decode_bench \
  --manifest /path/to/manifest.json \
  --mode=steady-state \
  --decode-tokens 16
```

Force the unified fused backend with the legacy env var spelling:

```bash
env \
  NEMOTRON_FORWARD_FUSED_MOE_PREFILL=1 \
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
  ./build/benchmarks/nano_fused_decode/nano_fused_decode_bench \
  --manifest /path/to/manifest.json \
  --mode=steady-state \
  --decode-tokens 16
```

The latest removal decision is based on `proj-2026-04-03-0318/artifacts/decode_backends/20260404T025731Z_pre_step4`:

- `fused_decode`: `1836.737 ms`
- `unified_fused`: `75.453 ms`
- `decode_cublaslt`: `68.872 ms`

## What To Measure

Primary step-6 metric:

- decode latency for `token_count == 1`

Recommended measurement method:

- Run `nano_fused_decode_bench` in `steady-state` mode.
- Use repeated `ContinueSingleToken(...)` calls after a fixed prompt prefill.
- Compare `benchmark.hot_steady_state_mean_ms` across the remaining forced backend configurations.

Why `steady-state` is acceptable for `token_count == 1`:

- each measured decode step is still a single-token decode call
- using multiple steps reduces noise versus a single profile-ready step

Trace-friendly alternative:

- `--mode=profile-ready` gives one prefill plus one measured decode step and is useful for Nsight capture

Expert-layer timing:

- `NEMOTRON_FORWARD_MOE_PROFILE=1` currently emits `moe_profile:` timings only for `BatchedCublasLtBackend`
- there is no equivalent per-layer internal timer for `UnifiedFusedBackend` or `DecodeCublasLtBackend`
- for those backends, use decode latency from the benchmark plus Nsight Systems or Nsight Compute when deeper attribution is needed

## Decision Rule

Use a legacy decode-only backend only if it earns its complexity.

- `FusedDecodeBackend` was removed after losing badly to unified fused on the RTX 5090 Nano decode benchmark.
- `DecodeCublasLtBackend` remains because it still beats unified fused for `token_count == 1` on that benchmark.
