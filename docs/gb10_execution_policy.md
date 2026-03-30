# GB10 Execution Policy

This note is intentionally short. It captures the current operator-format policy implied by the existing CUDA `13.2` benchmark artifacts and should be revised only when new benchmark coverage or correctness data changes the result.

Primary evidence:

- [artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_gemm_compare_20260329T014933Z_device.md)
- [docs/gb10_performance_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_performance_plan.md)

## Current Policy

### Use NVFP4 By Default

These families are already strong NVFP4 candidates on the current GB10 runtime-facing path:

- attention core projections matching the measured `4096 x 4096` family
- shared-expert `up_proj` / `down_proj` families matching the measured `5376 x 4096` and `4096 x 5376` shapes
- routed-expert GEMMs only where the measured family and service batch shape are known to benefit

Current evidence:

- `39 / 45` matched comparisons favor NVFP4 on pure GEMM hot latency
- `35 / 45` matched comparisons favor NVFP4 on runtime-facing hot latency
- strongest current win:
  - `shared_expert_up`, `m=256`, `workspace=0`
  - about `21.7x` compute-only speedup
  - about `12.8x` runtime-facing speedup

### Keep Higher Precision Or Case-Specific Policy

Do not apply a blanket NVFP4 policy yet to:

- the smallest `mtp_expert_up` / `mtp_expert_down` cases, especially `m=1`
- Mamba projection families
- attention KV projection families

Reasons:

- the smallest measured MTP cases still lose once activation staging is included
- Mamba and attention-KV families do not yet have matching NVFP4 runtime-path benchmark coverage

For now, treat these as `BF16 / FP32 correctness-first` or `benchmark-required before switching`.

## Attention Policy

For attention on GB10:

- `cuDNN Frontend` paged SDPA is the required v1 backend
- KV cache should target `BF16` pages first on GB10
- do not plan around FP8 attention on `SM121`
- the first BF16 cuDNN FE paged-SDPA decode path is now executing in the runtime and validated against a CPU reference; widen from that path rather than introducing a second attention backend
- do not hard-pin a single cuDNN SDPA implementation yet; the current runtime should allow cuDNN to auto-select the supported path for the shape

## Mamba Cache Policy

**Decision**: use `FP16 + stochastic rounding` as the production optimized-cache format. Accept the known first-user-tail weakness documented below. Re-assess when production traces are available.

### Format Summary

- `FP32`: correctness baseline
- `FP16 + stochastic rounding (fp16_sr)`: production optimized format
- `FP16` (plain truncation): rejected — worse mean drift on most phases
- grouped `INT8`: rejected — quantize/dequantize overhead erases the bandwidth win
- software Philox-style stochastic rounding is the `SM121` baseline (no hardware `cvt.rs.f16.f32` on CUDA `13.2`)

### Overall Advantage

Across a six-profile oracle-derived family (narrative, structured, tool-style, markdown-heavy, flat, headerless), `fp16_sr` beats plain `fp16` on mean drift at the majority of phase boundaries:

- assistant-decode phases: `fp16_sr` advantage roughly `1.9x-3.8x`
- later user-tail phases: `fp16_sr` advantage roughly `1.5x-4.1x`
- first user-tail phase: **mixed**, see below

### Known Weakness: First User-Tail Phase

The first user-tail phase (`user_turn_1_tail_prefill`) can show `fp16_sr` performing slightly worse than plain `fp16` on some profiles. This is the accepted weakness.

Observed guardrail anchors:

| Metric | Value |
|---|---:|
| Weakest `fp16/fp16_sr` ratio floor | `0.848009` at `8` requests (markdown-flat profile) |
| Weakest `fp16_sr` mean drift at that point | `0.000944` |
| Weakest `fp16` mean drift at that point | `0.000801` |
| Both formats' max absolute drift at that point | < `0.001` mean, < `3.1` max spike |

Both formats produce very small absolute drift. The regression is measurable but almost certainly below the threshold of observable model quality impact.

### Root Cause Of The Weakness

The regression correlates with the `dt` operand-distribution in the first user-tail phase. Profiles that spend more time in high-dt regimes (`>p75` share above ~0.3) tend to show the regression. However:

- the response to `dt` scaling is **non-monotonic** (1.25x dt scale *improves* the ratio while 1.0x and 1.5x are worse), so it is not a simple amplitude effect
- no single token pattern (bullets, headings, think-tags) is the root cause — controlled variants removing each pattern still show the weakness
- global `dt` threshold gating (SR→RN when `|dt| > 0.65`) improves the weak phase but causes unacceptable regressions in later phases (worst delta `-0.71`)

The weakness is a complex operand-distribution interaction specific to the early conversation state, not a correctable single-parameter defect.

### Key Artifacts For Re-Assessment

- Latest guardrail report: [gb10_mamba_trace_guardrails_20260329T152925Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_guardrails_20260329T152925Z.md)
- Latest operand report: [gb10_mamba_trace_operands_20260329T152925Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_operands_20260329T152925Z.md)
- dt-bucket analysis: [gb10_mamba_trace_dt_buckets_20260329T163241Z.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_dt_buckets_20260329T163241Z.md)
- dt-scale sweep (non-monotonicity evidence): [gb10_mamba_trace_dt_scale_sweep_20260329T164125Z_cuda132.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_trace_dt_scale_sweep_20260329T164125Z_cuda132.md)
- Threshold gating rejection: [gb10_mamba_dt_threshold_0p65_family_20260329T165952Z_cuda132.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_mamba_dt_threshold_0p65_family_20260329T165952Z_cuda132.md)
- Benchmark supports `--dt-threshold` for future selective-gating experiments

### When To Re-Assess

- When production traces are available to validate whether the synthetic-profile weakness reproduces under real serving load
- If a per-step or phase-local gating rule can improve the weak phase without later-phase cost
- If a new SM121 hardware path for stochastic rounding becomes available

## How To Use This Note

This note is for short-term execution choices:

- use it when deciding what operator family to wire into NVFP4 first
- use it when deciding which families stay correctness-first
- update it only when:
  - a new comparison artifact materially changes the result
  - a new operator family gets matched benchmark coverage
  - runtime correctness findings invalidate the current fast path
