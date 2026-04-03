# Existing 32-Token Baseline Provenance

Artifact inspected: `proj-2026-04-03-0318/moe_profile_baseline.log`

## What It Is

This log is produced by the internal runtime's expert-layer profiling path in:

- `runtime/src/backend/expert_layer.cpp`
- `PrintMoeProfileHistogram(...)`
- `PrintMoeProfileTimings(...)`

It is only emitted when `NEMOTRON_FORWARD_MOE_PROFILE=1` is enabled.

Each MoE layer contributes:

1. one histogram line with active-expert bucket counts
2. one timing line with `selection`, `routing`, `gemm_loop`, `shared`, `residual`, `alloc`, and `total`

So this artifact is an internal per-layer MoE microprofile, not an end-to-end request benchmark.

## What The File Contains

- `464` log lines total
- `232` per-layer records
- `10` complete MoE passes of `23` routed-MoE layers each
- `1` additional truncated pass that stops after layers `1` and `3`

The complete passes split into two repeated groups:

- passes `0-4`: layer-1 histogram `M=1:0 M=2:5 M=3:5 M=4+:118`
- passes `5-9`: layer-1 histogram `M=1:0 M=2:0 M=3:0 M=4+:128`

That means the file is already aggregating more than one workload shape, but it does not record the command line, benchmark case name, token count, prompt length, cache state, or whether the run was warmup vs measured.

## Why It Is Not A Direct vLLM-Comparable 32-Token Baseline

This artifact is not directly comparable to the planned vLLM TTFT baseline for three independent reasons.

First, it is not end-to-end TTFT. It only times the MoE layer's internal work. It excludes:

- request submission
- cache lookup / restore
- non-MoE layers
- logits / sampler work
- first-token decode completion as a wall-clock event

Second, the file does not preserve enough provenance to prove which benchmark case produced it. There is no attached command, no prefix length, and no benchmark summary file next to it.

Third, the histogram itself strongly suggests that the logged shape is not a `32`-token routed workload. For Nemotron Nano, `top_k=6`, so a true `32`-token MoE pass has `32 * 6 = 192` routed selections total. But the first repeated layer-1 histogram has a hard lower bound of:

- `5 * 2 + 5 * 3 + 118 * 4 = 497` routed selections

and the second repeated layer-1 histogram has a hard lower bound of:

- `128 * 4 = 512` routed selections

Both lower bounds are far above `192`, so this file cannot be a literal `32`-token routed-MoE capture.

## Conclusion

The current `moe_profile_baseline.log` should be treated as:

- a useful historical internal MoE-layer microprofile
- evidence that the host-routed per-expert GEMM loop dominates our current runtime

It should not be treated as:

- the authoritative `32`-token baseline
- an apples-to-apples external comparison against vLLM
- a complete decode or TTFT benchmark artifact
