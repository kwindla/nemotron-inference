# SM120 Backend Decision Gate

This step is a gate, not a kernel milestone. A custom SM120 backend is justified only if the unified fused path is already correct, the benchmark and correctness infrastructure is in place, and measured results still show a clear residual MoE gap versus vLLM `flashinfer_cutlass` on RTX 5090.

## Decision Target

If this gate opens, the target is narrow:

- build a fifth `MoeBackend` implementation for consumer Blackwell (`SM120`)
- close the residual MoE execution gap that remains after steps 1-7
- reuse the existing GPU `topk_ids` / `topk_weights` contract, prepared NVFP4 weights, and unified backend selection surface

This is not a license to replace the MoE stack, change routing, or create a separate architecture.

## Prerequisites Before Starting Custom Kernel Work

Do not start owned SM120 kernel work until all of the following are true:

1. The unified fused backend is correct and has been verified by `proj-2026-04-03-0318/verify_correctness.sh`.
2. Load-time weight preparation from step 3 is already in place and is the fast-path input surface.
3. Chunking is runner-level from step 4, not embedded inside a separate prefill-only kernel architecture.
4. The step-7 benchmark comparison against vLLM has been completed through the existing harness, with `proj-2026-04-03-0318/bench_full_comparison.sh` available as the measurement path.
5. The measured results show a reproducible performance gap between our unified fused backend and vLLM `flashinfer_cutlass`.

If the comparison artifacts are not complete enough to localize the residual gap, this gate stays closed.

## Decision Criteria

A custom SM120 backend is justified only if every point below is true:

1. The unified fused backend from step 5 still leaves a measurable, repeatable gap versus vLLM on the target RTX 5090 workloads.
2. The remaining gap is in MoE execution itself, not in routing, non-MoE layers, chunk orchestration, cache behavior, or other benchmark differences.
3. Routing histogram data shows which `M` buckets dominate the real workload, so any specialization is driven by measured occupancy rather than guessed bucket cutoffs.

This means:

- no custom backend if unified fused has already closed the practical gap
- no custom backend if the residual gap is outside MoE
- no custom backend if histogram data is too weak to justify bucket-specific design

## Scoping Constraints From The Plan

If this gate opens, scope the work narrowly:

1. Target the residual performance gap, not a full rewrite of the routed-expert path.
2. Use the real measured `M` histogram to choose which buckets deserve specialization.
3. Treat `M>=5` as a streamed or tiled regime. On SM120, the plan's Nano sizing math already puts `M * (2688 + 1856) * 4` bytes near the per-block shared-memory limit at `M=5`.
4. Plug the backend into the unified `MoeBackend` surface from `runtime/include/nemotron/moe_backend.h`, which means working through the existing `PrepareWeights(...)`, `Supports(...)`, and `Run(...)` contract.
5. Respect consumer Blackwell SM120 limits and assumptions:
   - no `tcgen05`
   - no TMEM
   - `128 KB` shared memory per SM
   - `99 KB` shared memory per block

The intended specialization target is therefore bucket-aware routed-expert execution on real SM120 bottlenecks, not a new end-to-end MoE architecture.

## Integration Path

The integration path should stay inside the existing backend framework:

1. Add the custom implementation as a fifth `MoeBackend`.
2. Give it highest priority when it is available and `Supports(...)` passes for the active configuration.
3. Feed it the same GPU top-k inputs and prepared weights already used by the unified fused backend.
4. Fall back to `UnifiedFusedBackend` when the custom backend does not support the configuration or cannot run successfully.

Current backend order is documented in `proj-2026-04-03-0318/BACKEND_SELECTION.md`. A custom SM120 backend should extend that order, not bypass it.

## What Not To Do

1. Do not start kernel work before benchmark data from step 7 exists and is good enough to explain the residual gap.
2. Do not optimize routing. GPU top-k routing is already the primary contract from step 2.
3. Do not change weight layout. Use the prepared weights from step 3.
4. Do not create a separate architecture. Plug into the existing unified backend surface.

## Go / No-Go Rule

Use this rule for the step-8 decision:

- `GO`: benchmark data shows a clear residual gap versus vLLM, attribution says the gap is in MoE execution, and histogram data identifies the dominant `M` buckets that should drive a narrow SM120 backend.
- `NO-GO`: any missing prerequisite, missing attribution, missing histogram support, or no material residual MoE gap.

If the answer is `NO-GO`, continue improving measurement quality or tuning the unified fused backend rather than opening custom kernel work.

## Reference Points

- `proj-2026-04-03-0318/PLAN.md`
- `proj-2026-04-03-0318/BACKEND_SELECTION.md`
- `runtime/include/nemotron/moe_backend.h`
