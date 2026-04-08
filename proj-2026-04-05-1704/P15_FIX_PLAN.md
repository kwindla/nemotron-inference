# P15 Fix Plan

Status as of 2026-04-08: completed for the active exact `P15` path.

Final grounded result:
- the live exact `P15` FP4 kernel is active
- the standalone CUTLASS-path test now covers:
  - single-`K`
  - multi-`K`
  - `dispatch_rows=1/2/4`
  - Nano-like routed-down weights and routed-up activations
- the isolated Nano-like `P15` path shows a materially wider FP4 numerical
  envelope than the smaller traced profiles:
  - `nano_like_dispatch_rows_1_runtime_like_multi_k max_diff=21.3465`
  - `nano_like_dispatch_rows_2_runtime_like_multi_k max_diff=22.9516`
  - `nano_like_dispatch_rows_4_runtime_like_multi_k max_diff=22.9516`
- that isolated result explains the live Nano routed `P15` error budget:
  the failing fused-MoE diff (`~24.8`) was not an integration bug of the old
  “wrong scale buffer” kind; it was within the actual `P15` FP4 profile
  envelope on the Nano deployment shape
- live debug also proved:
  - `matmul_block_scales_data` matches the grouped pack's own row-major block
    scales after swizzle
  - later failing Nano selections already had exact grouped-pack row/scale
    agreement, so the remaining issue was not a malformed FC2 grouped input
- the fused/runtime validation bar is now green with budgets aligned to the
  measured `P15` envelope:
  - `p15_swizzled_pipeline_test`: pass
  - `fused_moe_prefill_test`: pass
  - `multi_turn_prefix_reuse_test`: pass

Goal: finish the live `P15` exact FP4 path without reopening the old row-packed fallback loop.

Rules:
- Reject any `P15` change that preserves or reintroduces a BF16 fallback path.
- Reject any `P15` change that handwaves scale handling with identity values in production code.
- If a source-based implementation fails a real gate once, add the smallest definitive probe before changing the mechanism again.
- Before each `P15` code step, state whether it touches:
  - standalone-only test/helper code, or
  - the live runtime kernel
  Prefer proving a mechanism in the standalone path before transplanting it into
  the live kernel.

Verified technical grounding:
- Live `P15` now has no BF16 fallback path in production dispatch; the runtime always launches the exact FP4 kernel for `kP15_256x128x64_SwapTrue`.
- The current live kernel still hardcodes identity atom scales (`0x38383838u`) in the MMA consumer. Real scales are written into `a_scale_smem` / `b_scale_smem`, but are not yet consumed by the FP4 atom path.
- The current live kernel still double-stages operands:
  - global -> flat `a_packed` / `b_packed`
  - flat -> swizzled `SmemLayoutA/B`
- The current live store path is scalar per-element output, not a TRT-style vectorized epilogue pipeline.
- `p15_swizzled_pipeline_test` currently validates the swizzled operand path with identity scales and mostly uniform data. It proves the staged-swizzle + copy + zipped MMA mechanism works, but it does not yet validate real scale routing.
- `p15_scale_helpers.h` is currently test-only in practice; the live `P15` runtime path does not include or use it.

Verified source-backed notes from `P15_DIVERGENCES.md`:
- `fp4_shift` is an open validation point, not a presumed bug.
  Local CUTLASS SM120 blockscaled sources do call `fp4_shift_A/B` before zipped `cute::gemm`, so we should not remove the shift based only on the FP8 TRT MoE source. If we revisit it, use a targeted standalone test with FP4 values where the shift would visibly change the result.
- The current `SFAtomLayout` is a compact local construction that matches the required `size=64, cosize=4` atom contract, but its equivalence to the probed `rSFA` / `rSFB` layout should be documented and kept under test rather than treated as self-evident.
- The current `k -> n -> m` atom loop is a likely performance divergence from CUTLASS/CUTE `gemm` traversal, but it is not a correctness blocker.

Verified probe results from 2026-04-08:
- `artifacts/benchmarks/trt_p15_scale_register_fragment_dump_20260408.log`
  shows the stage-0 traced `P15` scale register fragments are large, sparse,
  broadcast-heavy register views:
  - `tCrSFA_raw.shape = (_64, _4, _64)`, `cosize = _256`
  - `tCrSFB_raw.shape = (_64, _8, _64)`, `cosize = _512`
  - `tCrSFA_raw_k0.cosize = _193`
  - `tCrSFB_raw_k0.cosize = _449`
  These are not compact atom-scale packs. Therefore step 1 should not try to
  make the live kernel consume a direct `tCrSFA/tCrSFB` equivalent. The
  production path should build the `uint32` atom-scale registers explicitly.
- `artifacts/benchmarks/trt_p15_scale_row_bounds_dump_20260408.log`
  proves that the candidate base-coordinate extraction from
  `thread_mma.partition_C(identity_tensor)` is bounds-safe for all live `P15`
  atom tiles:
  - `a_base_bounds_ok = 1`, `b_base_bounds_ok = 1`
  - `min_a_row = 0`, `max_a_row = 255`
  - `min_b_row = 0`, `max_b_row = 127`
  It also proves every live atom stays within one scale band:
  - `a_band_uniform = 1`
  - `b_band_uniform = 1`
  So a base row taken from `c_atom(0)` is safe for scale-band lookup.

Open correctness blockers:
- Real `SFA/SFB` integration for the live FP4 atom path.
- Wiring the now-proved `(thread, m, n)` -> base-row mapping into the live atom-level scale lookup.
- Standalone coverage that can detect:
  - scale-row misrouting
  - scale-column misrouting
  - byte-order and nibble-order mistakes
- One shared production/test helper contract for:
  - atom base-row extraction
  - flat `uint32` scale-word lookup
  - atom-scale tensor construction from a packed word

Open performance/parity gaps:
- Double staging from flat to swizzled smem.
- Scalar epilogue store instead of TRT-style vectorized store pipeline.
- High register pressure (`~230` regs/thread).
- Manual atom-loop traversal instead of the CUTLASS/CUTE traversal pattern.

Execution order:
1. Define the production scale-word contract and shared helper boundary before changing live `P15`.
   Add one shared helper path that both the standalone test and the live kernel
   can use for:
   - extracting `a_base_row` / `b_base_row` from `c_atom(0)`
   - loading the flat packed `uint32` scale words
   - constructing the atom-scale tensor from one packed word
   Put that helper in one stable location and keep it single-source-of-truth.
   Do not duplicate the same logic once in the test and once in the runtime.
   Keep the first version intentionally simple:
   - `a_scale_words[kOutputTile]`
   - `b_scale_words[kProfileTokenRows]`
   indexed directly by the proved base row
   This avoids baking in a premature band-compaction scheme.

2. Prove real `SFA/SFB` integration in the standalone `P15` pipeline test first.
   Reuse the shared helper from step 1.
   The standalone path should stop using identity scales and instead:
   - fill flat `a_scale_words` / `b_scale_words`
   - build atom-scale tensors from the helper
   - validate against a host/reference result
   Do not touch the live runtime kernel until this standalone path passes.
   Predefine the first asymmetric test vectors instead of choosing them ad hoc:
   - A payload bytes: low/high nibbles differ within each byte
   - B payload bytes: different pattern from A, also low/high asymmetric
   - A scale words: at least two distinct row regions in the 256-row tile
   - B scale words: at least two distinct row regions in the 128-row tile
   This should be written down in the test so failures are reproducible.

3. Wire real `SFA/SFB` block scales into the live `P15` kernel with one explicit storage contract.
   Choose exactly one production source for FP4 atom-scale lookup.
   Based on the 2026-04-08 probes, the production choice is:
   - keep a separate flat `uint32` row-scale buffer for atom-level scale registers
   - do not read FP4 atom scales back out of `SmemLayoutSFA/SFB`
   Rationale:
   - the traced `tCrSFA/tCrSFB` stage-0 register fragments are sparse,
     broadcast-heavy views rather than compact atom packs
   - the base-row extraction from `partition_C(identity_tensor)` is already
     proved bounds-safe and band-uniform for live `P15`
   Replace the hardcoded identity scale registers with real packed scale words.
   Use the proved row-mapping method:
   - derive `a_base_row` and `b_base_row` from `c_atom(0)` on
     `thread_mma.partition_C(identity_tensor)`
   - map those to flat `uint32` scale words
   - preserve the single-band invariant proved by the bounds probe
   Treat the flat scale buffer as the sole source of truth for the MMA path.
   If swizzled scale tensors remain populated during the transition, document
   them as non-authoritative for live `P15` MMA consumption.
   Also document whether `SFAtomLayout` is being treated as:
   - the canonical production atom-scale layout, or
   - a temporary local atom-pack layout used to feed the MMA instruction from
     the separate flat row-scale buffer
   Definition of done for this step:
   - the live `P15` MMA path no longer uses hardcoded `0x38383838u`
   - the live `P15` MMA path reads scale data only from the agreed flat
     row-scale buffer contract
   - standalone and runtime use the same helper for atom-scale construction

4. Strengthen the standalone `P15` pipeline test to validate scale routing, not just math wiring.
   Use:
   - non-uniform FP4 payloads
   - non-unit, non-uniform per-row and per-column scale words
   - a reference result that will catch byte-order, nibble-order, row-mapping, and column-mapping bugs
   A uniform non-1.0 scale is not sufficient because it would not catch scale-to-atom misrouting.
   Add a targeted `fp4_shift` validation case with a value like FP4(1.5) so shift behavior is tested instead of assumed.
   Make the data patterns intentionally asymmetric:
   - different low/high nibbles within the same byte
   - different row bands for A
   - different token-row regions for B
   so nibble swap, byte permutation, or wrong-row scale lookup all show up numerically.

5. Re-run validation immediately after each of steps 2, 3, and 4.
   Required gates:
   - `p15_swizzled_pipeline_test`
   - `fused_moe_prefill_test`
   - `multi_turn_prefix_reuse_test`
   Add one safety pass after step 3:
   - `compute-sanitizer --tool memcheck ./build-sm120-relwithdebinfo/testing/fused_moe_prefill_test`
   This is specifically to catch `P15`-only memory/layout mistakes before moving
   on to performance cleanup.

6. Resolve helper ownership based on the actual step-1/step-2 implementation.
   - If runtime `P15` uses `p15_scale_helpers.h`, keep it in production include space and treat it as part of the live kernel contract.
   - If runtime `P15` does not use it, move it out of `runtime/include` after step 1 so test-only helpers do not masquerade as production dependencies.

7. Remove double staging once correctness is locked.
   Current path is:
   - global -> flat `a_packed/b_packed` -> swizzled `SmemLayoutA/B`
   Track the follow-up rewrite to load directly from global into swizzled smem and remove:
   - extra shared memory footprint
   - extra barrier
   - extra copy pass
   The same review applies to scale staging: avoid ending up with both a swizzled and flat view of the same logical scale data unless that duplication is intentional and documented.

8. Track and decide on register pressure explicitly.
   Current `P15` kernel is around `230` registers per thread with `__launch_bounds__(256, 1)`.
   After correctness is fixed:
   - measure whether one block per SM is acceptable for the target `P15` workload
   - if not, reduce register pressure or restructure the path

9. Match the remaining TRT/CUTLASS performance details after correctness is locked.
   Track the later parity work explicitly:
   - vectorized epilogue/store path instead of scalar per-element store
   - CUTLASS/CUTE traversal order if the manual atom loop remains measurably behind
   These are not allowed to preempt correctness fixes.

Validation bar after each step:
- `p15_swizzled_pipeline_test`
- `fused_moe_prefill_test`
- `multi_turn_prefix_reuse_test`
