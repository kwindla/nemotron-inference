# NanoP1 Phase 3 — 2026-04-14 synopsis

The full April 14 probe transcript was archived because it had grown into a
long probe-by-probe log that was no longer a good handoff surface:

- full transcript:
  [nano_p1_phase3_session_full_2026-04-14.md](/home/khkramer/src/nemotron-inference/proj-2026-04-12-1022/probes/archive/2026-04-14-probe-chain/nano_p1_phase3_session_full_2026-04-14.md)
- probe-capture archive:
  [2026-04-14-probe-chain](/home/khkramer/src/nemotron-inference/proj-2026-04-12-1022/trtllm_reference/archive/2026-04-14-probe-chain)

## Current state

- Phase 1 synthetic all-ones still passes.
- The old flashinfer compatibility metric is still the same structured partial
  match: `4074 / 245760` bitwise matches against
  `golden_nano_k2688/bf16_gemm1_tactic1.bin`.
- A new local Phase 3 math oracle now exists in
  [nano_p1_mainloop_oracle_test.cpp](/home/khkramer/src/nemotron-inference/testing/backend/nano_p1_mainloop_oracle_test.cpp:973).
- That oracle dequantizes the saved Nano execution payload
  (`input_fp4_permuted.bin`, `input_sf_permuted.bin`, `inputs_w1_fp4.bin`,
  `inputs_w1_sf.bin`) using the execution-scale layout, accumulates in FP64,
  applies `g1_alpha`, rounds to BF16, and compares with a `1`-BF16-ULP budget.

## Important interpretation

- The local math oracle is **not** a drop-in replacement for the flashinfer
  dump. On the current tree, local-math vs flashinfer is only
  `15977 / 245760` within `1` ULP.
- That means the local oracle is a deliberate change of primary success
  criterion, not a validated reproduction of the flashinfer tactic-1 contract.
- Even so, the local oracle still exposes the same bug family: it preserves the
  structured row/column pattern instead of collapsing into random mismatches.

## Current Phase 3 numbers

From the current `nano_p1_mainloop_oracle_test` run:

- local math oracle: `8514 / 245760` within `1` BF16 ULP
- flashinfer bitwise compatibility: `4074 / 245760`
- local-math vs flashinfer compatibility: `15977 / 245760` within `1` ULP

The new local histogram is still strongly structured:

- dense rows: `M={0,2,24,26,32,34,56,58,64,66,88,90,96,98,120,122}`
- row 0 within-tolerance columns in the first 32:
  `0,2,8,10,16,18,24,26`

So the kernel bug is still real; only the oracle changed.

## What was ruled out by the archived probe pass

- another simple row-only B permutation
- odd-family `k_block` swap at MMA time
- flipping the current B copy-selector specialization
- treating `fp4_shift_B` as the primary culprit

## 2026-04-14 addendum: oracle validation and gate restructure

The local math oracle was audited against the flashinfer dump via five
Python validations under
`proj-2026-04-12-1022/probes/oracle_validation_{1..5}_2026-04-14.py`.
Findings, in order:

1. **Cross-tactic convergence holds.** tactic0 vs tactic1 bf16 dumps are
   byte-identical (245760/245760), confirming
   `trtllm_reference/NOTES.md` §5.
2. **Direct FP64 `hs @ w1_16.T` (no alpha) vs flashinfer reaches ~49%
   within 512 ULP, ~15% within 64 ULP, bitwise 0.15%.** This is the FP4
   precision floor: any FP64 re-implementation of the same dot product
   hits this same ceiling against an FP4 hardware result.
3. **The oracle math is correct.** `dq(fp4) @ dq(fp4).T * alpha`
   reaches ~46% within 512 ULP vs `hs @ w1_16.T`, essentially the same
   ceiling as FP16-direct. The oracle-vs-flashinfer 6.5% within 1 ULP
   number is not an oracle bug — it is the FP4→BF16 rounding floor.
4. **Row permutation confirmed (and the runtime's inverse formula is
   correct).** `input_fp4_permuted.bin` row k holds `hs[(k%8)*4 + k//8]`
   within each 32-row block. The kernel's
   `source_row = (a_in_blk%4)*8 + a_in_blk/4` at
   `fused_moe_prefill.cu:12331-12338` is the correct forward map from
   logical token → on-disk position. Rows 0 and 31 are the two fixed
   points of this permutation, which is why they show `0.045`
   L∞-error against identity — everything else shows ~0.6 (structural
   mismatch) unless compared against the `invSrcToDst` candidate,
   which matches at FP4 precision.
5. **`alpha * w1_gs ≈ 1`** (diff 2.6e-8). Alpha is exactly `1/w1_gs`, so
   the oracle's post-GEMM alpha multiplication exactly un-does the
   per-block scale that was absorbed into `dq_weight`.

### Consequent test-gate restructure

`testing/backend/nano_p1_mainloop_oracle_test.cpp` has been updated so
the Phase 3 primary gate is **runtime vs flashinfer bitwise**, not
runtime vs the local oracle within 1 ULP. The prior 1-ULP gate on the
local oracle was measuring FP4→BF16 precision noise and would fail on
*any* FP4 kernel, correct or not.

- New gate constants (near the top of the file):
  - `kPhase3FlashinferMaxAllowedBitwiseMismatches = 0` — strict primary
    gate: runtime must reproduce flashinfer tactic1 byte-for-byte.
  - `kPhase3OracleTelemetryUlpTight = 8`,
    `kPhase3OracleTelemetryUlpWide = 512` — declared for documentation
    only; actual telemetry uses a fixed
    {bitwise, 1, 8, 64, 512, 2048}-ULP distribution.
- New helper `PrintBf16DistributionSummary` prints six match counts per
  comparison instead of one. `RunPhase3NanoBucketK2688` now emits three
  distribution lines at the top:
  - `local math oracle vs flashinfer tactic1` (sanity check — should
    look like `<=512u=~49%`).
  - `runtime kernel vs flashinfer tactic1 (PRIMARY GATE)` (the gate).
  - `runtime kernel vs local math oracle (telemetry only)`.
- The per-axis histograms and the 8×8 dump are now keyed on the
  **bitwise-match-with-flashinfer** criterion, since that is the
  primary gate and pattern-based reasoning should reference it.
- `kPhase3MathOracleUlpTolerance` has been removed.

### Working rule going forward (supersedes the earlier bullet list)

- **Primary gate**: runtime vs flashinfer, strict bitwise (0
  mismatches). The cross-tactic convergence proof makes the dump
  canonical.
- **Telemetry**: local math oracle at distribution {bitwise, 1, 8, 64,
  512, 2048} ULPs is cheap to compute, lives alongside the gate, and
  serves as a sanity check that the dump file itself is interpretable
  (the oracle-vs-flashinfer summary should reach ~49% within 512 ULP on
  this bucket).
- **Localization**: the 2026-04-14 probe chain (reference B operand
  surface, runtime tag probe, bit-interleaved
  `desired_source_row = 2*local_row0 + 16*(byte&1) + 8*((byte>>1)&1) +
  32*(byte>>2)` encoding) is still the live bug lead for the remaining
  structured mismatch. Do not re-enter a runtime-kernel-edit loop on
  this test; use the probe chain to localize the B-side staging bug
  first, then fix.
- **Do not tighten the oracle tolerance.** Anything tighter than ~64
  ULP against the oracle is measuring FP4 precision noise, not runtime
  correctness, and will fail on any correct FP4 kernel.

## Next useful move

Do not add another broad flashinfer probe first.

Use the 2026-04-14 probe-chain findings (the bit-interleaved
`desired_source_row` encoding on the B-stage) to localize the last
structural runtime bug. The gate restructure above means any real
improvement will show up as a jump in the **bitwise** runtime-vs-
flashinfer column of the distribution summary, not as a shift in the
oracle tolerance column. Use that column to judge fixes, not the
oracle-match column — the oracle column responds to FP4 precision
noise and will be much noisier than the actual signal.
