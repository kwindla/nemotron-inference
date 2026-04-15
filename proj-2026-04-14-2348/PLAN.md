# Plan: NanoP1 mainloop rewrite via bitwise checkpoint ladder

Project directory: `./proj-2026-04-14-2348`

## Project context

From-scratch C++/CUDA inference engine for Nemotron-3 Nano (30B) hybrid
Mamba-attention-MoE model on RTX 5090. This plan is scoped narrowly to
one kernel: the routed MoE FC1 GEMM for the "P1" tile shape
(`CtaShape128x128x64B_Cluster1x1x1`, `SwapAB=false`) — one GEMM, one
expert, one layer — because the rest of the network waits on its
correctness.

Three prior plans (`proj-2026-04-12-1022/PLAN*.md`, plus the 2026-04-13
salvage and 2026-04-14 session work) tried to converge on a bitwise-
correct NanoP1 kernel via runtime bisection and synthetic-tag probes.
Progress over the full span:

| Day      | Phase 3 runtime bitwise vs flashinfer tactic1       |
| ---      | --------------------------------------------------- |
| 2026-04-12 | `2149 / 245760` (0.874%) before prior session      |
| 2026-04-13 | stuck; /loop mode oscillated between 230k and 243k mismatches (context kept compacting) |
| 2026-04-14 | `4074 / 245760` (1.658%) after commit `a008831`    |
| Now      | `4074 / 245760` (1.658%) — same state, gate fixed  |

The gate fix (commit `c26e390`) restructured the test so Phase 3 gates
on `runtime vs flashinfer bitwise mismatches == 0`, not on a tight ULP
comparison against a local FP64 oracle. Five Python validations under
`proj-2026-04-12-1022/probes/oracle_validation_{1..5}_2026-04-14.py`
established that `oracle vs flashinfer` reaches ~49% within 512 ULPs on
random FP4 data — the FP4 precision floor — so the prior 1-ULP oracle
gate was measuring quantization noise, not correctness. With the gate
fixed, the real number to move is `runtime vs flashinfer bitwise`.

Three days of runtime-bisection debugging did not move that number. The
session log in `proj-2026-04-12-1022/probes/nano_p1_phase3_session_2026-04-14.md`
lays out the trajectory; the short version is that every fix attempt
judged step K by an end-to-end bf16 output signal that only fires after
steps K, K+1, K+2 all work simultaneously, and the signal is blurred by
FP4 precision noise. We ran out of forward motion on 2026-04-14 after a
three-pass staging experiment (`position_map` zero-at-invalid,
`position_map` fallback-at-invalid, and a two-pass row-permute +
position-map override) each reduced the bitwise rate by ~10x. The
underlying helpers `NanoP1SourceRowForBStageOffset` and
`NanoP1PermuteBSourceRow` turned out to be partially circular —
verified by a probe whose own expected values were computed by the same
formulas it was validating.

This plan proposes a rewrite of the mainloop data-movement layer on a
bitwise checkpoint ladder, not another round of runtime bisection.

## Goal

Land a production NanoP1 mainloop kernel (`ComputeNanoP1AccumTile` and
its staging layer in
`runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`) whose Phase
3 Nano K=2688 output is byte-identical to
`proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/bf16_gemm1_tactic1.bin`.

The kernel must match flashinfer **bitwise**, not just within FP4
precision, and must do so via a rewrite whose correctness is established
at each of four intermediate bitwise checkpoints, not at the
end-to-end bf16 output alone. The CUTLASS type aliases
(`NanoP1TiledMma`, smem layouts, copy atoms, `NanoP1AccumLayout`) stay
unchanged — they are already verified bitwise-identical to flashinfer's
public `cutlass_kernels` collective-builder output via the compile-time
layout probes in `proj-2026-04-12-1022/probes/step_4a_probe_values.txt`
and the 2026-04-13 `ShowType<>` probe. The rewrite replaces only the
hand-rolled byte-copy layer and its helpers.

Correctness first. Performance is deferred. The end-state kernel is
allowed to be slower than the reverted partial-fix kernel; a follow-on
plan handles TMA enablement and optimization once bitwise parity is
locked in and the rest of the model can run.

## Scope

In scope:

- Flashinfer instrumentation to emit golden byte dumps of `smem_A`,
  `smem_B`, `smem_SFA`, `smem_SFB` at the moment before the first
  `cute::copy(s2r_copy_*, ...)` call, for the Nano K=2688 bucket,
  tactic 1, CTA (0, 0, 0), K iteration 0, stage 0.
- Flashinfer instrumentation to emit golden dumps of per-lane
  `tCrA`/`tCrB`/`tCrSFA`/`tCrSFB` register fragments post-`fp4_shift`,
  pre-gemm; and per-lane accumulator fragments post-gemm, pre-epilogue,
  after the last K iteration of the first CTA.
- A host-side `tools/smem_dump_diff` byte-diff utility for comparing
  saved smem/register dumps.
- Four new C++/CUDA test files
  (`nano_p1_checkpoint_{a,b,c,d}_test.cu`) that each run a minimal
  kernel, emit the matching checkpoint dump, and bytewise-diff it
  against the golden.
- A new equivalence ladder of four kernels, each differing from the
  previous by exactly one concrete change, each tested at its
  respective checkpoint:
  - V0 `flashinfer_passthrough_kernel.cu`
  - V1 `static_gmem_stage_kernel.cu`
  - V2 `swizzle_derived_stage_kernel.cu`
  - V3 `gmem_source_stage_kernel.cu`
- The final V4 replacement of `ComputeNanoP1AccumTile`'s staging with
  the verified V3 path.
- Deletion of the unverified helpers `NanoP1PermuteBSourceRow` and
  `NanoP1SourceRowForBStageOffset` in
  `runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh` once V4 is
  wired into production.
- Repurposing of `testing/backend/nano_p1_b_operand_probe.cu` and
  `nano_p1_reference_b_operand_probe.cu` as flashinfer-dump emitters,
  or their deletion once the checkpoint tests subsume them.

Out of scope (deferred):

- TMA enablement for the NanoP1 mainloop.
- Roofline / performance optimization.
- The direct-pack epilogue (`StoreNanoP1DirectPackCFragments`,
  `AccumulateNanoP1DirectPackRowMaxAbs`). These don't block bitwise
  parity at the BF16 checkpoint and get their own follow-on plan once
  NanoP1 lands.
- Phase 2 of the mainloop oracle test, which stays DEFERRED per prior
  plan decision.
- Other failing backend tests
  (`nvfp4_weight_test`, `expert_layer_oracle_test`,
  `expert_layer8_oracle_test`, `nano_p1_direct_pack_oracle_test`) —
  those predate NanoP1 and are tracked separately.

## Current state

- Current branch: `unified-routed-fp4`
- Current HEAD: `c26e390` (gate fix committed 2026-04-14)
- `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`: unchanged
  since `a008831` (partial TracedP5PermTileN row-only permute on the B
  side). Revert-verified after three failed position-map staging
  experiments.
- `testing/backend/nano_p1_mainloop_oracle_test.cpp`: primary gate is
  bitwise runtime-vs-flashinfer at `mismatches == 0`. Telemetry
  distribution summaries print for oracle-vs-flashinfer,
  runtime-vs-flashinfer, runtime-vs-oracle at {bitwise, 1, 8, 64, 512,
  2048} BF16 ULPs.
- Phase 1 (synthetic all-ones uniform fill): PASS.
- Phase 2 (capture-backed pre-quantized FP4): DEFERRED (by prior plan
  decision).
- Phase 3 (Nano K=2688 bucket): FAIL at
  `4074 / 245760` bitwise matches, `47.455%` within 512 ULPs.
- The CUTLASS type aliases (`NanoP1TiledMma`, `NanoP1SmemLayoutA`,
  `NanoP1SmemLayoutB`, `NanoP1SmemLayoutSFA`, `NanoP1SmemLayoutSFB`,
  `NanoP1SmemCopyAtomA`, `NanoP1SmemCopyAtomB`, `NanoP1AccumLayout`)
  are verified bitwise-identical to flashinfer's vendored
  `cutlass_kernels` CollectiveBuilder output. This verification is
  preserved in `proj-2026-04-12-1022/probes/nano_p1_phase3_layouts_2026-04-13.md`
  and the `ShowType<>` probe pattern.
- Flashinfer's SM120 FP4 MoE GEMM tactics 0..7 are cross-tactic
  bitwise-convergent on the Nano K=2688 bucket
  (`tactic0 vs tactic1 = 245760/245760 bitwise`; see
  `oracle_validation_2026-04-14.py` run 1).

## Design intent

Two principles. Everything else follows.

### P1: The unit of progress is a bytewise-equal dump at a known program point

Not "test passes". Not "mismatch pattern looks right". Not "probe
reports MATCH". An actual `memcmp(runtime_dump, flashinfer_dump) == 0`
at an agreed-upon checkpoint inside the kernel. If a change cannot be
verified by such a diff, it is not ready to land.

Every step of the ladder has a corresponding golden file, a
corresponding test that emits a dump under the same conditions, and a
corresponding `smem_dump_diff` invocation that either succeeds
silently or reports the first N differing offsets with hex context.

### P2: Flashinfer is the executable spec, not the CUTLASS source

We do not reason about what flashinfer's `CollectiveBuilder` composes
from `SmemLayoutAtomA`, `tile_to_shape`, `TracedP5PermTileN`, or the
SW64 swizzle. We instrument flashinfer's kernel to dump actual bytes at
actual program points, save those dumps as golden files, and our job is
to make a different kernel whose dumps match.

The moment anyone on this plan says "CUTE type X must mean Y because
the layout has shape Z," they are re-entering the three-day
debugging cycle that produced this plan. The correct move is to stop
reasoning and dump the bytes.

## Rules

### Kernel provenance (from PLAN_RULES.md)

This plan's target
`runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh` is **on the
prefill hot path**. Per PLAN_RULES.md "Kernel Provenance (hard
constraint)":

- **Allowed**: our own from-scratch CUDA code, `cute::MMA_Atom<cute::SM120_16x8x64_TN_VS<...>>`,
  `cute::TiledMMA`, `cute::Copy_Atom<cute::SM75_U32x4_LDSM_N>`,
  `cute::Layout`, `cute::Tensor`, swizzle functors, and
  `cutlass::detail::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`.
- **Forbidden**: `cutlass::gemm::collective::CollectiveBuilder`,
  `cutlass::gemm::device::GemmUniversal`, `KernelScheduleAuto`,
  `TmaWarpSpecialized*`, any header under
  `cutlass/gemm/collective/`, `cutlass/gemm/device/`,
  `cutlass/gemm/kernel/`, or `cutlass/epilogue/collective/`, and any
  FlashInfer, TRT-LLM, or vLLM runtime code.
- **This plan**: the existing NanoP1 CUTLASS type aliases
  (`NanoP1TiledMma`, `NanoP1SmemLayoutA/B/SFA/SFB`,
  `NanoP1SmemCopyAtomA/B`, `NanoP1AccumLayout`) are CUTE atoms and
  layout primitives, which is exactly the permitted set. No
  `CollectiveBuilder` call is introduced by this plan; every step
  stays inside the permitted CUTE-atom perimeter.
- **Flashinfer is a source reference, not a runtime dependency.** The
  flashinfer vendored path is only used in *tooling* (Step 1,
  instrumentation for dumping goldens) and *test* code (Step 4-7,
  static-input reference kernels), never linked into the runtime
  library.

### Safety and sequencing (from PLAN_RULES.md)

- Do not disable the current NanoP1 mainloop until the V4 replacement
  is bitwise-green at checkpoints A/B/C/D AND the full ctest sweep is
  no worse than HEAD. The partial-fix kernel at `4074 / 245760`
  bitwise stays live until V4 is wired in via Step 8.
- Reuse existing payload buffers unless duplication is required by
  the checkpoint ladder. The lookup-table produced in Step 7 is a
  generated constexpr header; no new runtime allocations.
- New kernels and launcher paths enter existing planning/tracing/
  benchmarking infrastructure. In this plan, the production change
  (Step 8) modifies `ComputeNanoP1AccumTile` in place; no new
  production launcher path is introduced.
- Kernels must support arbitrary token counts (1-512+), including
  partial row tiles. The Nano K=2688 golden captures `num_tokens=128`,
  but Step 8's production code must still handle `valid_rows < kTileM`
  correctly (the partial-row masking from the current
  `ComputeNanoP1AccumTile` implementation is preserved).

### Validation categories (from PLAN_RULES.md)

Ordered from highest signal to lowest:

- **Checkpoint diff** (this plan's primary instrument): bytewise
  `smem_dump_diff` at each of checkpoints A, B, C, D against
  flashinfer-derived goldens. Any mismatch is an immediate stop on
  the step that produced it.
- **Per-layer diagnostic** (`NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`):
  compares the prefill path against the single-token decode path per
  layer and per token. Run at Step 8 (V4 production wiring) and Step
  9 (cleanup).
- **Behavioral parity oracle on RTX 5090**:
  `tools/oracle/compare_chat_runtimes.py` against local vendored
  vLLM. Run at Step 8 after checkpoint D turns green.
- **Prompt-length sweep**:
  `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 uv run tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval`.
  Run at Step 8. Expected: `ALL CHECKS PASSED`.
- **24-token regression oracle**:
  `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build-sm120-relwithdebinfo/testing/nano_24_token_prefill_regression_test`.
  Run at Step 8. Must produce exact token-oracle matches for prompt23
  and prompt24.
- **Secondary external reference**: flashinfer's tactic 1 dump (the
  existing `bf16_gemm1_tactic1.bin`) remains the bitwise oracle for
  Checkpoint D.
- **Claude coherence scoring**: smoke test only, not acceptance
  proof.
- **Primary performance gate**: end-to-end prefill latency. Explicitly
  **deferred** to a follow-on plan. This plan's target kernel is
  allowed to be slower than the partial-fix kernel on HEAD.
- **Secondary diagnostic metrics**: hot kernel time and cold
  descriptor-build/setup overhead. Also deferred.

### Resource constraints (from PLAN_RULES.md)

- Per-CTA shared memory: do not increase beyond what the current
  `ComputeNanoP1AccumTile` uses (`NanoP1SharedStorage` in
  `nano_p1_kernel.cuh`). The generated lookup-table from Step 7 is a
  constant in gmem / constant memory, not additional smem.
- GPU memory: RTX 5090, 32GB, fully consumed by the model. The
  checkpoint test kernels run on a single CTA with minimal
  allocation; no new runtime allocations.

### Test protocol (from PLAN_RULES.md, adapted for this plan)

All GPU tests run **sequentially** (`-j1`) — the RTX 5090's 32GB is
fully consumed by the model, and parallel GPU tests will OOM.

**Known pre-existing failures** (do not count as regressions):

- `nvfp4_weight_test` — model weight loading
- `expert_layer_oracle_test` — model weight loading
- `expert_layer8_oracle_test` — model weight loading
- `nano_p1_direct_pack_oracle_test` — Phase 3 DEFERRED (same root
  cause as the NanoP1 mainloop bug this plan fixes; expected to
  recover once V4 lands, but not a gate for this plan)

Current Phase 3 NanoP1 mainloop oracle status (HEAD `c26e390`): FAIL
at `4074 / 245760` bitwise vs flashinfer tactic 1. Baseline
expected ctest pass count: stable at whatever HEAD reports with those
4 failing — any NEW failure beyond that set is a regression.

**After any code change**:

```bash
touch runtime/src/backend/fused_moe_prefill.cu && \
  cmake --build build-sm120-relwithdebinfo --parallel $(nproc)
```

The `touch` is mandatory whenever a `.cuh` fragment under
`runtime/src/backend/fused_moe_prefill/` is modified — per the
2026-04-14 session note, the header-dependency tracking misses these
edits and produces silent stale builds, which cost ~2 hours of
debugging before being diagnosed.

**After runtime/backend code changes**:

```bash
ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -j1
```

Any new failure beyond the 4 known pre-existing is a regression —
diagnose before proceeding.

**After kernel changes affecting MoE prefill** (Steps 7, 8):

```bash
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 \
  build-sm120-relwithdebinfo/testing/nano_24_token_prefill_regression_test
```

Must produce exact token-oracle matches for prompt23 and prompt24.

**After changes that could affect inference quality** (Step 8):

```bash
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 \
  uv run tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval
```

Expected: `ALL CHECKS PASSED`.

**For the final acceptance gate** (Step 8):

```bash
uv run tools/oracle/compare_chat_runtimes.py
```

Bitwise-identical against vLLM for the defined eval slice.

### Correctness (plan-specific)

- Do not reintroduce any path that materializes BF16 before `Relu²` in
  the production direct-pack epilogue. This plan fixes the **mainloop
  accumulator** only; the direct-pack epilogue in
  `AccumulateNanoP1DirectPackRowMaxAbs` /
  `StoreNanoP1DirectPackCFragments` stays on the FP32-accumulator
  direct-pack contract established in proj-2026-04-11-0400.
- The `DeviceNvfp4Matrix` FC2-facing contract must not change. Step 8
  only touches staging + `cute::copy` + `cute::gemm`; the store path
  is untouched.
- Do not change the canonical token-major `topk_ids`/`topk_weights`
  routing contract.

### Plan-specific rules

1. **No runtime bisection on the primary mainloop kernel.** Phase 3
   failures do not drive fixes to `nano_p1_kernel.cuh`. Only a failed
   checkpoint test drives a fix.
2. **No `DIAG printf` bisection on Phase 3 at any point in this plan.**
   The prior /loop session died in this pattern. Any diagnostic output
   goes into a dedicated checkpoint test or into
   `tools/smem_dump_diff`.
3. **Every step changes exactly one thing.** If step N+1 changes both
   "source tensor" and "swizzle formula", it gets split into N+1a and
   N+1b. If N+1 fails and the diff doesn't localize, we split further.
4. **No speculative layout helpers.** `NanoP1PermuteBSourceRow` and
   `NanoP1SourceRowForBStageOffset` are deleted the moment V4 is
   verified. No new helper is added unless it is verified against a
   golden dump, not against another helper.
5. **Probe circularity is banned.** A probe that verifies a staging
   function by writing tag values computed from the same function it
   is validating is a test artifact, not a verification. If we need a
   probe, the probe's "expected" value comes from a flashinfer dump,
   not from a formula we also use in the kernel.
6. **Flashinfer goldens are version-pinned.** The golden dump files
   carry a `manifest.json` recording the flashinfer commit hash, the
   `capture_bf16_gemm1.py` inputs (seed, shape, dtype), dump sizes,
   and sha256 of each file. Any change to the flashinfer build
   triggers re-running the capture and a bytewise diff against the
   prior golden — if it changes, stop and explain why before touching
   anything else.
7. **The CUTLASS type aliases stay frozen during this plan.** If a
   step requires changing `NanoP1TiledMma` or a smem layout, stop and
   open a separate plan; changing CUTLASS types invalidates every
   checkpoint in this ladder.
8. **Revert > amend.** If a step breaks, revert it and write a new
   step — do not patch on top of the broken one. The history is a
   ladder, not a staircase with broken treads.
9. **Full ctest sweep at each step**, not single-target verification.
   Per CLAUDE.md feedback memory on full test sweeps.
10. **Touch the `.cu` before building** when `.cuh` fragments change,
    per the 2026-04-14 session note. The `touch` is already baked
    into the Test protocol above.
11. **Source code wins over prose.** For any flashinfer behavior
    question, the canonical reference is the literal vendored
    `flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/`
    source, not any prose summary of it (including salvage notes and
    this plan).
12. **Probe-then-decide** (per CLAUDE.md): compile-time probes first
    (already done for type identity in step 4a), then byte-dump
    checkpoints, then end-to-end. This plan's entire structure is a
    probe-then-decide ladder.

## Success criteria

- Checkpoint A (pre-MMA smem state) tests pass bytewise for smem_A,
  smem_B, smem_SFA, smem_SFB on the Nano K=2688 bucket.
- Checkpoint B (post-s2r-copy, post-fp4_shift register fragments) tests
  pass bytewise for tCrA, tCrB, tCrSFA, tCrSFB per-lane for all 256
  threads of CTA (0, 0, 0).
- Checkpoint C (post-gemm, pre-epilogue accumulator fragments) tests
  pass bytewise for the final accumulator state.
- Checkpoint D = existing `nano_p1_mainloop_oracle_test` Phase 3:
  `bitwise_mismatches_vs_flashinfer == 0`.
- `Phase 1 synthetic all-ones` still PASSes.
- Full sm120 `ctest` sweep is no worse than HEAD on unrelated tests
  (the four pre-existing failures can stay failing, but nothing new
  breaks).
- `NanoP1PermuteBSourceRow` and `NanoP1SourceRowForBStageOffset` are
  deleted from `nvfp4_bridge.cuh`.
- The gate fix commit `c26e390` behavior is preserved: primary gate
  remains bitwise runtime-vs-flashinfer.

## Steps

### Step 1: Flashinfer golden-dump instrumentation

**Goal**: produce `smem_A_pre_mma.bin`, `smem_B_pre_mma.bin`,
`smem_SFA_pre_mma.bin`, `smem_SFB_pre_mma.bin`,
`tCrA_pre_gemm.bin`, `tCrB_pre_gemm.bin`, `tCrSFA_pre_gemm.bin`,
`tCrSFB_pre_gemm.bin`, and `accumulator_post_gemm.bin` under
`proj-2026-04-14-2348/golden/nano_k2688/`, from a patched run of
flashinfer's vendored SM120 FP4 MoE GEMM kernel, at tactic 1, CTA (0,
0, 0), K iteration 0 for the smem pre-MMA files, and after the last K
iteration for the accumulator file.

**Approach**: salvage the live-operand probe patch points recorded in
`proj-2026-04-12-1022/probes/nano_p1_reference_live_operand_patchpoints_2026-04-14.md`
and `proj-2026-04-12-1022/probes/archive/2026-04-14-probe-chain/nano_p1_phase3_session_full_2026-04-14.md`.
Those already identified where in
`flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/`
to patch, and confirmed the
`TmaWarpSpecializedGroupedGemmInput::ProbeArg` transport works end-to-end
for tactic 1. Reuse the transport, change the payload to whole-smem /
whole-fragment byte dumps, and write one file per artifact.

Extend `proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py`
(or fork it into this project dir) with a `--dump-smem-state` mode
that is tactic-1-only (per the 2026-04-14 session note on the
tactic-sweep CUDA error) and writes the dump files to a caller-
specified directory.

**Deliverables**:

- `proj-2026-04-14-2348/golden/nano_k2688/smem_{A,B,SFA,SFB}_pre_mma.bin`
- `proj-2026-04-14-2348/golden/nano_k2688/tCr{A,B,SFA,SFB}_pre_gemm.bin`
- `proj-2026-04-14-2348/golden/nano_k2688/accumulator_post_gemm.bin`
- `proj-2026-04-14-2348/golden/nano_k2688/manifest.json` recording the
  flashinfer commit, the capture inputs, the dump sizes, and sha256 of
  each file.
- `proj-2026-04-14-2348/tools/capture_goldens.sh` runnable wrapper.

**Validation**: run the capture twice and confirm the golden files are
byte-identical. Confirm tactic 0 and tactic 1 produce byte-identical
goldens for smem and tCr*, because both tactics converge on this
bucket.

**Estimated effort**: 1 day. The bulk of the work is already done by
the 2026-04-14 probe chain; this step is changing the probe payload
and output format, not rediscovering the patch points.

### Step 2: `tools/smem_dump_diff` byte-diff utility

**Goal**: a small C++ host binary under `tools/smem_dump_diff/` that
takes two .bin files and reports the first N (default 32) differing
offsets with hex+ASCII context plus a per-region summary.

Exit code 0 on bytewise equality, 1 on any mismatch. The output format
is line-oriented and greppable so test harnesses can parse it.

**Deliverables**:

- `tools/smem_dump_diff/smem_dump_diff.cpp`
- `tools/smem_dump_diff/CMakeLists.txt` (standalone, links nothing from
  `nemotron_runtime_*`)
- Usage: `smem_dump_diff <a.bin> <b.bin> [--max-diffs 32] [--skip-header N]`

**Validation**: run it on two identical files (exit 0, no output), on
two known-different files (reports the first differing offset and
exits 1), on files of different sizes (reports the size mismatch and
exits 1).

**Estimated effort**: a few hours. Pure host code, no GPU.

### Step 3: Checkpoint A test (pre-MMA smem state)

**Goal**: a C++/CUDA test
`testing/backend/nano_p1_checkpoint_a_test.cu` that launches a minimal
kernel, fills `smem_A`, `smem_B`, `smem_SFA`, `smem_SFB` from the
current `ComputeNanoP1AccumTile` staging path, dumps the four smem
buffers to host as .bin files, and runs `smem_dump_diff` against the
Step 1 goldens for each.

The test is parameterized on which staging version to exercise. The
first landing uses **V0-the-reference**: the test runs a kernel that
does a `cudaMemcpy` of the goldens directly into smem (i.e., no
staging, just a copy). This is a sanity check that the golden files
survive the round trip; by construction it must pass.

**Deliverables**:

- `testing/backend/nano_p1_checkpoint_a_test.cu`
- `testing/CMakeLists.txt` entry registering the test
- A helper header `testing/backend/nano_p1_checkpoint_dump.cuh`
  (test-only; not linked into the runtime library) that exposes a
  `DumpSmemStage0` device function that can be called from any
  checkpoint-capable test kernel.

**Validation**: test runs green on V0-the-reference (round-trip
identity). Running the test against the CURRENT partial-fix
`ComputeNanoP1AccumTile` staging will fail — that failure is the
first actionable localization signal the ladder produces. Capture the
diff output and reference it in Step 4.

**Estimated effort**: half a day.

### Step 4: V0 `flashinfer_passthrough_kernel.cu`

**Goal**: the first kernel in the ladder that actually runs a MMA.
This kernel does nothing except `memcpy` the golden `smem_{A,B,SFA,SFB}_pre_mma.bin`
from gmem directly into its smem buffers, then runs exactly the same
`cute::copy` → `fp4_shift` → `cute::gemm` sequence as
`ComputeNanoP1AccumTile`, then dumps the accumulator fragment for
checkpoint C.

The checkpoint B dump (post-s2r-copy, post-fp4_shift register state)
fires at the same program point `ComputeNanoP1AccumTile` currently
does its probe dump. Compare bytewise to the Step 1 goldens.

**Expected failure mode**: if checkpoint B passes and checkpoint C
fails, the MMA atom dispatch differs from flashinfer's. If B fails,
the copy atom or `fp4_shift` is wrong. If both pass, V0 is done and we
move to Step 5 knowing the CUTLASS-native compute layer is bitwise-
equivalent to flashinfer's.

**Deliverables**:

- `testing/backend/nano_p1_flashinfer_passthrough_kernel.cu`
- `testing/backend/nano_p1_checkpoint_b_test.cu` (launches
  flashinfer_passthrough_kernel, dumps tCr{A,B,SFA,SFB}, diffs
  against golden)
- `testing/backend/nano_p1_checkpoint_c_test.cu` (same kernel, dumps
  accumulator)

**Validation**: both B and C checkpoints pass bytewise. If either
fails, stop. The bug is in CUTLASS type declarations or the
`cute::gemm` call signature — do not proceed to Step 5 until this is
resolved. Expected resolution cost: hours, not days, because the blast
radius is one file.

**Estimated effort**: 1 day including debugging if B or C initially
fails.

### Step 5: V1 `static_gmem_stage_kernel.cu`

**Goal**: replace the `cudaMemcpy` of golden smem with a staging loop
that writes smem byte-by-byte, with the source byte still coming from
the saved golden dump. The destination offset comes from
`stage0_B(row, byte_index * 2)` (same CUTE layout as V4 will use). The
source byte is read from the golden dump indexed by the dump's linear
offset (not by the CUTE layout — that's Step 6).

This isolates **"can we use the CUTE swizzle to compute smem
destinations"** from **"do we know which source byte goes where"**.
The source is always right because it comes from the golden; the
destination offset is the thing under test.

**Validation**: checkpoint A passes bytewise. Both B and C still
pass.

**Deliverables**:

- `testing/backend/nano_p1_static_gmem_stage_kernel.cu`
- The checkpoint A test from Step 3 runs against this kernel in
  addition to V0.

**Estimated effort**: 1 day.

### Step 6: V2 `swizzle_derived_stage_kernel.cu`

**Goal**: replace the source (golden dump indexed linearly) with the
golden dump indexed by `stage0_B(row, byte_index * 2)`. That is: the
staging loop now reads and writes from the golden using the same
CUTE swizzle formula it uses for the destination. Both sides are the
same; the operation is effectively a copy. The byte pattern should
still be bytewise-identical to the golden.

This sounds trivial, but it's the step where we confirm that the CUTE
swizzle is reversible via the same formula — i.e., that "read at
stage0_B(row, b*2) from the dump" equals "write at stage0_B(row, b*2)
to smem" for every `(row, byte_index)` pair. If V2 fails where V1
passed, the CUTE layout composition is not the identity we thought it
was, and we have a highly-localized place to look.

**Deliverables**:

- `testing/backend/nano_p1_swizzle_derived_stage_kernel.cu`

**Validation**: checkpoint A passes bytewise.

**Estimated effort**: half a day.

### Step 7: V3 `gmem_source_stage_kernel.cu` — the heart of the rewrite

**Goal**: replace the source (golden smem dump) with the original FP4
gmem files (`input_fp4_permuted.bin`, `inputs_w1_fp4.bin` and
scale files). The staging loop now reads from the real input tensors
and must land bytes in smem such that `checkpoint A` still passes.

The formula: for each destination smem offset
`dst = stage0_B(row, byte_index * 2)`, determine the correct
`(source_row_within_tile, byte_index_within_tile)` by looking up `dst`
in a precomputed table. The table is built by running a one-time
host-side analysis: for each smem byte position, read the golden
value and find which byte of the gmem source tensor equals it (with
a tie-breaker via byte-pair clustering if multiple candidates match).
This produces a deterministic
`dst → (source_row, byte_index)` mapping, independent of any
handwritten formula.

The table is **valid across all K iterations and all CTAs** because:

1. The CUTE layouts are invariant across K iterations — the swizzle
   that maps `(row, byte_index) → dst` is determined by the smem
   layout type, not the K position. Every K iteration's staging loop
   writes the same pattern of destination slots.
2. The lookup table is CTA-local: it maps "destination slot X within
   this CTA's stage 0 smem" to "tile-local source row R, tile-local
   byte index I". The kernel adds `row_start` and `packed_byte_offset
   = k_base/2` at runtime to index the global gmem tensor.
3. The golden is captured at K iteration 0, CTA (0, 0, 0). The same
   mapping applies at every other K iteration and every other CTA as
   long as the CUTLASS type aliases don't change (Rule 7 forbids
   that).

For the scales, the same lookup is built against the SFA/SFB goldens.
The scale layout `sSFA/sSFB` is per-16-K-element block rather than
per-byte, so the scale lookup table has a different shape (indexed by
block within the CTA tile, not byte), but the same invariance
argument applies.

**Validation**: checkpoint A passes bytewise, checkpoints B, C still
pass, **and** checkpoint D (end-to-end Phase 3) passes with `bitwise
mismatches == 0`.

**Deliverables**:

- `proj-2026-04-14-2348/tools/build_stage_lookup.py` — host-side
  script that reads the golden smem dumps and the gmem source files,
  builds the lookup table, and emits it as a constexpr C++ header.
- `runtime/src/backend/fused_moe_prefill/nano_p1_stage_lookup.cuh`
  (generated, checked in — stable once built).
- `testing/backend/nano_p1_gmem_source_stage_kernel.cu`
- Checkpoint D (the existing
  `nano_p1_mainloop_oracle_test::RunPhase3NanoBucketK2688`) now runs
  against a test-time variant of the kernel that uses V3.

**Estimated effort**: 2 days including the lookup-builder script and
the generated header integration.

### Step 8: V4 — wire V3 into production `ComputeNanoP1AccumTile`

**Goal**: replace the hand-rolled staging loop in
`runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh` with the V3
path from Step 7. This is where the production kernel starts using
the lookup-table staging.

Delete `NanoP1PermuteBSourceRow` and `NanoP1SourceRowForBStageOffset`
from `nvfp4_bridge.cuh`. Delete `testing/backend/nano_p1_b_operand_probe.cu`
if its role is subsumed by the checkpoint A test (or convert it to
another flashinfer golden emitter). Delete
`testing/backend/nano_p1_reference_b_operand_probe.cu` if similar.

**Validation**:

- Checkpoint D (`nano_p1_mainloop_oracle_test` Phase 3) passes with
  `bitwise mismatches vs flashinfer == 0`.
- Phase 1 still passes.
- `nano_p1_direct_pack_oracle_test` (if it depends on the mainloop)
  shows improvement relative to HEAD, though its Phase 3 may still be
  DEFERRED due to epilogue issues.
- Full `ctest -C RelWithDebInfo` sweep: nothing that passed before now
  fails.

**Deliverables**:

- Edit
  `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`.
- Delete the dead helpers.
- Commit message records: checkpoint A/B/C/D status, full ctest
  status, lines deleted/changed, and the flashinfer golden manifest
  hash.

**Estimated effort**: half a day. Most of the work is in steps 1-7.

### Step 9: Clean up and archive

- Update `proj-2026-04-12-1022/probes/nano_p1_phase3_session_2026-04-14.md`
  with a "superseded by proj-2026-04-14-2348" pointer.
- Move the remaining oracle validation scripts under
  `proj-2026-04-12-1022/probes/` into
  `proj-2026-04-14-2348/archive/` if they are no longer load-bearing.
- Verify the gate fix commit `c26e390` is still the authoritative
  Phase 3 test gate structure. Update the test's comment block if the
  design changed.
- Record the new bitwise-match rate (target: `245760 / 245760`) in a
  short `proj-2026-04-14-2348/RESULTS.md`.

**Estimated effort**: 2 hours.

## Budget and stop conditions

Total estimated effort: 7 working days end-to-end (Steps 1-9), assuming
Step 1 is closer to 1 day than 3 because the prior probe-chain work
did most of the flashinfer patch-point discovery.

**Stop conditions**:

1. If Step 1 takes more than 3 working days, reassess. The probe-chain
   lineage should make this a 1-day job. If it's not, the flashinfer
   instrumentation strategy may need a different approach (e.g., host-
   side reconstruction from `input_fp4_permuted.bin` + the saved
   register probe data) rather than in-kernel dumps.
2. If Step 4 (V0 passthrough) fails both checkpoint B and checkpoint
   C, it means the CUTLASS type aliases are NOT identical to
   flashinfer's despite the compile-time layout probe saying they
   should be. Escalate to a separate type-identity investigation
   before continuing.
3. If Step 5 (V1 static-gmem-stage) fails after Step 4 succeeds, the
   issue is isolated to `stage0_B(row, byte_index * 2)` composition.
   That's a small, bounded bug to hunt.
4. If Step 6 (V2 swizzle-derived) fails after V1 passes, the CUTE
   swizzle is not self-consistent for the loop iteration pattern. This
   is also small and bounded.
5. If Step 7 (V3 gmem-source) fails and the lookup builder reports
   that multiple gmem bytes match a given smem golden position (a
   non-unique lookup), we have a genuine data-model mismatch: the
   goldens don't correspond to the input tensors we think they do. At
   that point stop and re-examine the flashinfer capture inputs.

## References

### Prior plans and debugging trail

- **Prior plan (partially successful; superseded by this one)**:
  `proj-2026-04-12-1022/PLAN.md`. Reached partial bitwise match
  (`4074 / 245760`) via commit `a008831` but did not converge to full
  bitwise parity.
- **Salvage notes from the 2026-04-13 prior-session rescue**:
  `proj-2026-04-12-1022/probes/nano_p1_phase3_salvage_2026-04-13.md`.
  Documents the 8.5 h /loop oscillation, the ruled-out hypotheses,
  and the operand-role investigation chain.
- **2026-04-14 session synopsis**:
  `proj-2026-04-12-1022/probes/nano_p1_phase3_session_2026-04-14.md`,
  with the full archived transcript at
  `proj-2026-04-12-1022/probes/archive/2026-04-14-probe-chain/nano_p1_phase3_session_full_2026-04-14.md`.
- **2026-04-14 in-session work (this session)**: failed staging
  experiments documented in chat; all reverted. No lingering working-
  tree changes.

### Instruments and oracles

- **Gate fix commit**: `c26e390` (restructures
  `testing/backend/nano_p1_mainloop_oracle_test.cpp` to gate on
  bitwise runtime-vs-flashinfer with multi-tolerance distribution
  telemetry).
- **Oracle validation chain**:
  `proj-2026-04-12-1022/probes/oracle_validation_{1..5}_2026-04-14.py`.
  Five Python scripts that proved the FP4→BF16 precision floor is
  ~49% within 512 ULPs vs flashinfer, and that the local FP64 math
  oracle is algorithmically correct.
- **Compile-time layout probe values**:
  `proj-2026-04-12-1022/probes/nano_p1_phase3_layouts_2026-04-13.md`
  and `proj-2026-04-12-1022/probes/step_4a_probe_values.txt`. Prove
  the CUTLASS type aliases are identical to flashinfer's public
  `cutlass_kernels` CollectiveBuilder output. These are load-bearing
  for this plan's P2 principle.
- **Flashinfer reference dump (current)**:
  `proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/bf16_gemm1_tactic1.bin`.
  Remains the Checkpoint D oracle. This plan's Step 1 adds
  byte-dump goldens at earlier checkpoints without invalidating this
  file.
- **Flashinfer patch-point history**:
  `proj-2026-04-12-1022/probes/nano_p1_reference_live_operand_patchpoints_2026-04-14.md`.
  Used as the starting point for Step 1's instrumentation.

### Flashinfer source reference

- **Vendored flashinfer path (in-tree, read-only reference)**:
  `.venv-trtllm/lib/python3.12/site-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/`.
  Note: despite the `nv_internal` path component, the subdirectory
  is the **public** `cutlass_kernels` variant (not
  `internal_cutlass_kernels`). This is the source that actually runs
  when `capture_bf16_gemm1.py` invokes
  `gen_cutlass_fused_moe_sm120_module().build_and_load()`. Verified
  in the 2026-04-14 symbol-table check against
  `libtensorrt_llm.so`.
- **TRT-LLM architecture notes (cross-reference only, since the
  flashinfer vendored copy is what runs):**
  `proj-2026-04-12-1022/trtllm_reference/NOTES.md`.

### Validation tools

- `tools/oracle/compare_chat_runtimes.py` — vLLM behavioral parity
  oracle (end-to-end acceptance gate at Step 8).
- `tools/oracle/prompt_length_sweep.py` — prompt-length regression
  sweep, invoked with
  `--runtimes native --skip-claude-eval`.
- `build-sm120-relwithdebinfo/testing/nano_24_token_prefill_regression_test`
  — focused 24-token regression oracle. Must be run with
  `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1`.

### Methodology and shared rules

- `PLAN_RULES.md` — shared rules (Kernel provenance, Safety and
  sequencing, Validation categories, Resource constraints, Test
  protocol). This plan includes them by reference and extends where
  noted.
- `CLAUDE.md` — project-level methodology, including the
  "Probe-then-decide" §1 that this plan operationalizes as its
  checkpoint ladder, and the development workflow rules
  (`/plan-init` → `/implement`).

## Progress

- [ ] Step 1: flashinfer golden-dump instrumentation
- [ ] Step 2: `tools/smem_dump_diff` byte-diff utility
- [ ] Step 3: checkpoint A test (pre-MMA smem state)
- [ ] Step 4: V0 `flashinfer_passthrough_kernel.cu`
- [ ] Step 5: V1 `static_gmem_stage_kernel.cu`
- [ ] Step 6: V2 `swizzle_derived_stage_kernel.cu`
- [ ] Step 7: V3 `gmem_source_stage_kernel.cu`
- [ ] Step 8: V4 wire into production
- [ ] Step 9: clean up and archive
