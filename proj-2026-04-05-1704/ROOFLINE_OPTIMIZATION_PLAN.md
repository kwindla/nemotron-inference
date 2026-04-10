# Roofline Optimization Plan

## Target

Stretch target: 20 ms cold TTFT for a 128-token prompt on RTX 5090.
This is about 2x faster than the local TRT-LLM baseline (`38 ms`) and
within about 1.7x of the corrected theoretical roofline floor
(`~11.8-12.0 ms`).

Planning target: drive the active `window=4096` production benchmark from
today's `~176 ms` at `prefix128 / tail4` to below `70 ms`, then reassess
against measured bandwidth and cache-utilization gains before committing to
the `50 ms`, `35 ms`, and `20 ms` milestones.

## Debugging Discipline

When a hardware-level test fails (illegal memory access, silent corruption,
hang), do NOT iterate by tweaking parameters at the same abstraction level.
After 2 failures with the same API/approach:

1. **Stop and re-read the reference implementation.** The answer is almost
   always that we're using the wrong abstraction, not the wrong parameters.
2. **Write a minimal copy-only smoke test** at the abstraction level the
   reference actually uses, before touching the full kernel again.
3. **Isolate one variable at a time** (operand isolation, copy-only vs
   copy+MMA, static vs dynamic smem) — but only after confirming you're at
   the right API level.

Concrete lesson: the Phase 0.5 TMA proof burned 5 loops trying to make raw
`SM90_TMA_LOAD_2D::copy(...)` work when TRT's reference code — already open
in context — uses `make_tma_copy(SM90_TMA_LOAD{}, ...) + cute::copy(...)`.
The fix was changing mechanism, not parameters.

Corollary for the next TMA pass:
- stay inside the CUTE abstraction by default
- use host-side `make_tma_copy(...)` objects plus kernel-side
  `copy(tma.with(barrier), partition_S, partition_D)`
- only go down to raw `cuTensorMapEncodeTiled` / PTX-level TMA if we are
  explicitly debugging CUTE itself

## Key Insight

Nemotron-3 Nano at prefix128 is **entirely memory-bandwidth-bound**.  The
Tensor Cores can finish the math in 0.3 ms.  The routed MoE weight traffic
alone is not the whole story: routed block scales add another ~12% to the
bytes that must move, pushing the routed MoE floor from `8.6 ms` to
`~9.7 ms` at peak bandwidth.  Every optimization that matters is about
moving bytes faster, moving fewer bytes, or hiding those bytes behind useful
work.

Current production state on the canonical surface:
- `prefix4 / tail4`: `68.780 ms`
- `prefix128 / tail4`: `175.692 ms`
- `prefix4096 / tail4`: `2337.938 ms`
- artifact:
  `artifacts/benchmarks/ttft_20260409T_plan_matrix_tail4.stdout.txt`

Reference baseline:
- TRT-LLM `prefix128`: `38.407 ms`

Interpretation:
- current native `prefix128` is about `15x` off the corrected `~11.8-12.0 ms`
  roofline floor
- current native `prefix128` is about `4.6x` slower than local TRT-LLM
- the plan must optimize the current `window=4096` production surface, not the
  earlier reduced-window smoke surfaces

## Numbers That Drive Every Decision

```
RTX 5090:
  GDDR7 bandwidth:   1792 GB/s peak, ~1300 GB/s effective
  L2 cache:          96 MB, ~7 TB/s to SMs
  L1/smem per SM:    128 KB
  Smem per CTA cap:  101376 B opt-in (~99 KB)
  Smem per SM cap:   102400 B (~100 KB)
  SMs:               170
  FP4 Tensor:        1676 TOPS (dense)
  Registers per SM:  256 KB

Nano per MoE layer:
  Routed weight:     661 MB (128 experts × FC1+FC2, FP4)
  Routed scales:     78.75 MiB (1 ue4m3 scale byte per 16 FP4 weights)
  Shared weight:     10 MB
  Routed compute:    15.9 GFLOP
  Shared compute:    5.1 GFLOP
  Arithmetic intensity: 31 FLOP/byte (need 935 to balance)
  Routed seam traffic:
    ~2.8 MiB BF16 FC1 intermediate
    ~0.7 MiB packed FC1->FC2 activation
    ~4-8 MiB FC2 output / finalize writes
    ~10-15 MiB total per layer, small for the roofline floor but important
    for Phase 3 seam reduction

One expert (FC1+FC2):
  Weight:            5.2 MB
  Scales:            0.615 MiB
  Tokens (avg):      6 at prefix128
  Compute:           62 MFLOP

L2 capacity:         ~18 experts at a time

Full forward pass:
  MoE weight (23 layers): 15.4 GB → 8.6 ms at peak BW
  MoE scales (23 layers): ~1.9 GB → ~1.1 ms at peak BW
  Routed MoE total: ~17.3 GB → ~9.7 ms at peak BW
  Mamba weight (23 layers): ~2.7 GB → 1.5 ms
  Attention weight (8 layers): ~0.5 GB → 0.3 ms
  LM head: ~0.7 GB → 0.4 ms
  Total weight+scale floor: ~21.2 GB → ~11.8 ms at peak BW
```

## Canonical Benchmark Surface

All optimization decisions in this plan use one benchmark contract only:

- `--moe-prefill-window-tokens 4096`
- `prefix-length 4`, `128`, and `4096`
- `tail-token-count 4`
- single active request

That avoids mixing the old reduced-window smoke cases with the production
surface.  All milestone numbers below refer to this contract unless
explicitly stated otherwise.

## Optimization Doctrine

### Bytes are the primary API

Every hot-path decision should be evaluated in terms of:

- mandatory HBM bytes for one forward pass
- actual HBM bytes moved by the runtime
- bytes duplicated across multiple layouts or scratch buffers
- bytes written to intermediate workspace and reread later
- achieved DRAM bandwidth as a percentage of practical peak

TTFT remains the user-facing score, but byte movement is the engineering
control surface.

### One execution layout on the active path

On the active routed path we want:

- one weight layout
- one scale layout
- one activation-pack layout
- one routed kernel family
- compile-time execution dimensions wherever the model family is static

If a buffer exists only to translate between two internal layouts, that buffer
is a likely tax that should be removed.  The ideal active path stores weights
in the exact execution form, packs activations directly into the exact
execution form, and consumes that form without repacking.

For the active Nano routed path, execution dimensions like padded
`routed_intermediate_size = 1920` should live in traits / template constants,
not as dynamic hot-path loop bounds, so shared-memory sizing and address
arithmetic can be constant-folded.

### Transport before compute

The next major wins should come from:

1. direct global → swizzled shared-memory transport
2. multi-stage overlap of transport and MMA
3. warp-specialized producer/consumer execution

This work comes before smaller math-side cleanups because the current gap is
not MMA issue rate.  It is the cost of staging bytes and exposing load
latency.

### Treat FC1 and FC2 as one routed system

The routed expert path should be optimized as one dataflow:

- FC1 input pack
- FC1 GEMM
- activation / requantization
- FC2 GEMM

The expensive seams are usually between kernels, not inside a single MMA
instruction.  The main question is therefore not “which kernel next?” but:

- what must hit global memory?
- what can remain in one request-scoped workspace?
- what can be packed once and consumed directly?

### Scheduling is a first-class optimization

If the routed path is bandwidth-bound, then *which expert tiles run when*
matters.  After the transport path is fixed, the next important lever is
expert-clustered execution and popularity-sorted scheduling to maximize L2
reuse of hot expert weights.

## Architecture Phases

### Phase 0: Foundation (done)

**Status:** already landed.

What is already true:
1. Routed intermediate dimension is padded from `1856` to execution
   `1920`.
2. Routed resident expert weights now use execution-form residency more
   closely, without a duplicate row-major resident scale copy.
3. `P5`, `P12`, `P13`, and `P15` are active on the unified routed FP4 body.
4. Old dedicated runtime `P15` path is deleted.
5. The full TTFT matrix at `window=4096` completes again without the old
   `prefix128` invalid-token failure.

**Current baseline to optimize from:**
- `prefix4 / tail4`: `68.780 ms`
- `prefix128 / tail4`: `175.692 ms`
- `prefix4096 / tail4`: `2337.938 ms`

### Phase 0.5: Measurement / Probe Gate

**Goal:** lock down the exact current bottleneck before touching the
transport path.

**Current status: complete enough to start Phase 1.**

Histogram deliverable is landed and measured on the canonical surface:
- raw artifact:
  `artifacts/benchmarks/routed_phase05_prefix128_20260409T061848Z.jsonl`
- summary artifact:
  `artifacts/benchmarks/routed_phase05_prefix128_20260409T061848Z.summary.txt`
- benchmark stdout:
  `artifacts/benchmarks/routed_phase05_prefix128_20260409T061848Z.stdout.txt`

Measured histogram facts from `cold_prefill_prefix128` at `window=4096`:
- `23` routed expert layers
- per-layer active experts range: `104 .. 127`
- per-layer top-5 expert share range: `10.81% .. 33.07%`
- per-layer top-10 expert share range: `19.79% .. 49.74%`
- per-layer max tokens for a single expert: up to `75`
- per-layer max CTAs for a single expert: up to `10`
- aggregate over the whole forward:
  - overall top-5 expert share: `6.86%`
  - overall top-10 expert share: `12.65%`
  - max aggregate tokens for one expert: `256`
  - max aggregate CTAs for one expert: `43`

Interpretation:
- the routed distribution is skewed at the per-layer level
- it is not globally concentrated enough to assume large L2 wins up front
- Phase 2 remains measurement-gated; transport and overlap are still the
  correct priorities ahead of persistent expert clustering

Profiler status:
- live `ncu` profiling now works on this machine
- canonical routed profile artifacts:
  - report:
    `artifacts/profiles/routed_phase05_20260409T064530Z/ncu_cold_prefill_prefix128_unified_routed_fp4.ncu-rep`
  - csv:
    `artifacts/profiles/routed_phase05_20260409T064530Z/ncu_cold_prefill_prefix128_unified_routed_fp4.csv`
  - summary:
    `artifacts/profiles/routed_phase05_20260409T064530Z/ncu_cold_prefill_prefix128_unified_routed_fp4.summary.txt`

Measured `ncu` facts for the first profiled unified routed FP4 kernel:
- kernel:
  `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel`
- launch:
  - block size `(384, 1, 1)`
  - grid size `(15, 208, 1)`
- per-CTA resources:
  - `168` registers per thread
  - `41.984 KiB` shared memory per block allocated
  - occupancy limited by both registers and shared memory to `1` block/SM
- bottleneck metrics:
  - DRAM throughput: `9.19%` of peak sustained
  - L2 sector hit rate: `25.26%`
  - active warps: `24.96%` of peak sustained active
  - eligible warps per cycle: `0.098`
  - issue active: `8.65%` of peak sustained active
  - tensor-pipe active: `4.83%` of peak sustained elapsed
  - kernel time: `2.367 ms`

Measured transport-contract facts from the compile-time dump:
- artifact:
  `artifacts/benchmarks/phase05_transport_contract_20260409T065531Z.txt`
- common SM120 transport contract across the active routed family:
  - `tma_threads_ref = 128`
  - `math_threads = 256`
  - `total_threads_per_block_ref = 384`
  - full barrier type: `cutlass::arch::ClusterTransactionBarrier`
  - empty barrier type: `cutlass::arch::ClusterBarrier`
  - operand copy atoms: `Copy_Atom<SM75_U32x4_LDSM_N, integer_subbyte<4,false>>`
  - scale copy atoms: `Copy_Atom<UniversalCopy<float_ue4m3_t,float_ue4m3_t>, float_ue4m3_t>`
- per-profile builder/runtime staging facts:
  - `P5`: `ab_stages=4`, `sf_stages=4`, single-stage runtime bytes `34816`, builder multistage bytes `139264`
  - `P12`: `ab_stages=4`, `sf_stages=4`, single-stage runtime bytes `34816`, builder multistage bytes `139264`
  - `P13`: `ab_stages=9`, `sf_stages=9`, single-stage runtime bytes `17408`, builder multistage bytes `156672`
  - `P15`: `ab_stages=6`, `sf_stages=6`, single-stage runtime bytes `26112`, builder multistage bytes `156672`
- critical implication:
  - the literal CUTLASS multistage storage for every active routed profile exceeds the per-CTA opt-in shared-memory cap of `101376` bytes
  - Phase 1 cannot be a naive “port the builder storage exactly” rewrite
  - Phase 1 must instead preserve the traced copy atoms / barrier contract while compressing stage storage to a runtime-feasible staged transport plan

Measured standalone `P5` TMA smoke status on the current branch:
  - `testing/p5_swizzled_pipeline_test_120f` now passes both:
  - `tma_copy_only_a_raw PASS`
  - `tma_copy_only_a_global_object PASS`
  - `tma_copy_only_a_offset_tile PASS`
  - `tma_copy_only_b_raw_full128 PASS`
  - `tma_copy_only_b_raw_logical32_valid5 PASS`
  - `tma_fragment_a PASS`
- and now also passes the first full-math numeric step above the fragment
  boundary:
  - `tma_single_tile_k64_dispatch_rows_5`
  - `tma_single_tile_k64_dispatch_rows_4`
  - `tma_single_tile_k128_dispatch_rows_5`
- meaning:
  - the host-built `make_tma_copy(SM90_TMA_LOAD{}, ...)` producer path is valid for the routed FP4 `P5` operand contract
  - the host-built `TmaCopyA` object remains valid when copied into device
    global memory and dereferenced from the kernel, which is the key ownership
    model needed for routed per-expert weight views
  - the same bare CUTE TMA object can select a nonzero output-row CTA tile and
    a nonzero batch/expert coordinate from a larger `(M,K,L)` tensor; the raw
    stage bytes still match the selected row-major packed source tile
    byte-for-byte
  - a B-side bare CUTE TMA object can load a 128x128 logical `(N,K)` packed
    source tile into `SmemLayoutB` and its raw stage bytes match the row-major
    packed source model byte-for-byte
  - the same B-side path can also use a logical 32x128 source tensor while
    retaining the 128x128 TMA tile; rows outside the 32-row tensor arrive as
    zero in shared memory.  This is the direct smoke proof needed for low-row
    grouped activations; live B TMA does not require overreading adjacent
    grouped rows or first padding the global activation workspace to 128 rows.
  - the immediate consumer boundary also survives intact:
    `SmemLayoutA -> make_tiled_copy_A(...) -> retile_D(...) -> fp4_shift_A(...)`
  - a single-tile `P5` math path with `A` on TMA and `B/scales` on the existing
    path is numerically valid for the low-row `dispatch_rows=4/5` regimes
  - the full 128-K staged tile is numerically consumable by the traced P5
    math path: P5 has `size<2>(tCrA)=size<2>(tCrB)=2`, and the consumer runs
    the two 64-K register k-blocks from one 128-K producer stage
- the two actual bugs in the earlier smoke were:
  - wrong full-barrier wait phase: the producer-complete wait must use phase `0`, matching TRT's initial consumer-side `ab_full_mbar.wait(ab_phase)` contract
  - wrong physical-byte reference model: the valid stage-0 `P5` raw-byte reference for this smoke is the packed row-major source bytes, not the earlier `stage0_A(...)/2` reconstruction
- `cuobjdump --dump-sass` on the smoke confirms the kernel really lowers to TMA:
  - `UTMACCTL.PF`
  - `UTMALDG.3D`
  - `SYNCS.ARRIVE.TRANS64`
  - `SYNCS.PHASECHK.TRANS64`

Runtime `P5` TMA descriptor-ownership status:
- implemented an optional resident-weight sidecar:
  - public weight-view field: `FusedNvfp4WeightView::p5_tma_load_a`
  - internal descriptor type:
    `nemotron::routed_p5_tma::P5TmaLoadA`
  - owner: `MonolithicNvfp4ExpertWeights`
  - descriptors are built on the host with the same bare CUTE
    `make_tma_copy(SM90_TMA_LOAD{}, ...)` pattern as the smoke, copied once
    to device memory, and exposed as one device-resident descriptor pointer
    per expert
- descriptor creation is optional rather than part of `valid()`:
  small/ragged test matrices remain valid with a null descriptor; TMA-eligible
  matrices are expected to publish non-null per-expert descriptor pointers
- `fused_moe_prefill_test` now has a focused 128x128 resident-weight upload
  check that verifies the descriptor pointers are present in the host views
  and survive the same host-to-device weight-view-table copy used by prefill
- important boundary:
  descriptor residency / pointer publication is wired and the live unified
  routed MMA kernel now has an incremental `P5` A/weight-operand TMA path
  behind a narrow eligibility gate.
- current live `P5` transport state:
  - `A`/weight uses resident per-expert host-built CUTE TMA descriptors
  - `B`/activation now also has a live TMA transport path with cached
    split ownership:
    - `DeviceMoeLaunchPlan` now keeps host mirrors of the exact active CTA
      row metadata (`cta_count`, `cta_row_starts`, `cta_valid_rows`) and a
      build epoch that advances each time the plan is rebuilt
    - `DeviceNvfp4Matrix` now caches one device array of bare CUTE
      `P5TmaLoadB` descriptors per `(packed_input allocation, launch_plan,
      build_epoch)` triple
    - the unified `P5` kernel consumes that cached device descriptor array
      directly, so the live path no longer pays per-launch D2H metadata copies
      or per-launch descriptor malloc/free churn
  - scales are now split the same way as operands:
    - `SFA`/weight scales use resident per-expert host-built CUTE TMA
      descriptors, published through `FusedNvfp4WeightView::p5_tma_load_sfa`
    - `SFB`/activation scales now use the aligned-slab hybrid fast path:
      - descriptors are still built from the full grouped execution-scale
        tensor, not tiny per-CTA row-sliced descriptors
      - each active CTA aligns its `SFB` transport base down to the enclosing
        `128`-row slab, TMA-loads that slab, then remaps only the local
        `valid_rows` window into the small-row shared tensor the consumer uses
      - this preserves the cheap `8/16`-row compute geometry for `A/B` while
        honoring the tile-aligned `SFB` contract CUTLASS expects
    - the live `P5` kernel now issues `A+SFA` on the A-side producer barrier;
      `B+SFB` now share the aligned-slab TMA path too
  - decisive probe result behind that landing:
    - CUTLASS `Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFA/SFB(...)`
      matches our runtime `matmul_block_scales_data` execution layout exactly
      for `kSwizzled128x4`
    - use that gmem layout for `SFA/SFB` descriptors; do not try to model
      the compact execution-scale buffer with `SmemLayoutSFA/SFB`
  - this is still not the final roofline-ready form because the B cache is
    tied to the current runtime objects rather than a fully persistent staged
    workspace, but the immediate launch-local rebuild overhead is now gone
- current live-gate status:
  - `fused_moe_prefill_test PASS`
  - `multi_turn_prefix_reuse_test PASS`
  - `testing/p5_swizzled_pipeline_test_120f` with
    `NEMOTRON_RUN_P5_TMA_SMOKE=1 PASS`
  - single-case canonical-capacity TTFT smoke:
    - after the first live B-side transport landing:
      - `cold_prefill_prefix128 = 142.097 ms` median
      - `cached_committed_head_prefix128_tail4 hot-prefix = 66.129 ms` median
      - `cached_global_root_prefix128_tail4 hot-prefix = 65.854 ms` median
    - after moving B-descriptor ownership/caching into the plan + matrix
      layer:
      - `cold_prefill_prefix128 = 142.474 ms` median
      - `cached_committed_head_prefix128_tail4 hot-prefix = 66.007 ms` median
      - `cached_global_root_prefix128_tail4 hot-prefix = 66.000 ms` median
    with `resolved_runtime_moe_prefill_window_tokens=4096`
  - interpretation:
    - removing the launch-local descriptor rebuild eliminates that known
      control overhead, but it does not materially move cold TTFT
    - the remaining gap is now dominated by the kernel body / staging path,
      which is the right point to hand over to `ncu`
  - direct grouped-kernel profiling surface:
    - `benchmarks/nano_moe_prefill/nano_routed_up_p5_grouped_bench`
      now isolates the routed FC1 grouped `P5` surface directly
    - current measured state on that harness:
      - `prefix_tokens=128`
      - `selection_count=768`
      - `selected_token_tile=8`
      - `b_tma_cached=yes`
      - `sfb_tma_cached=no`
      - `sfb_tma_launch_compatible=no`
      - `cold_ms=0.394`
      - `hot_mean_ms=0.591`
      - `hot_tflops=13.420`
      - `hot_weight_gib_per_s=585.850`
    - the same harness now confirms finite BF16 outputs on the isolated path,
      so it is a valid profiler surface instead of just a transport smoke
    - `P5`-only explicit slab experiment:
      - the benchmark can now force `token_tile=128` /
        `expert_row_alignment=128` without changing the default runtime path
      - this makes `SFB` TMA legal on the isolated surface:
        - `selected_token_tile=128`
        - `sfb_tma_cached=yes`
        - `sfb_tma_launch_compatible=yes`
      - the direct harness now reports median / p90 as well as mean, because
        single-run jitter was large enough to hide the real signal:
        - baseline auto-tile (`selected_token_tile=8`):
          - `hot_mean_ms=0.334`
          - `hot_median_ms=0.315`
          - `hot_p90_ms=0.317`
        - forced slab (`selected_token_tile=128`):
          - `hot_mean_ms=0.413`
          - `hot_median_ms=0.336`
          - `hot_p90_ms=0.773`
      - implication:
        - `128`-row slabs are sufficient to make the `SFB` transport contract
          legal
        - but this workload averages only `~6` valid rows per CTA, so a forced
          `128`-row slab overfetches heavily and is still slightly slower even
          after `SFB` becomes TMA-legal
        - do not promote the slab experiment to the default launch shape
  - post-stabilization profiling boundary:
    - the default full-model path now intentionally replays multi-token direct
      MoE rows through the single-row decode contract for correctness, so
      default TTFT smoke numbers are no longer a valid native-kernel profiler
      surface
    - use the explicit profiling override only for measurement:
      `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1`
    - with that override plus
      `--moe-prefill-window-tokens 133 --case cold_prefill_prefix128`, the
      live native routed path is visible again and currently measures
      - before live scale TMA:
        - `cold TTFT = 130.018 ms` median
      - after live `SFA/SFB` scale TMA:
        - `cold TTFT = 127.436 ms` median
      with:
      - `expert native multi-token runs = 23`
      - `expert row replay runs = 0`
  - first post-cleanup `ncu` read on that explicit profiling surface:
    - before removing the conservative full-stage `B` clear:
      - `Memory Throughput = 473.5 GB/s`
      - `Max Bandwidth = 26.83%`
      - `L2 Hit Rate = 43.82%`
      - `One or More Eligible = 11.70%`
      - `Issued Warp / Scheduler = 0.12`
      - `Warp Cycles / Issued Instruction = 25.58`
    - after removing the conservative full-stage `B` clear:
      - `Memory Throughput = 658.9 GB/s`
      - `Max Bandwidth = 37.34%`
      - `L2 Hit Rate = 65.17%`
      - `One or More Eligible = 15.29%`
      - `Issued Warp / Scheduler = 0.15`
      - `Warp Cycles / Issued Instruction = 19.44`
    - interpretation:
      - the `B`-stage clear was real overhead, not noise
      - the live kernel is still far from bandwidth-bound, but the next stall
        targets should be barrier structure / eligibility / occupancy, not
        descriptor transport
  - first post-scale-TMA `ncu` read on the same explicit profiling surface:
    - `Block Size = (384, 1, 1)`
    - `Grid Size = (15, 208, 1)`
    - `launch__registers_per_thread = 168`
    - `launch__shared_mem_per_block_allocated = 42.112 KB`
    - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed = 41.32%`
    - `lts__t_sector_hit_rate.pct = 68.82%`
    - `sm__warps_active.avg.pct_of_peak_sustained_active = 24.72%`
    - `smsp__warps_eligible.avg.per_cycle_active = 0.2145`
    - `smsp__issue_active.avg.pct_of_peak_sustained_active = 16.15%`
    - `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed = 12.38%`
    - profile artifact set:
      - `artifacts/profiles/routed_phase05_post_scale_tma_20260409T231946Z/ncu_cold_prefill_prefix128_unified_routed_fp4_post_scale_tma.ncu-rep`
      - `artifacts/profiles/routed_phase05_post_scale_tma_20260409T231946Z/ncu_cold_prefill_prefix128_unified_routed_fp4_post_scale_tma.csv`
      - `artifacts/profiles/routed_phase05_post_scale_tma_20260409T231946Z/ncu_cold_prefill_prefix128_unified_routed_fp4_post_scale_tma.summary.txt`
    - interpretation:
      - scale TMA moved the live path forward, but the kernel is still
        scheduler/eligibility limited rather than bandwidth limited
      - the next profitable work is inside the remaining consumer/control
        structure around the unified P5 body, not another transport-object
        rewrite
  - single-producer cleanup on the live `P5` fast path:
    - replaced the prior two-producer-warps / two-transaction-barriers
      structure with one elected producer thread and one shared transaction
      barrier for the full `A/B/SFA/SFB` stage
    - this is closer to CUTLASS
      `sm120_blockscaled_mma_tma.hpp`, which issues all four TMA copies under
      one producer barrier
    - correctness remained green:
      - `fused_moe_prefill_test`: PASS
      - `multi_turn_prefix_reuse_test`: PASS
      - `NEMOTRON_RUN_P5_TMA_SMOKE=1 p5_swizzled_pipeline_test`: PASS
    - explicit unsafe native prefix128 measurement moved only slightly:
      - before single-producer cleanup: `127.29-127.44 ms`
      - after single-producer cleanup: `127.118 ms`
    - interpretation:
      - this removes a real structural divergence from the CUTLASS producer
        contract, but it is not the missing large win
      - the dominant limiter is still deeper in the consumer-side wait/copy
        path or in the way the full-model profiling surface reaches the routed
        kernel
  - profiling-surface warning after the single-producer cleanup:
    - the full-model `ncu` surface is no longer a trustworthy direct read of
      the unified routed `P5` kernel
    - a whole-model metric capture on the unsafe native prefix128 bench now
      intermittently aborts early with driver `UnknownError` while profiling
      unrelated kernels, and the surviving report rows prominently show
      `cutlass3x_sm120...` kernels before the grouped routed `P5` body
    - implication:
      - do not treat whole-model `ncu` on the unsafe override as the canonical
        optimization loop for the grouped routed kernel anymore
      - the next measurement step should be a direct grouped-kernel harness
        around `LaunchPlannedPackedInputMatVec{,Bf16}` or an equivalent focused
        `fused_moe_prefill` microbench so the unified routed `P5` body can be
        profiled in isolation
  - occupancy diagnosis on the same explicit native P5 profiling surface:
    - `Registers Per Thread = 168`
    - `Static Shared Memory Per Block = 40984 B`
    - `Shared Memory Configuration Size = 65536 B`
    - `Block Limit Registers = 1`
    - `Block Limit Shared Mem = 1`
    - `Theoretical Occupancy = 25%`
    - `Achieved Occupancy = 25.03%`
    - `Achieved Active Warps Per SM = 12.02`
    - interpretation:
      - live P5 is hard-capped at one CTA per SM by footprint, not by launch
        overhead
      - optimizing descriptor ownership was necessary, but it is no longer the
        dominant limiter
  - first direct grouped-kernel `ncu` read on the isolated harness:
    - `dram__throughput.avg.pct_of_peak_sustained_elapsed = 53.58%`
    - `lts__t_sector_hit_rate.pct = 61.97%`
    - `Achieved Occupancy = 24.89%`
    - `Registers Per Thread = 164`
    - `Static Shared Memory Per Block = 40.97 KiB`
    - `One or More Eligible = 14.59%`
    - `Eligible Warps Per Scheduler = 0.20`
    - `Issued Warp Per Scheduler = 0.15`
    - `Warp Cycles Per Issued Instruction = 20.42`
    - conclusion:
      - the isolated grouped `P5` kernel is still one-CTA-per-SM limited by
        registers plus shared memory
      - the next optimization target is warp eligibility / consumer-side stall
        reduction, not another descriptor-ownership tweak
  - first post-hybrid-`SFB` direct grouped-kernel `ncu` read:
    - profile artifact set:
      - `artifacts/profiles/p5_hybrid_sfb_20260410T031056Z/ncu_p5_grouped_prefix128.ncu-rep`
      - `artifacts/profiles/p5_hybrid_sfb_20260410T031056Z/ncu_p5_grouped_prefix128.csv`
      - `artifacts/profiles/p5_hybrid_sfb_20260410T031056Z/ncu_p5_grouped_prefix128.summary.txt`
    - measured facts:
      - `launch__registers_per_thread = 168`
      - `launch__shared_mem_per_block_allocated = 46.21 KiB`
      - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed = 56.48%`
      - `lts__t_sector_hit_rate.pct = 59.09%`
      - `sm__warps_active.avg.pct_of_peak_sustained_active = 25.40%`
      - `smsp__warps_eligible.avg.per_cycle_active = 0.193`
      - `smsp__issue_active.avg.pct_of_peak_sustained_active = 14.47%`
      - `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed = 11.91%`
    - conclusion:
      - the hybrid `SFB` path improves transport utilization a bit, but it
        does not materially change the scheduler-limited shape of the kernel
      - relative to the pre-hybrid isolated read, DRAM throughput is up a few
        points, while eligible warps and issue activity are effectively flat
      - the next leverage remains consumer-side wait/copy/store and warp
        eligibility, not more scale-transport legality work
      - dominant post-hybrid stall buckets are now:
        - `sleeping = 6.83`
        - `barrier = 3.83`
        - `wait = 2.81`
        - `long_scoreboard = 1.72`
        - `short_scoreboard = 1.22`
      - that points directly at pipeline structure and synchronization as the
        next code target
  - first direct grouped-kernel source-counter read on the same harness:
    - `706560` excessive global sectors (`40%` of `1788480`)
    - `1532160` excessive shared wavefronts (`10%` of `15240960`)
    - branch ratio is negligible (`0.13%`), so control divergence is not the
      dominant issue
    - implication:
      - with `A`, `B`, and `SFA` already on TMA in this harness, the remaining
        obvious uncoalesced hot-path traffic was the `SFB` fallback plus the
        surrounding consumer/store scaffolding
      - that profiler snapshot is now historical context: `SFB` no longer
        falls back on the default `P5` launch geometry after the aligned-slab
        hybrid remap landed
      - the next structural win is now consumer/store-side eligibility work,
        not more `SFB` descriptor archaeology
  - source-counter result for the forced `128`-row slab:
    - excessive global sectors drop from `40%` (`706560 / 1788480`) to `8%`
      (`61440 / 807300`)
    - excessive shared wavefronts stay similar (`10%` -> `9%`)
    - issue activity does not improve (`14.49%` -> `14.00%`)
    - implication:
      - making `SFB` legal/coalesced is real progress, not a phantom
      - but after that fix, the next limiter is deeper in the consumer-side
        wait/copy/store structure, not in raw global-memory coalescing
  - standalone `SFB` offset smoke on the existing `P5` TMA harness:
    - added a new gated case under
      `NEMOTRON_RUN_P5_TMA_SMOKE=1 p5_swizzled_pipeline_test`
    - exact question:
      - can a full-tensor `TMA_SFB` descriptor service a small-row offset
        (`row_start=8`, `valid_rows=8`) by shifting the tensor in device code
        with `cute::domain_offset(...)`, without rebasing the descriptor
        pointer per CTA?
    - definitive result:
      - no, not with the current `get_tma_tensor(...) -> domain_offset(...) ->
        local_tile(...) -> get_slice(...).partition_S(...)` path
      - the smoke fails deterministically at
        `row=0 col=64 tma=0 ref=137`
    - implication:
      - the old per-CTA sliced-descriptor model is not the only problem
      - a naive full-tensor `domain_offset` rewrite is also insufficient
      - follow-up source check:
        - the local SM120 array collective
          (`sm120_blockscaled_mma_array_tma.hpp`) does not use the SM100
          `make_tma_atom_* + tma_partition(...)` contract for `SFB`
        - it uses `make_tma_copy(...) -> get_tma_tensor(...) -> local_tile(...)`
          and then `block_tma_sfb.partition_S(...)`
      - follow-up result:
        - the grouped shadow-descriptor path is now also modeled in the
          standalone harness
        - the local helper-built `SFB` TMA object matches the
          CUTLASS-selected `GmemTiledCopySFB` contract byte-for-byte:
          `tma_copy_only_sfb_builder_contract PASS`
        - the real bug was in our helper, not CUTLASS:
          `MakeP5ScaleLayoutSFB(token_rows, input_cols)` was passing
          `tile_atom_to_shape_SFB(make_shape(token_rows, input_cols, 1))`
        - for `tile_atom_to_shape_SFB`, a rank-3 problem shape is interpreted
          as `(M, N, K)`, so that helper built a layout for
          `N=input_cols, K=1` instead of `N=token_rows, K=input_cols`
        - after fixing the helper to pass
          `make_shape(1, token_rows, input_cols, 1)`:
          - `tma_copy_only_sfb_execution_layout_contract PASS`
          - `tma_copy_only_sfb_tileindex_row0 PASS`
          - `tma_copy_only_sfb_tileindex_row128 PASS`
          - `tma_copy_only_sfb_grouped_tileindex_row128 PASS`
        - the exact host-side alias probe now reports:
          `layout=(((_32,_4),2),((_16,_4),2),(_1,1)):
          (((_16,_4),1024),((_0,_1),_512),(_0,2048))`
          with `cosize=2048`
      - consequence:
        - `tma_partition(...)` is the wrong abstraction for this exact
          collective, and the grouped descriptor-update lifecycle was not the
          missing fix
        - the aligned `SFB` gmem contract is now proven correct for the
          swizzled execution-scale buffer
        - the remaining failure in the old harness is the intentionally
          unaligned direct-offset path:
          `tma_copy_only_sfb_offset_row8 FAIL row=24 col=0 tma=0 ref=33`
        - the new hybrid smoke now proves the right fix for the active routed
          launch geometry:
          - `tma_copy_only_sfb_hybrid_row8_valid8 PASS`
          - aligned slab load + logical-row remap is sufficient
        - the corrected runtime helper now reaches the default grouped live
          path too:
          - `nano_routed_up_p5_grouped_bench --prefix-tokens 128`
          - `selected_token_tile=8`
          - `sfb_tma_cached=yes`
          - `sfb_tma_launch_compatible=yes`
          - hot-path reruns are currently around
            `hot_mean_ms=0.536..0.649`, `hot_median_ms=0.410..0.457`
        - the forced-slab path is still slower even though it is fully legal:
          - `--force-token-tile 128`
          - `selected_token_tile=128`
          - `sfb_tma_cached=yes`
          - `sfb_tma_launch_compatible=yes`
          - `hot_mean_ms=0.630`, `hot_median_ms=0.479`
  - standalone `SFB` consumer-copy probe on the aligned slab:
    - added a separate gated probe under
      `NEMOTRON_RUN_P5_SFB_DIRECT_SLICE_PROBE=1` in
      `p5_swizzled_pipeline_test_120f`
    - exact question:
      - after loading the aligned `128`-row `SFB` slab with TMA, can the live
        `SmemCopyAtomSFB` consumer path read a row-offset source view directly
        from that slab, without the shared-memory remap step?
    - definitive result:
      - yes, but only inside the first `32`-row chunk of the aligned slab
      - exact passing cases:
        - `offset=0, valid_rows=8`: `PASS`
        - `offset=1,2,4,8,16,17,24`, `valid_rows=8`: `PASS`
      - exact failing cases:
        - `offset=25, valid_rows=8`: fragment mismatches
        - `offset=30, valid_rows=8`: fragment mismatches
        - `offset=31, valid_rows=8`: fragment mismatches
        - `offset=24, valid_rows=9`: fragment mismatches
        - `offset=32, valid_rows=8`: `ERROR misaligned address`
        - `offset=64, valid_rows=8`: `ERROR misaligned address`
    - implication:
      - the direct-slice path is not generically illegal; the live failure was
        not caused by "any nonzero row offset"
      - the current source-view legality rule is empirical and narrow:
        - safe only when `row_offset < 32` and `row_offset + valid_rows <= 32`
      - outside that first `32`-row chunk, the current
        `domain_offset(...) -> partition_S(...) -> copy(...)` source path is
        not usable for `SmemCopyAtomSFB`
    - live-kernel follow-up:
      - the architectural branch in `fused_moe_prefill.cu` now keeps the
        direct-slice experiment only for that proven-safe window and restores
        the old shared-memory remap everywhere else
      - correctness is green again:
        - `fused_moe_prefill_test`: PASS
        - `multi_turn_prefix_reuse_test`: PASS
      - direct grouped `P5` bench after the guard:
        - `selected_token_tile=8`
        - `sfb_tma_cached=yes`
        - `sfb_tma_launch_compatible=yes`
        - `sfb_direct_ctas=8`
        - `sfb_direct_slice_safe_ctas=24`
        - `sfb_remap_fallback_ctas=96`
        - `hot_mean_ms=0.553`
        - `hot_median_ms=0.374`
    - consequence:
      - this is real architectural progress, but it does not remove the
        dominant `SFB` remap/sync cost on the current `prefix128` grouped
        surface because `96 / 128` CTAs still require the fallback path
      - the next exact target is no longer "is direct slice legal?" but "how
        do we eliminate the remap for row offsets beyond the first `32`
        rows?" The most plausible next approaches are:
        - a consumer-local register assembly path from the aligned slab, or
        - a rebased/chunked source-view construction that respects the
          swizzle-period boundary instead of crossing it
  - traced thread-shape fact from the existing transport-contract dump:
    - `math_threads = 256`
    - `math_warps = 8`
    - `tma_threads_ref = 128`
    - `total_threads_per_block_ref = 384`
    - this means the traced `384`-thread P5 shape is not arbitrary padding; it
      is a `256`-thread math block plus `128` reference TMA threads
    - our current live kernel now uses that TMA budget for `A/B/SFA/SFB`; the
      active routed geometry reaches `SFB` through the aligned-slab remap path
  - rejected shortcut:
    - a targeted occupancy experiment that combined:
      - `__launch_bounds__(384, 2)` on the unified routed kernel, and
      - preferred shared-memory carveout `100` on the P5 programmatic launch
    - built cleanly, but the unsafe native prefix128 profiling bench stopped
      making forward progress on the first measured iteration and had to be
      killed
    - conclusion:
      - do not retry brute-force occupancy pressure as the next step
      - preserve the known-good post-`B`-clear-removal kernel body
      - the next TRT-aligned move is to spend the reserved TMA thread budget on
        the missing scale path, not to force a smaller launch shape

Immediate implication for Phase 1:
- stop questioning whether CUTE TMA works for routed FP4 `P5`
- the now-proven `A`-operand producer+consumer path has been lifted into the
  live unified routed kernel for eligible `P5` tiles. The next boundary is
  measurement plus the B-operand live contract, not more A-side smoke.
- resolved shape seam:
  - the host TMA object uses the traced top-level `128x128` operand tile
  - the traced P5 math tile is `128x32x64`
  - the P5 register consumer has two K blocks per loaded stage
  - therefore the live P5 staged-K step is now `128`, matching the producer
    tile; do not re-split P5 back into artificial 64-K load iterations
- treat B-side TMA as transport-proven and now minimally wired in the live
  kernel:
  the smoke covers both the full 128-row B tile and the live-like
  logical-32 / valid-5 activation case with TMA out-of-bounds zero fill, and
  the live `P5` path now consumes cached per-CTA `B` descriptors via the
  launch-plan + matrix ownership split.
  The next boundary is no longer descriptor ownership; it is direct
  measurement of the live kernel body and then removing the remaining known
  temporary costs inside that body.
- keep one temporary correctness-landing caveat explicit before reading too
  much into the first transport measurements:
  - the old thread-coded `B` packed-data loop still exists structurally around
    the `if (!use_p5_tma_b)` fallback while the scale scatter runs. Confirm
    with SASS / `ncu` whether the remaining control flow is negligible on the
    TMA path or whether it still needs a dedicated scale/TMA producer split.
- next TRT-aligned kernel step after the scale-TMA landing:
  - the missing P5 scale transport is no longer the blocker
  - next measurement boundary:
    - profile the live `A+SFA` / `B+SFB` path with `ncu` on the explicit
      unsafe native profiling surface
    - check whether the remaining stall mix is still barrier/eligibility
      dominated or has shifted toward the consumer-side scale/copy path
  - next implementation boundary:
    - remove or isolate any residual thread-coded scale/control scaffolding
      that still structurally surrounds the TMA fast path
    - keep non-`P5` profiles on the old path until they are replaced by the
      same unified swizzled/TMA contract
  - only after reading that post-scale-TMA profile should we revisit
    occupancy/launch-shape work
- do not go back down to raw descriptor/PTX debugging unless the runtime-like integration breaks at a new boundary

Live-integration boundary decisions:
- the standalone proof deliberately uses the same bare TRT-style
  `make_tma_copy(SM90_TMA_LOAD{}, tensor_A, SmemLayoutA{}(_,_,0), ...)`
  pattern used by the local SM120 reference.  Keep it unless the live path
  proves a CTA-coordinate / view-offset bug.
- if live per-expert weights or multi-CTA coordinates fail, first compare the
  exact selected packed source tile against the raw stage-0 bytes.  Only suspect
  the higher-level `make_tma_atom_*` / CTA-v-coordinate wrapper after that
  raw-byte check fails.
- the raw-byte proof also established an important storage fact: the physical
  `SmemLayoutA` stage allocation is 1 byte per logical FP4 element, while the
  packed TMA payload occupies the first half of the per-stage physical byte
  span. Account for that 2x physical stage footprint when sizing the eventual
  multistage pipeline.

Interpretation:
- the current unified routed kernel is not bandwidth-limited
- it is also not tensor-core-limited
- it is issue-starved and underfed, with too few eligible warps and too much
  front-end / transport overhead per useful MMA cycle
- this strongly supports the next planned direction:
  direct-to-swizzled transport, multistage overlap, and producer/consumer warp
  specialization before any attempt to chase persistent expert clustering

**Work:**
1. Run `ncu` on the current unified routed FP4 path for representative
   `P5/P12/P13/P15` regimes.
2. Capture:
   - DRAM throughput
   - L2 hit rate
   - warp stall reasons
   - occupancy / active warps
   - shared-memory and register usage
3. Dump the actual routed expert selection histogram for the canonical
   `prefix128` surface:
   - per-layer expert popularity
   - tokens per expert
   - CTA count per expert under the current launch plan
4. Use TRT/CUTLASS source plus compile-time probes to recover the exact SM120
   staged-copy / barrier / stage-count contract needed for the TMA rewrite.
   Status: done for the current unified routed profile family.
5. When profiling the live `P5` TMA path, remove and then remeasure the two
   known temporary costs:
   - the conservative full-stage `B` smem clear before TMA
   - any residual control-flow overhead from the fallback `B` packed-data loop
6. Run `ncu` on the cached-descriptor live `P5` path and capture the first
   transport-focused before/after counters:
   - DRAM throughput
   - L2 hit rate
   - warp stall reasons
   - active warps / occupancy
   - shared-memory instructions and barriers

**Reference points to start from, not rediscover:**
- External references:
  - Colfax, *Mastering the NVIDIA TMA*:
    `https://research.colfax-intl.com/tutorial-hopper-tma/`
    This is the canonical end-to-end explanation of the host
    `make_tma_copy(...)` + kernel `copy(tma.with(barrier), ...)` pattern.
  - Colfax, *CUTLASS tutorial: sub-byte GEMM on NVIDIA Blackwell GPUs*:
    `https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/`
    This is the most relevant external explanation of sub-byte / FP4 TMA
    behavior on Blackwell, including the packed-to-packed versus auto-unpack
    tensor-map modes.
  - CUTLASS issue `#2906`:
    `https://github.com/NVIDIA/cutlass/issues/2906`
    Track this as the main alignment hazard reference for SM120 NVFP4 TMA:
    descriptor alignment and scale-SMEM alignment are both easy footguns.
  - CUTLASS issue `#3096`:
    `https://github.com/NVIDIA/cutlass/issues/3096`
    Track this as the main Blackwell grouped-FP4 correctness reference,
    especially the `compute_120f` build caution.
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/sm120_blockwise_gemm/sm120_utils.cuh`
  lines `271-307`: `TMA_A`, `TMA_B`, `TMA_SFA`, `TMA_SFB`, `SmemCopyAtomSF`,
  and `TmaTransactionBytes*`.  This is where the local SM120 path confirms
  that the current TRT/CUTLASS builder still uses `SM90_TMA_LOAD` copy atoms
  on SM120.
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/sm120_blockwise_gemm/sm120_fp8_moe_gemm_1d1d.cuh`
  lines `395-530`: `load_ab()` and `load_sf()` producer paths, barrier usage,
  and `as_position_independent_swizzle_tensor(...)`.
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/sm120_blockwise_gemm/sm120_fp8_moe_gemm_1d1d.cuh`
  lines `190-253`: `SM120BlockScaledMoeScheduler`.
- Local Phase 0.5 harnesses:
  - `proj-2026-04-05-1704/run_phase05_routed_histogram.sh`
  - `proj-2026-04-05-1704/summarize_routed_phase05_histogram.py`
  - `proj-2026-04-05-1704/run_phase05_routed_ncu.sh`
  - `proj-2026-04-05-1704/summarize_phase05_ncu_csv.py`

**Exit criteria:**
- we have one measured bottleneck summary for the current unified routed FP4
  path
- we know whether the expert distribution is skewed enough for Phase 2 to be
  worth doing, and which experts dominate routed traffic
- we have the exact source/probe facts needed to implement the SM120 staged
  transport contract without guesswork
- we know that the literal builder multistage storage is too large for the
  per-CTA shared-memory cap, so Phase 1 must use a compressed runtime stage
  plan instead of a direct storage clone

### Phase 1: Weight Loading Pipeline (biggest lever)

**Goal:** Get the weight bytes from DRAM to the Tensor Core as fast as
the hardware allows.  This phase targets the corrected `~9.7 ms` routed
MoE weight+scale floor, not the older `8.6 ms` weight-only floor. The
implementation constraint is now also explicit: keep the traced SM120 copy
atoms and barrier contract, but do not try to replicate the builder’s full
multistage storage literally when it exceeds the `101376`-byte per-CTA limit.

**Why this is first:** At prefix128, routed MoE weight transport dominates the
cost surface.  Every meaningful improvement in effective bandwidth moves TTFT.
No smaller compute-side cleanup has comparable leverage.

#### 1a. TMA for Operand Loads

Replace the byte-level swizzled scatter with TMA (Tensor Memory
Accelerator) loads.  TMA loads from global memory directly into swizzled
shared memory in hardware, without consuming SM instruction slots.

- Current: threads compute swizzled offsets, issue individual byte stores
  to smem.  This saturates the SM instruction pipeline on address
  computation instead of letting DRAM bandwidth be the bottleneck.
- Target: one TMA descriptor per K-tile, issued by a single producer
  thread.  Frees all other threads for MMA compute.

The exact staged-copy primitive and barrier structure are now pinned down by the
Phase 0.5 probe. The remaining design choice is the compressed runtime stage
count/layout that preserves those contracts while fitting within the per-CTA
shared-memory limit.

Phase 1a implementation rule:
- prototype transport through the same CUTE contract TRT uses:
  - host: `make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout, ...)`
  - kernel: `copy(tma.with(barrier), partition_S(...), partition_D(...))`
- do not start from raw `CUtensorMap` descriptors or raw
  `SM90_TMA_LOAD_*::copy(...)` wrappers unless the purpose of the test is to
  debug CUTE itself

FP4-specific hazards that must be treated as fixed constraints:
- sub-byte TMA mode selection matters:
  - packed-to-packed FP4 transport corresponds to the `16U4_ALIGN8B` contract
  - mixed auto-unpack modes such as `16U4_ALIGN16B` are a different path
- if we do descend to raw tensor-map descriptors during debugging, remember the
  documented sub-byte constraints:
  - GMEM base alignment is stricter than generic byte tensors
  - logical K/box dimensions are constrained for FP4 tensor maps
  - only the documented sub-byte swizzle modes are legal
- TMA descriptor storage must respect the stricter alignment used by
  `prefetch.tensormap` on Blackwell-class paths
- scale-factor SMEM should be treated as requiring at least `128`-byte
  alignment on the FP4 path
- build/runtime validation for SM120 grouped FP4 should keep the
  `compute_120f` caution in view; do not assume `compute_120` is enough just
  because dense FP4 kernels compile

Concrete source anchors for this phase:
- `sm120_utils.cuh` lines `271-307` for descriptor and transaction-byte types
- `sm120_fp8_moe_gemm_1d1d.cuh` lines `395-530` for producer-side load methods
- `sm120_fp8_moe_gemm_1d1d.cuh` lines `190-253` for the scheduler shape that
  will later matter for persistent routing work

TMA requires:
- the exact SM120 copy atom / descriptor contract used by the traced builder
- tensor map descriptors set up on the host before launch
- barrier-synchronized producer/consumer warp specialization

This is a significant kernel restructuring but it directly follows the local
SM120 TRT/CUTLASS producer path shape in `sm120_utils.cuh` and
`sm120_fp8_moe_gemm_1d1d.cuh`.

**Expected impact:** Removes the current thread-coded staging path as the main
front-end bottleneck and enables a real bandwidth-limited experiment.  Exact
gain is measurement-gated.

#### 1b. Multi-Stage Pipeline

Double-buffer (or triple-buffer) the smem K-tiles so that TMA loads for
K-tile `i+1` overlap with MMA compute on K-tile `i`.

- Current: load all → `__syncthreads` → compute all → `__syncthreads`.
  DRAM latency (~400-600 ns) is fully exposed on every K-tile boundary.
- Target: producer warps fill the next stage via TMA while consumer warps
  run MMA on the current stage.  Pipeline barriers replace `__syncthreads`.

With 15 K-tiles per expert (at K=1920, TileK=128): 14 of 15 loads overlap
with compute.  Only the first load is latency-exposed.

Measured single-stage runtime storage from Phase 0.5 is:
- `P5/P12`: `34816` bytes per stage
- `P13`: `17408` bytes per stage
- `P15`: `26112` bytes per stage

This means the feasible compressed runtime stage counts are roughly:
- `P5/P12`: at most `2` stages (`69632` bytes) before extra barrier/epilogue
  state; `3` stages would already be `104448` bytes and exceed the cap
- `P13`: up to `5` stages in principle on raw stage bytes, though barrier and
  epilogue state will reduce that headroom
- `P15`: at most `3` stages on raw stage bytes; `4` stages would already be
  `104448` bytes and exceed the cap

So Phase 1b should start from a compressed `2`-stage design for `P5/P12`, with
profile-specific expansion only where the measured storage budget allows it.

**Expected impact:** Hides most K-tile load latency behind MMA.  Exact gain is
measurement-gated.

Phase 1b should now proceed with an explicit software-pipelined K-loop
checklist on the isolated grouped `P5` surface before any profile fan-out:

1. Use the direct grouped harness as the canonical measurement loop:
   - kernel: `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel`
   - bench:
     `benchmarks/nano_moe_prefill/nano_routed_up_p5_grouped_bench`
   - do not use whole-model `ncu` as the primary read for this step

2. Start from the current single-stage `P5` body in
   `runtime/src/backend/fused_moe_prefill.cu` and split the K-loop into three
   explicit regions:
   - prologue: issue the first stage load only
   - steady state: issue stage `next` while consumers drain stage `curr`
   - epilogue: drain the final loaded stage after producer issue stops

3. Introduce a `2`-entry stage ring for the compressed runtime transport:
   - operand/storage state for `A`, `B`, `SFA`, `SFB`
   - per-stage full/empty barrier state
   - stage index rotation `curr = iter & 1`, `next = (iter + 1) & 1`
   - confirm first whether the current `SmemLayout*` types already carry a
     pipeline-stage mode and the runtime allocation is merely using one live
     stage, or whether the collective/builder parameters must be changed to
     request `2` stages explicitly
   - keep `kMacroTileK = 128` for the first pipeline attempt unless the
     measured runtime stage footprint changes; the current `34816` bytes/stage
     number implies `2` stages at `K=128` still fit (`69632` bytes) and do not
     require halving `K` to `64`

4. Keep the current proven transport contract intact while changing only the
   schedule:
   - `A/B/SFA/SFB` continue to use the existing CUTE TMA objects
   - the hybrid `SFB` aligned-slab remap remains the legal small-row path
   - do not re-open descriptor/layout questions during the pipeline step

5. Move all TMA issue responsibility into producer warps/threads only:
   - consumers never issue transport
   - producers prefetch descriptors once, then loop over stage issue
   - producers call `arrive_and_expect_tx(...)` per stage, not per full-K body
   - start from the current single elected async producer thread; only widen to
     a dedicated producer warp if measurement shows the single-thread issue path
     is insufficient for the `2`-stage steady state

6. Convert the consumer side from whole-body waits to per-stage waits:
   - wait only on `curr`
   - copy fragments from the current stage into registers
   - run the MMA micro-loop for the current stage
   - signal stage completion so the producer can safely reuse that slot

7. Remove whole-CTA synchronization from the steady-state K-loop where the
   staged barriers already provide ordering:
   - `__syncthreads()` should remain only where required for non-pipelined
     shared state, not as the main stage boundary
   - the steady-state target is zero whole-CTA syncs; at most one prologue
     fence may remain before the first stage is made visible to consumers
   - the post-hybrid `ncu` stall read says `sleeping`, `barrier`, and `wait`
     are the first synchronization targets to reduce

8. Validate each substep before proceeding:
   - correctness:
     `testing/fused_moe_prefill_test`
   - runtime regression:
     `testing/multi_turn_prefix_reuse_test`
   - direct grouped perf surface:
     `benchmarks/nano_moe_prefill/nano_routed_up_p5_grouped_bench --prefix-tokens 128`
   - if transport legality is touched, rerun
     `NEMOTRON_RUN_P5_TMA_SMOKE=1 testing/p5_swizzled_pipeline_test_120f`

9. Reprofile after the first working `2`-stage pipeline lands:
   - compare against the current post-hybrid baseline:
     - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed = 56.48%`
     - `smsp__warps_eligible.avg.per_cycle_active = 0.193`
     - `smsp__issue_active.avg.pct_of_peak_sustained_active = 14.47%`
   - success criterion for Phase 1b is not just more DRAM throughput; it must
     reduce the dominant `sleeping` / `barrier` / `wait` stall buckets too

10. Only after the `P5` `2`-stage loop is stable and measurably better should
    the same structure be ported to `P12/P13/P15`

Current Phase 1b experiment record:
- Attempt A: coarse `2`-stage ring on the live `P5` body, with producer
  pre-issuing stage `next` and consumers draining stage `curr`
  - result: correctness passed, but the direct grouped bench did not show a
    stable win and `ncu` stayed effectively flat against the post-hybrid
    baseline
  - baseline vs Attempt A (`prefix128`, isolated grouped `P5`, direct `ncu`):
    - eligible warps/cycle: `0.193180 -> 0.193126`
    - issue active: `14.471734% -> 14.467279%`
    - sleeping stall ratio: `6.833301 -> 6.757392`
    - barrier stall ratio: `3.829301 -> 3.829051`
    - wait stall ratio: `2.813678 -> 2.813553`
  - conclusion: stage-ring transport alone did not buy a meaningful pipeline
    win; the dominant synchronization picture was unchanged
- Attempt B: add CUTLASS-style intra-stage `copy(next_kblock) / gemm(curr_kblock)`
  to the consumer path and then move the empty-slot release earlier, before the
  final gemm on the current stage
  - result: correctness still passed, but the direct grouped bench regressed
    and the aggressive early-release variant regressed further
  - representative direct grouped bench reads on this machine:
    - accepted post-hybrid baseline body: `hot_mean_ms ~= 0.549`
    - Attempt B with k-block pipelining: `hot_mean_ms ~= 0.605`
    - Attempt B plus early empty-slot release: `hot_mean_ms ~= 0.570`
  - conclusion: these first manual CUTLASS-inspired consumer rewrites did not
    outperform the accepted baseline body and were reverted
- Attempt C: port the grouped `P5` body to an explicit `cutlass::PipelineTmaAsync<2>`
  schedule with producer `acquire/get_barrier/tail`, consumer
  `wait/release`, and CUTLASS-style intra-stage `copy_kblock/gemm_kblock`
  sequencing
  - result: correctness still passed, but after rebuilding the statically
    linked grouped bench target the direct `P5` surface regressed sharply
  - rebuilt isolated grouped `P5` bench read on this machine:
    - accepted post-hybrid baseline body: `hot_mean_ms ~= 0.549`
    - Attempt C exact `PipelineTmaAsync` port: `hot_mean_ms = 1.098`
  - conclusion: the exact pipeline object port is not a drop-in win for the
    current grouped `P5` body; it was reverted
- Attempt D: keep the existing grouped `P5` body, but add a lighter-weight
  two-slot barrier ring on top of the direct-global `SFB` base
  - producer-side invariant hoist:
    - move `get_tma_tensor/local_tile/partition_S/partition_D` setup for
      `A/B/SFA` out of the per-tile issue lambda and prebuild stage-0/stage-1
      source/destination partitions once
  - consumer-side overlap:
    - prime `k_block=0`
    - then run `copy(next_kblock)` before `gemm(curr_kblock)` inside each
      macro-`K` tile
    - keep `SFB` direct-global and sync-free
  - result: correctness passed and the isolated grouped `P5` bench showed the
    first clear Phase `1b` win on this branch
  - direct grouped bench progression on this machine:
    - cleaned direct-global `SFB` baseline: `hot_mean_ms=0.578`,
      `hot_median_ms=0.580`
    - first barrier-ring landing before hoists: `hot_mean_ms=0.591`
    - after producer invariant hoist: `hot_mean_ms=0.560`,
      `hot_median_ms=0.359`
    - after inner `copy(next_kblock) / gemm(curr_kblock)` overlap:
      `hot_mean_ms=0.436`, `hot_median_ms=0.342`, `hot_tflops=18.167`
    - immediate reruns on the same rebuilt binary are noisy but stay well below
      the cleaned baseline:
      - `hot_mean_ms=0.449`, `hot_median_ms=0.363`
      - `hot_mean_ms=0.411`, `hot_median_ms=0.342`
    - one later read regressed to `hot_mean_ms=0.601`, so grouped-bench wall
      clock should be treated as machine-state-sensitive; the scheduler metrics
      below are the stronger signal for this attempt
  - targeted isolated-kernel `ncu` read after Attempt D:
    - profile artifact set:
      - `artifacts/profiles/p5_pipeline_overlap_20260410T103111Z/ncu_p5_grouped_prefix128.csv`
    - averaged key metrics across captured launches:
      - `launch__registers_per_thread = 166`
      - `launch__shared_mem_per_block_allocated = 83072 B`
      - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed = 54.75%`
      - `lts__t_sector_hit_rate.pct = 57.32%`
      - `sm__warps_active.avg.pct_of_peak_sustained_active = 18.77%`
      - `smsp__warps_eligible.avg.per_cycle_active = 0.44`
      - `smsp__issue_active.avg.pct_of_peak_sustained_active = 36.04%`
      - `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed = 11.55%`
  - conclusion:
    - the win is not a DRAM-throughput story; it is a scheduler/issue story
    - compared with the post-hybrid baseline (`eligible=0.193`,
      `issue_active=14.47%`), the new grouped `P5` body is issuing far more
      work despite lower occupancy from the `2`-stage footprint
    - the direct-global `SFB` base plus intra-stage copy/MMA overlap is the
      first Phase `1b` structure worth keeping and extending

- Attempt E: move the next-stage handoff into the last `k_block` of the current
  macro tile, CUTLASS-style, instead of waiting for the next stage only at the
  top of the next outer tile
  - variant E1:
    - added a cross-consumer named barrier before `consumer_release`, then
      waited the next full stage and preloaded `k_block 0` of the next tile
      before finishing `gemm(k_block=1)` on the current tile
  - variant E2:
    - removed the named barrier but kept the earlier `consumer_release /
      consumer_wait(next) / copy(next, k_block 0)` handoff
  - correctness:
    - `fused_moe_prefill_test`: PASS
    - `multi_turn_prefix_reuse_test`: PASS
    - `p5_swizzled_pipeline_test_120f`: PASS
  - grouped `P5` bench reads on this machine:
    - E1 first read: `hot_mean_ms=0.532`, `hot_median_ms=0.363`
    - E2 first read: `hot_mean_ms=0.504`, `hot_median_ms=0.337`
  - targeted isolated-kernel `ncu` reads:
    - E1 artifact set:
      - `artifacts/profiles/p5_stage_transition_20260410T104620Z/ncu_p5_grouped_prefix128.csv`
    - E2 artifact set:
      - `artifacts/profiles/p5_stage_transition_noblockbar_20260410T104801Z/ncu_p5_grouped_prefix128.csv`
    - averaged key metrics:
      - E1: `eligible=0.190`, `issue_active=15.89%`,
        `dram=67.03%`, `tensor_pipe=14.16%`
      - E2: `eligible=0.190`, `issue_active=16.07%`,
        `dram=67.59%`, `tensor_pipe=14.24%`
  - conclusion:
    - both variants are worse than Attempt D on the stronger scheduler metrics
      (`eligible=0.44`, `issue_active=36.04%`)
    - the wall-clock grouped bench can look superficially better on a noisy
      machine state, but the isolated `ncu` read says this handoff shape
      over-serializes the consumers on our current producer design
    - do not keep the next-stage `consumer_wait(next)` inside the last
      `k_block` of the current tile on the live `P5` path without a stronger
      producer-side redesign

Phase 1b measurement hygiene note:
- `benchmarks/nano_moe_prefill/nano_routed_up_p5_grouped_bench` is statically
  linked. Rebuilding only `fused_moe_prefill_test` is not sufficient after
  runtime kernel edits; the grouped bench target itself must be rebuilt before
  trusting wall-clock or `ncu` results.

Phase 1b next-step constraint:
- do not reintroduce Attempts A, B, or C as-is
- Attempt D is now the active Phase `1b` base
- next work should keep the direct-global `SFB` and inner `k_block` overlap,
  but it should not pull the next-stage full-barrier wait into the last
  `k_block` of the current tile the way Attempt E did
- next work should instead target producer-side issue/setup overhead from the
  stronger Attempt D base instead of revisiting transport legality

Phase 1b end-to-end integration checkpoint (`fc9dde5`, clean worktree):
- isolated grouped-kernel result:
  - `benchmarks/nano_moe_prefill/nano_routed_up_p5_grouped_bench`
  - `prefix_tokens=128`
  - `selected_token_tile=8`
  - `sfb_live_mode=direct_gmem_sparse`
  - `hot_mean_ms=0.595`
  - `hot_median_ms=0.379`
  - `hot_weight_gib_per_s=581.778`
- safe full-model TTFT result:
  - `SingleTokenForwardConfig.moe_prefill_window_tokens` still defaults to
    `23` in `runtime/src/api/single_token_forward_model.cpp`
  - at `dispatch_rows=23`, `SelectRoutedGemm1Profile` and
    `SelectRoutedGemm2Profile` both return `legacy`
  - the stable runtime therefore still routes direct multi-token MoE prefill
    through row replay in `runtime/src/backend/expert_layer.cpp`
  - clean `fc9dde5` TTFT results on the safe path:
    - `cold_prefill_prefix128`: `1010.200 ms`
    - `cached_committed_head_prefix128_tail4`: `50.196 ms`
    - `cached_global_root_prefix128_tail4`: `50.353 ms`
    - `cached_committed_head_prefix256_tail128`: `1020.110 ms`
    - `cached_global_root_prefix256_tail128`: `1019.302 ms`
    - all of those runs report `expert native multi-token runs=0` and high
      `expert row replay runs`
- unsafe profiling-only result:
  - enabling `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1` allows the
    full model to use native multi-token routed MoE again
  - clean `fc9dde5` TTFT results on that surface:
    - `cold_prefill_prefix128`: `375.189 ms`
    - `cached_committed_head_prefix128_tail4`: `59.397 ms`
    - `cached_global_root_prefix128_tail4`: `56.753 ms`
    - `cached_committed_head_prefix256_tail128`: `377.474 ms`
    - `cached_global_root_prefix256_tail128`: `378.508 ms`
    - those runs report `expert native multi-token runs>0` and
      `expert row replay runs=0`
- correctness gate:
  - safe `continuation_prefill_oracle_test`: PASS
  - unsafe native direct MoE: FAIL
    - `split_single_tail_argmax_match=0`
    - `full_argmax=1710`
    - `single_argmax=1584`
- conclusion:
  - the committed `P5` Phase `1b` kernel win is real, but it cannot move the
    stable end-to-end path until native multi-token direct MoE continuation
    correctness is fixed
  - the next project is therefore not more `P5` microkernel tuning; it is to
    make native multi-token direct MoE pass the continuation oracle, then
    raise or remove the `23`-token safety clamp so the routed kernels can
    become active on the stable full-model surface

#### 1c. Warp-Specialized Producer/Consumer

Dedicate a subset of warps to TMA production and the rest to MMA
consumption.  This is the standard TRT-LLM pattern:

```
Warps 0-5:  MMA consumers (math)
Warp 6:     TMA A/B loader (operand producer)
Warp 7:     TMA SFA/SFB loader (scale producer)
```

The producers issue TMA loads and signal barriers.  The consumers wait on
barriers, run MMA, and signal completion.  No warp ever does both.

This pairs with Phase 1b (multi-stage pipeline) — the producer fills
stage `i+1` while consumers drain stage `i`.

**Expected impact:** Eliminates warp scheduling contention between load and
compute instructions and makes the transport pipeline TRT-like.

### Phase 2: L2 Cache Optimization (measurement-gated)

**Goal:** Maximize reuse of expert weights through L2 cache, reducing
effective DRAM traffic.

#### 2a. Expert-Clustered CTA Scheduling

Replace the default grid launch (which distributes CTAs round-robin
across SMs) with a persistent kernel that processes all CTAs for one
expert before moving to the next.

- Current: 40,000+ CTAs launched, distributed across 170 SMs by the
  hardware scheduler.  Expert A's weight tiles and Expert B's tiles
  compete for L2 space.
- Target: 170 persistent CTAs (one per SM).  Each SM works through a
  global work queue, processing all output tiles for expert A, then
  all tiles for expert B, etc.

When an SM finishes expert A's tiles, expert A's weight data is hot in
L2 from the TMA loads.  The next SM that starts expert A's tiles gets
L2 hits instead of DRAM loads.

With 128 experts and 170 SMs: at any moment, ~170 SMs are working on a
small number of experts.  If 10 SMs work on the same expert
simultaneously, each TMA load from DRAM is reused 10x through L2.

This should only proceed if Phase 0.5 / Phase 1 measurements show that routed
weight traffic still misses L2 heavily enough for expert clustering to matter.
The Phase 0.5 expert histogram is the gating input here: if a small set of
experts dominates routed tokens at `prefix128`, clustering and popularity
sorting become much more attractive; if the distribution is flat, Phase 2
should be deprioritized.

Concrete reference patterns:
- `SM120BlockScaledMoeScheduler` in `sm120_fp8_moe_gemm_1d1d.cuh` lines
  `190-253` shows the current per-expert token-offset/block assignment model
  that a clustered persistent scheduler would replace or wrap.
- `cutlass::gemm::StaticPersistentScheduler` is referenced in
  `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h`
  and provides the generic persistent-scheduler pattern worth borrowing where
  it matches our launch model.

**Expected impact:** potentially material for popular experts, but must be
validated against measured L2 hit rate and expert popularity histograms before
the work is justified.

#### 2b. Popularity-Sorted Expert Scheduling

Sort experts by token count (descending) before scheduling.  Popular
experts (more tokens, more output tiles) run first while L2 is cold and
has maximum capacity.  Unpopular experts (1-2 tokens) run last — they
each load one weight set and produce minimal output, so L2 thrashing
from them is irrelevant.

This pairs with Phase 2a — the persistent kernel's work queue is sorted
by expert popularity.

**Expected impact:** secondary improvement on top of expert clustering if the
measured L2 behavior justifies it.

### Phase 3: Routed Dataflow Restructuring

**Goal:** optimize the routed FC1→activation→FC2 system as one dataflow, not
as isolated kernel islands.

#### 3a. Minimize the FC1→FC2 seam

Evaluate the whole routed layer boundary in byte terms:

- which FC1 outputs must materialize to global memory?
- which buffers can remain request-scoped only once?
- which activation packs can be produced once and consumed directly by FC2?

This phase should remove redundant intermediate traffic before adding more
profile coverage.

#### 3a.5. Experimental expert-local FC1→activation→FC2 fusion

After Phase 1 transport/pipeline work is in place, evaluate a more aggressive
expert-local fusion path for the low-token regime:

- one persistent CTA lifetime per expert tile family
- FC1 compute
- activation / requantization
- FC2 consume
- only the final routed output hits global memory

At the canonical `prefix128` surface, the average expert sees about `6` tokens,
so a BF16 FC1 intermediate of `6 × 1920` is only about `23 KiB`.  That is
small enough to consider keeping the expert-local intermediate in shared memory
while FC2 consumes it directly.

This is not the default plan because routed weight bytes still dominate the
roofline, but it is the most promising Phase 3 variant if the seam bytes and
launch boundaries remain visible after Phase 1-2.

#### 3b. Split Fused FC1 into Gemm1 + Activation

The current FC1 kernel fuses gemm + Relu² + FP4 requantization.  This:
- Forces the compiler to keep all three stages' registers live
  simultaneously (high register pressure).
- Prevents TMA/pipeline optimizations because the fused store path
  can't overlap with the next K-tile load.
- Differs from TRT-LLM, which runs activation as a separate kernel.

Split into:
1. Gemm1 kernel: unified FP4 body with `swap_ab=false` support.
   Produces BF16 intermediate output.
2. Activation kernel: Relu² + FP4 requantization.  Lightweight, reads
   BF16, writes packed FP4.

#### 3c. Port Remaining FC1 Profiles to the Unified Kernel

`P5` is already on the unified routed FP4 body.  The remaining FC1 work is:

- `P0`
- `P1`
- `P4`
- `P7`

Add the remaining FC1 `swap_ab=false` and FC1-side `swap_ab=true` profile
instantiations to the unified kernel traits layer.  The kernel body should stay
the same; only the per-profile builder/layout parameters change.

The TMA/pipeline/scheduling optimizations from Phase 1-2 should then apply to
all routed FC1 and FC2 profiles through one unified body.

### Phase 4: Shared Expert Optimization

**Goal:** Bring the shared expert (~20% of MoE time) onto the same
optimized path.

The shared expert is a single expert that processes ALL tokens (not
routed).  At prefix128: 128 tokens × hidden(2688) → intermediate(3712)
→ hidden(2688).

This is a regular dense GEMM, not a grouped MoE GEMM.  It has much
better arithmetic intensity (128 tokens × 3712 × 2688 × 2 / (3712 ×
2688 × 0.5) = 512 FLOP/byte).  This is close to the compute-bandwidth
balance point.

Options:
- Reuse the unified FP4 kernel body (treat as 1 expert, many tokens).
- Or use cuBLAS for the dense GEMM (may be faster for well-shaped
  dense problems).

Profile first.  Only optimize if shared expert is still >10% of TTFT
after routed expert optimization.

### Phase 5: Mamba Prefill

**Goal:** Optimize Mamba SSM prefill for long sequences.

At prefix128, Mamba is ~8% of TTFT (~11 ms).  At prefix4096, it becomes
a larger fraction because the scan cost grows with sequence length while
MoE cost grows sublinearly (more tokens per expert = better weight
amortization).

The current implementation uses `MambaSsdPrefillFixedKernel` with a
fixed tile size.  TRT-LLM and vLLM use chunked-scan approaches that
parallelize the scan across sequence chunks.

This is lower priority than MoE optimization for prefix128 but becomes
critical for prefix4096.

### Phase 6: System-Level

**Goal:** Optimize everything outside the per-layer compute.

#### 6a. Model Load Time

Current model loading unpacks weights from the checkpoint format and
repacks them into runtime layout.  The 1920-padding (Phase 0) adds a
small cost here.  Opportunities:
- Memory-map the weight file directly if the runtime format matches
  the on-disk format.
- Pre-pack weights into the runtime format in a one-time offline step.
- Parallelize weight loading across CPU cores.

#### 6b. Prefix Cache

The model has Mamba recurrent state and Attention KV-cache that can be
cached for prefix reuse.  Opportunities:
- Efficient state serialization/deserialization.
- Hierarchical cache (L2 → VRAM → host memory → disk).
- Speculative prefix matching.

#### 6c. Batch Scheduling

For serving scenarios with multiple concurrent requests:
- Dynamic batching to increase tokens-per-expert ratio (better weight
  amortization).
- Expert-aware request scheduling (group requests that activate
  similar experts).

## Measurement Framework

Every phase must be validated against concrete roofline metrics, not
just wall-clock TTFT:

1. **DRAM bandwidth utilization:** `nsys` or `ncu` reported memory
   throughput as a percentage of 1792 GB/s.  Target: >60%.

2. **L2 hit rate:** `ncu` L2 cache metrics.  Target: >30% for MoE
   weight accesses.

3. **SM occupancy:** Warps active per cycle.  Not a primary target
   (bandwidth-bound kernels don't need high occupancy) but monitor for
   regressions.

4. **Tensor Core utilization:** Percentage of cycles with active MMA.
   At prefix128 this will always be low (<10%) because we're
   bandwidth-bound.  But it should not be zero (which would indicate
   a broken pipeline).

5. **Kernel launch overhead:** Total CPU-side launch time across all
   kernels per forward pass.  Target: <1 ms.

6. **Per-layer breakdown:** `nsys` trace with per-kernel timing for
   one full forward pass.  Identifies which layers/components are
   above their roofline share.

7. **Intermediate byte accounting:** total bytes written to routed
   intermediate workspace and reread later.  This is the key metric for
   judging whether FC1→activation→FC2 restructuring is actually reducing
   traffic.

8. **Launch count across the routed layer:** number of kernel launches per
   routed layer forward.  If we are still moving the same bytes through too
   many boundaries, launch count will reveal it.

9. **Traffic decomposition:** break total routed-layer bytes into:
   - weight bytes
   - scale bytes
   - FC1→FC2 seam / intermediate bytes
   This is the only way to tell whether a speedup came from better transport,
   better overlap, or simply shifting traffic between buffers.

## Execution Order

The work should proceed in this order:

1. Freeze the canonical `window=4096` benchmark surface.
2. Run the Phase 0.5 measurement/probe gate and quantify current transport
   bottlenecks.
3. Replace thread-driven staging with direct-to-swizzled staged transport.
4. Add multi-stage overlap and warp-specialized producer/consumer execution.
5. Measure again.
6. Only then optimize expert scheduling / clustering if the measured L2
   behavior justifies it.
7. Then finish the routed FC1 dataflow and remaining FC1 profile coverage on
   top of the new transport path.
8. Optimize shared expert and Mamba only after the routed path has been pushed
   close enough to the roofline that they become meaningful fractions of TTFT.

## Phase Dependencies

```
Phase 0 (foundation)
  ↓
Phase 0.5 (measurement / probe gate)
  ↓
Phase 1a (TMA loads)
  ↓
Phase 1b (multi-stage pipeline) ← requires 1a
  ↓
Phase 1c (warp specialization) ← requires 1b
  ↓
Phase 2a (expert clustering) ← requires 1a, can overlap with 1b/1c
  ↓
Phase 2b (popularity sorting) ← requires 2a
  ↓
Phase 3 (routed FC1 / dataflow) ← requires Phase 1 complete, benefits from Phase 2
  ↓
Phase 4 (shared expert) ← profile after Phase 1-3
  ↓
Phase 5 (Mamba) ← independent, can start after Phase 0
  ↓
Phase 6 (system) ← independent, can start anytime
```

## Current P5 Status

- The branch moved one architectural step further than the old hybrid-TMA
  `SFB` path:
  - `SFB` is now loaded directly from global execution-scale memory into the
    consumer fragment on the live grouped `P5` path
  - there is no `SFB` shared-memory remap or CTA-wide sync on that path
- The standalone proof for that direct-global path is green:
  - `tma_fragment_sfb_global_assembly offset=25 PASS`
  - the fragment now uses sparse physical writes, not a 64-element logical
    splat
- The first direct-global live result was correctness-green but instruction
  heavy:
  - `hot_mean_ms=0.619`
  - `sleeping` dropped sharply versus the post-hybrid baseline, which confirms
    the CTA-wide `SFB` sync/remap was removed from the critical path
- The next direct-global optimization was the real win:
  - writing only the 4 physical scale bytes per atom instead of all 64 logical
    aliases improved the grouped `P5` bench to:
    - `hot_mean_ms=0.575`
    - `hot_median_ms=0.577`
  - that is still slightly slower than the older remap-free chunked path
    (`~0.556 ms`), but it preserves the cleaner direct-global architecture
- A definitive warp-sharing probe now explains the remaining load shape:
  - for each active `SFB` slot, warp 0 sees exactly
    `warp0_unique=0,1,2,3,4,5,6,7`
  - lane pattern is contiguous 4-lane groups:
    `0,0,0,0,1,1,1,1,...,7,7,7,7`
  - implication:
    - only an 8-word-per-warp reduction is available
    - the scale loads are shareable, but not “one word per warp” shareable
- I tested the obvious subgroup optimization on that pattern:
  - one loader lane per contiguous 4-lane subgroup plus `__shfl_sync`
  - correctness remained green
  - live grouped `P5` performance regressed slightly:
    - `hot_mean_ms=0.593`
    - `hot_median_ms=0.588`
  - targeted `ncu` comparison says the saved global loads are not paying for
    the added shuffle/control overhead on this kernel
- Cleaned-base status as of the current `P5` checkpoint:
  - the subgroup broadcast optimization has been removed from the live kernel
  - the grouped `P5` bench now reports the actual active scale path:
    - `sfb_live_mode=direct_gmem_sparse`
    - `sfb_tma_cached=yes`
    - `sfb_tma_launch_compatible=yes`
    - `sfb_global_fragment_ctas=128`
  - the canonical direct-global proof is now a default regression in
    `p5_swizzled_pipeline_test_120f`, not an opt-in env probe:
    - `tma_fragment_sfb_global_assembly offset=25 PASS`
  - current validated grouped `P5` baseline on this cleaned base:
    - `cold_ms=0.600`
    - `hot_mean_ms=0.578`
    - `hot_median_ms=0.580`
    - `hot_p90_ms=0.589`
    - `hot_tflops=13.715`
- Current conclusion:
  - direct-global `SFB` is still the right architectural base
  - sparse 4-byte fragment fill is worth keeping
  - 4-lane subgroup broadcast is not worth keeping and is no longer live code
  - Phase `1b` is now live on the isolated grouped `P5` surface:
    - `cold_ms=0.395`
    - `hot_mean_ms=0.436`
    - `hot_median_ms=0.342`
    - `hot_p90_ms=0.344`
    - `hot_tflops=18.167`
  - the next profitable work should stay on this pipeline branch:
    clean up remaining stage-transition overhead and only then consider porting
    the same structure to the other routed profiles

## What We're NOT Doing (and why)

- **Atom-loop serpentine optimization:** Saves <0.3 ms across all layers
  (compute is 30x faster than bandwidth).  Not worth the complexity
  until we're within 2x of roofline.

- **Vectorized epilogue first:** output writes are not the first roofline
  limiter.  Defer until transport, overlap, and routing seams are fixed.

- **Host-side `cudaMemPrefetchAsync` expert prefetch:** our routed expert
  weights are ordinary device allocations, not managed-memory pages, so this is
  not a reliable way to stage the next expert into L2.  If we need explicit
  cache warming later, it should come from kernel-side transport/scheduling or
  a proven L2 access-policy mechanism, not per-expert host-side prefetch calls.

- **Weight compression below FP4:** Would require model retraining or
  quality-loss-tolerant deployment.  Out of scope for the runtime.

- **Cross-layer weight sharing:** Not possible — each layer has
  independent expert weights.

- **cuBLAS/cuDNN replacement for Attention:** Only 8 layers, already
  well-optimized by NVIDIA libraries.  Not on the critical path.

## Milestones

| Milestone | TTFT target | Key change |
|---|---|---|
| Phase 0 complete | baseline established | padded/unified production baseline |
| Phase 0.5 complete | measured bottleneck locked | `ncu` + source/probe gate |
| Phase 1a (TMA) | ≤140 ms prefix128 | hardware-accelerated staged loads |
| Phase 1b+c (pipeline) | ≤70 ms prefix128 | latency hiding + warp specialization |
| Phase 2 (L2 opt) | ≤50 ms prefix128 | measured cache-aware scheduling |
| Phase 3 (FC1) | ≤35 ms prefix128 | full routed path optimized |
| Phase 4+5 | stretch: ≤20 ms prefix128 | shared expert + Mamba |

These are estimates based on the corrected roofline analysis, including routed
block-scale traffic.  Each phase will be re-evaluated against measured
`nsys`/`ncu` data before proceeding to the next.
