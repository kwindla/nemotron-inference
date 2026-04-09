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
  - `B`/activation now also has a live TMA transport path, but its ownership
    is deliberately launch-local:
    - host copies the exact `cta_row_starts` / `cta_valid_rows` metadata
    - host builds one bare CUTE `P5TmaLoadB` descriptor per active CTA
    - descriptors are copied to device for the launch and consumed by a
      dedicated producer warp in the unified `P5` kernel
  - scales and non-`P5` profiles still use the existing thread-coded
    global -> swizzled-smem scatter
  - this is a transport-correctness landing, not the final roofline-ready
    ownership model: per-launch host descriptor rebuild is expected to carry
    overhead and should be treated as temporary
- current live-gate status:
  - `fused_moe_prefill_test PASS`
  - `multi_turn_prefix_reuse_test PASS`
  - `testing/p5_swizzled_pipeline_test_120f` with
    `NEMOTRON_RUN_P5_TMA_SMOKE=1 PASS`
  - single-case canonical-capacity TTFT smoke:
    - `cold_prefill_prefix128 = 142.097 ms` median
    - `cached_committed_head_prefix128_tail4 hot-prefix = 66.129 ms` median
    - `cached_global_root_prefix128_tail4 hot-prefix = 65.854 ms` median
    with `resolved_runtime_moe_prefill_window_tokens=4096`
  - still not a claimed full-matrix TTFT win: the launch-local `B`
    descriptor path is correct enough to benchmark, but the ownership/caching
    model still needs to move closer to the runtime plan / workspace layer

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
  the live `P5` path now consumes per-CTA launch-local `B` descriptors.
  The next boundary is no longer "can live B TMA work?" but "how do we move
  descriptor ownership/caching out of the per-launch host rebuild so the
  transport experiment is representative?"
- keep two temporary correctness-landing caveats explicit before reading too
  much into the first transport measurements:
  - the live `P5` B path still zero-fills the entire swizzled `B` stage before
    issuing TMA, even though TMA already zero-fills out-of-bounds rows; remove
    that conservative clear before serious roofline profiling
  - the old thread-coded `B` packed-data loop still exists structurally around
    the `if (!use_p5_tma_b)` fallback; confirm with SASS / `ncu` that the
    empty loop body is fully elided on the TMA path instead of paying a hidden
    control-flow cost
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
5. Replace the temporary launch-local `P5` B-descriptor rebuild with a
   runtime-owned descriptor path:
   - either cache per-CTA `P5TmaLoadB` descriptors alongside the launch plan
     metadata for a stable workspace pointer
   - or derive an equivalent launch-time object without host round-tripping
     the CTA metadata
6. When profiling the live `P5` TMA path, remove and then remeasure the two
   known temporary costs:
   - the conservative full-stage `B` smem clear before TMA
   - any residual control-flow overhead from the fallback `B` packed-data loop

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
