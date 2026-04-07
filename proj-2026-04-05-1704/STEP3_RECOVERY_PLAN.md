# Step 3 Recovery Plan

Project directory: `./proj-2026-04-05-1704`

## Goal

Recover from the current custom-MoE-prefill regression, validate that the new
device-side contract is still the right architectural direction, and then push
past the best pre-refactor TTFT numbers.

## Current Priority: Prefill Optimization

The contract work is now good enough to treat as fixed for this phase:

- one active runtime code path
- device-only runtime contract
- no hot-path DtoH
- behavioral reuse equivalence restored on the committed-head reuse path
- TRT-style grouped padded FC1/FC2 activation-pack production now exists in the
  static MoE workspace via custom per-expert NVFP4 packing code

Latest checkpoint:

- FC1 now consumes a TRT-like packed source contract: normalized activations
  are packed once and permuted in packed form into grouped padded rows
- the active packed path no longer computes the dead per-expert activation
  scales that its current consumer ignores
- focused gates are green again:
  `device_nvfp4_matrix_test`, `moe_launch_plan_device_test`,
  `fused_moe_prefill_test`, `multi_turn_prefix_reuse_test`
- focused TTFT on `prefix128 / tail4` is now:
  `cold_prefill_prefix128 = 125.663 ms`,
  `cached_committed_head_prefix128_tail4 hot-prefix = 54.480 ms`,
  `cached_global_root_prefix128_tail4 hot-prefix = 54.246 ms`

Interpretation:

- the FC1 contract shift is now stable and slightly positive
- the next missing TRT piece is FC2 / Gemm1-output scale handling, not more
  FC1 gather-side work

Current in-progress follow-up:

- grouped per-expert packing now again honors expert tensor scales
- routed-down packed input now consumes that FC2 expert-scale contract
- focused gates remain green:
  `device_nvfp4_matrix_test`, `moe_launch_plan_device_test`,
  `fused_moe_prefill_test`, `multi_turn_prefix_reuse_test`

Focused TTFT on the same `prefix128 / tail4` gate after that FC2 step is:

- `cold_prefill_prefix128 = 126.679 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.423 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.766 ms`

The next TRT-style contract jump is now landed:

- the active packed path computes Gemm1/activation output scales directly
- activated FC1 output is packed straight into the grouped FC2 input contract
- `routed_up_scratch` is no longer the active FC1->FC2 seam
- packed focused tests now run with `routed_up_scratch = nullptr`

Focused TTFT on the same gate after removing that FP32 boundary is:

- `cold_prefill_prefix128 = 126.959 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.447 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.684 ms`

Interpretation:

- the direct Gemm1-output contract is correct and behaviorally stable
- this step is structurally necessary, but still roughly performance-neutral
- the remaining gap is now squarely in the grouped math core, not the packed
  FC1->FC2 boundary

The next routed-stage alignment step then landed:

- launch-plan aliases now expose the TRT-style grouped metadata names:
  `permuted_idx_to_token_idx`, `total_num_padded_tokens`,
  `num_non_exiting_ctas`, `cta_idx_xy_to_batch_idx`,
  `cta_idx_xy_to_mn_limit`
- the active routed BF16 path no longer pre-packs FC1 input activations into an
  NVFP4 matrix
- the active path now does `BF16 grouped Gemm1 -> gemm1_output_scale -> packed
  Gemm2 input`, which matches TRT's BF16 x NVFP4 routed entry far better than
  the previous packed-FC1 experiment

Focused TTFT on the same `prefix128 / tail4` gate after that cutover is:

- `cold_prefill_prefix128 = 126.012 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.447 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.616 ms`

Interpretation:

- this confirms that the packed FC1 input path should not remain active for the
  BF16 Nemotron Nano route
- the remaining gap is now even more clearly the grouped FC1/FC2 math core
  itself, not the routed launch metadata or BF16 FC1 entry contract

The next routed-stage cutover is now also landed:

- the active grouped FC1 kernel stays on the transposed WMMA body, but now
  materializes BF16 `gemm1_output` directly instead of re-running FC1 to derive
  the FC2 input contract
- activation scales are computed from BF16 `Relu2(gemm1_output)` rows, and
  those activated BF16 rows are packed directly into the grouped FC2 input
  contract
- the active grouped path no longer depends on `routed_up_scratch`; packed
  focused tests again run with `routed_up_scratch = nullptr`

Focused TTFT on the same `prefix128 / tail4` gate after this BF16 seam cutover
is:

- `cold_prefill_prefix128 = 126.390 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.336 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.333 ms`

Interpretation:

- this removes duplicate FC1 work and keeps the routed boundary much closer to
  TRT's `gemm1_output` / scale contract
- correctness and behavioral reuse remain green
- the gain is still modest, which confirms the next real win must come from
  the grouped FC1/FC2 math core itself

The next routed-stage contract cutover is now also landed:

- `gemm1_output_scale` / `activation_output_scale` are now real per-row /
  per-block routed dequant scales on the active path, written directly by the
  BF16->NVFP4 packer
- the active FC2 consumer now decodes from those routed dequant scales
  directly, instead of reconstructing scale from `expert tensor scale + encoded
  block scales`
- the active routed FC1 and FC2 consumers now derive token work from
  `cta_idx_xy_to_batch_idx + cta_idx_xy_to_mn_limit + expert_first_token_offsets +
  selected_token_tile`, rather than the older explicit
  `cta_row_starts + cta_valid_rows` contract

Focused TTFT on the same `prefix128 / tail4` gate after these cutovers is:

- `cold_prefill_prefix128 = 126.178 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.361 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.652 ms`

The next routed-stage cutover is now in place too:

- normalized activations are packed once into the static prefill source pack
- routed FC1 gathers packed rows with `permuted_idx_to_token_idx` into the
  grouped FC1 input pack
- the active FC1 consumer now runs from packed routed input directly into BF16
  `gemm1_output`, instead of going through the FP32 gather path first

Focused TTFT on the same gate after that cutover is:

- `cold_prefill_prefix128 = 125.623 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.361 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.224 ms`

The next grouped-body step is now active too:

- routed FC1/FC2 WMMA consumers use a TRT-like `tileTokensDim=16`,
  `transposeMmaOutput=true`, `epilogueTileM=128` execution shape
- in our implementation that is `kPlannedOutputTile = 128`,
  `kPlannedThreadsPerBlock = 256`, and `8` warps per CTA for the routed
  grouped path

Focused TTFT on the same gate after the larger grouped tile cutover is:

- `cold_prefill_prefix128 = 125.324 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.358 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.597 ms`

Interpretation:

- the active routed path is now carrying and consuming a much more faithful
  TRT-style grouped contract
- the TTFT result is still close to flat even after moving FC1 onto packed
  routed input
- the larger grouped tile does help cold prefill a bit, which is evidence that
  we are finally moving the active bottleneck inside the kernel body rather than
  just rearranging contracts
- that means the remaining gap is now squarely in the grouped FC1/FC2 kernel
  body itself, not the routed metadata or FC1->FC2 scale handoff
- TRT runtime trace update (`2026-04-07`):
  - the real local `TRT-LLM` NemotronH serve path is not using the
    `trtllmGenFp8BlockScaleMoe` runner we first inspected
  - it is using the CUTLASS fused-MoE custom op selected via
    `trtllm::fused_moe::gemm1` / `trtllm::fused_moe::gemm2`
  - observed tactic IDs from the real serve path are:
    - warmup max-context `(4607, 1344)`: `gemm1=1`, `gemm2=13`
    - decode-like `(1, 1344)`: `gemm1=0`, `gemm2=13`
    - `prefix4` `(4, 1344)`: `gemm1=5`, `gemm2=15`
    - `prefix128` `(128, 1344)`: `gemm1=4`, `gemm2=12`
    - `prefix4096` `(4096, 1344)`: `gemm1=1`, `gemm2=13`
  - the packed routed shape on that path is:
    - `input_shape=(tokens, 1344)`
    - `fc1_shape=(128, 1920, 168)`
    - `fc2_shape=(128, 2688, 120)`
    - `top_k=6`
    - `act_dtype=torch.uint8`, `weight_dtype=torch.int64`,
      `output_dtype=torch.bfloat16`
  - implication:
    the native grouped-kernel target must now be treated as a packed
    low-precision, shape-aware tactic family rather than one fixed routed
    grouped kernel
  - actual CUTLASS tactic descriptors recovered from the live
    `FusedMoeRunner` path:
    - `prefix4`
      - `gemm1=5`: TMA Warp Specialized, `tile=128x128x64`,
        `cluster=1x1x1`, `swap_ab=true`, `epilogue_fusion=0`
      - `gemm2=15`: TMA Warp Specialized, `tile=256x128x64`,
        `cluster=1x1x1`, `swap_ab=true`, `epilogue_fusion=1`
    - `prefix128`
      - `gemm1=4`: TMA Warp Specialized, `tile=128x128x128`,
        `cluster=1x1x1`, `swap_ab=true`, `epilogue_fusion=0`
      - `gemm2=13`: TMA Warp Specialized, `tile=128x128x64`,
        `cluster=1x1x1`, `swap_ab=true`, `epilogue_fusion=1`
    - `prefix4096`
      - `gemm1=1`: TMA Warp Specialized, `tile=128x128x64`,
        `cluster=1x1x1`, `swap_ab=false`, `epilogue_fusion=0`
      - `gemm2=13`: TMA Warp Specialized, `tile=128x128x64`,
        `cluster=1x1x1`, `swap_ab=true`, `epilogue_fusion=1`
  - implication:
    - `gemm2` is nearly stable across `128` and `4096`, but `prefix4` wants a
      larger `256x128x64` tile
    - `gemm1` changes both `tile_k` and `swap_ab` across regimes
    - the next native grouped-kernel rewrite should therefore target a small
      TRT-like tactic family rather than one fixed routed body
- granular tactic-sweep update (`2026-04-07`):
  - artifacts:
    - `artifacts/benchmarks/trtllm_fused_moe_tactic_sweep_20260407.log`
    - `artifacts/benchmarks/trtllm_fused_moe_tactic_boundary_sweep_20260407.log`
  - the live CUTLASS fused-MoE path is not using a simple monotonic threshold
    over `num_rows`
  - observed islands from the finer sweep:
    - `1`: `g1=0 128x128x128 swap_ab=false`, `g2=13 128x128x64`
    - `2..3`: `g1=7 256x128x64 swap_ab=true`, `g2=15 256x128x64`
    - `4..7`: `g1=5 128x128x64 swap_ab=true`, `g2=15 256x128x64`
    - `8`: `g1=1 128x128x64 swap_ab=false`, `g2=13 128x128x64`
    - `9..15`: `g1=1 128x128x64 swap_ab=false`, `g2=12 128x128x128`
    - `16`: `g1=1 128x128x64 swap_ab=false`, `g2=15 256x128x64`
    - `24..31`: `g1=1 128x128x64 swap_ab=false`, `g2=13 128x128x64`
    - `32..112`: `g1=0 128x128x128 swap_ab=false`, `g2=13 128x128x64`
    - `120..127`: `g1=4 128x128x128 swap_ab=true`, `g2=13 128x128x64`
    - `128..192`: `g1=5 128x128x64 swap_ab=true`, `g2=12 128x128x128`
    - `200..248`: `g1=4 128x128x128 swap_ab=true`, `g2=12 128x128x128`
    - `256`: `g1=5 128x128x64 swap_ab=true`, `g2=13 128x128x64`
    - `320`: `g1=1 128x128x64 swap_ab=false`, `g2=12 128x128x128`
    - `384`: `g1=5 128x128x64 swap_ab=true`, `g2=13 128x128x64`
    - `448..511`: `g1=1 128x128x64 swap_ab=false`, `g2=12 128x128x128`
    - `512..992`: `g1=5 128x128x64 swap_ab=true`, `g2=12 128x128x128`
    - `1024..4607`: `g1=1 128x128x64 swap_ab=false`, `g2=13 128x128x64`
  - implication:
    - the native routed grouped-kernel selector should be an explicit regime
      table rather than a guessed threshold function
    - `gemm1` and `gemm2` need independent regime selection
  - the concrete implementation sub-plan for that rewrite now lives in the
    canonical optimization doc:
    `proj-2026-04-05-0445/PLAN.md`,
    section `Sub-Plan: From-Scratch Routed Grouped GEMM Family`
- grouped-body operand-layout cutover (`2026-04-07`):
  - active file:
    `runtime/src/backend/fused_moe_prefill.cu`
  - change:
    the routed FC1/FC2 grouped kernels now follow TRT operand-layout semantics
    instead of treating both operands as row-major
    - `swap_ab=false`: activations as `A` row-major, weights as `B`
      column-major
    - `swap_ab=true`: weights as `A` row-major, activations as `B`
      column-major, with transposed output scatter
  - focused validation:
    - `fused_moe_prefill_test`
    - `moe_launch_plan_device_test`
    - `multi_turn_prefix_reuse_test`
  - artifacts:
    - `artifacts/benchmarks/ttft_20260407_trt_layout_grouped_body_prefix128_tail4.stdout.txt`
    - `artifacts/benchmarks/ttft_20260407_trt_layout_grouped_body_prefix4096_tail4.stdout.txt`
  - result:
    - `cold_prefill_prefix128 = 126.168 ms`
    - `cached_global_root_prefix128_tail4 hot-prefix = 54.304 ms`
    - `cold_prefill_prefix4096 = 2488.473 ms`
    - `cached_global_root_prefix4096_tail4 hot-prefix = 96.850 ms`
  - conclusion:
    - this closes the remaining semantic/layout mismatch with TRT at the routed
      grouped-kernel boundary
    - it does not yet recover the big performance gap
    - the next bottleneck is the grouped mainloop itself: scalar NVFP4 decode,
      shared-memory staging, and the lack of TRT-like TMA/block-scaled
      transport into the MMA pipeline
    - fallback for untraced tactic islands remains the legacy WMMA row-tile
      routed path in `runtime/src/backend/fused_moe_prefill.cu`
- grouped-body blockwise decode cutover (`2026-04-07`):
  - active file:
    `runtime/src/backend/fused_moe_prefill.cu`
  - change:
    the active traced grouped routed kernels now stage complete `16`-value
    NVFP4 blocks into shared memory, instead of repeatedly decoding individual
    FP4 elements and reloading the same block scale for each element
  - focused validation:
    - `fused_moe_prefill_test`
    - `moe_launch_plan_device_test`
    - `multi_turn_prefix_reuse_test`
  - artifacts:
    - `artifacts/benchmarks/ttft_20260407_blockwise_grouped_body_prefix128_tail4.stdout.txt`
    - `artifacts/benchmarks/ttft_20260407_blockwise_grouped_body_prefix4096_tail4.stdout.txt`
  - result:
    - `cold_prefill_prefix128 = 125.578 ms`
    - `cached_global_root_prefix128_tail4 hot-prefix = 54.328 ms`
    - `cold_prefill_prefix4096 = 2479.858 ms`
    - `cached_global_root_prefix4096_tail4 hot-prefix = 96.994 ms`
  - conclusion:
    - blockwise decode is worth retaining: it improves cold prefill at both
      `prefix128` and `prefix4096` while keeping reuse correctness green
    - hot-prefix TTFT is effectively flat, which means the remaining routed
      gap is now deeper than scalar FP4 decode alone
    - the next native rewrite should target the grouped MMA mainloop proper,
      not more contract cleanup

- full TRT mainloop alignment plan (`2026-04-07`):
  - canonical plan:
    `proj-2026-04-05-0445/PLAN.md`,
    section `Full TRT Mainloop Alignment Plan`
  - objective:
    match the traced local TRT routed mainloop as faithfully as possible in
    native custom code, not just the routed metadata and tile-selection layer
  - immediate work:
    - freeze the current routed external contract
    - recover the remaining TRT internal mainloop facts with targeted tracing
      and profiling
    - build a native producer/consumer grouped-kernel substrate
    - port the traced routed tactic family onto that substrate
  - current blockers:
    - missing internal pipeline facts such as stage count and
      producer/consumer warp-role split
    - no native TMA/warp-specialized substrate yet
    - partial epilogue recovery only
    - some internal tile/swizzle details still inferred rather than observed
    - irregular traced tactic family requires a real profile family, not one
      universal kernel

- targeted TRT routed `nsys` profile (`2026-04-07`):
  - artifacts:
    - `artifacts/profiles/trtllm_mainloop_20260407/nsys_prefix128.nsys-rep`
    - `artifacts/profiles/trtllm_mainloop_20260407/nsys_prefix128.cuda_gpu_kern_sum.csv`
    - `artifacts/profiles/trtllm_mainloop_20260407/nsys_prefix128.cuda_gpu_trace.csv`
  - recovered facts:
    - the live routed GEMMs are
      `cutlass::gemm::kernel::GemmUniversal<...MainloopSm120ArrayTmaWarpSpecializedBlockScaled...>`
    - live grouped GEMM launches use `BlockX=384`
    - TRT helper stages are separate kernels:
      - `computeStridesTmaWarpSpecializedKernel<...>` with `BlockX=128`
      - `doActivationKernel<...>` with `BlockX=256`
    - the live routed SM120 path is block-scaled FP4 grouped GEMM
      (`cutlass::float_e2m1_t` operands), not BF16 WMMA
  - implication:
    - native helper-stage alignment is now materially closer
    - the remaining routed gap is specifically the true FP4/TMA grouped
      mainloop body

- routed activation-pack helper alignment (`2026-04-07`):
  - active native change:
    - replace the generic BF16 `PackDeviceRowMajorBf16ToNvfp4PerExpert(...)`
      boundary with a routed, padded, programmatic-launch activation pack
      kernel in `runtime/src/backend/fused_moe_prefill.cu`
    - rows-per-CTA policy matches TRT helper behavior:
      `kProcessRows = 1 / 2 / 4`
  - focused validation:
    - `fused_moe_prefill_test`
    - `moe_launch_plan_device_test`
    - `multi_turn_prefix_reuse_test`
  - artifact:
    - `artifacts/benchmarks/ttft_20260407_routed_activation_pack_prefix128_tail4.stdout.txt`
  - result:
    - `cold_prefill_prefix128 = 126.004 ms`
    - `cached_committed_head_prefix128_tail4 hot-prefix = 54.410 ms`
    - `cached_global_root_prefix128_tail4 hot-prefix = 54.373 ms`
  - conclusion:
    - keep the routed activation-pack helper alignment
    - helper-stage structure is no longer the main routed gap

- grouped routed CTA-shape alignment (`2026-04-07`):
  - active native change:
    - move the active grouped routed FC1/FC2 kernels to a TRT-like `384`
      thread CTA
    - keep eight consumer warps on the WMMA tile and use the extra four warps
      on the staging side
  - focused validation:
    - `fused_moe_prefill_test`
    - `moe_launch_plan_device_test`
    - `multi_turn_prefix_reuse_test`
  - artifact:
    - `artifacts/benchmarks/ttft_20260407_grouped_384cta_prefix128_tail4.stdout.txt`
  - result:
    - `cold_prefill_prefix128 = 125.839 ms`
    - `cached_committed_head_prefix128_tail4 hot-prefix = 54.766 ms`
    - `cached_global_root_prefix128_tail4 hot-prefix = 54.540 ms`
  - conclusion:
    - keep the larger grouped CTA shape
    - this improves cold prefill slightly and keeps reuse correctness green
    - hot-prefix remains flat, so the next bottleneck is the true FP4/TMA
      mainloop rather than CTA size alone

That means the priority is no longer "prove the contract." The priority is:

1. optimize prefill first
2. recover the cold TTFT baseline
3. beat the baseline with a stronger math core

Everything else is secondary until cold prefill is back under control.

External benchmark commands, artifacts, and cross-runtime profiling notes now
live in `proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md`.

Canonical optimization planning now lives in
`proj-2026-04-05-0445/PLAN.md`. This file should be treated as the detailed
recovery log and artifact-backed execution record for that plan.

This recovery plan should also be read as the continuation of the earlier
optimization contract in `proj-2026-04-05-0445/PLAN.md`, especially:

- `Prior Art` and `Locked Contract`
- step `2`, which locked `FP4xFP4` tensor-core math with one activation
  quantization step per layer
- step `3`, which set the intended end state as grouped `FP4xFP4` MoE math,
  not scalar row-wise dot products

## External Race Plan

The next objective is no longer just "recover the old baseline." It is:

1. beat local `vLLM` cold prefill
2. beat local `TRT-LLM` PyTorch cold prefill
3. do it without adding a second runtime path or weakening reuse correctness

### Competition Target

Use the direct exact-token prefill profiling surface from
`EXTERNAL_BASELINES_NOTES.md` as the primary cold-prefill race metric.

The implementation direction for beating those numbers is still anchored by the
earlier optimization decisions in `proj-2026-04-05-0445/PLAN.md`: grouped
`FP4xFP4` tensor-core MoE, explicit launch/runtime structure, and no return to
host-orchestrated per-expert execution.

Current measured times on the local Nano checkpoint / RTX 5090:

| runtime | prefix4 | prefix128 | prefix4096 |
|---|---:|---:|---:|
| native runtime | `54.016 ms` | `127.698 ms` | `2503.116 ms` |
| vLLM | `48.011 ms` | `40.396 ms` | `82.938 ms` |
| TRT-LLM | `35.922 ms` | `38.407 ms` | `66.789 ms` |

To beat both external baselines, native runtime must get under:

- `prefix4 < 35.922 ms`
- `prefix128 < 38.407 ms`
- `prefix4096 < 66.789 ms`

Current gap versus the stronger external comparator:

- `prefix4`: native is about `1.50x` slower than TRT-LLM
- `prefix128`: native is about `3.32x` slower than TRT-LLM
- `prefix4096`: native is about `37.48x` slower than TRT-LLM

### What The Current Profiles Say

The external profile comparison makes the work order clearer than the older
internal-only analysis:

- `prefix4`: native still spends most of prefill in routed expert custom math
  plus shared expert WMMA
- `prefix128`: native is dominated by routed expert custom math
- `prefix4096`: native is dominated by routed expert custom math and then by
  Mamba prefill

By contrast, both `vLLM` and `TRT-LLM` spend the same region mostly in grouped
CUTLASS / FP4 GEMM kernels, with much smaller glue overhead and much smaller
Mamba cost at long prefix.

That means:

- the routed expert math core is still the first blocker
- the shared expert path is still worth fixing early, especially for `prefix4`
- the long-prefix `4096` race cannot be won without a serious Mamba prefill
  improvement after routed MoE
- attention is not the first optimization target

### Execution Order

#### Phase 1. Routed MoE Parity

Goal:

- make routed MoE look structurally like the external baselines:
  grouped/tiled FP4 GEMM, not a custom row kernel

This phase is the direct continuation of
`proj-2026-04-05-0445/PLAN.md` step `3` (`Grouped FP4xFP4 tensor core MoE
kernel`). The earlier project already settled the architectural argument. The
remaining work is to actually land that end state in the current runtime and
make it beat the external baselines on RTX 5090.

Required work:

- replace the remaining custom routed-expert math core with a TRT-LLM-like
  grouped GEMM implementation over the current device launch plan
- keep one runtime path and the existing device-only contract
- remove or fold standalone row-oriented staging where possible:
  - explicit row expansion
  - explicit scatter-style row handling
  - explicit quantize-dequantize passes that only exist to feed the current
    custom row kernels

Stage gate:

- routed-expert kernels are no longer the dominant majority bucket at
  `prefix128`
- native `prefix128 < 70 ms`
- native `prefix4 < 45 ms`

Why this gate:

- if routed MoE still dominates after the first rewrite, we are still not close
  enough to the external execution shape
- `prefix128` is the design-center case where routed MoE should pay off first

#### Phase 2. Shared Expert Alignment

Goal:

- move shared experts onto a kernel family that is competitive with the grouped
  external paths instead of leaving short-prefix prefill exposed to a large
  standalone shared-expert cost

Required work:

- keep the current deterministic tile selection idea if it still helps
- but treat shared-up/shared-down as first-class math-core work, not as a late
  cleanup item
- reuse the same specialized SM120 execution principles as the routed path

Stage gate:

- shared-expert kernels are no longer one of the top two buckets at `prefix4`
- native `prefix4 < 40 ms`
- native `prefix128 < 50 ms`

Why this gate:

- `prefix4` is where shared expert still matters most after routed work
- winning `prefix4` and `prefix128` requires both routed and shared paths to be
  competitive

#### Phase 3. Long-Prefix Mamba Recovery

Goal:

- collapse the long-prefix Mamba gap that currently blocks `prefix4096`

Required work:

- compare our `MambaSsdPrefillFixedKernel` path directly against the chunk-scan
  style work visible in `vLLM` / `TRT-LLM`
- determine whether the right move is:
  - chunked scan / chunked state progression
  - larger-tile scan structure
  - better state layout / memory traffic
  - a more fused prefill step boundary

Stage gate:

- native `prefix4096 < 250 ms`
- Mamba prefill is below `10%` of the `prefix4096` GPU-kernel mix

Why this gate:

- we do not need final victory at this phase
- but if `4096` is still in the hundreds of milliseconds, we are nowhere near
  the external frontier

#### Phase 4. Final Race Tuning

Goal:

- close the remaining gap to the best external direct-prefill numbers

Expected remaining work:

- attention polish only if it has become one of the top remaining buckets
- launch-shape cleanup
- short-prefix under-fill handling
- small residual routing/finalize overhead

Final gate:

- native exact-token cold prefill beats both external baselines at `4`, `128`,
  and `4096`
- no regression on behaviorally exact reuse
- no material regression in hot-prefix TTFT or single-request decode throughput

### Rules For This Race

1. Optimize against the external direct-prefill numbers first, not only against
   our historical baselines.
2. Treat `prefix128` as the design-center optimization case.
3. Do not claim progress from short-prefix wins if `prefix4096` remains blocked
   by Mamba.
4. Do not spend time on attention while routed MoE or Mamba is still the
   dominant gap.
5. After each landed phase, rerun:
   - native direct/profile comparison
   - the `4 / 128 / 4096` native TTFT surface
   - `multi_turn_prefix_reuse_test`

### Practical Interpretation

The path to beating `vLLM` and `TRT-LLM` is:

1. make routed MoE look like grouped GEMM
2. make shared MoE stop distorting short-prefix cold prefill
3. make long-prefix Mamba look more like the external chunk-scan cost profile
4. only then spend time on smaller polish items

Anything that does not move one of those three fronts is probably not on the
critical path.

## Specialization Assumptions

This plan is intentionally specialized for one concrete deployment target:

- model: official Nemotron 3 Nano NVFP4 checkpoint
- GPU: GeForce RTX 5090 (`SM120`, Blackwell)
- routed expert shape:
  - hidden size `2688`
  - routed intermediate size `1856`
  - shared intermediate size `3712`
  - routed experts `128`
  - routed experts selected per token `6`

That means:

- the first objective is not a generic MoE kernel family
- we should tune tile sizes, staging strategy, launch shape, and scratch layout
  for these exact dimensions
- we should optimize for RTX 5090 execution characteristics directly
- we should prefer a single specialized fast path over a more abstract slower
  implementation, as long as we do not create a second runtime path
- one-time load-time specialization or repacking is acceptable if it improves
  the active path; per-request repacking is not

Generality matters only where it is required to preserve correctness across the
benchmarked prompt lengths, cache states, and serving scenarios for this same
model and GPU target.

## Prefill Optimization Plan

This is the active plan for the next phase of work.

### Success Criteria

Primary success criteria:

- beat the old cold-prefill baseline on the plan-aligned benchmark surface
- keep hot-prefix TTFT at or below the old baseline
- preserve one runtime path and the current device-only contract
- preserve behaviorally exact reuse on the Nano oracle path

Behavioral reuse requirement for this phase:

- cold full prefill and split / resumed prefill must remain behaviorally
  equivalent on the Nano oracle path
- the acceptance bar is stable reuse behavior, not bitwise-identical tensors
- small numerical drift is acceptable only if it does not change committed-head
  reuse behavior, cache correctness, or greedy token decisions

Required correctness gates:

- `fused_moe_prefill_test`
- `expert_routing_device_test`
- `multi_turn_prefix_reuse_test`
- focused interactive-forward sanity checks

Required benchmark gates before claiming success:

- plan-aligned TTFT sweep from `PLAN.md`
- throughput bench
- full `ctest`

### Test Strategy

To stay focused on specialization without losing basic kernel coverage:

- deployment-shape MoE correctness tests should use the exact Nemotron Nano
  NVFP4 dimensions by default
- synthetic contract tests should remain only where they catch edge conditions
  that Nano-shaped tests would miss, such as ragged tails, tiny dimensions, and
  optional-output contract validation

For the current MoE prefill work, that means:

- `fused_moe_prefill_test` should contain both:
  - a minimal synthetic contract case
  - a Nano deployment-shape correctness case
- `expert_routing_device_test` remains synthetic and contract-focused

### Guiding Principles

1. Do not add a second implementation path.
2. Do not add host-side control or DtoH in the hot path.
3. Specialize aggressively for the deployed Nemotron Nano NVFP4 on RTX 5090
   shape when that improves the active runtime path.
4. Use TRT-LLM as the required structural reference for operation boundaries,
   work decomposition, and staging.
5. Do not spend more time polishing scalar dot-product kernels once the data
   says the architecture is wrong.
6. Replace one operation boundary at a time and benchmark after each change.
7. Keep the runtime contract stable while changing the math core underneath it.

### Required TRT-LLM Comparison Process

Before implementing or changing any major prefill operation, compare our code
against the closest TRT-LLM operation and record the comparison in this project
file.

Required comparison targets:

- routing / renormalize
- token permutation / expert-major layout
- grouped first projection
- activation / requantization
- grouped second projection
- finalize / unpermute

Questions to answer for each operation:

- what work decomposition does TRT-LLM use
- where does TRT-LLM create expert-major or tile-major parallelism
- where does TRT-LLM preserve locality between stages
- what parts of TRT-LLM are useful structure for us to copy directly
- what parts should be specialized more aggressively for `SM120` and the exact
  Nano dimensions

## Assumption Audit

This section records the key assumptions behind the current plan so they stay
explicit and reviewable.

### 1. Shape regime matters, and it changes with prefix length

For Nano:

- `prefix4`: `24` routed selections total, average `0.1875` tokens per expert
- `prefix128`: `768` routed selections total, average `6` tokens per expert
- `prefix4096`: `24,576` routed selections total, average `192` tokens per
  expert

These are materially different execution regimes:

- `prefix4` is effectively a sparse tiny-batch regime
- `prefix128` is an awkward small-batch grouped regime
- `prefix4096` is a real grouped-GEMM regime

Plan consequence:

- we still want one runtime code path
- we do not want three separate math stacks
- therefore the chosen math core has to tolerate under-filled small-`M` cases
  while still scaling well at `prefix128` and `prefix4096`

Design-center assumption:

- optimize primarily for `prefix128` and `prefix4096`
- keep `prefix4` correct and accept that it is the least hardware-friendly
  point on the benchmark surface
- if small-`M` handling is needed, prefer padding or tile under-fill inside the
  same grouped kernel family rather than dispatching to a second algorithmic
  path

### 2. Launch count was a structural problem, but the current routed baseline already fixed most of it

The earlier grouped implementation really did suffer from huge routed launch
fan-out. That was a valid concern.

But the retained single-launch grouped baseline has already collapsed the
routed path from per-expert launches to one routed-up and one routed-down
launch per expert layer:

- routed grouped expert matvec launches in current `nsys`:
  `230` over `5` measured iterations = `46` per iteration = `23` layers x `2`
  routed stages

The shared path still contributes another `46` dense expert matvec launches per
iteration, so the total expert-math launch count is still not minimal, but the
main routed launch explosion is no longer the dominant issue.

Plan consequence:

- further launch-count reduction is welcome
- but the current cold-prefill gap is now mostly a math-core efficiency
  problem, not a `5,888`-launch structural problem

### 3. Tensor-pipe use must become a hard success criterion

The current `ncu` profile shows effectively `0%` tensor-pipe activity in the
dominant routed expert math kernel.

Plan consequence:

- any replacement routed-up or routed-down design that still shows effectively
  zero tensor-pipe activity should be treated as suspect
- "clear microbench win" is not enough by itself; the win should come from a
  better execution mechanism, not another scalar dequant-plus-dot-product loop

Operational gate:

- first routed-up replacement must move tensor-pipe activity above zero
- later iterations should aim to raise it materially, not merely make it
  nonzero

### 4. cuBLASLt is not the active implementation target for this path

This is an explicit architectural assumption, not an oversight.

We are not currently planning to use cuBLASLt as the shipping prefill math core
for this MoE path.

Reasons:

- we intentionally moved this path away from a library-controlled expert
  execution model to keep the runtime contract fully under our control
- we want one device-driven runtime path, not a path that depends on opaque
  library algorithm selection and host-orchestrated grouped GEMM setup
- the awkward `M` distribution for `prefix4` and `prefix128` would still force
  us to make padding and scheduling decisions; cuBLASLt does not remove that
  problem, it only moves the math into a library call
- TRT-LLM is the architectural reference, but the goal is to reproduce its
  strong operation shape with a much narrower custom implementation specialized
  to this model and GPU

Plan consequence:

- TRT-LLM remains the reference for operation structure
- cuBLASLt can still be used as a benchmark or sanity yardstick if needed
- but it is not the assumed end-state implementation for the active prefill
  path

### 5. Shared experts are not a late afterthought

The current single-launch profile still shows the shared expert dense path at
about `23%` of GPU kernel time on cold `prefix128`.

Plan consequence:

- routed expert math is still the first priority because it is larger
- but shared expert execution is large enough that we should evaluate it early
  once the routed-up replacement shape is clear
- shared should not remain parked at the end of the plan as a vague cleanup
  task

## Routed-Up Operation Comparison

This is the first required operation-level comparison between the current code
and TRT-LLM.

### Current runtime path

Current routed-up in `runtime/src/backend/fused_moe_prefill.cu`:

- input contract is already reduced to grouped routed rows:
  `selection_count x hidden_size`
- launch shape is one block per `(expert, output_row)`
- grid size is `n_experts * routed_intermediate_size`
- each block decodes FP4 weights on the fly and computes scalar dot products
  over the full hidden dimension
- each block handles up to `kGroupedTokenTile = 8` routed rows at a time
- output layout is row-major `selection_count x routed_intermediate_size`

Implications:

- work decomposition is output-row-major, not expert-tile-major
- `M` only changes how many inner-loop row tiles each block processes
- tensor-pipe usage is effectively zero
- the kernel remains scalar and control-heavy even when `M` becomes large

### TRT-LLM reference path

Relevant files:

- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu`
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cuteDslKernels/moeUtils.h`

TRT-LLM routed-up structure:

- routing builds permutation and launch metadata:
  - `permutedIdxToTokenIdx`
  - `numNonExitingCtas`
  - `ctaIdxXyToBatchIdx`
  - `ctaIdxXyToMnLimit`
- `PermuteGemm1::Runner::run()` executes a grouped batched GEMM over the
  routed tokens and experts
- work is organized around expert-major or tile-major grouped GEMM launch
  metadata rather than one block per output row
- the grouped GEMM runner is configured with `tileTokensDim`, `routeAct`, and
  transpose/epilogue behavior so the routed-up stage stays in a GEMM-friendly
  layout for the next stage

### Structural mismatch

The key mismatch is not only "we use a different kernel." It is:

- TRT-LLM decomposes routed-up as grouped GEMM over expert-major tiles
- our current kernel decomposes routed-up as scalar row-wise matvec over
  `n_experts * output_rows`

That means TRT-LLM creates parallelism along the tile axes that matter for
tensor-core math, while our current kernel creates a huge number of scalar
blocks that each do their own FP4 decode and reduction work.

This comparison reinforces the current architectural target:

- keep the runtime contract
- replace the routed-up math core
- move the work decomposition toward expert-major grouped tiles

## Routed-Up Microbench Baseline

The first Phase-1 harness is now implemented at:

- `benchmarks/nano_moe_prefill/nano_routed_up_bench.cpp`

It benchmarks the current active routed-up math core directly through
`RunGroupedNvfp4ExpertMatVec()` using the exact Nano routed-up dimensions:

- hidden size `2688`
- routed intermediate size `1856`
- routed experts `128`
- top-k `6`

Current measured results with `--warmup 1 --iterations 5`:

- `prefix4`
  - selection count `24`
  - active experts `24`
  - hot mean `0.520 ms`
  - effective routed-up throughput `0.461 TFLOP/s`
- `prefix128`
  - selection count `768`
  - active experts `128`
  - hot mean `10.282 ms`
  - effective routed-up throughput `0.745 TFLOP/s`
- `prefix4096`
  - selection count `24,576`
  - active experts `128`
  - hot mean `320.900 ms`
  - effective routed-up throughput `0.764 TFLOP/s`

Immediate interpretation:

- the kernel does not materially improve its effective compute rate from
  `prefix128` to `prefix4096`
- that is strong evidence that the ceiling is the scalar math core itself, not
  only small-`M` inefficiency
- the routed-up kernel alone already reproduces the same ~`0.75 TFLOP/s`
  ceiling implied by the end-to-end profile

This is the current baseline to beat for the routed-up replacement.

## TRT-LLM-Aligned Routed-Up Microbench Update

Artifacts:

- checked benchmark run:
  - `artifacts/benchmarks/nano_routed_up_bench_20260406_row_coop_checked.stdout.txt`
- checked `ncu` report:
  - `artifacts/profiles/nano_routed_up_20260406_row_coop/ncu_routed_up_prefix128_row_coop_checked.csv`

Implementation note:

- the routed-up microbench now includes a correctness cross-check against the
  current baseline kernel
- the candidate path builds padded expert-major token tiles, runs a
  row-cooperative CUDA kernel, then unpads the result back to logical token
  order before comparing outputs
- this keeps the experiment benchmark-only while using a zero-diff numerical
  diagnostic against the current baseline before any runtime integration

Candidate shape:

- one CTA owns one `(expert, token_tile, output_row_tile)`
- token tile size `8`
- output row tile size `4`
- one warp computes one output row
- the CTA stages the token tile in shared memory and reuses it across the
  output-row tile
- this is structurally closer to TRT-LLM `permute -> gemm1` than the old
  one-block-per-row baseline, but it still uses scalar NVFP4 decode and scalar
  accumulation

Checked results:

- `prefix4`
  - baseline hot mean `0.520 ms`
  - row-coop hot mean `2.380 ms`
  - exact match: `max_abs_diff_vs_baseline = 0.000`
- `prefix128`
  - baseline hot mean `10.251 ms`
  - row-coop hot mean `12.659 ms`
  - exact match: `max_abs_diff_vs_baseline = 0.000`
- `prefix4096`
  - baseline hot mean `319.538 ms`
  - row-coop hot mean `302.364 ms`
  - exact match: `max_abs_diff_vs_baseline = 0.000`

Interpretation:

- the candidate is not good enough for the primary target case
  `prefix128`; it is about `23.5%` slower there
- it is much worse for `prefix4` because padding from sparse per-expert token
  counts dominates the small amount of work
- it does show a small `~5.4%` improvement at `prefix4096`, which means the
  shared-input reuse is directionally helpful once `M` is large enough
- that is useful evidence, but not a runtime integration signal

Checked `ncu` comparison on `prefix128`:

- block size `128`
- grid size `59392`
- registers per thread `48`
- static shared memory per block `2048` bytes
- achieved occupancy `82.64%`
- eligible warps per scheduler `0.27`
- issue active `18.33%`

Comparison to the old standalone baseline:

- warp eligibility improved from about `0.15` to `0.27`
- issue rate improved from about `11.8%` to `18.3%`
- but that scheduler improvement still did not overcome the extra padded work
  and per-step synchronization on the `prefix128` case

Conclusion:

- this experiment rules out a tempting but insufficient next step:
  “expert-major permutation plus shared-input reuse” by itself is not enough
  for the behaviorally reuse-stable path we need
- the next routed-up prototype should preserve the correctness guard, but it
  needs to attack the remaining problem more directly:
  - reduce or avoid padded token work for the `prefix128` regime
  - amortize synchronization better than one shared-load barrier per `K` step
  - move farther toward GEMM-like work decomposition rather than only
    reorganizing scalar matvec

### Follow-Up: Ragged Row-Coop Routed-Up Candidate

Artifacts:

- checked benchmark run:
  - `artifacts/benchmarks/nano_routed_up_bench_20260406_ragged_row_coop.stdout.txt`
- checked `ncu` report:
  - `artifacts/profiles/nano_routed_up_20260406_row_coop/ncu_routed_up_prefix128_ragged_row_coop.csv`

Change:

- keep the same row-cooperative kernel shape
- remove padded expert-major token storage entirely
- build CTA metadata directly from the true `expert_offsets`
- load only the valid token rows for each expert tile
- keep the same zero-diff numerical cross-check against the routed-up baseline

This isolates the effect of padded-token overhead from the effect of the
shared-input row-cooperative math itself.

Checked results:

- `prefix4`
  - baseline hot mean `0.520 ms`
  - ragged row-coop hot mean `0.411 ms`
  - speedup `1.26x`
  - exact match: `max_abs_diff_vs_baseline = 0.000`
- `prefix128`
  - baseline hot mean `10.240 ms`
  - ragged row-coop hot mean `9.560 ms`
  - speedup `1.07x`
  - exact match: `max_abs_diff_vs_baseline = 0.000`
- `prefix4096`
  - baseline hot mean `319.513 ms`
  - ragged row-coop hot mean `297.429 ms`
  - speedup `1.07x`
  - exact match: `max_abs_diff_vs_baseline = 0.000`

Interpretation:

- removing the padded-token work completely flipped the `prefix128` result
  from a regression into a real win
- that confirms the previous section’s diagnosis: the padded-token overhead was
  large enough to erase the scheduler gains on the priority case
- the row-cooperative shape is still only a modest improvement, but it is now
  a correct benchmark-only candidate worth considering for runtime integration

Checked `ncu` comparison on `prefix128`:

- block size `128`
- grid size `59392`
- registers per thread `47`
- static shared memory per block `2048` bytes
- achieved occupancy `82.67%`
- eligible warps per scheduler `0.24`
- issue active `16.13%`

Comparison to the old standalone baseline:

- warp eligibility improved from about `0.15` to `0.24`
- issue rate improved from about `11.8%` to `16.1%`
- the gain is smaller than the padded version, but without the padded-token
  tax it is enough to win on `prefix4`, `prefix128`, and `prefix4096`

Updated conclusion:

- the next practical step is to port this routed-up shape into the runtime
  path behind the existing device-only contract and measure TTFT impact
- because the gain is only about `7%` on the key routed-up microbench case,
  this is likely not the whole recovery
- but it is the first benchmark-only routed-up design that beats the current
  baseline across the Nano benchmark surface while still matching it exactly in
  this narrow numerical experiment

### Rejected Runtime Integration: Upper-Bound Device Mapping

Artifact:

- `artifacts/benchmarks/ttft_20260406_runtime_ragged_row_coop_prefix128_tail4.stdout.txt`

Attempt:

- replace the runtime grouped routed matvec launch with the ragged row-coop
  kernel
- keep the runtime device-only contract by avoiding host-built CTA metadata
- map `(expert, row_tile)` work inside the kernel from an upper-bound
  `linear_row_tile` index derived from `selection_count`

Why it was attractive:

- no DtoH
- no second runtime code path
- no new routing-side host metadata

Measured result on the fast TTFT gate:

- `cold_prefill_prefix128`
  - retained runtime baseline: `681.460 ms`
  - upper-bound runtime mapping: `971.287 ms`
  - regression: `1.43x` slower
- `cached_committed_head_prefix128_tail4`
  - retained runtime baseline hot-prefix TTFT: `52.054 ms`
  - upper-bound runtime mapping hot-prefix TTFT: `66.660 ms`
  - regression: `1.28x` slower

Interpretation:

- the benchmark-only routed-up gain did not survive this runtime mapping
- the device-only upper-bound launch created too much extra scheduling and
  empty-work overhead
- the routed-up kernel itself was not the problem; the way runtime work was
  enumerated was

Action taken:

- reverted the runtime grouped-matvec integration after the TTFT regression
- kept the benchmark-only ragged row-coop candidate and its correctness guard
- kept the benchmark and profiling artifacts because the routed-up kernel shape
  is still promising

Updated next step:

- do not retry runtime integration with an upper-bound linear row-tile scan
- if we integrate this routed-up shape into runtime, we need a lighter
  device-only work description:
  - device-built CTA metadata
  - or another exact mapping that does not explode the number of launched CTAs

## Runtime Launch-Plan Architecture

This is the concrete next runtime step.

Keep these parts unchanged:

- `DeviceExpertRouting` remains the logical routing result
- routed gather stays expert-major through `sorted_token_indices`
- finalize continues to use `selection_to_sorted`
- no hot-path DtoH is introduced

Add one new device-resident object:

- `DeviceMoeLaunchPlan`

Current first-version contents:

- `row_tile_count`
- `row_tile_expert_ids`
- `row_tile_row_starts`
- `row_tile_valid_rows`

Why this boundary:

- the retained benchmark-only ragged row-coop kernel already proved that exact
  `(expert, row_tile)` descriptors are enough to beat the scalar routed-up
  baseline
- the rejected runtime attempt failed because CTAs rediscovered their work from
  an upper-bound linear row-tile index inside the kernel
- TRT-LLM uses the same architectural split: routing emits compact launch
  metadata, and grouped math consumes it

Current implementation strategy:

1. run `RunDeviceExpertRouting()` exactly as today
2. build `DeviceMoeLaunchPlan` on device from:
   - `expert_offsets`
   - `active_expert_count`
   - `active_expert_ids`
3. launch routed-up and routed-down against that plan instead of scanning
   `expert_offsets` inside the math kernel
4. keep shared expert and finalize unchanged for this step

Careful-review notes:

- this is not throwaway work if we later move to a richer TRT-style CTA map
- the reusable part is the separation between logical routing and device launch
  metadata
- the row-tile plan is only the first plan format, not the final ceiling
- the runtime consumer may still use a bounded overlaunch on the host, but the
  kernel must index exact descriptors directly and must not scan to find work
- if this integration still leaves too much empty-CTA overhead, the next
  extension should be richer CTA metadata or a persistent worker schedule, not
  a return to kernel-side discovery

Implementation status:

- the device launch-plan object and builder are now worth keeping
- a first runtime consumer was tested against the retained `prefix128/tail4`
  TTFT gate and regressed badly
- so the active runtime path should stay on the retained grouped kernel until
  the plan-driven consumer wins both its microbench and the TTFT gate
- failed consumer artifacts:
  - `artifacts/benchmarks/ttft_20260406_launch_plan_prefix128_tail4.stdout.txt`
  - `artifacts/benchmarks/ttft_20260406_launch_plan_routed_up_only_prefix128_tail4.stdout.txt`
- retained grouped-baseline recheck after reverting the active consumer:
  - `artifacts/benchmarks/ttft_20260406_launch_plan_foundation_prefix128_tail4.stdout.txt`

## Routed-Up Standalone `ncu` Baseline

Artifact:

- `artifacts/profiles/nano_routed_up_20260406/ncu_routed_up_prefix128.csv`

Profile target:

- `nano_routed_up_bench --case prefix128`
- kernel:
  `Nvfp4GroupedExpertMatVecRowsKernel(const float*, const int*, unsigned long, const FusedNvfp4WeightView*, unsigned long, float*)`

Key metrics:

- block size `128`
- grid size `14848`
- registers per thread `48`
- static shared memory per block `6144` bytes
- theoretical occupancy `83.33%`
- achieved occupancy `80.51%`
- achieved active warps per SM `38.64`
- eligible warps per scheduler `0.15`
- issued warps per scheduler `0.11`
- `No Eligible = 88.94%`
- memory throughput about `23.24 GB/s`
- DRAM throughput `1.32%` of peak

Interpretation:

- the current routed-up kernel is not primarily occupancy-limited
- it is also still nowhere near a bandwidth limit
- the dominant problem is issue starvation in a scalar dependency-heavy kernel
  with very few eligible warps per cycle despite high achieved occupancy

This matters because it changes the next optimization rule:

- do not chase occupancy for its own sake
- do not assume a larger shared-memory tile automatically helps
- the next routed-up experiment needs a different math mechanism or much higher
  instruction-level parallelism, not just a different launch geometry

One attempted expert-major tile rewrite was tried and then reverted before
landing because it regressed the routed-up microbench and temporarily broke the
synthetic contract test. The result reinforces the point above: changing the
block shape alone is not enough if the arithmetic core stays scalar.

### Benchmark Contract

Fast iteration benchmarks:

- `cold_prefill_prefix128`
- `cached_committed_head_prefix128_tail4`

Official checkpoint benchmarks:

- `cold_prefill_prefix4`
- `cold_prefill_prefix128`
- `cold_prefill_prefix4096`
- `cached_committed_head_prefix4_tail4`
- `cached_committed_head_prefix128_tail4`
- `cached_committed_head_prefix4096_tail4`

Interpretation rules:

- cold prefill is the primary optimization target
- cached hot-prefix TTFT must not regress while recovering cold prefill
- throughput is a validation gate, not the tuning loop metric

### Phase 0. Freeze the Current Baseline

Objective:

- treat the current grouped single-launch path as the baseline to beat

Work:

- keep the current `nsys` and `ncu` artifacts as the control profile
- record the retained benchmark numbers in this project file
- use the current device-only grouped path as the reference implementation for
  correctness during the rewrite

Stop condition:

- do not change benchmark surface or contract assumptions during this phase

### Phase 1. Build Operation-Level Tuning Harnesses

Objective:

- make each prefill operation measurable on its own

Work:

- add a microbench for routed row gather / permute
- add a microbench for routed-up
- add a microbench for activation + requantization
- add a microbench for routed-down
- add a microbench for finalize / unpermute / weighted reduction
- keep all timed regions device-only
- size the first version of every harness for the exact Nano dimensions before
  considering any broader parameterization

Reference requirement:

- for each harness, point to the closest TRT-LLM operation and note the current
  structural mismatch before writing new kernel code

Deliverable:

- one small benchmark target or harness per operation

Stop condition:

- do not start new kernel work until the touched operation has a measurable
  microbench

### Phase 2. Replace Routed-Up with a Real Grouped Math Core

Objective:

- eliminate the scalar routed-up matvec as the dominant cold-prefill hotspot

Target shape:

- expert-major or tile-major work decomposition
- many tokens per launch, not one output row per block
- explicit staging of input tiles and weight tiles
- enough regularity to support tensor-like or tensor-core-friendly math

Reference shape:

- TRT-LLM `route -> permute -> grouped gemm1`

Work:

- compare our current routed-up path against TRT-LLM `PermuteGemm1` first
- design the routed-up scratch layout around grouped tiles rather than scalar
  output rows
- choose tile sizes and staging specifically for `hidden=2688`,
  `routed=1856`, `top_k=6`, and RTX 5090 execution behavior
- replace `Nvfp4GroupedExpertMatVecRowsKernel` for routed-up first
- benchmark the routed-up microbench before wiring it into end-to-end prefill
- run fast TTFT benchmarks after integration

Success gate:

- routed-up microbench shows a clear win
- routed-up profile shows nonzero tensor-pipe activity
- cold `prefix128` improves materially

### Phase 3. Replace Activation + Requantization with a Tiled Pipeline Stage

Objective:

- keep the grouped routed-up output in a layout that feeds routed-down without
  falling back to a scalar cleanup phase

Target shape:

- vectorized activation
- vectorized requantization
- minimal extra global-memory traffic
- no thread-0 scalar passes

Reference shape:

- TRT-LLM activation/quant stage between grouped `gemm1` and grouped `gemm2`

Work:

- compare our current activation stage against TRT-LLM activation/quant staging
  first
- define the intermediate layout needed by both routed-up and routed-down
- fuse where the code stays understandable
- keep the intermediate layout specialized for the routed-up / routed-down tile
  shape we actually deploy
- benchmark this stage independently before full integration

Important note:

- this phase does not assume activation/requant must remain a permanent
  standalone stage
- if the chosen grouped math core makes this boundary simpler or partially
  fusable, take the simpler design

Success gate:

- activation/quant is no longer an architectural blocker between grouped
  routed-up and grouped routed-down

### Phase 4. Replace Routed-Down and Finalize

Objective:

- complete the routed path with a grouped routed-down stage and an explicit
  finalize step

Target shape:

- grouped routed-down math over expert-major tiles
- explicit finalize / unpermute / weighted reduction back to token order
- no scalar in-kernel accumulation as the primary combine strategy

Reference shape:

- TRT-LLM `grouped gemm2 -> finalize`

Work:

- compare our current routed-down and weighted reduction against TRT-LLM
  `Gemm2 -> finalize` first
- replace the current routed-down scalar grouped matvec
- make finalize its own explicit operation boundary
- keep weighted reduction semantics identical to the current contract

Success gate:

- routed path becomes fully grouped end to end
- cold `prefix128` moves substantially closer to baseline

### Phase 5. Rework the Shared Expert Path

Objective:

- decide whether the shared path should reuse the grouped core or stay as a
  smaller specialized path

Work:

- measure the shared path separately after the routed path is upgraded
- compare against TRT-LLM shared-expert separation before changing this path
- if it is still a meaningful cold-prefill cost, replace its scalar matvec
  stages too
- otherwise keep it simple and leave it alone

Decision rule:

- optimize shared only after routed no longer dominates
- if the new grouped math core is already natural for the shared dense path,
  pull shared forward rather than waiting for a separate late phase

### Phase 6. Recover and Beat the Baseline

Objective:

- move from "less bad" to "better than before"

Work:

- rerun the full plan-aligned TTFT sweep
- rerun throughput
- rerun full `ctest`
- rerun interactive conversation sanity checks
- compare directly against the old recorded baseline artifacts

Finish line:

- cold prefill better than the pre-refactor baseline
- cached hot-prefix at least maintained
- no correctness regressions

This phase is not "rewrite everything again". It is:

1. profile the current custom kernel stack
2. compare each kernel operation against stronger reference implementations
3. optimize one operation at a time with micro tests and benchmark checkpoints
4. stop only when we are better than the pre-refactor baseline or we have
   proven that a specific operation boundary must be redesigned

## Current Status

Checkpoint commit:

- `9c93ed2` `Checkpoint custom MoE prefill refactor`

Current known regression from the native TTFT run in
`artifacts/benchmarks/nano_prefix_cache_ttft_20260405T_native.stdout.txt`:

- `cold_prefill_prefix256`: `2234.009 ms`
- `cold_prefill_prefix1024`: `7781.018 ms`
- `cold_prefill_prefix4096`: `28173.273 ms`
- `cached_committed_head_prefix256_tail32` hot-prefix: `1151.702 ms`
- `cached_committed_head_prefix1024_tail32` hot-prefix: `1159.389 ms`

Recent pre-refactor baselines:

- `proj-2026-04-05-0445/step7_runs/20260405T/nano_prefix_cache_ttft_initial.txt`
  - `cold_prefill_prefix256`: `293.446 ms`
  - `cold_prefill_prefix1024`: `439.427 ms`
  - `cold_prefill_prefix4096`: `1385.281 ms`
  - `cached_committed_head_prefix256_tail32` hot-prefix: `162.210 ms`
  - `cached_committed_head_prefix1024_tail32` hot-prefix: `178.329 ms`
- `proj-2026-04-05-0445/step9_runs/20260405T/post_attention_ttft.txt`
  - `cold_prefill_prefix1024`: `440.312 ms`
  - `cold_prefill_prefix4096`: `1369.694 ms`

Current slowdown versus the fresher baselines:

- `cold_prefill_prefix256`: `7.61x`
- `cold_prefill_prefix1024`: `17.67x`
- `cold_prefill_prefix4096`: `20.57x`
- `cached_committed_head_prefix256_tail32` hot-prefix: `7.10x`
- `cached_committed_head_prefix1024_tail32` hot-prefix: `6.50x`

Post-correctness update on the current working tree:

- committed-head reuse is green again after fixing the custom MoE prefill
  semantic mismatch so the prefill path returns `routed + shared`, matching the
  decode path
- the focused checks now pass again:
  - `fused_moe_prefill_test`
  - `multi_turn_prefix_reuse_test`
- the remaining standalone diagnostic
  `testing/api/split_prefill_localization_test.cpp` still reports a
  `full vs prefix-only prefix attention KV` mismatch for the isolated
  2-token prefix snapshot case, but that mismatch does not reproduce in the
  committed-head reuse path

Plan-aligned rerun on the current working tree from
`artifacts/benchmarks/ttft_20260406_tail4_prefix4_128_4096.stdout.txt`:

- cold medians:
  - `cold_prefill_prefix4`: `1133.700 ms` vs `40.415 ms` baseline (`28.05x`)
  - `cold_prefill_prefix128`: `1153.988 ms` vs `142.156 ms` baseline (`8.12x`)
  - `cold_prefill_prefix4096`: `28235.126 ms` vs `1191.797 ms` baseline (`23.69x`)
- committed-head hot-prefix medians:
  - `prefix4_tail4`: `1145.679 ms` vs `35.281 ms` baseline (`32.47x`)
  - `prefix128_tail4`: `1143.402 ms` vs `40.655 ms` baseline (`28.12x`)
  - `prefix4096_tail4`: `1190.753 ms` vs `83.871 ms` baseline (`14.20x`)
- short-tail behavior is especially informative: `prefix4_tail4` and
  `prefix128_tail4` are both pinned near `1.14 s`, which suggests a large
  roughly fixed prefill cost rather than a tail-length-specific problem
- the long-prefix cached case still gets a real cache win (`24.684x` vs cold),
  but the hot-prefix tail prefill itself is still far above baseline at
  `1155.880 ms`

Re-profile update on the clean checkpoint `22b0463`:

- `nsys` traces were captured for:
  - `artifacts/profiles/ttft_prefill_20260406/nsys_cold_prefill_prefix128.nsys-rep`
  - `artifacts/profiles/ttft_prefill_20260406/nsys_cached_committed_head_prefix128_tail4.nsys-rep`
- the cold `prefix128` trace shows the custom MoE prefill kernel still
  dominating the end-to-end wall time:
  - `FusedMoePrefillKernel`: `5602.587 ms` total over `115` launches
    (`48.718 ms` average), which is about `1120 ms` per measured benchmark
    iteration and therefore nearly the whole cold TTFT
  - the next largest GPU bucket is `MambaSsdPrefillFixedKernel<128>` at only
    `45.257 ms` total
- the cached `prefix128_tail4` trace tells the same story:
  - `FusedMoePrefillKernel`: `16790.389 ms` total over `345` launches
    (`48.668 ms` average)
  - the hot-prefix tail prefill itself was `1126.400 ms`, again pointing at the
    expert path rather than restore or decode
- `ncu` on the first `FusedMoePrefillKernel` launch from cold `prefix128`
  (`artifacts/profiles/ttft_prefill_20260406/ncu_cold_prefill_prefix128_fused_moe.csv`)
  shows a serialized kernel rather than a bandwidth-limited kernel:
  - block size `256`, grid size `128`
  - `54` registers per thread
  - `26880` bytes shared memory per block
  - occupancy limited to `3` blocks/SM by shared memory and `4` by registers
  - DRAM throughput only `2.18%` of peak
  - scheduler eligibility only about `0.09` warps per cycle active
- that profile is consistent with the code structure in
  `runtime/src/backend/fused_moe_prefill.cu`, where input quantization and both
  activation-plus-requantization stages are still serialized on `tid == 0`
- immediate next optimization step: keep the current contract and kernel shape,
  but parallelize the thread-0 quantize / `Relu2` / requantize sections before
  attempting a larger grouped-math redesign

First Priority-1 recovery step on top of `22b0463`:

- changed `fused_decode::QuantizeDequantizeNvfp4Row` to a block-parallel row
  implementation instead of a single-thread walk
- changed `FusedMoePrefillKernel` so the initial selected-expert copy and both
  `Relu2` stages are block-parallel instead of `tid == 0`
- correctness stayed green on:
  - `fused_moe_prefill_test`
  - `multi_turn_prefix_reuse_test`
- focused TTFT rerun on `prefix128` / `tail4` improved, but only modestly:
  - `cold_prefill_prefix128`: `1153.988 ms` -> `1114.050 ms` (`1.036x`, `3.46%`)
  - `cached_committed_head_prefix128_tail4`: `1143.402 ms` -> `1099.152 ms`
    (`1.040x`, `3.87%`)
- conclusion: the thread-0 sections were real, but they were not the dominant
  source of the regression. The next step should stop polishing around these
  scalar loops and move to the larger structural gap versus vLLM / TRT-LLM:
  grouped routed-expert math with token-by-expert scheduling and weight reuse

Priority-2 grouped routed-expert rewrite on top of `cdd890d`:

- replaced the one-block-per-token routed path with a grouped routed-expert
  prefill implementation in `runtime/src/backend/fused_moe_prefill.cu`
  built around `DeviceExpertRouting` plus the existing request/workspace
  scratch tensors
- the current grouped path keeps a single active implementation path and now
  does:
  - device routing into sorted token/expert order
  - grouped gather over routed rows
  - grouped routed-up and routed-down custom NVFP4 matvec launches
  - ordered top-k reduction using `selection_to_sorted`
  - separate shared-expert custom path using the same scratch discipline
- correctness stayed green on:
  - `fused_moe_prefill_test`
  - `expert_routing_device_test`
  - `multi_turn_prefix_reuse_test`
- focused TTFT rerun recorded in
  `artifacts/benchmarks/ttft_20260406_grouped_prefix128_tail4.stdout.txt`
  shows a large recovery on the plan-aligned `prefix128` / `tail4` case:
  - `cold_prefill_prefix128`: `1114.050 ms` -> `719.458 ms`
    (`1.548x`, `35.42%` faster)
  - `cached_committed_head_prefix128_tail4`: `1099.152 ms` -> `65.347 ms`
    (`16.82x`, `94.05%` faster)
- relative to the pre-refactor plan baseline:
  - `cold_prefill_prefix128`: still `5.06x` slower than `142.156 ms`
  - `cached_committed_head_prefix128_tail4`: now only `1.61x` slower than
    `40.655 ms`
- interpretation:
  - the grouped rewrite fixed most of the short-tail hot-prefix disaster
  - cold prefill is still far too slow, so the remaining bottleneck is likely
    inside the routed expert math itself rather than cache restore, snapshot,
    or the earlier thread-0 scalar sections
  - because the kernel shape has changed materially, the next step should be a
    fresh `nsys` / `ncu` pass on this grouped version before more surgery

Grouped-path reprofiling update:

- fresh cold `prefix128` `nsys` trace recorded at:
  - `artifacts/profiles/ttft_prefill_20260406_grouped/nsys_cold_prefill_prefix128.nsys-rep`
  - `artifacts/profiles/ttft_prefill_20260406_grouped/nsys_cold_prefill_prefix128.stats_cuda_gpu_kern_sum.csv`
- unlike the old one-block-per-token kernel, the grouped implementation is now
  dominated almost entirely by the custom NVFP4 matvec stages:
  - routed grouped matvec
    `Nvfp4MatVecRowsKernel<(bool)1>`: `2605.270 ms` total over `29440`
    launches, `88.494 us` average, `74.0%` of GPU time
  - shared contiguous matvec
    `Nvfp4MatVecRowsKernel<(bool)0>`: `774.534 ms` total over `230`
    launches, `3.368 ms` average, `22.0%` of GPU time
  - together they account for about `96%` of GPU kernel time on cold
    `prefix128`
- that means the current bottleneck is no longer quantize / activation glue.
  It is the custom NVFP4 math core itself.

Grouped routed-matvec `ncu` update:

- focused `ncu` report for the first routed grouped matvec launch is recorded
  at:
  - `artifacts/profiles/ttft_prefill_20260406_grouped/ncu_cold_prefill_prefix128_matvec_exact.csv`
- key metrics for
  `Nvfp4MatVecRowsKernel<(bool)1>(...)`:
  - block size `256`, grid size `1856`
  - `56` registers per thread
  - `17408` bytes shared memory per block
  - occupancy limited to `4` blocks/SM by registers and `5` by shared memory
  - `sm__warps_active.avg.per_cycle_active = 30.12`, but
    `smsp__warps_eligible.avg.per_cycle_active = 0.15`
  - `sm__issue_active.avg.pct_of_peak_sustained_elapsed = 11.80%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed = 0.40%`
  - `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed = 15.14%`
  - tensor pipe activity is `0%`
- interpretation:
  - this kernel is not bandwidth-bound
  - it is also not using tensor cores
  - the main problem is scheduler starvation in a scalar/control-heavy matvec
    shape with poor warp eligibility

Failed micro-optimization note:

- a follow-up experiment replaced the per-tile 256-thread shared-memory
  reduction in `Nvfp4MatVecRowsKernel` with warp-level reduction plus a
  cross-warp finalize
- correctness stayed green, but TTFT regressed on
  `artifacts/benchmarks/ttft_20260406_grouped_warpreduce_prefix128_tail4.stdout.txt`
  compared with the retained grouped baseline in
  `artifacts/benchmarks/ttft_20260406_grouped_prefix128_tail4.stdout.txt`
  - `cold_prefill_prefix128`: `719.458 ms` -> `832.126 ms`
  - `cached_committed_head_prefix128_tail4`: `65.347 ms` -> `67.580 ms`
- that change was reverted
- conclusion: more reduction micro-polish on the current scalar matvec is not
  the right next bet

Updated next step:

- keep the grouped-path baseline from
  `artifacts/benchmarks/ttft_20260406_grouped_prefix128_tail4.stdout.txt`
- stop polishing around reduction details
- redesign the custom NVFP4 expert math around a more parallel launch shape:
  - fewer per-expert launches
  - better cross-token reuse inside the routed-up / routed-down kernels
  - a tiled/grouped math core that can materially increase warp eligibility
    and instruction issue rate

Single-launch grouped-expert follow-up:

- implemented a device-resident routed weight-view table so the grouped routed
  matvec pass can launch once per routed-up or routed-down stage instead of
  once per expert
- this does not add any runtime DtoH transfer:
  - the routed weight views are already known on the host at layer
    construction time
  - they are uploaded once to device memory during setup
  - the active runtime path uses only the device routing buffers, device
    scratch buffers, and the device weight-view table
- follow-up cleanup removed the per-call host-view dependency from
  `RunFusedMoePrefill`; the runtime prefill contract is device-only again
- focused TTFT rerun recorded at
  `artifacts/benchmarks/ttft_20260406_grouped_singlelaunch_prefix128_tail4.stdout.txt`
  improved further over the retained grouped baseline:
  - `cold_prefill_prefix128`: `719.458 ms` -> `681.460 ms`
    (`1.056x`, `5.28%` faster)
  - `cached_committed_head_prefix128_tail4`: `65.347 ms` -> `52.054 ms`
    (`1.255x`, `20.34%` faster)
- relative to the pre-refactor plan baseline:
  - `cold_prefill_prefix128`: now `4.79x` slower than `142.156 ms`
  - `cached_committed_head_prefix128_tail4`: now only `1.28x` slower than
    `40.655 ms`
- interpretation:
  - per-expert launch fan-out was a real part of the remaining regression
  - cached tail prefill is getting close to baseline
  - cold prefill is still dominated by the scalar custom NVFP4 math kernel, so
    the next step should target the math core itself rather than launch count

Reference-scope note:

- vLLM comparison in this plan uses the CUDA FP4 fused-MoE stack plus the
  Nemotron-H shared-expert wiring, which is the closest in-tree match to the
  Nano routed-plus-shared shape.
- TRT-LLM comparison on this platform should be treated as CUTLASS-path
  guidance. Its `TRTLLMGenFusedMoE` min-latency backend explicitly excludes
  `SM120`, so it is useful for operation structure but not as a direct
  same-platform execution path for the local RTX 5090 target.

## Working Hypothesis

The new contract still looks directionally right:

- no host `expert_offsets` readback
- no host loop over active experts
- no cuBLASLt plan churn on the hot path
- no runtime requirement to pack full-token activations for a library call

The likely failure is implementation quality inside the new kernel, not the
top-level contract. The benchmark counters still show the same layer run counts
as the old path, which argues against a scheduler explosion or a duplicated
high-level execution path.

The immediate suspicion is that the current custom kernel is spending most of
its time in scalarized per-token work:

- one block per token
- serial expert loop inside the block
- thread-0-only quantization and activation passes
- no grouped/token-batched expert GEMM
- no overlap between routed and shared expert work

## Benchmark Contract

Primary checkpoint cases:

- `cold_prefill_prefix1024`
- `cold_prefill_prefix4096`
- `cached_committed_head_prefix1024_tail32`
- `cached_committed_head_prefix4096_tail32`

Fast iteration cases:

- `cold_prefill_prefix256`
- `cached_committed_head_prefix256_tail32`
- a dedicated fused-MoE microbench with representative Nano dimensions

Success gates:

1. recover to at least the pre-refactor baseline
2. beat the baseline on `cold_prefill_prefix1024` and
   `cached_committed_head_prefix1024_tail32`
3. keep correctness green on focused dummy-data and manifest-smoke coverage

Stop conditions:

- If a profiling pass shows the regression is dominated by a single operation,
  optimize that operation before touching anything else.
- If two successive optimization steps each improve TTFT by less than `10%`,
  and the profile still shows the same dominating scalar hotspot, stop and
  redesign that operation boundary instead of polishing around it.
- If the current one-block-per-token design cannot get within `20%` of the old
  baseline after fixing clear serialization problems, treat that as evidence
  that the grouping/scheduling shape must change.

## Operation Catalog

This section is the backbone of the work. Every optimization step must map to
one of these operations.

### 1. Routing Contract Ingress

Our code now assumes routing is already done before fused expert execution:

- inputs: `selected_indices`, `selected_weights`, `input`, `normalized`
- no router logits inside the kernel

Current implementation:

- our kernel only copies `top_k` expert ids and weights into block-shared state
- no internal routing or token regrouping

Reference implementations:

- vLLM keeps routing separate but uses fused/grouped top-k helpers when
  available:
  - `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
- TRT-LLM supports both "routing outside the fused op" and an integrated
  min-latency form:
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py`

Assessment:

- this part is probably not the main regression source
- keep the current contract stable unless profiling proves otherwise

### 2. Token-to-Expert Work Decomposition

Current implementation:

- one CUDA block per token
- serial loop over `top_k` routed experts inside the block
- shared expert runs in the same block after routed experts finish

Reference implementations:

- vLLM converts `topk_ids` into expert-grouped token order using
  `moe_align_block_size()`:
  - `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`
- vLLM then shuffles rows and executes grouped expert matmuls over all routed
  tokens for an expert set:
  - `vllm/model_executor/layers/fused_moe/cutlass_moe.py`
- TRT-LLM follows the same high-level shape:
  - routing
  - scatter / token grouping
  - grouped `gemm1`
  - activation
  - grouped `gemm2`
  - finalize route

Assessment:

- this is the most likely architectural performance gap
- our current kernel gives up all cross-token reuse
- if profiling shows routed up/down projections dominate, we should expect this
  operation to become the main redesign target

### 3. Input Quantization / Activation Formatting

Current implementation:

- thread `0` runs `QuantizeDequantizeNvfp4Row(normalized_row, quantized_input)`
- the entire block waits at a barrier

Reference implementations:

- vLLM quantizes expert inputs in dedicated kernels over grouped routed rows:
  - `scaled_fp4_experts_quant`
  - `silu_and_mul_scaled_fp4_experts_quant`
  - invoked from `cutlass_moe.py`
- TRT-LLM describes this as dynamic quantization before or inside the fused MoE
  operator depending on backend and mode:
  - `fused_moe_cutlass.py`
  - `fused_moe_trtllm_gen.py`

Assessment:

- our contract to avoid host-side pack/scaling is fine
- the current implementation is almost certainly too serialized
- first optimization candidate inside the existing kernel shape: parallelize
  row quantization across the block and measure how much that alone recovers

### 4. Routed Up Projection

Current implementation:

- each thread computes different rows of the routed expert intermediate
- dot products are scalar row-major loops via `Nvfp4RowMajorDot`
- no cross-token batching
- no tensor-core grouped matmul

Reference implementations:

- vLLM uses grouped FP4/CUTLASS MoE matmuls over shuffled tokens:
  - `vllm/model_executor/layers/fused_moe/cutlass_moe.py`
- TRT-LLM CUTLASS path follows the same grouped `gemm1` structure:
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`

Assessment:

- this is a prime suspect for the catastrophic slowdown
- if ncu shows low tensor utilization or massive instruction count in this
  operation, the fix is not micro-polish; it is a real custom grouped matmul
  step

### 5. Routed Activation

Current implementation:

- thread `0` loops over the entire routed intermediate buffer
- applies `Relu2`
- then re-quantizes the same buffer in place

Reference implementations:

- vLLM either fuses activation with quantization or runs vectorized activation
  before quantization:
  - `cutlass_moe.py`
- TRT-LLM documents activation as a distinct step inside the fused path, not a
  thread-0 scalar loop

Assessment:

- this is another obvious serialization point
- optimize here before changing less important code

### 6. Routed Down Projection + Weighted Reduction

Current implementation:

- each routed expert contribution is projected back to hidden size
- routing weight is applied per column
- accumulation is done directly into `output_row`

Reference implementations:

- vLLM runs grouped second-stage expert GEMM, then reshuffles rows back and
  performs weighted reduction:
  - `cutlass_moe.py`
- TRT-LLM uses grouped `gemm2` followed by finalize-route

Assessment:

- direct accumulation is a reasonable contract
- but the current per-token/per-expert scalar projection is a likely hotspot
- if this dominates, direct accumulation should stay but the math kernel must
  change

### 7. Shared Expert Execution

Current implementation:

- shared up/down work is embedded into the same per-token block after routed
  experts complete
- it reuses the same scratch and the same one-token execution shape

Reference implementations:

- vLLM keeps shared experts logically separate:
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`
- TRT-LLM `SharedMoE` also computes routed and shared outputs as separate
  components and then combines them:
  - `tensorrt_llm/layers/moe.py`

Assessment:

- the current fused treatment may be correct functionally but is not aligned
  with the strongest reference shapes
- we should profile routed-only and shared-only cost separately
- likely optimization direction: split shared expert execution back out so it
  can be tuned and optionally overlapped independently

### 8. Final Output Writeback

Current implementation:

- initialize `output_row = input_row`
- add routed weighted contributions
- add shared contribution
- optional debug outputs for routed/shared contributions

Reference implementations:

- both vLLM and TRT-LLM finalize after the routed path and then combine shared
  expert output outside the grouped expert math core

Assessment:

- this is not the first place to optimize
- keep it simple unless profiling shows unexpected memory traffic

## Profiling Strategy

### Phase A. Controlled Benchmark Confirmation

Before new profiles, rerun only:

- `cold_prefill_prefix256`
- `cold_prefill_prefix1024`
- `cached_committed_head_prefix256_tail32`
- `cached_committed_head_prefix1024_tail32`

Requirements:

- isolate single-case runs
- no concurrent GPU jobs
- record exact command lines and output artifacts in this project directory

Purpose:

- verify the regression is stable and reproducible
- reduce turnaround time versus the full sweep

### Phase B. End-to-End `nsys`

Profile:

- `cold_prefill_prefix1024`
- `cached_committed_head_prefix1024_tail32`

Questions to answer:

- what fraction of TTFT is now in fused MoE prefill
- is the cost dominated by routed path, shared path, or both
- are there still hidden syncs or allocator churn around the new kernel
- how does the kernel-time split compare to
  `proj-2026-04-05-0445/post_attention_root_cause.md`

Output:

- one short root-cause note with before/after kernel-time tables

### Phase C. `ncu` on the New Kernel

Use a reduced, representative case with Nano dimensions and one or a few tokens.

Questions to answer:

- occupancy
- registers per thread
- shared-memory pressure
- local-memory spills
- tensor core usage or lack of it
- memory throughput
- dominant warp stall reasons
- instruction mix in `Nvfp4RowMajorDot` and the thread-0 quantize/activation
  sections

Purpose:

- decide whether the next step is a local parallelization fix or a math-kernel
  redesign

### Phase D. Add Micro Instrumentation

Create narrow timing hooks or micro tests around these operations:

- input row quantize/dequantize
- routed up projection
- routed activation + re-quantize
- routed down projection + weighted accumulate
- shared up/down projection

Rules:

- dummy data only
- no host copies in the timed region
- keep one micro test per operation so improvements are attributable

## Optimization Order

Optimize in descending order of likely impact and lowest redesign cost.

### Priority 1. Remove obvious serialization in the existing kernel

Candidates:

- parallelize input quantization across the block
- parallelize `Relu2`
- parallelize intermediate re-quantization
- reduce unnecessary barriers

Checkpoint after each change:

- run focused micro test
- run `fused_moe_prefill_test`
- rerun `cold_prefill_prefix256`
- rerun `cached_committed_head_prefix256_tail32`

### Priority 2. Decide whether one-block-per-token is salvageable

If Priority 1 does not recover a large fraction of the regression:

- prototype a custom grouped routed-up operation
- keep the current external contract
- move from per-token scheduling to token-grouped-by-expert scheduling

This is the likely pivot point. The references do not preserve the current
one-token scheduling shape for the expert GEMMs.

### Priority 3. Split shared experts from routed experts

If routed and shared cost are materially different:

- move shared expert execution into a separate custom path
- keep the final combine explicit
- tune routed and shared paths independently

### Priority 4. Replace scalar dot-product math with custom grouped math

If `Nvfp4RowMajorDot` dominates:

- write a real custom grouped expert matmul path
- retain the Step 3 contract
- add a dedicated microbench for:
  - grouped routed up
  - grouped routed down
  - shared up/down

## Validation Strategy

After every optimization step:

1. micro test for the touched operation
2. `fused_moe_prefill_test`
3. `expert_layer_fastpath_test`
4. `full_forward_manifest_smoke_test`
5. targeted TTFT rerun on the fast cases

Before claiming recovery:

1. rerun the full TTFT sweep
2. rerun throughput bench
3. rerun full `ctest`
4. sanity-check the interactive conversation utility

## Point-by-Point Review

1. The current regression is too large for blind tuning. Profiling first is
   mandatory.
2. The run-count parity with earlier artifacts suggests the contract is not
   fundamentally broken at the scheduler level.
3. The current kernel has multiple thread-0-only sections. Those are immediate
   targets and easy to validate with micro tests.
4. The biggest structural gap versus vLLM and TRT-LLM is token grouping for
   expert GEMMs. We should expect that to matter.
5. Shared experts should not stay fused into the same scheduling shape unless
   profiling proves the coupling is helping.
6. The stop conditions are explicit. If the current schedule shape cannot get
   close to baseline, we should stop polishing and replace the offending
   operation with a better custom kernel.
7. Baseline recovery is not the end state. The finish line for this phase is
   "better than the old baseline", not merely "less bad than today".

## First-Principles Roofline Snapshot

This section is the current reality check for the custom MoE prefill path on
the Nano 30B configuration and the local RTX 5090 target.

### Nano expert dimensions

From `runtime/src/api/single_token_forward_model.cpp`:

- hidden size: `2688`
- routed expert intermediate size: `1856`
- shared expert intermediate size: `3712`
- routed experts: `128`
- routed experts selected per token: `6`
- expert layers per forward pass on the current benchmark surface: `23`

### Expert math per token

Per expert layer, the prefill path computes:

- routed up + down: `2 * hidden * routed * top_k`
- shared up + down: `2 * hidden * shared`

For the Nano configuration, that is:

- `79,822,848` FMAs per token per expert layer
- `159,645,696` FLOPs per token per expert layer

For `prefix128` across `23` expert layers, that is:

- `234,998,464,512` FMAs
- `469,996,929,024` FLOPs

### Expert weight traffic lower bound

If the grouped kernel shape is working well, the routed path should stream each
active expert weight once per layer and reuse it across the routed tokens
assigned to that expert. For a `prefix128` batch on this model, that usually
means essentially all `128` routed experts are active.

With packed FP4 weights plus one FP8 block scale per `16` weights, the storage
cost is approximately `0.5625` bytes per weight.

That implies:

- routed + shared weights per expert layer: about `695.83 MiB`
- routed + shared weights across all `23` expert layers: about `15.63 GiB`

Using the RTX 5090 Blackwell memory-bandwidth figure of `1.792 TB/s`, the
ideal expert-weight streaming lower bound for `prefix128` is only about
`9.36 ms`.

Source:

- NVIDIA RTX Blackwell architecture appendix listing `1792 GB/sec` memory
  bandwidth for GeForce RTX 5090:
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

### Compute lower bound

Using the same NVIDIA architecture appendix, the RTX 5090 peak tensor rates
are vastly above the current effective throughput:

- peak BF16 tensor throughput with FP32 accumulate: `209.5 TFLOP/s`
- peak INT8 tensor throughput: `1676 TOPS`

Even if we pessimistically compare our expert path against the BF16 tensor
figure, the raw compute lower bound for the `prefix128` expert work is about
`2.24 ms`. So the physical floor for this path is not set by arithmetic
capacity. It is set by moving the expert weights efficiently and keeping the
machine busy while doing so.

### What the current profile says

From the retained grouped single-launch run:

- `cold_prefill_prefix128 = 681.460 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 52.054 ms`

From `artifacts/profiles/ttft_prefill_20260406_singlelaunch/`:

- total GPU kernel time per measured cold `prefix128` iteration: about
  `674.392 ms`
- grouped routed expert matvec:
  `Nvfp4GroupedExpertMatVecRowsKernel = 72.9%`
- shared expert matvec:
  `Nvfp4MatVecRowsKernel<(bool)0> = 23.0%`
- total expert matvec time per measured cold `prefix128` iteration: about
  `646.571 ms`
- all other GPU kernels combined: about `27.821 ms`

That implies:

- current effective expert throughput: about `0.73 TFLOP/s`
- current effective expert-weight streaming rate: about `24.2 GiB/s`

Relative to hardware capability, both numbers are tiny. Relative even to the
old `142.156 ms` baseline, the current cold path is still far from the limit.
The old baseline itself only implied roughly `110 GiB/s` of effective
expert-weight streaming, so it was not especially close to hardware roofline
either.

### Conclusion from first principles

The current `4.79x` cold regression is not because the old code was already
optimal. It is because the current grouped path still uses a scalar dot-product
math core that leaves most of the GPU idle.

The `ncu` data already says the same thing from the scheduler side:

- tensor pipe activity is effectively `0%`
- DRAM throughput is only `0.40%` of peak
- issue rate is low
- warp eligibility is poor

So the path forward is not "make this scalar matvec a bit nicer." The path
forward is to replace the math core with a launch shape that creates enough
parallel work, enough reuse, and enough regularity for the hardware to run near
its real limits.

## Current Focus: Math-Core Recovery Strategy

The earlier priority steps already removed the obvious serialization and
collapsed per-expert launch fan-out. The remaining work is now centered on the
math core itself.

### Operation target shape

Our current `RunFusedMoePrefill()` path still looks like this:

1. route and sort token selections
2. gather routed rows into `selection_count x hidden`
3. quantize/dequantize routed rows
4. grouped routed-up scalar matvec
5. activation + requantization
6. grouped routed-down scalar matvec
7. weighted reduction back to token order
8. separate shared up/down scalar matvec path
9. accumulate shared output into final token output

The TRT-LLM reference shape is materially different:

1. routing / renormalize
2. permute tokens into expert-major tiles
3. grouped `gemm1`
4. activation / quantization
5. grouped `gemm2`
6. finalize / unpermute / combine

That shape appears in:

- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu`
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cuteDslKernels/moeUtils.h`

The next phase should move our implementation toward that structure while
preserving a single device-only runtime contract.

### Architectural objective

For the active prefill path, the core kernel stack should eventually satisfy
all of these:

- one runtime code path
- no DtoH on the hot path
- token-to-expert regrouping fully on device
- expert-major or tile-major work decomposition
- grouped routed-up and routed-down math
- enough work per launch to keep warps eligible
- enough regularity to make tensor-core or tensor-like batched math viable
- explicit finalize/unpermute step instead of implicit scalar accumulation in
  the matvec kernel

### Priorities from here

1. Treat the current scalar grouped matvec as the baseline to beat, not the
   structure to preserve.
2. Build a dedicated operation-by-operation microbench for the expert math
   core:
   - route/permute only
   - grouped routed-up only
   - activation/quant only
   - grouped routed-down only
   - finalize/unpermute only
3. Replace routed-up first, because it is the largest cross-token reuse
   opportunity and the cleanest place to prove a new tiled math core.
4. Then replace routed-down with the same scheduling discipline.
5. After the routed path is stable and fast, re-evaluate whether the shared
   path should use the same grouped math core or a simpler specialized path.
6. Do not spend more time on reduction micro-polish unless a new profile shows
   that a replacement math core has already shifted the bottleneck elsewhere.

### Stop conditions for the current scalar kernel family

The current scalar grouped-matvec family should be considered finished unless
it can beat these bars:

- materially higher warp eligibility
- materially higher issue rate
- materially higher effective expert-weight streaming rate
- clear progress toward the old `142.156 ms` cold `prefix128` baseline

If a change does not move those metrics, it is not the right kind of change.

## Exact CTA Launch-Plan Checkpoint

The TRT-LLM-style launch-plan foundation is now integrated into the active
runtime path for both routed expert stages:

- `RunFusedMoePrefill()` now builds the device CTA plan immediately after
  `RunDeviceExpertRouting()`
- the active routed-up and routed-down stages consume
  `RunLaunchPlannedNvfp4ExpertMatVec(...)`
- the launch plan is still statically preallocated in the request-scoped MoE
  workspace; there is no hot-path DtoH and no hot-path `cudaMalloc`

Focused validation after switching the active path:

- `single_token_forward_model_test`: pass
- `moe_launch_plan_device_test`: pass
- `fused_moe_prefill_test`: pass
- `multi_turn_prefix_reuse_test`: pass

Focused design-center TTFT gate, artifact:

- `artifacts/benchmarks/ttft_20260406_launch_plan_exact_prefix128_tail4.stdout.txt`

Results against the retained grouped-launch baseline
(`artifacts/benchmarks/ttft_20260406_launch_plan_foundation_prefix128_tail4.stdout.txt`):

- `cold_prefill_prefix128`: `681.114 ms` -> `654.778 ms`
  (`1.040x`, `3.87%` faster)
- `cached_committed_head_prefix128_tail4`: `51.638 ms` -> `46.787 ms`
  (`1.104x`, `9.39%` faster)

Interpretation:

- the exact device CTA map is a real runtime win, not just a microbench win
- the earlier bad runtime result was largely due to the looser row-tile
  overlaunch strategy, not the launch-plan architecture itself

## Static Capacity vs Active Window

The next result is important for deployment strategy.

The full short-tail sweep artifact
`artifacts/benchmarks/ttft_20260406_launch_plan_exact_tail4_prefix4_128_4096.stdout.txt`
was run with the default runtime resolution and therefore reported:

- `resolved_runtime_moe_prefill_capacity_tokens=4096`
- `resolved_runtime_moe_prefill_window_tokens=4096`

Short-tail results in that fully static `4096` setup:

- `cold_prefill_prefix4`: `67.083 ms`
- `cold_prefill_prefix128`: `661.138 ms`
- `cold_prefill_prefix4096`: `20353.969 ms`
- `cached_committed_head_prefix4_tail4` hot-prefix: `67.877 ms`
- `cached_committed_head_prefix128_tail4` hot-prefix: `69.369 ms`
- `cached_committed_head_prefix4096_tail4` hot-prefix: `110.831 ms`

That exposed an architectural problem: the runtime had been coupling
preallocated MoE workspace capacity to the active MoE prefill window.

That coupling is now removed:

- `moe_prefill_capacity_tokens` controls static workspace size
- `moe_prefill_window_tokens` now clamps the active prefill window without
  shrinking the preallocated workspace
- the TTFT bench header now reports those two values separately

Proof artifact with explicit small window and large static allocation:

- `artifacts/benchmarks/ttft_20260406_launch_plan_capacity4096_window133_prefix128_tail4.stdout.txt`
- reported header:
  - `resolved_runtime_moe_prefill_capacity_tokens=4096`
  - `resolved_runtime_moe_prefill_window_tokens=133`

Measured result in that decoupled configuration:

- `cold_prefill_prefix128`: `657.918 ms`
- `cached_committed_head_prefix128_tail4` hot-prefix: `68.912 ms`

Interpretation:

- decoupling capacity from window is required and now works
- but the large-request/static-4096 short-tail slowdown is **not** primarily
  the MoE window anymore
- most of the remaining short-tail penalty under the full-static setup is
  coming from some other max-context or request-sizing effect

Current best explanation:

- exact CTA launch metadata improved the active MoE math path
- explicit small MoE window did not recover the short-tail regression once the
  request itself was still sized for `4096`
- so the next profiling target should be broader request-sized overhead:
  snapshot size, KV reservation, request views, or another max-context-scaled
  component outside the MoE window logic

## Short-Tail Static-Compare Profile

Focused `nsys` comparison artifacts:

- `artifacts/profiles/ttft_prefill_20260406_static_compare/nsys_cached_committed_head_prefix128_tail4_capacity133_window133.nsys-rep`
- `artifacts/profiles/ttft_prefill_20260406_static_compare/nsys_cached_committed_head_prefix128_tail4_capacity4096_window133.nsys-rep`

Compared setups:

- small request-sized workspace:
  - `resolved_runtime_moe_prefill_capacity_tokens=133`
  - `resolved_runtime_moe_prefill_window_tokens=133`
- large static workspace with explicit small active window:
  - `resolved_runtime_moe_prefill_capacity_tokens=4096`
  - `resolved_runtime_moe_prefill_window_tokens=133`

The `nsys` diff showed that the regression was not a broad request-sized tax.
Almost all of the extra time was in the new launch-plan path itself:

- `Nvfp4LaunchPlannedExpertMatVecRowsKernel`: `+166.781 ms` total over the
  5-iteration trace
- `BuildLaunchPlanKernel`: `+5.803 ms` total over the 5-iteration trace
- everything else was noise by comparison

Important API implication:

- `cudaFree`, `cudaStreamSynchronize`, and `cudaMalloc` also increased, but
  those deltas are downstream symptoms of slower kernels on the same stream,
  not the root cause

Root cause in our implementation:

- the runtime launch-plan buffers were statically allocated at max workspace
  capacity, which is correct
- but `LaunchPlannedMatVec()` was launching `grid.x = launch_plan->cta_capacity()`
  instead of an upper bound derived from the **current** active selection count
- `BuildLaunchPlanKernel` was also filling sentinel entries up to that same max
  capacity

For the failing case, that meant:

- actual resumed tail prefill work: `selection_count = 4 * 6 = 24`
- small request-sized workspace upper bound: `CtaCapacity(128, 798) = 211`
- large static workspace upper bound: `CtaCapacity(128, 24576) = 3184`
- ideal current-batch upper bound for the actual tail work:
  `CtaCapacity(128, 24) = 24`

So even the good small-workspace case was still overlaunching, but the full
static workspace case was overlaunching much more severely.

## TRT-LLM Comparison: The Missing Piece

TRT-LLM does two separate things here:

1. it statically allocates buffers large enough for the configured maximum
2. it computes the host launch upper bound from the **current batch token
   count**, not from the maximum buffer capacity

Relevant reference points:

- `runner.h:getMaxNumCtasInBatchDim(...)`
- `routingRenormalize/launchBlockKernel.cu`

What TRT-LLM does:

- host computes `maxNumCtasInBatchDim` from `numTokens`, `topK`,
  `numExperts`, and `tileTokensDim`
- routing writes exact device metadata:
  - `ctaIdxXyToBatchIdx`
  - `ctaIdxXyToMnLimit`
  - `numNonExitingCtas`
- grouped kernels launch against the current-batch upper bound and let the
  device metadata stop non-participating CTAs

What we were missing:

- we copied the "device exact CTA metadata" half
- but we were still using max workspace capacity for the host-side upper bound

## Dynamic Upper-Bound Fix

The runtime now keeps the static launch-plan buffers but uses the **current**
active selection count for both launch-plan build and launch:

- `BuildDeviceMoeLaunchPlan(..., active_selection_count, ...)`
- `RunLaunchPlannedNvfp4ExpertMatVec(..., active_selection_count, ...)`

That removes both sources of max-capacity tax:

- `BuildLaunchPlanKernel` no longer clears the whole max-capacity tail
- `LaunchPlannedMatVec()` no longer launches `grid.x = max_buffer_ctas`

Focused validation remains green:

- `moe_launch_plan_device_test`
- `fused_moe_prefill_test`
- `multi_turn_prefix_reuse_test`

Focused benchmark after the fix, artifact:

- `artifacts/benchmarks/ttft_20260406_launch_plan_dynamic_upper_bound_capacity4096_window133_prefix128_pair.stdout.txt`

Key result:

- with `resolved_runtime_moe_prefill_capacity_tokens=4096` and
  `resolved_runtime_moe_prefill_window_tokens=133`,
  `cached_committed_head_prefix128_tail4` improved from `68.912 ms` to
  `46.835 ms`

Interpretation:

- the remaining short-tail regression was not an unavoidable consequence of
  static preallocation
- it was a missing TRT-LLM-style dynamic CTA upper bound in our launch-plan
  integration
- static buffers plus current-batch launch bounds is the correct architecture

## Padded Routed-Row Checkpoint

The next TRT-LLM-aligned step is now landed in the active runtime contract:

- `DeviceMoeLaunchPlan` now carries padded routed-row metadata in addition to
  exact CTA metadata:
  - `permuted_token_indices`
  - `sorted_to_permuted_indices`
  - `cta_m_limits`
- request-scoped MoE workspace scratch for routed prefill is now sized to
  `PaddedRowCapacity(num_experts, selection_capacity)`, not just raw
  `selection_capacity`
- `RunFusedMoePrefill()` now uses that padded routed-row layout end to end for
  the routed path:
  - padded gather / permute from `normalized`
  - routed-up on padded rows
  - activation / requant on padded rows
  - routed-down on padded rows
  - finalize via `selection_to_sorted -> sorted_to_permuted`

Focused correctness after switching the active path:

- `moe_launch_plan_device_test`: pass
- `fused_moe_prefill_test`: pass
- `multi_turn_prefix_reuse_test`: pass

Focused design-center TTFT rerun, command:

- `NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json ./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench --prefix-length 128 --tail-token-count 4 --warmup 0 --iterations 5`

Measured result on the padded routed-row path:

- `cold_prefill_prefix128`: `654.286 ms`
- `cached_committed_head_prefix128_tail4` hot-prefix: `46.454 ms`
- `cached_global_root_prefix128_tail4` hot-prefix: `46.440 ms`

Comparison against the pre-padded-layout design-center baseline
(`artifacts/benchmarks/ttft_20260406_post_dynamic_bounds_prefix128_tail4.stdout.txt`):

- `cold_prefill_prefix128`: `651.942 ms` -> `654.286 ms`
  (`0.36%` slower)
- `cached_committed_head_prefix128_tail4`: `46.381 ms` -> `46.454 ms`
  (`0.16%` slower)

Interpretation:

- the padded routed-row layout is architecturally correct and now validated on
  the active runtime path
- but it is essentially TTFT-neutral on the design-center case
- this is strong evidence that the remaining gap is not the routed control
  plane or the compact-vs-padded layout boundary
- the remaining recovery work must target the routed and shared expert
  math cores themselves

## Routed-Up Microbench Realignment

After switching `RunLaunchPlannedNvfp4ExpertMatVec(...)` to the padded
runtime contract, the standalone routed-up benchmark needed one more fix:

- it now starts from `token_count x hidden`
- baseline and ragged variants explicitly gather compact expert-major rows via
  routing before timing the math kernel
- the launch-plan variant explicitly gathers padded routed rows via
  `permuted_token_indices`
- output comparison now remaps padded output back through
  `sorted_to_permuted`

Focused `prefix128` result after that realignment:

- `baseline`: `10.235 ms`
- `ragged_row_coop`: `9.554 ms`
- `launch_plan_upper_bound`: `9.794 ms`

Interpretation:

- the benchmark now matches the runtime contract again
- the padded launch-plan math kernel remains slightly slower than the
  benchmark-only ragged row-coop variant
- that is consistent with the near-flat TTFT result from the active runtime
  path
- the next useful change must improve the math core itself, not just the
  surrounding layout contract

## TRT-LLM Review Checkpoint

Current review against the TRT-LLM reference points to five concrete gaps that
still matter:

1. Routed expert math is still scalar row-wise matvec, not grouped GEMM.
   The active kernel still decodes FP4 weights inline and accumulates row-wise
   over `K`, while TRT-LLM feeds `PermuteGemm1` / `Gemm2` through the grouped
   batched GEMM runner.

2. Routing metadata is richer now, but the active runtime does not consume the
   full tile contract yet.
   We now build `cta_m_limits`, padded-token maps, and reverse maps, but the
   routed math and activation stages still mostly operate as row kernels over
   padded storage rather than tile-aware kernels that use the full CTA plan.

3. Permute and activation are still generic row kernels.
   TRT-LLM permutes and activates through tile-aware kernels keyed by
   `tile_idx_to_mn_limit` and `num_non_exiting_ctas`. Our current prefill path
   still does:
   - FP32 gather / permute
   - FP32 `Relu2`
   - FP32 quantize-dequantize

4. Shared expert math is still on the old scalar contiguous path.
   Even after routed launch-plan work, the shared up/down path still uses the
   same scalar matvec family and remains a substantial part of cold-prefill
   time.

5. The launch-plan builder itself is still much simpler and more serialized
   than TRT-LLM routing.
   TRT-LLM parallelizes histogram, scans, CTA-map build, and permutation in
   one routing step. Our launch-plan builder is still a small serial kernel.
   This is no longer the main design-center bottleneck, but it remains a gap
   for larger prompt lengths.

Practical conclusion:

- the contract work is now close enough to TRT-LLM that more progress from
  control-plane changes alone should be expected to be small
- the next material win must come from replacing the routed and then shared
  scalar matvec math cores with a tile- / GEMM-like implementation that
  actually uses the padded launch-plan contract

## FP32 Accumulation Checkpoint

The next benchmark-only question was whether our custom NVFP4 math core was
paying an avoidable cost by widening accumulation all the way to `double`
instead of staying closer to TRT-LLM's Blackwell-style low-precision MMA +
FP32-accumulator model.

TRT-LLM reference points:

- grouped GEMM options default the accumulator dtype to `Fp32`
- Blackwell MoE GEMM options allow low-precision MMA input types including
  `E2m1` / `E4m3`
- this reinforces that our scalar FP4 decode + `double` reduction path is not
  representative of the intended arithmetic model

First focused benchmark-only experiment:

- keep the active padded launch-plan contract and overlaunch shape unchanged
- replace only the routed-up accumulator type with `float`
- leave the rest of the kernel structure unchanged

Measured routed-up microbench results after that change:

- `prefix4`
  - `baseline`: `0.543 ms`
  - `launch_plan_fp32_accum`: `0.154 ms`
- `prefix128`
  - `baseline`: `11.129 ms` before the runtime change
  - benchmark-only `launch_plan_fp32_accum`: `2.785 ms`
- `prefix4096`
  - `baseline`: `320.731 ms`
  - `launch_plan_fp32_accum`: `98.112 ms`

The narrow benchmark diff guard did not show any observed numerical drift on
these cases, which is consistent with the project requirement now being
behavioral reuse equivalence rather than bitwise identity.

Active runtime change:

- all three custom NVFP4 matvec kernels in
  `runtime/src/backend/fused_moe_prefill.cu` now accumulate in `float`
  instead of `double`:
  - grouped routed expert matvec
  - launch-planned routed expert matvec
  - contiguous shared expert matvec

Focused correctness after landing the runtime change:

- `fused_moe_prefill_test`: pass
- `expert_routing_device_test`: pass
- `multi_turn_prefix_reuse_test`: pass

Focused plan-aligned TTFT rerun under the full-static `tail4` matrix:

- command:
  - `NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json ./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench --prefix-length 4 --prefix-length 128 --prefix-length 4096 --tail-token-count 4 --warmup 0 --iterations 5`
- measured cold medians:
  - `cold_prefill_prefix4`: `30.770 ms`
  - `cold_prefill_prefix128`: `197.220 ms`
  - `cold_prefill_prefix4096`: `6350.265 ms`
- measured cached committed-head hot-prefix medians:
  - `prefix4_tail4`: `28.996 ms`
  - `prefix128_tail4`: `31.415 ms`
  - `prefix4096_tail4`: `73.509 ms`

Comparison against the committed working baseline in `PLAN.md`:

- cached `tail4` now beats the old baseline at all three prefix lengths
- cold `prefix4` also beats the old baseline
- cold `prefix128` improves dramatically versus the recent regression state,
  but is still above the old `142.156 ms` baseline
- cold `prefix4096` remains far slower than baseline, so long-prefill recovery
  is still unresolved

Interpretation:

- widening custom NVFP4 accumulation to `double` was a major self-inflicted
  performance loss
- switching to FP32 accumulation is safe enough for the current behavioral
  reuse requirement on the focused oracle path
- the short / medium routed and shared expert paths recovered strongly from
  this change
- the next profiling step should re-check `cold_prefill_prefix128` to see how
  much MoE gap remains, and separately confirm whether long-prefill work has
  now shifted decisively toward Mamba / non-MoE costs at `prefix4096`

## Launch-Pattern Checkpoint

After the FP32-accum change, the next focused question was whether the active
launch-planned routed kernel was still leaving performance on the table simply
because of launch ordering and CTA metadata use.

Focused routed-up microbench on `prefix128`:

- retained runtime-like `launch_plan_upper_bound`: `2.776 ms`
- benchmark-local row-tile-major reordering: `2.635 ms`
- exact ragged benchmark CTA list: `1.889 ms`

That small row-tile-major win was real enough to land in the active runtime
path:

- `Nvfp4LaunchPlannedExpertMatVecRowsKernel` now launches with `grid.x =
  output_row_tile_count`, `grid.y = current_cta_capacity`, so consecutive CTAs
  stay on the same routed-row tile while sweeping output-row tiles

Focused TTFT rerun after that runtime launch-order change:

- `cold_prefill_prefix128`: `197.220 ms -> 193.538 ms`
- `cached_committed_head_prefix128_tail4` hot-prefix TTFT:
  `31.415 ms -> 30.366 ms`
- `cached_global_root_prefix128_tail4` hot-prefix TTFT:
  `31.580 ms -> 30.445 ms`

The more important result came from `ncu` on the routed-up microbench:

- exact ragged kernel:
  - grid size: `59392`
  - duration: `2.377 ms`
  - eligible warps / scheduler: `1.93`
- launch-planned kernel after flattening the launch order:
  - grid size: `96512`
  - duration: `3.441 ms`
  - eligible warps / scheduler: `1.07`

The exact CTA count for this case is only `128` routed-row tiles, but the
active launch-planned path still launches against the static upper bound
`current_cta_capacity = 208`. That multiplies by `464` output-row tiles to
produce `96512` launched CTAs instead of the `59392` real CTAs in the exact
ragged benchmark. The remaining routed-up gap is therefore not mainly
`blockIdx` ordering anymore; it is capacity-based CTA overlaunch.

TRT-LLM comparison:

- TRT routing writes exact CTA metadata such as `numNonExitingCtas`,
  `ctaIdxXyToBatchIdx`, and `ctaIdxXyToMnLimit`
- grouped Gemm1 / Gemm2 consume that exact metadata directly
- this lets TRT amortize the control-plane work inside a much stronger
  grouped-GEMM math core, instead of paying row-kernel overhead per tiny CTA

Practical conclusion:

- the current row-kernel path can still win from cleaner launch patterns, but
  the remaining gap is fundamentally tied to using a small-work-per-CTA custom
  routed kernel with an upper-bound launch count
- another round of control-plane polish is unlikely to recover the full gap
- the next material step should increase work per CTA, using the existing
  launch-plan contract as input to a more tile- / GEMM-like routed-up math
  core rather than another tiny row kernel

## Wider-CTA Routed-Up Checkpoint

The next benchmark-only question was whether simply increasing work per CTA
already moves us toward the TRT-LLM grouped-GEMM shape, even before changing
the math family itself.

New routed-up benchmark variants:

- `ragged_row_coop_tile8`
  - exact host-built CTA list
  - `8` output rows / CTA
  - `256` threads / CTA
- `launch_plan_flattened_tile8`
  - same `8` output rows / CTA
  - same static launch-plan contract
  - still launches against the upper-bound CTA capacity

Measured routed-up microbench results:

- `prefix128`
  - `ragged_row_coop`: `1.890 ms`
  - `ragged_row_coop_tile8`: `1.531 ms`
  - `launch_plan_upper_bound` with retained runtime tile8 path: `2.522 ms`
  - `launch_plan_flattened_tile8`: `2.524 ms`
- `prefix4096`
  - `ragged_row_coop`: `52.899 ms`
  - `ragged_row_coop_tile8`: `35.651 ms`
  - `launch_plan_upper_bound` with retained runtime tile8 path: `81.299 ms`
  - `launch_plan_flattened_tile8`: `81.274 ms`

Focused runtime TTFT after promoting `8` output rows / CTA to the active
launch-planned routed kernel:

- `cold_prefill_prefix128`: `193.538 ms -> 190.134 ms`
- `cached_committed_head_prefix128_tail4` hot-prefix TTFT:
  `30.366 ms -> 30.547 ms`
- `cached_global_root_prefix128_tail4` hot-prefix TTFT:
  `30.445 ms -> 30.358 ms`

Interpretation:

- increasing work per CTA is absolutely the right direction
- the exact-CTA tile8 ragged kernel is much faster than the tile4 ragged kernel
- but the upper-bound launch-plan path still captures only a small fraction of
  that benefit

This clarifies the architecture gap relative to TRT-LLM:

- TRT routing produces exact CTA metadata such as `numNonExitingCtas` and CTA
  maps that grouped Gemm1 / Gemm2 consume directly
- our active runtime still relies on an upper-bound CTA launch and early exits
- for `prefix128`, that overlaunch factor is still large
- for `prefix4096`, where overlaunch is small, the remaining gap indicates we
  also need a richer exact task map, not just a larger output tile

Practical conclusion:

- the next useful runtime contract extension is an exact routed task map that
  is closer to TRT-LLM's CTA map shape
- specifically, the launch plan should be able to materialize one task record
  per real `(expert, row_tile, output_row_tile)` block, not just per routed-row
  tile
- that still keeps the runtime device-only and statically allocated, but gives
  the math kernel the exact grouped-task list it needs

## Active Micro-Plan

Current state, cleaned up:

- routed-up math is now much better than the regression state, but still
  dominated by a custom row kernel
- wider `8`-row CTAs are the right direction
- the active runtime path still leaves performance on the table because its
  launch plan stops at routed-row tiles instead of continuing to exact routed
  tasks
- behavioral reuse is the correctness gate; bitwise identity is not required

Immediate micro-plan:

1. Extend `DeviceMoeLaunchPlan` so it can store an exact routed task map, not
   just routed-row tiles.
2. Build that exact task map on device after the existing routed-row plan is
   built, using statically preallocated buffers sized at workspace creation.
3. Add a benchmark-only exact-task consumer that takes the device-built task
   map instead of the current host-built exact ragged metadata.
4. Compare that benchmark-only exact-task consumer against:
   - retained active runtime-like launch-plan tile8
   - host-built exact ragged tile8
5. If the device-built exact-task consumer wins materially, promote it into the
   runtime routed-up path first.
6. After routed-up wins in runtime TTFT, apply the same exact-task contract to
   routed-down.
7. Only after routed-up and routed-down consume the richer exact-task contract
   should we spend more time on shared-expert math-core changes.

TRT-LLM comparison, line by line:

1. TRT routing computes `numCtaPerExpert`, `ctaOffsetPerExpert`, and
   `numNonExitingCtas`.
   Our next step: compute exact routed task count on device from the existing
   routed-row plan and store it in the launch plan.

2. TRT routing writes `ctaIdxXyToBatchIdx`.
   Our next step: write one device task record per real
   `(expert, row_tile, output_row_tile)` block. The first version can encode
   this as `task_cta_index + task_output_row_base`.

3. TRT routing writes `ctaIdxXyToMnLimit`.
   Our next step: keep `cta_valid_rows` / `cta_m_limits` on the routed-row plan
   and let each exact task reference that row-tile metadata.

4. TRT grouped Gemm1 / Gemm2 consume exact routing metadata directly.
   Our next step: make the benchmark and then the runtime routed kernels consume
   the exact task map directly instead of deriving work from an upper-bound
   launch shape.

5. TRT keeps everything device-side and statically provisioned for the chosen
   workspace.
   Our next step: do the same. No hot-path DtoH, no per-call allocation, and no
   alternate runtime path.

Intentional divergence to watch carefully:

- TRT-LLM has a mature grouped-GEMM launch/runtime stack that can consume exact
  CTA metadata directly.
- We do not have that stack yet.
- So the first local end-state is not “fully port TRT-LLM”; it is “make our
  runtime consume an exact device-built routed task map, with a kernel that
  benefits from it materially.”
- That work is not throwaway because the exact task map is the same contract a
  stronger grouped-GEMM-style kernel will eventually want.

### Exact-task checkpoint

Status:

- landed a device-built exact task map in `DeviceMoeLaunchPlan`
- first version stored `(task_cta_index, task_output_row_base)`
- second version flattened that to direct task records:
  `task_expert_id`, `task_row_start`, `task_valid_rows`,
  `task_output_row_base`
- both variants passed focused correctness tests

Focused routed-up microbench result:

- `prefix128`
  - `ragged_row_coop_tile8`: `1.530 ms`
  - `launch_plan_upper_bound`: `2.528 ms`
  - `launch_plan_exact_task_tile8`: `2.537 ms`
- `prefix4096`
  - `ragged_row_coop_tile8`: `35.666 ms`
  - `launch_plan_upper_bound`: `81.360 ms`
  - `launch_plan_exact_task_tile8`: `82.324 ms`

Conclusion:

- flattening the exact task record removed an extra metadata indirection, but it
  did not materially move the routed-up kernel
- the remaining gap is therefore not mainly `task -> cta` pointer chasing
- comparing against TRT-LLM makes the deeper mismatch clear: our current exact
  task map is still the task map for a row kernel, not the CTA map for grouped
  GEMM tiles

TRT-LLM-aligned interpretation:

- TRT routing writes `ctaIdxXyToBatchIdx`, `ctaIdxXyToMnLimit`,
  `numNonExitingCtas`, and padded-token metadata for grouped GEMM runners
- TRT does not externalize output-row tiles as a separate task list the way our
  current `task_output_row_base` path does
- TRT grouped GEMM kernels consume CTA metadata where the CTA is already a GEMM
  work tile; output/N tiling is internal to the math kernel
- our current launch-plan exact-task path still uses CTA = `(expert row tile,
  output row tile)` for a scalar row-coop kernel, so even “exact task” remains
  the wrong work unit

Updated immediate next step:

1. Stop using host-ragged row-kernel speed as the architectural target.
2. Keep the exact launch-plan foundation; it is still useful.
3. Build the next routed-up microbench around TRT-like grouped work units:
   CTA metadata should identify routed batch/expert work and token limits, while
   the kernel computes a larger `M x N` tile internally.
4. Remove `task_output_row_base` from the design center once the grouped-tile
   microbench exists, because that field is a symptom of the row-kernel model,
   not the TRT grouped-GEMM model.

### First grouped-tile attempt

Implemented:

- a benchmark-only `launch_plan_grouped_tile32` routed-up kernel
- scheduler shape:
  - `grid.y = cta_count` from the routed launch plan
  - `grid.x = ceil_div(intermediate_size, 32)`
  - each CTA consumes the existing routed-row launch-plan metadata directly

Focused routed-up microbench result:

- `prefix128`
  - `launch_plan_upper_bound`: `2.528 ms`
  - `launch_plan_grouped_tile32`: `2.774 ms`
- `prefix4096`
  - `launch_plan_upper_bound`: `81.537 ms`
  - `launch_plan_grouped_tile32`: `89.809 ms`

Interpretation against TRT-LLM:

- this was the right scheduler direction, but still the wrong math core
- the current `tile32` kernel is still a row kernel internally:
  each warp loops over multiple output rows and accumulates them serially
- TRT grouped GEMM kernels do not serialize output rows that way; warps
  cooperate on a GEMM tile and the `N` dimension is internal to the MMA/tile
  algorithm, not a per-warp serial loop
- so this result should not push us back to row-task variants; it tells us the
  next attempt must change the intra-CTA work decomposition, not just the CTA
  scheduler

Updated next implementation step:

1. keep the current launch-plan scheduler as the control-plane baseline
2. prototype a benchmark-only routed-up kernel where warps cooperate on an
   `M x N` tile instead of one warp owning one output row stream
3. only compare new variants against TRT-like grouped scheduling principles:
   direct CTA metadata, internal `N` tiling, and larger work units per CTA

### Cooperative `M8 x N32` checkpoint

Implemented:

- benchmark-only cooperative routed-up kernel with:
  - CTA scheduler from the current launch plan
  - `M = 8` routed rows per CTA
  - `N = 32` output rows per CTA
  - shared-memory staging for both the input tile and the decoded weight tile
  - one thread per output element inside the `8 x 32` tile
- promoted that cooperative kernel into the active runtime
  `LaunchPlannedMatVec()` path

Focused routed-up microbench after runtime promotion:

- `prefix128`
  - retained row-style launch-planned runtime path had been about `2.528 ms`
  - active `launch_plan_upper_bound`: `2.409 ms`
  - `launch_plan_cooperative_m8n32`: `2.408 ms`
- `prefix4096`
  - retained row-style launch-planned runtime path had been about `81.537 ms`
  - active `launch_plan_upper_bound`: `71.207 ms`
  - `launch_plan_cooperative_m8n32`: `71.213 ms`
- `launch_plan_microtile2_m8n32` regressed badly and is not a candidate:
  - `prefix128`: `4.191 ms`
  - `prefix4096`: `137.772 ms`

Focused TTFT after runtime promotion:

- command:
  - `env NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json ./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench --prefix-length 128 --tail-token-count 4 --warmup 0 --iterations 5`
- results:
  - `cold_prefill_prefix128`: `184.083 ms`
  - `cached_committed_head_prefix128_tail4` hot-prefix TTFT: `32.423 ms`
  - `cached_global_root_prefix128_tail4` hot-prefix TTFT: `32.323 ms`

Interpretation:

- this is the first post-regression runtime change that clearly moves the
  active launch-planned path toward TRT-style grouped execution and improves the
  design-center cold prefill case materially
- relative to the previously retained active runtime checkpoint
  (`cold_prefill_prefix128 ~= 190.134 ms`, hot-prefix `~= 30.547 ms`),
  the cooperative kernel improved cold prefill by about `3.2%` while regressing
  hot-prefix TTFT by about `6%`
- that trade is acceptable for now because cold prefill is the primary recovery
  target
- the remaining gap to the fast ragged benchmark confirms that the next step is
  still the math core, not more scheduler bookkeeping

Next TRT-aligned step:

1. keep the cooperative `M8 x N32` kernel as the active runtime baseline
2. remove dead-end row-task assumptions from the launch-plan design center
3. prototype the next grouped routed-up math core around a true thread
   micro-fragment / GEMM tile decomposition, not around serial row groups
4. only after routed-up advances again should routed-down be migrated to the
   same math family

### Transposed BF16 WMMA checkpoint

Behavioral correctness requirement for this phase:

- the gate is behavioral reuse equivalence, not bitwise identity
- benchmark-level BF16 drift is acceptable if:
  - `multi_turn_prefix_reuse_test` still passes
  - committed-head reuse does not diverge
  - TTFT/cached behavior remains correct on the Nano oracle path

TRT-LLM alignment that mattered here:

- TRT `PermuteGemm1` runs with `routeAct=true` and
  `transposeMmaOutput=true`
- in practice that means grouped GEMM treats routed weights as the MMA `A`
  operand, routed activations as `B`, and computes an output tile in the
  transposed orientation before the later remap/finalize stage
- our first WMMA benchmark prototype did not follow that shape and produced
  large numeric errors even though it was fast

Implemented:

- benchmark-only transposed BF16 WMMA routed-up kernel in
  `benchmarks/nano_moe_prefill/nano_routed_up_bench.cpp`
- active runtime launch-planned matvec migrated to the same transposed WMMA
  math core in `runtime/src/backend/fused_moe_prefill.cu`
- kept the existing device launch plan and static workspace contract unchanged

Focused routed-up microbench:

- artifact:
  `artifacts/benchmarks/nano_routed_up_bench_20260406_wmma.stdout.txt`
- `prefix128`
  - current runtime-like cooperative path:
    `launch_plan_upper_bound = 2.408 ms`
  - buggy non-TRT WMMA prototype:
    `launch_plan_wmma_bf16_m16n32k16 = 1.110 ms`
    with `max_abs_diff_vs_baseline = 16.448`
  - TRT-aligned transposed WMMA:
    `launch_plan_wmma_bf16_transposed_m16n32k16 = 1.507 ms`
    with `max_abs_diff_vs_baseline = 0.020`
- `prefix4096`
  - current runtime-like cooperative path:
    `launch_plan_upper_bound = 71.212 ms`
  - TRT-aligned transposed WMMA:
    `launch_plan_wmma_bf16_transposed_m16n32k16 = 27.060 ms`
    with `max_abs_diff_vs_baseline = 0.020`
- `prefix4`
  - current runtime-like cooperative path:
    `launch_plan_upper_bound = 0.154 ms`
  - TRT-aligned transposed WMMA:
    `launch_plan_wmma_bf16_transposed_m16n32k16 = 0.370 ms`
    with `max_abs_diff_vs_baseline = 0.020`

Interpretation:

- this is the first tensor-core routed-up prototype that is both fast and
  numerically well-behaved enough to satisfy the behavioral reuse bar
- it is a clear cold-prefill win for `prefix128` and `prefix4096`
- it is a clear small-`M` loss at `prefix4`, which matches the earlier regime
  analysis: this first WMMA kernel pads the token/tile `M` dimension
  aggressively and wastes work when only a few routed rows are active

Validation after runtime promotion:

- `fused_moe_prefill_test`: pass
- `expert_routing_device_test`: pass
- `multi_turn_prefix_reuse_test`: pass

Focused TTFT after runtime promotion:

- artifact:
  `artifacts/benchmarks/ttft_20260406_transposed_wmma_prefix128_tail4.stdout.txt`
- command:
  - `env NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json ./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench --warmup 1 --iterations 5 --prefix-length 128 --tail-token-count 4 --case cold_prefill_prefix128 --case cached_committed_head_prefix128_tail4 --case cached_global_root_prefix128_tail4`
- results:
  - `cold_prefill_prefix128 = 137.825 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix TTFT = 40.085 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix TTFT = 39.951 ms`

Interpretation:

- cold prefill now beats the older recorded baseline
  (`~142.156 ms`) on the design-center case
- resumed `tail4` TTFT regressed relative to the prior cooperative WMMA-free
  runtime checkpoint because the current transposed WMMA kernel is inefficient
  in the tiny-`M` regime
- despite that regression, the resumed `tail4` case is still essentially on
  top of the older project baseline (`~40.655 ms`), so this is an acceptable
  prefill-first checkpoint

Updated cold-prefill profile:

- artifact:
  `artifacts/profiles/ttft_prefill_20260406_wmma/cold_prefill_prefix128.cuda_gpu_kern_sum.csv`
- top kernels:
  - `Nvfp4LaunchPlannedExpertMatVecRowsKernel = 53.1%`
  - `Nvfp4MatVecRowsKernel<false> = 24.4%`
  - Mamba prefill kernels together are far behind those two

Interpretation against TRT-LLM:

- routed experts are much healthier, but the active path is still not a full
  TRT-style grouped GEMM scheduler
- the shared expert path is now a much larger fraction of cold prefill than it
  was earlier because it still runs on the old scalar contiguous matvec kernel
- the next optimization target should therefore split in two:
  - move shared experts onto the same stronger tensor-core math family
  - continue improving small-`M` routed scheduling so resumed short tails do
    not pay the full padded `M16` cost

Next step:

1. keep the transposed WMMA routed path as the active baseline
2. profile or microbench the shared expert path directly
3. implement a TRT-aligned tensor-core contiguous/shared expert kernel
4. after shared no longer consumes ~24% of cold prefill, return to small-`M`
   routed scheduling and pack the tiny resumed-tail regime more efficiently

### Shared expert transposed WMMA checkpoint

Why this was the next target:

- after the routed WMMA promotion, cold-prefill profiling showed
  `Nvfp4MatVecRowsKernel<false>` at `24.4%` of cold `prefix128`
- that kernel is the shared expert path (`shared_up` and `shared_down`)
- TRT-LLM `Gemm2` uses the same broad policy shape as `PermuteGemm1`:
  `transposeMmaOutput = true`, but `routeAct = false`

Implemented:

- moved `LaunchContiguousMatVec()` in
  `runtime/src/backend/fused_moe_prefill.cu` onto the same transposed BF16
  WMMA math family as the routed path
- kept a single kernel family and added a deterministic host-side output-tile
  chooser for the shared path
- hardware/model-specific inputs to that chooser:
  - RTX 5090 device reports `170` SMs
  - WMMA token tile is fixed at `M = 16`
  - candidate output tiles are `32`, `64`, and `128`
  - choose the largest output tile whose CTA count
    `ceil(N / tile) * ceil(M / 16)` still covers the SM count; otherwise use
    `32`

Observed tile behavior before the selector:

- fixed `32` tile:
  - `cold_prefill_prefix128 = 127.768 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 54.565 ms`
- fixed `64` tile:
  - `cold_prefill_prefix128 = 126.926 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 54.752 ms`
- fixed `128` tile:
  - `cold_prefill_prefix128 = 126.042 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 59.187 ms`

Interpretation:

- a larger shared output tile helps cold prefill because `prefix128` has enough
  token tiles to keep the GPU busy
- the same larger tile hurts resumed `tail4` because `M = 4` only produces a
  single token tile, so overly large output tiles underfill the SMs and waste
  work
- this is exactly the sort of shape-sensitive tradeoff TRT-LLM resolves with
  config selection rather than one universal constant

Deterministic selector checkpoint:

- artifact:
  `artifacts/benchmarks/ttft_20260406_shared_wmma_selected_prefix128_tail4.stdout.txt`
- results:
  - `cold_prefill_prefix128 = 126.201 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix TTFT = 54.239 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix TTFT = 54.332 ms`

Interpretation:

- the selector recovered the best parts of the fixed-tile experiments:
  - essentially the best cold-prefill result from the larger shared tile family
  - slightly better resumed `tail4` TTFT than the fixed `32` and fixed `64`
    variants
- this is still much slower than the older non-WMMA resumed-tail checkpoint,
  so the next recovery target is still small-`M` routed/shared scheduling, not
  more large-`M` cold tuning

Updated cold-prefill profile after shared WMMA + deterministic selection:

- artifact:
  `artifacts/profiles/ttft_prefill_20260406_shared_wmma64/cold_prefill_prefix128.cuda_gpu_kern_sum.csv`
- top kernels:
  - `Nvfp4LaunchPlannedExpertMatVecRowsKernel = 57.9%`
  - `Nvfp4ContiguousWmmaMatVecRowsKernel = 17.5%`

Interpretation:

- the shared expert path dropped from `24.4%` to `17.5%` of cold prefill
- routed experts are now clearly the dominant remaining cold-prefill bucket
- the next step should return to routed small-`M` / short-tail efficiency from
  this stronger cold-prefill baseline
## TRT-LLM Nano Serve Setup

For local TRT-LLM comparison work on RTX 5090, prefer the PyTorch backend serve
path over TensorRT engine build for NemotronH. In our local `1.3.0rc11`
checkout, `NemotronHForCausalLM` is supported and benchmarkable through
`trtllm-serve` / `trtllm-bench` on the PyTorch backend, while the older engine
`MODEL_MAP` path still rejects NemotronH.

Local files:
- `proj-2026-04-05-1704/trtllm_nano_serve.yaml`
- `proj-2026-04-05-1704/run_trtllm_nano_serve.sh`

Current local Nano-on-5090 serve assumptions:
- backend: `pytorch`
- single GPU: `tensor_parallel_size=1`, `pipeline_parallel_size=1`
- `moe_config.backend: CUTLASS`
- `kv_cache_config.dtype: fp8`
- `kv_cache_config.mamba_ssm_cache_dtype: float32`
- chunked prefill enabled

If we later prove that `float16` Mamba cache with stochastic rounding is stable
enough on Nano, we can revisit that knob for memory/perf tradeoffs. The initial
local target is a working, benchmarkable server path that stays close to the
official cookbook structure while fitting RTX 5090 constraints.

Local benchmark entry points:
- `proj-2026-04-05-1704/trtllm_nano_ttft_random_tokens.py`
- `proj-2026-04-05-1704/trtllm_nano_throughput_random_tokens.py`

These wrappers:
- launch the local `trtllm-serve` Nano server
- wait for `/v1/models`
- drive `benchmark_serving.py` against the OpenAI chat endpoint
- use `prompt_token_ids` with random fixed-length token inputs to avoid
  chat-template/tokenization drift
- shut the server back down after each benchmark run

Current local external baselines on RTX 5090:

- TRT-LLM PyTorch serve artifacts:
  - `artifacts/benchmarks/trtllm_serve_ttft_prefix4_out1_20260406_current.json`
  - `artifacts/benchmarks/trtllm_serve_ttft_prefix128_out1_20260406_current.json`
  - `artifacts/benchmarks/trtllm_serve_ttft_prefix4096_out1_20260406_current.json`
  - `artifacts/benchmarks/trtllm_serve_throughput_prompt16_gen16_20260406_current.json`
- TRT-LLM PyTorch serve results:
  - TTFT median:
    - `prefix4 -> 92.335 ms`
    - `prefix128 -> 160.844 ms`
    - `prefix4096 -> 375.780 ms`
  - throughput (`prompt16/gen16`, non-streaming):
    - `request_throughput = 12.640 req/s`
    - `output_throughput = 179.073 tok/s`
    - `total_token_throughput = 381.320 tok/s`
    - `mean_tpot = 115.851 ms`
- vLLM artifacts:
  - `artifacts/benchmarks/vllm_ttft_20260406_clean_prefix1_4_128_4096.json`
  - `artifacts/benchmarks/vllm_generate_throughput_16prompt_16gen_20260406_current.json`
- vLLM results:
  - TTFT median:
    - `prefix4 -> 34.445 ms`
    - `prefix128 -> 31.469 ms`
    - `prefix4096 -> 78.328 ms`
  - throughput (`prompt16/gen16`):
    - `full_decode_tokens_per_second = 40.856 tok/s`

Caveats:

- The current vLLM setup still reports `num_cached_tokens = 0` on every
  iteration, so its cache-hit/decode-side numbers are not trustworthy prefix
  reuse comparisons yet.
- The TRT-LLM throughput run is `--non-streaming`, so its TTFT field is the
  upstream sentinel `-1000 ms`; use the dedicated one-token TTFT artifacts for
  TTFT and the throughput artifact only for throughput / TPOT.
- The TRT-LLM path that works locally for NemotronH is the PyTorch backend
  serve path, not TensorRT engine build.

External comparison work now has a dedicated execution log in:

- `proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md`

That note is the source of truth for:

- exact external benchmark commands
- artifact paths
- apples-to-apples throughput work status
- vLLM prefix-caching investigation status
- cross-codebase prefill profiling status
