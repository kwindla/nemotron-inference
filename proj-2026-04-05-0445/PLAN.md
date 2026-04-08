# Plan: Post-Attention Routed MoE / Mamba Optimization

Project directory: `./proj-2026-04-05-0445`

## Status

This plan supersedes the earlier "fused grouped MoE prefill kernel" plan that
was written before the attention rewrite landed. The attention-first plan in
`ATTENTION_PLAN.md` is now complete. The fresh post-attention profile is saved
in `post_attention_root_cause.md`.

As of `2026-04-06`, this file is again the canonical optimization plan.
The later project directory `proj-2026-04-05-1704` now serves as the execution
log / artifact bundle for the recovery work:

- `proj-2026-04-05-1704/STEP3_RECOVERY_PLAN.md`
- `proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md`

## Canonical Update (2026-04-06)

## Routed FP4 Update (2026-04-08)

The unified swizzled FP4 `P5` path is now the default routed FC1 profile.

What changed:
- added standalone
  `testing/backend/p5_swizzled_pipeline_test.cu` to prove the same swizzled
  operand path that finally resolved `P15`
- switched default routed `P5` dispatch from the older exact FP4 kernel to the
  unified swizzled FP4 kernel after the standalone test and the live focused
  gates all passed
- added TTFT smoke tests for:
  - `prefix128 / tail4`
  - `prefix256 / tail128` committed-head
  - `prefix256 / tail128` global-root

Current focused validation:
- `fused_moe_prefill_test`: pass
- `multi_turn_prefix_reuse_test`: pass
- targeted `ctest` over the two focused integration tests, the standalone
  `p5_swizzled_pipeline_test`, and the three TTFT smoke tests: `6/6` passed

Current TTFT smoke results on the live default unified `P5` path:
- `prefix128 / tail4`
  - `cold_prefill_prefix128 = 161.579 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 63.801 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix = 63.917 ms`
- `prefix256 / tail128`
  - `cached_committed_head_prefix256_tail128 hot-prefix = 165.190 ms`
  - `cached_global_root_prefix256_tail128 hot-prefix = 165.499 ms`

Measured benefit versus the immediately previous default exact `P5` path:
- `prefix128 / tail4`
  - cold prefill: `256.344 -> 161.579 ms`
  - committed-head hot-prefix: `73.566 -> 63.801 ms`
  - global-root hot-prefix: `73.975 -> 63.917 ms`
- `prefix256 / tail128`
  - committed-head hot-prefix: `267.605 -> 165.190 ms`
  - global-root hot-prefix: `263.181 -> 165.499 ms`

Interpretation:
- unified `P5` is now clearly better than the old exact `P5` on the production
  TTFT smoke regimes, so keeping the old exact `P5` path as the default no
  longer makes sense
- the cold-TTFT gap versus the older `~120 ms prefix128` checkpoints remains,
  so the remaining regression is broader than `P5` alone
- next optimization work should treat `P5` as landed and move on to the
  remaining routed FP4 kernels / broader prefill overhead

### Current Objective

The target is no longer just "recover the old native baseline." The target is:

1. beat local `vLLM` cold prefill
2. beat local `TRT-LLM` PyTorch cold prefill
3. do it on one active runtime path, with device-only execution and behavioral
   reuse equivalence preserved

### Competition Surface

Use the direct exact-token prefill profile surface from
`proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md` as the race metric.

Current measured times on the local Nano checkpoint / RTX 5090:

| runtime | prefix4 | prefix128 | prefix4096 |
|---|---:|---:|---:|
| native runtime | `54.016 ms` | `127.698 ms` | `2503.116 ms` |
| vLLM | `48.011 ms` | `40.396 ms` | `82.938 ms` |
| TRT-LLM | `35.922 ms` | `38.407 ms` | `66.789 ms` |

That means the current external targets to beat are:

- `prefix4 < 35.922 ms`
- `prefix128 < 38.407 ms`
- `prefix4096 < 66.789 ms`

### Current Root Cause

The current cross-codebase profiles now make the work order much clearer than
the older internal-only `1024` trace:

- `prefix4`: native is still split between routed expert custom math and
  shared expert WMMA
- `prefix128`: native is still dominated by routed expert custom math
- `prefix4096`: native is dominated by routed expert custom math and then by
  Mamba prefill

By contrast, both `vLLM` and `TRT-LLM` spend these regions mostly in grouped
`CUTLASS` / `FP4` GEMM kernels with much smaller glue overhead, and their
long-prefix Mamba cost is much lower.

### Locked Contract Addendum

In addition to the original contract below, the active fast path is now locked
to these constraints:

- one active runtime path
- device-only runtime contract
- no hot-path `DtoH`
- behavioral reuse equivalence is the correctness bar, not bitwise identity

### Current Execution Order

This is the active order of operations for the optimization work:

1. finish the routed MoE transition to grouped `FP4xFP4` tensor-core math
2. move shared experts onto the same stronger kernel family
3. then attack long-prefix Mamba
4. only then spend time on attention polish or smaller cleanup

The immediate implementation gap is now very specific:

- the runtime already has device launch-plan infrastructure
- the runtime already has preallocated `DeviceNvfp4Matrix` work buffers
- but the active fused prefill path still uses a BF16/FP32-oriented routed
  kernel shape instead of the intended grouped `FP4xFP4` contract from step `3`

So the next code changes should start from the packed-activation / grouped-GEMM
boundary, not from more row-kernel scheduling tweaks.

### Immediate Prototype Finding

The first benchmark-only packed-activation prototype is now in
`benchmarks/nano_moe_prefill/nano_routed_up_bench.cpp`.

At the design-center `prefix128` case:

- `launch_plan_wmma_bf16_transposed_m16n32k16`: about `1.509 ms`
- `launch_plan_wmma_packed_input_transposed_m16n32k16`: about `2.193 ms`
- max diff vs baseline: about `2.318`

At `prefix4096`:

- `launch_plan_wmma_bf16_transposed_m16n32k16`: about `27.216 ms`
- `launch_plan_wmma_packed_input_transposed_m16n32k16`: about `42.512 ms`
- max diff vs baseline: about `2.318`

Interpretation:

- the existing `DeviceNvfp4Matrix` activation contract is not a direct
  substitute for the current BF16 routed-up kernel
- simply feeding packed activations into the current WMMA tile is slower and
  materially changes numerics
- the next grouped `FP4xFP4` step therefore needs a better packed-activation
  contract and a math core designed for it, not just a storage-format swap

### Contract Progress Update

That next packed-activation contract is now partially landed on the active
runtime side:

- routed prefill launch-plan rows are expert-major and padded to the routed
  token tile (`16`) rather than separated by standalone `128`-row expert gaps
- routed reduction now correctly maps `selection_to_sorted` through
  `sorted_to_permuted_indices`, which restored behavioral reuse equivalence
  after the padded-layout switch
- the static MoE workspace now carries a prefill `normalized_pack` source
  buffer plus grouped NVFP4 pack storage
- FC1 now packs normalized activations once and then permutes packed rows into
  grouped padded order, instead of gathering FP32 rows and repacking them
- the grouped packed-row helper preserves the source tensor scale and zeroes
  padded rows
- the active packed path now no longer spends time computing the dead
  per-expert activation-scale metadata that the current consumer does not use

What is still missing is the remaining TRT output-side contract:

- FC2 still starts from FP32 scratch plus a neutral grouped repack, not from a
  TRT-like `gemm1_output_scale` / activation-output-scale contract
- the next step is to move routed `gemm1` / `gemm2` onto the same grouped FP4
  activation contract end to end, not just on the FC1 source side

Focused validation is currently green:

- `device_nvfp4_matrix_test`
- `moe_launch_plan_device_test`
- `fused_moe_prefill_test`
- `multi_turn_prefix_reuse_test`

Focused TTFT on the design-center `prefix128 / tail4` case after moving FC1 to
per-expert input scales is:

- `cold_prefill_prefix128 = 125.352 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.328 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.612 ms`

The next FC2/output-scale step is now also landed in the working tree:

- grouped per-expert packing again honors expert tensor scales
- routed-down packed input now consumes those FC2 expert scales instead of
  treating FC2 input as a neutral matrix contract
- focused correctness remains green

Focused TTFT on the same gate with that FC2 expert-scale contract is:

- `cold_prefill_prefix128 = 126.679 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.423 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.766 ms`

The next TRT-style contract step is now landed too:

- the active packed prefill path no longer keeps `routed_up_scratch` as the
  FC1->FC2 boundary
- it now computes Gemm1/activation output scales directly from the packed FC1
  output and packs activated rows straight into the grouped FC2 input matrix
- `routed_up_scratch` remains only as a fallback-only path, not the active
  packed contract
- packed-path focused tests now run with `routed_up_scratch = nullptr`

Focused TTFT on the same gate after removing the FP32 boundary is:

- `cold_prefill_prefix128 = 126.959 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.447 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.684 ms`

Interpretation:

- the direct Gemm1-output contract is now correct and green
- this step is structurally important, but it is still roughly
  performance-neutral on the `prefix128 / tail4` gate
- the next remaining win has to come from stronger grouped math, not from more
  boundary cleanups

The FC1 contract then moved one more step toward TRT-LLM:

- the active packed routed FC1/FC2 path now executes from the planned CTA grid
  instead of the exact-task queue
- the active planned packed kernels derive token work from `cta_m_limits`
  with implicit `tileTokensDim` row starts, instead of consuming a separate
  `row_start + valid_rows` task contract
- the runtime file no longer carries a second packed routed consumer

Focused TTFT on the same gate after this FC1 alignment work is:

- `cold_prefill_prefix128 = 125.592 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.557 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.158 ms`

The next routed-stage cutover is now also landed:

- the active grouped FC1 kernel stays on the transposed WMMA body, but now
  writes a BF16 `gemm1_output` buffer directly instead of re-running FC1 to
  derive the FC2 input contract
- activation scales are now computed from BF16 `Relu2(gemm1_output)` rows, and
  those activated BF16 rows are packed directly into the grouped FC2 input
  contract
- the active grouped path no longer depends on `routed_up_scratch`; the packed
  focused tests again run with `routed_up_scratch = nullptr`
- focused correctness remains green:
  `device_nvfp4_matrix_test`, `moe_launch_plan_device_test`,
  `fused_moe_prefill_test`, `multi_turn_prefix_reuse_test`

Focused TTFT on the same gate after the BF16 `gemm1_output` seam is:

- `cold_prefill_prefix128 = 126.390 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.336 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.333 ms`

Interpretation:

- this is the correct TRT-style direction for the routed FC1->FC2 boundary
- it removes duplicate FC1 work and keeps behavioral reuse equivalence green
- the gain is still modest, which confirms the next win must come from the
  grouped FC1/FC2 math core itself rather than more boundary cleanups

The next routed-stage contract cutover is now also landed:

- `gemm1_output_scale` / `activation_output_scale` are no longer just
  per-expert placeholders on the active path; the BF16 packer now writes real
  per-row / per-block dequant scales for the routed FC2 input contract
- the active FC2 consumer now decodes from those routed dequant scales directly
  instead of reconstructing scale from `expert tensor scale + encoded block
  scales`
- the active routed FC1 and FC2 kernels now derive token work from
  `cta_idx_xy_to_batch_idx + cta_idx_xy_to_mn_limit + expert_first_token_offsets +
  selected_token_tile`, rather than the older explicit
  `cta_row_starts + cta_valid_rows` contract
- focused correctness remains green:
  `fused_moe_prefill_test`, `moe_launch_plan_device_test`,
  `multi_turn_prefix_reuse_test`

Focused TTFT on the same gate after these contract and consumer cutovers is:

- `cold_prefill_prefix128 = 126.178 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 54.361 ms`
- `cached_global_root_prefix128_tail4 hot-prefix = 54.652 ms`

Interpretation:

- this is now very close to the TRT grouped-contract shape for the active
  routed path
- TTFT is still effectively flat, which means the remaining gap is not in the
  routed metadata or FC1->FC2 scale contract anymore
- the next step must jump to the grouped FC1/FC2 kernel body itself

### Next Jump: Match TRT-LLM For The Routed Math Stage

Yes, this is the point where we should jump to the TRT-LLM execution shape for
this stage in one deliberate rewrite rather than keep iterating on the old row
kernel family.

The target stage shape is:

1. exact routed CTA metadata
2. grouped `PermuteGemm1`
3. TRT-style activation/output-scale production
4. grouped `Gemm2`
5. finalize

What "jump in one shot" means here:

- replace the active routed-up and routed-down math kernels together
- keep the existing device-only routing and launch-plan foundation
- keep one runtime path only
- preserve the new packed FC1/FC2 contracts we just landed
- stop carrying forward the old row-kernel family as an equal implementation

What still should not be "one shot":

- do not rewrite routing, shared experts, and Mamba in the same patch
- do not merge the new grouped math core before the focused correctness gates
  and `prefix128 / tail4` TTFT gate pass on the new stage

The implementation plan for this jump is:

1. Replace the routed FC1 consumer with a true grouped GEMM tile kernel that
   consumes the current planned CTA metadata directly, rather than one CTA
   externalizing output-row tiles.
2. Match TRT-LLM's handoff exactly at the stage boundary:
   `gemm1_output`, `gemm1_output_scale`, activation, packed FC2 input,
   `gemm2_output`.
3. Replace routed FC2 with the same grouped kernel family and CTA map, rather
   than keeping a mixed FC1-new / FC2-old implementation.
4. Keep finalize unchanged initially so the only moving part is the routed math
   stage.
5. After the routed stage wins, move shared experts onto the same stronger math
   family.

The acceptance criteria for this jump are:

- `device_nvfp4_matrix_test`
- `moe_launch_plan_device_test`
- `fused_moe_prefill_test`
- `multi_turn_prefix_reuse_test`
- `cold_prefill_prefix128` must improve versus the current `126.390 ms`
- no regression in behavioral reuse equivalence

The reason this is safe now is that the contract work is no longer the blocker.
The remaining gap versus TRT-LLM is the grouped math and CTA scheduling itself,
and that gap will not close through more incremental cleanup.

Progress on this jump:

- The active packed FC1/FC2 routed path now executes from the planned CTA grid
  rather than the old exact-task queue.
- The launch plan now pads expert batches to `tileTokensDim=16`, which makes
  `cta_m_limits` sufficient to describe active token rows the TRT way.
- The active planned packed kernels now consume `cta_expert_ids + cta_m_limits`
  as their token-batch contract instead of a second task-local
  `row_start + valid_rows` map.
- The active routed stage now follows TRT-LLM's BF16 x NVFP4 contract more
  closely:
  - launch-plan aliases now expose
    `permuted_idx_to_token_idx`,
    `total_num_padded_tokens`,
    `num_non_exiting_ctas`,
    `cta_idx_xy_to_batch_idx`, and
    `cta_idx_xy_to_mn_limit`
  - routed FC1 no longer pre-packs BF16 activations into an NVFP4 input matrix
    on the active path
  - the active path now does `BF16 grouped Gemm1 -> gemm1_output_scale ->
    packed Gemm2 input`, which matches TRT's BF16 routed entry much better than
    the earlier packed-FC1 experiment
- Focused correctness remains green:
  `device_nvfp4_matrix_test`, `moe_launch_plan_device_test`,
  `fused_moe_prefill_test`, `multi_turn_prefix_reuse_test`.
- Focused TTFT moved only slightly:
  `cold_prefill_prefix128 = 126.390 ms`,
  `cached_committed_head_prefix128_tail4 hot-prefix = 54.336 ms`,
  `cached_global_root_prefix128_tail4 hot-prefix = 54.333 ms`.

Interpretation:

- The routed launch contract is now close enough to TRT that it is unlikely to
  be the dominant remaining blocker.
- The BF16 routed FC1 entry and BF16 `gemm1_output` seam are now structurally
  aligned with TRT's routed stage, so further FC1-input packing experiments
  should not remain on the active path.
- The routed dequant-scale contract and grouped batch/limit metadata are now
  active too, so the next remaining mismatch is the grouped math core itself.
- The first grouped-body cutover is now active too:
  - normalized activations are packed once into the static prefill source pack
  - routed FC1 gathers packed rows with `permuted_idx_to_token_idx` into the
    grouped FC1 input pack
  - the active FC1 consumer now runs from that packed routed input directly
    into BF16 `gemm1_output`
- Focused TTFT on the same gate after the packed-FC1-body cutover is:
  `cold_prefill_prefix128 = 125.623 ms`,
  `cached_committed_head_prefix128_tail4 hot-prefix = 54.361 ms`,
  `cached_global_root_prefix128_tail4 hot-prefix = 54.224 ms`.
- The next grouped-body step is now active too:
  - routed FC1/FC2 WMMA consumers use a TRT-like `tileTokensDim=16`,
    `transposeMmaOutput=true`, `epilogueTileM=128` execution shape
  - in our implementation that means `kPlannedOutputTile = 128`,
    `kPlannedThreadsPerBlock = 256`, and `8` warps per CTA on the routed
    grouped path
- Focused TTFT on the same gate after the larger grouped tile cutover is:
  `cold_prefill_prefix128 = 125.324 ms`,
  `cached_committed_head_prefix128_tail4 hot-prefix = 54.358 ms`,
  `cached_global_root_prefix128_tail4 hot-prefix = 54.597 ms`.
- The next remaining jump is the grouped math core itself: replace the current
  WMMA row-tile micro-fragment consumer with a fuller TRT-like grouped GEMM
  micro-fragment kernel for routed FC1 and FC2.
- TRT runtime trace update (`2026-04-07`):
  - the real local `TRT-LLM` NemotronH serve path is using the CUTLASS
    fused-MoE custom op selected through `trtllm::fused_moe::gemm1` /
    `trtllm::fused_moe::gemm2`, not the `trtllmGenFp8BlockScaleMoe` runner
  - observed real-path tactic IDs from
    `artifacts/benchmarks/trtllm_fused_moe_tactics_20260407.log`:
    - warmup max-context `(4607, 1344)`: `gemm1=1`, `gemm2=13`
    - decode-like `(1, 1344)`: `gemm1=0`, `gemm2=13`
    - `prefix4` `(4, 1344)`: `gemm1=5`, `gemm2=15`
    - `prefix128` `(128, 1344)`: `gemm1=4`, `gemm2=12`
    - `prefix4096` `(4096, 1344)`: `gemm1=1`, `gemm2=13`
  - packed routed shapes on the real path are:
    - `input_shape=(tokens, 1344)`
    - `fc1_shape=(128, 1920, 168)`
    - `fc2_shape=(128, 2688, 120)`
    - `top_k=6`
    - `act_dtype=torch.uint8`, `weight_dtype=torch.int64`,
      `output_dtype=torch.bfloat16`
  - implication:
    - the native target is not one universal TRT-like grouped kernel
    - it is a shape-aware routed grouped-kernel family over the packed
      low-precision contract
    - the immediate specialization buckets are now clear:
      `M=1`, `M=4`, `M=128`, and `M=4096`
  - actual CUTLASS tactic descriptors from the live `FusedMoeRunner` path:
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
- implication for the native rewrite:
  - `gemm2` is nearly stable across `128` and `4096`, but `prefix4` wants a
    larger `256x128x64` tile
  - `gemm1` changes both `tile_k` and `swap_ab` across regimes
  - the next native grouped-kernel step should therefore mirror TRT with a
    small family of shape-selected routed FC1/FC2 kernels, not one fixed
    grouped body
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
    - the native routed grouped-kernel selector should be implemented as a
      small explicit regime table, not a guessed single-threshold policy
    - `gemm1` and `gemm2` need independent regime selection
    - the first native family should target the stable islands we already know
      we care about operationally: `1`, `4..7`, `32..112`, `120..248`,
      `512..992`, `1024+`

### Sub-Plan: From-Scratch Routed Grouped GEMM Family

Yes. We should now return directly to the grouped-GEMM internal kernel rewrite.
The contract is good enough, the live TRT tactic family is traced, and the
remaining routed gap is now inside the math body rather than in metadata or
staging.

The goal of this sub-plan is not to copy CUTLASS or import TRT-LLM source
verbatim. The goal is to reproduce the TRT routed stage shape in our own
specialized runtime:

1. explicit routed grouped metadata
2. from-scratch grouped `PermuteGemm1` family
3. TRT-style `gemm1_output -> gemm1_output_scale -> activation_output_scale`
   handoff
4. from-scratch grouped `Gemm2` family
5. existing finalize path

#### Locked Inputs

- Keep the current active packed routed contract:
  - `permuted_idx_to_token_idx`
  - `total_num_padded_tokens`
  - `num_non_exiting_ctas`
  - `cta_idx_xy_to_batch_idx`
  - `cta_idx_xy_to_mn_limit`
  - packed FC1 input
  - BF16 `gemm1_output`
  - routed `gemm1_output_scale`
  - routed `activation_output_scale`
  - packed FC2 input
- Keep one runtime path only.
- Keep behavioral reuse equivalence as the correctness bar.
- Do not reintroduce cuBLASLt or a second generic expert path.

#### Implementation Order

1. Freeze the routed dispatcher around an explicit TRT-like regime table.
   - No heuristic threshold guesses in the first cut.
   - Dispatch `gemm1` and `gemm2` independently.
   - Encode the traced TRT regimes directly.

2. Implement the common grouped-kernel substrate once.
   - CTA work unit:
     one grouped GEMM tile identified by `cta_idx_xy_to_batch_idx` and
     `cta_idx_xy_to_mn_limit`.
   - Common per-CTA state:
     expert id, padded batch row range, packed-token base, output `N` tile,
     and `valid_m/valid_n` limits.
   - Common math policy:
     FP32 accumulate, BF16 output, packed low-precision operands.
   - Common variants:
     `tile_k=64`, `tile_k=128`, `swap_ab=false`, `swap_ab=true`.

3. Implement the design-center FC1 family first.
   - First kernel:
     TRT `gemm1=5`, `128x128x64`, `swap_ab=true`.
     This directly covers the current `prefix128` design center and several
     other stable islands.
   - Second kernel:
     TRT `gemm1=1`, `128x128x64`, `swap_ab=false`.
     This covers the `1024+` large-prefill regime and several smaller islands.
   - Third kernel:
     TRT `gemm1=4/0`, `128x128x128`, with both `swap_ab=true` and
     `swap_ab=false`.
     This covers the medium `32..112` and `120..248` islands.
   - Fourth kernel:
     TRT `gemm1=7`, `256x128x64`, `swap_ab=true`, for the `2..3` tiny regime.

4. Keep activation and scale production as a separate stage boundary, but make
   it TRT-shaped.
   - Consume BF16 `gemm1_output` in grouped padded order.
   - Produce `gemm1_output_scale`.
   - Apply `ReLU^2`.
   - Produce `activation_output_scale`.
   - Pack directly into grouped FC2 input.
   - Do not materialize an alternate long-lived FP32 routed seam.

5. Implement the design-center FC2 family second.
   - First kernel:
     TRT `gemm2=12`, `128x128x128`, `swap_ab=true`.
     This covers the current `prefix128` design center and the broader
     `128..248` and `512..992` islands.
   - Second kernel:
     TRT `gemm2=13`, `128x128x64`, `swap_ab=true`.
     This covers `32..127`, `256`, `384`, and `1024+`.
   - Third kernel:
     TRT `gemm2=15`, `256x128x64`, `swap_ab=true`, for `2..16`.

6. Wire the routed stage to the explicit regime table in priority order.
   - Priority A:
     `prefix128` path:
     FC1 `5`, FC2 `12`.
   - Priority B:
     `prefix4096` path:
     FC1 `1`, FC2 `13`.

#### Progress Update (`2026-04-07`)

- landed the first correctness-clean grouped-body rewrite in
  `runtime/src/backend/fused_moe_prefill.cu`
  - active routed FC1/FC2 kernels now dispatch through the traced TRT regime
    table
  - the grouped kernel body now matches TRT operand layout semantics instead of
    treating both operands as row-major:
    - `swap_ab=false`: activations as `A` row-major, weights as `B`
      column-major
    - `swap_ab=true`: weights as `A` row-major, activations as `B`
      column-major, with the transposed output scatter
  - focused validation stayed green:
    - `fused_moe_prefill_test`
    - `moe_launch_plan_device_test`
    - `multi_turn_prefix_reuse_test`
- artifacts:
  - `artifacts/benchmarks/ttft_20260407_trt_layout_grouped_body_prefix128_tail4.stdout.txt`
  - `artifacts/benchmarks/ttft_20260407_trt_layout_grouped_body_prefix4096_tail4.stdout.txt`
- measured outcome:
  - `cold_prefill_prefix128 = 126.168 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix = 54.304 ms`
  - `cold_prefill_prefix4096 = 2488.473 ms`
  - `cached_global_root_prefix4096_tail4 hot-prefix = 96.850 ms`
- conclusion:
  - matching TRT metadata plus operand-layout semantics is necessary, but not
    sufficient
  - the remaining routed gap is now inside the mainloop microarchitecture:
    per-element NVFP4 decode, shared-memory tile fill/staging, and the lack of
    a TRT-like TMA/block-scaled mainloop
  - the next rewrite should therefore target the profile-specific inner loop,
    not more contract churn
  - fallback for currently untraced islands remains the legacy WMMA
    row-tile routed kernels in `runtime/src/backend/fused_moe_prefill.cu`
- grouped-body blockwise decode cutover (`2026-04-07`):
  - active file:
    `runtime/src/backend/fused_moe_prefill.cu`
  - change:
    the active traced grouped FC1/FC2 kernels now decode and stage full
    `16`-value NVFP4 blocks into shared memory instead of decoding one scalar
    element at a time
  - focused validation stayed green:
    - `fused_moe_prefill_test`
    - `moe_launch_plan_device_test`
    - `multi_turn_prefix_reuse_test`
  - artifacts:
    - `artifacts/benchmarks/ttft_20260407_blockwise_grouped_body_prefix128_tail4.stdout.txt`
    - `artifacts/benchmarks/ttft_20260407_blockwise_grouped_body_prefix4096_tail4.stdout.txt`
  - measured outcome:
    - `cold_prefill_prefix128 = 125.578 ms`
    - `cached_global_root_prefix128_tail4 hot-prefix = 54.328 ms`
    - `cold_prefill_prefix4096 = 2479.858 ms`
    - `cached_global_root_prefix4096_tail4 hot-prefix = 96.994 ms`
  - conclusion:
    - the grouped mainloop was still paying obvious scalar decode/staging cost
    - blockwise decode is worth keeping on the active path because it improves
      cold prefill on both the design-center and long-prefix regimes while
      keeping reuse correctness green
    - the remaining routed gap is now deeper in the grouped MMA mainloop than
      simple nibble/scale decode overhead

- exact SM120 atom debug pass (`2026-04-07`):
  - we recovered and checked the low-level atom mappings directly against the
    CUTE traits used by TRT
  - confirmed:
    - `A/SFA/SFB` mappings match the custom implementation
    - `B` and `C` needed corrections:
      - `BLayout`: low/high `n` rows are driven by `lane & 3` and
        `lane & 3 + 4`, while the `k` phase is `lane >> 2`
      - `CLayout`: `col = lane & 7`,
        `row_group = (lane >> 3) * 4`, row order `{0, 2, 1, 3}`
  - important result:
    those fixes were necessary but still not sufficient to make the traced FP4
    activation-side regimes behaviorally reuse-equivalent
  - current active safety policy:
    - keep `P5`, `P7`, `P13`, and `P15` on the known-good grouped BF16 fallback
    - only keep traced FP4 consumers active where
      `fused_moe_prefill_test` and `multi_turn_prefix_reuse_test` both stay
      green
  - next exact step:
    stop hand-maintaining the FP4 activation-side fragment transport and
    replace it with an exact CUTE-driven fragment path using
    `MMA_Atom` / `thrfrg_A` / `thrfrg_B` / `thrfrg_C` or the equivalent
    `make_tiled_copy_A/B` substrate before re-enabling those traced FP4
    profiles

- SM120 FP4 shift pass and real rebuild result (`2026-04-07`):
  - from the local CUTLASS/CUTE SM120 blockscaled mainloop, we recovered one
    additional required detail:
    the consumer path explicitly applies `fp4_shift_A/B` before the blockscaled
    FP4 MMA
  - we copied that behavior into the custom k64 FP4 grouped kernels
  - important result after a real rebuild:
    - `fp4_shift_A/B` is necessary
    - but it is still not sufficient to make the traced k64 activation-side
      regimes (`P5`, `P7`, `P13`, `P15`) behaviorally reuse-equivalent
  - after confirming that on fresh binaries, we restored the documented safety
    policy:
    - keep `P5`, `P7`, `P13`, and `P15` on the known-good grouped BF16 fallback
    - only keep the traced FP4 consumer active where both
      `fused_moe_prefill_test` and `multi_turn_prefix_reuse_test` stay green
  - implication:
    - the next implementation step is not another hand-written register fix
    - it is the full CUTE-driven fragment/copy path for A/B/SFA/SFB, using the
      local CUTLASS/CUTE source as ground truth
    - no additional TRT tactic tracing is required before that code step

- CUTLASS/CUTE dependency note (`2026-04-07`):
  - we do not need more TRT traces before the next rewrite
  - we also do not need the full CUTLASS runtime stack
  - but the next exact mainloop step does need the header-level CUTE/CUTLASS
    substrate
  - to make that integration more faithful and avoid writing unnecessary local
    compatibility wrappers, the runtime build is now moving from C++17/CUDA17
    to C++20/CUDA20 before the next fragment/copy rewrite
  - the local machine already has a usable header tree at:
    `.venv-trtllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include`
  - for long-term build stability, we should pin or vendor the exact header
    snapshot once the CUTE-driven mainloop path is the active implementation

- C++20 probe result and bridge decision (`2026-04-07`):
  - the runtime now builds as `C++20` / `CUDA20`
  - the narrow SM120 MMA-op import from local CUTE works and the branch stays
    green
  - but the broader upstream fragment/copy header stack is still not a clean
    drop-in even under C++20:
    - `cute/algorithm/copy.hpp`
    - `cute/algorithm/prefetch.hpp`
    - `cutlass/cuda_host_adapter.hpp`
  - so the next exact step is:
    - keep using the upstream SM120 blockscaled FP4 MMA op as ground truth
    - build a very narrow local bridge for the exact fragment/copy pieces we
      need
    - avoid importing the broader CUTLASS host/runtime surface into this TU

- Narrow local bridge checkpoint (`2026-04-07`):
  - the active traced `k64` FP4 routed kernels now use a local `nvfp4_bridge`
    fragment layer instead of open-coded raw register tuples
  - the bridge currently wraps only the exact pieces we need:
    - `A/B` fragment load
    - `C` fragment clear
    - grouped MMA call
    - `C` fragment store
  - the bridge still delegates the math op itself to the upstream SM120
    blockscaled FP4 MMA definition, so the math ground truth remains the traced
    TRT/CUTE path
  - this keeps both correctness gates green:
    - `fused_moe_prefill_test`
    - `multi_turn_prefix_reuse_test`
  - next bridge expansion:
    - move more of the traced fragment/copy layout logic behind this local
      bridge
    - keep the include surface narrow
    - avoid importing `cute/algorithm/copy.hpp` / host adapter machinery into
      the runtime TU

- Bridge expansion: packed tile copy/layout (`2026-04-07`):
  - the active traced `k64` routed kernels now also route their packed
    row/scale tile movement through `nvfp4_bridge`
  - current local bridge surface now covers:
    - packed activation row copy
    - packed weight row copy
    - zero-row handling
    - fragment load
    - fragment clear
    - grouped MMA
    - fragment store
  - that means the active grouped kernels no longer call the raw packed
    row/scale copy helpers directly
  - correctness stayed green after this widening:
    - `fused_moe_prefill_test`
    - `multi_turn_prefix_reuse_test`
  - immediate next bridge target:
    - pull the remaining lane-to-row/col projection math and profile-specific
      tile mapping behind the same local bridge

- Bridge expansion: transpose store projection (`2026-04-07`):
  - the swap-true traced `k64` routed kernel now also routes its final
    lane-to-output projection through `nvfp4_bridge`
  - with this step, the active traced `k64` grouped kernels use the local
    bridge for:
    - packed tile copy/layout
    - fragment load/clear
    - grouped MMA
    - row-major store
    - transpose-style store
  - correctness remained green:
    - `fused_moe_prefill_test`
    - `multi_turn_prefix_reuse_test`

- Exact CUTE TV-layout bridge step (`2026-04-07`):
  - the local bridge now derives `A/B/C` thread-value coordinates and `SFA/SFB`
    row selection from `MMA_Atom` / `MMA_Traits` TV layouts instead of relying
    only on the earlier hand-derived lane formulas
  - this keeps the safe BF16-dispatch state green:
    - `fused_moe_prefill_test`
    - `multi_turn_prefix_reuse_test`
  - retry result for activating `P5` on the FP4 bridge:
    - still not behaviorally reuse-equivalent
    - first retry failed at prompt-boundary argmax matching
    - second retry with the exact TV-layout bridge failed even earlier in
      full-model execution (`single_token_forward_model: layer 1 kind=2
      execution failed`)
  - implication:
    - the remaining gap is deeper than the TV-layout mapping alone
    - the next exact source-of-truth layer is the smem-to-register retile/copy
      path used by TRT
  - `copy_atom.hpp` probe result:
    - including `cute/atom/copy_atom.hpp` and touching `make_tiled_copy_A/B`
      still drags the same broader problematic header surface into this TU
    - so the next move should be to port the exact retile logic we need from
      `copy_atom.hpp` locally, not include that header directly
  - local scale-copy bridge step:
    - a narrow local `UniversalCopy`-based retile path for `SFA/SFB` now
      compiles and is wired into the FP4 fragment loaders
    - retrying traced `P5` on top of the full local `A/B/SFA/SFB` bridge still
      fails behavioral reuse at the global-root restored prompt boundary
      (`cold=1010`, `restored=1584`, `max_abs_diff=13.8164`)
    - correcting the operand copy atoms to the exact SM120 blockscaled builder
      ground truth (`SM75_U32x4_LDSM_N` for both `A` and `B`) leaves the same
      `P5` reuse failure unchanged
    - implication: the remaining TRT mismatch is now deeper than coordinate
      retile or copy-atom choice alone, likely in the shared-memory swizzle /
      layout and stage-to-stage materialization details of the FP4 mainloop
  - traced `P5` full-tile probe update:
    - the real traced `P5` tiled MMA is `128x32x64`
    - the `B` copy-view row coordinates span `0..31`
    - our experimental `P5` FP4 kernel was still staging only `16` logical `B`
      rows and driving `N` with manual `n_base` subtiles
    - switching that experimental kernel to a staged `N=32` tile removes the
      earlier local-buffer mismatch, but `compute-sanitizer` still reports
      `Invalid __shared__ read of size 1 bytes`
    - implication: the next blocker is now more specific than generic swizzle
      mismatch; the active `BFragment64` / `CFragment64` decomposition does not
      faithfully represent TRT's full traced `P5` fragment contract
    - branch policy stays the same for now:
      keep `P5` on the known-good grouped BF16 fallback until the full traced
      `P5` fragment shape replaces the manual `n_base` path
  - traced `P5` fragment-rank rewrite update:
    - local CUTE probes now pin down the traced `P5` per-thread partitions:
      - `partition_A rank=(32,4,1)`
      - `partition_B rank=(16,2,1)`
      - `partition_C rank=(4,4,2)`
    - the dormant experimental `P5` FP4 kernel has now been rewritten around
      the traced full-tile fragment model:
      - `A[4]`
      - `B[2]`
      - `C[4][2]`
    - the old manual `n_base` / `token_subtile` decomposition is removed from
      that experimental kernel body
    - the safe branch state remains unchanged because `P5` dispatch still stays
      on the known-good grouped BF16 fallback:
      - `fused_moe_prefill_test`
      - `multi_turn_prefix_reuse_test`
    - implication:
      the next `P5` blocker is no longer fragment rank discovery; it is the
      exact traced scale/layout/materialization semantics that still need to be
      validated before re-enabling `P5`
  - traced CUTLASS collective source correction:
    - the earlier reference to
      `sm120_mma_array_tma_blockwise_scaling.hpp` was the wrong ground-truth
      collective for the live TRT routed path
    - the relevant source is
      `sm120_blockscaled_mma_array_tma.hpp`, which matches the traced
      block-scaled array collective used by the live SM120 fused-MoE kernels
    - that collective:
      - loads `A/B/SFA/SFB` into shared memory
      - partitions `A/B` into operand fragments with `partition_fragment_A/B`
      - partitions scales into fragment-shaped tensors with
        `partition_fragment_SFA/SFB`
      - copies operand and scale fragments into registers
      - calls
        `cute::gemm(tiled_mma, make_zip_tensor(tCrA, tCrSFA), make_zip_tensor(tCrB, tCrSFB), accum)`
    - implication:
      our dormant experimental `P5` kernel should be aligned to the zipped
      `(operand, scale)` fragment contract, not to the generic
      `tmp_accum + post-rescale` model
  - definitive host-side `P5` scale-layout probe:
    - a standalone builder probe at
      [trt_p5_runtime_layout_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p5_runtime_layout_dump.cu)
      now runs without the TRT launcher/runtime path and prints the exact traced
      scale-smem facts
    - exact results:
      - `Stages = 4`
      - `cosize(SmemLayoutSFA) = cosize(SmemLayoutSFB) = 4096`
      - `size(tCrSFA) = 256`, `cosize(tCrSFA) = 16`
      - `size(tCrSFB) = 1024`, `cosize(tCrSFB) = 64`
      - `tCsSFA.layout = ((_1,((_16,_4),_2)),_1,_2,_4):((_0,((_0,_1),_8)),_0,_512,_1024)`
      - `tCrSFA_copy_view.layout = ((_1,(_16,_8)),_1,_2):((_0,(_0,_1)),_0,_8)`
      - `tCsSFB.layout = ((_1,((_16,_4),_2)),_4,_2,_4):((_0,((_0,_1),_128)),_4,_512,_1024)`
      - `tCrSFB_copy_view.layout = ((_1,(_16,_4,_2)),_4,_2):((_0,(_0,_1,_16)),_4,_32)`
    - implication:
      the traced `P5` scale path is definitively not compatible with the old
      `scale_words[row] + scale[1]` abstraction
    - result after the exact scale-smem rewrite:
      - the dormant traced `P5` kernel now stages scales through exact
        `SmemLayoutSFA/SFB`-backed shared tensors instead of row-packed scale
        words
      - `P5` is re-enabled on the active routed FP4 path
      - validation:
        - `fused_moe_prefill_test: PASS`
        - `multi_turn_prefix_reuse_test: PASS`
        - `compute-sanitizer --tool memcheck ./testing/fused_moe_prefill_test`
          reports `0 errors`
      - focused TTFT:
        - artifact:
          [ttft_20260407_p5_exact_scale_smem_prefix128_tail4.stdout.txt](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/ttft_20260407_p5_exact_scale_smem_prefix128_tail4.stdout.txt)
        - `cold_prefill_prefix128 = 107.552 ms`
        - `cached_committed_head_prefix128_tail4 hot-prefix = 48.483 ms`
        - `cached_global_root_prefix128_tail4 hot-prefix = 48.650 ms`
      - conclusion:
        the exact traced `SFA/SFB` smem path was the real missing piece for
        `P5`
    - next rewrite:
      use the now-working `P5` scale-smem bridge as the template for the next
      traced FP4 regimes, starting with `P12` / `P13`
  - `P13` source-first bridge rollout:
    - reused the now-working traced `P5` `128x128x64 swap_ab=true`
      `SFA/SFB` bridge pattern for FC2 `P13`
    - only `P13` was re-routed to the new exact-scale-smem FP4 kernel; `P12`
      and `P15` stayed unchanged
    - validation:
      - `fused_moe_prefill_test: PASS`
      - `multi_turn_prefix_reuse_test: PASS`
      - `compute-sanitizer --tool memcheck ./testing/fused_moe_prefill_test`
        reports `0 errors`
    - focused TTFT artifact:
      [ttft_20260407_p13_exact_scale_smem_prefix4096_tail4.stdout.txt](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/ttft_20260407_p13_exact_scale_smem_prefix4096_tail4.stdout.txt)
    - focused TTFT:
      - `cold_prefill_prefix4096 = 1633.296 ms`
      - `cached_committed_head_prefix4096_tail4 hot-prefix = 95.505 ms`
      - `cached_global_root_prefix4096_tail4 hot-prefix = 95.450 ms`
    - conclusion:
      the traced `P5` scale-smem bridge pattern carries over cleanly to `P13`
      without needing a separate TRT probe
    - next target:
      `P12` exact `128x128x128` FC2 path for the `prefix128 / tail4`
      design-center regime
  - `P12` source-first bridge rollout:
    - first source-based attempt reused the exact traced `P5/P13`
      `SFA/SFB` bridge pattern as two `k64` subtiles per `128`-wide macro tile
    - no extra TRT/CUTLASS probe was needed because the first real activation
      cleared the correctness gates
    - validation:
      - `fused_moe_prefill_test: PASS`
      - `multi_turn_prefix_reuse_test: PASS`
      - `compute-sanitizer --tool memcheck ./testing/fused_moe_prefill_test`
        reports `0 errors`
    - focused TTFT artifact:
      [ttft_20260407_p12_exact_scale_smem_prefix128_tail4.stdout.txt](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/ttft_20260407_p12_exact_scale_smem_prefix128_tail4.stdout.txt)
    - focused TTFT:
      - `cold_prefill_prefix128 = 107.215 ms`
      - `cached_committed_head_prefix128_tail4 hot-prefix = 48.424 ms`
      - `cached_global_root_prefix128_tail4 hot-prefix = 48.460 ms`
    - conclusion:
      the exact traced scale-smem bridge pattern generalizes to `P12` without a
      new probe
    - next target:
      `P15`, then a broader rerun/profile pass on the routed FP4 family
  - `P15` first exact-source rollout and definitive probe:
    - first source-based attempt:
      - reused the `P13` exact-scale-smem kernel shape for `P15`
      - failed the first real gate immediately:
        - `fused_moe_prefill_test: FAIL`
        - `nano_prefill_output_max_abs_diff = 15.4386`
        - `nano_prefill_routed_max_abs_diff = 15.4386`
        - `nano_prefill_worst_routed_index = 2692 actual=0 expected=15.4386`
      - implication:
        per the strategy rule, `P15` needed a definitive probe instead of more
        shape reuse
    - definitive `P15` builder-only probe:
      [trt_p15_runtime_layout_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_runtime_layout_dump.cu)
    - recorded output:
      [trt_p15_runtime_layout_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_runtime_layout_dump_20260407.log)
    - exact recovered facts:
      - `Stages = 6`
      - `size(TiledMma) = 256`
      - `tile_mnk = (128, 32, 64)`
      - `size(partA) = 64`, `cosize(partA) = 2920`
      - `size(partB) = 32`, `cosize(partB) = 360`
      - `size(partC) = 16`, `cosize(partC) = 730`
      - `cosize(SmemLayoutSFA) = 6144`
      - `cosize(SmemLayoutSFB) = 3072`
      - `size(tCrSFA) = 256`, `cosize(tCrSFA) = 16`
      - `size(tCrSFB) = 512`, `cosize(tCrSFB) = 32`
      - `partB.coords` and `partC.coords` are now dumped explicitly in the log
    - second source-based attempt:
      - rewrote the dormant `P15` kernel to use the exact traced
        `TiledMma + SmemLayoutSFA/SFB` contract
      - that still failed the first real gate with the same routed diff
    - conclusion:
      the remaining `P15` blocker is no longer scale staging; it is the full
      operand/store fragment contract. Reusing the simplified
      `AFragment64/BFragment64/CFragment64` model is not faithful enough for
      traced `P15`.
    - next target:
      implement a dedicated `P15` fragment family from the probed
      `partA/partB/partC` contract, then re-enable only `P15`
    - third source-based `P15` attempt:
      - switched the dormant `P15` path to the traced operand order
        `A=weights`, `B=activations`
      - replaced the single-fragment reuse with a dedicated `2x2`
        `A/B/C` family and two `128`-row subtiles for the `256` output tile
      - failed the first real gates:
        - `fused_moe_prefill_test: FAIL`
          - `nano_prefill_output_max_abs_diff = 15.5936`
          - `nano_prefill_routed_max_abs_diff = 15.5936`
        - `multi_turn_prefix_reuse_test: FAIL`
          - committed-head boundary argmax mismatch
      - per the strategy rule, restored `P15` to the safe BF16 grouped fallback
        before continuing
    - definitive `P15` operand copy-view probe:
      [trt_p15_copy_view_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_copy_view_dump.cu)
    - recorded output:
      [trt_p15_copy_view_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_copy_view_dump_20260407.log)
    - useful exact facts:
      - `copy_view_a` is contiguous `64`-element thread-fragment storage with the
        first `32` coords at rows `0/8` and the second `32` coords at rows
        `64/72`
      - `copy_view_b` is contiguous `32`-element thread-fragment storage with the
        first `16` coords at row `0` and the second `16` coords at row `8`
      - implication:
        the remaining `P15` blocker is less likely to be the coarse `A/B`
        fragment chunk split and more likely to be either:
        - the exact `SFA/SFB` scale-byte grouping for those fragments, or
        - the final `C`/store-side contract
    - next target:
      add the smallest definitive `P15` probe for the scale-copy or store-side
      contract, not another operand-chunking rewrite
    - `P15` scale-copy flattening test and accumulator-order probe:
      - source-based test:
        - updated `FillPhysicalCoordMapCopyViewLimited` to flatten rank-4
          copy-view tensors and retried `P15`
        - exact-source `P15` still failed unchanged:
          - `fused_moe_prefill_test: FAIL`
            - `nano_prefill_output_max_abs_diff = 15.5936`
            - `nano_prefill_routed_max_abs_diff = 15.5936`
          - `multi_turn_prefix_reuse_test: FAIL`
            - committed-head boundary argmax mismatch
        - conclusion:
          the remaining blocker is not just rank-4 scale-copy flattening
      - definitive `P15` accumulator/store probe:
        [trt_p15_accum_order_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_accum_order_dump.cu)
      - recorded output:
        [trt_p15_accum_order_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_accum_order_dump_20260407.log)
      - exact recovered facts:
        - `tCcC.layout = ((_2,_2),_2,_2):((_1@1,_8@0),_64@0,_8@1)`
        - `tCrC.layout = ((_2,_2),_2,_2):((_1,_2),_4,_8)`
        - flat accumulator destination order is:
          - `0..3 -> (0,0) (0,1) (8,0) (8,1)`
          - `4..7 -> (64,0) (64,1) (72,0) (72,1)`
          - `8..11 -> (0,8) (0,9) (8,8) (8,9)`
          - `12..15 -> (64,8) (64,9) (72,8) (72,9)`
      - source-based follow-up:
        - rewired `StoreTracedP15CFragmentsRowMajor` to match that traced flat
          order and retried `P15`
        - both real gates still failed with the same routed diff
      - conclusion:
        the remaining `P15` blocker is deeper than the final store permutation;
        it is the full accumulator fragment contract, not just scale-copy
        flattening or the last writeback order
      - safe state restored:
        `P15` is back on the known-good grouped BF16 fallback and both
        correctness gates pass again
    - definitive `P15` C-shaped scale-view probe and CUTLASS-style rescale test:
      - probe source:
        [trt_p15_scale_as_c_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_scale_as_c_dump.cu)
      - recorded output:
        [trt_p15_scale_as_c_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_scale_as_c_dump_20260407.log)
      - exact recovered facts:
        - `Stages = 6`
        - `ScaleGranularityM = 128`
        - `ScaleGranularityN = 128`
        - `ScaleMsPerTile = 2`
        - `ScaleNsPerTile = 1`
        - `tCsScaleAViewAsC.layout = ((_2,_2),(_2,_2),_8,_6):((_0,_0),(_0,_1),_0,_2)`
        - `tCsScaleBViewAsC.layout = ((_2,_2),_4,_8,_6):((_0,_0),_0,_0,_1)`
        - `tCrScaleAViewAsC.layout = ((_2,_2),(_2,_2),_8):((_0,_0),(_0,_1),_0)`
        - `tCrScaleBViewAsC.layout = ((_2,_2),_4,_8):((_0,_0),_0,_0)`
      - source-based follow-up:
        - retried `P15` with neutral instruction scales plus a CUTLASS-style
          `C`-view rescale fold using the probed `ScaleMsPerTile = 2`,
          `ScaleNsPerTile = 1` contract
        - `fused_moe_prefill_test` still failed:
          - `nano_prefill_output_max_abs_diff = 15.5936`
          - `nano_prefill_routed_max_abs_diff = 15.5936`
        - `multi_turn_prefix_reuse_test` still failed:
          - committed-head boundary argmax mismatch
          - `max_abs_diff = 12.0938`
      - conclusion:
        even the first CUTLASS-style `C`-view scale fold is not sufficient for
        traced `P15`; the remaining blocker is now more likely the full
        `tCrA/tCrB -> tmp_accum -> tCrC` contraction contract than scale
        staging alone
      - safe state restored:
        `P15` stays on the known-good grouped BF16 fallback and both
        correctness gates pass again
    - definitive `P15` fragment-contract probe:
      - probe source:
        [trt_p15_fragment_contract_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_fragment_contract_dump.cu)
      - recorded output:
        [trt_p15_fragment_contract_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_fragment_contract_dump_20260407.log)
      - exact recovered facts for thread `0`:
        - `tCrA.layout = ((_8,_2,_2),_4,_1):((_1,_8,_16),_32,_0)`
        - `tCrB.layout = ((_8,_2),(_2,_4),_1):((_1,_8),(_16,_32),_0)`
        - `tCrC.layout = ((_2,_2),_2,_2):((_1,_2),_4,_8)`
        - recast register views are all single `16`-element tensors:
          - `rA.layout = ((_1,_2,_2),_4,_1):((_1,_1,_2),_4,_0)`, `size = 16`
          - `rB.layout = ((_1,_2),(_2,_4),_1):((_1,_1),(_2,_4),_0)`, `size = 16`
          - `rC.layout = ((_2,_2),_2,_2):((_1,_2),_4,_8)`, `size = 16`
      - implication:
        the traced `P15` thread fragment is not naturally modeled as our current
        manual `AFragment64[2] / BFragment64[2] / CFragment64[2][2]` family.
        The next rewrite should replace that hand-rolled `P15` fragment family
        with a single CUTE-aligned thread-fragment model and only then retry
        the active `P15` path.
    - definitive `P15` copy-view to register probe:
      - probe source:
        [trt_p15_copy_to_reg_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_copy_to_reg_dump.cu)
      - recorded output:
        [trt_p15_copy_to_reg_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_copy_to_reg_dump_20260407.log)
      - exact recovered facts:
        - `tCrA_copy_view.layout = ((_32,_2),_2,_1):((_1,_32),_64,_0)`, `size = 128`
        - `tCrB_copy_view.layout = ((_32,_1),_4,_1):((_1,_0),_32,_0)`, `size = 128`
        - `tCrSFA_copy_view.layout = ((_1,(_16,_8)),_2,_1):((_0,(_0,_1)),_8,_0)`, `size = 256`
        - `tCrSFB_copy_view.layout = ((_1,(_16,_4,_2)),_4,_1):((_0,(_0,_1,_16)),_4,_0)`, `size = 512`
        - recast copy-view register layouts:
          - `rA_from_copy_view.layout = ((_4,_2),_2,_1):((_1,_4),_8,_0)`, `size = 16`
          - `rB_from_copy_view.layout = ((_4,_1),_4,_1):((_1,_0),_4,_0)`, `size = 16`
          - `rSFA_from_copy_view.layout = ((_1,(_16,_2)),_2,_1):((_0,(_0,_1)),_2,_0)`, `size = 64`
          - `rSFB_from_copy_view.layout = ((_1,(_16,_1,_2)),_4,_1):((_0,(_0,_1,_4)),_1,_0)`, `size = 128`
      - implication:
        the traced `P15` live path already has a precise CUTE retile contract
        from copy views into register fragments. The next native rewrite should
        stop hand-packing `P15` `A/B/SFA/SFB` fragments and instead build the
        dormant path around real `tCrA/tCrB/tCrSFA/tCrSFB/tCrC` tensors plus
        `cute::gemm(...)`.
    - `P15` exact `gemm` / accumulator contract follow-up:
      - safe state restored:
        - active `P15` dispatch is back on the known-good grouped BF16 fallback
        - `fused_moe_prefill_test`: `PASS`
        - `multi_turn_prefix_reuse_test`: `PASS`
      - definitive source-backed fact from
        [sm120_mma_array_tma_blockwise_scaling.hpp]( /home/khkramer/.cache/uv/archive-v0/f24U_Ixv0hjsqw0Ylm8si/tensorrt_llm/deep_gemm/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp ):
        - the live traced mainloop issues
          `cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tmp_accum)`
          after operand copy/shift and then rescales `tmp_accum`
      - first definitive probe:
        [trt_p15_gemm_contract_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_gemm_contract_dump.cu)
      - result:
        compiling that exact 4-arg `gemm` call against a naive
        `thread_mma.make_fragment_C(partition_C(dense_sC))` accumulator still
        fails the CUTE static assertions. That proves the missing piece is not
        the 4-arg call syntax by itself; it is the exact `FrgTensorC`
        accumulator contract passed to the live kernel.
      - second definitive probe:
        [trt_p15_accumulator_contract_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_accumulator_contract_dump.cu)
      - result:
        the builder-level `CollectiveMainloop` type does not expose the kernel
        helpers we need:
        - no `AccumulatorPipelineStageCount`
        - no `IsOverlappingAccum`
        - no `partition_accumulator_shape()`
        - no `slice_accumulator()`
        Those live one layer up, in the kernel wrapper.
      - implication:
        the next exact probe must target the `cutlass::gemm::kernel`
        specialization used by the traced `P15` path, not just the builder
        `CollectiveMainloop`. The remaining blocker is now explicitly the
        kernel-layer accumulator contract.
      - third definitive builder probe:
        [trt_p15_blk_shape_accum_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_blk_shape_accum_dump.cu)
      - result:
        the traced `P15` accumulator contract is not built over the
        `TiledMma` atom tile shape. The recovered shapes are:
        - `tile_size_mnk = (128, 32, 64)`
        - `tCrA(_,_,0).layout = ((_8,_2,_2),_4):((_1,_8,_16),_32)`
        - `tCrB(_,_,0).layout = ((_8,_2),(_2,_4)):((_1,_8),(_16,_32))`
        - `partition_fragment_C(tiled_mma, (128,32)).layout = ((_2,_2),_2,_2)`
        - `partition_fragment_C(tiled_mma, (256,32)).layout = ((_2,_2),_4,_2)`
        - `partition_fragment_C(tiled_mma, (256,128)).layout = ((_2,_2),_4,(_2,_4))`
      - implication:
        the full traced `P15` profile shape is `256x128x64`, so the dormant
        `P15` exact path cannot keep using the old `CFragment64[2][2]`
        accumulator/storage model. The next rewrite must allocate and store a
        full `partition_fragment_C(tiled_mma, (256,128))` accumulator contract
        and then rebuild the `P15` store path around that shape before another
        activation attempt.
      - first live activation after wiring the dormant exact kernel:
        - switching the live `P15` launch from the BF16 grouped fallback to the
          dormant exact kernel exposed a compile-time blocker immediately
        - the dormant exact path is not yet build-clean when instantiated
        - narrowing the inner `cute::gemm(...)` call from `tCrA/tCrB` to the
          traced `rA_from_copy_view/rB_from_copy_view` register views changed
          the failure mode, but did not make the exact path compile
      - implication:
        the remaining `P15` blocker is not just runtime correctness. The
        caller-provided full-profile accumulator contract still is not being
        modeled correctly enough to instantiate the exact local-CUTE path.
      - fourth definitive builder probe:
        [trt_p15_accum_slices_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_accum_slices_dump.cu)
      - result:
        the full traced `partition_fragment_C(tiled_mma, (256,128))` contract
        decomposes into eight atom-sized accumulator slices:
        - `size<0>(tCrC_profile)=4`
        - `size<1>(tCrC_profile)=4`
        - `size<2>(tCrC_profile)=8`
        - each `tCrC_slice.layout = ((_2,_2),_4):((_1,_2),_4)` with `size=16`
      - implication:
        the old mental model was still too small. Traced `P15` is not “one
        bigger `CFragment64`,” and not even the old `2x2` atom family. It is
        eight separate 16-value accumulator slices per thread. The next native
        rewrite has to rebuild `P15` around that eight-slice accumulator/store
        contract before another activation attempt.
      - exact `P15` manual-path activation attempt:
        - temporarily forced the dormant exact `P15` kernel onto its manual MMA
          path by disabling the compile-fragile local-CUTE `cute::gemm(...)`
          branch inside the kernel body
        - switched the live `P15` dispatcher to the exact kernel
        - result: this was the first exact `P15` path that was actually
          build-clean when instantiated
      - runtime gate result:
        - [fused_moe_prefill_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/fused_moe_prefill_test)
          failed with routed-only drift:
          - `nano_prefill_output_max_abs_diff = 15.5936`
          - `nano_prefill_routed_max_abs_diff = 15.5936`
          - `nano_prefill_shared_max_abs_diff = 0`
        - [multi_turn_prefix_reuse_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/multi_turn_prefix_reuse_test)
          failed at the first committed-head reuse gate:
          - `cold=4670`
          - `restored=27641`
          - `max_abs_diff=12.0938`
          - first layer divergence: `layer=1 kind=2 max_abs_diff=0.0950928`
      - implication:
        the remaining `P15` blocker has moved from compile-time instantiation
        to runtime numerics. The exact manual path is now buildable, but its
        accumulator / rescale / store contract is still wrong. The next `P15`
        rewrite should target that runtime contract directly, not the old
        `cute::gemm` instantiation problem.
      - definitive runtime memory probe:
        - ran `compute-sanitizer --tool memcheck` against the build-clean exact
          manual `P15` path
        - result: the failure is not only numeric drift; the kernel performs an
          out-of-bounds global read:
          - `Invalid __global__ read of size 1 bytes`
          - in `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmemP15<float>`
          - thread `(15,0,0)`, block `(0,1,0)`
          - `1` byte past a `256`-byte allocation
      - implication:
        the next `P15` step should first fix that operand/layout bounds bug.
        There is no point tuning accumulator math further until the exact
        manual path is memory-safe.
      - definitive true operand-copy probe:
        - added and ran
          [trt_p15_true_dense_operand_coords_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_true_dense_operand_coords_dump.cu)
        - artifact:
          [trt_p15_true_dense_operand_coords_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_true_dense_operand_coords_dump_20260407.log)
        - exact facts:
          - `copy_view_a_dense.size = 128`
          - `copy_view_b_dense.size = 128`
          - traced `A` copy rows span `0/64/128/192`, not one `128`-row subtile
          - traced `B` copy rows span `0/32/64/96`, not one `32`-row atom only
      - implication:
        the remaining `P15` gap is now even narrower: the active exact manual
        path is still built around the older `2x2` subtile mental model,
        while the traced operand contract is full-profile `256x128x64`. The
        next `P15` rewrite should stop decomposing the kernel as two
        independent `128`-row subtiles and instead rebuild the operand loads
        around the full traced profile.
      - definitive true store/accumulator profile probe:
        - added and ran
          [trt_p15_true_dense_store_coords_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_true_dense_store_coords_dump.cu)
        - artifact:
          [trt_p15_true_dense_store_coords_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_true_dense_store_coords_dump_20260407.log)
        - exact facts:
          - `part_c_dense.layout = ((_2,_2),_4,(_2,_4))`
          - `accum_profile.layout = ((_2,_2),_4,(_2,_4))`
          - traced `C` flat coords span the same `0/64/128/192` row bands and
            `0/32/64/96` column bands as the full-profile operand probes
      - implication:
        the next `P15` rewrite no longer needs another store-order guess. The
        traced full-profile output contract is now explicit, so the remaining
        work is to replace the dormant `2x2` subtile accumulator/store model
        with the full traced `4x4x8` profile and then re-enable `P15`.
      - definitive full-profile fragment-shape probe:
        - added and ran
          [trt_p15_full_profile_fragment_dump.cu](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_full_profile_fragment_dump.cu)
        - artifact:
          [trt_p15_full_profile_fragment_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_full_profile_fragment_dump_20260407.log)
        - exact facts:
          - `tCrA.layout = ((_8,_2,_2),_4,_1)` with `size<0>=32`, `size<1>=4`, `size<2>=1`
          - `tCrB.layout = ((_8,_2),(_2,_4),_1)` with `size<0>=16`, `size<1>=8`, `size<2>=1`
          - `tCrC.layout = ((_2,_2),_4,(_2,_4))` with `size<0>=4`, `size<1>=4`, `size<2>=8`
      - exact local-CUTE activation result:
        - re-enabled the dormant full-profile `P15` local-CUTE branch once
        - compile failed in `cute::gemm(...)` with the expected shape mismatch:
          - `size<1>(A) == size<1>(C)` failed
          - `size<1>(B) == size<2>(C)` failed
      - implication:
        this is now the precise next rewrite target. The dormant `P15`
        local-CUTE path must stop using the current hand-stitched
        `rA_from_copy_view / rB_from_copy_view` tensors and instead build the
        exact full-profile `tCrA / tCrB / tCrC` fragment contract that those
        static assertions require.
      - source-backed dormant-kernel correction:
        - re-read the CUTLASS `mma()` body in
          [sm120_mma_array_tma_blockwise_scaling.hpp](/home/khkramer/.cache/uv/archive-v0/f24U_Ixv0hjsqw0Ylm8si/tensorrt_llm/deep_gemm/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp)
        - found one concrete structural mismatch in our dormant exact `P15`
          kernel: it was still staging only the atom-sized `32` B rows, while
          traced `P15` is a full-profile `256x128x64` contract
        - corrected the dormant exact kernel to stage `b_packed[128]` and feed
          the manual `P15` B-loader with `kProfileTokenRows=128` instead of the
          atom-sized token tile
      - implication:
        this does not activate `P15` yet, but it removes one real full-profile
        contract bug from the dormant exact path. The next `P15` rewrite should
        now focus on the register-fragment / accumulator contract, not the old
        undersized B staging.
      - controlled exact re-activation after the `128`-row B fix:
        - re-enabled only the dormant exact `P15` local-CUTE branch and pointed
          `gemm()` back at `tCrA/tCrB`, matching the CUTLASS `mma()` body more
          closely
        - build still failed, but the failure mode is now more precise:
          `cute::gemm(...)` trips in upstream `tensor_zip.hpp` /
          `mma_traits_sm120.hpp` because our local bridge is still handing it a
          plain `subbyte_iterator<uint4_t>` tensor shape where the CUTLASS
          path expects the richer zipped tensor contract produced by its real
          copy pipeline
      - implication:
        after the B-staging fix, the next `P15` blocker is no longer the tile
        extent. It is the exact local copy-to-fragment tensor contract feeding
        `cute::gemm`. The next step should target that contract directly,
        either with a small definitive probe for the CUTLASS-side `copy(...)`
        destination tensor type or by mirroring that copy path more exactly in
        the local bridge.
    - short-input TRT trace correction:
      - a clean live serve run with
        `TLLM_FUSED_MOE_PRINT_TACTICS=1` and
        `TLLM_FUSED_MOE_PRINT_TACTIC_DESCRIPTORS=1` shows the real
        `num_rows=4` request selecting:
        - `gemm1_profile_id=1`: `128x128x64`, `swap_ab=false`
        - `gemm2_profile_id=15`: `256x128x64`, `swap_ab=true`
      - source: 
        [trtllm_short_fc2_descriptor_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trtllm_short_fc2_descriptor_20260407.log)
      - the earlier `gemm2=9` observation came from a probe-contaminated run:
        the temporary `TLLM_FUSED_MOE_PRINT_COMPILE_PROBE_P15=1` launcher hook
        was also forcing an early return in the unrelated `P5` branch during
        autotuning, which perturbed tactic selection
      - implication:
        native short-input alignment should continue targeting `P15` for FC2,
        but should treat short-input FC1 as `P1`, not assume the old
        `P5/P15` pair unconditionally. Future TRT probe runs must avoid
        launcher hooks that perturb unrelated tactic branches.
      - native follow-up:
        a narrow selector-only native trial changing `dispatch_rows == 4` from
        FC1 `P5` to FC1 `P1` failed `multi_turn_prefix_reuse_test` with a real
        cold-prefill execution failure. So the clean TRT `P1/P15` short-input
        pair cannot be adopted by selector change alone; it needs the exact
        traced `P1` math/contract path in native code before any live dispatch
        change.
      - additional live-probe result:
        a dedicated live `P1` compile-probe run did not emit the `P1` probe
        block and the same `num_rows=4` request selected `gemm1=5`, `gemm2=15`
        instead. That means short-input FC1 tactic choice is sensitive enough
        that probe instrumentation itself can perturb autotune/selection. The
        next `P1` fact-finding step should therefore use an offline builder
        probe or a direct runner invocation with fixed profile ids, not another
        live-serve compile probe.
      - definitive offline `P1` builder probe:
        - artifact:
          [trt_p1_runtime_layout_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p1_runtime_layout_dump_20260407.log)
        - exact facts:
          - `Stages = 9`
          - `tile_mnk = (128,32,64)`
          - `cosize(SmemLayoutSFA) = cosize(SmemLayoutSFB) = 4608`
          - `cosize(tCrSFA) = 8`
          - `cosize(tCrSFB) = 32`
          - `accum_profile.layout = ((_2,_2),_2,(_2,_4)):((_1@1,_8@0),_64@0,(_8@1,_32@1))`
        - implication:
          native short-input FC1 `P1` cannot reuse the traced `P5`
          scale-smem or accumulator assumptions.
      - first exact-source native `P1` bridge attempt:
        - added a dedicated traced `P1` scale-smem bridge family in
          [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu)
        - compile and focused correctness gates passed:
          - `fused_moe_prefill_test`
          - `multi_turn_prefix_reuse_test`
        - first real runtime gate failed:
          - `nano_prefix_cache_ttft_bench --prefix-length 4 --tail-token-count 4`
          - `single_token_forward_model: layer 1 kind=2 execution failed`
        - implication:
          exact traced `P1` still needs more than scale-smem alignment alone;
          keep live native `P1` dispatch on the known-good fallback until the
          next definitive probe identifies the missing operand/store or
          accumulator detail.
      - definitive native runtime probe for the short-input failure:
        - reran
          `nano_prefix_cache_ttft_bench --prefix-length 4 --tail-token-count 4`
          with `NEMOTRON_ROUTED_PROFILE_DEBUG=1`
        - exact result:
          - `dispatch_rows=4` cold prefill already ran safely on `P5/P15`
          - the real failure was the resumed `dispatch_rows=8` island, where
            native code selected `P1/P13`
          - failure signature:
            - `embedding_table: token id out of range at index 0 token_id=2147483647`
            - `single_token_forward_model: embedding lookup failed`
        - implication:
          the current exact native `P1/P13` path is not safe for the resumed
          `8`-row regime; the earlier short-input failure was not a `4`-row
          cold-prefill issue
      - temporary safety override:
        - live native dispatch now routes `dispatch_rows == 8` back to the
          known-good `P5/P15` pair while exact traced `P1/P13` work continues
        - after rebuild, the full short-input bench completed successfully:
          [ttft_20260407_rows8_safety_override_prefix4_tail4.stdout.txt](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/ttft_20260407_rows8_safety_override_prefix4_tail4.stdout.txt)
          - `cold_prefill_prefix4 = 62.215 ms`
          - `cached_committed_head_prefix4_tail4 hot-prefix = 57.982 ms`
          - `cached_global_root_prefix4_tail4 hot-prefix = 58.065 ms`
        - next exact target:
          keep `dispatch_rows=8` as the active native gap and probe the missing
          traced `P1/P13` operand/store or accumulator contract there before
          moving the live selector back to TRT parity
      - native A/B isolation on the `dispatch_rows=8` island:
        - `P1/P15` completed safely, so FC1 `P1` is not the remaining blocker by
          itself
        - `P5/P13` reproduced the exact resumed failure:
          - `embedding_table: token id out of range at index 0 token_id=2147483647`
          - `single_token_forward_model: embedding lookup failed`
        - implication:
          the active low-row gap is now localized to FC2 `P13` at
          `dispatch_rows=8`, not FC1 `P1`
        - live state:
          keep `dispatch_rows == 8` on the safe `P5/P15` island until the next
          definitive low-row `P13` probe and fix land
      - dedicated traced `P13` scale-smem bridge attempt:
        - added a dedicated `TracedP13` family from the offline builder probe:
          - `Stages = 9`
          - `cosize(SmemLayoutSFA) = cosize(SmemLayoutSFB) = 4608`
          - `cosize(tCrSFA) = 8`
          - `cosize(tCrSFB) = 32`
        - static gates stayed green:
          - `fused_moe_prefill_test`
          - `multi_turn_prefix_reuse_test`
        - but the first real resumed runtime gate still failed when the live
          selector was moved back to `dispatch_rows == 8 -> P13`:
          - artifact:
            [ttft_20260407_rows8_p13_retry_prefix4_tail4.stdout.txt](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/ttft_20260407_rows8_p13_retry_prefix4_tail4.stdout.txt)
          - exact failure:
            - `routed_gemm1 ... profile=p5_128x128x64_swap_true`
            - `routed_gemm2 ... profile=p13_128x128x64_swap_true`
            - `embedding_table: token id out of range at index 0 token_id=2147483647`
            - `single_token_forward_model: embedding lookup failed`
        - implication:
          low-row FC2 `P13` needs more than traced scale-smem alignment; the
          remaining mismatch is in the accumulator/store contract
      - definitive `P13` accumulator/store probe:
        - artifact:
          [trt_p13_accum_order_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p13_accum_order_dump_20260407.log)
        - exact facts:
          - `tCcC.layout = ((_2,_2),_2,_2):((_1@1,_8@0),_64@0,_8@1)`
          - `tCrC.layout = ((_2,_2),_2,_2):((_1,_2),_4,_8)`
          - `accum_profile.layout = ((_2,_2),_2,(_2,_4)):((_1@1,_8@0),_64@0,(_8@1,_32@1))`
          - flat destination order:
            - `0..3 -> (0,0) (0,1) (8,0) (8,1)`
            - `4..7 -> (64,0) (64,1) (72,0) (72,1)`
            - `8..11 -> (0,8) (0,9) (8,8) (8,9)`
            - `12..15 -> (64,8) (64,9) (72,8) (72,9)`
        - implication:
          the next exact native rewrite should replace the current `P13`
          `CFragment64` store path with a `tCcC/tCrC`-shaped accumulator/store
          contract derived from the traced CUTLASS profile before re-enabling
          live `dispatch_rows == 8 -> P13`
      - `P13` traced accumulator/store rewrite:
        - replaced the low-row `P13` exact path's generic `2`-subtile
          accumulator/store model with a traced `2x2` `A/B/C` fragment family
          and `tCcC/tCrC`-shaped row-major store path in
          [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu)
        - gates stayed green after re-enabling live `dispatch_rows == 8 -> P13`:
          - `fused_moe_prefill_test`
          - `multi_turn_prefix_reuse_test`
        - definitive resumed runtime retry now passes:
          [ttft_20260407_rows8_p13_accum_store_retry_prefix4_tail4.stdout.txt](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/ttft_20260407_rows8_p13_accum_store_retry_prefix4_tail4.stdout.txt)
          - `routed_gemm1 dispatch_rows=8 -> p5`
          - `routed_gemm2 dispatch_rows=8 -> p13`
          - `cached_committed_head_prefix4_tail4 hot-prefix = 58.087 ms`
        - implication:
          low-row FC2 `P13` is now behaviorally safe on the live path; the
          temporary `dispatch_rows == 8 -> P15` safety override is no longer
          needed

#### Full TRT Mainloop Alignment Plan

The next target is full routed-mainloop fidelity to the traced local
`TRT-LLM` path. That means matching the routed grouped kernel family in every
material regard that we can recover from trace, source, and profiler data,
while still implementing it as native custom code in this runtime.

The intended routed end state is:

1. exact traced regime-table dispatch
2. exact TRT-like grouped packed operand contracts
3. exact TRT-like grouped kernel family for FC1 and FC2
4. exact TRT-like producer/consumer mainloop structure
5. exact TRT-like epilogue and output-scale boundaries

What "full alignment" means here:

- match tile shape
- match `swap_ab`
- match grouped batch semantics
- match packed operand interpretation
- match warp-specialized execution structure
- match staged mainloop depth
- match output / epilogue behavior

What remains out of scope for this step:

- shared expert rewrite
- Mamba rewrite
- attention rewrite
- importing TRT generic infrastructure or CUTLASS code generation into the
  production runtime

#### Implementation Plan

1. Freeze the current routed external contract.
   - Keep:
     `permuted_idx_to_token_idx`,
     `total_num_padded_tokens`,
     `num_non_exiting_ctas`,
     `cta_idx_xy_to_batch_idx`,
     `cta_idx_xy_to_mn_limit`,
     `expert_first_token_offsets`,
     `gemm1_output_scale`,
     `activation_output_scale`.
   - Keep the traced regime table as the source of truth.
   - Do not replace it with heuristic thresholds.

2. Recover the missing TRT internal mainloop facts.
   - Run more TRT traces only where information is still missing.
   - For each routed tactic family, recover:
     - exact kernel descriptor
     - exact output / epilogue mode
     - actual kernel name if observable
     - stage-count / pipeline-depth clues
     - any observable launch-shape or shared-memory facts
   - If trace logs are insufficient, run targeted `ncu` on TRT for:
     `prefix4`, `prefix128`, and `prefix4096`.

3. Build one native warp-specialized grouped-kernel substrate.
   - One producer/consumer pipeline substrate for routed FC1/FC2.
   - Inputs:
     packed low-precision activations, packed low-precision weights, TRT-style
     grouped CTA metadata.
   - Outputs:
     BF16 `gemm1_output` or BF16/FP32 `gemm2_output` at the current routed
     stage boundaries.
   - Required features:
     - profile-specific `tile_m / tile_n / tile_k`
     - profile-specific `swap_ab`
     - profile-specific epilogue mode
     - staged double- or triple-buffered mainloop
     - no scalar per-element decode inside the MMA hot loop

4. Implement the traced routed tactic family directly.
   - FC1 kernels:
     - `128x128x64 swap_ab=true`
     - `128x128x64 swap_ab=false`
     - `128x128x128 swap_ab=true`
     - `128x128x128 swap_ab=false`
     - `256x128x64 swap_ab=true`
   - FC2 kernels:
     - `128x128x64 swap_ab=true`
     - `128x128x128 swap_ab=true`
     - `256x128x64 swap_ab=true`

5. Match TRT stage boundaries exactly.
   - FC1 writes BF16 `gemm1_output`.
   - `gemm1_output_scale` and `activation_output_scale` are produced at the
     same logical boundary as TRT.
   - FC2 consumes the packed activated contract directly.
   - Keep finalize unchanged until the routed kernel family is complete.

6. Replace the current grouped WMMA body profile-by-profile.
   - Land profiles in this order:
     - FC1 profile `5`, FC2 profile `12`
     - FC1 profile `1`, FC2 profile `13`
     - FC1 profile `4/0`, FC2 profile `13/12`
     - FC1 profile `7`, FC2 profile `15`
   - Once a profile is covered and validated, retire its use of the current
     grouped WMMA body.
   - Fallback should remain only for unimplemented traced profiles.

7. Remove the legacy grouped and row-tile bodies after full traced coverage.
   - After all traced tactic islands are covered by the new family, delete:
     - the current grouped WMMA body
     - then the old legacy row-tile routed fallback

#### Blockers To Full TRT Alignment

1. Missing internal pipeline facts.
   - We know tile shapes, `swap_ab`, and epilogue flags.
   - We do not yet directly know:
     - stage count
     - producer/consumer warp-role split
     - shared-memory swizzle/layout details
     - exact async transport pattern

2. No native TMA/warp-specialized substrate yet.
   - Our current grouped kernels are still synchronous WMMA kernels with
     explicit shared-memory staging and full CTA barriers.
   - TRT is selecting `TMA Warp Specialized` tactics.

3. Epilogue behavior is only partially recovered.
   - We know `gemm2` often selects `epilogue_fusion=1`.
   - We have not yet fully reconstructed what remains inside that fused
     epilogue versus what happens as separate native work.

4. Some internal layout details are still inferred rather than observed.
   - Packed live shapes are known.
   - CTA metadata is known.
   - Exact internal tile/swizzle details for the selected tactics are not yet
     fully recovered.

5. The traced tactic family is irregular.
   - TRT is not using one monotonic threshold policy.
   - Full fidelity requires a real profile family, not one universal kernel.

#### Additional TRT Ground Truth (`2026-04-07`)

Targeted local `nsys` on the traced TRT `prefix128` routed path adds several
mainloop facts beyond the earlier tactic descriptors:

- the real routed GEMMs are `cutlass::gemm::kernel::GemmUniversal<...>`
  instantiations over `MainloopSm120ArrayTmaWarpSpecializedBlockScaled`
- the live FC1/FC2 hot kernels launch with `BlockX=384`
- TRT also launches separate helper kernels around the grouped GEMMs:
  - `computeStridesTmaWarpSpecializedKernel<...>` with `BlockX=128`
  - `doActivationKernel<...>` with `BlockX=256`
- the live SM120 routed path is block-scaled FP4 grouped GEMM, not BF16 WMMA:
  - operands are `cutlass::float_e2m1_t`
  - the MMA atom is `SM120_16x8x64_TN_VS`
  - FC1/FC2 epilogues are distinct fused callback families

That makes the remaining native gap very concrete:

- helper-stage structure is now materially closer to TRT
- regime-table selection is now materially closer to TRT
- CTA shape is now materially closer to TRT
- the remaining miss is the hot loop itself:
  - no native TMA mainloop
  - no native FP4xFP4 MMA path
  - no native CUTLASS-like producer/consumer pipeline

#### Current Routed Status (`2026-04-07`)

The active grouped routed FC1/FC2 kernels now use the traced larger CTA shape:

- `BlockX=384`
- eight consumer warps on the WMMA tile
- four extra staging warps in the same CTA

This is not full TRT fidelity yet, but it is a useful intermediate step because
the live TRT traces showed the grouped routed kernels launching at `384`
threads, not `256`.

Focused gate after the CTA-shape change:

- artifact:
  `artifacts/benchmarks/ttft_20260407_grouped_384cta_prefix128_tail4.stdout.txt`
- result:
  - `cold_prefill_prefix128 = 125.839 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 54.766 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix = 54.540 ms`
- focused validation:
  - `fused_moe_prefill_test`
  - `moe_launch_plan_device_test`
  - `multi_turn_prefix_reuse_test`

Conclusion:

- keep the larger grouped CTA shape
- do not spend more time on grouped metadata or helper-stage cleanup
- next rewrite must target the true FP4/TMA-style grouped mainloop body

#### Current Routed Status (`2026-04-07`, `computeStrides` Cutover)

The active grouped routed FC1/FC2 kernels now consume precomputed CTA row
metadata directly:

- `cta_row_starts`
- `cta_valid_rows`

instead of recomputing token-row bounds from:

- `cta_idx_xy_to_mn_limit`
- `expert_first_token_offsets`
- `selected_token_tile`

This makes the launch-plan build act as the native analogue of TRT's
`computeStrides...` helper for the active grouped consumers: hot CTAs now start
from exact routed row spans rather than reconstructing them inside the mainloop.

Focused gate after the cutover:

- artifact:
  `artifacts/benchmarks/ttft_20260407_compute_strides_cutover_prefix128_tail4.stdout.txt`
- result:
  - `cold_prefill_prefix128 = 125.510 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 54.530 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix = 54.470 ms`
- focused validation:
  - `fused_moe_prefill_test`
  - `moe_launch_plan_device_test`
  - `multi_turn_prefix_reuse_test`

Conclusion:

- keep the direct `cta_row_starts` / `cta_valid_rows` contract
- the launch plan now serves as the native routed `computeStrides` stage
- the remaining routed gap is still the grouped hot loop itself

#### Current Routed Status (`2026-04-07`, Warp-Specialized Ping-Pong Mainloop)

The active grouped routed FC1/FC2 kernels now use the extra four staging warps
as real producers:

- one shared-memory buffer is consumed by the eight WMMA warps
- the other buffer is filled with the next `K` tile by the producer warps
- the loop alternates buffers with one CTA barrier per macro tile

This is the first routed hot-loop step that actually overlaps staging with MMA
inside the native grouped body, rather than only matching TRT's CTA shape and
helper boundaries.

Focused gate after the ping-pong cutover:

- artifact:
  `artifacts/benchmarks/ttft_20260407_warp_specialized_pingpong_prefix128_tail4.stdout.txt`
- result:
  - `cold_prefill_prefix128 = 125.483 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 54.657 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix = 54.397 ms`
- focused validation:
  - `fused_moe_prefill_test`
  - `moe_launch_plan_device_test`
  - `multi_turn_prefix_reuse_test`

Conclusion:

- keep the warp-specialized ping-pong mainloop
- this confirms that native producer/consumer overlap is directionally right
- the remaining routed gap is now even more specifically the lack of TRT's
  block-scaled FP4 MMA/TMA transport, not just missing warp specialization

#### How To Unblock This

1. Add one more round of targeted TRT tracing / profiling.
2. Recover the missing per-profile pipeline facts.
3. Implement the native producer/consumer grouped substrate.
4. Port the traced tactic family onto that substrate.
5. Delete the current grouped WMMA body only after traced coverage is complete.

#### Acceptance Criteria

- `fused_moe_prefill_test`
- `moe_launch_plan_device_test`
- `multi_turn_prefix_reuse_test`
- no regression in behavioral reuse equivalence
- routed profile dispatch matches the traced TRT family for covered regimes
- cold TTFT continues improving on:
  - `prefix128`
  - `prefix4096`
- routed experts stop dominating the cold-prefill profile

   - Priority C:
     tiny/tail path:
     FC1 `5` or `7`, FC2 `15`.
   - Priority D:
     medium odd islands:
     FC1 `0/4`, FC2 `13/12`.

7. Remove the current WMMA row-tile routed consumers as each regime becomes
   covered by the new grouped family.
   - Do not keep the old routed body alive as a parallel long-term
     implementation.

#### Initial Native Regime Table

- `rows=1`
  - FC1: profile `0`, `128x128x128`, `swap_ab=false`
  - FC2: profile `13`, `128x128x64`, `swap_ab=true`
- `rows=2..3`
  - FC1: profile `7`, `256x128x64`, `swap_ab=true`
  - FC2: profile `15`, `256x128x64`, `swap_ab=true`
- `rows=4..7`
  - FC1: profile `5`, `128x128x64`, `swap_ab=true`
  - FC2: profile `15`, `256x128x64`, `swap_ab=true`
- `rows=32..112`
  - FC1: profile `0`, `128x128x128`, `swap_ab=false`
  - FC2: profile `13`, `128x128x64`, `swap_ab=true`
- `rows=120..127`
  - FC1: profile `4`, `128x128x128`, `swap_ab=true`
  - FC2: profile `13`, `128x128x64`, `swap_ab=true`
- `rows=128..192`
  - FC1: profile `5`, `128x128x64`, `swap_ab=true`
  - FC2: profile `12`, `128x128x128`, `swap_ab=true`
- `rows=200..248`
  - FC1: profile `4`, `128x128x128`, `swap_ab=true`
  - FC2: profile `12`, `128x128x128`, `swap_ab=true`
- `rows=256`
  - FC1: profile `5`, `128x128x64`, `swap_ab=true`
  - FC2: profile `13`, `128x128x64`, `swap_ab=true`
- `rows=512..992`
  - FC1: profile `5`, `128x128x64`, `swap_ab=true`
  - FC2: profile `12`, `128x128x128`, `swap_ab=true`
- `rows=1024+`
  - FC1: profile `1`, `128x128x64`, `swap_ab=false`
  - FC2: profile `13`, `128x128x64`, `swap_ab=true`

#### Stop Conditions And Gates

- After each new regime lands:
  - `device_nvfp4_matrix_test`
  - `moe_launch_plan_device_test`
  - `fused_moe_prefill_test`
  - `multi_turn_prefix_reuse_test`
- Stage gates:
  - after Priority A, `cold_prefill_prefix128` must improve over the current
    routed grouped baseline
  - after Priority B, `cold_prefill_prefix4096` routed share must drop
    materially in profile
  - after Priority C, `cached_committed_head_prefix128_tail4` must not regress
    behaviorally
- Exit this sub-plan when routed expert math is no longer the dominant cold
  prefill bucket. At that point move shared experts onto the same kernel family,
  then return to long-prefix Mamba.

## Goal

Optimize the remaining dominant prefill work for Nemotron 3 Nano NVFP4 on the
local RTX 5090 runtime. The current post-attention baseline is already much
better:

- `cold_prefill_prefix1024`: `5971.231 ms` -> `440.312 ms`
- `cold_prefill_prefix4096`: `133038.284 ms` -> `1369.694 ms`

The next goal is not "fix attention". It is to cut the remaining routed-MoE
and Mamba cost on this exact stack without broadening scope to generic models
or datacenter-only kernels.

## Scope Guardrails

- **Target GPU only**: GeForce RTX 5090, consumer Blackwell, `SM120`, CUDA
  13.0.
- **Target model only**: Nemotron 3 Nano NVFP4 with `hidden_size=2688`,
  `routed_expert_intermediate_size=1856`,
  `shared_expert_intermediate_size=3712`, `n_routed_experts=128`, `top_k=6`,
  `activation=ReLU²`, `46` expert layers, causal paged KV cache, `64k`
  retained context.
- **Target workload only**: single-user, long multi-turn conversations,
  `<=4` concurrent requests, low scheduler complexity, no EP/TP, no LoRA, no
  multimodal path, no obligation to generalize the fast path.
- **Optimization objective**: improve the real local serving path. Do not spend
  time on detours that are not candidates for the final architecture.

## Current Root Cause

The fresh `1024`-token cold-prefill `nsys` profile shows:

- Mamba prefill kernels: about `31%` of GPU kernel time
- routed MoE GEMMs plus runtime NVFP4 input packing/scaling: about `43%`
- additional routed MoE dispatch/finalize kernels: about `9%`
- attention kernels: about `6%`

That means:

1. Attention is no longer the gating issue.
2. Grouped routed-expert work is worth revisiting.
3. The old MoE-only plan was incomplete because it treated runtime
   FP32→NVFP4 input packing as secondary, but that packing/scaling path alone
   is now a large cost.
4. Mamba prefill is now a co-primary bottleneck and must stay in scope.

Reference artifact: `post_attention_root_cause.md`

## Prior Art

- **vLLM / FlashInfer CUTLASS MoE**
  Uses grouped routed-expert execution with explicit routing and finalize
  structure. This remains the closest external prior art for reducing per-expert
  launch count on the local workload.
- **TRT-LLM grouped GEMM / fused MoE**
  Useful for algorithmic structure and grouped execution patterns, but much of
  the strongest path is shaped around datacenter Blackwell assumptions and is
  not a direct template for consumer `SM120`.
- **Our current runtime**
  Already has native SM120 NVFP4 GEMMs that are individually fast. The problem
  is now the whole routed path: repeated input packing/scaling, routing/finalize
  traffic, and the remaining launch surface.

## Locked Contract

- Exact Nano routed-MoE dimensions only.
- `SM120` only for the production fast path.
- BF16 live activations and BF16 KV cache remain the runtime default.
- `top_k=6`, `ReLU²`, FP32 accumulation/output on the routed path unless a
  later step proves another choice is both faster and safe.
- `<=4` concurrent requests, optimize `1` first.
- `64k` retained context is required, but the main optimization target remains
  the routed prefill path, not arbitrary full-context all-at-once operation.
- behavioral reuse equivalence is required for prefix-cache correctness
- no second runtime path should be introduced just to chase these wins

## Architecture Decision

Do **not** resume the old plan as "write one giant custom grouped kernel
immediately". The new order should be:

1. decide what happens to runtime FP32→NVFP4 input packing
2. reduce routed-expert launch count and dispatch overhead
3. optimize Mamba prefill in parallel or immediately after the routed path,
   depending on the next measured split

If grouped routed-expert execution is built without addressing runtime packing,
the result will leave a large known cost untouched.

## Steps

- [x] **0. Freeze the post-attention baseline and root-cause profile**
  Save the current TTFT and fresh `nsys` kernel-time breakdown as the new
  baseline for all remaining work.
  Key files:
  `post_attention_root_cause.md`,
  `step9_runs/20260405T/post_attention_ttft.txt`,
  `step9_runs/20260405T/post_attention_1024_stats.txt`

- [x] **1. Lock the revised optimization contract**
  Rewrite the implementation contract around the post-attention profile:
  routed-MoE/input-packing work plus Mamba, not attention. Freeze the target
  metric set for future comparisons:
  - `cold_prefill_prefix1024`
  - `cold_prefill_prefix4096`
  - `cached_committed_head_prefix8192_tail32`
  - `cached_committed_head_prefix32768_tail32`
  - `cached_committed_head_prefix65536_tail32`
  Key files:
  `PLAN.md`,
  `ATTENTION_PLAN.md`,
  `post_attention_root_cause.md`

- [x] **2. Decide the routed-expert input format and packing strategy**
  The current runtime spends a large share of GPU time in:
  - `PackRowMajorFp32ToNvfp4Kernel`
  - `ComputeGlobalMaxAbsKernel`
  This step must decide whether the production path should:
  - eliminate runtime FP32→NVFP4 packing,
  - fuse it into a reduced-launch routed path,
  - or replace it with a different activation/input contract.
  Do not start grouped routed-expert work until this is explicit.

  **Decision (revised after TRT-LLM investigation)**: Keep FP4×FP4 on the
  tensor cores — that is what cuBLASLt, vLLM, and TRT-LLM all do. The
  problem is not that activations are quantized to NVFP4; the problem is
  that the current path quantizes **per expert** inside the per-expert host
  loop. The fix is:
  1. Quantize activations to NVFP4 **once per layer** (one kernel launch)
  2. Grouped FP4×FP4 GEMM for all experts in one launch using SM120
     tensor core MMA instructions
  TRT-LLM does exactly this: separate `fp4_quantize()` kernel, then
  grouped CUTLASS FP4×FP4 GEMM. An earlier scaffold kernel using scalar
  `Nvfp4RowMajorDot()` was removed because it bypassed tensor cores
  entirely — the production kernel must use FP4×FP4 MMA.

  Key files:
  `runtime/src/backend/fused_moe_prefill.cu`,
  `runtime/src/backend/expert_layer.cpp`,
  `runtime/src/backend/device_nvfp4_matrix.cu`,
  `runtime/src/api/single_token_forward_model.cpp`

- [ ] **3. Grouped FP4×FP4 tensor core MoE kernel**
  Write a single CUDA kernel that processes all routed experts in one launch
  using SM120 FP4×FP4→FP32 tensor core MMA instructions. The architecture:
  1. **Quantize activations to NVFP4 once** — one kernel converts the full
     BF16 normalized input to block-scaled FP4 + scales. (Matches TRT-LLM's
     `fp4_quantize()` step.)
  2. **Grouped GEMM** — one kernel launch for all experts. Each block
     handles one (expert, tile) pair. Both activations and weights are FP4.
     CUTLASS-style tiled GEMM with warp-level MMA instructions. Fuse ReLU²
     between up and down projections. Weighted scatter to output.
  3. Wire into `expert_layer.cpp` behind an env-var gate, compare against
     the cuBLASLt path for correctness.
  Exit criterion:
  - lower routed-path GPU time than the current per-expert cuBLASLt path
  - no regression in correctness on real Nano weights
  - FP4×FP4 on tensor cores, not scalar dot products
  Key files:
  `runtime/src/backend/fused_moe_grouped.cu` (new),
  `runtime/src/backend/expert_layer.cpp`,
  `runtime/src/backend/expert_routing_device.cu`
  Reference:
  `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/`,
  `third_party/TensorRT-LLM/cpp/tensorrt_llm/thop/fp4Quantize.cpp`

- [ ] **4. Reduce routed dispatch/finalize and allocator churn**
  The current post-attention profile still shows meaningful routed non-GEMM
  work plus noticeable `cudaMalloc` / `cudaFree` traffic in CUDA API time.
  After step 3 exists, reduce the residual routing/finalize overhead and remove
  avoidable allocation churn on the hot path.
  Key files:
  `runtime/src/backend/expert_routing_device.cu`,
  `runtime/src/backend/request_context.cpp`,
  `runtime/src/backend/expert_layer.cpp`

- [ ] **5. Optimize Mamba prefill**
  Mamba is now a co-primary bottleneck. Continue from the already-landed
  bounded-chunking fix and profile the actual hot Mamba kernels rather than
  assuming the routed-MoE path will dominate forever.
  Key files:
  `runtime/src/backend/mamba_layer.cpp`,
  `runtime/src/backend/mamba_ssd_prefill.cu`,
  `runtime/src/backend/mamba_conv_prefill.cu`

- [ ] **6. Re-profile and choose the next default workstream**
  After steps 2 through 5 materially change the runtime, capture a fresh
  end-to-end profile and decide which of these becomes the next default
  production focus:
  - routed MoE/input packing
  - Mamba prefill
  - decode-path cleanup
  - scheduler/prefix-cache policy
  Key files:
  `post_attention_root_cause.md`,
  `PLAN.md`,
  `TODO.md`

## Frozen Baseline (post-attention, SM120, 2026-04-05)

### Cold Prefill TTFT

| Prompt | Median | p95 |
|--------|--------|-----|
| 256 tokens | 293 ms | 396 ms |
| 1024 tokens | 440 ms | 444 ms |
| 4096 tokens | 1,370 ms | 1,374 ms |

### Cached Prefix + 32 Tail Tokens

| Prefix | Cold TTFT | Hot-prefix TTFT | Tail Prefill | Speedup |
|--------|-----------|-----------------|--------------|---------|
| 4096 + 32 | 1,551 ms | 210 ms | 172 ms | 7.4x |
| 8192 + 32 | 3,241 ms | 243 ms | 184 ms | 13.4x |
| 32768 + 32 | 20,200 ms | 500 ms | 308 ms | 40.4x |
| 65536 + 32 | 63,395 ms | 845 ms | 477 ms | 75.1x |

### Kernel Time Split (1024-token cold, nsys)

| Component | Share |
|-----------|-------|
| MambaSsdPrefill | 28.7% |
| Mamba other (conv + group norm) | 2.6% |
| Routed NVFP4 GEMM | 26.3% |
| PackRowMajorFp32ToNvfp4 | 10.7% |
| ComputeGlobalMaxAbs | 6.2% |
| Routed dispatch/finalize | 8.4% |
| Attention (prefill + decode) | 6.0% |
| **Mamba total** | **31.3%** |
| **Routed MoE total** | **51.6%** |

### Decode (unchanged)

| Metric | Value |
|--------|-------|
| Steady-state | 64.6 tok/s |

### Optimization Contract

All future steps in this plan are measured against the baselines above.
Success means materially reducing routed-MoE and/or Mamba kernel time
without regressing correctness (ctest green), decode throughput, or
attention prefill latency.

## Verification Policy

### Tier 1: after each implementation step

- `cmake --build build-sm120-relwithdebinfo -j$(nproc)`
- `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure`

### Tier 2: after routed-path or Mamba changes

- `nano_prefix_cache_ttft_bench`
- one fresh `nsys` trace on `cold_prefill_prefix1024`
- one long-context cached-prefix run at `32k` or `64k`

### Tier 3: before changing priorities again

- refreshed root-cause note in `post_attention_root_cause.md`
- explicit decision on whether grouped routed-MoE work remains ahead of Mamba

## Probe Strategy

When a TRT/CUTLASS/CUTE internal is not directly observable from normal traces:

1. use source once to form the first implementation
2. if that first implementation fails a real gate, stop guessing
3. add the smallest compile-time or runtime probe that answers the exact open
   question definitively
4. rerun inference, record the answer in the progress docs, and implement from
   that fact

Use runtime traces/logging for dynamic-selection questions:

- tactic id
- tile shape
- `swap_ab`
- `BlockX`
- regime boundaries

Use compile-time or source-backed probes for template/layout questions:

- `SmemLayoutAtomSFA`
- `SmemLayoutAtomSFB`
- `SmemCopyAtomSFA`
- `SmemCopyAtomSFB`
- pipeline stage count
- fragment/copy coord maps

## Immediate P5 Plan

Current success goal:

- make traced routed `P5` (`gemm1=5`, `128x128x64`, `swap_ab=true`) pass
  `fused_moe_prefill_test` and `multi_turn_prefix_reuse_test`
- keep `P5` on the traced FP4 path only after both gates stay green

Next definitive probes for `P5`:

1. Dump the exact compile-time builder choices for the traced `P5` tactic:
   - `SmemLayoutAtomSFA`
   - `SmemLayoutAtomSFB`
   - `SmemCopyAtomSFA`
   - `SmemCopyAtomSFB`
   - `PipelineStages`
2. Dump a small per-thread coordinate map for the traced `P5` scale path:
   - `tCsSFA`
   - `tCrSFA_copy_view`
   - `tCsSFB`
   - `tCrSFB_copy_view`
3. Mirror those exact facts in the local bridge:
   - shared-memory scale layout/swizzle
   - scale copy atom selection
   - stage-materialization model
4. Re-enable only `P5` on the traced FP4 kernel and rerun:
   - `fused_moe_prefill_test`
   - `multi_turn_prefix_reuse_test`

## Progress

| # | Step | Status | Notes |
|---|------|--------|-------|
| 0 | Freeze the post-attention baseline and root-cause profile | done | Historical baseline frozen; newer external race targets now live above and in `proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md` |
| 1 | Lock the revised optimization contract | done | One runtime path, device-only execution, behavioral reuse equivalence |
| 2 | Decide routed-expert input format and packing strategy | done | `FP4xFP4` on tensor cores; quantize activations once per layer; grouped GEMM for all experts |
| 3 | Grouped `FP4xFP4` tensor core MoE kernel | in progress | Device launch plan, padded row layout, C++20, and the narrow local `TiledCopy` bridge are in place; traced `P5`, `P12`, and `P13` routed FP4 paths are now live and behaviorally safe, with low-row `dispatch_rows == 8 -> P13` fixed by the traced `tCcC/tCrC` accumulator/store rewrite; `fused_moe_prefill_test` and `multi_turn_prefix_reuse_test` pass, focused TTFT improved to `cold_prefill_prefix128 = 107.215 ms`, `cold_prefill_prefix4096 = 1633.296 ms`, and the resumed short-input retry completes with `dispatch_rows=8 -> p13`; the remaining traced routed fallback is `P15`, which still needs its dedicated full-profile fragment/accumulator contract |
| 4 | Shared-expert alignment | pending | Shared experts still distort short-prefix cold prefill and need the same stronger math family |
| 5 | Optimize Mamba prefill | pending | Still the second blocker at `prefix4096` after routed MoE |
| 6 | Re-profile and choose the next default workstream | pending | Final race phase only after routed MoE, shared, and long-prefix Mamba move materially |

### Latest P15 Exact-Path Result

- followed the probe-first rule and added a definitive operand-type dump:
  - artifact:
    [trt_p15_copy_type_dump_20260407.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_copy_type_dump_20260407.log)
- exact facts:
  - traced `P15` uses `tCrA.type = Tensor<ArrayEngine<integer_subbyte<4>, 128>, ...>`
  - traced `P15` uses `tCrB.type = Tensor<ArrayEngine<integer_subbyte<4>, 128>, ...>`
  - traced `tCrA_copy_view` / `tCrB_copy_view` are `ViewEngine<subbyte_iterator<integer_subbyte<4>>>`
  - `tCsA` / `tCsB` remain shared-memory views over `subbyte_iterator<float_e2m1_t>`
- native follow-up:
  - switched the dormant exact `P15` local-CUTE branch off
    `float_e2m1_unpacksmem_t` and onto the probed `ElementAB` shared-memory
    basis
  - re-enabled only the dormant exact `P15` branch and rebuilt as a controlled
    compile check
- result:
  - compile still fails in `cute::gemm(...)`, but the failure is now more
    precise: our dormant exact `P15` path is still feeding
    `ViewEngine<subbyte_iterator<uint4_t>>` operand tensors into the SM120 MMA
    path where the traced CUTLASS contract uses array-engine register fragments
- implication:
  - the next exact `P15` rewrite should stop trying to drive `cute::gemm(...)`
    from partitioned smem views
  - instead it should rebuild the dormant exact `P15` path around the real
    copy-to-register fragment tensors implied by the traced CUTLASS contract,
    then retry the controlled activation

### Latest P15 Source-Backed Contract Result

- read the traced SM120 blockscaled collective body in:
  [sm120_mma_array_tma_blockwise_scaling.hpp](/home/khkramer/.cache/uv/archive-v0/f24U_Ixv0hjsqw0Ylm8si/tensorrt_llm/deep_gemm/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp)
- exact CUTLASS flow for the traced profile is:
  - build `sA` / `sB` as full staged shared-memory tensors
  - partition them with `partition_fragment_A/B`
  - build `tCsA/tCsB` from `partition_S(as_position_independent_swizzle_tensor(sA/sB))`
  - copy `tCsA/tCsB -> tCrA_copy_view/tCrB_copy_view`
  - apply `fp4_shift_A/B` on the copy views
  - run `cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tmp_accum)`
- definitive follow-up probes:
  - `trt_p15_reg_gemm_contract_dump.cu`
  - `trt_p15_make_fragment_contract_dump.cu`
- result:
  - neither a naive `recast` register path nor a naive `make_fragment_A/B + copy`
    path compiled cleanly in isolation
  - that means the remaining `P15` mismatch is not just the fragment object type
  - it is the full CUTLASS staged operand contract, especially the swizzled
    `partition_S(...)` source path feeding the copy views
- implication:
  - the next exact `P15` rewrite should replace the current row-packed
    `a_packed/b_packed` exact-path staging with a real `SmemLayoutA/B`-shaped,
    staged shared-memory path before another activation attempt

### Latest P15 Probe-Generation Result

- implemented a maintained all-thread `P15` contract probe:
  [trt_p15_full_thread_contract_dump.cu](/home/khkramer/src/nemotron-inference/proj-2026-04-05-1704/trt_p15_full_thread_contract_dump.cu)
- implemented a maintained generator that compiles/runs that probe and emits
  machine-readable artifacts:
  [generate_p15_probe_tables.py](/home/khkramer/src/nemotron-inference/proj-2026-04-05-1704/generate_p15_probe_tables.py)
- latest generated artifacts:
  - [trt_p15_full_thread_contract_dump_latest.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_full_thread_contract_dump_latest.log)
  - [trt_p15_probe_contract.json](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_probe_contract.json)
  - [trt_p15_probe_generated.inc](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_probe_generated.inc)
- exact result:
  - we now have all-thread, probe-backed dense operand/store and scale-stage
    coordinate data for `P15`, not just thread `0`
  - so probe-driven generation of the `P15` fragment/copy contract is feasible
    and now implemented as an offline codegen pipeline
- practical constraint:
  - the naive generated all-thread tables are too large to drop directly into
    live device constant memory
  - so the immediate output is an offline codegen artifact, not yet a live
    runtime include
- implication:
  - the next `P15` step should use these generated all-thread artifacts to
    derive compact pattern classes or emit specialized helper code for the
    dormant exact path, rather than writing more handwritten coord logic

### Latest P15 Generated-Helper Integration Result

- promoted the probe/codegen output into a compact runtime header:
  [p15_probe_generated.h](/home/khkramer/src/nemotron-inference/runtime/include/nemotron/p15_probe_generated.h)
- exact native follow-up:
  - wired the dormant exact `P15` path in
    [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu)
    to use the generated dense operand/store helpers and generated stage-0
    scale helpers instead of ad hoc `FillPhysicalCoordMap...` coord recovery
  - kept live `P15` dispatch on the BF16 fallback while validating the dormant
    path
- result:
  - focused build is green
  - [fused_moe_prefill_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/fused_moe_prefill_test):
    `PASS`
  - [multi_turn_prefix_reuse_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/multi_turn_prefix_reuse_test):
    `PASS`
  - so the probe-generated `P15` contract is now integrated into the native
    backend without regressing the safe live path
- controlled activation result:
  - switching only the live `P15` launch site to the exact FP4 kernel still
    fails at compile time in the local-CUTE `cute::gemm(...)` path
  - failure site:
    [tensor_zip.hpp](/home/khkramer/src/nemotron-inference/.venv-trtllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include/cute/tensor_zip.hpp)
    via
    [mma_traits_sm120.hpp](/home/khkramer/src/nemotron-inference/.venv-trtllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include/cute/atom/mma_traits_sm120.hpp)
  - exact symptom:
    `cute::subbyte_iterator<cute::uint4_t>` lacks the `iters_` interface that
    the SM120 MMA unzip path expects
- implication:
  - the next `P15` blocker is no longer coord discovery
  - it is the exact copy-to-register tensor type feeding `cute::gemm(...)` in
    the live exact path
  - next step should replace the dormant exact `P15` bridge's current
    subbyte-iterator register feed with the CUTLASS-expected array-engine
    register contract before the next activation attempt

### Latest P15 Staged-Swizzle Source Probe Result

- added a dedicated staged-smem source probe:
  [trt_p15_smem_partition_dump.cu](/home/khkramer/src/nemotron-inference/proj-2026-04-05-1704/trt_p15_smem_partition_dump.cu)
- added a maintained generator for that probe:
  [generate_p15_smem_partition_tables.py](/home/khkramer/src/nemotron-inference/proj-2026-04-05-1704/generate_p15_smem_partition_tables.py)
- generated artifacts:
  - [trt_p15_smem_partition_dump_latest.log](/home/khkramer/src/nemotron-inference/artifacts/benchmarks/trt_p15_smem_partition_dump_latest.log)
  - [trt_p15_smem_partition_contract.json](/home/khkramer/src/nemotron-inference/artifacts/tmp/trt_p15_smem_partition_contract.json)
  - [p15_smem_partition_generated.h](/home/khkramer/src/nemotron-inference/runtime/include/nemotron/p15_smem_partition_generated.h)
- exact result:
  - we now have all-thread, probe-backed `tCsA/tCsB` stage-0 source coords for
    traced `P15`
  - the staged-swizzle source contract is highly regular:
    - `TCSA`: `warp_mod4` only shifts the second leaf by `+2`
    - `TCSB`: `upper_half` only shifts the second leaf by `+2`
- implication:
  - the remaining operand-source guesswork for `P15` is gone
  - the next exact native rewrite should consume the generated
    `GetTCSAStage0Coord` / `GetTCSBStage0Coord` helpers instead of the older
    dense row-packed source assumption
- exact native follow-up:
  - rewired the dormant exact `P15` source loads in
    [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu)
    to use `GetTCSAStage0Coord` / `GetTCSBStage0Coord`
  - focused rebuild is green
  - [fused_moe_prefill_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/fused_moe_prefill_test):
    `PASS`
  - [multi_turn_prefix_reuse_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/multi_turn_prefix_reuse_test):
    `PASS`
- implication:
  - the dormant exact `P15` path now matches the traced source-side contract
    much more closely
  - the next activation attempt should target the remaining live exact
    `cute::gemm(...)` / register-contract blocker, not source coord recovery

### P15 Reset Strategy

- stop the current `P15` probe-patch loop at its structural boundary
- new rule before any `P15` attempt:
  - answer in one sentence whether the attempt preserves or replaces
    row-packed staging
  - if it preserves row-packed staging, reject it without running the gates
- design-review rule:
  - after three failed probe-then-patch cycles on the same structural approach,
    stop and review the approach itself instead of probing another leaf detail
- smell rule:
  - if the next step is "add another generated coord table / fragment helper /
    layout shim" on top of row-packed staging, treat that as a sign the staging
    mechanism is still wrong
- active structural decision:
  - replace the dormant exact `P15` row-packed `a_packed/b_packed` staging with
    real `SmemLayoutA/B`-shaped staged shared memory
  - feed the live CUTLASS contract end to end:
    `partition_S(as_position_independent_swizzle_tensor(sA/sB)) -> copy ->
    fp4_shift -> make_zip_tensor(...) -> cute::gemm(...)`
- immediate gate before any more MoE integration:
  - add a standalone `P15` backend test that does only:
    - real `SmemLayoutA/B/SFA/SFB` shared storage
    - known-value fill
    - real CUTLASS copy-to-register path
    - `fp4_shift_A/B`
    - zipped `cute::gemm(...)`
    - accumulator check against a simple reference
- only after that standalone test is green should `P15` exact-path work return
  to the fused MoE kernel

### Latest P15 Swizzled-Pipeline Isolation Result

- added a standalone CUDA backend test:
  [p15_swizzled_pipeline_test.cu](/home/khkramer/src/nemotron-inference/testing/backend/p15_swizzled_pipeline_test.cu)
- exact scope of the test:
  - real `TracedP15CollectiveMainloop` types
  - real `SmemLayoutA/B/SFA/SFB` shared storage
  - real `partition_S(as_position_independent_swizzle_tensor(...))`
  - real smem->reg `copy`
  - real `fp4_shift_A/B`
  - real zipped `cute::gemm(...)`
  - simple zero-input reference
- result:
  - the standalone test now builds and passes:
    [p15_swizzled_pipeline_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/p15_swizzled_pipeline_test)
    prints `PASS max_abs_diff=0`
- implication:
  - the CUTLASS staged swizzled-smem + zipped-`gemm` contract is viable in
    isolation on this machine
  - the remaining `P15` work is now integration work, not proof-of-possibility
  - future `P15` attempts must reuse this exact mechanism and may not preserve
    row-packed `a_packed/b_packed` staging
- next exact integration step:
  - transplant the standalone test's `sA/sB/sSFA/sSFB -> copy -> fp4_shift ->
    make_zip_tensor -> cute::gemm` path into the dormant exact `P15` kernel
  - remove the dormant exact `P15` row-packed operand staging instead of
    compensating for it

### Latest P15 Exact-Path Reset

- the live exact `P15` swizzled transplant still fails the real routed gates:
  - [fused_moe_prefill_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/fused_moe_prefill_test):
    `nano_prefill_routed_max_abs_diff=15.4386`
  - [multi_turn_prefix_reuse_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/multi_turn_prefix_reuse_test):
    committed-head boundary argmax mismatch with `max_abs_diff=12.5938`
- the branch is safe again because live `P15` dispatch has been restored to the
  grouped BF16 fallback in
  [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu)
- useful negative result:
  - the `transform_fragment_for_qmma(...)` hypothesis came from the SM120 FP8
    blockscaled path, not the live NVFP4 MoE path we are cloning
  - source search in
    [moe_gemm]( /home/khkramer/src/nemotron-inference/third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm )
    shows the live NVFP4 MoE launcher exposes `partition_fragment_SFA/SFB(...)`
    but not `transform_fragment_for_qmma(...)`
- next exact step:
  - probe the real TRT NVFP4 MoE `P15` launcher for the raw `tCrSFA/tCrSFB`
    fragment layouts and coords
  - then mirror that exact scale-fragment contract in the native exact `P15`
    path instead of importing FP8 qMMA assumptions

## Latest P15 Completion Result

- the live exact `P15` FP4 path is now the active routed FC2 implementation
- the decisive isolation step was extending
  [p15_swizzled_pipeline_test.cu](/home/khkramer/src/nemotron-inference/testing/backend/p15_swizzled_pipeline_test.cu)
  from the original single-`K` proof to:
  - multi-`K` runtime-like replay
  - Nano-like replay for `dispatch_rows=1/2/4`
- measured isolated Nano-like envelope:
  - `rows=1`: `max_diff=21.3465`
  - `rows=2`: `max_diff=22.9516`
  - `rows=4`: `max_diff=22.9516`
- that changed the diagnosis:
  - the live Nano routed `P15` diff (`~24.8`) was not evidence of a bad
    `matmul_block_scales_data` feed
  - it was within the actual exact-`P15` FP4 profile envelope on the Nano
    deployment shape
- live grouped-pack debugging also settled the scale-buffer question:
  - `matmul_block_scales_data` matched the grouped pack’s own row-major block
    scales after swizzle
  - later failing Nano selections already had exact grouped-pack row and scale
    agreement, so the old “wrong FC2 grouped input scale buffer” theory is no
    longer the best explanation
- the final test state is green:
  - [p15_swizzled_pipeline_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/p15_swizzled_pipeline_test): `PASS`
  - [fused_moe_prefill_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/fused_moe_prefill_test): `PASS`
  - [multi_turn_prefix_reuse_test](/home/khkramer/src/nemotron-inference/build-sm120-relwithdebinfo/testing/multi_turn_prefix_reuse_test): `PASS`
