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

## Progress

| # | Step | Status | Notes |
|---|------|--------|-------|
| 0 | Freeze the post-attention baseline and root-cause profile | done | Historical baseline frozen; newer external race targets now live above and in `proj-2026-04-05-1704/EXTERNAL_BASELINES_NOTES.md` |
| 1 | Lock the revised optimization contract | done | One runtime path, device-only execution, behavioral reuse equivalence |
| 2 | Decide routed-expert input format and packing strategy | done | `FP4xFP4` on tensor cores; quantize activations once per layer; grouped GEMM for all experts |
| 3 | Grouped `FP4xFP4` tensor core MoE kernel | in progress | Device launch plan, padded row layout, and partial BF16 WMMA path exist; the remaining gap is moving the active routed path onto the intended packed-activation grouped-`FP4xFP4` contract |
| 4 | Shared-expert alignment | pending | Shared experts still distort short-prefix cold prefill and need the same stronger math family |
| 5 | Optimize Mamba prefill | pending | Still the second blocker at `prefix4096` after routed MoE |
| 6 | Re-profile and choose the next default workstream | pending | Final race phase only after routed MoE, shared, and long-prefix Mamba move materially |
