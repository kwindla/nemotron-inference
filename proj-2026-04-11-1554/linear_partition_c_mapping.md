# P5 linear `part_c(i)` mapping — native direct-path correction

## Confirmation method

The earlier note was wrong about the linear ordering. The old table reused the
`FillPhysicalCoordMapCopyViewLimited` / nested-loop dump from
`proj-2026-04-11-0400/partition_c_analysis.md`, but the live native direct path
indexes the accumulator with the true linear `part_c(i)` order.

The current confirmation path is:

- `RunP5NativeDirectPackOracleForTesting` in `fused_moe_prefill.cu`, which
  populates a synthetic 128x128 tile through the real
  `TracedP5AccumProfileLayout` and then calls the live
  `StoreUnifiedRoutedFp4DirectPack` helper.
- `testing/backend/staged_fp4_pack_test.cpp`, which bitwise-compares that live
  path against the staged/reference native oracle using non-uniform
  per-row tensor scales.

This caught the stale lookup table immediately and forced the retable below.

## Actual linear order

For thread `0` (`warp_id=0`, `lane_id=0`), the validated `part_c(i)` order is:

```text
i=00 -> (m0,  n0)
i=01 -> (m0,  n1)
i=02 -> (m8,  n0)
i=03 -> (m8,  n1)
i=04 -> (m64, n0)
i=05 -> (m64, n1)
i=06 -> (m72, n0)
i=07 -> (m72, n1)
i=08 -> (m0,  n8)
i=09 -> (m0,  n9)
i=10 -> (m8,  n8)
i=11 -> (m8,  n9)
i=12 -> (m64, n8)
i=13 -> (m64, n9)
i=14 -> (m72, n8)
i=15 -> (m72, n9)
```

with:

```text
m0  = 16*g_m + q
m8  = 16*g_m + 8 + q
m64 = 64 + 16*g_m + q
m72 = 64 + 16*g_m + 8 + q

n0  = 16*g_n + 2*r
n1  = 16*g_n + 2*r + 1
n8  = 16*g_n + 8 + 2*r
n9  = 16*g_n + 9 + 2*r
```

This order is different from the old nested-loop note, which interleaved
`n0/n8` before `n1/n9`.

## Correct constexpr tables

The live native direct packer now uses:

```text
kLinearToBlockHalf = {0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1}
kLinearToTokenGroup = {0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3}
kLinearToMHalf = {0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1}

kBlockElemMh0 = {0,1,8,9,4,5,12,13}
kBlockElemMh1 = {2,3,10,11,6,7,14,15}
```

with `token_group` indexing `n_coords = {n0, n1, n8, n9}`.

## Validated invariants

1. `cute::size(part_c) == 16`.
2. One warp owns each 16-wide FP4 block.
3. Eight lanes participate in each warp-local reduction and pack.
4. `row_alpha` remains token-row-local and must preserve
   `input_per_row_tensor_scales` when present.
5. The native direct path must use `Relu2(row_alpha * accum)` with no BF16
   truncation before activation.
