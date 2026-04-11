# P5 linear `part_c(i)` mapping — confirmed by invariant validation

## Confirmation method

The `ValidateP5WarpLocalInvariants` function (fused_moe_prefill.cu, debug builds) programmatically
validates the expected ownership pattern from `proj-2026-04-11-0400/partition_c_analysis.md` using
the linear `part_c(i)` iteration order. The probe `DumpP5LinearPartitionCLayout` is available
under `NEMOTRON_P5_LINEAR_PARTITION_DEBUG` for visual inspection if needed.

A compile-time `static_assert(decltype(cute::size(part_c))::value == 16)` inside
`StageUnifiedRoutedFp4DirectActivated` confirms exactly 16 valid elements per consumer thread.

## Validated invariants

1. **16 elements per thread** — `cute::size(part_c) == 16`, not 32.
2. **Two FP4 blocks per thread** — one in `m=0..63` (low_block) and one in `m=64..127` (high_block),
   with `high_block == low_block + 4`.
3. **One warp per 16-wide FP4 block** — within a given n-half and block column, all 16 elements
   are owned by lanes in the same warp.
4. **8 participating lanes per block** — stride-4 pattern: lanes `{0,4,8,12,16,20,24,28}` shifted
   by `lane_group = lane_id & 3`. Lane mask: `0x11111111u << lane_group`.
5. **4 token rows per thread** — n_coords are deterministic per `(warp_id, lane_id)`:
   - `n0 = 16*n_half + 2*r`
   - `n1 = 16*n_half + 2*r + 1`
   - `n2 = 16*n_half + 8 + 2*r`
   - `n3 = 16*n_half + 9 + 2*r`
   where `r = lane_id & 3`, `n_half = warp_id / 4`.
6. **4 elements per token row** — each of the 4 token rows has exactly 4 elements (2 in low block,
   2 in high block).

## Mapping formula (from partition_c_analysis.md, confirmed by validator)

- `warp_id = 4*g_n + g_m` where `g_n in {0,1}`, `g_m in {0,1,2,3}`
- `lane_id = 4*q + r` where `q in [0,7]`, `r in [0,3]`
- m_set: `{16*g_m+q, 16*g_m+8+q, 64+16*g_m+q, 72+16*g_m+q}`
- n_set: `{16*g_n+2*r, 16*g_n+2*r+1, 16*g_n+8+2*r, 16*g_n+9+2*r}`

## Implication for step 2

The linear `part_c(i)` ordering matches the nested-loop ordering from the step-2 analysis.
The existing tables can be reused directly as constexpr lookup arrays indexed by `i` (0..15).
