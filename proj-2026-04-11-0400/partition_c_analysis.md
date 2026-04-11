# P5 `partition_C` ownership analysis

## Capture notes

- The requested default run

  ```bash
  NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build/testing/nano_24_token_prefill_regression_test
  ```

  passes, but it does **not** hit the P5 routed FC1 kernel on this workload:
  - prompt23 uses `dispatch_rows=23` and falls back to `legacy`
  - prompt24 uses `dispatch_rows=24` and selects `P1`
  - raw output: `proj-2026-04-11-0400/partition_c_dump.txt`

- To probe the live P5 kernel, I reran with the existing debug dispatch override:

  ```bash
  NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 \
  NEMOTRON_DEBUG_USE_SELECTED_TOKEN_TILE_FOR_DISPATCH=1 \
  NEMOTRON_ROUTED_PROFILE_DEBUG=1 \
  build/testing/nano_24_token_prefill_regression_test
  ```

  That forces routed GEMM1 to `dispatch_rows=8`, which selects `p5_128x128x64_swap_true`, emits the dump, and then fails the oracle later because the override changes kernel selection semantics. The dump itself was captured before that failure.
  - raw P5 dump: `proj-2026-04-11-0400/partition_c_dump_forced.txt`
  - structured mapping: `proj-2026-04-11-0400/partition_c_mapping.csv`

## What the dump shows

- The unified P5 kernel's `thread_mma.partition_C(identity_tensor)` produces **16 populated coordinates per consumer thread**, not 32.
- In the diagnostic, `physical_idx=16..31` are `(-1, -1)` after pre-filling the arrays, so those slots are not populated by the `partition_C` copy-view traversal.
- The valid ownership domain across the 256 consumer threads is:
  - `m_coord in [0, 127]`
  - `n_coord in [0, 31]`
- Warp ownership splits cleanly by `n_coord` half:
  - warps `0..3` own `n_coord in [0, 15]`
  - warps `4..7` own `n_coord in [16, 31]`

## 16-column FP4 block ownership

Interpret the FP4 scale group as the usual routed-activation block: a fixed token row (`n_coord`) and a 16-column span in `m_coord`.

- For a fixed `n_coord`, each 16-column block is owned by **exactly one warp**:
  - `m=0..15` and `m=64..79`: warp `0` for `n=0..15`, warp `4` for `n=16..31`
  - `m=16..31` and `m=80..95`: warp `1` / warp `5`
  - `m=32..47` and `m=96..111`: warp `2` / warp `6`
  - `m=48..63` and `m=112..127`: warp `3` / warp `7`
- Within that warp, the 16 elements are split across **8 lanes**, with **2 columns per participating lane**.
- Therefore the block-max / max-abs reduction can be done **within a single warp**.
- It is **not** single-quad-local in this concrete mapping. The participating lanes are strided across the warp, e.g. for `n=0` and `m=0..15` the owners are lanes `0, 4, 8, 12, 16, 20, 24, 28`.

## Exact thread to `(m,n)` formula

For the populated part of the dump, the mapping is exact and regular.

Let:

- `warp_id = 4 * g_n + g_m`
- `g_n in {0, 1}` chooses the `n` half: `0..15` or `16..31`
- `g_m in {0, 1, 2, 3}` chooses the 16-column `m` block modulo 64
- `lane_id = 4 * q + r`
- `q in [0, 7]`
- `r in [0, 3]`

Then the per-lane coordinate sets are:

```text
m_set = {
  16*g_m + q,
  16*g_m + 8 + q,
  64 + 16*g_m + q,
  72 + 16*g_m + q
}

n_set = {
  16*g_n + 2*r,
  16*g_n + 2*r + 1,
  16*g_n + 8 + 2*r,
  16*g_n + 9 + 2*r
}
```

And the populated physical slots are ordered as:

```text
p00 = (m0,  n0)   p01 = (m0,  n8)   p02 = (m64, n0)   p03 = (m64, n8)
p04 = (m0,  n1)   p05 = (m0,  n9)   p06 = (m64, n1)   p07 = (m64, n9)
p08 = (m8,  n0)   p09 = (m8,  n8)   p10 = (m72, n0)   p11 = (m72, n8)
p12 = (m8,  n1)   p13 = (m8,  n9)   p14 = (m72, n1)   p15 = (m72, n9)
p16..p31 = unset in the observed `partition_C(identity_tensor)` dump
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

I verified this formula against all 256 dumped consumer threads.

## Practical conclusion for step 2

- The CUTLASS/CUTE baseline assumption that a 16-element FP4 scale group can be reduced **within a warp** is supported by the concrete P5 dump.
- The stronger "single quad" assumption is **not** supported by this mapping.
- There is an additional concrete concern for later steps:
  - the current unified P5 helper still iterates `physical=0..31`
  - but the observed `partition_C(identity_tensor)` dump populates only `0..15`
  - step 4/5 should treat that as a real contract mismatch to audit before relying on the old 32-slot indexing scheme
