"""Follow-up to oracle_validation_2026-04-14.py.

The first run showed that direct FP16 GEMM on the saved pre-quantization
`hs` / `w1_16` tensors disagrees with the flashinfer dump even at a 512-ULP
budget, while the oracle-style FP4 dequant+accumulate disagrees at 6.50%
matching. This script tests the orientation hypothesis: is the flashinfer
dump simply in a different M-index order than the saved hs row order, or
in a different N-index order than the saved w1_16 row order?

Strategy: take the straight fp64 `hs @ w1_16[0].T * alpha` result and
search for where flashinfer dump[0,0] (as a float) lives in that result,
and vice versa. If there is a single-row permutation that aligns the two,
we can pin it down by indexed comparison.

Also prints literal values at a few anchor points so we can see what's
actually there.
"""
from __future__ import annotations

import numpy as np
import pathlib
import sys

import torch


ROOT = pathlib.Path(
    "/home/khkramer/src/nemotron-inference/proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688"
)

NUM_ROWS = 128
HIDDEN = 2688
INTER = 1920


def bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    u32 = bits.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_flashinfer_floats() -> np.ndarray:
    raw = (ROOT / "bf16_gemm1_tactic1.bin").read_bytes()
    bits = np.frombuffer(raw[64:], dtype=np.uint16).reshape(NUM_ROWS, INTER)
    return bf16_bits_to_float(bits)


def bf16_bits_from_float(values: np.ndarray) -> np.ndarray:
    f32 = values.astype(np.float32)
    bits32 = f32.view(np.uint32)
    lsb = (bits32 >> 16) & 1
    rounding_bias = lsb + 0x7FFF
    rounded = (bits32.astype(np.uint64) + rounding_bias.astype(np.uint64)) & 0xFFFFFFFF
    return (rounded >> 16).astype(np.uint16)


def ulp_diff_scalar(a: int, b: int) -> int:
    def ordered(x: int) -> int:
        if (x >> 15) & 1:
            return -(x & 0x7FFF)
        return x
    return abs(ordered(a) - ordered(b))


def make_src_to_dst_perm(n: int) -> np.ndarray:
    """Inverse of the 32-row shuffle: destination index -> source index.
    srcToDstBlk32RowMap: (r%4)*8 + r//4 within each 32-row block.
    """
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        block = i // 32
        r = i % 32
        out[i] = block * 32 + (r % 4) * 8 + r // 4
    return out


def main() -> int:
    alpha = float(np.frombuffer((ROOT / "inputs_g1_alphas.bin").read_bytes(), dtype=np.float32)[0])
    print(f"alpha = {alpha!r}")

    flash_f = load_flashinfer_floats()
    print(f"flashinfer float shape = {flash_f.shape}")

    hs = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["hs"]
    w1_16 = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["w1_16"]
    print(f"hs  shape={tuple(hs.shape)}  dtype={hs.dtype}")
    print(f"w1_16 shape={tuple(w1_16.shape)}  dtype={w1_16.dtype}")

    hs64 = hs.to(torch.float64).numpy()
    w64 = w1_16[0].to(torch.float64).numpy()  # (1920, 2688) = (N, K)
    ref = alpha * (hs64 @ w64.T)               # (128, 1920)
    ref_bits = bf16_bits_from_float(ref)

    def as_bits(arr_f: np.ndarray) -> np.ndarray:
        return bf16_bits_from_float(arr_f)

    flash_bits = bf16_bits_from_float(flash_f)

    def match_count(a_bits: np.ndarray, b_bits: np.ndarray, tol: int = 1) -> int:
        a_ord = np.where((a_bits >> 15) & 1, -(a_bits & 0x7FFF).astype(np.int32), a_bits.astype(np.int32))
        b_ord = np.where((b_bits >> 15) & 1, -(b_bits & 0x7FFF).astype(np.int32), b_bits.astype(np.int32))
        return int(np.sum(np.abs(a_ord - b_ord) <= tol))

    total = NUM_ROWS * INTER

    def pct(n: int) -> str:
        return f"{n}/{total} ({100.0*n/total:.3f}%)"

    print(f"flash[0,:4] = {flash_f[0,:4]}")
    print(f"ref_fp64[0,:4]  = {ref[0,:4]}")
    print(f"ref_fp64[0,0] = {ref[0,0]!r}")
    print(f"flash[0,0]    = {flash_f[0,0]!r}")

    # Base orientation.
    print(f"ref vs flashinfer 1ulp  : {pct(match_count(ref_bits, flash_bits, 1))}")
    print(f"ref vs flashinfer 8ulp  : {pct(match_count(ref_bits, flash_bits, 8))}")
    print(f"ref vs flashinfer 64ulp : {pct(match_count(ref_bits, flash_bits, 64))}")
    print(f"ref vs flashinfer 512ulp: {pct(match_count(ref_bits, flash_bits, 512))}")

    # -------- Try applying the 32-row shuffle to hs (source->dst) --------
    perm = make_src_to_dst_perm(NUM_ROWS)
    print(f"perm[0..15] = {perm[:16]}")

    # shuffle hs by inverse srcToDst (reorder rows):
    # pattern 1: hs[perm[i]] -> i (so the i-th row of hs_perm is hs[perm[i]])
    hs_p = hs64[perm]
    ref_p = alpha * (hs_p @ w64.T)
    ref_p_bits = bf16_bits_from_float(ref_p)
    print(f"ref(hs[perm]) vs flashinfer 1ulp : {pct(match_count(ref_p_bits, flash_bits, 1))}")
    print(f"ref(hs[perm]) vs flashinfer 64ulp: {pct(match_count(ref_p_bits, flash_bits, 64))}")
    print(f"ref(hs[perm]) vs flashinfer 512ulp: {pct(match_count(ref_p_bits, flash_bits, 512))}")

    # pattern 2: output rows of ref are in shuffled order -> unshuffle flashinfer rows to compare.
    inv_perm = np.zeros_like(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    print(f"inv_perm[0..15] = {inv_perm[:16]}")
    flash_unshuffled = flash_f[inv_perm]
    flash_unshuf_bits = bf16_bits_from_float(flash_unshuffled)
    print(f"ref vs flashinfer[inv_perm rows] 1ulp : {pct(match_count(ref_bits, flash_unshuf_bits, 1))}")
    print(f"ref vs flashinfer[inv_perm rows] 64ulp: {pct(match_count(ref_bits, flash_unshuf_bits, 64))}")

    # -------- Try N (col) shuffle via the same pattern (with N=1920, blocks of 32) --------
    if INTER % 32 == 0:
        n_perm = make_src_to_dst_perm(INTER)
        # Shuffle weight rows by n_perm (corresponds to shuffling N).
        w_p = w64[n_perm]
        ref_npn = alpha * (hs64 @ w_p.T)
        ref_npn_bits = bf16_bits_from_float(ref_npn)
        print(f"ref(w[perm]) vs flashinfer 1ulp  : {pct(match_count(ref_npn_bits, flash_bits, 1))}")
        print(f"ref(w[perm]) vs flashinfer 64ulp : {pct(match_count(ref_npn_bits, flash_bits, 64))}")
        print(f"ref(w[perm]) vs flashinfer 512ulp: {pct(match_count(ref_npn_bits, flash_bits, 512))}")

    # -------- Search for flash[0,0] inside ref and vice versa --------
    target = flash_f[0, 0]
    # Find the closest position in `ref` to target.
    diff = np.abs(ref - target)
    near_idx = int(np.argmin(diff))
    nm, nn = near_idx // INTER, near_idx % INTER
    print(f"flash[0,0] = {target!r}; closest ref[m,n] is at ({nm},{nn}) value={ref[nm,nn]!r} abs_diff={diff[nm,nn]!r}")
    target2 = ref[0, 0]
    diff2 = np.abs(flash_f - target2)
    near_idx2 = int(np.argmin(diff2))
    fm, fn = near_idx2 // INTER, near_idx2 % INTER
    print(f"ref[0,0]   = {target2!r}; closest flash[m,n] is at ({fm},{fn}) value={flash_f[fm,fn]!r} abs_diff={diff2[fm,fn]!r}")

    # -------- Print small block comparisons --------
    print()
    print("flash[0,:8]  =", flash_f[0, :8])
    print("ref  [0,:8]  =", ref[0, :8])
    print("flash[8,:8]  =", flash_f[8, :8])
    print("ref  [8,:8]  =", ref[8, :8])
    print("flash[32,:8] =", flash_f[32, :8])
    print("ref  [32,:8] =", ref[32, :8])

    return 0


if __name__ == "__main__":
    sys.exit(main())
