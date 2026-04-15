"""Final cross-check: oracle_dequant vs hs@w1_16.T directly.

If the oracle's FP4 dequant is faithful to the original BF16 tensors, then
`(dq_input @ dq_weight.T)` should match `(hs @ w1_16.T) * w_gs` very
closely (within FP4 quantization noise). This cross-checks the dequant
path without depending on the flashinfer dump at all.

Also separately validates the ~49% vs-flashinfer match ceiling as an
FP4 precision floor, not an oracle bug, by testing with looser tolerances.
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
BLOCK_W = 16
SCALE_TILE_K = 4
SCALE_TILE_M = 128

FP4_DECODE_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float64,
)


def bf16_bits_from_float(values: np.ndarray) -> np.ndarray:
    f32 = values.astype(np.float32)
    bits32 = f32.view(np.uint32)
    lsb = (bits32 >> 16) & 1
    rounding_bias = lsb + 0x7FFF
    rounded = (bits32.astype(np.uint64) + rounding_bias.astype(np.uint64)) & 0xFFFFFFFF
    return (rounded >> 16).astype(np.uint16)


def ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def ordered(x: np.ndarray) -> np.ndarray:
        sign = (x >> 15) & 1
        return np.where(sign == 1, -(x & 0x7FFF).astype(np.int32), x.astype(np.int32))
    return np.abs(ordered(a.astype(np.int32)) - ordered(b.astype(np.int32)))


def decode_fp8_e4m3(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.uint32)
    sign = (raw >> 7) & 0x1
    exp = (raw >> 3) & 0xF
    mant = raw & 0x7
    value_normal = (1.0 + mant.astype(np.float64) / 8.0) * np.power(2.0, (exp.astype(np.float64) - 7.0))
    value_subnormal = (mant.astype(np.float64) / 8.0) * np.power(2.0, -6.0)
    result = np.where(exp != 0, value_normal, np.where(mant != 0, value_subnormal, 0.0))
    sign_mask = np.where(sign == 1, -1.0, 1.0)
    nan_mask = (raw == 0x7F) | (raw == 0xFF)
    return np.where(nan_mask, np.nan, result * sign_mask)


def execution_scale_offset(row: int, block_col: int, padded_blocks_per_row: int) -> int:
    num_k_tiles = padded_blocks_per_row // SCALE_TILE_K
    k_tile = block_col // SCALE_TILE_K
    inner_k = block_col & 3
    m_tile = row // SCALE_TILE_M
    outer_m = row & 31
    inner_m = (row >> 5) & 3
    return (((m_tile * num_k_tiles) + k_tile) << 9) | (outer_m << 4) | (inner_m << 2) | inner_k


def dequantize_execution_nvfp4(
    packed: np.ndarray,
    execution_scales: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    logical_blocks_per_row = cols // BLOCK_W
    padded_blocks_per_row = ((logical_blocks_per_row + 3) // 4) * 4
    out = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        for block in range(logical_blocks_per_row):
            scale_off = execution_scale_offset(row, block, padded_blocks_per_row)
            scale_byte = execution_scales[scale_off]
            scale = decode_fp8_e4m3(np.array([scale_byte], dtype=np.uint8))[0]
            col_start = block * BLOCK_W
            for offset in range(0, BLOCK_W, 2):
                byte_idx = row * (cols // 2) + (col_start + offset) // 2
                byte = packed[byte_idx]
                out[row, col_start + offset + 0] = FP4_DECODE_LUT[byte & 0x0F] * scale
                out[row, col_start + offset + 1] = FP4_DECODE_LUT[(byte >> 4) & 0x0F] * scale
    return out


def match_summary(label: str, lhs_bits: np.ndarray, rhs_bits: np.ndarray) -> None:
    d = ulp_diff(lhs_bits.reshape(-1), rhs_bits.reshape(-1))
    total = d.size
    bitwise = int(np.sum(d == 0))
    within_1 = int(np.sum(d <= 1))
    within_8 = int(np.sum(d <= 8))
    within_64 = int(np.sum(d <= 64))
    within_512 = int(np.sum(d <= 512))
    within_2k = int(np.sum(d <= 2048))
    max_ulp = int(np.max(d)) if total else 0
    print(
        f"{label}:\n"
        f"    bitwise={bitwise}/{total} ({100.0*bitwise/total:.3f}%)\n"
        f"    <=1u={within_1} ({100.0*within_1/total:.3f}%)\n"
        f"    <=8u={within_8} ({100.0*within_8/total:.3f}%)\n"
        f"    <=64u={within_64} ({100.0*within_64/total:.3f}%)\n"
        f"    <=512u={within_512} ({100.0*within_512/total:.3f}%)\n"
        f"    <=2048u={within_2k} ({100.0*within_2k/total:.3f}%)\n"
        f"    max={max_ulp}"
    )


def main() -> int:
    alpha = float(np.frombuffer((ROOT / "inputs_g1_alphas.bin").read_bytes(), dtype=np.float32)[0])
    w1_gs = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["w1_gs"].item()
    print(f"alpha  = {alpha!r}")
    print(f"w1_gs  = {w1_gs!r}")
    print(f"|alpha*w1_gs - 1| = {abs(alpha*w1_gs - 1)!r}")

    flash_bits = np.frombuffer((ROOT / "bf16_gemm1_tactic1.bin").read_bytes(), dtype=np.uint8)[64:]
    flash_bits = np.frombuffer(flash_bits.tobytes(), dtype=np.uint16).reshape(NUM_ROWS, INTER)

    hs = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["hs"]
    w1_16 = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["w1_16"]
    hs64 = hs.to(torch.float64).numpy()
    w64 = w1_16[0].to(torch.float64).numpy()

    # BF16 originals, no alpha
    bf16_dot = hs64 @ w64.T
    print("\n=== Baseline: bf16 originals dot product vs flashinfer ===")
    match_summary("bf16_dot vs flashinfer (no alpha)", bf16_bits_from_float(bf16_dot), flash_bits)

    # Load and dequantize FP4
    input_fp4 = np.frombuffer((ROOT / "input_fp4_permuted.bin").read_bytes(), dtype=np.uint8)
    input_sf = np.frombuffer((ROOT / "input_sf_permuted.bin").read_bytes(), dtype=np.uint8)
    weight_fp4 = np.frombuffer((ROOT / "inputs_w1_fp4.bin").read_bytes(), dtype=np.uint8)
    weight_sf = np.frombuffer((ROOT / "inputs_w1_sf.bin").read_bytes(), dtype=np.uint8)
    print("\ndequantizing input...")
    dq_input = dequantize_execution_nvfp4(input_fp4, input_sf, NUM_ROWS, HIDDEN)
    print("dequantizing weight...")
    dq_weight = dequantize_execution_nvfp4(weight_fp4, weight_sf, INTER, HIDDEN)

    # Rescaled oracle
    oracle = dq_input @ dq_weight.T           # at (w_gs * bf16) scale, per hypothesis
    oracle_scaled = oracle * alpha            # at bf16 scale

    # Cross-check 1: oracle vs bf16_dot (ignoring flashinfer). Direct test of
    # whether FP4 dequantize+accumulate faithfully reproduces the original
    # bf16 dot product at FP4 precision.
    print("\n=== Cross-check: oracle (rescaled) vs bf16 originals dot product ===")
    match_summary("oracle*alpha vs bf16_dot (hs@w1.T)",
                  bf16_bits_from_float(oracle_scaled), bf16_bits_from_float(bf16_dot))

    # Cross-check 2: oracle / w_gs vs bf16_dot (alternative scaling).
    match_summary("oracle/w_gs vs bf16_dot",
                  bf16_bits_from_float(oracle / w1_gs), bf16_bits_from_float(bf16_dot))

    # Cross-check 3: the first-row dequantized activations should match hs BUT
    # in shuffled order. Compare dq_input[0, :8] with hs[perm_applied, :8] to
    # verify the dequant is reading something that makes sense.
    print(f"\ndq_input[0, :8]  = {dq_input[0, :8]}")
    print(f"hs[0, :8]         = {hs64[0, :8]}")
    # Since input_fp4_permuted is shuffled, dq_input[0] should correspond to
    # some permuted row of hs. Try standard srcToDstBlk32Row:
    for candidate in [0, 1, 2, 3, 4, 8, 16, 24]:
        d = np.abs(dq_input[0] - hs64[candidate]).max()
        print(f"max|dq_input[0] - hs[{candidate}]| = {d!r}")

    # -------- Weight-side check --------
    print(f"\ndq_weight[0, :8]  = {dq_weight[0, :8]}")
    print(f"w1_16[0, 0, :8]   = {w64[0, :8]}")
    print(f"w1_16[0, 0, :8] * w1_gs = {w64[0, :8] * w1_gs}")
    for candidate in [0, 1, 2, 3, 4, 8, 16, 24]:
        d = np.abs(dq_weight[0] - w64[candidate] * w1_gs).max()
        print(f"max|dq_weight[0] - w1_16[{candidate}]*w_gs| = {d!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
