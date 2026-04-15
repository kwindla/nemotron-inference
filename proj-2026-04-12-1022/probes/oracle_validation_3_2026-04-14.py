"""Third validation step: test the 'alpha should not be applied' hypothesis.

oracle_validation_2_2026-04-14.py showed that alpha*hs@w1.T is ~6030x
smaller than flash[0,0]. alpha = 1.315e-4, 1/alpha ≈ 7604 — same order.
Hypothesis: flashinfer's gemm1 output (the dump) is PRE-alpha, so the
oracle's post-hoc alpha multiplication is inverting the scale.

Run the same comparisons with alpha removed from the reference. Should
match within ~FP4 quantization noise (~50-500 BF16 ULP).
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


def load_flashinfer_bits() -> np.ndarray:
    raw = (ROOT / "bf16_gemm1_tactic1.bin").read_bytes()
    return np.frombuffer(raw[64:], dtype=np.uint16).reshape(NUM_ROWS, INTER)


def load_flashinfer_floats() -> np.ndarray:
    bits = load_flashinfer_bits()
    return bits.astype(np.uint32).__lshift__(16).view(np.float32)


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
    max_ulp = int(np.max(d)) if total else 0
    print(
        f"{label}: bitwise={bitwise}/{total} ({100.0*bitwise/total:.3f}%) "
        f"<=1u={within_1} ({100.0*within_1/total:.3f}%) "
        f"<=8u={within_8} ({100.0*within_8/total:.3f}%) "
        f"<=64u={within_64} ({100.0*within_64/total:.3f}%) "
        f"<=512u={within_512} ({100.0*within_512/total:.3f}%) "
        f"max={max_ulp}"
    )


def main() -> int:
    alpha = float(np.frombuffer((ROOT / "inputs_g1_alphas.bin").read_bytes(), dtype=np.float32)[0])
    print(f"alpha = {alpha!r}")

    flash_bits = load_flashinfer_bits()
    print(f"flashinfer bits shape = {flash_bits.shape}")

    hs = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["hs"]
    w1_16 = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["w1_16"]
    hs64 = hs.to(torch.float64).numpy()
    w64 = w1_16[0].to(torch.float64).numpy()  # (1920, 2688)

    # No-alpha reference.
    ref_no_alpha = hs64 @ w64.T
    print(f"ref_no_alpha[0,:4] = {ref_no_alpha[0,:4]}")

    ref_no_alpha_bits = bf16_bits_from_float(ref_no_alpha)
    match_summary("FP16-level hs@w1.T (NO alpha)", ref_no_alpha_bits, flash_bits)

    # With alpha for comparison.
    ref_with_alpha = alpha * ref_no_alpha
    ref_with_alpha_bits = bf16_bits_from_float(ref_with_alpha)
    match_summary("FP16-level hs@w1.T * alpha   ", ref_with_alpha_bits, flash_bits)

    # With 1/alpha.
    ref_inv_alpha = ref_no_alpha / alpha
    ref_inv_alpha_bits = bf16_bits_from_float(ref_inv_alpha)
    match_summary("FP16-level hs@w1.T / alpha   ", ref_inv_alpha_bits, flash_bits)

    # Also test w_gs (global scale) = 1/alpha.
    print(f"1/alpha = {1.0/alpha!r}")

    # -------- Now oracle FP4 path with no alpha --------
    print("dequantizing input...")
    input_fp4 = np.frombuffer((ROOT / "input_fp4_permuted.bin").read_bytes(), dtype=np.uint8)
    input_sf = np.frombuffer((ROOT / "input_sf_permuted.bin").read_bytes(), dtype=np.uint8)
    weight_fp4 = np.frombuffer((ROOT / "inputs_w1_fp4.bin").read_bytes(), dtype=np.uint8)
    weight_sf = np.frombuffer((ROOT / "inputs_w1_sf.bin").read_bytes(), dtype=np.uint8)
    dq_input = dequantize_execution_nvfp4(input_fp4, input_sf, NUM_ROWS, HIDDEN)
    print("dequantizing weights...")
    dq_weight = dequantize_execution_nvfp4(weight_fp4, weight_sf, INTER, HIDDEN)
    oracle_no_alpha = dq_input @ dq_weight.T
    oracle_with_alpha = alpha * oracle_no_alpha
    oracle_inv_alpha = oracle_no_alpha / alpha
    match_summary("oracle dq@dq (NO alpha)  ", bf16_bits_from_float(oracle_no_alpha), flash_bits)
    match_summary("oracle dq@dq * alpha      ", bf16_bits_from_float(oracle_with_alpha), flash_bits)
    match_summary("oracle dq@dq / alpha      ", bf16_bits_from_float(oracle_inv_alpha), flash_bits)

    # Also check the scale applied to FP4 (alpha already absorbed during dequant via w_gs).
    # Look up w1_gs from inputs.pt for completeness.
    w1_gs = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["w1_gs"].item()
    print(f"w1_gs = {w1_gs!r}; alpha (from disk) = {alpha!r}; 1/w1_gs = {1.0/w1_gs!r}")

    # Direct comparison at (0,0).
    print(f"\nflash[0,0]                = {flash_bits[0,0]:#06x}  "
          f"value={(flash_bits[0,0].astype(np.uint32)<<16).view(np.float32)}")
    print(f"ref_no_alpha[0,0]         = {ref_no_alpha[0,0]!r}")
    print(f"ref_with_alpha[0,0]       = {ref_with_alpha[0,0]!r}")
    print(f"oracle_no_alpha[0,0]      = {oracle_no_alpha[0,0]!r}")
    print(f"oracle_with_alpha[0,0]    = {oracle_with_alpha[0,0]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
