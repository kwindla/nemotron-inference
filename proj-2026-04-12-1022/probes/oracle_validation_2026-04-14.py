"""Cheap validation of the local-math oracle vs flashinfer reference.

Uses only the already-captured Nano K=2688 inputs under
`proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/`. No new GPU run,
no new capture. The goal is to answer the question "does a Python
re-implementation of the oracle's math agree with the flashinfer dump?"
separately from "does the runtime kernel agree with the oracle?".

Three comparisons are emitted:

  A. FP16-level dot product on the pre-quantization bf16 originals vs the
     flashinfer dump. This tells us whether the M/N orientation and alpha
     application match flashinfer at all, independent of FP4 quantization.

  B. Dequant-and-accumulate via a direct Python reimplementation of the
     oracle's formulas on the saved FP4 bytes + swizzled 128x4 scales vs
     the flashinfer dump. This is the exact check the C++ oracle performs,
     in a faster-to-iterate language.

  C. The same Python dequant with the nibble pack order FLIPPED vs the
     flashinfer dump. This is the single-variable toggle for the "nibble
     high/low packing" candidate.

Match rate thresholds interpreted as:

  >= 80% : approximately correct (off-by-one in low-magnitude positions OK)
  40-80% : partial / some systematic error
  < 15%  : structural wrongness (incompatible layouts or formulas)
"""

from __future__ import annotations

import numpy as np
import pathlib
import struct
import sys

import torch


ROOT = pathlib.Path(
    "/home/khkramer/src/nemotron-inference/proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688"
)

NUM_ROWS = 128
HIDDEN = 2688
INTER = 1920
BLOCK_W = 16
SCALE_TILE_M = 128
SCALE_TILE_K = 4

# NVFP4 E2M1 value lookup: 4-bit code -> float
# E2M1 is 1 sign bit + 2 exp bits + 1 mantissa bit, bias=1. Values:
#   0x0 = +0.0, 0x1 = +0.5, 0x2 = +1.0, 0x3 = +1.5,
#   0x4 = +2.0, 0x5 = +3.0, 0x6 = +4.0, 0x7 = +6.0,
#   0x8 = -0.0, 0x9 = -0.5, 0xA = -1.0, 0xB = -1.5,
#   0xC = -2.0, 0xD = -3.0, 0xE = -4.0, 0xF = -6.0
FP4_DECODE_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float64,
)


def decode_fp8_e4m3(raw: np.ndarray) -> np.ndarray:
    """Decode uint8 bytes as FP8 E4M3 into float64. Vectorized. """
    raw = raw.astype(np.uint32)
    sign = (raw >> 7) & 0x1
    exp = (raw >> 3) & 0xF
    mant = raw & 0x7
    # E4M3 format: 1 sign + 4 exp + 3 mant, bias=7, no infinity, NaN is 0xff/0x7f.
    # Normal: value = (-1)^sign * 2^(exp-7) * (1 + mant/8)
    # Subnormal (exp=0): value = (-1)^sign * 2^(-6) * (mant/8)
    # NaN: 0x7f, 0xff
    result = np.zeros_like(raw, dtype=np.float64)
    normal = exp != 0
    subnormal = (exp == 0) & (mant != 0)
    value_normal = (1.0 + mant.astype(np.float64) / 8.0) * np.power(
        2.0, (exp.astype(np.float64) - 7.0)
    )
    value_subnormal = (mant.astype(np.float64) / 8.0) * np.power(2.0, -6.0)
    result = np.where(normal, value_normal, result)
    result = np.where(subnormal, value_subnormal, result)
    sign_mask = np.where(sign == 1, -1.0, 1.0)
    # NaN convention (E4M3): 0x7f or 0xff
    nan_mask = (raw == 0x7F) | (raw == 0xFF)
    result = np.where(nan_mask, np.nan, result * sign_mask)
    return result


def execution_scale_offset(row: int, block_col: int, padded_blocks_per_row: int) -> int:
    """Matches both the runtime's kSwizzled128x4 and the C++ oracle's formula."""
    num_k_tiles = padded_blocks_per_row // SCALE_TILE_K
    k_tile = block_col // SCALE_TILE_K
    inner_k = block_col & 3
    m_tile = row // SCALE_TILE_M
    outer_m = row & 31
    inner_m = (row >> 5) & 3
    return (((m_tile * num_k_tiles) + k_tile) << 9) | (outer_m << 4) | (inner_m << 2) | inner_k


def dequantize_execution_nvfp4(
    packed: np.ndarray,          # shape (rows * cols/2,) uint8 — row-major K-contiguous
    execution_scales: np.ndarray, # shape (rows * padded_blocks_per_row,) uint8
    rows: int,
    cols: int,
    flip_nibble_order: bool = False,
) -> np.ndarray:
    """Reimplements the C++ oracle's DequantizeExecutionLayoutNvfp4Matrix in numpy.

    Returns float64 matrix of shape (rows, cols).
    """
    logical_blocks_per_row = cols // BLOCK_W
    padded_blocks_per_row = ((logical_blocks_per_row + 3) // 4) * 4
    out = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        for block in range(logical_blocks_per_row):
            scale_off = execution_scale_offset(row, block, padded_blocks_per_row)
            scale_byte = execution_scales[scale_off]
            scale = decode_fp8_e4m3(np.array([scale_byte], dtype=np.uint8))[0]
            col_start = block * BLOCK_W
            # Each block is BLOCK_W=16 FP4 values = 8 packed bytes.
            for offset in range(0, BLOCK_W, 2):
                byte_idx = row * (cols // 2) + (col_start + offset) // 2
                byte = packed[byte_idx]
                low_nibble = byte & 0x0F
                high_nibble = (byte >> 4) & 0x0F
                if flip_nibble_order:
                    out[row, col_start + offset + 0] = FP4_DECODE_LUT[high_nibble] * scale
                    out[row, col_start + offset + 1] = FP4_DECODE_LUT[low_nibble] * scale
                else:
                    out[row, col_start + offset + 0] = FP4_DECODE_LUT[low_nibble] * scale
                    out[row, col_start + offset + 1] = FP4_DECODE_LUT[high_nibble] * scale
    return out


def read_bytes(path: pathlib.Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=np.uint8)


def read_bf16_file(path: pathlib.Path, expected_elems: int) -> np.ndarray:
    raw = path.read_bytes()
    # Dump header is 5 x u32 = 20 bytes for version/rows/cols/elem_bytes/is_gated,
    # but inputs_bin_manifest says "bf16_header_bytes": 64 (per metadata.json).
    # Header layout: first u32 is version. Body = rows*cols*2 bytes. Try both.
    if len(raw) == 64 + expected_elems * 2:
        body = raw[64:]
    elif len(raw) == 20 + expected_elems * 2:
        body = raw[20:]
    elif len(raw) == expected_elems * 2:
        body = raw
    else:
        raise ValueError(f"unexpected file size {len(raw)} for {path}")
    return np.frombuffer(body, dtype=np.uint16)


def load_flashinfer_dump(tactic_id: int) -> np.ndarray:
    path = ROOT / f"bf16_gemm1_tactic{tactic_id}.bin"
    # Read u16 bits; total 491584 bytes = 64 header + 128*1920*2 body = 491584. Confirmed.
    raw = path.read_bytes()
    body = raw[64:]
    bits = np.frombuffer(body, dtype=np.uint16)
    assert bits.size == NUM_ROWS * INTER, f"size mismatch: {bits.size} vs {NUM_ROWS*INTER}"
    return bits.reshape(NUM_ROWS, INTER)


def bf16_bits_from_float_np(values: np.ndarray) -> np.ndarray:
    """RTNE convert float64 -> bf16 bits (uint16), matching __float2bfloat16."""
    f32 = values.astype(np.float32)
    bits32 = f32.view(np.uint32)
    # RTNE rounding: add (((bits32 >> 16) & 1) + 0x7FFF) and shift right 16.
    lsb = (bits32 >> 16) & 1
    rounding_bias = lsb + 0x7FFF
    rounded = (bits32.astype(np.uint64) + rounding_bias.astype(np.uint64)) & 0xFFFFFFFF
    return (rounded >> 16).astype(np.uint16)


def ulp_diff(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Bf16 ULP difference via ordered-int conversion."""
    def ordered(x: np.ndarray) -> np.ndarray:
        sign = (x >> 15) & 1
        neg = np.where(sign == 1, -(x & 0x7FFF).astype(np.int32), x.astype(np.int32))
        return neg
    a = ordered(lhs.astype(np.int32))
    b = ordered(rhs.astype(np.int32))
    return np.abs(a - b).astype(np.int32)


def load_pt_tensor(key: str) -> torch.Tensor:
    return torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")[key]


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
        f"<=1ulp={within_1} ({100.0*within_1/total:.3f}%) "
        f"<=8ulp={within_8} ({100.0*within_8/total:.3f}%) "
        f"<=64ulp={within_64} ({100.0*within_64/total:.3f}%) "
        f"<=512ulp={within_512} ({100.0*within_512/total:.3f}%) "
        f"max={max_ulp}"
    )


def main() -> int:
    if not ROOT.exists():
        print(f"missing root dir: {ROOT}", file=sys.stderr)
        return 1

    alpha = float(np.frombuffer((ROOT / "inputs_g1_alphas.bin").read_bytes(), dtype=np.float32)[0])
    print(f"alpha = {alpha!r}")

    flash_bits = load_flashinfer_dump(1)  # tactic 1 = P1
    print(f"flashinfer dump shape = {flash_bits.shape}")

    # Cross-tactic convergence check (NOTES.md §5 claim).
    tac0_bits = load_flashinfer_dump(0)
    match_summary("tactic0 vs tactic1 bitwise", tac0_bits, flash_bits)

    # -------- Comparison A: FP16-level GEMM ----------
    # Treat hs and w1_16 as the ground-truth pre-quantization values.
    # Compute output[m, n] = alpha * sum_k hs[m, k] * w1_16[0, n, k]
    hs = load_pt_tensor("hs")               # shape (128, 2688) bf16
    w1_16 = load_pt_tensor("w1_16")         # shape (1, 1920, 2688) bf16
    print(f"hs shape={tuple(hs.shape)} dtype={hs.dtype}")
    print(f"w1_16 shape={tuple(w1_16.shape)} dtype={w1_16.dtype}")

    hs64 = hs.to(torch.float64).numpy()
    w64 = w1_16[0].to(torch.float64).numpy()
    out_fp16_level = alpha * (hs64 @ w64.T)
    out_fp16_level_bits = bf16_bits_from_float_np(out_fp16_level)
    match_summary("A. FP16-level hs@w1_16.T * alpha vs flashinfer", out_fp16_level_bits, flash_bits)

    # -------- Comparison B: oracle-style dequant from saved FP4 bytes ----------
    input_fp4 = read_bytes(ROOT / "input_fp4_permuted.bin")
    input_sf = read_bytes(ROOT / "input_sf_permuted.bin")
    weight_fp4 = read_bytes(ROOT / "inputs_w1_fp4.bin")
    weight_sf = read_bytes(ROOT / "inputs_w1_sf.bin")
    print(f"input_fp4 bytes={input_fp4.size}")
    print(f"input_sf bytes={input_sf.size}")
    print(f"weight_fp4 bytes={weight_fp4.size}")
    print(f"weight_sf bytes={weight_sf.size}")

    print("dequantizing input (oracle nibble order)...", flush=True)
    dq_input = dequantize_execution_nvfp4(input_fp4, input_sf, NUM_ROWS, HIDDEN, flip_nibble_order=False)
    print("dequantizing weights (oracle nibble order)...", flush=True)
    dq_weight = dequantize_execution_nvfp4(weight_fp4, weight_sf, INTER, HIDDEN, flip_nibble_order=False)

    out_oracle = alpha * (dq_input @ dq_weight.T)
    out_oracle_bits = bf16_bits_from_float_np(out_oracle)
    match_summary("B. oracle dequant(FP4) * alpha vs flashinfer", out_oracle_bits, flash_bits)
    match_summary("B vs A (oracle vs fp16-level)", out_oracle_bits, out_fp16_level_bits)

    # -------- Comparison C: nibble order FLIPPED ----------
    print("dequantizing input (FLIPPED nibble order)...", flush=True)
    dq_input_f = dequantize_execution_nvfp4(input_fp4, input_sf, NUM_ROWS, HIDDEN, flip_nibble_order=True)
    print("dequantizing weights (FLIPPED nibble order)...", flush=True)
    dq_weight_f = dequantize_execution_nvfp4(weight_fp4, weight_sf, INTER, HIDDEN, flip_nibble_order=True)

    out_oracle_f = alpha * (dq_input_f @ dq_weight_f.T)
    out_oracle_f_bits = bf16_bits_from_float_np(out_oracle_f)
    match_summary("C. oracle dequant(FP4)[FLIPPED] * alpha vs flashinfer", out_oracle_f_bits, flash_bits)

    return 0


if __name__ == "__main__":
    sys.exit(main())
