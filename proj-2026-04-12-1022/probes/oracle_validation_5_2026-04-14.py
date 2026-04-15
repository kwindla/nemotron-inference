"""Fifth validation: is `input_fp4_permuted.bin` actually row-permuted?

The salvage notes claim that `do_shuffle=True` in flashinfer nvfp4_quantize
applies a 32-row srcToDstBlk32RowMap permutation. The runtime kernel
inverts this permutation when staging A. Validation 4 suggested that
dq_input[0] is closest to hs[0], not to any permuted alternative, which
would mean the row permutation claim is WRONG. Pin this down decisively.

For k=0..31, measure the L-infinity error between dq_input[k] and hs[k']
for k' ∈ {identity, srcToDst, inv_srcToDst}. The one that consistently
produces ~FP4 precision error tells us the true layout.
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
BLOCK_W = 16
SCALE_TILE_K = 4
SCALE_TILE_M = 128

FP4_DECODE_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float64,
)


def decode_fp8_e4m3_scalar(raw: int) -> float:
    sign = (raw >> 7) & 0x1
    exp = (raw >> 3) & 0xF
    mant = raw & 0x7
    if raw == 0x7F or raw == 0xFF:
        return float("nan")
    if exp == 0:
        if mant == 0:
            return 0.0
        value = (mant / 8.0) * (2 ** -6)
    else:
        value = (1.0 + mant / 8.0) * (2 ** (exp - 7))
    return -value if sign else value


def execution_scale_offset(row: int, block_col: int, padded_blocks_per_row: int) -> int:
    num_k_tiles = padded_blocks_per_row // SCALE_TILE_K
    k_tile = block_col // SCALE_TILE_K
    inner_k = block_col & 3
    m_tile = row // SCALE_TILE_M
    outer_m = row & 31
    inner_m = (row >> 5) & 3
    return (((m_tile * num_k_tiles) + k_tile) << 9) | (outer_m << 4) | (inner_m << 2) | inner_k


def dequantize_row(
    row: int,
    packed: np.ndarray,
    execution_scales: np.ndarray,
    hidden: int,
) -> np.ndarray:
    logical_blocks = hidden // BLOCK_W
    padded_blocks = ((logical_blocks + 3) // 4) * 4
    out = np.zeros(hidden, dtype=np.float64)
    for block in range(logical_blocks):
        scale_off = execution_scale_offset(row, block, padded_blocks)
        scale = decode_fp8_e4m3_scalar(int(execution_scales[scale_off]))
        col_start = block * BLOCK_W
        for offset in range(0, BLOCK_W, 2):
            byte_idx = row * (hidden // 2) + (col_start + offset) // 2
            byte = int(packed[byte_idx])
            out[col_start + offset + 0] = FP4_DECODE_LUT[byte & 0x0F] * scale
            out[col_start + offset + 1] = FP4_DECODE_LUT[(byte >> 4) & 0x0F] * scale
    return out


def src_to_dst_32(r: int) -> int:
    """Forward map from salvage notes: r → (r%4)*8 + r//4 within each 32-block."""
    blk = r // 32
    inner = r % 32
    return blk * 32 + (inner % 4) * 8 + inner // 4


def inv_src_to_dst_32(r: int) -> int:
    """Inverse: (r%8)*4 + r//8 within each 32-block."""
    blk = r // 32
    inner = r % 32
    return blk * 32 + (inner % 8) * 4 + inner // 8


def main() -> int:
    hs = torch.load(ROOT / "inputs.pt", weights_only=False, map_location="cpu")["hs"]
    hs64 = hs.to(torch.float64).numpy()

    input_fp4 = np.frombuffer((ROOT / "input_fp4_permuted.bin").read_bytes(), dtype=np.uint8)
    input_sf = np.frombuffer((ROOT / "input_sf_permuted.bin").read_bytes(), dtype=np.uint8)

    print("row    hs_magnitude_max    Linf_identity   Linf_srcToDst     Linf_invSrcToDst")
    for k in range(32):
        dq_k = dequantize_row(k, input_fp4, input_sf, HIDDEN)
        id_err = float(np.max(np.abs(dq_k - hs64[k])))
        std_k = src_to_dst_32(k)
        inv_k = inv_src_to_dst_32(k)
        std_err = float(np.max(np.abs(dq_k - hs64[std_k])))
        inv_err = float(np.max(np.abs(dq_k - hs64[inv_k])))
        hs_mag = float(np.max(np.abs(hs64[k])))
        print(
            f"{k:3d}    {hs_mag:12.6f}    "
            f"{id_err:12.6f} ({k:3d})    "
            f"{std_err:12.6f} ({std_k:3d})    "
            f"{inv_err:12.6f} ({inv_k:3d})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
