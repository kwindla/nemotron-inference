#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def swap_nibbles(value: int) -> int:
    return ((value & 0xF) << 4) | (value >> 4)


def dup_lo(value: int) -> int:
    lo = value & 0xF
    return (lo << 4) | lo


def dup_hi(value: int) -> int:
    hi = value >> 4
    return (hi << 4) | hi


def parse_live_byte(copy_sig: str) -> int:
    first = copy_sig.split(",", 1)[0]
    _, byte_value = first.split(":", 1)
    return int(byte_value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether live activation-stage bytes on the real-input compare "
            "match the dumped gemm1_input bytes under simple coordinate or byte transforms."
        )
    )
    parser.add_argument("--compare-json", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--row-bytes", type=int, default=1344)
    parser.add_argument("--search-row-radius", type=int, default=4)
    parser.add_argument("--search-byte-radius", type=int, default=8)
    args = parser.parse_args()

    compare = json.loads(args.compare_json.read_text())
    input_bytes = args.input_file.read_bytes()
    samples = compare.get("samples", [])
    row_count = len(input_bytes) // args.row_bytes

    transforms = {
        "id": lambda value: value,
        "swap_nibbles": swap_nibbles,
        "dup_lo": dup_lo,
        "dup_hi": dup_hi,
        "xor_ff": lambda value: value ^ 0xFF,
        "xor_55": lambda value: value ^ 0x55,
        "xor_aa": lambda value: value ^ 0xAA,
    }
    nibble_pairs = {
        "lo_i_hi_i+1": lambda a, b: (a & 0xF) | ((b >> 4) << 4),
        "hi_i_lo_i+1": lambda a, b: ((a >> 4) & 0xF) | ((b & 0xF) << 4),
        "lo_i_lo_i+1": lambda a, b: (a & 0xF) | ((b & 0xF) << 4),
        "hi_i_hi_i+1": lambda a, b: ((a >> 4) & 0xF) | (((b >> 4) & 0xF) << 4),
        "lo_i+1_hi_i": lambda a, b: (b & 0xF) | ((a >> 4) << 4),
        "hi_i+1_lo_i": lambda a, b: ((b >> 4) & 0xF) | ((a & 0xF) << 4),
    }

    transform_matches = {name: 0 for name in transforms}
    neighbor_matches = {delta: 0 for delta in range(-args.search_byte_radius, args.search_byte_radius + 1)}
    nibble_pair_matches = {name: 0 for name in nibble_pairs}
    sample_rows: list[str] = []

    for sample in samples:
        live = sample["live"]
        row, col = live["local0"]
        byte_index = col // 2
        live_byte = parse_live_byte(live["copy_sig"])
        file_offset = row * args.row_bytes + byte_index
        file_byte = input_bytes[file_offset]

        for name, transform in transforms.items():
            if transform(file_byte) == live_byte:
                transform_matches[name] += 1

        for delta in neighbor_matches:
            neighbor_byte_index = byte_index + delta
            if 0 <= neighbor_byte_index < args.row_bytes:
                if input_bytes[row * args.row_bytes + neighbor_byte_index] == live_byte:
                    neighbor_matches[delta] += 1

        next_byte_index = min(byte_index + 1, args.row_bytes - 1)
        next_file_byte = input_bytes[row * args.row_bytes + next_byte_index]
        for name, transform in nibble_pairs.items():
            if transform(file_byte, next_file_byte) == live_byte:
                nibble_pair_matches[name] += 1

        nearby_hits: list[tuple[int, int]] = []
        for row_delta in range(-args.search_row_radius, args.search_row_radius + 1):
            neighbor_row = row + row_delta
            if not (0 <= neighbor_row < row_count):
                continue
            for byte_delta in range(-args.search_byte_radius, args.search_byte_radius + 1):
                neighbor_byte_index = byte_index + byte_delta
                if not (0 <= neighbor_byte_index < args.row_bytes):
                    continue
                if input_bytes[neighbor_row * args.row_bytes + neighbor_byte_index] == live_byte:
                    nearby_hits.append((row_delta, byte_delta))
        sample_rows.append(
            "key={} coord=({}, {}) stage_offset={} file_byte={} live_byte={} nearby_hits={}".format(
                tuple(sample["key"]),
                row,
                byte_index,
                live["stage0_offsets"][0],
                file_byte,
                live_byte,
                nearby_hits[:8],
            )
        )

    print(f"samples={len(samples)} row_bytes={args.row_bytes}")
    print("simple_transform_matches:")
    for name, count in transform_matches.items():
        print(f"  {name}: {count}")
    print("same-row neighbor-byte matches:")
    for delta, count in neighbor_matches.items():
        print(f"  byte_delta={delta:+d}: {count}")
    print("adjacent-byte nibble-pair matches:")
    for name, count in nibble_pair_matches.items():
        print(f"  {name}: {count}")
    print("sample_rows:")
    for row in sample_rows:
        print(f"  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
