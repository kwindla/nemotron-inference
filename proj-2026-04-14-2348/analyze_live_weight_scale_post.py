#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


LIVE_ENTRY_RE = re.compile(
    r"entry=\d+\s+tid=(?P<tid>\d+)\s+.*?"
    r"n_tile=(?P<n_tile>\d+)\s+"
    r"k_block=(?P<k_block>\d+)\s+.*?"
    r"part_c0=\((?P<part_output_col0>\d+),(?P<part_token_row0>\d+)\)\s+.*?"
    r"scale_reg_post=\((?P<scale_reg_post>0x[0-9a-f]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LiveEntry:
    tid: int
    n_tile: int
    k_block: int
    part_output_col0: int
    part_token_row0: int
    scale_reg_post: int


def execution_scale_offset(row: int, block_col: int, padded_blocks_per_row: int) -> int:
    num_k_tiles = padded_blocks_per_row // 4
    k_tile = block_col // 4
    inner_k = block_col & 3
    m_tile = row // 128
    outer_m = row & 31
    inner_m = (row >> 5) & 3
    return ((((m_tile * num_k_tiles) + k_tile) << 9) |
            (outer_m << 4) |
            (inner_m << 2) |
            inner_k)


def parse_live_entries(path: Path) -> list[LiveEntry]:
    entries: list[LiveEntry] = []
    for line in path.read_text().splitlines():
        match = LIVE_ENTRY_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        entries.append(
            LiveEntry(
                tid=int(groups["tid"]),
                n_tile=int(groups["n_tile"]),
                k_block=int(groups["k_block"]),
                part_output_col0=int(groups["part_output_col0"]),
                part_token_row0=int(groups["part_token_row0"]),
                scale_reg_post=int(groups["scale_reg_post"], 16),
            )
        )
    return entries


def load_scale_word(blob: bytes, row: int, block_base: int, padded_blocks_per_row: int) -> int:
    raw = bytes(
        blob[execution_scale_offset(row, block_base + block, padded_blocks_per_row)]
        for block in range(4)
    )
    return int.from_bytes(raw, "little")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze live tactic-5 weight scale_reg_post words.")
    parser.add_argument("--live-probe", type=Path, required=True)
    parser.add_argument("--scale-file", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--weight-rows", type=int, default=1920)
    parser.add_argument("--tids", type=str, default="0,2")
    parser.add_argument("--sample-limit", type=int, default=12)
    args = parser.parse_args()

    tracked_tids = {int(token) for token in args.tids.split(",") if token.strip()}
    blocks_per_row = args.hidden_size // 16
    padded_blocks_per_row = ((blocks_per_row + 3) // 4) * 4
    scale_blob = args.scale_file.read_bytes()

    live_entries = [
        entry for entry in parse_live_entries(args.live_probe)
        if entry.tid in tracked_tids
    ]
    live_entries.sort(key=lambda entry: (entry.tid, entry.n_tile, entry.k_block))

    print(f"tracked_entries={len(live_entries)}")
    for entry in live_entries:
        hits: list[tuple[int, int]] = []
        if entry.scale_reg_post != 0:
            for row in range(args.weight_rows):
                for block_base in range(0, blocks_per_row - 3, 4):
                    if load_scale_word(scale_blob, row, block_base, padded_blocks_per_row) == entry.scale_reg_post:
                        hits.append((row, block_base))
        hit_preview = ",".join(f"({row},{block_base})" for row, block_base in hits[: args.sample_limit])
        print(
            f"tid={entry.tid} n_tile={entry.n_tile} k_block={entry.k_block} "
            f"part=({entry.part_output_col0},{entry.part_token_row0}) "
            f"scale_reg_post=0x{entry.scale_reg_post:08x} "
            f"hit_count={len(hits)} hits={hit_preview}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
