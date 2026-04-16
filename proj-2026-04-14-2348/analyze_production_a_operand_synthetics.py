#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path


LIVE_ENTRY_RE = re.compile(
    r"entry=\d+\s+tid=(?P<tid>-?\d+)\s+.*?"
    r"n_tile=(?P<n_tile>-?\d+)\s+"
    r"k_block=(?P<k_block>-?\d+)\s+.*?"
    r"part_c0=\((?P<part_output_col0>-?\d+),(?P<part_token_row0>-?\d+)\)\s+.*?"
    r"reg_pre=\((?P<reg0_pre>0x[0-9a-f]+),(?P<reg1_pre>0x[0-9a-f]+)\)\s+"
    r"reg_post=\((?P<reg0_post>0x[0-9a-f]+),(?P<reg1_post>0x[0-9a-f]+)\)"
)

MAINLOOP_ENTRY_RE = re.compile(
    r"nano_p1_mainloop_oracle_test: probe a_operand\s+"
    r"tid=(?P<tid>-?\d+)\s+"
    r"(?:k_step=(?P<k_step>-?\d+)\s+k_base=(?P<k_base>-?\d+)\s+)?"
    r"n_tile=(?P<n_tile>-?\d+)\s+"
    r"k_block=(?P<k_block>-?\d+)\s+"
    r"part_token_row0=(?P<part_token_row0>-?\d+)\s+"
    r"part_output_col0=(?P<part_output_col0>-?\d+)\s+.*?"
    r"reg_pre=\((?P<reg0_pre>0x[0-9a-f]+),(?P<reg1_pre>0x[0-9a-f]+)\)\s+"
    r"reg_post=\((?P<reg0_post>0x[0-9a-f]+),(?P<reg1_post>0x[0-9a-f]+)\)"
)


@dataclasses.dataclass(frozen=True)
class Key:
    tid: int
    n_tile: int
    k_block: int


@dataclasses.dataclass(frozen=True)
class Entry:
    tid: int
    n_tile: int
    k_block: int
    part_output_col0: int
    part_token_row0: int
    reg_bytes: tuple[int, ...]

    @property
    def key(self) -> Key:
        return Key(self.tid, self.n_tile, self.k_block)


def u32_hex_to_le_bytes(value: str) -> tuple[int, int, int, int]:
    word = int(value, 16)
    return tuple((word >> (8 * i)) & 0xFF for i in range(4))


def pick_reg_bytes(groups: dict[str, str], which: str) -> tuple[int, ...]:
    return u32_hex_to_le_bytes(groups[f"reg0_{which}"]) + u32_hex_to_le_bytes(groups[f"reg1_{which}"])


def parse_entries(path: Path, pattern: re.Pattern[str], which: str) -> dict[Key, Entry]:
    entries: dict[Key, Entry] = {}
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        entry = Entry(
            tid=int(groups["tid"]),
            n_tile=int(groups["n_tile"]),
            k_block=int(groups["k_block"]),
            part_output_col0=int(groups["part_output_col0"]),
            part_token_row0=int(groups["part_token_row0"]),
            reg_bytes=pick_reg_bytes(groups, which),
        )
        entries[entry.key] = entry
    return entries


def parse_live(path: Path, which: str) -> dict[Key, Entry]:
    return parse_entries(path, LIVE_ENTRY_RE, which)


def parse_mainloop(path: Path, which: str) -> dict[Key, Entry]:
    return parse_entries(path, MAINLOOP_ENTRY_RE, which)


def merge_high_bits(low_entry: Entry, high_entry: Entry) -> tuple[int, ...]:
    if len(low_entry.reg_bytes) != len(high_entry.reg_bytes):
        raise ValueError("mismatched byte counts while merging high bits")
    return tuple(low | (high << 8) for low, high in zip(low_entry.reg_bytes, high_entry.reg_bytes))


def normalize_runtime_kblock_col4_swap(entries: dict[Key, Entry]) -> dict[Key, Entry]:
    normalized: dict[Key, Entry] = {}
    for key, entry in entries.items():
        new_key = key
        if (entry.part_output_col0 % 16) == 4:
            new_key = Key(entry.tid, entry.n_tile, entry.k_block ^ 1)
        normalized[new_key] = entry
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare live tactic-5 weight synthetics against the production mainloop A-operand probe."
    )
    parser.add_argument("--which", choices=("pre", "post"), default="post")
    parser.add_argument("--live-row-low", type=Path, required=True)
    parser.add_argument("--live-byte-low", type=Path, required=True)
    parser.add_argument("--runtime-row-low", type=Path, required=True)
    parser.add_argument("--runtime-byte-low", type=Path, required=True)
    parser.add_argument("--live-row-high", type=Path)
    parser.add_argument("--live-byte-high", type=Path)
    parser.add_argument("--runtime-row-high", type=Path)
    parser.add_argument("--runtime-byte-high", type=Path)
    parser.add_argument(
        "--normalize-runtime-kblock-col4-swap",
        action="store_true",
        help="Flip runtime k_block for entries whose part_output_col0 %% 16 == 4.",
    )
    parser.add_argument("--sample-limit", type=int, default=16)
    args = parser.parse_args()

    live_row_low = parse_live(args.live_row_low, args.which)
    live_byte_low = parse_live(args.live_byte_low, args.which)
    runtime_row_low = parse_mainloop(args.runtime_row_low, args.which)
    runtime_byte_low = parse_mainloop(args.runtime_byte_low, args.which)

    live_row_high = parse_live(args.live_row_high, args.which) if args.live_row_high else None
    live_byte_high = parse_live(args.live_byte_high, args.which) if args.live_byte_high else None
    runtime_row_high = parse_mainloop(args.runtime_row_high, args.which) if args.runtime_row_high else None
    runtime_byte_high = parse_mainloop(args.runtime_byte_high, args.which) if args.runtime_byte_high else None

    if args.normalize_runtime_kblock_col4_swap:
        runtime_row_low = normalize_runtime_kblock_col4_swap(runtime_row_low)
        runtime_byte_low = normalize_runtime_kblock_col4_swap(runtime_byte_low)
        if runtime_row_high is not None:
            runtime_row_high = normalize_runtime_kblock_col4_swap(runtime_row_high)
        if runtime_byte_high is not None:
            runtime_byte_high = normalize_runtime_kblock_col4_swap(runtime_byte_high)

    shared = sorted(
        set(live_row_low)
        & set(live_byte_low)
        & set(runtime_row_low)
        & set(runtime_byte_low)
        & (set(live_row_high) if live_row_high is not None else set(live_row_low))
        & (set(live_byte_high) if live_byte_high is not None else set(live_byte_low))
        & (set(runtime_row_high) if runtime_row_high is not None else set(runtime_row_low))
        & (set(runtime_byte_high) if runtime_byte_high is not None else set(runtime_byte_low)),
        key=lambda key: (key.tid, key.n_tile, key.k_block),
    )

    print(f"which={args.which}")
    print(f"shared_entries={len(shared)}")
    if not shared:
      return 0

    part_matches = 0
    entry_mismatches = 0
    lane_mismatches = 0
    samples: list[dict[str, object]] = []

    for key in shared:
        live_row_lo = live_row_low[key]
        live_byte_lo = live_byte_low[key]
        runtime_row_lo = runtime_row_low[key]
        runtime_byte_lo = runtime_byte_low[key]
        live_row_bytes = (
            merge_high_bits(live_row_lo, live_row_high[key]) if live_row_high is not None else live_row_lo.reg_bytes
        )
        live_byte_bytes = (
            merge_high_bits(live_byte_lo, live_byte_high[key]) if live_byte_high is not None else live_byte_lo.reg_bytes
        )
        runtime_row_bytes = (
            merge_high_bits(runtime_row_lo, runtime_row_high[key])
            if runtime_row_high is not None
            else runtime_row_lo.reg_bytes
        )
        runtime_byte_bytes = (
            merge_high_bits(runtime_byte_lo, runtime_byte_high[key])
            if runtime_byte_high is not None
            else runtime_byte_lo.reg_bytes
        )

        live_part = (live_row_lo.part_output_col0, live_row_lo.part_token_row0)
        runtime_part = (runtime_row_lo.part_output_col0, runtime_row_lo.part_token_row0)
        if live_part == runtime_part:
            part_matches += 1

        live_tuples = list(zip(live_row_bytes, live_byte_bytes))
        runtime_tuples = list(zip(runtime_row_bytes, runtime_byte_bytes))
        mismatching_lanes = [
            lane for lane, (live_tuple, runtime_tuple) in enumerate(zip(live_tuples, runtime_tuples))
            if live_tuple != runtime_tuple
        ]
        lane_mismatches += len(mismatching_lanes)
        if mismatching_lanes:
            entry_mismatches += 1
            if len(samples) < args.sample_limit:
                samples.append(
                    {
                        "key": dataclasses.asdict(key),
                        "part": {"live": live_part, "runtime": runtime_part},
                        "mismatching_lanes": mismatching_lanes,
                        "live_tuples": live_tuples,
                        "runtime_tuples": runtime_tuples,
                    }
                )

    print(f"part_matches={part_matches}/{len(shared)}")
    print(f"entry_tuple_mismatches={entry_mismatches}")
    print(f"lane_tuple_mismatches={lane_mismatches}")
    print("samples=" + json.dumps(samples, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
