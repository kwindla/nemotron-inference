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
RUNTIME_TID_RE = re.compile(r"nano_p1_a_operand_probe: ---- tid=(?P<tid>-?\d+) ----")
RUNTIME_ENTRY_RE = re.compile(
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
    reg_pre_bytes: tuple[int, ...]

    @property
    def key(self) -> Key:
        return Key(self.tid, self.n_tile, self.k_block)


def u32_hex_to_le_bytes(value: str) -> tuple[int, int, int, int]:
    word = int(value, 16)
    return tuple((word >> (8 * i)) & 0xFF for i in range(4))


def pick_reg_bytes(groups: dict[str, str], which: str) -> tuple[int, ...]:
    return u32_hex_to_le_bytes(groups[f"reg0_{which}"]) + u32_hex_to_le_bytes(groups[f"reg1_{which}"])


def parse_live(path: Path, which: str) -> dict[Key, Entry]:
    entries: dict[Key, Entry] = {}
    for line in path.read_text().splitlines():
        match = LIVE_ENTRY_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        reg_pre_bytes = pick_reg_bytes(groups, which)
        entry = Entry(
            tid=int(groups["tid"]),
            n_tile=int(groups["n_tile"]),
            k_block=int(groups["k_block"]),
            part_output_col0=int(groups["part_output_col0"]),
            part_token_row0=int(groups["part_token_row0"]),
            reg_pre_bytes=reg_pre_bytes,
        )
        entries[entry.key] = entry
    return entries


def parse_runtime(path: Path, which: str) -> dict[Key, Entry]:
    entries: dict[Key, Entry] = {}
    current_tid: int | None = None
    for line in path.read_text().splitlines():
        tid_match = RUNTIME_TID_RE.search(line)
        if tid_match is not None:
            current_tid = int(tid_match.group("tid"))
            continue
        if current_tid is None:
            continue
        entry_match = RUNTIME_ENTRY_RE.search(line)
        if entry_match is None:
            continue
        groups = entry_match.groupdict()
        reg_pre_bytes = pick_reg_bytes(groups, which)
        entry = Entry(
            tid=current_tid,
            n_tile=int(groups["n_tile"]),
            k_block=int(groups["k_block"]),
            part_output_col0=int(groups["part_output_col0"]),
            part_token_row0=int(groups["part_token_row0"]),
            reg_pre_bytes=reg_pre_bytes,
        )
        entries[entry.key] = entry
    return entries


def parse_probe(path: Path, kind: str, which: str) -> dict[Key, Entry]:
    if kind == "live":
        return parse_live(path, which)
    if kind == "runtime":
        return parse_runtime(path, which)
    raise ValueError(f"unsupported kind {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode logical (row, byte) tuples from row_index/byte_index reg_pre bytes."
    )
    parser.add_argument("--which", choices=("pre", "post"), default="pre")
    parser.add_argument("--live-row", type=Path, required=True)
    parser.add_argument("--live-byte", type=Path, required=True)
    parser.add_argument("--runtime-row", type=Path, required=True)
    parser.add_argument("--runtime-byte", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=16)
    args = parser.parse_args()

    live_row = parse_probe(args.live_row, "live", args.which)
    live_byte = parse_probe(args.live_byte, "live", args.which)
    runtime_row = parse_probe(args.runtime_row, "runtime", args.which)
    runtime_byte = parse_probe(args.runtime_byte, "runtime", args.which)

    print(f"which={args.which}")
    shared = sorted(
        set(live_row) & set(live_byte) & set(runtime_row) & set(runtime_byte),
        key=lambda k: (k.tid, k.n_tile, k.k_block),
    )

    print(f"shared_entries={len(shared)}")
    if not shared:
        return 0

    lane_mismatch_count = 0
    entry_mismatch_count = 0
    samples: list[dict[str, object]] = []

    for key in shared:
        live_row_entry = live_row[key]
        live_byte_entry = live_byte[key]
        runtime_row_entry = runtime_row[key]
        runtime_byte_entry = runtime_byte[key]

        live_tuples = list(zip(live_row_entry.reg_pre_bytes, live_byte_entry.reg_pre_bytes))
        runtime_tuples = list(zip(runtime_row_entry.reg_pre_bytes, runtime_byte_entry.reg_pre_bytes))
        mismatching_lanes = [
            lane for lane, (live_tuple, runtime_tuple) in enumerate(zip(live_tuples, runtime_tuples))
            if live_tuple != runtime_tuple
        ]
        lane_mismatch_count += len(mismatching_lanes)
        if mismatching_lanes:
            entry_mismatch_count += 1
            if len(samples) < args.sample_limit:
                samples.append(
                    {
                        "key": dataclasses.asdict(key),
                        "part": {
                            "live": (live_row_entry.part_output_col0, live_row_entry.part_token_row0),
                            "runtime": (runtime_row_entry.part_output_col0, runtime_row_entry.part_token_row0),
                        },
                        "mismatching_lanes": mismatching_lanes,
                        "live_tuples": live_tuples,
                        "runtime_tuples": runtime_tuples,
                    }
                )

    print(f"entry_mismatches={entry_mismatch_count}")
    print(f"lane_mismatches={lane_mismatch_count}")
    print("samples=" + json.dumps(samples, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
