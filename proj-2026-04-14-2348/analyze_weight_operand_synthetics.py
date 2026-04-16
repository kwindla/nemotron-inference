#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path


LIVE_ENTRY_RE = re.compile(
    r"entry=\d+\s+tid=(?P<tid>-?\d+)\s+.*?"
    r"n_tile=(?P<n_tile>-?\d+)\s+"
    r"k_block=(?P<k_block>-?\d+)\s+.*?"
    r"part_c0=\((?P<part_output_col0>-?\d+),(?P<part_token_row0>-?\d+)\)\s+.*?"
    r"copy_sig_count=\d+\s+copy_sig=(?P<copy_sig>[^ ]+)"
)
RUNTIME_TID_RE = re.compile(r"nano_p1_a_operand_probe: ---- tid=(?P<tid>-?\d+) ----")
RUNTIME_ENTRY_RE = re.compile(
    r"n_tile=(?P<n_tile>-?\d+)\s+"
    r"k_block=(?P<k_block>-?\d+)\s+"
    r"part_token_row0=(?P<part_token_row0>-?\d+)\s+"
    r"part_output_col0=(?P<part_output_col0>-?\d+)"
)
RUNTIME_COPY_SIG_RE = re.compile(r"copy_sig=(?P<copy_sig>.*)")


@dataclasses.dataclass(frozen=True)
class Key:
    tid: int
    n_tile: int
    k_block: int


@dataclasses.dataclass(frozen=True)
class SyntheticEntry:
    tid: int
    n_tile: int
    k_block: int
    part_output_col0: int
    part_token_row0: int
    stage0_offset: int
    payload_byte: int

    @property
    def key(self) -> Key:
        return Key(self.tid, self.n_tile, self.k_block)


def parse_copy_sig(copy_sig: str) -> tuple[int, int]:
    first = copy_sig.split(",", 1)[0].strip()
    stage0_offset_s, payload_byte_s = first.split(":")
    return int(stage0_offset_s), int(payload_byte_s)


def parse_live(path: Path) -> dict[Key, SyntheticEntry]:
    entries: dict[Key, SyntheticEntry] = {}
    for line in path.read_text().splitlines():
      match = LIVE_ENTRY_RE.search(line)
      if match is None:
          continue
      groups = match.groupdict()
      stage0_offset, payload_byte = parse_copy_sig(groups["copy_sig"])
      entry = SyntheticEntry(
          tid=int(groups["tid"]),
          n_tile=int(groups["n_tile"]),
          k_block=int(groups["k_block"]),
          part_output_col0=int(groups["part_output_col0"]),
          part_token_row0=int(groups["part_token_row0"]),
          stage0_offset=stage0_offset,
          payload_byte=payload_byte,
      )
      entries[entry.key] = entry
    return entries


def parse_runtime(path: Path) -> dict[Key, SyntheticEntry]:
    entries: dict[Key, SyntheticEntry] = {}
    current_tid: int | None = None
    pending_key: tuple[int, int, int, int, int] | None = None

    for line in path.read_text().splitlines():
      tid_match = RUNTIME_TID_RE.search(line)
      if tid_match is not None:
          current_tid = int(tid_match.group("tid"))
          continue
      if current_tid is None:
          continue

      entry_match = RUNTIME_ENTRY_RE.search(line)
      if entry_match is not None:
          groups = entry_match.groupdict()
          pending_key = (
              current_tid,
              int(groups["n_tile"]),
              int(groups["k_block"]),
              int(groups["part_output_col0"]),
              int(groups["part_token_row0"]),
          )
          continue

      if pending_key is None or "copy_sig=" not in line:
          continue

      copy_match = RUNTIME_COPY_SIG_RE.search(line)
      if copy_match is None:
          continue
      stage0_offset, payload_byte = parse_copy_sig(copy_match.group("copy_sig"))
      tid, n_tile, k_block, part_output_col0, part_token_row0 = pending_key
      entry = SyntheticEntry(
          tid=tid,
          n_tile=n_tile,
          k_block=k_block,
          part_output_col0=part_output_col0,
          part_token_row0=part_token_row0,
          stage0_offset=stage0_offset,
          payload_byte=payload_byte,
      )
      entries[entry.key] = entry
      pending_key = None
    return entries


def parse_probe(path: Path, kind: str) -> dict[Key, SyntheticEntry]:
    if kind == "live":
        return parse_live(path)
    if kind == "runtime":
        return parse_runtime(path)
    raise ValueError(f"unsupported kind {kind!r}")


def format_part(entry: SyntheticEntry) -> str:
    return f"({entry.part_output_col0},{entry.part_token_row0})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode logical (source_row, byte_index) tags from paired row_index and byte_index probe captures."
    )
    parser.add_argument("--live-row", type=Path, required=True)
    parser.add_argument("--live-byte", type=Path, required=True)
    parser.add_argument("--runtime-row", type=Path, required=True)
    parser.add_argument("--runtime-byte", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=16)
    args = parser.parse_args()

    live_row = parse_probe(args.live_row, "live")
    live_byte = parse_probe(args.live_byte, "live")
    runtime_row = parse_probe(args.runtime_row, "runtime")
    runtime_byte = parse_probe(args.runtime_byte, "runtime")

    shared = sorted(set(live_row) & set(live_byte) & set(runtime_row) & set(runtime_byte), key=lambda k: (k.tid, k.n_tile, k.k_block))

    print(f"shared_entries={len(shared)}")
    if not shared:
        return 0

    tuple_mismatches = []
    for key in shared:
        live_row_entry = live_row[key]
        live_byte_entry = live_byte[key]
        runtime_row_entry = runtime_row[key]
        runtime_byte_entry = runtime_byte[key]

        live_tuple = (live_row_entry.payload_byte, live_byte_entry.payload_byte)
        runtime_tuple = (runtime_row_entry.payload_byte, runtime_byte_entry.payload_byte)
        if live_tuple != runtime_tuple:
            tuple_mismatches.append(
                (
                    key,
                    live_row_entry,
                    live_byte_entry,
                    runtime_row_entry,
                    runtime_byte_entry,
                    live_tuple,
                    runtime_tuple,
                )
            )

    print(f"logical_tuple_mismatches={len(tuple_mismatches)}")
    for item in tuple_mismatches[: args.sample_limit]:
        key, live_row_entry, live_byte_entry, runtime_row_entry, runtime_byte_entry, live_tuple, runtime_tuple = item
        print(
            "sample"
            f" key={key}"
            f" part_live={format_part(live_row_entry)}"
            f" part_runtime={format_part(runtime_row_entry)}"
            f" live_tuple={live_tuple}"
            f" runtime_tuple={runtime_tuple}"
            f" live_offsets=({live_row_entry.stage0_offset},{live_byte_entry.stage0_offset})"
            f" runtime_offsets=({runtime_row_entry.stage0_offset},{runtime_byte_entry.stage0_offset})"
        )

    print("\nper_tid_samples:")
    last_tid = None
    shown = 0
    for key in shared:
        if shown >= args.sample_limit:
            break
        live_row_entry = live_row[key]
        live_byte_entry = live_byte[key]
        runtime_row_entry = runtime_row[key]
        runtime_byte_entry = runtime_byte[key]
        if key.tid != last_tid:
            print(f"tid={key.tid}")
            last_tid = key.tid
        print(
            f"  n_tile={key.n_tile} k_block={key.k_block} "
            f"part={format_part(live_row_entry)} "
            f"live_tuple=({live_row_entry.payload_byte},{live_byte_entry.payload_byte}) "
            f"runtime_tuple=({runtime_row_entry.payload_byte},{runtime_byte_entry.payload_byte})"
        )
        shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
