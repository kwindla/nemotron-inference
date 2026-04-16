#!/usr/bin/env python3
import argparse
import dataclasses
import pathlib
import re
import sys


LIVE_ENTRY_RE = re.compile(
    r"entry=\d+\s+"
    r"tid=(?P<tid>-?\d+)\s+"
    r".*?"
    r"n_tile=(?P<n_tile>-?\d+)\s+"
    r"k_block=(?P<k_block>-?\d+)\s+"
    r".*?"
    r"part_c0=\((?P<part_output_col0>-?\d+),(?P<part_token_row0>-?\d+)\)\s+"
    r"local0=\((?P<local_row0>-?\d+),(?P<local_col0>-?\d+)\)\s+"
    r".*?"
    r"stage0_offsets=\((?P<stage0_offset0>-?\d+),(?P<stage0_offset1>-?\d+)\)\s+"
    r"reg_pre=\(0x(?P<reg0_pre>[0-9a-fA-F]+),0x(?P<reg1_pre>[0-9a-fA-F]+)\)"
)

RUNTIME_TID_RE = re.compile(r"^nano_p1_b_operand_probe: ---- tid=(?P<tid>-?\d+) ----$")
RUNTIME_ENTRY_RE = re.compile(
    r"^\s*n_tile=(?P<n_tile>-?\d+)\s+"
    r".*?"
    r"k_block=(?P<k_block>-?\d+)\s+"
    r".*?"
    r"part_token_row0=(?P<part_token_row0>-?\d+)\s+"
    r"part_output_col0=(?P<part_output_col0>-?\d+)\s+"
    r"local_row0=(?P<local_row0>-?\d+)\s+"
    r"local_col0=(?P<local_col0>-?\d+)\s+"
    r".*?"
    r"stage0_offset0=(?P<stage0_offset0>-?\d+)\s+"
    r"stage0_offset1=(?P<stage0_offset1>-?\d+)\s+"
    r".*?"
    r"byte_tag_reg0_pre=0x(?P<reg0_pre>[0-9a-fA-F]+)\s+"
    r"byte_tag_reg1_pre=0x(?P<reg1_pre>[0-9a-fA-F]+)"
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
    local_row0: int
    local_col0: int
    stage0_offset0: int
    reg0_pre: int
    reg1_pre: int

    @property
    def key(self) -> Key:
        return Key(self.tid, self.n_tile, self.k_block)


def parse_live(path: pathlib.Path) -> dict[Key, Entry]:
    entries: dict[Key, Entry] = {}
    for line in path.read_text().splitlines():
        match = LIVE_ENTRY_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        entry = Entry(
            tid=int(groups["tid"]),
            n_tile=int(groups["n_tile"]),
            k_block=int(groups["k_block"]),
            part_output_col0=int(groups["part_output_col0"]),
            part_token_row0=int(groups["part_token_row0"]),
            local_row0=int(groups["local_row0"]),
            local_col0=int(groups["local_col0"]),
            stage0_offset0=int(groups["stage0_offset0"]),
            reg0_pre=int(groups["reg0_pre"], 16),
            reg1_pre=int(groups["reg1_pre"], 16),
        )
        entries[entry.key] = entry
    return entries


def parse_runtime(path: pathlib.Path) -> dict[Key, Entry]:
    entries: dict[Key, Entry] = {}
    current_tid: int | None = None
    for line in path.read_text().splitlines():
        tid_match = RUNTIME_TID_RE.match(line)
        if tid_match is not None:
            current_tid = int(tid_match.group("tid"))
            continue
        match = RUNTIME_ENTRY_RE.match(line)
        if match is None or current_tid is None:
            continue
        groups = match.groupdict()
        entry = Entry(
            tid=current_tid,
            n_tile=int(groups["n_tile"]),
            k_block=int(groups["k_block"]),
            part_output_col0=int(groups["part_output_col0"]),
            part_token_row0=int(groups["part_token_row0"]),
            local_row0=int(groups["local_row0"]),
            local_col0=int(groups["local_col0"]),
            stage0_offset0=int(groups["stage0_offset0"]),
            reg0_pre=int(groups["reg0_pre"], 16),
            reg1_pre=int(groups["reg1_pre"], 16),
        )
        entries[entry.key] = entry
    return entries


def same_coords(lhs: Entry, rhs: Entry) -> bool:
    return (
        lhs.part_output_col0 == rhs.part_output_col0
        and lhs.part_token_row0 == rhs.part_token_row0
        and lhs.local_row0 == rhs.local_row0
        and lhs.local_col0 == rhs.local_col0
        and lhs.stage0_offset0 == rhs.stage0_offset0
    )


def same_regs(lhs: Entry, rhs: Entry) -> bool:
    return lhs.reg0_pre == rhs.reg0_pre and lhs.reg1_pre == rhs.reg1_pre


def fmt_regs(entry: Entry) -> str:
    return f"(0x{entry.reg0_pre:08x},0x{entry.reg1_pre:08x})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare live K=128 flashinfer w1_fp4 probe output against runtime nano_p1_b_operand_probe output."
    )
    parser.add_argument("--live", required=True, type=pathlib.Path, help="Live operand_probe_tactic1.txt path")
    parser.add_argument("--runtime", required=True, type=pathlib.Path, help="Runtime nano_p1_b_operand_probe output path")
    parser.add_argument(
        "--show-mismatches",
        action="store_true",
        help="Print each mismatching shared entry",
    )
    args = parser.parse_args()

    live = parse_live(args.live)
    runtime = parse_runtime(args.runtime)

    shared_keys = sorted(set(live) & set(runtime), key=lambda key: (key.tid, key.n_tile, key.k_block))
    coord_mismatches = 0
    reg_mismatches = 0
    even_family_reg_mismatches = 0
    odd_family_reg_mismatches = 0
    odd_family_swap_candidates = 0
    reg_mismatches_after_odd_family_swap = 0
    mismatches: list[str] = []

    for key in shared_keys:
        live_entry = live[key]
        runtime_entry = runtime[key]
        if not same_coords(live_entry, runtime_entry):
            coord_mismatches += 1
            mismatches.append(
                f"coord key={key} runtime(part=({runtime_entry.part_output_col0},{runtime_entry.part_token_row0}) "
                f"local=({runtime_entry.local_row0},{runtime_entry.local_col0}) stage0={runtime_entry.stage0_offset0}) "
                f"live(part=({live_entry.part_output_col0},{live_entry.part_token_row0}) "
                f"local=({live_entry.local_row0},{live_entry.local_col0}) stage0={live_entry.stage0_offset0})"
            )
            continue

        family_is_odd = (runtime_entry.local_col0 % 2) == 1
        if not same_regs(live_entry, runtime_entry):
            reg_mismatches += 1
            if family_is_odd:
                odd_family_reg_mismatches += 1
            else:
                even_family_reg_mismatches += 1

            swapped_key = Key(key.tid, key.n_tile, 1 - key.k_block)
            swapped_entry = runtime.get(swapped_key)
            swap_matches = (
                family_is_odd
                and swapped_entry is not None
                and same_regs(live_entry, swapped_entry)
            )
            if swap_matches:
                odd_family_swap_candidates += 1

            mismatches.append(
                f"reg key={key} local_col0={runtime_entry.local_col0} stage0_offset0={runtime_entry.stage0_offset0} "
                f"runtime={fmt_regs(runtime_entry)} live={fmt_regs(live_entry)} "
                f"odd_family_swap_match={'yes' if swap_matches else 'no'}"
            )

        compare_entry = runtime_entry
        if family_is_odd:
            swapped_entry = runtime.get(Key(key.tid, key.n_tile, 1 - key.k_block))
            if swapped_entry is not None:
                compare_entry = dataclasses.replace(
                    swapped_entry,
                    tid=runtime_entry.tid,
                    n_tile=runtime_entry.n_tile,
                    k_block=runtime_entry.k_block,
                    part_output_col0=runtime_entry.part_output_col0,
                    part_token_row0=runtime_entry.part_token_row0,
                    local_row0=runtime_entry.local_row0,
                    local_col0=runtime_entry.local_col0,
                    stage0_offset0=runtime_entry.stage0_offset0,
                )
        if not same_regs(live_entry, compare_entry):
            reg_mismatches_after_odd_family_swap += 1

    print(
        f"live_entries={len(live)} runtime_entries={len(runtime)} shared_entries={len(shared_keys)}"
    )
    print(f"missing_in_runtime={len(live) - len(shared_keys)} missing_in_live={len(runtime) - len(shared_keys)}")
    print(f"coord_mismatches={coord_mismatches}/{len(shared_keys)}")
    print(f"reg_mismatches={reg_mismatches}/{len(shared_keys)}")
    print(f"even_family_reg_mismatches={even_family_reg_mismatches}")
    print(f"odd_family_reg_mismatches={odd_family_reg_mismatches}")
    print(f"odd_family_swap_candidates={odd_family_swap_candidates}")
    print(
        f"reg_mismatches_after_odd_family_kblock_swap={reg_mismatches_after_odd_family_swap}/{len(shared_keys)}"
    )

    if args.show_mismatches:
      for line in mismatches:
          print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
