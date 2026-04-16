#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_BIN = ROOT / "build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe"
ARCHIVE_ROOT = ROOT / "proj-2026-04-12-1022/trtllm_reference/archive/2026-04-14-probe-chain"
OUT_DIR = ROOT / "proj-2026-04-14-2348/archive/focused_b_attempt/stage_slot_compare"
TRACKED_TIDS = "32,48,64,80"

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
    r".*?"
    r"reg_pre=\(0x(?P<reg0_pre>[0-9a-fA-F]+),0x(?P<reg1_pre>[0-9a-fA-F]+)\)"
)
RUNTIME_TID_RE = re.compile(r"^\s*nano_p1_b_operand_probe: ---- tid=(?P<tid>-?\d+) ----$")
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
    r"source_row_reg0_pre=0x(?P<source_row_reg0_pre>[0-9a-fA-F]+)\s+"
    r"source_row_reg1_pre=0x(?P<source_row_reg1_pre>[0-9a-fA-F]+)\s+"
    r"byte_tag_reg0_pre=0x(?P<byte_tag_reg0_pre>[0-9a-fA-F]+)\s+"
    r"byte_tag_reg1_pre=0x(?P<byte_tag_reg1_pre>[0-9a-fA-F]+)"
)


@dataclasses.dataclass(frozen=True)
class Key:
    tid: int
    n_tile: int
    k_block: int


@dataclasses.dataclass(frozen=True)
class LiveEntry:
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


@dataclasses.dataclass(frozen=True)
class RuntimeEntry:
    tid: int
    n_tile: int
    k_block: int
    part_output_col0: int
    part_token_row0: int
    local_row0: int
    local_col0: int
    stage0_offset0: int
    source_row_reg0_pre: int
    source_row_reg1_pre: int
    byte_tag_reg0_pre: int
    byte_tag_reg1_pre: int

    @property
    def key(self) -> Key:
        return Key(self.tid, self.n_tile, self.k_block)


def parse_live(path: pathlib.Path) -> dict[Key, LiveEntry]:
    entries: dict[Key, LiveEntry] = {}
    for line in path.read_text().splitlines():
        match = LIVE_ENTRY_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        entry = LiveEntry(
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


def parse_runtime(path: pathlib.Path) -> dict[Key, RuntimeEntry]:
    entries: dict[Key, RuntimeEntry] = {}
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
        entry = RuntimeEntry(
            tid=current_tid,
            n_tile=int(groups["n_tile"]),
            k_block=int(groups["k_block"]),
            part_output_col0=int(groups["part_output_col0"]),
            part_token_row0=int(groups["part_token_row0"]),
            local_row0=int(groups["local_row0"]),
            local_col0=int(groups["local_col0"]),
            stage0_offset0=int(groups["stage0_offset0"]),
            source_row_reg0_pre=int(groups["source_row_reg0_pre"], 16),
            source_row_reg1_pre=int(groups["source_row_reg1_pre"], 16),
            byte_tag_reg0_pre=int(groups["byte_tag_reg0_pre"], 16),
            byte_tag_reg1_pre=int(groups["byte_tag_reg1_pre"], 16),
        )
        entries[entry.key] = entry
    return entries


def same_coords(live: LiveEntry, runtime: RuntimeEntry) -> bool:
    return (
        live.part_output_col0 == runtime.part_output_col0
        and live.part_token_row0 == runtime.part_token_row0
        and live.local_row0 == runtime.local_row0
        and live.local_col0 == runtime.local_col0
        and live.stage0_offset0 == runtime.stage0_offset0
    )


def runtime_regs(entry: RuntimeEntry, reg_selector: str) -> tuple[int, int]:
    if reg_selector == "source":
        return entry.source_row_reg0_pre, entry.source_row_reg1_pre
    return entry.byte_tag_reg0_pre, entry.byte_tag_reg1_pre


def summarize(
    live: dict[Key, LiveEntry],
    runtime: dict[Key, RuntimeEntry],
    reg_selector: str,
) -> dict[str, int]:
    shared_keys = sorted(set(live) & set(runtime), key=lambda key: (key.tid, key.n_tile, key.k_block))
    coord_mismatches = 0
    reg_mismatches = 0
    even_family_reg_mismatches = 0
    odd_family_reg_mismatches = 0
    odd_family_swap_candidates = 0
    reg_mismatches_after_swap = 0
    for key in shared_keys:
        live_entry = live[key]
        runtime_entry = runtime[key]
        if not same_coords(live_entry, runtime_entry):
            coord_mismatches += 1
            continue
        live_regs = (live_entry.reg0_pre, live_entry.reg1_pre)
        regs = runtime_regs(runtime_entry, reg_selector)
        if regs == live_regs:
            continue
        reg_mismatches += 1
        if (runtime_entry.local_col0 % 2) == 0:
            even_family_reg_mismatches += 1
            reg_mismatches_after_swap += 1
            continue
        odd_family_reg_mismatches += 1
        swapped = runtime.get(Key(key.tid, key.n_tile, 1 - key.k_block))
        if swapped is not None and runtime_regs(swapped, reg_selector) == live_regs:
            odd_family_swap_candidates += 1
        else:
            reg_mismatches_after_swap += 1
    return {
        "shared_entries": len(shared_keys),
        "coord_mismatches": coord_mismatches,
        "reg_mismatches": reg_mismatches,
        "even_family_reg_mismatches": even_family_reg_mismatches,
        "odd_family_reg_mismatches": odd_family_reg_mismatches,
        "odd_family_swap_candidates": odd_family_swap_candidates,
        "reg_mismatches_after_odd_family_kblock_swap": reg_mismatches_after_swap,
    }


def stage_live_path(mode: str) -> pathlib.Path:
    if mode == "stage_slot_low_byte":
        return ARCHIVE_ROOT / "golden_probe_stage_slot_low_byte_live_fullfamily_overwriteall" / "operand_probe_tactic1.txt"
    if mode.startswith("stage_slot_bit"):
        bit = mode.removeprefix("stage_slot_bit")
        return ARCHIVE_ROOT / f"golden_probe_stage_slot_bit{bit}_live_fullfamily_overwriteall" / "operand_probe_tactic1.txt"
    raise ValueError(f"unsupported mode: {mode}")


def configure_runtime_env(mode: str) -> dict[str, str]:
    env = {
        "NEMOTRON_NANO_P1_B_PROBE_TIDS": TRACKED_TIDS,
        "NEMOTRON_NANO_P1_B_PROBE_SOURCE_ROW_MODE": "position_map",
    }
    if mode == "stage_slot_low_byte":
        env["NEMOTRON_NANO_P1_B_PROBE_STAGE_SLOT_LOW_BYTE"] = "1"
    elif mode.startswith("stage_slot_bit"):
        env["NEMOTRON_NANO_P1_B_PROBE_STAGE_SLOT_BIT"] = mode.removeprefix("stage_slot_bit")
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return env


def run_runtime_probe(mode: str, output_path: pathlib.Path) -> pathlib.Path:
    env = os.environ.copy()
    env.update(configure_runtime_env(mode))
    result = subprocess.run(
        [str(PROBE_BIN)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    output_path.write_text(result.stdout)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare runtime probe against archived live stage-slot probe families.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["stage_slot_low_byte"] + [f"stage_slot_bit{bit}" for bit in range(8)],
        help="Stage-slot synthetic probe families to compare.",
    )
    args = parser.parse_args()

    if not PROBE_BIN.exists():
        print(f"missing probe binary: {PROBE_BIN}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "mode shared_entries coord_mismatches reg_mismatches "
        "even_family_reg_mismatches odd_family_reg_mismatches "
        "odd_family_swap_candidates reg_mismatches_after_odd_family_kblock_swap"
    )
    for mode in args.modes:
        live_path = stage_live_path(mode)
        runtime_path = OUT_DIR / f"{mode}_runtime.txt"
        live = parse_live(live_path)
        runtime = parse_runtime(run_runtime_probe(mode, runtime_path))
        stats = summarize(live, runtime, reg_selector="source")
        print(
            f"{mode} "
            f"{stats['shared_entries']} "
            f"{stats['coord_mismatches']}/{stats['shared_entries']} "
            f"{stats['reg_mismatches']}/{stats['shared_entries']} "
            f"{stats['even_family_reg_mismatches']} "
            f"{stats['odd_family_reg_mismatches']} "
            f"{stats['odd_family_swap_candidates']} "
            f"{stats['reg_mismatches_after_odd_family_kblock_swap']}/{stats['shared_entries']}"
        )
    print(f"artifacts={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
