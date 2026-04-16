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
ARCHIVE_DIR = ROOT / "proj-2026-04-14-2348/archive/focused_b_attempt"
LIVE_REAL_DIR = ARCHIVE_DIR / "live_capture_consumer_tids"
LIVE_REAL_PROBE = LIVE_REAL_DIR / "operand_probe_tactic1.txt"
LIVE_REAL_INPUT = LIVE_REAL_DIR / "input_fp4_permuted.bin"
LIVE_ROW_PROBE = ROOT / "proj-2026-04-12-1022/trtllm_reference/archive/2026-04-14-probe-chain/golden_probe_w1_row_index_k128/operand_probe_tactic1.txt"
LIVE_BYTE_PROBE = ROOT / "proj-2026-04-12-1022/trtllm_reference/archive/2026-04-14-probe-chain/golden_probe_w1_byte_index_k128/operand_probe_tactic1.txt"
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


def count_reg_mismatches(
    live: dict[Key, LiveEntry],
    runtime: dict[Key, RuntimeEntry],
    reg_selector: str,
) -> tuple[int, int]:
    shared_keys = sorted(set(live) & set(runtime), key=lambda key: (key.tid, key.n_tile, key.k_block))
    mismatches = 0
    for key in shared_keys:
        live_entry = live[key]
        runtime_entry = runtime[key]
        if not same_coords(live_entry, runtime_entry):
            mismatches += 1
            continue
        if reg_selector == "source":
            runtime_regs = (runtime_entry.source_row_reg0_pre, runtime_entry.source_row_reg1_pre)
        else:
            runtime_regs = (runtime_entry.byte_tag_reg0_pre, runtime_entry.byte_tag_reg1_pre)
        live_regs = (live_entry.reg0_pre, live_entry.reg1_pre)
        if runtime_regs != live_regs:
            mismatches += 1
    return mismatches, len(shared_keys)


def count_real_input_mismatches(
    live: dict[Key, LiveEntry],
    runtime: dict[Key, RuntimeEntry],
) -> tuple[int, int, int]:
    shared_keys = sorted(set(live) & set(runtime), key=lambda key: (key.tid, key.n_tile, key.k_block))
    mismatches = 0
    odd_family_swap_candidates = 0
    for key in shared_keys:
        live_entry = live[key]
        runtime_entry = runtime[key]
        if not same_coords(live_entry, runtime_entry):
            mismatches += 1
            continue
        runtime_regs = (runtime_entry.byte_tag_reg0_pre, runtime_entry.byte_tag_reg1_pre)
        live_regs = (live_entry.reg0_pre, live_entry.reg1_pre)
        if runtime_regs == live_regs:
            continue
        mismatches += 1
        if (runtime_entry.local_col0 % 2) == 1:
            swapped = runtime.get(Key(key.tid, key.n_tile, 1 - key.k_block))
            if swapped is not None and (swapped.byte_tag_reg0_pre, swapped.byte_tag_reg1_pre) == live_regs:
                odd_family_swap_candidates += 1
    return mismatches, len(shared_keys), odd_family_swap_candidates


def run_probe(
    mode: str,
    output_path: pathlib.Path,
    use_input_file: bool,
    input_stage_mode: str | None = None,
) -> None:
    env = os.environ.copy()
    env["NEMOTRON_NANO_P1_B_PROBE_TIDS"] = TRACKED_TIDS
    env["NEMOTRON_NANO_P1_B_PROBE_SOURCE_ROW_MODE"] = mode
    if use_input_file:
        env["NEMOTRON_NANO_P1_B_INPUT_FILE"] = str(LIVE_REAL_INPUT)
        env["NEMOTRON_NANO_P1_B_INPUT_ROWS"] = "128"
        env["NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES"] = "1344"
        if input_stage_mode is not None:
            env["NEMOTRON_NANO_P1_B_INPUT_STAGE_MODE"] = input_stage_mode
    else:
        env.pop("NEMOTRON_NANO_P1_B_INPUT_FILE", None)
        env.pop("NEMOTRON_NANO_P1_B_INPUT_ROWS", None)
        env.pop("NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES", None)
        env.pop("NEMOTRON_NANO_P1_B_INPUT_STAGE_MODE", None)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the focused B-boundary probe sweep.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["row_only_permute", "position_map", "position_map_odd_plus4"],
        help="Source-row modes to sweep.",
    )
    parser.add_argument(
        "--input-stage-modes",
        nargs="+",
        default=["packed_bytes", "unpack_nibbles", "unpack_nibbles_swap_halves"],
        help="Input staging modes to sweep for real-input runs.",
    )
    args = parser.parse_args()

    if not PROBE_BIN.exists():
        print(f"missing probe binary: {PROBE_BIN}", file=sys.stderr)
        return 2

    sweep_dir = ARCHIVE_DIR / "probe_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    live_row = parse_live(LIVE_ROW_PROBE)
    live_byte = parse_live(LIVE_BYTE_PROBE)
    live_real = parse_live(LIVE_REAL_PROBE)

    print(
        "source_row_mode input_stage_mode "
        "row_tag_reg_mismatches byte_tag_reg_mismatches real_input_reg_mismatches odd_swap_candidates"
    )
    for mode in args.modes:
        synthetic_out = sweep_dir / f"{mode}_synthetic.txt"
        run_probe(mode, synthetic_out, use_input_file=False)
        synthetic_runtime = parse_runtime(synthetic_out)

        row_mismatches, row_total = count_reg_mismatches(live_row, synthetic_runtime, "source")
        byte_mismatches, byte_total = count_reg_mismatches(live_byte, synthetic_runtime, "byte")
        for input_stage_mode in args.input_stage_modes:
            real_out = sweep_dir / f"{mode}_{input_stage_mode}_real_input.txt"
            run_probe(mode, real_out, use_input_file=True, input_stage_mode=input_stage_mode)
            real_runtime = parse_runtime(real_out)
            real_mismatches, real_total, odd_swaps = count_real_input_mismatches(live_real, real_runtime)
            print(
                f"{mode} "
                f"{input_stage_mode} "
                f"{row_mismatches}/{row_total} "
                f"{byte_mismatches}/{byte_total} "
                f"{real_mismatches}/{real_total} "
                f"{odd_swaps}"
            )

    print(f"artifacts={sweep_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
