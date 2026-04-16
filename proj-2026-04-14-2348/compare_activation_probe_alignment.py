#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


LIVE_ENTRY_RE = re.compile(
    r"entry=\d+ tid=(\d+).*?n_tile=(\d+).*?k_block=(\d+).*?"
    r"part_c0=\(([-\d]+),([-\d]+)\) "
    r"local0=\(([-\d]+),([-\d]+)\) "
    r"local1=\(([-\d]+),([-\d]+)\) "
    r"stage0_offsets=\(([-\d]+),([-\d]+)\) "
    r"copy_sig_count=(\d+) copy_sig=([^\n]*?) reg_"
)
RUNTIME_TID_RE = re.compile(r"nano_p1_[ab]_operand_probe: ---- tid=(\d+) ----")
RUNTIME_ENTRY_RE = re.compile(
    r"n_tile=(\d+).*?k_block=(\d+).*?"
    r"part_token_row0=(\d+).*?part_output_col0=(\d+).*?"
    r"local_row0=(\d+).*?local_col0=(\d+).*?"
    r"local_row1=(\d+).*?local_col1=(\d+).*?"
    r"stage0_offset0=(\d+).*?stage0_offset1=(\d+)"
)
RUNTIME_COPY_SIG_RE = re.compile(r"copy_sig=(.*)")


def parse_live(path: Path) -> dict[tuple[int, int, int], dict[str, object]]:
    entries: dict[tuple[int, int, int], dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("entry="):
            continue
        match = LIVE_ENTRY_RE.search(line)
        if not match:
            continue
        tid, n_tile, k_block = map(int, match.group(1, 2, 3))
        entries[(tid, n_tile, k_block)] = {
            "part_c0": tuple(map(int, match.group(4, 5))),
            "local0": tuple(map(int, match.group(6, 7))),
            "local1": tuple(map(int, match.group(8, 9))),
            "stage0_offsets": tuple(map(int, match.group(10, 11))),
            "copy_sig_count": int(match.group(12)),
            "copy_sig": match.group(13).strip(),
        }
    return entries


def parse_runtime(path: Path) -> dict[tuple[int, int, int], dict[str, object]]:
    entries: dict[tuple[int, int, int], dict[str, object]] = {}
    current_tid: int | None = None
    pending_key: tuple[int, int, int] | None = None

    for line in path.read_text().splitlines():
        tid_match = RUNTIME_TID_RE.search(line)
        if tid_match:
            current_tid = int(tid_match.group(1))
            continue
        if current_tid is None:
            continue

        if line.startswith("  n_tile="):
            match = RUNTIME_ENTRY_RE.search(line)
            if not match:
                continue
            n_tile, k_block = map(int, match.group(1, 2))
            pending_key = (current_tid, n_tile, k_block)
            entries[pending_key] = {
                "part_c0": tuple(map(int, match.group(3, 4))),
                "local0": tuple(map(int, match.group(5, 6))),
                "local1": tuple(map(int, match.group(7, 8))),
                "stage0_offsets": tuple(map(int, match.group(9, 10))),
                "copy_sig_count": 0,
                "copy_sig": "",
            }
            continue

        if pending_key is not None and "copy_sig=" in line:
            match = RUNTIME_COPY_SIG_RE.search(line)
            if match:
                copy_sig = match.group(1).strip()
                entries[pending_key]["copy_sig"] = copy_sig
                entries[pending_key]["copy_sig_count"] = copy_sig.count(",") + 1 if copy_sig else 0
            pending_key = None

    return entries


def format_sample(
    shared_key: tuple[int, int, int],
    field: str,
    live_entries: dict[tuple[int, int, int], dict[str, object]],
    runtime_entries: dict[tuple[int, int, int], dict[str, object]],
) -> str:
    return (
        f"sample key={shared_key} "
        f"live={live_entries[shared_key][field]!r} "
        f"runtime={runtime_entries[shared_key][field]!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare live FlashInfer activation probe output against Nano activation probe output."
    )
    parser.add_argument("--live", type=Path, required=True, help="Path to operand_probe_tactic*.txt")
    parser.add_argument(
        "--runtime",
        type=Path,
        required=True,
        help="Path to nano_p1_b_operand_probe text output",
    )
    args = parser.parse_args()

    live_entries = parse_live(args.live)
    runtime_entries = parse_runtime(args.runtime)
    shared = sorted(set(live_entries) & set(runtime_entries))

    print(f"live_entries={len(live_entries)} runtime_entries={len(runtime_entries)} shared={len(shared)}")
    if not shared:
        return 0

    fields = ["part_c0", "local0", "local1", "stage0_offsets", "copy_sig_count", "copy_sig"]
    for field in fields:
        mismatches = [key for key in shared if live_entries[key][field] != runtime_entries[key][field]]
        print(f"{field}_mismatches={len(mismatches)}")
        for key in mismatches[:5]:
            print(f"  {format_sample(key, field, live_entries, runtime_entries)}")

    swapped_part_c = [
        key
        for key in shared
        if live_entries[key]["part_c0"] == tuple(reversed(runtime_entries[key]["part_c0"]))  # type: ignore[arg-type]
    ]
    print(f"part_c0_swapped_matches={len(swapped_part_c)}")
    for key in swapped_part_c[:5]:
        print(
            f"  swapped_sample key={key} "
            f"live={live_entries[key]['part_c0']!r} "
            f"runtime={runtime_entries[key]['part_c0']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
