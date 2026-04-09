#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


FIELDS = [
    "Kernel Name",
    "Block Size",
    "Grid Size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "launch__shared_mem_config_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
]


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"empty ncu csv: {path}")

    header = None
    data_rows = []
    for row in rows:
        if not row:
            continue
        if row[0] == "ID":
            header = row
            continue
        if header is None:
            continue
        if len(row) != len(header):
            continue
        if not row[0].strip().isdigit():
            continue
        data_rows.append(row)

    if header is None or not data_rows:
        raise ValueError(f"unexpected ncu csv shape: {path}")
    return header, data_rows


def with_ms_suffix(value: str) -> str:
    try:
        return f"{float(value):.3f} ms"
    except ValueError:
        return value


def summarize(path: Path) -> int:
    header, data_rows = load_rows(path)
    first = data_rows[0]
    print(f"phase05_ncu_csv={path}")
    for field in FIELDS:
        if field not in header:
            continue
        value = first[header.index(field)]
        if field == "gpu__time_duration.sum":
            value = with_ms_suffix(value)
        print(f"{field}={value}")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_phase05_ncu_csv.py <ncu.csv>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"missing ncu csv: {path}", file=sys.stderr)
        return 1

    try:
        return summarize(path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
