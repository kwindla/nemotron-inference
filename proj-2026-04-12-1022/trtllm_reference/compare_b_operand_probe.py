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
    r"(?:dispatch=\((?P<dispatch_stages>-?\d+),(?P<dispatch_scheduler_stages>-?\d+)\)\s+)?"
    r"(?:k_counts=\((?P<tcrb_k_blocks>-?\d+),(?P<tcrb_copy_view_k_blocks>-?\d+),(?P<tcsb_coord_k_blocks>-?\d+)\)\s+)?"
    r".*?"
    r"(?:sf_counts=\((?P<tcrsfb_k_blocks>-?\d+),(?P<tcrsfb_copy_view_k_blocks>-?\d+),(?P<tcssfb_coord_k_blocks>-?\d+)\)\s+)?"
    r".*?"
    r"part_c0=\((?P<part_c_row0>-?\d+),(?P<part_c_col0>-?\d+)\)\s+"
    r"local0=\((?P<local_row0>-?\d+),(?P<local_col0>-?\d+)\)\s+"
    r"local1=\((?P<local_row1>-?\d+),(?P<local_col1>-?\d+)\)\s+"
    r"stage0_offsets=\((?P<stage0_offset0>-?\d+),(?P<stage0_offset1>-?\d+)\)\s+"
    r"reg_pre=\(0x(?P<reg0_pre>[0-9a-fA-F]+),0x(?P<reg1_pre>[0-9a-fA-F]+)\)"
)

SF_COUNTS_RE = re.compile(
    r"sf_counts=\((?P<tcrsfb_k_blocks>-?\d+),(?P<tcrsfb_copy_view_k_blocks>-?\d+),(?P<tcssfb_coord_k_blocks>-?\d+)\)"
)

SF_LAYOUT_RE = re.compile(
    r"sf_layout=\((?P<sfb_stage_elems>-?\d+),(?P<sfb_total_elems>-?\d+)"
    r"(?:,(?P<sfb_stage_shape_dim0>-?\d+),(?P<sfb_stage_shape_dim1>-?\d+),"
    r"(?P<layout_sfb_tv_dim0>-?\d+),(?P<layout_sfb_tv_dim1>-?\d+)"
    r"|,(?P<layout_sfb_tv_dim0_legacy>-?\d+),(?P<layout_sfb_tv_dim1_legacy>-?\d+))\)"
)
SF_ATOM_RE = re.compile(
    r"sf_atom=\((?P<sf_vec_size>-?\d+),(?P<smem_layout_atom_sfb_dim0>-?\d+),"
    r"(?P<smem_layout_atom_sfb_dim1>-?\d+)\)"
)

RUNTIME_TID_RE = re.compile(r"^\s*nano_p1_(?:reference_)?b_operand_probe: ---- tid=(?P<tid>-?\d+) ----$")
RUNTIME_ENTRY_RE = re.compile(
    r"^\s*n_tile=(?P<n_tile>-?\d+)\s+"
    r".*?"
    r"k_block=(?P<k_block>-?\d+)\s+"
    r".*?"
    r"(?:dispatch=\((?P<dispatch_stages>-?\d+),(?P<dispatch_scheduler_stages>-?\d+)\)\s+)?"
    r"(?:k_counts=\((?P<tcrb_k_blocks>-?\d+),(?P<tcrb_copy_view_k_blocks>-?\d+),(?P<tcsb_coord_k_blocks>-?\d+)\)\s+)?"
    r".*?"
    r"(?:sf_counts=\((?P<tcrsfb_k_blocks>-?\d+),(?P<tcrsfb_copy_view_k_blocks>-?\d+),(?P<tcssfb_coord_k_blocks>-?\d+)\)\s+)?"
    r".*?"
    r"part_token_row0=(?P<part_token_row0>-?\d+)\s+"
    r"part_output_col0=(?P<part_output_col0>-?\d+)\s+"
    r"local_row0=(?P<local_row0>-?\d+)\s+"
    r"local_col0=(?P<local_col0>-?\d+)\s+"
    r"local_row1=(?P<local_row1>-?\d+)\s+"
    r"local_col1=(?P<local_col1>-?\d+)\s+"
    r".*?"
    r"stage0_offset0=(?P<stage0_offset0>-?\d+)\s+"
    r"stage0_offset1=(?P<stage0_offset1>-?\d+)\s+"
    r"source_row_reg0_pre=0x(?P<reg0_pre>[0-9a-fA-F]+)\s+"
    r"source_row_reg1_pre=0x(?P<reg1_pre>[0-9a-fA-F]+)"
)


@dataclasses.dataclass(frozen=True)
class Key:
    tid: int
    n_tile: int
    k_block: int


@dataclasses.dataclass
class Entry:
    tid: int
    n_tile: int
    k_block: int
    dispatch_stages: int
    dispatch_scheduler_stages: int
    tcrb_k_blocks: int
    tcrb_copy_view_k_blocks: int
    tcsb_coord_k_blocks: int
    tcrsfb_k_blocks: int
    tcrsfb_copy_view_k_blocks: int
    tcssfb_coord_k_blocks: int
    sf_vec_size: int
    smem_layout_atom_sfb_dim0: int
    smem_layout_atom_sfb_dim1: int
    sfb_stage_elems: int
    sfb_total_elems: int
    sfb_stage_shape_dim0: int
    sfb_stage_shape_dim1: int
    layout_sfb_tv_dim0: int
    layout_sfb_tv_dim1: int
    part_c_row0: int
    part_c_col0: int
    local_row0: int
    local_col0: int
    local_row1: int
    local_col1: int
    stage0_offset0: int
    stage0_offset1: int
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
        groups = {}
        for k, v in match.groupdict().items():
            if k.startswith("reg"):
                groups[k] = int(v, 16)
            else:
                groups[k] = -1 if v is None else int(v)
        sf_match = SF_COUNTS_RE.search(line)
        if sf_match is not None:
            groups.update({k: int(v) for k, v in sf_match.groupdict().items()})
        sf_layout_match = SF_LAYOUT_RE.search(line)
        sf_atom_match = SF_ATOM_RE.search(line)
        groups["sf_vec_size"] = -1
        groups["smem_layout_atom_sfb_dim0"] = -1
        groups["smem_layout_atom_sfb_dim1"] = -1
        if sf_atom_match is not None:
            groups["sf_vec_size"] = int(sf_atom_match.group("sf_vec_size"))
            groups["smem_layout_atom_sfb_dim0"] = int(sf_atom_match.group("smem_layout_atom_sfb_dim0"))
            groups["smem_layout_atom_sfb_dim1"] = int(sf_atom_match.group("smem_layout_atom_sfb_dim1"))
        groups["sfb_stage_elems"] = -1
        groups["sfb_total_elems"] = -1
        groups["sfb_stage_shape_dim0"] = -1
        groups["sfb_stage_shape_dim1"] = -1
        groups["layout_sfb_tv_dim0"] = -1
        groups["layout_sfb_tv_dim1"] = -1
        if sf_layout_match is not None:
            groups["sfb_stage_elems"] = int(sf_layout_match.group("sfb_stage_elems"))
            groups["sfb_total_elems"] = int(sf_layout_match.group("sfb_total_elems"))
            if sf_layout_match.group("sfb_stage_shape_dim0") is not None:
                groups["sfb_stage_shape_dim0"] = int(sf_layout_match.group("sfb_stage_shape_dim0"))
                groups["sfb_stage_shape_dim1"] = int(sf_layout_match.group("sfb_stage_shape_dim1"))
                groups["layout_sfb_tv_dim0"] = int(sf_layout_match.group("layout_sfb_tv_dim0"))
                groups["layout_sfb_tv_dim1"] = int(sf_layout_match.group("layout_sfb_tv_dim1"))
            else:
                groups["layout_sfb_tv_dim0"] = int(sf_layout_match.group("layout_sfb_tv_dim0_legacy"))
                groups["layout_sfb_tv_dim1"] = int(sf_layout_match.group("layout_sfb_tv_dim1_legacy"))
        entry = Entry(**groups)
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
        sf_match = SF_COUNTS_RE.search(line)
        sf_layout_match = SF_LAYOUT_RE.search(line)
        sf_atom_match = SF_ATOM_RE.search(line)
        sf_vec_size = -1
        smem_layout_atom_sfb_dim0 = -1
        smem_layout_atom_sfb_dim1 = -1
        if sf_atom_match is not None:
            sf_vec_size = int(sf_atom_match.group("sf_vec_size"))
            smem_layout_atom_sfb_dim0 = int(sf_atom_match.group("smem_layout_atom_sfb_dim0"))
            smem_layout_atom_sfb_dim1 = int(sf_atom_match.group("smem_layout_atom_sfb_dim1"))
        sfb_stage_elems = -1
        sfb_total_elems = -1
        sfb_stage_shape_dim0 = -1
        sfb_stage_shape_dim1 = -1
        layout_sfb_tv_dim0 = -1
        layout_sfb_tv_dim1 = -1
        if sf_layout_match is not None:
            sfb_stage_elems = int(sf_layout_match.group("sfb_stage_elems"))
            sfb_total_elems = int(sf_layout_match.group("sfb_total_elems"))
            if sf_layout_match.group("sfb_stage_shape_dim0") is not None:
                sfb_stage_shape_dim0 = int(sf_layout_match.group("sfb_stage_shape_dim0"))
                sfb_stage_shape_dim1 = int(sf_layout_match.group("sfb_stage_shape_dim1"))
                layout_sfb_tv_dim0 = int(sf_layout_match.group("layout_sfb_tv_dim0"))
                layout_sfb_tv_dim1 = int(sf_layout_match.group("layout_sfb_tv_dim1"))
            else:
                layout_sfb_tv_dim0 = int(sf_layout_match.group("layout_sfb_tv_dim0_legacy"))
                layout_sfb_tv_dim1 = int(sf_layout_match.group("layout_sfb_tv_dim1_legacy"))
        entry = Entry(
            tid=current_tid,
            n_tile=int(groups["n_tile"]),
            k_block=int(groups["k_block"]),
            dispatch_stages=-1 if groups["dispatch_stages"] is None else int(groups["dispatch_stages"]),
            dispatch_scheduler_stages=(
                -1 if groups["dispatch_scheduler_stages"] is None else int(groups["dispatch_scheduler_stages"])
            ),
            tcrb_k_blocks=-1 if groups["tcrb_k_blocks"] is None else int(groups["tcrb_k_blocks"]),
            tcrb_copy_view_k_blocks=(
                -1 if groups["tcrb_copy_view_k_blocks"] is None else int(groups["tcrb_copy_view_k_blocks"])
            ),
            tcsb_coord_k_blocks=-1 if groups["tcsb_coord_k_blocks"] is None else int(groups["tcsb_coord_k_blocks"]),
            tcrsfb_k_blocks=(
                -1
                if sf_match is None
                else int(sf_match.group("tcrsfb_k_blocks"))
            ),
            tcrsfb_copy_view_k_blocks=(
                -1
                if sf_match is None
                else int(sf_match.group("tcrsfb_copy_view_k_blocks"))
            ),
            tcssfb_coord_k_blocks=(
                -1
                if sf_match is None
                else int(sf_match.group("tcssfb_coord_k_blocks"))
            ),
            sf_vec_size=sf_vec_size,
            smem_layout_atom_sfb_dim0=smem_layout_atom_sfb_dim0,
            smem_layout_atom_sfb_dim1=smem_layout_atom_sfb_dim1,
            sfb_stage_elems=sfb_stage_elems,
            sfb_total_elems=sfb_total_elems,
            sfb_stage_shape_dim0=sfb_stage_shape_dim0,
            sfb_stage_shape_dim1=sfb_stage_shape_dim1,
            layout_sfb_tv_dim0=layout_sfb_tv_dim0,
            layout_sfb_tv_dim1=layout_sfb_tv_dim1,
            part_c_row0=int(groups["part_output_col0"]),
            part_c_col0=int(groups["part_token_row0"]),
            local_row0=int(groups["local_row0"]),
            local_col0=int(groups["local_col0"]),
            local_row1=int(groups["local_row1"]),
            local_col1=int(groups["local_col1"]),
            stage0_offset0=int(groups["stage0_offset0"]),
            stage0_offset1=int(groups["stage0_offset1"]),
            reg0_pre=int(groups["reg0_pre"], 16),
            reg1_pre=int(groups["reg1_pre"], 16),
        )
        entries[entry.key] = entry
    return entries


def describe_entry(prefix: str, entry: Entry) -> str:
    return (
        f"{prefix} tid={entry.tid} n_tile={entry.n_tile} k_block={entry.k_block} "
        f"dispatch=({entry.dispatch_stages},{entry.dispatch_scheduler_stages}) "
        f"k_counts=({entry.tcrb_k_blocks},{entry.tcrb_copy_view_k_blocks},{entry.tcsb_coord_k_blocks}) "
        f"sf_counts=({entry.tcrsfb_k_blocks},{entry.tcrsfb_copy_view_k_blocks},{entry.tcssfb_coord_k_blocks}) "
        f"sf_atom=({entry.sf_vec_size},{entry.smem_layout_atom_sfb_dim0},{entry.smem_layout_atom_sfb_dim1}) "
        f"sf_layout=({entry.sfb_stage_elems},{entry.sfb_total_elems},"
        f"{entry.sfb_stage_shape_dim0},{entry.sfb_stage_shape_dim1},"
        f"{entry.layout_sfb_tv_dim0},{entry.layout_sfb_tv_dim1}) "
        f"part_c0=({entry.part_c_row0},{entry.part_c_col0}) "
        f"local0=({entry.local_row0},{entry.local_col0}) "
        f"local1=({entry.local_row1},{entry.local_col1}) "
        f"stage0_offsets=({entry.stage0_offset0},{entry.stage0_offset1}) "
        f"reg_pre=(0x{entry.reg0_pre:08x},0x{entry.reg1_pre:08x})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", required=True, type=pathlib.Path)
    parser.add_argument("--runtime", required=True, type=pathlib.Path)
    parser.add_argument("--show-limit", type=int, default=8)
    args = parser.parse_args()

    live = parse_live(args.live)
    runtime = parse_runtime(args.runtime)

    live_keys = set(live)
    runtime_keys = set(runtime)
    shared_keys = sorted(live_keys & runtime_keys, key=lambda key: (key.tid, key.n_tile, key.k_block))

    print(f"live_entries={len(live)} runtime_entries={len(runtime)} shared_entries={len(shared_keys)}")
    missing_in_runtime = sorted(live_keys - runtime_keys, key=lambda key: (key.tid, key.n_tile, key.k_block))
    missing_in_live = sorted(runtime_keys - live_keys, key=lambda key: (key.tid, key.n_tile, key.k_block))
    print(f"missing_in_runtime={len(missing_in_runtime)} missing_in_live={len(missing_in_live)}")

    coord_mismatches: list[tuple[Key, Entry, Entry]] = []
    dispatch_mismatches: list[tuple[Key, Entry, Entry]] = []
    k_count_mismatches: list[tuple[Key, Entry, Entry]] = []
    sf_count_mismatches: list[tuple[Key, Entry, Entry]] = []
    sf_atom_mismatches: list[tuple[Key, Entry, Entry]] = []
    sf_layout_mismatches: list[tuple[Key, Entry, Entry]] = []
    reg_mismatches: list[tuple[Key, Entry, Entry]] = []
    for key in shared_keys:
        live_entry = live[key]
        runtime_entry = runtime[key]
        if (
            live_entry.part_c_row0,
            live_entry.part_c_col0,
            live_entry.local_row0,
            live_entry.local_col0,
            live_entry.local_row1,
            live_entry.local_col1,
            live_entry.stage0_offset0,
            live_entry.stage0_offset1,
        ) != (
            runtime_entry.part_c_row0,
            runtime_entry.part_c_col0,
            runtime_entry.local_row0,
            runtime_entry.local_col0,
            runtime_entry.local_row1,
            runtime_entry.local_col1,
            runtime_entry.stage0_offset0,
            runtime_entry.stage0_offset1,
        ):
            coord_mismatches.append((key, live_entry, runtime_entry))
        if (
            live_entry.dispatch_stages,
            live_entry.dispatch_scheduler_stages,
        ) != (
            runtime_entry.dispatch_stages,
            runtime_entry.dispatch_scheduler_stages,
        ):
            dispatch_mismatches.append((key, live_entry, runtime_entry))
        if (
            live_entry.tcrb_k_blocks,
            live_entry.tcrb_copy_view_k_blocks,
            live_entry.tcsb_coord_k_blocks,
        ) != (
            runtime_entry.tcrb_k_blocks,
            runtime_entry.tcrb_copy_view_k_blocks,
            runtime_entry.tcsb_coord_k_blocks,
        ):
            k_count_mismatches.append((key, live_entry, runtime_entry))
        if (
            live_entry.tcrsfb_k_blocks,
            live_entry.tcrsfb_copy_view_k_blocks,
            live_entry.tcssfb_coord_k_blocks,
        ) != (
            runtime_entry.tcrsfb_k_blocks,
            runtime_entry.tcrsfb_copy_view_k_blocks,
            runtime_entry.tcssfb_coord_k_blocks,
        ):
            sf_count_mismatches.append((key, live_entry, runtime_entry))
        if (
            live_entry.sf_vec_size,
            live_entry.smem_layout_atom_sfb_dim0,
            live_entry.smem_layout_atom_sfb_dim1,
        ) != (
            runtime_entry.sf_vec_size,
            runtime_entry.smem_layout_atom_sfb_dim0,
            runtime_entry.smem_layout_atom_sfb_dim1,
        ):
            sf_atom_mismatches.append((key, live_entry, runtime_entry))
        if (
            live_entry.sfb_stage_elems,
            live_entry.sfb_total_elems,
            live_entry.sfb_stage_shape_dim0,
            live_entry.sfb_stage_shape_dim1,
            live_entry.layout_sfb_tv_dim0,
            live_entry.layout_sfb_tv_dim1,
        ) != (
            runtime_entry.sfb_stage_elems,
            runtime_entry.sfb_total_elems,
            runtime_entry.sfb_stage_shape_dim0,
            runtime_entry.sfb_stage_shape_dim1,
            runtime_entry.layout_sfb_tv_dim0,
            runtime_entry.layout_sfb_tv_dim1,
        ):
            sf_layout_mismatches.append((key, live_entry, runtime_entry))
        if (live_entry.reg0_pre, live_entry.reg1_pre) != (runtime_entry.reg0_pre, runtime_entry.reg1_pre):
            reg_mismatches.append((key, live_entry, runtime_entry))

    print(f"coord_mismatches={len(coord_mismatches)}/{len(shared_keys)}")
    print(f"dispatch_mismatches={len(dispatch_mismatches)}/{len(shared_keys)}")
    print(f"k_count_mismatches={len(k_count_mismatches)}/{len(shared_keys)}")
    print(f"sf_count_mismatches={len(sf_count_mismatches)}/{len(shared_keys)}")
    print(f"sf_atom_mismatches={len(sf_atom_mismatches)}/{len(shared_keys)}")
    print(f"sf_layout_mismatches={len(sf_layout_mismatches)}/{len(shared_keys)}")
    print(f"reg_mismatches={len(reg_mismatches)}/{len(shared_keys)}")

    for label, mismatches in (
        ("coord", coord_mismatches),
        ("dispatch", dispatch_mismatches),
        ("k_count", k_count_mismatches),
        ("sf_count", sf_count_mismatches),
        ("sf_atom", sf_atom_mismatches),
        ("sf_layout", sf_layout_mismatches),
        ("reg", reg_mismatches),
    ):
        for idx, (_, live_entry, runtime_entry) in enumerate(mismatches[: args.show_limit], start=1):
            print(f"{label}_mismatch[{idx}]")
            print(describe_entry("  live   ", live_entry))
            print(describe_entry("  runtime", runtime_entry))

    return 0


if __name__ == "__main__":
    sys.exit(main())
