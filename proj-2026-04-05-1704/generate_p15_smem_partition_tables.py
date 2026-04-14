#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import re
import subprocess
from pathlib import Path


THREAD_COUNT = 256
STAGE0_SIZE = 128


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_include_roots(root: Path) -> list[Path]:
    candidates = [
        root / ".venv-trtllm/lib/python3.12/site-packages/flashinfer/data/cutlass/include",
    ]
    candidates.extend(
        Path(path)
        for path in glob.glob(
            str(Path.home() / ".cache/uv/archive-v0/*/tensorrt_llm/deep_gemm/include")
        )
    )
    return [path for path in candidates if path.exists()]


def compile_probe(root: Path, probe: Path, binary: Path) -> None:
    include_roots = find_include_roots(root)
    if not include_roots:
        raise RuntimeError("Unable to locate a CUTLASS/CUTE include root for the P15 smem probe")
    cmd = ["nvcc", "-std=c++20", str(probe), "-O0", "-g", "-o", str(binary)]
    for include_root in include_roots:
        cmd.extend(["-I", str(include_root)])
    cmd.extend(["-I", "/usr/local/cuda/include"])
    subprocess.run(cmd, cwd=root, check=True)


def parse_int(token: str) -> int:
    return 0 if token.startswith("_") else int(token)


def parse_probe_output(text: str) -> dict:
    meta: dict[str, int] = {}
    data = {
        "TCSA": [[None] * STAGE0_SIZE for _ in range(THREAD_COUNT)],
        "TCSB": [[None] * STAGE0_SIZE for _ in range(THREAD_COUNT)],
    }
    current_thread: int | None = None

    meta_re = re.compile(r"^META\s+(\w+)\s+_?(\d+)$")
    thread_re = re.compile(r"^THREAD\s+(\d+)$")
    coord_re = re.compile(
        r"^(TCSA|TCSB)\s+(\d+)\s+\(\(([-_0-9]+),([-_0-9]+)\),\(([-_0-9]+),([-_0-9]+)\),\(([-_0-9]+),([-_0-9]+)\)\)$"
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if match := meta_re.match(line):
            meta[match.group(1)] = int(match.group(2))
            continue
        if match := thread_re.match(line):
            current_thread = int(match.group(1))
            continue
        if current_thread is None:
            raise RuntimeError(f"Encountered payload before thread header: {line}")
        if match := coord_re.match(line):
            tag, idx, *vals = match.groups()
            data[tag][current_thread][int(idx)] = [parse_int(v) for v in vals]
            continue
        raise RuntimeError(f"Unrecognized probe line: {line}")

    expected_meta = {
        "threads": THREAD_COUNT,
        "tcsa_stage0_size": STAGE0_SIZE,
        "tcsb_stage0_size": STAGE0_SIZE,
    }
    for key, expected in expected_meta.items():
        actual = meta.get(key)
        if actual != expected:
            raise RuntimeError(f"Probe meta mismatch for {key}: expected {expected}, got {actual}")

    for tag, threads in data.items():
        for thread_idx, coords in enumerate(threads):
            missing = [idx for idx, coord in enumerate(coords) if coord is None]
            if missing:
                raise RuntimeError(f"Missing {tag} coords for thread {thread_idx}: {missing[:8]}")

    return {"meta": meta, "data": data}


def constant_delta(data: dict, tag: str, dst_thread: int, src_thread: int, component: int) -> int:
    dst = data[tag][dst_thread]
    src = data[tag][src_thread]
    deltas = {src[i][component] - dst[i][component] for i in range(len(dst))}
    if len(deltas) != 1:
        raise RuntimeError(
            f"{tag} delta for src={src_thread} dst={dst_thread} component={component} is not constant: {sorted(deltas)}"
        )
    return next(iter(deltas))


def emit_header(parsed: dict) -> str:
    meta = parsed["meta"]
    data = parsed["data"]

    def emit_base_block(name: str, tag: str) -> list[str]:
        lines = [f"inline constexpr P15StageCoord3 {name}[kP15LaneCount][kP15Stage0CoordCount] = {{"]
        for lane in range(32):
            coords = ", ".join(
                f"{{{{{c0}, {c1}}}, {{{c2}, {c3}}}, {{{c4}, {c5}}}}}"
                for c0, c1, c2, c3, c4, c5 in data[tag][lane]
            )
            lines.append(f"    {{{coords}}},")
        lines.append("};")
        return lines

    tcsa_deltas = [constant_delta(data, "TCSA", 0, 32, i) for i in range(6)]
    tcsa_upper = [constant_delta(data, "TCSA", 0, 128, i) for i in range(6)]
    tcsb_deltas = [constant_delta(data, "TCSB", 0, 32, i) for i in range(6)]
    tcsb_upper = [constant_delta(data, "TCSB", 0, 128, i) for i in range(6)]

    def emit_adjust(name: str, deltas: list[int], uppers: list[int]) -> list[str]:
        return [
            f"inline constexpr int {name}WarpMod4_0 = {deltas[0]};",
            f"inline constexpr int {name}WarpMod4_1 = {deltas[1]};",
            f"inline constexpr int {name}WarpMod4_2 = {deltas[2]};",
            f"inline constexpr int {name}WarpMod4_3 = {deltas[3]};",
            f"inline constexpr int {name}WarpMod4_4 = {deltas[4]};",
            f"inline constexpr int {name}WarpMod4_5 = {deltas[5]};",
            f"inline constexpr int {name}UpperHalf_0 = {uppers[0]};",
            f"inline constexpr int {name}UpperHalf_1 = {uppers[1]};",
            f"inline constexpr int {name}UpperHalf_2 = {uppers[2]};",
            f"inline constexpr int {name}UpperHalf_3 = {uppers[3]};",
            f"inline constexpr int {name}UpperHalf_4 = {uppers[4]};",
            f"inline constexpr int {name}UpperHalf_5 = {uppers[5]};",
        ]

    lines = [
        "// Generated by proj-2026-04-05-1704/generate_p15_smem_partition_tables.py",
        "// from proj-2026-04-05-1704/trt_p15_smem_partition_dump.cu.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "#define NEMOTRON_P15_SMEM_HD __host__ __device__",
        "",
        "namespace nemotron::p15_smem_partition_generated {",
        "",
        "struct P15Coord2 {",
        "  std::uint16_t x;",
        "  std::uint16_t y;",
        "};",
        "",
        "struct P15StageCoord3 {",
        "  P15Coord2 a;",
        "  P15Coord2 b;",
        "  P15Coord2 c;",
        "};",
        "",
        f"inline constexpr int kP15ThreadCount = {meta['threads']};",
        "inline constexpr int kP15LaneCount = 32;",
        f"inline constexpr int kP15Stage0CoordCount = {meta['tcsa_stage0_size']};",
        "",
    ]
    lines.extend(emit_base_block("kP15TCSAStage0Base", "TCSA"))
    lines.append("")
    lines.extend(emit_base_block("kP15TCSBStage0Base", "TCSB"))
    lines.append("")
    lines.extend(emit_adjust("kP15TCSA", tcsa_deltas, tcsa_upper))
    lines.append("")
    lines.extend(emit_adjust("kP15TCSB", tcsb_deltas, tcsb_upper))
    lines.extend([
        "",
        "NEMOTRON_P15_SMEM_HD constexpr int P15WarpMod4(int thread_idx) { return (thread_idx >> 5) & 0x3; }",
        "NEMOTRON_P15_SMEM_HD constexpr int P15WarpUpperHalf(int thread_idx) { return thread_idx >> 7; }",
        "",
        "NEMOTRON_P15_SMEM_HD constexpr P15StageCoord3 GetTCSAStage0Coord(int thread_idx, int logical_index) {",
        "  const auto base = kP15TCSAStage0Base[thread_idx & 31][logical_index];",
        "  const int warp = P15WarpMod4(thread_idx);",
        "  const int upper = P15WarpUpperHalf(thread_idx);",
        "  return {",
        "      {static_cast<std::uint16_t>(base.a.x + warp * kP15TCSAWarpMod4_0 + upper * kP15TCSAUpperHalf_0),",
        "       static_cast<std::uint16_t>(base.a.y + warp * kP15TCSAWarpMod4_1 + upper * kP15TCSAUpperHalf_1)},",
        "      {static_cast<std::uint16_t>(base.b.x + warp * kP15TCSAWarpMod4_2 + upper * kP15TCSAUpperHalf_2),",
        "       static_cast<std::uint16_t>(base.b.y + warp * kP15TCSAWarpMod4_3 + upper * kP15TCSAUpperHalf_3)},",
        "      {static_cast<std::uint16_t>(base.c.x + warp * kP15TCSAWarpMod4_4 + upper * kP15TCSAUpperHalf_4),",
        "       static_cast<std::uint16_t>(base.c.y + warp * kP15TCSAWarpMod4_5 + upper * kP15TCSAUpperHalf_5)},",
        "  };",
        "}",
        "",
        "NEMOTRON_P15_SMEM_HD constexpr P15StageCoord3 GetTCSBStage0Coord(int thread_idx, int logical_index) {",
        "  const auto base = kP15TCSBStage0Base[thread_idx & 31][logical_index];",
        "  const int warp = P15WarpMod4(thread_idx);",
        "  const int upper = P15WarpUpperHalf(thread_idx);",
        "  return {",
        "      {static_cast<std::uint16_t>(base.a.x + warp * kP15TCSBWarpMod4_0 + upper * kP15TCSBUpperHalf_0),",
        "       static_cast<std::uint16_t>(base.a.y + warp * kP15TCSBWarpMod4_1 + upper * kP15TCSBUpperHalf_1)},",
        "      {static_cast<std::uint16_t>(base.b.x + warp * kP15TCSBWarpMod4_2 + upper * kP15TCSBUpperHalf_2),",
        "       static_cast<std::uint16_t>(base.b.y + warp * kP15TCSBWarpMod4_3 + upper * kP15TCSBUpperHalf_3)},",
        "      {static_cast<std::uint16_t>(base.c.x + warp * kP15TCSBWarpMod4_4 + upper * kP15TCSBUpperHalf_4),",
        "       static_cast<std::uint16_t>(base.c.y + warp * kP15TCSBWarpMod4_5 + upper * kP15TCSBUpperHalf_5)},",
        "  };",
        "}",
        "",
        "}  // namespace nemotron::p15_smem_partition_generated",
        "",
        "#undef NEMOTRON_P15_SMEM_HD",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    root = repo_root()
    probe = Path(__file__).with_name("trt_p15_smem_partition_dump.cu")
    artifacts_tmp = root / "artifacts/tmp"
    artifacts_bench = root / "artifacts/benchmarks"
    artifacts_tmp.mkdir(parents=True, exist_ok=True)
    artifacts_bench.mkdir(parents=True, exist_ok=True)
    binary = artifacts_tmp / "trt_p15_smem_partition_dump_nvcc"
    log_path = artifacts_bench / "trt_p15_smem_partition_dump_latest.log"
    json_path = artifacts_tmp / "trt_p15_smem_partition_contract.json"
    header_path = root / "runtime/include/nemotron/p15_smem_partition_generated.h"

    compile_probe(root, probe, binary)
    output = subprocess.run([str(binary)], cwd=root, check=True, text=True, capture_output=True).stdout
    log_path.write_text(output)
    parsed = parse_probe_output(output)
    json_path.write_text(json.dumps(parsed, indent=2))
    header_path.write_text(emit_header(parsed))

    print(f"Wrote {log_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {header_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
