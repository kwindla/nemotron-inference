#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/khkramer/src/nemotron-inference"
PROJ_REF_DIR="${REPO_ROOT}/proj-2026-04-12-1022/trtllm_reference"
GOLDEN_DIR="${PROJ_REF_DIR}/golden_nano_k2688"

export NEMOTRON_HARNESS_M=128
export NEMOTRON_HARNESS_K=2688
export NEMOTRON_HARNESS_N=1920
export NEMOTRON_HARNESS_E=1
export NEMOTRON_HARNESS_TOPK=1

bash "${PROJ_REF_DIR}/run_capture.sh" --golden-dir "${GOLDEN_DIR}"
"${REPO_ROOT}/vllm-env-cu128/bin/python" "${PROJ_REF_DIR}/dump_inputs_bin.py" --golden-dir "${GOLDEN_DIR}"

python3 - "${GOLDEN_DIR}" <<'PY'
import hashlib
import itertools
import pathlib
import struct
import sys

golden_dir = pathlib.Path(sys.argv[1])
tactic_paths = [golden_dir / f"bf16_gemm1_tactic{i}.bin" for i in range(8)]

def load_dump(path: pathlib.Path):
    data = path.read_bytes()
    if len(data) < 64:
        raise SystemExit(f"{path} is too small to contain the 64-byte dump header")
    if data[:8] != b"NEMOP1\x00\x00":
        raise SystemExit(f"{path} has an unexpected dump header")
    rows = struct.unpack_from("<I", data, 12)[0]
    cols = struct.unpack_from("<I", data, 16)[0]
    elem_bytes = struct.unpack_from("<I", data, 20)[0]
    body = data[64:]
    expected_bytes = rows * cols * elem_bytes
    if len(body) != expected_bytes:
        raise SystemExit(
            f"{path} body size mismatch: got={len(body)} expected={expected_bytes}"
        )
    return {
        "rows": rows,
        "cols": cols,
        "elem_bytes": elem_bytes,
        "body": body,
        "md5": hashlib.md5(data).hexdigest(),
    }

dumps = [(path.name, load_dump(path)) for path in tactic_paths]
distinct_md5s = sorted({entry["md5"] for _, entry in dumps})

def diff_count(lhs: bytes, rhs: bytes, elem_bytes: int) -> int:
    if len(lhs) != len(rhs):
        raise SystemExit("cannot diff dumps with different body sizes")
    mismatches = 0
    for offset in range(0, len(lhs), elem_bytes):
        if lhs[offset:offset + elem_bytes] != rhs[offset:offset + elem_bytes]:
            mismatches += 1
    return mismatches

lines = [
    "# Tactic Divergence Report",
    "",
    "Nano bucket: `M=128`, `K=2688`, `N=1920`, `n_experts=1`, `top_k=1`.",
    "",
    "## MD5",
    "",
]
for name, entry in dumps:
    lines.append(f"- `{name}`: `{entry['md5']}`")
lines.extend(
    [
        "",
        f"Distinct md5 count: `{len(distinct_md5s)}`",
        "",
        "## Pairwise Element Diff Counts",
        "",
    ]
)
for (lhs_name, lhs_entry), (rhs_name, rhs_entry) in itertools.combinations(dumps, 2):
    mismatches = diff_count(lhs_entry["body"], rhs_entry["body"], lhs_entry["elem_bytes"])
    lines.append(f"- `{lhs_name}` vs `{rhs_name}`: `{mismatches}` differing bf16 elements")

report_path = golden_dir / "tactic_divergence_report.md"
report_path.write_text("\n".join(lines) + "\n")

if len(distinct_md5s) < 2:
    raise SystemExit(
        "ERROR: all 8 tactic md5 hashes are identical at K=2688; stopping per step-5 gate"
    )
PY

md5sum "${GOLDEN_DIR}"/bf16_gemm1_tactic*.bin
echo "[run_capture_nano] wrote ${GOLDEN_DIR}/tactic_divergence_report.md"
