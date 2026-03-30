#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import tempfile


def build_probe_ptx(instruction: str) -> str:
    return f""".version 8.8
.target sm_121
.address_size 64

.visible .entry probe(
    .param .u64 in_ptr,
    .param .u64 out_ptr
)
{{
    .reg .f32 %f;
    .reg .b16 %h;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.global.f32 %f, [%rd1];
    {instruction}
    st.global.u16 [%rd2], %h;
    ret;
}}
"""


def run_ptxas(ptxas: str, instruction: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="sm121_sr_probe_") as tmpdir:
        tmp = pathlib.Path(tmpdir)
        ptx_path = tmp / "probe.ptx"
        cubin_path = tmp / "probe.cubin"
        ptx_path.write_text(build_probe_ptx(instruction), encoding="utf-8")
        command = [ptxas, "--gpu-name", "sm_121", str(ptx_path), "-o", str(cubin_path)]
        proc = subprocess.run(command, capture_output=True, text=True)
        return {
            "instruction": instruction,
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "supported": proc.returncode == 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    ptxas = shutil.which("ptxas")
    if ptxas is None:
        raise SystemExit("ptxas not found on PATH")

    version_proc = subprocess.run([ptxas, "--version"], capture_output=True, text=True)
    probes = [
        ("control_rn_f16", "cvt.rn.f16.f32 %h, %f;"),
        ("candidate_rs_f16", "cvt.rs.f16.f32 %h, %f;"),
    ]

    results = {}
    for name, instruction in probes:
        results[name] = run_ptxas(ptxas, instruction)

    summary = {
        "control_rn_f16_supported": results["control_rn_f16"]["supported"],
        "candidate_rs_f16_supported": results["candidate_rs_f16"]["supported"],
    }
    summary["conclusion"] = (
        "sm_121 ptxas accepted direct cvt.rs.f16.f32"
        if summary["candidate_rs_f16_supported"]
        else "sm_121 ptxas did not accept direct cvt.rs.f16.f32; no direct PTX fp32->fp16 stochastic-rounding conversion path was found in this probe"
    )

    artifact = {
        "environment": {
            "ptxas_path": ptxas,
            "ptxas_version_stdout": version_proc.stdout,
            "ptxas_version_stderr": version_proc.stderr,
            "cuda_home": os.environ.get("CUDA_HOME", ""),
        },
        "summary": summary,
        "results": results,
    }

    output_path = pathlib.Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
