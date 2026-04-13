#!/usr/bin/env python3
"""Dump `golden/inputs.pt` tensors to raw `.bin` files.

Step 4c's C++ oracle test needs to load the step-2 capture inputs without
pulling libtorch into the runtime test target (libtorch's `find_package`
clobbers `CMAKE_CUDA_FLAGS` and strips the `a` arch variant from sibling
CUDA targets). Dumping raw `.bin` files here makes the test a pure
C++/CUDA program that reads uint8/int32/float32/bfloat16 bytes straight
from disk.

Usage:
    .venv-trtllm/bin/python proj-2026-04-12-1022/trtllm_reference/dump_inputs_bin.py

Output:
    golden/inputs_<key>.bin  (one per tensor in inputs.pt)
    golden/inputs_bin_manifest.json  (shape/dtype/byte-count manifest)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "golden",
        help="Directory containing inputs.pt and where raw .bin dumps are written.",
    )
    args = parser.parse_args()

    golden = args.golden_dir
    inputs_pt = golden / "inputs.pt"
    if not inputs_pt.exists():
        raise SystemExit(f"missing {inputs_pt} — run run_capture.sh first")

    inputs = torch.load(str(inputs_pt), weights_only=False)

    manifest = {
        "generated_by": "proj-2026-04-12-1022/trtllm_reference/dump_inputs_bin.py",
        "source_pt": "inputs.pt",
        "tensors": {},
    }

    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach().cpu().contiguous()
        # bf16 is not directly supported by numpy; view as uint8 to preserve bits.
        byte_view = t.view(torch.uint8) if t.dtype == torch.bfloat16 else t
        try:
            raw = byte_view.numpy().tobytes()
        except TypeError:
            raw = bytes(byte_view.untyped_storage())
        bin_path = golden / f"inputs_{k}.bin"
        bin_path.write_bytes(raw)
        manifest["tensors"][k] = {
            "file": f"inputs_{k}.bin",
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "element_size": t.element_size(),
            "nbytes": t.element_size() * t.numel(),
        }

    (golden / "inputs_bin_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for k, meta in sorted(manifest["tensors"].items()):
        print(f"  {meta['file']}: {meta['dtype']} shape={meta['shape']} bytes={meta['nbytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
