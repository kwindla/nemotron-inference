#!/usr/bin/env python3

"""Regression coverage for the current Nemotron 3 Super manifest contract.

This test is intentionally checkpoint-family-specific. It builds a tiny
checkpoint-shaped safetensors shard that follows the current Nemotron 3 Super
NVFP4 routing layout:

- routed expert `up_proj` / `down_proj` weights carry raw `weight_scale_2`
  through the generic NVFP4 `tensor_scale` auxiliary
- routed expert standalone `input_scale` tensors stay visible as separate
  manifest entries

The exact routed standalone count asserted here is only for this current
Nemotron 3 Super layout slice, not a generic invariant for every architecture.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "tools" / "manifest" / "generate_forward_manifest.py"

ROUTED_PREFIX = "backbone.layers.1.mixer.experts.0"
SHARED_DOWN_PREFIX = "backbone.layers.1.mixer.shared_experts.down_proj"

EXPECTED_ROUTED_STANDALONE_INPUT_SCALE_NAMES = {
    f"{ROUTED_PREFIX}.up_proj.input_scale",
    f"{ROUTED_PREFIX}.down_proj.input_scale",
}
EXPECTED_ROUTED_STANDALONE_INPUT_SCALE_COUNT = len(
    EXPECTED_ROUTED_STANDALONE_INPUT_SCALE_NAMES
)

ROUTED_RUNTIME_FUSION_BOUNDARY = (
    "runtime/cache may later fuse standalone input_scale with raw weight_scale_2"
)
SHARED_DOWN_RUNTIME_FUSION_BOUNDARY = (
    "tensor_scale remains raw weight_scale_2 with no separate input_scale fusion"
)


def dtype_nbytes(dtype: str) -> int:
    sizes = {
        "F32": 4,
        "BF16": 2,
        "F16": 2,
        "I32": 4,
        "U8": 1,
        "F8_E4M3FN": 1,
        "F8_E4M3": 1,
    }
    if dtype not in sizes:
        raise KeyError(f"unsupported test dtype {dtype!r}")
    return sizes[dtype]


def tensor_nbytes(dtype: str, shape: list[int]) -> int:
    element_count = math.prod(shape) if shape else 1
    return element_count * dtype_nbytes(dtype)


def write_fake_safetensors_shard(
    path: Path, tensors: list[tuple[str, str, list[int]]]
) -> None:
    header: dict[str, dict[str, object]] = {}
    data = bytearray()
    cursor = 0
    for name, dtype, shape in tensors:
        nbytes = tensor_nbytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + nbytes],
        }
        data.extend(bytes([len(data) % 251]) * nbytes)
        cursor += nbytes

    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)


def build_checkpoint_slice(model_dir: Path) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    shard_path = model_dir / shard_name

    tensors = [
        (f"{ROUTED_PREFIX}.up_proj.weight", "U8", [2, 8]),
        (f"{ROUTED_PREFIX}.up_proj.weight_scale", "F8_E4M3FN", [2, 1]),
        (f"{ROUTED_PREFIX}.up_proj.weight_scale_2", "F32", []),
        (f"{ROUTED_PREFIX}.up_proj.input_scale", "F32", []),
        (f"{ROUTED_PREFIX}.down_proj.weight", "U8", [2, 8]),
        (f"{ROUTED_PREFIX}.down_proj.weight_scale", "F8_E4M3FN", [2, 1]),
        (f"{ROUTED_PREFIX}.down_proj.weight_scale_2", "F32", []),
        (f"{ROUTED_PREFIX}.down_proj.input_scale", "F32", []),
        (f"{SHARED_DOWN_PREFIX}.weight", "U8", [2, 8]),
        (f"{SHARED_DOWN_PREFIX}.weight_scale", "F8_E4M3FN", [2, 1]),
        (f"{SHARED_DOWN_PREFIX}.weight_scale_2", "F32", []),
        (f"{SHARED_DOWN_PREFIX}.input_scale", "F32", []),
    ]
    write_fake_safetensors_shard(shard_path, tensors)

    index = {
        "weight_map": {name: shard_name for name, _, _ in tensors},
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def is_routed_expert_standalone_input_scale(name: str) -> bool:
    return name.startswith("backbone.layers.") and ".mixer.experts." in name and (
        name.endswith(".up_proj.input_scale") or name.endswith(".down_proj.input_scale")
    )


def find_auxiliary(tensor: dict[str, object], auxiliary_name: str) -> dict[str, object]:
    for auxiliary in tensor.get("auxiliaries", []):
        if auxiliary.get("name") == auxiliary_name:
            return auxiliary
    raise KeyError(f"missing {auxiliary_name!r} auxiliary on tensor {tensor['name']!r}")


def without_weight_suffix(name: str) -> str:
    suffix = ".weight"
    if not name.endswith(suffix):
        raise ValueError(f"expected NVFP4 weight tensor name, found {name!r}")
    return name[: -len(suffix)]


class ForwardManifestCurrentCheckpointRegressionTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(
            prefix="nemotron_forward_manifest_regression_"
        )
        cls.model_dir = Path(cls.temp_dir.name)
        build_checkpoint_slice(cls.model_dir)
        cls.manifest_path = cls.model_dir / "forward_manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--model-dir",
                str(cls.model_dir),
                "--output-manifest",
                str(cls.manifest_path),
                "--source-revision",
                "test-source-revision",
                "--tokenizer-revision",
                "test-tokenizer-revision",
                "--packer-version",
                "forward-manifest-regression-test",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.tensors_by_name = {
            tensor["name"]: tensor for tensor in cls.manifest.get("tensors", [])
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_routed_standalone_input_scales_remain_visible(self) -> None:
        routed_input_scales = {
            tensor_name
            for tensor_name in self.tensors_by_name
            if is_routed_expert_standalone_input_scale(tensor_name)
        }
        self.assertGreater(
            len(routed_input_scales),
            0,
            "current Nemotron 3 Super checkpoint family should keep routed standalone input_scale entries in the manifest",
        )
        self.assertEqual(
            routed_input_scales,
            EXPECTED_ROUTED_STANDALONE_INPUT_SCALE_NAMES,
            "checkpoint-family-specific routed standalone input_scale set changed",
        )
        self.assertEqual(
            len(routed_input_scales),
            EXPECTED_ROUTED_STANDALONE_INPUT_SCALE_COUNT,
            "checkpoint-family-specific routed standalone input_scale count changed",
        )

        routed_weight_scale_2_entries = [
            tensor_name
            for tensor_name in self.tensors_by_name
            if ".mixer.experts." in tensor_name and tensor_name.endswith(".weight_scale_2")
        ]
        self.assertEqual(
            routed_weight_scale_2_entries,
            [],
            "routed raw weight_scale_2 should stay attached to NVFP4 weight entries rather than appear as standalone manifest tensors",
        )

    def test_nvfp4_weight_entries_expose_step6_provenance_metadata(self) -> None:
        routed_weight_names = [
            f"{ROUTED_PREFIX}.up_proj.weight",
            f"{ROUTED_PREFIX}.down_proj.weight",
        ]
        for weight_name in routed_weight_names:
            tensor = self.tensors_by_name[weight_name]
            tensor_scale = find_auxiliary(tensor, "tensor_scale")
            weight_prefix = without_weight_suffix(weight_name)
            expected_source_name = weight_prefix + ".weight_scale_2"
            expected_input_scale_name = weight_prefix + ".input_scale"
            self.assertEqual(
                tensor_scale.get("source_tensor_name"),
                expected_source_name,
                "routed NVFP4 tensor_scale auxiliary should point at raw checkpoint weight_scale_2",
            )
            self.assertEqual(
                tensor_scale.get("provenance"),
                "raw_weight_scale_2",
                "routed NVFP4 tensor_scale auxiliary should expose explicit raw provenance",
            )
            self.assertEqual(
                tensor_scale.get("manifest_semantics"),
                "checkpoint_oriented",
                "routed NVFP4 tensor_scale auxiliary should declare checkpoint-oriented semantics",
            )
            self.assertEqual(
                tensor_scale.get("separate_manifest_input_scale"),
                expected_input_scale_name,
                "routed NVFP4 tensor_scale auxiliary should record the separate standalone input_scale entry",
            )
            self.assertEqual(
                tensor_scale.get("runtime_fusion_boundary"),
                ROUTED_RUNTIME_FUSION_BOUNDARY,
                "routed NVFP4 tensor_scale auxiliary should document the runtime/cache fusion boundary",
            )

        shared_down_tensor = self.tensors_by_name[f"{SHARED_DOWN_PREFIX}.weight"]
        shared_down_tensor_scale = find_auxiliary(shared_down_tensor, "tensor_scale")
        self.assertEqual(
            shared_down_tensor_scale.get("source_tensor_name"),
            f"{SHARED_DOWN_PREFIX}.weight_scale_2",
            "shared-down NVFP4 tensor_scale auxiliary should point at raw checkpoint weight_scale_2",
        )
        self.assertEqual(
            shared_down_tensor_scale.get("provenance"),
            "raw_weight_scale_2",
            "shared-down NVFP4 tensor_scale auxiliary should expose explicit raw provenance",
        )
        self.assertEqual(
            shared_down_tensor_scale.get("manifest_semantics"),
            "checkpoint_oriented",
            "shared-down NVFP4 tensor_scale auxiliary should declare checkpoint-oriented semantics",
        )
        self.assertNotIn(
            "separate_manifest_input_scale",
            shared_down_tensor_scale,
            "shared-down NVFP4 tensor_scale auxiliary should not claim routed standalone input_scale fusion metadata",
        )
        self.assertEqual(
            shared_down_tensor_scale.get("runtime_fusion_boundary"),
            SHARED_DOWN_RUNTIME_FUSION_BOUNDARY,
            "shared-down NVFP4 tensor_scale auxiliary should preserve the raw weight_scale_2 boundary description",
        )


if __name__ == "__main__":
    unittest.main()
