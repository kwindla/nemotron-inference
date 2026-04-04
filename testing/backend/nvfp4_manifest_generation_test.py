#!/usr/bin/env python3

import importlib.util
import tempfile
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = SOURCE_ROOT / "tools" / "manifest" / "generate_forward_manifest.py"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("generate_forward_manifest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    manifest = load_manifest_module()
    with tempfile.TemporaryDirectory(prefix="nemotron_nvfp4_manifest_generation_test_") as temp_dir:
        model_dir = Path(temp_dir)
        weight_map = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": "weights/expert_up_packed.safetensors",
            "backbone.layers.0.mixer.experts.0.up_proj.weight_scale": "weights/expert_up_scales.safetensors",
            "backbone.layers.0.mixer.experts.0.up_proj.weight_scale_2": "weights/expert_up_tensor_scale.safetensors",
        }
        shard_headers = {
            "weights/expert_up_packed.safetensors": {
                "header_size": 64,
                "tensors": {
                    "backbone.layers.0.mixer.experts.0.up_proj.weight": {
                        "dtype": "U8",
                        "shape": [1856, 1344],
                        "data_offsets": [0, 1856 * 1344],
                    }
                },
            },
            "weights/expert_up_scales.safetensors": {
                "header_size": 32,
                "tensors": {
                    "backbone.layers.0.mixer.experts.0.up_proj.weight_scale": {
                        "dtype": "F8_E4M3FN",
                        "shape": [1856, 168],
                        "data_offsets": [0, 1856 * 168],
                    }
                },
            },
            "weights/expert_up_tensor_scale.safetensors": {
                "header_size": 16,
                "tensors": {
                    "backbone.layers.0.mixer.experts.0.up_proj.weight_scale_2": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                },
            },
        }

        entry = manifest.build_nvfp4_entry(
            model_dir=model_dir,
            weight_map=weight_map,
            shard_headers=shard_headers,
            tensor_name="backbone.layers.0.mixer.experts.0.up_proj.weight",
            compute_checksums=False,
        )

        assert entry["storage_dtype"] == "nvfp4_e2m1"
        assert entry["layout_tag"] == "cublaslt_fp4_tn_v1"
        assert entry["packed_shape"] == [1856, 1344]
        assert entry["logical_shape"] == [1856, 2688]
        assert entry["packed_file"].endswith("weights/expert_up_packed.safetensors")
        assert entry["nbytes"] == 1856 * 1344
        assert [aux["name"] for aux in entry["auxiliaries"]] == ["block_scales", "tensor_scale"]
        assert entry["auxiliaries"][0]["file"].endswith("weights/expert_up_scales.safetensors")
        assert entry["auxiliaries"][0]["nbytes"] == 1856 * 168
        assert entry["auxiliaries"][1]["file"].endswith("weights/expert_up_tensor_scale.safetensors")
        assert entry["auxiliaries"][1]["nbytes"] == 4
        assert entry["source_tensor_name"] == "backbone.layers.0.mixer.experts.0.up_proj.weight"

    print("nvfp4_manifest_generation_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
