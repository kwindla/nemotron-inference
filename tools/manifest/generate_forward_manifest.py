#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DEFAULT_GPU_FAMILY = "RTX5090"
DEFAULT_COMPUTE_CAPABILITY = "12.0"
DEFAULT_SOURCE_REVISION = "b1ffe4992d7db6d768453a551a656b8d12c638fb"
DEFAULT_TOKENIZER_REVISION = DEFAULT_SOURCE_REVISION
CHECKSUM_PLACEHOLDER = "fnv1a64:0000000000000000"

RUNTIME_PROFILE = {
    "kv_bytes_per_token": 4096,
    "mamba_state_bytes_fp16": 87162880,
    "mamba_state_bytes_fp32": 174325760,
}

GLOBAL_TENSORS = {
    "backbone.embeddings.weight",
    "backbone.norm_f.weight",
    "lm_head.weight",
}

ATTENTION_SUFFIXES = {
    "norm.weight",
    "mixer.q_proj.weight",
    "mixer.k_proj.weight",
    "mixer.v_proj.weight",
    "mixer.o_proj.weight",
}

MAMBA_SUFFIXES = {
    "norm.weight",
    "mixer.norm.weight",
    "mixer.in_proj.weight",
    "mixer.in_proj.weight_scale",
    "mixer.in_proj.input_scale",
    "mixer.conv1d.weight",
    "mixer.conv1d.bias",
    "mixer.A_log",
    "mixer.D",
    "mixer.dt_bias",
    "mixer.out_proj.weight",
    "mixer.out_proj.weight_scale",
    "mixer.out_proj.input_scale",
}

EXPERT_STANDALONE_SUFFIXES = {
    "norm.weight",
    "mixer.gate.weight",
    "mixer.gate.e_score_correction_bias",
    "mixer.fc1_latent_proj.weight",
    "mixer.fc1_latent_proj.weight_scale",
    "mixer.fc1_latent_proj.input_scale",
    "mixer.fc2_latent_proj.weight",
    "mixer.shared_experts.up_proj.weight",
    "mixer.shared_experts.up_proj.weight_scale",
    "mixer.shared_experts.up_proj.input_scale",
    "mixer.shared_experts.down_proj.weight_scale",
    "mixer.shared_experts.down_proj.input_scale",
}

NVFP4_WEIGHT_SUFFIXES = {
    "mixer.in_proj.weight",
    "mixer.out_proj.weight",
    "mixer.shared_experts.up_proj.weight",
    "mixer.shared_experts.down_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a runtime-consumable forward-pass manifest for the Nemotron checkpoint."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--packer-version", default="forward-manifest-v1-dev")
    parser.add_argument(
        "--gpu-family",
        default=DEFAULT_GPU_FAMILY,
        help=(
            "Target GPU family label stored in manifest target_platform. "
            f"Default: {DEFAULT_GPU_FAMILY}."
        ),
    )
    parser.add_argument(
        "--compute-capability",
        default=DEFAULT_COMPUTE_CAPABILITY,
        help=(
            "Target compute capability stored in manifest target_platform as a dotted string "
            "(for example 12.0, 12.1). "
            f"Default: {DEFAULT_COMPUTE_CAPABILITY}."
        ),
    )
    parser.add_argument(
        "--compute-checksums",
        action="store_true",
        help="Compute real fnv1a64 checksums over each tensor range. This is much slower on the full 120B checkpoint.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_revisions(model_dir: Path, source_revision: str, tokenizer_revision: str) -> tuple[str, str]:
    if source_revision and tokenizer_revision:
        return source_revision, tokenizer_revision

    preflight_report = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "preflight"
        / "checkpoint_report.json"
    )
    inferred_source = source_revision
    inferred_tokenizer = tokenizer_revision
    if preflight_report.exists():
        payload = read_json(preflight_report)
        repo_sha = payload.get("repo_sha")
        if repo_sha:
            if not inferred_source:
                inferred_source = repo_sha
            if not inferred_tokenizer:
                inferred_tokenizer = repo_sha

    if not inferred_source:
        inferred_source = DEFAULT_SOURCE_REVISION
    if not inferred_tokenizer:
        inferred_tokenizer = inferred_source or DEFAULT_TOKENIZER_REVISION
    return inferred_source, inferred_tokenizer


def read_safetensors_header(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    return header_size, header


def load_checkpoint_index(
    model_dir: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    weight_map = read_json(model_dir / "model.safetensors.index.json")["weight_map"]
    shard_headers: dict[str, dict[str, Any]] = {}
    for shard_name in sorted(set(weight_map.values())):
        header_size, header = read_safetensors_header(model_dir / shard_name)
        shard_headers[shard_name] = {
            "header_size": header_size,
            "tensors": header,
        }
    return weight_map, shard_headers


def tensor_record(
    model_dir: Path,
    weight_map: dict[str, str],
    shard_headers: dict[str, dict[str, Any]],
    tensor_name: str,
) -> dict[str, Any]:
    shard_name = weight_map[tensor_name]
    shard_record = shard_headers[shard_name]
    metadata = shard_record["tensors"][tensor_name]
    header_size = int(shard_record["header_size"])
    start, end = metadata["data_offsets"]
    return {
        "name": tensor_name,
        "dtype": metadata["dtype"],
        "shape": list(metadata["shape"]),
        "file": str((model_dir / shard_name).resolve()),
        "offset_bytes": 8 + header_size + int(start),
        "nbytes": int(end) - int(start),
    }


def canonical_shape(shape: list[int]) -> list[int]:
    return shape if shape else [1]


def map_dense_dtype(dtype: str) -> str:
    mapping = {
        "F32": "fp32",
        "BF16": "bf16",
        "F16": "fp16",
        "I32": "int32",
        "U8": "uint8",
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dense dtype {dtype!r}")
    return mapping[dtype]


def map_scaled_fp8_dtype(dtype: str) -> str:
    if dtype not in {"F8_E4M3FN", "F8_E4M3"}:
        raise ValueError(f"expected F8_E4M3FN weight, found {dtype!r}")
    return "fp8_e4m3fn"


def is_layer_tensor(name: str) -> bool:
    return name.startswith("backbone.layers.")


def layer_suffix(name: str) -> str:
    prefix = "backbone.layers."
    rest = name[len(prefix) :]
    layer_index, dot, suffix = rest.partition(".")
    if not dot or not layer_index.isdigit():
        return ""
    return suffix


def is_routed_expert_nvfp4_weight(suffix: str) -> bool:
    return suffix.startswith("mixer.experts.") and (
        suffix.endswith(".up_proj.weight") or suffix.endswith(".down_proj.weight")
    )


def is_nvfp4_weight(name: str) -> bool:
    if not is_layer_tensor(name):
        return False
    suffix = layer_suffix(name)
    return suffix in NVFP4_WEIGHT_SUFFIXES or is_routed_expert_nvfp4_weight(suffix)


def is_needed_tensor(name: str) -> bool:
    if name in GLOBAL_TENSORS:
        return True
    if not is_layer_tensor(name):
        return False
    suffix = layer_suffix(name)
    return (
        suffix in ATTENTION_SUFFIXES
        or suffix in MAMBA_SUFFIXES
        or suffix in EXPERT_STANDALONE_SUFFIXES
        or suffix in NVFP4_WEIGHT_SUFFIXES
        or is_routed_expert_nvfp4_weight(suffix)
    )


def op_class_for_tensor(name: str) -> str:
    if name == "backbone.embeddings.weight":
        return "embedding"
    if name == "backbone.norm_f.weight":
        return "final_norm"
    if name == "lm_head.weight":
        return "logits"
    suffix = layer_suffix(name)
    if suffix.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")):
        return "attention"
    if suffix.endswith(("in_proj.weight", "out_proj.weight")):
        return "mamba_linear"
    if suffix in {"mixer.conv1d.weight", "mixer.conv1d.bias", "mixer.A_log", "mixer.D", "mixer.dt_bias"}:
        return "mamba_param"
    if suffix.endswith(("fc1_latent_proj.weight", "fc2_latent_proj.weight")):
        return "expert_linear"
    if suffix.endswith("mixer.gate.weight"):
        return "router"
    if suffix.endswith("mixer.gate.e_score_correction_bias"):
        return "router_bias"
    if suffix.endswith("mixer.shared_experts.up_proj.weight"):
        return "shared_expert_up"
    if suffix.endswith("mixer.shared_experts.down_proj.weight"):
        return "shared_expert_down"
    if ".experts." in suffix and suffix.endswith(".up_proj.weight"):
        return "routed_expert_up"
    if ".experts." in suffix and suffix.endswith(".down_proj.weight"):
        return "routed_expert_down"
    if suffix.endswith(".weight_scale") or suffix.endswith(".input_scale"):
        return "scale"
    if suffix.endswith("norm.weight"):
        return "norm"
    return "parameter"


def compute_fnv1a64(path: Path, offset_bytes: int, nbytes: int) -> str:
    fnv_offset_basis = 1469598103934665603
    fnv_prime = 1099511628211
    hash_value = fnv_offset_basis
    with path.open("rb") as handle:
        handle.seek(offset_bytes)
        remaining = nbytes
        while remaining > 0:
            chunk = handle.read(min(1 << 20, remaining))
            if not chunk:
                raise RuntimeError(f"short read while hashing {path}")
            for value in chunk:
                hash_value ^= value
                hash_value = (hash_value * fnv_prime) & 0xFFFFFFFFFFFFFFFF
            remaining -= len(chunk)
    return f"fnv1a64:{hash_value:016x}"


def make_checksum(path: Path, offset_bytes: int, nbytes: int, compute_checksums: bool) -> str:
    if not compute_checksums:
        return CHECKSUM_PLACEHOLDER
    return compute_fnv1a64(path, offset_bytes, nbytes)


def build_dense_entry(
    model_dir: Path,
    weight_map: dict[str, str],
    shard_headers: dict[str, dict[str, Any]],
    tensor_name: str,
    compute_checksums: bool,
) -> dict[str, Any]:
    record = tensor_record(model_dir, weight_map, shard_headers, tensor_name)
    storage_dtype = map_dense_dtype(record["dtype"])
    compute_dtype = "bf16" if storage_dtype == "bf16" else "fp32"
    return {
        "name": tensor_name,
        "op_class": op_class_for_tensor(tensor_name),
        "logical_shape": canonical_shape(record["shape"]),
        "packed_shape": canonical_shape(record["shape"]),
        "storage_dtype": storage_dtype,
        "compute_dtype": compute_dtype,
        "layout_tag": "row_major",
        "alignment_bytes": 16,
        "packed_file": record["file"],
        "offset_bytes": record["offset_bytes"],
        "nbytes": record["nbytes"],
        "source_tensor_name": tensor_name,
        "checksum": make_checksum(
            Path(record["file"]), record["offset_bytes"], record["nbytes"], compute_checksums
        ),
    }


def build_scaled_fp8_entry(
    model_dir: Path,
    weight_map: dict[str, str],
    shard_headers: dict[str, dict[str, Any]],
    tensor_name: str,
    compute_checksums: bool,
) -> dict[str, Any]:
    record = tensor_record(model_dir, weight_map, shard_headers, tensor_name)
    return {
        "name": tensor_name,
        "op_class": op_class_for_tensor(tensor_name),
        "logical_shape": canonical_shape(record["shape"]),
        "packed_shape": canonical_shape(record["shape"]),
        "storage_dtype": map_scaled_fp8_dtype(record["dtype"]),
        "compute_dtype": "fp32",
        "layout_tag": "row_major",
        "alignment_bytes": 16,
        "packed_file": record["file"],
        "offset_bytes": record["offset_bytes"],
        "nbytes": record["nbytes"],
        "source_tensor_name": tensor_name,
        "checksum": make_checksum(
            Path(record["file"]), record["offset_bytes"], record["nbytes"], compute_checksums
        ),
    }


def build_nvfp4_entry(
    model_dir: Path,
    weight_map: dict[str, str],
    shard_headers: dict[str, dict[str, Any]],
    tensor_name: str,
    compute_checksums: bool,
) -> dict[str, Any]:
    weight = tensor_record(model_dir, weight_map, shard_headers, tensor_name)
    block_scales_name = tensor_name[:-len(".weight")] + ".weight_scale"
    tensor_scale_candidates = [
        tensor_name[:-len(".weight")] + ".weight_scale_2",
        tensor_name[:-len(".weight")] + ".input_scale",
    ]
    tensor_scale_name = next(
        (candidate for candidate in tensor_scale_candidates if candidate in weight_map),
        "",
    )
    if not tensor_scale_name:
        raise KeyError(f"missing tensor-scale auxiliary for NVFP4 tensor {tensor_name}")
    block_scales = tensor_record(model_dir, weight_map, shard_headers, block_scales_name)
    tensor_scale = tensor_record(model_dir, weight_map, shard_headers, tensor_scale_name)
    logical_cols = block_scales["shape"][1] * 16
    logical_shape = [weight["shape"][0], logical_cols]
    return {
        "name": tensor_name,
        "op_class": op_class_for_tensor(tensor_name),
        "logical_shape": logical_shape,
        "packed_shape": weight["shape"],
        "storage_dtype": "nvfp4_e2m1",
        "compute_dtype": "fp32_accum",
        "layout_tag": "cublaslt_fp4_tn_v1",
        "alignment_bytes": 16,
        "block_scale_mode": "vec16_e4m3",
        "block_scale_dtype": "fp8_e4m3fn",
        "tensor_scale_dtype": "fp32",
        "packed_file": weight["file"],
        "offset_bytes": weight["offset_bytes"],
        "nbytes": weight["nbytes"],
        "auxiliaries": [
            {
                "name": "block_scales",
                "file": block_scales["file"],
                "offset_bytes": block_scales["offset_bytes"],
                "nbytes": block_scales["nbytes"],
            },
            {
                "name": "tensor_scale",
                "file": tensor_scale["file"],
                "offset_bytes": tensor_scale["offset_bytes"],
                "nbytes": tensor_scale["nbytes"],
            },
        ],
        "source_tensor_name": tensor_name,
        "checksum": make_checksum(
            Path(weight["file"]), weight["offset_bytes"], weight["nbytes"], compute_checksums
        ),
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir).resolve()
    weight_map, shard_headers = load_checkpoint_index(model_dir)
    source_revision, tokenizer_revision = infer_revisions(
        model_dir, args.source_revision, args.tokenizer_revision
    )

    tensors = []
    for tensor_name in sorted(weight_map):
        if not is_needed_tensor(tensor_name):
            continue

        suffix = layer_suffix(tensor_name)
        if suffix in {
            "mixer.shared_experts.down_proj.weight_scale",
            "mixer.shared_experts.down_proj.input_scale",
        }:
            shared_down_weight = tensor_record(
                model_dir,
                weight_map,
                shard_headers,
                tensor_name.rsplit(".", 1)[0] + ".weight",
            )
            if shared_down_weight["dtype"] == "U8":
                continue

        record = tensor_record(model_dir, weight_map, shard_headers, tensor_name)
        if is_nvfp4_weight(tensor_name) and record["dtype"] == "U8":
            tensors.append(
                build_nvfp4_entry(
                    model_dir, weight_map, shard_headers, tensor_name, args.compute_checksums
                )
            )
            continue

        if record["dtype"] in {"F8_E4M3FN", "F8_E4M3"}:
            tensors.append(
                build_scaled_fp8_entry(
                    model_dir, weight_map, shard_headers, tensor_name, args.compute_checksums
                )
            )
        else:
            tensors.append(
                build_dense_entry(
                    model_dir, weight_map, shard_headers, tensor_name, args.compute_checksums
                )
            )

    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "source_revision": source_revision,
        "tokenizer_revision": tokenizer_revision,
        "packer_version": args.packer_version,
        "target_platform": {
            "gpu_family": args.gpu_family,
            "compute_capability": args.compute_capability,
        },
        "runtime_profile": dict(RUNTIME_PROFILE),
        "tensors": tensors,
    }


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_manifest": str(output_path.resolve()),
                "tensor_count": len(manifest["tensors"]),
                "compute_checksums": args.compute_checksums,
                "source_revision": manifest["source_revision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
