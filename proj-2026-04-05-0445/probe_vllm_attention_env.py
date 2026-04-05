#!/usr/bin/env python3

import contextlib
import io
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import torch
import vllm
from vllm.platforms import current_platform
from vllm.utils.flashinfer import supports_trtllm_attention


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bench_dir = repo_root / "third_party" / "vllm" / "benchmarks" / "attention_benchmarks"
    sys.path.insert(0, str(bench_dir))

    from common import BenchmarkConfig
    from runner import _create_vllm_config

    cfg = BenchmarkConfig(
        backend="FLASH_ATTN",
        batch_spec="q256",
        num_layers=6,
        head_dim=128,
        num_q_heads=32,
        num_kv_heads=2,
        block_size=16,
        device="cuda:0",
        kv_cache_dtype="auto",
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        vllm_config = _create_vllm_config(cfg, max_num_blocks=16)

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "py_executable": sys.executable,
        "py_path_prefix": sys.path[:5],
        "cwd": os.getcwd(),
        "py_env_pythonpath": os.environ.get("PYTHONPATH", ""),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_module": vllm.__file__,
        "current_platform_device_name": current_platform.device_name,
        "current_platform_is_cuda": current_platform.is_cuda(),
        "current_platform_capability": list(current_platform.get_device_capability()),
        "current_platform_family100": current_platform.is_device_capability_family(100),
        "supports_trtllm_attention": supports_trtllm_attention(),
        "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "resolved_vllm_benchmark_dtype": str(vllm_config.model_config.dtype),
    }

    for module_name in ("flashinfer", "flash_attn", "triton"):
        try:
            __import__(module_name)
            result[f"module_{module_name}"] = "import_ok"
        except Exception as exc:  # pragma: no cover - diagnostic helper
            result[f"module_{module_name}"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
