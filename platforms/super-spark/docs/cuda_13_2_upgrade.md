# CUDA 13.2 Upgrade on DGX Spark

Date: 2026-03-28. Updated: 2026-03-29 (cuDNN install confirmed).

## Installed State (after upgrade)

- DGX OS: 7.4.0
- Driver: 580.142 (nvidia-driver-580-open) — unchanged, using cuda-compat
- CUDA Toolkit: **13.2** (V13.2.51) at `/usr/local/cuda-13.2/`
- cuBLAS: **13.3.0.5** (3x NVFP4/MXFP8 DGX Spark improvement)
- cuDNN: **9.20.0** (headers at `/usr/include/aarch64-linux-gnu/cudnn.h`, lib at `/usr/lib/aarch64-linux-gnu/libcudnn.so.9.20.0`)
- cuDNN Frontend: HEAD `6943af9` (March 2026) at `third_party/cudnn-frontend/`
- cuda-compat: libcuda.so 595.58.03 at `/usr/local/cuda-13.2/compat/`
- Kernel: 6.17.0-1014-nvidia (aarch64)

## Previous State (before upgrade)

- CUDA Toolkit: 13.0.2
- cuBLAS: 13.1.0.3
- cuDNN: not installed

## Target

CUDA 13.2 toolkit + cuBLAS 13.3.0.5 (the version with 3x NVFP4/MXFP8 improvement on DGX Spark).

Keep driver at 580.142. Do NOT upgrade to 590 or 595.

## Why Not Upgrade the Driver

- Driver 590 has a confirmed UMA memory leak on DGX Spark (~80 GB not released after CUDA process exit). Forum: https://forums.developer.nvidia.com/t/driver-590-48-01-regression-uma-memory-not-released-after-cuda-process-exit-works-on-580-126-09/359969
- NVIDIA forum moderator: "driver 590 and HWE kernels are not yet supported on the Spark"
- Driver 595 is not yet validated for DGX Spark. It exists in cuda-compute-repo as beta but is not in the DGX OS update channel.
- NVIDIA employee @aplattner recommended waiting for the official DGX OS update channel.
- Next DGX OS release (7.5) expected around August 2026, likely bundling driver 595 + CUDA 13.2.

## Why cuda-compat Works

CUDA forward compatibility allows a newer toolkit to run on an older kernel driver by providing user-mode CUDA driver libraries (libcuda.so, libnvidia-ptxjitcompiler.so) from driver 595, installed to `/usr/local/cuda-13.2/compat/`. The kernel module stays at 580.

- CUDA 13.x minor version compatibility requires driver >= 580 (satisfied).
- The cuBLAS 3x NVFP4/MXFP8 improvement is library-side, not driver-dependent.
- Only known limitation: OpenGL/Vulkan interop breaks (irrelevant for inference).

Forum reference: https://forums.developer.nvidia.com/t/using-cuda-13-2-on-the-dgx-spark/364305

## Install Commands

```bash
sudo apt update
sudo apt install cuda-toolkit-13-2 cuda-compat-13-2
```

No reboot required.

## Post-Install: Switch Active Toolkit

```bash
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-13.2 132
sudo update-alternatives --set cuda /usr/local/cuda-13.2
```

## Runtime Environment

Any process that needs CUDA 13.2 must set:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

For the nemotron-runtime build, update CMakeLists.txt to point at `/usr/local/cuda-13.2/` or rely on the updated `/usr/local/cuda` symlink.

## Verification

```bash
/usr/local/cuda-13.2/bin/nvcc --version
# Should show: release 13.2

cat /usr/local/cuda-13.2/version.json | grep -A2 '"libcublas"'
# Should show: 13.3.0.5 (the cuBLAS version with 3x DGX Spark improvement)

nvidia-smi
# Driver stays at 580.142. CUDA Version line may still say 13.0 (that's the driver-side CUDA, not the toolkit).

LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat python3 -c "import ctypes; lib = ctypes.CDLL('libcuda.so'); print('cuda-compat loaded')"
# Should succeed without error.
```

## Rollback

The old toolkit stays at `/usr/local/cuda-13.0/` untouched. To revert:

```bash
sudo update-alternatives --set cuda /usr/local/cuda-13.0
```

To fully remove:

```bash
sudo apt remove cuda-toolkit-13-2 cuda-compat-13-2
```

## cuDNN 9.20 + cuDNN Frontend

cuDNN is required for paged attention (the only working attention backend on SM121/GB10).

### Install cuDNN dev packages

```bash
sudo apt install libcudnn9-dev-cuda-13 cudnn9-cuda-13-2
```

This installs cuDNN 9.20.0 (runtime, headers, static library).

### cuDNN Frontend (header-only C++ wrapper)

The cuDNN Frontend is a header-only C++ library that provides the graph API for paged attention, MoE grouped matmul, etc. The Ubuntu-packaged `libcudnn-frontend-dev` is version 0.9.2 (too old). Use the current version from GitHub.

This is a build prerequisite. If setting up a fresh checkout, clone it before building:

```bash
git clone --depth 1 https://github.com/NVIDIA/cudnn-frontend.git nemotron-runtime/third_party/cudnn-frontend
```

Current pinned version: `6943af9` (March 2026 HEAD). The clone lives in the source tree at `third_party/cudnn-frontend/` and should be included when sharing the project.

CMake should add the include path:

```cmake
target_include_directories(... PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cudnn-frontend/include)
```

### Verification

```bash
ls /usr/include/aarch64-linux-gnu/cudnn.h           # backend C header (aarch64-specific path)
ls /usr/include/aarch64-linux-gnu/cudnn_version.h   # version defines (CUDNN_MAJOR=9)
ls /usr/lib/aarch64-linux-gnu/libcudnn.so.9.20.0    # runtime library
ls third_party/cudnn-frontend/include/cudnn_frontend.h  # C++ frontend graph API
```

Note: On aarch64/DGX Spark, cuDNN headers install to `/usr/include/aarch64-linux-gnu/`, not `/usr/include/` directly. CMake's `find_package(CUDNN)` or `find_path(CUDNN_INCLUDE_DIR cudnn.h)` may need the arch-specific path hint.

## After Install: Rebuild and Re-Probe

1. Rebuild nemotron-runtime against the new toolkit.
2. Re-run the NVFP4 cuBLASLt probe to check if CUBLAS_STATUS_NOT_SUPPORTED is resolved.
3. Re-run the dense GEMM benchmark suite to capture CUDA 13.2 performance on GB10.
