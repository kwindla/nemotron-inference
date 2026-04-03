// CUTLASS FP8 dense GEMM for SM120/SM121 decode tiles.
// Follows the repository's NVFP4 CUTLASS runner structure, but for a single
// dense GEMM with a simple linear-combination epilogue.

#include "nemotron/cutlass_fp8_gemm.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define NEMOTRON_CUTLASS_FP8_GEMM_SM121_AVAILABLE 1
#else
#define NEMOTRON_CUTLASS_FP8_GEMM_SM121_AVAILABLE 0
#endif

#if NEMOTRON_CUTLASS_FP8_GEMM_SM121_AVAILABLE

namespace nemotron {
namespace {

using ProblemShape = Shape<int, int, int, int>;
using ElementAB = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = void;
using ElementD = float;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ThreadBlockShape = Shape<_16, _64, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using EpilogueTile = Shape<_16, _32>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
    EpilogueTile,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementD,
    LayoutD,
    AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementAB,
    LayoutA,
    AlignmentA,
    ElementAB,
    LayoutB,
    AlignmentB,
    ElementAccumulator,
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using MainloopArguments = typename GemmKernel::MainloopArguments;
using EpilogueArguments = typename GemmKernel::EpilogueArguments;
using Arguments = typename Gemm::Arguments;

struct CutlassFp8GemmShape {
  int m = 0;
  int n = 0;
  int k = 0;

  bool operator<(const CutlassFp8GemmShape& other) const {
    if (m != other.m) {
      return m < other.m;
    }
    if (n != other.n) {
      return n < other.n;
    }
    return k < other.k;
  }
};

struct CutlassFp8GemmCallSignature {
  const void* fp8_a = nullptr;
  const void* fp8_b = nullptr;
  float* output = nullptr;
  std::uint32_t alpha_bits = 0;

  bool Matches(const CutlassFp8GemmCallSignature& other) const {
    return fp8_a == other.fp8_a &&
           fp8_b == other.fp8_b &&
           output == other.output &&
           alpha_bits == other.alpha_bits;
  }
};

struct CutlassFp8GemmState {
  Gemm gemm_op;
  std::size_t required_workspace_size = 0;
  bool initialized = false;
  std::optional<Arguments> cached_arguments;
  std::optional<CutlassFp8GemmCallSignature> cached_signature;
};

struct CutlassFp8GemmCache {
  int device = -1;
  void* workspace = nullptr;
  std::size_t workspace_size = 0;
  std::map<CutlassFp8GemmShape, CutlassFp8GemmState> states;

  ~CutlassFp8GemmCache() { Reset(); }

  void Reset() {
    states.clear();
    if (workspace != nullptr) {
      cudaFree(workspace);
      workspace = nullptr;
    }
    workspace_size = 0;
  }

  bool EnsureDevice() {
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      return false;
    }
    if (device != current_device) {
      Reset();
      device = current_device;
    }
    return true;
  }

  bool EnsureWorkspace(std::size_t required_workspace_size) {
    if (required_workspace_size <= workspace_size) {
      return true;
    }

    void* new_workspace = nullptr;
    if (required_workspace_size > 0 &&
        cudaMalloc(&new_workspace, required_workspace_size) != cudaSuccess) {
      return false;
    }

    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    workspace = new_workspace;
    workspace_size = required_workspace_size;

    // Params produced by initialize()/update() embed the workspace pointer.
    // Rebuild per-shape state after any workspace growth.
    states.clear();
    return true;
  }
};

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float bit width mismatch");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

CutlassFp8GemmCallSignature MakeCallSignature(
    const void* fp8_a,
    const void* fp8_b,
    float alpha,
    float* output) {
  CutlassFp8GemmCallSignature signature;
  signature.fp8_a = fp8_a;
  signature.fp8_b = fp8_b;
  signature.output = output;
  signature.alpha_bits = FloatBits(alpha);
  return signature;
}

Arguments BuildArguments(
    int m,
    int n,
    int k,
    const void* fp8_a,
    const void* fp8_b,
    float alpha,
    float* output) {
  auto a_ptr = static_cast<ElementAB const*>(fp8_a);
  auto b_ptr = static_cast<ElementAB const*>(fp8_b);

  auto problem_shape = ProblemShape{m, n, k, 1};
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  MainloopArguments mainloop_args{a_ptr, stride_a, b_ptr, stride_b};

  EpilogueArguments epilogue_args{};
  epilogue_args.thread.alpha = alpha;
  epilogue_args.thread.beta = 0.0f;
  epilogue_args.thread.alpha_ptr = nullptr;
  epilogue_args.thread.beta_ptr = nullptr;
  epilogue_args.ptr_C = output;
  epilogue_args.dC = stride_c;
  epilogue_args.ptr_D = output;
  epilogue_args.dD = stride_d;

  return Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      cutlass::KernelHardwareInfo{},
      {}
  };
}

CutlassFp8GemmCache& GetCutlassFp8GemmCache() {
  thread_local CutlassFp8GemmCache cache;
  return cache;
}

}  // namespace

bool CutlassFp8DenseGemmAvailable() { return true; }

bool RunCutlassFp8DenseGemm(
    int m,
    int n,
    int k,
    const void* fp8_a,
    const void* fp8_b,
    float alpha,
    float* output,
    cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0 || fp8_a == nullptr || fp8_b == nullptr || output == nullptr) {
    return false;
  }
  if (m > 16) {
    return false;
  }

  auto& cache = GetCutlassFp8GemmCache();
  if (!cache.EnsureDevice()) {
    std::cerr << "cutlass_fp8_gemm: cudaGetDevice failed\n";
    return false;
  }

  const CutlassFp8GemmShape shape{m, n, k};
  const auto signature = MakeCallSignature(fp8_a, fp8_b, alpha, output);
  const auto arguments = BuildArguments(m, n, k, fp8_a, fp8_b, alpha, output);

  auto state_it = cache.states.find(shape);
  if (state_it == cache.states.end()) {
    auto status = Gemm::can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "cutlass_fp8_gemm: can_implement: " << cutlassGetStatusString(status) << "\n";
      return false;
    }

    const std::size_t workspace_size = Gemm::get_workspace_size(arguments);
    if (!cache.EnsureWorkspace(workspace_size)) {
      std::cerr << "cutlass_fp8_gemm: workspace alloc failed\n";
      return false;
    }

    state_it = cache.states.emplace(shape, CutlassFp8GemmState{}).first;
    auto& state = state_it->second;
    auto init_status = state.gemm_op.initialize(arguments, cache.workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      std::cerr << "cutlass_fp8_gemm: initialize: " << cutlassGetStatusString(init_status) << "\n";
      return false;
    }
    state.required_workspace_size = workspace_size;
    state.initialized = true;
    state.cached_arguments = arguments;
    state.cached_signature = signature;
  }

  auto& state = state_it->second;
  cutlass::Status status = cutlass::Status::kSuccess;
  if (!state.cached_signature.has_value() || !state.cached_signature->Matches(signature)) {
    status = state.gemm_op.update(arguments, cache.workspace);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "cutlass_fp8_gemm: update: " << cutlassGetStatusString(status) << "\n";
      return false;
    }
    state.cached_arguments = arguments;
    state.cached_signature = signature;
  }

  status = state.gemm_op.run(stream);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "cutlass_fp8_gemm: GEMM: " << cutlassGetStatusString(status) << "\n";
    return false;
  }

  return true;
}

}  // namespace nemotron

#else

namespace nemotron {

bool CutlassFp8DenseGemmAvailable() { return false; }

bool RunCutlassFp8DenseGemm(
    int,
    int,
    int,
    const void*,
    const void*,
    float,
    float*,
    cudaStream_t) {
  return false;
}

}  // namespace nemotron

#endif
