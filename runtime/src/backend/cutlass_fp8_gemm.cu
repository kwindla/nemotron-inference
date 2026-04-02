// CUTLASS FP8 dense GEMM for SM120/SM121 decode tiles.
// Follows the repository's NVFP4 CUTLASS runner structure, but for a single
// dense GEMM with a simple linear-combination epilogue.

#include "nemotron/cutlass_fp8_gemm.h"

#include <cuda_runtime.h>
#include <iostream>

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

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_args,
      epilogue_args,
      cutlass::KernelHardwareInfo{},
      {}
  };

  Gemm gemm_op;
  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "cutlass_fp8_gemm: can_implement: " << cutlassGetStatusString(status) << "\n";
    return false;
  }

  std::size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size > 0 && cudaMalloc(&workspace, workspace_size) != cudaSuccess) {
    std::cerr << "cutlass_fp8_gemm: workspace alloc failed\n";
    return false;
  }

  status = gemm_op.initialize(arguments, workspace, stream);
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }

  if (workspace != nullptr) {
    cudaFree(workspace);
  }

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
