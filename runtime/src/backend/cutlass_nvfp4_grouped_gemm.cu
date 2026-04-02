// CUTLASS NVFP4 grouped GEMM for SM121 (DGX Spark / GB10).
// Follows CUTLASS example 79d structure with FP32 output.

#include "nemotron/cutlass_nvfp4_grouped_gemm.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define NEMOTRON_FUSED_MOE_SM121_AVAILABLE 1
#else
#define NEMOTRON_FUSED_MOE_SM121_AVAILABLE 0
#endif

#if NEMOTRON_FUSED_MOE_SM121_AVAILABLE

namespace nemotron {

// ---- CUTLASS type configuration ----

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e2m1_t;

// A/B: NVFP4 block-scaled. The `*` after LayoutTag enables grouped/pointer-array mode.
using ElementA = cutlass::nv_float4_t<ElementInput>;
using LayoutATag = cutlass::layout::RowMajor;
using ElementB = cutlass::nv_float4_t<ElementInput>;
using LayoutBTag = cutlass::layout::ColumnMajor;

// Output: FP32 (matches runtime hidden-state format)
using ElementC = float;
using ElementD = float;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutDTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Resolved types from the template.
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;
using InternalElementA = typename Gemm::ElementA;
using InternalElementB = typename Gemm::ElementB;
using ElementSF = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Simple RAII helper.
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  ~DevBuf() { if (ptr) cudaFree(ptr); }
  bool alloc(std::size_t n) { return cudaMalloc(&ptr, n * sizeof(T)) == cudaSuccess; }
  bool upload(const T* h, std::size_t n) {
    return cudaMemcpy(ptr, h, n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
};

template <typename T>
__global__ void BuildContiguousScalarPointerArrayKernel(
    const T* values,
    const T** output,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = values + index;
  }
}

template <typename T>
bool BuildContiguousScalarPointerArray(
    const T* values,
    const T** output,
    std::size_t count,
    cudaStream_t stream) {
  if (values == nullptr || output == nullptr || count == 0) {
    return false;
  }
  constexpr int kThreadsPerBlock = 256;
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  BuildContiguousScalarPointerArrayKernel<<<grid, block, 0, stream>>>(values, output, count);
  return cudaGetLastError() == cudaSuccess;
}

bool CutlassNvfp4GroupedGemmAvailable() { return true; }

bool RunCutlassNvfp4GroupedGemm(
    int group_count, int m, int n, int k,
    const void* const* host_a_ptrs_raw,
    const void* const* host_a_sf_ptrs_raw,
    const void* const* host_b_ptrs_raw,
    const void* const* host_b_sf_ptrs_raw,
    const void* const* host_c_ptrs_raw,
    void* const* host_d_ptrs_raw,
    const float* device_alpha_values,
    float beta,
    cudaStream_t stream) {

  if (group_count <= 0 || m <= 0 || n <= 0 || k <= 0 || device_alpha_values == nullptr) {
    return false;
  }

  // Cast void pointer arrays to the CUTLASS-typed pointer arrays.
  auto host_a_ptrs = reinterpret_cast<const InternalElementA* const*>(host_a_ptrs_raw);
  auto host_sfa_ptrs = reinterpret_cast<const ElementSF* const*>(host_a_sf_ptrs_raw);
  auto host_b_ptrs = reinterpret_cast<const InternalElementB* const*>(host_b_ptrs_raw);
  auto host_sfb_ptrs = reinterpret_cast<const ElementSF* const*>(host_b_sf_ptrs_raw);
  auto host_c_ptrs = reinterpret_cast<const ElementC* const*>(host_c_ptrs_raw);
  auto host_d_ptrs = reinterpret_cast<ElementD* const*>(host_d_ptrs_raw);
  const auto gc = static_cast<std::size_t>(group_count);

  // Host problem sizes + strides (uniform for MoE).
  std::vector<typename ProblemShape::UnderlyingProblemShape> h_problems(gc);
  std::vector<StrideA> h_sA(gc);
  std::vector<StrideB> h_sB(gc);
  std::vector<StrideC> h_sC(gc);
  std::vector<StrideD> h_sD(gc);
  std::vector<LayoutSFA> h_lSFA(gc);
  std::vector<LayoutSFB> h_lSFB(gc);
  for (std::size_t i = 0; i < gc; ++i) {
    h_problems[i] = {m, n, k};
    h_sA[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    h_sB[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    h_sC[i] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    h_sD[i] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    // SF layouts derived from the problem shape via the CUTLASS config.
    h_lSFA[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    h_lSFB[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  }

  // Upload everything to device.
  DevBuf<typename ProblemShape::UnderlyingProblemShape> d_problems;
  DevBuf<const InternalElementA*> d_a; DevBuf<const InternalElementB*> d_b;
  DevBuf<const ElementSF*> d_sfa; DevBuf<const ElementSF*> d_sfb;
  DevBuf<const ElementC*> d_c; DevBuf<ElementD*> d_d;
  DevBuf<const ElementAccumulator*> d_alpha_ptrs;
  DevBuf<StrideA> d_sA; DevBuf<StrideB> d_sB;
  DevBuf<StrideC> d_sC; DevBuf<StrideD> d_sD;
  DevBuf<LayoutSFA> d_lSFA; DevBuf<LayoutSFB> d_lSFB;

  if (!d_problems.alloc(gc) || !d_problems.upload(h_problems.data(), gc) ||
      !d_a.alloc(gc) || !d_a.upload(host_a_ptrs, gc) ||
      !d_b.alloc(gc) || !d_b.upload(host_b_ptrs, gc) ||
      !d_sfa.alloc(gc) || !d_sfa.upload(host_sfa_ptrs, gc) ||
      !d_sfb.alloc(gc) || !d_sfb.upload(host_sfb_ptrs, gc) ||
      !d_c.alloc(gc) || !d_c.upload(host_c_ptrs, gc) ||
      !d_d.alloc(gc) || !d_d.upload(host_d_ptrs, gc) ||
      !d_alpha_ptrs.alloc(gc) ||
      !d_sA.alloc(gc) || !d_sA.upload(h_sA.data(), gc) ||
      !d_sB.alloc(gc) || !d_sB.upload(h_sB.data(), gc) ||
      !d_sC.alloc(gc) || !d_sC.upload(h_sC.data(), gc) ||
      !d_sD.alloc(gc) || !d_sD.upload(h_sD.data(), gc) ||
      !d_lSFA.alloc(gc) || !d_lSFA.upload(h_lSFA.data(), gc) ||
      !d_lSFB.alloc(gc) || !d_lSFB.upload(h_lSFB.data(), gc)) {
    std::cerr << "fused_moe: device allocation failed\n";
    return false;
  }

  if (!BuildContiguousScalarPointerArray(
          device_alpha_values, d_alpha_ptrs.ptr, gc, stream)) {
    return false;
  }

  // Build arguments matching the CUTLASS collective layout:
  // mainloop: {ptr_A, dA, ptr_B, dB, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB}
  // epilogue: {{alpha_ptr_array, beta}, ptr_C, dC, ptr_D, dD}
  using EpilogueArgs = typename GemmKernel::CollectiveEpilogue::Arguments;
  EpilogueArgs epilogue_args{};
  epilogue_args.thread.alpha = 0.0f;
  epilogue_args.thread.beta = beta;
  epilogue_args.thread.alpha_ptr = nullptr;
  epilogue_args.thread.beta_ptr = nullptr;
  epilogue_args.thread.alpha_ptr_array = d_alpha_ptrs.ptr;
  epilogue_args.thread.beta_ptr_array = nullptr;
  epilogue_args.ptr_C = d_c.ptr;
  epilogue_args.dC = d_sC.ptr;
  epilogue_args.ptr_D = d_d.ptr;
  epilogue_args.dD = d_sD.ptr;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, d_problems.ptr, h_problems.data()},
      {d_a.ptr, d_sA.ptr, d_b.ptr, d_sB.ptr,
       d_sfa.ptr, d_lSFA.ptr, d_sfb.ptr, d_lSFB.ptr},
      epilogue_args
  };

  Gemm gemm_op;
  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "fused_moe: can_implement: " << cutlassGetStatusString(status) << "\n";
    return false;
  }

  std::size_t ws_size = Gemm::get_workspace_size(arguments);
  void* ws = nullptr;
  if (ws_size > 0 && cudaMalloc(&ws, ws_size) != cudaSuccess) {
    std::cerr << "fused_moe: workspace alloc failed\n";
    return false;
  }

  status = gemm_op.initialize(arguments, ws, stream);
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }
  if (ws) cudaFree(ws);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "fused_moe: GEMM: " << cutlassGetStatusString(status) << "\n";
    return false;
  }
  return true;
}

// Device-pointer variant: pointer arrays are already on device.
// Strides and SF layouts are still uploaded per call (uniform, small).
// TODO: cache these per-shape to eliminate the last per-call allocations.
bool RunCutlassNvfp4GroupedGemmFromDevice(
    int group_count, int m, int n, int k,
    const void* device_a_ptrs_raw,
    const void* device_a_sf_ptrs_raw,
    const void* device_b_ptrs_raw,
    const void* device_b_sf_ptrs_raw,
    const void* device_c_ptrs_raw,
    void* device_d_ptrs_raw,
    const float* device_alpha_values,
    float beta,
    cudaStream_t stream) {

  if (group_count <= 0 || m <= 0 || n <= 0 || k <= 0 || device_alpha_values == nullptr) {
    return false;
  }
  const auto gc = static_cast<std::size_t>(group_count);

  // The CUTLASS collective's ElementA is float_e2m1_t (sizeof=1).
  // Our packed NVFP4 data pointers are uint8_t* — same byte layout.
  // The reinterpret_cast is safe because both are 1-byte element pointer arrays.
  using CollectiveElementA = typename GemmKernel::CollectiveMainloop::ElementA;
  using CollectiveElementB = typename GemmKernel::CollectiveMainloop::ElementB;
  static_assert(sizeof(CollectiveElementA) == 1, "ElementA must be 1 byte");
  static_assert(sizeof(CollectiveElementB) == 1, "ElementB must be 1 byte");
  static_assert(sizeof(ElementSF) == 1, "ElementSF must be 1 byte");

  // Use memcpy to cast the pointer values into the exact types CUTLASS expects,
  // avoiding C++ strict aliasing issues with reinterpret_cast on pointer-to-pointer.
  using MainloopArgs = typename GemmKernel::CollectiveMainloop::Arguments;
  using EpilogueArgs = typename GemmKernel::CollectiveEpilogue::Arguments;

  // Build host-side metadata (uniform shapes — small, could be cached).
  std::vector<typename ProblemShape::UnderlyingProblemShape> h_problems(gc);
  std::vector<StrideA> h_sA(gc);
  std::vector<StrideB> h_sB(gc);
  std::vector<StrideC> h_sC(gc);
  std::vector<StrideD> h_sD(gc);
  std::vector<LayoutSFA> h_lSFA(gc);
  std::vector<LayoutSFB> h_lSFB(gc);
  for (std::size_t i = 0; i < gc; ++i) {
    h_problems[i] = {m, n, k};
    h_sA[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    h_sB[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    h_sC[i] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    h_sD[i] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    h_lSFA[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    h_lSFB[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  }

  // Upload strides and layouts (small, uniform metadata).
  DevBuf<typename ProblemShape::UnderlyingProblemShape> d_problems;
  DevBuf<const ElementAccumulator*> d_alpha_ptrs;
  DevBuf<StrideA> d_sA; DevBuf<StrideB> d_sB;
  DevBuf<StrideC> d_sC; DevBuf<StrideD> d_sD;
  DevBuf<LayoutSFA> d_lSFA; DevBuf<LayoutSFB> d_lSFB;

  if (!d_problems.alloc(gc) || !d_problems.upload(h_problems.data(), gc) ||
      !d_alpha_ptrs.alloc(gc) ||
      !d_sA.alloc(gc) || !d_sA.upload(h_sA.data(), gc) ||
      !d_sB.alloc(gc) || !d_sB.upload(h_sB.data(), gc) ||
      !d_sC.alloc(gc) || !d_sC.upload(h_sC.data(), gc) ||
      !d_sD.alloc(gc) || !d_sD.upload(h_sD.data(), gc) ||
      !d_lSFA.alloc(gc) || !d_lSFA.upload(h_lSFA.data(), gc) ||
      !d_lSFB.alloc(gc) || !d_lSFB.upload(h_lSFB.data(), gc)) {
    return false;
  }

  if (!BuildContiguousScalarPointerArray(
          device_alpha_values, d_alpha_ptrs.ptr, gc, stream)) {
    return false;
  }

  // Construct mainloop arguments using field assignment to avoid
  // aggregate-init type mismatch with CUTLASS's internal pointer types.
  MainloopArgs mainloop_args;
  std::memcpy(&mainloop_args.ptr_A, &device_a_ptrs_raw, sizeof(void*));
  mainloop_args.dA = d_sA.ptr;
  std::memcpy(&mainloop_args.ptr_B, &device_b_ptrs_raw, sizeof(void*));
  mainloop_args.dB = d_sB.ptr;
  std::memcpy(&mainloop_args.ptr_SFA, &device_a_sf_ptrs_raw, sizeof(void*));
  mainloop_args.layout_SFA = d_lSFA.ptr;
  std::memcpy(&mainloop_args.ptr_SFB, &device_b_sf_ptrs_raw, sizeof(void*));
  mainloop_args.layout_SFB = d_lSFB.ptr;

  EpilogueArgs epilogue_args{};
  epilogue_args.thread.alpha = 0.0f;
  epilogue_args.thread.beta = beta;
  epilogue_args.thread.alpha_ptr = nullptr;
  epilogue_args.thread.beta_ptr = nullptr;
  epilogue_args.thread.alpha_ptr_array = d_alpha_ptrs.ptr;
  epilogue_args.thread.beta_ptr_array = nullptr;
  std::memcpy(&epilogue_args.ptr_C, &device_c_ptrs_raw, sizeof(void*));
  epilogue_args.dC = d_sC.ptr;
  std::memcpy(&epilogue_args.ptr_D, &device_d_ptrs_raw, sizeof(void*));
  epilogue_args.dD = d_sD.ptr;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, d_problems.ptr, h_problems.data()},
      mainloop_args,
      epilogue_args
  };

  Gemm gemm_op;
  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return false;

  std::size_t ws_size = Gemm::get_workspace_size(arguments);
  void* ws = nullptr;
  if (ws_size > 0 && cudaMalloc(&ws, ws_size) != cudaSuccess) return false;

  status = gemm_op.initialize(arguments, ws, stream);
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }
  if (ws) cudaFree(ws);
  return status == cutlass::Status::kSuccess;
}

// ---- Pre-allocated plan ----

struct CutlassNvfp4GroupedGemmPlan::Impl {
  int group_count = 0;
  int m = 0;
  int n = 0;
  int k = 0;

  // Device-resident constant metadata (allocated once, reused every token).
  std::vector<typename ProblemShape::UnderlyingProblemShape> h_problems;
  DevBuf<typename ProblemShape::UnderlyingProblemShape> d_problems;
  DevBuf<StrideA> d_sA;
  DevBuf<StrideB> d_sB;
  DevBuf<StrideC> d_sC;
  DevBuf<StrideD> d_sD;
  DevBuf<LayoutSFA> d_lSFA;
  DevBuf<LayoutSFB> d_lSFB;
  DevBuf<const ElementAccumulator*> d_alpha_ptrs;

  // Pre-allocated workspace.
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;

  ~Impl() {
    if (workspace) cudaFree(workspace);
  }
};

std::unique_ptr<CutlassNvfp4GroupedGemmPlan> CutlassNvfp4GroupedGemmPlan::Create(
    int group_count, int m, int n, int k) {
  if (group_count <= 0 || m <= 0 || n <= 0 || k <= 0) return nullptr;
  const auto gc = static_cast<std::size_t>(group_count);

  auto impl = std::make_unique<Impl>();
  impl->group_count = group_count;
  impl->m = m;
  impl->n = n;
  impl->k = k;

  // Build host metadata.
  impl->h_problems.resize(gc);
  std::vector<StrideA> h_sA(gc);
  std::vector<StrideB> h_sB(gc);
  std::vector<StrideC> h_sC(gc);
  std::vector<StrideD> h_sD(gc);
  std::vector<LayoutSFA> h_lSFA(gc);
  std::vector<LayoutSFB> h_lSFB(gc);
  for (std::size_t i = 0; i < gc; ++i) {
    impl->h_problems[i] = {m, n, k};
    h_sA[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    h_sB[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    h_sC[i] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    h_sD[i] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    h_lSFA[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    h_lSFB[i] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  }

  // Upload to device (once).
  if (!impl->d_problems.alloc(gc) || !impl->d_problems.upload(impl->h_problems.data(), gc) ||
      !impl->d_sA.alloc(gc) || !impl->d_sA.upload(h_sA.data(), gc) ||
      !impl->d_sB.alloc(gc) || !impl->d_sB.upload(h_sB.data(), gc) ||
      !impl->d_sC.alloc(gc) || !impl->d_sC.upload(h_sC.data(), gc) ||
      !impl->d_sD.alloc(gc) || !impl->d_sD.upload(h_sD.data(), gc) ||
      !impl->d_lSFA.alloc(gc) || !impl->d_lSFA.upload(h_lSFA.data(), gc) ||
      !impl->d_lSFB.alloc(gc) || !impl->d_lSFB.upload(h_lSFB.data(), gc) ||
      !impl->d_alpha_ptrs.alloc(gc)) {
    return nullptr;
  }

  // Pre-allocate workspace by probing with dummy pointer arrays.
  // CUTLASS workspace size depends only on the shapes, not on the actual data.
  DevBuf<const void*> dummy_ptrs;
  if (!dummy_ptrs.alloc(gc)) return nullptr;
  cudaMemset(dummy_ptrs.ptr, 0, gc * sizeof(void*));

  using MainloopArgs = typename GemmKernel::CollectiveMainloop::Arguments;
  using EpilogueArgs = typename GemmKernel::CollectiveEpilogue::Arguments;

  MainloopArgs mainloop_args;
  std::memcpy(&mainloop_args.ptr_A, &dummy_ptrs.ptr, sizeof(void*));
  mainloop_args.dA = impl->d_sA.ptr;
  std::memcpy(&mainloop_args.ptr_B, &dummy_ptrs.ptr, sizeof(void*));
  mainloop_args.dB = impl->d_sB.ptr;
  std::memcpy(&mainloop_args.ptr_SFA, &dummy_ptrs.ptr, sizeof(void*));
  mainloop_args.layout_SFA = impl->d_lSFA.ptr;
  std::memcpy(&mainloop_args.ptr_SFB, &dummy_ptrs.ptr, sizeof(void*));
  mainloop_args.layout_SFB = impl->d_lSFB.ptr;

  EpilogueArgs epilogue_args{};
  epilogue_args.thread.alpha = 1.0f;
  epilogue_args.thread.beta = 0.0f;
  epilogue_args.thread.alpha_ptr = nullptr;
  epilogue_args.thread.beta_ptr = nullptr;
  epilogue_args.thread.alpha_ptr_array = nullptr;
  epilogue_args.thread.beta_ptr_array = nullptr;
  std::memcpy(&epilogue_args.ptr_C, &dummy_ptrs.ptr, sizeof(void*));
  epilogue_args.dC = impl->d_sC.ptr;
  std::memcpy(&epilogue_args.ptr_D, &dummy_ptrs.ptr, sizeof(void*));
  epilogue_args.dD = impl->d_sD.ptr;

  typename Gemm::Arguments probe_args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, impl->d_problems.ptr, impl->h_problems.data()},
      mainloop_args,
      epilogue_args
  };

  Gemm gemm_op;
  if (gemm_op.can_implement(probe_args) != cutlass::Status::kSuccess) {
    return nullptr;
  }

  impl->workspace_bytes = Gemm::get_workspace_size(probe_args);
  if (impl->workspace_bytes > 0) {
    if (cudaMalloc(&impl->workspace, impl->workspace_bytes) != cudaSuccess) {
      return nullptr;
    }
  }

  return std::unique_ptr<CutlassNvfp4GroupedGemmPlan>(
      new CutlassNvfp4GroupedGemmPlan(std::move(impl)));
}

CutlassNvfp4GroupedGemmPlan::CutlassNvfp4GroupedGemmPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CutlassNvfp4GroupedGemmPlan::CutlassNvfp4GroupedGemmPlan(CutlassNvfp4GroupedGemmPlan&&) noexcept = default;
CutlassNvfp4GroupedGemmPlan& CutlassNvfp4GroupedGemmPlan::operator=(CutlassNvfp4GroupedGemmPlan&&) noexcept = default;
CutlassNvfp4GroupedGemmPlan::~CutlassNvfp4GroupedGemmPlan() = default;

bool CutlassNvfp4GroupedGemmPlan::valid() const { return impl_ != nullptr; }
int CutlassNvfp4GroupedGemmPlan::group_count() const { return impl_ ? impl_->group_count : 0; }
int CutlassNvfp4GroupedGemmPlan::m() const { return impl_ ? impl_->m : 0; }
int CutlassNvfp4GroupedGemmPlan::n() const { return impl_ ? impl_->n : 0; }
int CutlassNvfp4GroupedGemmPlan::k() const { return impl_ ? impl_->k : 0; }

bool CutlassNvfp4GroupedGemmPlan::Run(
    const void* device_a_ptrs,
    const void* device_a_sf_ptrs,
    const void* device_b_ptrs,
    const void* device_b_sf_ptrs,
    const void* device_c_ptrs,
    void* device_d_ptrs,
    const float* device_alpha_values,
    float beta,
    cudaStream_t stream) const {
  if (!valid() || device_alpha_values == nullptr) {
    return false;
  }

  if (!BuildContiguousScalarPointerArray(
          device_alpha_values,
          impl_->d_alpha_ptrs.ptr,
          static_cast<std::size_t>(impl_->group_count),
          stream)) {
    return false;
  }

  using MainloopArgs = typename GemmKernel::CollectiveMainloop::Arguments;
  using EpilogueArgs = typename GemmKernel::CollectiveEpilogue::Arguments;

  MainloopArgs mainloop_args;
  std::memcpy(&mainloop_args.ptr_A, &device_a_ptrs, sizeof(void*));
  mainloop_args.dA = impl_->d_sA.ptr;
  std::memcpy(&mainloop_args.ptr_B, &device_b_ptrs, sizeof(void*));
  mainloop_args.dB = impl_->d_sB.ptr;
  std::memcpy(&mainloop_args.ptr_SFA, &device_a_sf_ptrs, sizeof(void*));
  mainloop_args.layout_SFA = impl_->d_lSFA.ptr;
  std::memcpy(&mainloop_args.ptr_SFB, &device_b_sf_ptrs, sizeof(void*));
  mainloop_args.layout_SFB = impl_->d_lSFB.ptr;

  EpilogueArgs epilogue_args{};
  epilogue_args.thread.alpha = 0.0f;
  epilogue_args.thread.beta = beta;
  epilogue_args.thread.alpha_ptr = nullptr;
  epilogue_args.thread.beta_ptr = nullptr;
  epilogue_args.thread.alpha_ptr_array = impl_->d_alpha_ptrs.ptr;
  epilogue_args.thread.beta_ptr_array = nullptr;
  std::memcpy(&epilogue_args.ptr_C, &device_c_ptrs, sizeof(void*));
  epilogue_args.dC = impl_->d_sC.ptr;
  std::memcpy(&epilogue_args.ptr_D, &device_d_ptrs, sizeof(void*));
  epilogue_args.dD = impl_->d_sD.ptr;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {impl_->group_count, impl_->d_problems.ptr, impl_->h_problems.data()},
      mainloop_args,
      epilogue_args
  };

  Gemm gemm_op;
  auto status = gemm_op.initialize(arguments, impl_->workspace, stream);
  if (status == cutlass::Status::kSuccess) {
    status = gemm_op.run(stream);
  }
  return status == cutlass::Status::kSuccess;
}

}  // namespace nemotron

#else

namespace nemotron {
bool CutlassNvfp4GroupedGemmAvailable() { return false; }

bool RunCutlassNvfp4GroupedGemm(
    int, int, int, int,
    const void* const*, const void* const*,
    const void* const*, const void* const*,
    const void* const*, void* const*,
    const float*, float, cudaStream_t) {
  return false;
}

bool RunCutlassNvfp4GroupedGemmFromDevice(
    int, int, int, int,
    const void*, const void*,
    const void*, const void*,
    const void*, void*,
    const float*, float, cudaStream_t) {
  return false;
}

std::unique_ptr<CutlassNvfp4GroupedGemmPlan> CutlassNvfp4GroupedGemmPlan::Create(int, int, int, int) {
  return nullptr;
}
CutlassNvfp4GroupedGemmPlan::CutlassNvfp4GroupedGemmPlan(std::unique_ptr<Impl>) {}
CutlassNvfp4GroupedGemmPlan::CutlassNvfp4GroupedGemmPlan(CutlassNvfp4GroupedGemmPlan&&) noexcept = default;
CutlassNvfp4GroupedGemmPlan& CutlassNvfp4GroupedGemmPlan::operator=(CutlassNvfp4GroupedGemmPlan&&) noexcept = default;
CutlassNvfp4GroupedGemmPlan::~CutlassNvfp4GroupedGemmPlan() = default;
bool CutlassNvfp4GroupedGemmPlan::valid() const { return false; }
int CutlassNvfp4GroupedGemmPlan::group_count() const { return 0; }
int CutlassNvfp4GroupedGemmPlan::m() const { return 0; }
int CutlassNvfp4GroupedGemmPlan::n() const { return 0; }
int CutlassNvfp4GroupedGemmPlan::k() const { return 0; }
bool CutlassNvfp4GroupedGemmPlan::Run(
    const void*, const void*, const void*, const void*,
    const void*, void*, const float*, float, cudaStream_t) const {
  return false;
}

}  // namespace nemotron

#endif
