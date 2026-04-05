// SM120 FP4×FP4 Grouped GEMM smoke test
// Verifies CUTLASS GroupProblemShape with block-scaled FP4 on RTX 5090
// using the TRT-LLM-aligned recipe.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cute/tensor.hpp>

#pragma GCC diagnostic pop

using namespace cute;

// ── Type definitions matching TRT-LLM SM120 grouped FP4 recipe ──────────

using ArchTag = cutlass::arch::Sm120;
using TensorOp = cutlass::arch::OpClassBlockScaledTensorOp;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = void;
using ElementD = float;
using ElementAccum = float;
using ElementCompute = float;
using ElementSF = cutlass::float_ue4m3_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignA = 32;
constexpr int AlignB = 32;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int64_t, int64_t, int64_t>>;
using MmaTileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// SM120 epilogue: TmaWarpSpecialized (TRT-LLM EpilogueScheduleSM120)
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAccum, ElementC, ElementAccum>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, TensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementCompute,
    ElementC, LayoutD*, AlignD,
    ElementD, LayoutD*, AlignD,
    EpilogueSchedule,
    FusionOp>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, TensorOp,
    ElementA, LayoutA*, AlignA,
    ElementB, LayoutB*, AlignB,
    ElementAccum,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Internal (non-pointer) types for building host arrays
using InternalStrideA = typename GemmKernel::InternalStrideA;
using InternalStrideB = typename GemmKernel::InternalStrideB;
using InternalStrideD = typename GemmKernel::InternalStrideD;
using InternalLayoutSFA = typename GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using InternalLayoutSFB = typename GemmKernel::CollectiveMainloop::InternalLayoutSFB;

// ── Helpers ─────────────────────────────────────────────────────────────

bool CheckCuda(cudaError_t s, const char* msg = "") {
  if (s != cudaSuccess) {
    std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(s) << "\n";
    return false;
  }
  return true;
}

template <typename T>
T* Alloc(std::size_t count) {
  T* p = nullptr;
  if (!CheckCuda(cudaMalloc(&p, count * sizeof(T)), "alloc")) return nullptr;
  return p;
}

template <typename T>
bool Upload(T* dst, const T* src, std::size_t count) {
  return CheckCuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

// ── Test ────────────────────────────────────────────────────────────────

bool TestGroupedGemm() {
  constexpr int num_groups = 2;
  constexpr int M0 = 128, M1 = 256, N = 128, K = 128;

  const std::size_t A0_bytes = M0 * (K / 2);
  const std::size_t A1_bytes = M1 * (K / 2);
  const std::size_t B_bytes = N * (K / 2);

  // Scale layouts (int shapes to match InternalLayout types)
  auto layout_SFA0 = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M0, N, K, 1));
  auto layout_SFB0 = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M0, N, K, 1));
  auto layout_SFA1 = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M1, N, K, 1));
  auto layout_SFB1 = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M1, N, K, 1));

  // Device memory
  auto* d_A0 = Alloc<uint8_t>(A0_bytes);
  auto* d_A1 = Alloc<uint8_t>(A1_bytes);
  auto* d_B  = Alloc<uint8_t>(B_bytes);
  auto* d_D0 = Alloc<float>(M0 * N);
  auto* d_D1 = Alloc<float>(M1 * N);
  auto* d_SFA0 = Alloc<uint8_t>(cute::cosize(layout_SFA0));
  auto* d_SFA1 = Alloc<uint8_t>(cute::cosize(layout_SFA1));
  auto* d_SFB0 = Alloc<uint8_t>(cute::cosize(layout_SFB0));
  auto* d_SFB1 = Alloc<uint8_t>(cute::cosize(layout_SFB1));
  auto* d_alpha0 = Alloc<float>(1);
  auto* d_alpha1 = Alloc<float>(1);

  if (!d_A0 || !d_A1 || !d_B || !d_D0 || !d_D1 ||
      !d_SFA0 || !d_SFA1 || !d_SFB0 || !d_SFB1 || !d_alpha0 || !d_alpha1)
    return false;

  // Fill random + identity scales
  std::mt19937 rng(42);
  auto fill = [&](uint8_t* p, std::size_t n) {
    std::vector<uint8_t> h(n);
    for (auto& b : h) b = rng() & 0xFF;
    return Upload(p, h.data(), n);
  };
  float one = 1.0f;
  if (!fill(d_A0, A0_bytes) || !fill(d_A1, A1_bytes) || !fill(d_B, B_bytes) ||
      !CheckCuda(cudaMemset(d_SFA0, 0x7E, cute::cosize(layout_SFA0))) ||
      !CheckCuda(cudaMemset(d_SFA1, 0x7E, cute::cosize(layout_SFA1))) ||
      !CheckCuda(cudaMemset(d_SFB0, 0x7E, cute::cosize(layout_SFB0))) ||
      !CheckCuda(cudaMemset(d_SFB1, 0x7E, cute::cosize(layout_SFB1))) ||
      !CheckCuda(cudaMemset(d_D0, 0, M0 * N * sizeof(float))) ||
      !CheckCuda(cudaMemset(d_D1, 0, M1 * N * sizeof(float))) ||
      !Upload(d_alpha0, &one, 1) || !Upload(d_alpha1, &one, 1))
    return false;

  using ElemFP4 = cutlass::float_e2m1_t;

  // Host arrays
  typename ProblemShape::UnderlyingProblemShape h_shapes[2] = {
      cute::make_shape(int64_t(M0), int64_t(N), int64_t(K)),
      cute::make_shape(int64_t(M1), int64_t(N), int64_t(K))};

  const ElemFP4* h_ptr_A[2] = {(const ElemFP4*)d_A0, (const ElemFP4*)d_A1};
  const ElemFP4* h_ptr_B[2] = {(const ElemFP4*)d_B, (const ElemFP4*)d_B};
  float* h_ptr_D[2] = {d_D0, d_D1};
  const ElementSF* h_ptr_SFA[2] = {(const ElementSF*)d_SFA0, (const ElementSF*)d_SFA1};
  const ElementSF* h_ptr_SFB[2] = {(const ElementSF*)d_SFB0, (const ElementSF*)d_SFB1};
  const float* h_alpha_ptrs[2] = {d_alpha0, d_alpha1};

  // Grouped strides: arrays of InternalStride
  InternalStrideA h_dA[2] = {
      cute::make_int_tuple_from<InternalStrideA>(int64_t(K), int64_t(0)),
      cute::make_int_tuple_from<InternalStrideA>(int64_t(K), int64_t(0))};
  InternalStrideB h_dB[2] = {
      cute::make_int_tuple_from<InternalStrideB>(int64_t(K), int64_t(0)),
      cute::make_int_tuple_from<InternalStrideB>(int64_t(K), int64_t(0))};
  InternalStrideD h_dD[2] = {
      cute::make_int_tuple_from<InternalStrideD>(int64_t(N), int64_t(0)),
      cute::make_int_tuple_from<InternalStrideD>(int64_t(N), int64_t(0))};

  InternalLayoutSFA h_lSFA[2] = {layout_SFA0, layout_SFA1};
  InternalLayoutSFB h_lSFB[2] = {layout_SFB0, layout_SFB1};

  // Upload arrays to device
  auto* d_shapes = Alloc<typename ProblemShape::UnderlyingProblemShape>(2);
  auto* d_ptr_A = Alloc<const ElemFP4*>(2);
  auto* d_ptr_B = Alloc<const ElemFP4*>(2);
  auto* d_ptr_D = Alloc<float*>(2);
  auto* d_ptr_SFA = Alloc<const ElementSF*>(2);
  auto* d_ptr_SFB = Alloc<const ElementSF*>(2);
  auto* d_dA = Alloc<InternalStrideA>(2);
  auto* d_dB = Alloc<InternalStrideB>(2);
  auto* d_dD = Alloc<InternalStrideD>(2);
  auto* d_lSFA = Alloc<InternalLayoutSFA>(2);
  auto* d_lSFB = Alloc<InternalLayoutSFB>(2);
  auto* d_alpha_ptrs = Alloc<const float*>(2);

  if (!d_shapes || !d_ptr_A || !d_ptr_B || !d_ptr_D || !d_ptr_SFA || !d_ptr_SFB ||
      !d_dA || !d_dB || !d_dD || !d_lSFA || !d_lSFB || !d_alpha_ptrs)
    return false;

  if (!Upload(d_shapes, h_shapes, 2) ||
      !Upload(d_ptr_A, h_ptr_A, 2) || !Upload(d_ptr_B, h_ptr_B, 2) ||
      !Upload(d_ptr_D, h_ptr_D, 2) ||
      !Upload(d_ptr_SFA, h_ptr_SFA, 2) || !Upload(d_ptr_SFB, h_ptr_SFB, 2) ||
      !Upload(d_dA, h_dA, 2) || !Upload(d_dB, h_dB, 2) || !Upload(d_dD, h_dD, 2) ||
      !Upload(d_lSFA, h_lSFA, 2) || !Upload(d_lSFB, h_lSFB, 2) ||
      !Upload(d_alpha_ptrs, h_alpha_ptrs, 2))
    return false;

  // CUTLASS grouped arguments — all mainloop/epilogue fields are device pointers to arrays
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape = ProblemShape{num_groups, d_shapes, nullptr};

  args.mainloop.ptr_A = d_ptr_A;
  args.mainloop.dA = d_dA;
  args.mainloop.ptr_B = d_ptr_B;
  args.mainloop.dB = d_dB;
  args.mainloop.ptr_SFA = d_ptr_SFA;
  args.mainloop.layout_SFA = d_lSFA;
  args.mainloop.ptr_SFB = d_ptr_SFB;
  args.mainloop.layout_SFB = d_lSFB;

  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = d_dD;
  args.epilogue.ptr_D = d_ptr_D;
  args.epilogue.dD = d_dD;
  args.epilogue.thread.alpha_ptr_array = d_alpha_ptrs;
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;

  args.hw_info.device_id = 0;
  cudaDeviceGetAttribute(&args.hw_info.sm_count, cudaDevAttrMultiProcessorCount, 0);
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

  Gemm gemm;

  auto can_impl = gemm.can_implement(args);
  if (can_impl != cutlass::Status::kSuccess) {
    std::cerr << "can_implement: " << cutlass::cutlassGetStatusString(can_impl) << "\n";
    return false;
  }

  std::size_t ws_size = gemm.get_workspace_size(args);
  std::cout << "Grouped workspace: " << ws_size << " bytes\n";

  void* d_ws = nullptr;
  if (ws_size > 0 && !CheckCuda(cudaMalloc(&d_ws, ws_size), "workspace"))
    return false;

  auto init = gemm.initialize(args, d_ws, nullptr);
  if (init != cutlass::Status::kSuccess) {
    std::cerr << "initialize: " << cutlass::cutlassGetStatusString(init) << "\n";
    return false;
  }

  auto run = gemm.run(args, d_ws, nullptr);
  if (run != cutlass::Status::kSuccess) {
    std::cerr << "run: " << cutlass::cutlassGetStatusString(run) << "\n";
    return false;
  }

  if (!CheckCuda(cudaDeviceSynchronize(), "sync")) return false;

  std::vector<float> out0(M0 * N), out1(M1 * N);
  CheckCuda(cudaMemcpy(out0.data(), d_D0, out0.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CheckCuda(cudaMemcpy(out1.data(), d_D1, out1.size() * sizeof(float), cudaMemcpyDeviceToHost));

  float sum0 = 0, sum1 = 0;
  for (float v : out0) sum0 += std::abs(v);
  for (float v : out1) sum1 += std::abs(v);
  std::cout << "Group 0 (M=" << M0 << "): L1=" << sum0 << "\n";
  std::cout << "Group 1 (M=" << M1 << "): L1=" << sum1 << "\n";

  // Cleanup
  cudaFree(d_A0); cudaFree(d_A1); cudaFree(d_B);
  cudaFree(d_D0); cudaFree(d_D1);
  cudaFree(d_SFA0); cudaFree(d_SFA1); cudaFree(d_SFB0); cudaFree(d_SFB1);
  cudaFree(d_alpha0); cudaFree(d_alpha1);
  cudaFree(d_shapes); cudaFree(d_ptr_A); cudaFree(d_ptr_B); cudaFree(d_ptr_D);
  cudaFree(d_ptr_SFA); cudaFree(d_ptr_SFB);
  cudaFree(d_dA); cudaFree(d_dB); cudaFree(d_dD);
  cudaFree(d_lSFA); cudaFree(d_lSFB); cudaFree(d_alpha_ptrs);
  if (d_ws) cudaFree(d_ws);

  return sum0 > 1e-6f && sum1 > 1e-6f;
}

int main() {
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  std::cout << "Device: " << props.name << " (SM " << props.major << props.minor << ")\n";
  if (props.major * 10 + props.minor < 120) {
    std::cout << "SKIP (requires SM120+)\n";
    return 0;
  }
  if (TestGroupedGemm()) {
    std::cout << "cutlass_fp4_grouped_smoke_test: PASS\n";
    return 0;
  } else {
    std::cout << "cutlass_fp4_grouped_smoke_test: FAIL\n";
    return 1;
  }
}
