#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <cutlass/arch/barrier.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/layout/layout.h>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/tensor_impl.hpp>

namespace {

#ifndef NEMOTRON_INCLUDE_STAGE_KERNEL
#define NEMOTRON_INCLUDE_STAGE_KERNEL 0
#endif

namespace nvfp4_cute {

using ElementAB = cute::float_e2m1_t;
using ElementSFCompute = cute::float_ue4m3_t;
static constexpr int kScaleVecSize = 16;

using MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    ElementAB,
    ElementAB,
    float,
    ElementSFCompute,
    kScaleVecSize>;

}  // namespace nvfp4_cute

using ARegister = std::remove_extent_t<typename nvfp4_cute::MmaOp::ARegisters>;
using BRegister = std::remove_extent_t<typename nvfp4_cute::MmaOp::BRegisters>;
using Atom = cute::MMA_Atom<nvfp4_cute::MmaOp>;
using TracedP5LayoutA = cutlass::layout::RowMajor;
using TracedP5LayoutB = cutlass::layout::ColumnMajor;
using TracedP5LayoutC = cutlass::layout::ColumnMajor;
using TracedP5LayoutD = cutlass::layout::ColumnMajor;
using TracedP5MmaTileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using TracedP5ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using TracedP5EpilogueTensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using TracedP5TensorOp = cutlass::arch::OpClassBlockScaledTensorOp;
using TracedP5ElementD = __nv_bfloat16;
using TracedP5Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5EpilogueTensorOp,
    TracedP5MmaTileShape,
    TracedP5ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    TracedP5ElementD,
    TracedP5LayoutC*,
    32,
    TracedP5ElementD,
    TracedP5LayoutD*,
    32,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using TracedP5StageCountAutoCarveout =
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename TracedP5Epilogue::SharedStorage))>;
using TracedP5CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    TracedP5TensorOp,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutA*,
    32,
    cutlass::nv_float4_t<nvfp4_cute::ElementAB>,
    TracedP5LayoutB*,
    32,
    float,
    TracedP5MmaTileShape,
    TracedP5ClusterShape,
    TracedP5StageCountAutoCarveout,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using TracedP5TiledMma = typename TracedP5CollectiveMainloop::TiledMma;
using TracedP5SmemLayoutA = typename TracedP5CollectiveMainloop::SmemLayoutA;
using TracedP5SmemLayoutB = typename TracedP5CollectiveMainloop::SmemLayoutB;
using TracedP5SmemLayoutSFA = typename TracedP5CollectiveMainloop::SmemLayoutSFA;
using TracedP5SmemLayoutSFB = typename TracedP5CollectiveMainloop::SmemLayoutSFB;
using TracedP5SmemCopyAtomA = typename TracedP5CollectiveMainloop::SmemCopyAtomA;
using TracedP5SmemCopyAtomB = typename TracedP5CollectiveMainloop::SmemCopyAtomB;
using TracedP5SmemCopyAtomSFA = typename TracedP5CollectiveMainloop::SmemCopyAtomSFA;
using TracedP5SmemCopyAtomSFB = typename TracedP5CollectiveMainloop::SmemCopyAtomSFB;

struct P5Traits {
  using CollectiveMainloop = TracedP5CollectiveMainloop;
  using TiledMma = TracedP5TiledMma;
  using SmemLayoutA = TracedP5SmemLayoutA;
  using SmemLayoutB = TracedP5SmemLayoutB;
  using SmemLayoutSFA = TracedP5SmemLayoutSFA;
  using SmemLayoutSFB = TracedP5SmemLayoutSFB;
  using SmemCopyAtomA = TracedP5SmemCopyAtomA;
  using SmemCopyAtomB = TracedP5SmemCopyAtomB;
  using SmemCopyAtomSFA = TracedP5SmemCopyAtomSFA;
  using SmemCopyAtomSFB = TracedP5SmemCopyAtomSFB;
  using SmemAllocA = typename TiledMma::ValTypeA;
  using SmemAllocB = typename TiledMma::ValTypeB;

  static constexpr int kScaleSmemCosizeA = cute::cosize_v<SmemLayoutSFA>;
  static constexpr int kScaleSmemCosizeB = cute::cosize_v<SmemLayoutSFB>;
  static constexpr int kSwizzledAElems = cute::size(cute::take<0, 2>(SmemLayoutA{}));
  static constexpr int kSwizzledBElems = cute::size(cute::take<0, 2>(SmemLayoutB{}));
};

constexpr int kP5ThreadsPerBlock = (cute::size(P5Traits::TiledMma{}) / 32) * 32;
constexpr int kP5PipelineStages = 2;
constexpr int kP5SfbTmaStageElems = P5Traits::kScaleSmemCosizeB;
constexpr int kFp32StageElems = 128 * 128;

template <class T>
__device__ __forceinline__ void KeepPointer(T* ptr) {
  asm volatile("" : : "l"(ptr) : "memory");
}

template <class Traits>
struct CurrentMainloopStorage {
  alignas(1024) cute::array_aligned<typename Traits::SmemAllocA, Traits::kSwizzledAElems * kP5PipelineStages>
      smem_swizzled_a_storage;
  alignas(1024) cute::array_aligned<typename Traits::SmemAllocB, Traits::kSwizzledBElems * kP5PipelineStages>
      smem_swizzled_b_storage;
  alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeA * kP5PipelineStages>
      a_scale_smem_storage;
  alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeB>
      b_scale_smem_storage;
  alignas(128) cute::array_aligned<nvfp4_cute::ElementSFCompute, kP5SfbTmaStageElems>
      p5_sfb_tma_smem_storage;
  alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType
      p5_tma_full_mbar_storage[kP5PipelineStages];
  alignas(16) cutlass::arch::ClusterBarrier::ValueType
      p5_tma_empty_mbar_storage[kP5PipelineStages];
};

template <class Traits>
struct AliasedSharedStorage {
  union alignas(1024) Storage {
    CurrentMainloopStorage<Traits> mainloop;
    cute::array_aligned<float, kFp32StageElems> fp32_staging_storage;
  } storage;
  int p13_debug_capture_cta;
};

template <class Traits>
__global__ void CurrentSeparateSharedKernel(std::uintptr_t* sink) {
  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocA, Traits::kSwizzledAElems * kP5PipelineStages>
      smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocB, Traits::kSwizzledBElems * kP5PipelineStages>
      smem_swizzled_b_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeA * kP5PipelineStages>
      a_scale_smem_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeB>
      b_scale_smem_storage;
  __shared__ alignas(128) cute::array_aligned<nvfp4_cute::ElementSFCompute, kP5SfbTmaStageElems>
      p5_sfb_tma_smem_storage;
  __shared__ alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType
      p5_tma_full_mbar_storage[kP5PipelineStages];
  __shared__ alignas(16) cutlass::arch::ClusterBarrier::ValueType
      p5_tma_empty_mbar_storage[kP5PipelineStages];
  __shared__ int p13_debug_capture_cta;

  KeepPointer(smem_swizzled_a_storage.data());
  KeepPointer(smem_swizzled_b_storage.data());
  KeepPointer(a_scale_smem_storage.data());
  KeepPointer(b_scale_smem_storage.data());
  KeepPointer(p5_sfb_tma_smem_storage.data());
  KeepPointer(&p5_tma_full_mbar_storage[0]);
  KeepPointer(&p5_tma_empty_mbar_storage[0]);
  KeepPointer(&p13_debug_capture_cta);
  if (threadIdx.x == 0 && sink != nullptr) {
    sink[0] = reinterpret_cast<std::uintptr_t>(smem_swizzled_a_storage.data()) ^
              reinterpret_cast<std::uintptr_t>(smem_swizzled_b_storage.data()) ^
              reinterpret_cast<std::uintptr_t>(a_scale_smem_storage.data()) ^
              reinterpret_cast<std::uintptr_t>(b_scale_smem_storage.data()) ^
              reinterpret_cast<std::uintptr_t>(p5_sfb_tma_smem_storage.data());
  }
}

#if NEMOTRON_INCLUDE_STAGE_KERNEL
template <class Traits>
__global__ void CurrentSeparatePlusStagingKernel(std::uintptr_t* sink) {
  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocA, Traits::kSwizzledAElems * kP5PipelineStages>
      smem_swizzled_a_storage;
  __shared__ alignas(1024) cute::array_aligned<typename Traits::SmemAllocB, Traits::kSwizzledBElems * kP5PipelineStages>
      smem_swizzled_b_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeA * kP5PipelineStages>
      a_scale_smem_storage;
  __shared__ alignas(1024) cute::array_aligned<nvfp4_cute::ElementSFCompute, Traits::kScaleSmemCosizeB>
      b_scale_smem_storage;
  __shared__ alignas(128) cute::array_aligned<nvfp4_cute::ElementSFCompute, kP5SfbTmaStageElems>
      p5_sfb_tma_smem_storage;
  __shared__ alignas(16) cutlass::arch::ClusterTransactionBarrier::ValueType
      p5_tma_full_mbar_storage[kP5PipelineStages];
  __shared__ alignas(16) cutlass::arch::ClusterBarrier::ValueType
      p5_tma_empty_mbar_storage[kP5PipelineStages];
  __shared__ alignas(1024) cute::array_aligned<float, kFp32StageElems> fp32_staging_storage;
  __shared__ int p13_debug_capture_cta;

  KeepPointer(smem_swizzled_a_storage.data());
  KeepPointer(smem_swizzled_b_storage.data());
  KeepPointer(a_scale_smem_storage.data());
  KeepPointer(b_scale_smem_storage.data());
  KeepPointer(p5_sfb_tma_smem_storage.data());
  KeepPointer(&p5_tma_full_mbar_storage[0]);
  KeepPointer(&p5_tma_empty_mbar_storage[0]);
  KeepPointer(fp32_staging_storage.data());
  KeepPointer(&p13_debug_capture_cta);
  if (threadIdx.x == 0 && sink != nullptr) {
    sink[0] = reinterpret_cast<std::uintptr_t>(smem_swizzled_a_storage.data()) ^
              reinterpret_cast<std::uintptr_t>(fp32_staging_storage.data());
  }
}
#endif

template <class Traits>
__global__ void AliasedSharedKernel(std::uintptr_t* sink) {
  __shared__ AliasedSharedStorage<Traits> shared;

  KeepPointer(shared.storage.mainloop.smem_swizzled_a_storage.data());
  KeepPointer(shared.storage.mainloop.smem_swizzled_b_storage.data());
  KeepPointer(shared.storage.mainloop.a_scale_smem_storage.data());
  KeepPointer(shared.storage.mainloop.b_scale_smem_storage.data());
  KeepPointer(shared.storage.mainloop.p5_sfb_tma_smem_storage.data());
  KeepPointer(&shared.storage.mainloop.p5_tma_full_mbar_storage[0]);
  KeepPointer(&shared.storage.mainloop.p5_tma_empty_mbar_storage[0]);
  KeepPointer(shared.storage.fp32_staging_storage.data());
  KeepPointer(&shared.p13_debug_capture_cta);
  if (threadIdx.x == 0 && sink != nullptr) {
    sink[0] = reinterpret_cast<std::uintptr_t>(shared.storage.mainloop.smem_swizzled_a_storage.data()) ^
              reinterpret_cast<std::uintptr_t>(shared.storage.fp32_staging_storage.data());
  }
}

template <class Kernel>
void ReportKernel(const char* name, Kernel kernel) {
  cudaFuncAttributes attrs{};
  cudaError_t status = cudaFuncGetAttributes(&attrs, kernel);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: cudaFuncGetAttributes failed: %s\n", name, cudaGetErrorString(status));
    return;
  }

  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm,
      kernel,
      kP5ThreadsPerBlock,
      0);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: occupancy query failed: %s\n", name, cudaGetErrorString(status));
    return;
  }

  std::printf(
      "%-28s shared=%6zu B  local=%6zu B  regs=%3d  maxThreads=%4d  occupancy_blocks_per_sm=%d\n",
      name,
      attrs.sharedSizeBytes,
      attrs.localSizeBytes,
      attrs.numRegs,
      attrs.maxThreadsPerBlock,
      active_blocks_per_sm);
}

void PrintStorageSummary() {
  using Mainloop = CurrentMainloopStorage<P5Traits>;
  using Aliased = AliasedSharedStorage<P5Traits>;
  using AStorage =
      cute::array_aligned<typename P5Traits::SmemAllocA, P5Traits::kSwizzledAElems * kP5PipelineStages>;
  using BStorage =
      cute::array_aligned<typename P5Traits::SmemAllocB, P5Traits::kSwizzledBElems * kP5PipelineStages>;
  using ScaleAStorage =
      cute::array_aligned<nvfp4_cute::ElementSFCompute, P5Traits::kScaleSmemCosizeA * kP5PipelineStages>;
  using ScaleBStorage =
      cute::array_aligned<nvfp4_cute::ElementSFCompute, P5Traits::kScaleSmemCosizeB>;
  using TmaScaleStorage =
      cute::array_aligned<nvfp4_cute::ElementSFCompute, kP5SfbTmaStageElems>;
  using StageStorage = cute::array_aligned<float, kFp32StageElems>;

  std::printf("P5 threads/block: %d\n", kP5ThreadsPerBlock);
  std::printf("P5 swizzled elems: A=%d B=%d\n", P5Traits::kSwizzledAElems, P5Traits::kSwizzledBElems);
  std::printf("P5 scale cosize:   A=%d B=%d\n", P5Traits::kScaleSmemCosizeA, P5Traits::kScaleSmemCosizeB);
  std::printf("Shared component sizes:\n");
  std::printf("  A mainloop storage:      %6zu B\n", sizeof(AStorage));
  std::printf("  B mainloop storage:      %6zu B\n", sizeof(BStorage));
  std::printf("  A scale storage:         %6zu B\n", sizeof(ScaleAStorage));
  std::printf("  B scale storage:         %6zu B\n", sizeof(ScaleBStorage));
  std::printf("  SFB TMA stage storage:   %6zu B\n", sizeof(TmaScaleStorage));
  std::printf("  Barrier storage:         %6zu B\n",
              sizeof(cutlass::arch::ClusterTransactionBarrier::ValueType) * kP5PipelineStages +
                  sizeof(cutlass::arch::ClusterBarrier::ValueType) * kP5PipelineStages);
  std::printf("  128x128 FP32 staging:    %6zu B\n", sizeof(StageStorage));
  std::printf("Aggregate storage models:\n");
  std::printf("  Current mainloop struct: %6zu B\n", sizeof(Mainloop));
  std::printf("  Aliased union struct:    %6zu B\n", sizeof(Aliased));
  std::printf("  Current + staging naive: %6zu B\n", sizeof(Mainloop) + sizeof(StageStorage) + sizeof(int));
}

}  // namespace

int main() {
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed\n");
    return 1;
  }

  std::printf("GPU: %s\n", props.name);
  std::printf(
      "cc=%d.%d sharedMemPerMultiprocessor=%zu sharedMemPerBlockOptin=%zu\n",
      props.major,
      props.minor,
      static_cast<std::size_t>(props.sharedMemPerMultiprocessor),
      static_cast<std::size_t>(props.sharedMemPerBlockOptin));

  PrintStorageSummary();

  std::uintptr_t* sink = nullptr;
  if (cudaMalloc(&sink, sizeof(std::uintptr_t)) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed\n");
    return 1;
  }

  ReportKernel("current_separate", CurrentSeparateSharedKernel<P5Traits>);
  ReportKernel("aliased_union", AliasedSharedKernel<P5Traits>);
#if NEMOTRON_INCLUDE_STAGE_KERNEL
  ReportKernel("current_plus_stage", CurrentSeparatePlusStagingKernel<P5Traits>);
#endif

  CurrentSeparateSharedKernel<P5Traits><<<1, kP5ThreadsPerBlock>>>(sink);
  AliasedSharedKernel<P5Traits><<<1, kP5ThreadsPerBlock>>>(sink);
#if NEMOTRON_INCLUDE_STAGE_KERNEL
  CurrentSeparatePlusStagingKernel<P5Traits><<<1, kP5ThreadsPerBlock>>>(sink);
#endif
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "kernel execution failed\n");
    cudaFree(sink);
    return 1;
  }

  cudaFree(sink);
  return 0;
}
