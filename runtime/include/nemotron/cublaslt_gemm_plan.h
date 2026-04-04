#pragma once

#include <cstddef>
#include <optional>

#include "nemotron/gemm_execution.h"

namespace nemotron {

enum class CublasLtTransform {
  kNone,
  kTranspose,
};

enum class CublasLtMatrixOrder {
  kRowMajor,
  kColumnMajor,
};

enum class CublasLtContract {
  kRowMajorA_N_RowMajorB_T,
  kColumnMajorA_T_ColumnMajorB_N,
};

enum class CublasLtScaleMode {
  kNone,
  kVec16UE4M3,
};

const char* ToString(CublasLtTransform transform);
const char* ToString(CublasLtMatrixOrder order);
const char* ToString(CublasLtContract contract);
const char* ToString(CublasLtScaleMode scale_mode);

struct CublasLtGemmPlan {
  PreparedGemmExecution execution;
  CublasLtContract contract = CublasLtContract::kRowMajorA_N_RowMajorB_T;
  CublasLtTransform transform_a = CublasLtTransform::kNone;
  CublasLtTransform transform_b = CublasLtTransform::kTranspose;
  CublasLtMatrixOrder order_a = CublasLtMatrixOrder::kRowMajor;
  CublasLtMatrixOrder order_b = CublasLtMatrixOrder::kRowMajor;
  CublasLtMatrixOrder order_c = CublasLtMatrixOrder::kRowMajor;
  CublasLtScaleMode scale_mode = CublasLtScaleMode::kNone;
  std::size_t lda = 0;
  std::size_t ldb = 0;
  std::size_t ldc = 0;
  std::size_t required_alignment_bytes = 16;
  bool packed_alignment_ok = false;
  bool block_scales_alignment_ok = false;
  bool tensor_scale_alignment_ok = false;
};

struct CublasLtPlanRejectInfo {
  const char* reason = nullptr;
  bool packed_alignment_ok = false;
  bool block_scales_alignment_ok = false;
  bool tensor_scale_alignment_ok = false;
};

std::optional<CublasLtGemmPlan> BuildCublasLtGemmPlan(
    const PreparedGemmExecution& execution,
    CublasLtPlanRejectInfo* reject_info = nullptr);

}  // namespace nemotron
