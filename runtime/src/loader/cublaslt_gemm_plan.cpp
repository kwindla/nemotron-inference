#include "nemotron/cublaslt_gemm_plan.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace nemotron {
namespace {

bool IsAligned(const std::uint8_t* ptr, std::size_t alignment_bytes) {
  if (ptr == nullptr || alignment_bytes == 0) {
    return false;
  }
  return (reinterpret_cast<std::uintptr_t>(ptr) % alignment_bytes) == 0;
}

bool DebugEnabled() {
  return std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
}

void ResetRejectInfo(CublasLtPlanRejectInfo* reject_info) {
  if (reject_info != nullptr) {
    *reject_info = CublasLtPlanRejectInfo{};
  }
}

void LogPlanReject(
    const PreparedGemmExecution& execution,
    const char* reason,
    CublasLtPlanRejectInfo* reject_info = nullptr,
    bool packed_alignment_ok = false,
    bool block_scales_alignment_ok = false,
    bool tensor_scale_alignment_ok = false) {
  if (reject_info != nullptr) {
    reject_info->reason = reason;
    reject_info->packed_alignment_ok = packed_alignment_ok;
    reject_info->block_scales_alignment_ok = block_scales_alignment_ok;
    reject_info->tensor_scale_alignment_ok = tensor_scale_alignment_ok;
  }
  if (!DebugEnabled()) {
    return;
  }
  std::cerr << "cublaslt_gemm_plan: reject"
            << " backend=" << ToString(execution.backend_kind)
            << " M=" << execution.launch_plan.m
            << " N=" << execution.launch_plan.n
            << " K=" << execution.launch_plan.k
            << " reason=" << reason
            << " packed_alignment_ok=" << packed_alignment_ok
            << " block_scales_alignment_ok=" << block_scales_alignment_ok
            << " tensor_scale_alignment_ok=" << tensor_scale_alignment_ok
            << "\n";
}

void ApplyContract(CublasLtContract contract, CublasLtGemmPlan* plan) {
  if (plan == nullptr) {
    return;
  }

  plan->contract = contract;
  switch (contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      plan->transform_a = CublasLtTransform::kNone;
      plan->transform_b = CublasLtTransform::kTranspose;
      plan->order_a = CublasLtMatrixOrder::kRowMajor;
      plan->order_b = CublasLtMatrixOrder::kRowMajor;
      plan->order_c = CublasLtMatrixOrder::kRowMajor;
      return;
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      plan->transform_a = CublasLtTransform::kTranspose;
      plan->transform_b = CublasLtTransform::kNone;
      plan->order_a = CublasLtMatrixOrder::kColumnMajor;
      plan->order_b = CublasLtMatrixOrder::kColumnMajor;
      plan->order_c = CublasLtMatrixOrder::kColumnMajor;
      return;
  }
}

}  // namespace

const char* ToString(CublasLtTransform transform) {
  switch (transform) {
    case CublasLtTransform::kNone:
      return "none";
    case CublasLtTransform::kTranspose:
      return "transpose";
  }
  return "unknown";
}

const char* ToString(CublasLtMatrixOrder order) {
  switch (order) {
    case CublasLtMatrixOrder::kRowMajor:
      return "row_major";
    case CublasLtMatrixOrder::kColumnMajor:
      return "column_major";
  }
  return "unknown";
}

const char* ToString(CublasLtContract contract) {
  switch (contract) {
    case CublasLtContract::kRowMajorA_N_RowMajorB_T:
      return "row_nt_mk_nk";
    case CublasLtContract::kColumnMajorA_T_ColumnMajorB_N:
      return "col_tn_km_kn";
  }
  return "unknown";
}

const char* ToString(CublasLtScaleMode scale_mode) {
  switch (scale_mode) {
    case CublasLtScaleMode::kNone:
      return "none";
    case CublasLtScaleMode::kVec16UE4M3:
      return "vec16_ue4m3";
  }
  return "unknown";
}

std::optional<CublasLtGemmPlan> BuildCublasLtGemmPlan(
    const PreparedGemmExecution& execution,
    CublasLtPlanRejectInfo* reject_info) {
  ResetRejectInfo(reject_info);
  if (execution.launch_plan.m == 0 ||
      execution.launch_plan.n == 0 ||
      execution.launch_plan.k == 0 ||
      !execution.launch_plan.packed_bytes.valid()) {
    LogPlanReject(execution, "invalid_launch_plan", reject_info);
    return std::nullopt;
  }

  CublasLtGemmPlan plan;
  plan.execution = execution;
  plan.lda = execution.launch_plan.k;
  plan.ldb = execution.launch_plan.k;
  plan.ldc = execution.launch_plan.n;
  plan.required_alignment_bytes = 16;
  plan.packed_alignment_ok =
      IsAligned(execution.launch_plan.packed_bytes.data, plan.required_alignment_bytes);

  switch (execution.backend_kind) {
    case GemmBackendKind::kCublasLtDense:
      ApplyContract(CublasLtContract::kRowMajorA_N_RowMajorB_T, &plan);
      plan.scale_mode = CublasLtScaleMode::kNone;
      plan.block_scales_alignment_ok = !execution.requires_block_scales;
      plan.tensor_scale_alignment_ok = !execution.requires_tensor_scale;
      break;
    case GemmBackendKind::kCublasLtNvfp4BlockScaled:
      if (!execution.launch_plan.block_scales_bytes.valid() ||
          !execution.launch_plan.tensor_scale_bytes.valid()) {
        LogPlanReject(
            execution,
            "missing_nvfp4_scale_buffers",
            reject_info,
            plan.packed_alignment_ok,
            false,
            false);
        return std::nullopt;
      }
      // The validated local GB10 cuBLASLt FP4 contract uses row-major A(m,k) with
      // row-major packed B(n,k) and transB = T.
      ApplyContract(CublasLtContract::kRowMajorA_N_RowMajorB_T, &plan);
      plan.scale_mode = CublasLtScaleMode::kVec16UE4M3;
      plan.block_scales_alignment_ok =
          IsAligned(execution.launch_plan.block_scales_bytes.data, plan.required_alignment_bytes);
      plan.tensor_scale_alignment_ok =
          IsAligned(execution.launch_plan.tensor_scale_bytes.data, plan.required_alignment_bytes);
      break;
  }

  if (!plan.packed_alignment_ok) {
    LogPlanReject(
        execution,
        "packed_pointer_alignment",
        reject_info,
        false,
        plan.block_scales_alignment_ok,
        plan.tensor_scale_alignment_ok);
    return std::nullopt;
  }
  if (execution.requires_block_scales && !plan.block_scales_alignment_ok) {
    LogPlanReject(
        execution,
        "block_scale_pointer_alignment",
        reject_info,
        plan.packed_alignment_ok,
        false,
        plan.tensor_scale_alignment_ok);
    return std::nullopt;
  }
  if (execution.requires_tensor_scale && !plan.tensor_scale_alignment_ok) {
    LogPlanReject(
        execution,
        "tensor_scale_pointer_alignment",
        reject_info,
        plan.packed_alignment_ok,
        plan.block_scales_alignment_ok,
        false);
    return std::nullopt;
  }

  return plan;
}

}  // namespace nemotron
