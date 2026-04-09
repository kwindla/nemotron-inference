#pragma once

#include <cstddef>
#include <string_view>

namespace nemotron {

constexpr std::size_t kRoutedExpertExecutionAlignment = 128;
constexpr std::size_t kNvfp4ScaleBlockWidthRuntime = 16;

constexpr std::size_t RoundUpToMultiple(std::size_t value, std::size_t multiple) {
  return multiple == 0 ? value : ((value + multiple - 1u) / multiple) * multiple;
}

constexpr std::size_t DefaultRoutedExpertIntermediateSizePadded(std::size_t logical_size) {
  return RoundUpToMultiple(logical_size, kRoutedExpertExecutionAlignment);
}

constexpr std::size_t ResolveRoutedExpertIntermediateSizeExecution(
    std::size_t logical_size,
    std::size_t padded_size) {
  return padded_size == 0 ? logical_size : padded_size;
}

constexpr bool RoutedExpertIntermediateSizePaddingValid(
    std::size_t logical_size,
    std::size_t padded_size) {
  const std::size_t execution_size =
      ResolveRoutedExpertIntermediateSizeExecution(logical_size, padded_size);
  return logical_size != 0 &&
         execution_size >= logical_size &&
         (execution_size % kNvfp4ScaleBlockWidthRuntime) == 0;
}

inline bool IsRoutedExpertUpOpClass(std::string_view op_class) {
  return op_class == "routed_expert_up";
}

inline bool IsRoutedExpertDownOpClass(std::string_view op_class) {
  return op_class == "routed_expert_down";
}

inline bool IsRoutedExpertOpClass(std::string_view op_class) {
  return IsRoutedExpertUpOpClass(op_class) || IsRoutedExpertDownOpClass(op_class);
}

}  // namespace nemotron
