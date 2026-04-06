#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using nemotron::DeviceTensorFp32;
using nemotron::DeviceExpertRouting;
using nemotron::RunDeviceExpertSelection;
using nemotron::RunDeviceExpertRouting;
using nemotron::kMaxDeviceExpertRoutingExperts;

// Synthetic contract test: this file intentionally keeps small, explicit
// routing fixtures so edge cases stay easy to reason about. Deployment-shape
// MoE correctness lives in fused_moe_prefill_test with Nano-sized dimensions.

template <typename T>
class DeviceBuffer {
 public:
  static std::unique_ptr<DeviceBuffer> Create(std::size_t count) {
    if (!HasCudaDevice() || count == 0) {
      return nullptr;
    }

    T* data = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)) !=
        cudaSuccess) {
      return nullptr;
    }
    return std::unique_ptr<DeviceBuffer>(new DeviceBuffer(data, count));
  }

  DeviceBuffer(DeviceBuffer&&) noexcept = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const {
    return data_;
  }

  std::size_t count() const {
    return count_;
  }

  bool CopyFromHost(const std::vector<T>& values) const {
    return values.size() == count_ &&
           cudaMemcpy(
               data_,
               values.data(),
               count_ * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess;
  }

 private:
  DeviceBuffer(T* data, std::size_t count) : data_(data), count_(count) {}

  static bool HasCudaDevice() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

template <typename T>
bool CopyDeviceValues(
    const T* device_data,
    std::size_t count,
    std::vector<T>* host_values) {
  if (device_data == nullptr || host_values == nullptr) {
    return false;
  }
  host_values->assign(count, T{});
  return cudaMemcpy(
             host_values->data(),
             device_data,
             count * sizeof(T),
             cudaMemcpyDeviceToHost) == cudaSuccess;
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

bool ExpectEqualVector(
    const std::vector<int>& actual,
    const std::vector<int>& expected,
    const std::string& label) {
  if (actual.size() != expected.size()) {
    std::cerr << "FAIL: " << label << " size mismatch: actual="
              << actual.size() << " expected=" << expected.size() << "\n";
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "FAIL: " << label << " mismatch at " << i
                << ": actual=" << actual[i]
                << " expected=" << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

bool ExpectEqualPrefix(
    const std::vector<int>& actual,
    const std::vector<int>& expected,
    const std::string& label) {
  if (actual.size() < expected.size()) {
    std::cerr << "FAIL: " << label << " size mismatch: actual="
              << actual.size() << " expected>=" << expected.size() << "\n";
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "FAIL: " << label << " mismatch at " << i
                << ": actual=" << actual[i]
                << " expected=" << expected[i] << "\n";
      return false;
    }
  }
  return true;
}

float Sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

struct SelectionParityCase {
  std::string label;
  std::size_t token_count = 0;
  std::size_t n_routed_experts = 0;
  std::size_t top_k = 0;
  std::size_t n_group = 0;
  std::size_t topk_group = 0;
  float routed_scaling_factor = 1.0f;
  bool norm_topk_prob = false;
  std::vector<float> router_logits;
  std::vector<float> correction_bias;
};

struct SelectionCandidate {
  int expert_index = -1;
  float score = -INFINITY;
  float raw_weight = 0.0f;
};

bool SelectionCandidateIsBetter(
    const SelectionCandidate& candidate,
    const SelectionCandidate& current) {
  if (candidate.score > current.score) {
    return true;
  }
  if (candidate.score < current.score) {
    return false;
  }
  return candidate.expert_index < current.expert_index;
}

void BuildVllmGroupedTopkReference(
    const SelectionParityCase& test_case,
    std::vector<int>* topk_ids,
    std::vector<float>* topk_weights) {
  topk_ids->assign(test_case.token_count * test_case.top_k, -1);
  topk_weights->assign(test_case.token_count * test_case.top_k, 0.0f);

  const std::size_t group_size = test_case.n_routed_experts / test_case.n_group;
  for (std::size_t token_index = 0; token_index < test_case.token_count; ++token_index) {
    std::vector<float> raw_scores(test_case.n_routed_experts, 0.0f);
    std::vector<float> choice_scores(test_case.n_routed_experts, 0.0f);
    for (std::size_t expert_index = 0; expert_index < test_case.n_routed_experts; ++expert_index) {
      const float raw_score = Sigmoid(
          test_case.router_logits[token_index * test_case.n_routed_experts + expert_index]);
      raw_scores[expert_index] = raw_score;
      choice_scores[expert_index] = raw_score + test_case.correction_bias[expert_index];
    }

    std::vector<float> group_scores(test_case.n_group, -INFINITY);
    for (std::size_t group_index = 0; group_index < test_case.n_group; ++group_index) {
      std::vector<SelectionCandidate> group_candidates;
      group_candidates.reserve(group_size);
      for (std::size_t slot = 0; slot < group_size; ++slot) {
        const std::size_t expert_index = group_index * group_size + slot;
        group_candidates.push_back(SelectionCandidate{
            static_cast<int>(expert_index),
            choice_scores[expert_index],
            raw_scores[expert_index],
        });
      }
      std::stable_sort(
          group_candidates.begin(),
          group_candidates.end(),
          [](const SelectionCandidate& lhs, const SelectionCandidate& rhs) {
            return SelectionCandidateIsBetter(lhs, rhs);
          });
      const float top1 = group_candidates.empty() ? -INFINITY : group_candidates[0].score;
      const float top2 =
          group_candidates.size() > 1 ? group_candidates[1].score : top1;
      group_scores[group_index] = top1 + top2;
    }

    std::vector<std::size_t> ranked_groups(test_case.n_group, 0);
    std::iota(ranked_groups.begin(), ranked_groups.end(), 0);
    std::stable_sort(
        ranked_groups.begin(),
        ranked_groups.end(),
        [&](std::size_t lhs, std::size_t rhs) {
          if (group_scores[lhs] > group_scores[rhs]) {
            return true;
          }
          if (group_scores[lhs] < group_scores[rhs]) {
            return false;
          }
          return lhs < rhs;
        });

    std::vector<bool> selected_groups(test_case.n_group, false);
    for (std::size_t rank = 0;
         rank < std::min(test_case.topk_group, test_case.n_group);
         ++rank) {
      selected_groups[ranked_groups[rank]] = true;
    }

    std::vector<SelectionCandidate> candidates;
    candidates.reserve(test_case.n_routed_experts);
    for (std::size_t expert_index = 0; expert_index < test_case.n_routed_experts; ++expert_index) {
      const std::size_t group_index = expert_index / group_size;
      if (!selected_groups[group_index]) {
        continue;
      }
      candidates.push_back(SelectionCandidate{
          static_cast<int>(expert_index),
          choice_scores[expert_index],
          raw_scores[expert_index],
      });
    }
    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const SelectionCandidate& lhs, const SelectionCandidate& rhs) {
          return SelectionCandidateIsBetter(lhs, rhs);
        });

    float weight_sum = 0.0f;
    for (std::size_t slot = 0; slot < test_case.top_k; ++slot) {
      const SelectionCandidate& candidate = candidates[slot];
      (*topk_ids)[token_index * test_case.top_k + slot] = candidate.expert_index;
      (*topk_weights)[token_index * test_case.top_k + slot] = candidate.raw_weight;
      weight_sum += candidate.raw_weight;
    }

    if (test_case.norm_topk_prob) {
      const float denominator = weight_sum + 1.0e-20f;
      for (std::size_t slot = 0; slot < test_case.top_k; ++slot) {
        (*topk_weights)[token_index * test_case.top_k + slot] /= denominator;
      }
    }
    for (std::size_t slot = 0; slot < test_case.top_k; ++slot) {
      (*topk_weights)[token_index * test_case.top_k + slot] *=
          test_case.routed_scaling_factor;
    }
  }
}

SelectionParityCase MakeTieSensitiveNanoCase() {
  SelectionParityCase test_case;
  test_case.label = "tie_sensitive_nano";
  test_case.token_count = 2;
  test_case.n_routed_experts = 512;
  test_case.top_k = 22;
  test_case.n_group = 1;
  test_case.topk_group = 1;
  test_case.routed_scaling_factor = 5.0f;
  test_case.norm_topk_prob = true;
  test_case.router_logits.assign(
      test_case.token_count * test_case.n_routed_experts,
      -8.0f);
  test_case.correction_bias.assign(test_case.n_routed_experts, 0.0f);

  for (std::size_t token_index = 0; token_index < test_case.token_count; ++token_index) {
    for (std::size_t expert_index = 0; expert_index < test_case.n_routed_experts; ++expert_index) {
      test_case.router_logits[token_index * test_case.n_routed_experts + expert_index] =
          -8.0f + 0.001f * static_cast<float>(expert_index);
    }
  }

  for (std::size_t expert_index = 0; expert_index < 32; ++expert_index) {
    test_case.router_logits[expert_index] =
        1.10f - 0.02f * static_cast<float>(expert_index);
    test_case.router_logits[test_case.n_routed_experts + expert_index] =
        0.95f - 0.018f * static_cast<float>(expert_index);
  }

  test_case.router_logits[7] = 0.8425f;
  test_case.router_logits[8] = 0.8425f;
  test_case.router_logits[20] = 0.5875f;
  test_case.router_logits[21] = 0.5875f;
  test_case.router_logits[test_case.n_routed_experts + 5] = 0.7935f;
  test_case.router_logits[test_case.n_routed_experts + 6] = 0.7935f;
  test_case.router_logits[test_case.n_routed_experts + 18] = 0.4715f;
  test_case.router_logits[test_case.n_routed_experts + 19] = 0.4715f;

  return test_case;
}

SelectionParityCase MakeCorrectionBiasNanoCase() {
  SelectionParityCase test_case;
  test_case.label = "correction_bias_nano";
  test_case.token_count = 3;
  test_case.n_routed_experts = 512;
  test_case.top_k = 22;
  test_case.n_group = 1;
  test_case.topk_group = 1;
  test_case.routed_scaling_factor = 5.0f;
  test_case.norm_topk_prob = true;
  test_case.router_logits.assign(
      test_case.token_count * test_case.n_routed_experts,
      -8.0f);
  test_case.correction_bias.assign(test_case.n_routed_experts, 0.0f);

  for (std::size_t expert_index = 0; expert_index < test_case.n_routed_experts; ++expert_index) {
    test_case.correction_bias[expert_index] =
        (static_cast<float>(static_cast<int>((expert_index * 17) % 13) - 6) *
         0.025f);
    for (std::size_t token_index = 0; token_index < test_case.token_count; ++token_index) {
      test_case.router_logits[token_index * test_case.n_routed_experts + expert_index] =
          -8.0f + 0.001f * static_cast<float>((expert_index + token_index * 11) % 97);
    }
  }

  for (std::size_t token_index = 0; token_index < test_case.token_count; ++token_index) {
    for (std::size_t expert_index = 0; expert_index < 28; ++expert_index) {
      test_case.router_logits[token_index * test_case.n_routed_experts + expert_index] =
          0.95f - 0.028f * static_cast<float>(expert_index) +
          0.01f * static_cast<float>(token_index);
    }
  }

  test_case.router_logits[22] = 0.31f;
  test_case.correction_bias[22] = 0.42f;
  test_case.router_logits[23] = 0.30f;
  test_case.correction_bias[23] = 0.42f;
  test_case.router_logits[24] = 0.54f;
  test_case.correction_bias[24] = -0.18f;
  test_case.router_logits[test_case.n_routed_experts + 11] = 0.36f;
  test_case.correction_bias[11] = 0.33f;
  test_case.router_logits[(2 * test_case.n_routed_experts) + 9] = 0.48f;
  test_case.correction_bias[9] = -0.12f;

  return test_case;
}

SelectionParityCase MakeScalingWithoutRenormCase() {
  SelectionParityCase test_case;
  test_case.label = "scaling_without_renorm";
  test_case.token_count = 2;
  test_case.n_routed_experts = 512;
  test_case.top_k = 22;
  test_case.n_group = 1;
  test_case.topk_group = 1;
  test_case.routed_scaling_factor = 1.75f;
  test_case.norm_topk_prob = false;
  test_case.router_logits.assign(
      test_case.token_count * test_case.n_routed_experts,
      -8.0f);
  test_case.correction_bias.assign(test_case.n_routed_experts, 0.0f);

  for (std::size_t token_index = 0; token_index < test_case.token_count; ++token_index) {
    for (std::size_t expert_index = 0; expert_index < test_case.n_routed_experts; ++expert_index) {
      test_case.router_logits[token_index * test_case.n_routed_experts + expert_index] =
          -7.5f + 0.002f * static_cast<float>((expert_index * 13 + token_index * 29) % 211);
    }
    for (std::size_t expert_index = 0; expert_index < 26; ++expert_index) {
      test_case.router_logits[token_index * test_case.n_routed_experts + expert_index] =
          0.88f - 0.024f * static_cast<float>(expert_index) +
          0.006f * static_cast<float>(token_index);
    }
  }

  return test_case;
}

bool RunDeviceSelectionParityCase(const SelectionParityCase& test_case) {
  auto router_logits_device =
      DeviceTensorFp32::Create({test_case.token_count, test_case.n_routed_experts});
  auto correction_bias_device = DeviceTensorFp32::Create({test_case.n_routed_experts});
  auto topk_ids_device =
      DeviceBuffer<int>::Create(test_case.token_count * test_case.top_k);
  auto topk_weights_device =
      DeviceBuffer<float>::Create(test_case.token_count * test_case.top_k);
  if (!Expect(
          router_logits_device != nullptr &&
              correction_bias_device != nullptr &&
              topk_ids_device != nullptr &&
              topk_weights_device != nullptr,
          test_case.label + ": device buffers should allocate") ||
      !Expect(
          router_logits_device->CopyFromHost(
              test_case.router_logits.data(),
              test_case.router_logits.size()),
          test_case.label + ": router logits should upload") ||
      !Expect(
          correction_bias_device->CopyFromHost(
              test_case.correction_bias.data(),
              test_case.correction_bias.size()),
          test_case.label + ": correction bias should upload") ||
      !Expect(
          RunDeviceExpertSelection(
              *router_logits_device,
              *correction_bias_device,
              test_case.n_routed_experts,
              test_case.top_k,
              test_case.n_group,
              test_case.topk_group,
              test_case.routed_scaling_factor,
              test_case.norm_topk_prob,
              topk_ids_device->data(),
              topk_weights_device->data()),
          test_case.label + ": RunDeviceExpertSelection should succeed") ||
      !Expect(
          cudaDeviceSynchronize() == cudaSuccess,
          test_case.label + ": selection kernel should synchronize")) {
    return false;
  }

  std::vector<int> actual_topk_ids;
  std::vector<float> actual_topk_weights;
  if (!Expect(
          CopyDeviceValues(
              topk_ids_device->data(),
              test_case.token_count * test_case.top_k,
              &actual_topk_ids),
          test_case.label + ": topk ids should download") ||
      !Expect(
          CopyDeviceValues(
              topk_weights_device->data(),
              test_case.token_count * test_case.top_k,
              &actual_topk_weights),
          test_case.label + ": topk weights should download")) {
    return false;
  }

  std::vector<int> expected_topk_ids;
  std::vector<float> expected_topk_weights;
  BuildVllmGroupedTopkReference(
      test_case,
      &expected_topk_ids,
      &expected_topk_weights);

  return ExpectEqualVector(
             actual_topk_ids,
             expected_topk_ids,
             test_case.label + ": topk_ids") &&
         Expect(
             MaxAbsDiff(actual_topk_weights, expected_topk_weights) <= 1.0e-5f,
             test_case.label + ": topk_weights should match vLLM reference");
}

void FillSelectionPattern(
    std::size_t n_experts,
    std::size_t token_count,
    std::size_t top_k,
    int variant,
    std::vector<int>* selected_indices,
    std::vector<float>* selected_weights) {
  const std::size_t selection_count = token_count * top_k;
  selected_indices->assign(selection_count, 0);
  selected_weights->assign(selection_count, 0.0f);

  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    for (std::size_t slot = 0; slot < top_k; ++slot) {
      const std::size_t selection_index = token_index * top_k + slot;
      std::size_t expert_index =
          ((token_index * static_cast<std::size_t>(17 + variant)) +
           (slot * static_cast<std::size_t>(29 + (variant * 3))) +
           (static_cast<std::size_t>(variant) * 7)) %
          n_experts;
      if (((token_index + static_cast<std::size_t>(variant)) % 5) == 0) {
        expert_index = (slot * 3 + static_cast<std::size_t>(variant)) % n_experts;
      }
      if (((slot + static_cast<std::size_t>(variant)) % 4) == 0) {
        expert_index = (token_index * 7 + 1) % n_experts;
      }
      if (((token_index + slot + static_cast<std::size_t>(variant)) % 9) == 0) {
        expert_index = 0;
      }

      (*selected_indices)[selection_index] = static_cast<int>(expert_index);

      const std::size_t raw =
          ((selection_index + 1) * static_cast<std::size_t>(13 + variant * 5) +
           7) %
          113;
      (*selected_weights)[selection_index] =
          0.05f + static_cast<float>(raw) / 127.0f;
    }
  }
}

void BuildCpuReference(
    std::size_t n_experts,
    std::size_t token_count,
    std::size_t top_k,
    const std::vector<int>& selected_indices,
    const std::vector<float>& selected_weights,
    std::vector<int>* expert_offsets,
    std::vector<int>* sorted_token_indices,
    std::vector<float>* sorted_token_weights,
    std::vector<int>* selection_to_sorted,
    std::vector<int>* active_expert_ids) {
  const std::size_t selection_count = token_count * top_k;
  std::vector<int> expert_counts(n_experts, 0);
  for (int expert_index : selected_indices) {
    ++expert_counts[static_cast<std::size_t>(expert_index)];
  }

  expert_offsets->assign(n_experts + 1, 0);
  for (std::size_t expert_index = 0; expert_index < n_experts; ++expert_index) {
    (*expert_offsets)[expert_index + 1] =
        (*expert_offsets)[expert_index] + expert_counts[expert_index];
  }

  sorted_token_indices->assign(selection_count, 0);
  sorted_token_weights->assign(selection_count, 0.0f);
  selection_to_sorted->assign(selection_count, -1);
  std::vector<int> write_offsets = *expert_offsets;
  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    for (std::size_t slot = 0; slot < top_k; ++slot) {
      const std::size_t selection_index = token_index * top_k + slot;
      const int expert_index = selected_indices[selection_index];
      const int write_index = write_offsets[static_cast<std::size_t>(expert_index)]++;
      (*sorted_token_indices)[static_cast<std::size_t>(write_index)] =
          static_cast<int>(token_index);
      (*sorted_token_weights)[static_cast<std::size_t>(write_index)] =
          selected_weights[selection_index];
      (*selection_to_sorted)[selection_index] = write_index;
    }
  }

  active_expert_ids->clear();
  for (std::size_t expert_index = 0; expert_index < n_experts; ++expert_index) {
    if (expert_counts[expert_index] > 0) {
      active_expert_ids->push_back(static_cast<int>(expert_index));
    }
  }
}

bool RunParityCase(
    std::size_t n_experts,
    std::size_t token_count,
    std::size_t top_k,
    const std::vector<int>& variants) {
  const std::size_t selection_count = token_count * top_k;
  auto selected_indices_device = DeviceBuffer<int>::Create(selection_count);
  auto selected_weights_device = DeviceBuffer<float>::Create(selection_count);
  auto routing = DeviceExpertRouting::Create(n_experts, selection_count);

  if (!Expect(
          selected_indices_device != nullptr &&
              selected_weights_device != nullptr &&
              routing != nullptr &&
              routing->valid(),
          "device allocations should succeed")) {
    return false;
  }

  for (int variant : variants) {
    std::vector<int> selected_indices_host;
    std::vector<float> selected_weights_host;
    FillSelectionPattern(
        n_experts,
        token_count,
        top_k,
        variant,
        &selected_indices_host,
        &selected_weights_host);

    if (!Expect(
            selected_indices_device->CopyFromHost(selected_indices_host),
            "selected indices upload should succeed") ||
        !Expect(
            selected_weights_device->CopyFromHost(selected_weights_host),
            "selected weights upload should succeed") ||
        !Expect(
            RunDeviceExpertRouting(
                selected_indices_device->data(),
                selected_weights_device->data(),
                token_count,
                top_k,
                routing.get()),
            "RunDeviceExpertRouting should succeed") ||
        !Expect(
            cudaDeviceSynchronize() == cudaSuccess,
            "routing kernels should complete")) {
      return false;
    }

    std::vector<int> actual_expert_offsets;
    std::vector<int> actual_sorted_token_indices;
    std::vector<float> actual_sorted_token_weights;
    std::vector<int> actual_selection_to_sorted;
    std::vector<int> actual_active_expert_count;
    std::vector<int> actual_active_expert_ids;
    if (!Expect(
            CopyDeviceValues(
                routing->expert_offsets(),
                n_experts + 1,
                &actual_expert_offsets),
            "expert offsets download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->sorted_token_indices(),
                selection_count,
                &actual_sorted_token_indices),
            "sorted token indices download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->sorted_token_weights(),
                selection_count,
                &actual_sorted_token_weights),
            "sorted token weights download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->selection_to_sorted(),
                selection_count,
                &actual_selection_to_sorted),
            "selection_to_sorted download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->active_expert_count(),
                1,
                &actual_active_expert_count),
            "active expert count download should succeed") ||
        !Expect(
            CopyDeviceValues(
                routing->active_expert_ids(),
                n_experts,
                &actual_active_expert_ids),
            "active expert ids download should succeed")) {
      return false;
    }

    std::vector<int> expected_expert_offsets;
    std::vector<int> expected_sorted_token_indices;
    std::vector<float> expected_sorted_token_weights;
    std::vector<int> expected_selection_to_sorted;
    std::vector<int> expected_active_expert_ids;
    BuildCpuReference(
        n_experts,
        token_count,
        top_k,
        selected_indices_host,
        selected_weights_host,
        &expected_expert_offsets,
        &expected_sorted_token_indices,
        &expected_sorted_token_weights,
        &expected_selection_to_sorted,
        &expected_active_expert_ids);

    if (!ExpectEqualVector(
            actual_expert_offsets,
            expected_expert_offsets,
            "expert_offsets") ||
        !ExpectEqualVector(
            actual_sorted_token_indices,
            expected_sorted_token_indices,
            "sorted_token_indices") ||
        !Expect(
            MaxAbsDiff(actual_sorted_token_weights, expected_sorted_token_weights) ==
                0.0f,
            "sorted_token_weights should match exactly") ||
        !ExpectEqualVector(
            actual_selection_to_sorted,
            expected_selection_to_sorted,
            "selection_to_sorted") ||
        !Expect(
            actual_active_expert_count.size() == 1,
            "active expert count should be scalar") ||
        !Expect(
            actual_active_expert_count[0] ==
                static_cast<int>(expected_active_expert_ids.size()),
            "active expert count should match reference") ||
        !ExpectEqualPrefix(
            actual_active_expert_ids,
            expected_active_expert_ids,
            "active_expert_ids")) {
      return false;
    }
  }

  return true;
}

bool RunCreateValidationCase() {
  return Expect(
      DeviceExpertRouting::Create(
          kMaxDeviceExpertRoutingExperts + 1,
          8) == nullptr,
      "routing buffers should reject unsupported expert counts");
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::cout << "expert_routing_device_test: SKIP (no CUDA device)\n";
    return 0;
  }

  if (!RunCreateValidationCase() ||
      !RunDeviceSelectionParityCase(MakeTieSensitiveNanoCase()) ||
      !RunDeviceSelectionParityCase(MakeCorrectionBiasNanoCase()) ||
      !RunDeviceSelectionParityCase(MakeScalingWithoutRenormCase()) ||
      !RunParityCase(128, 32, 6, {0, 1}) ||
      !RunParityCase(17, 5, 3, {2})) {
    return 1;
  }

  std::cout << "expert_routing_device_test: PASS\n";
  return 0;
}
