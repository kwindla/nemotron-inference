#include "nemotron/fused_moe_prefill.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::size_t> DefaultRows() {
  return {
      1,  2,  7,   8,   9,   16,  23,  24,  31,  32,  64,  112, 120,
      127, 128, 192, 200, 248, 256, 320, 384, 448, 512, 992, 1024,
  };
}

bool ParseRows(const std::string& value, std::vector<std::size_t>* rows) {
  if (rows == nullptr) {
    return false;
  }
  rows->clear();
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0') {
      return false;
    }
    rows->push_back(static_cast<std::size_t>(parsed));
  }
  return !rows->empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::size_t> rows = DefaultRows();
  for (int arg_index = 1; arg_index < argc; ++arg_index) {
    const std::string arg = argv[arg_index];
    if (arg == "--rows" && arg_index + 1 < argc) {
      if (!ParseRows(argv[++arg_index], &rows)) {
        std::cerr << "routed_profile_matrix_probe: failed to parse --rows\n";
        return 1;
      }
      continue;
    }
    if (arg == "--help") {
      std::cout << "usage: routed_profile_matrix_probe [--rows 8,16,23,...]\n";
      return 0;
    }
    std::cerr << "routed_profile_matrix_probe: unknown argument: " << arg << "\n";
    return 1;
  }

  std::cout << "| dispatch_rows | fc1_profile | fc1_class | fc2_profile | fc2_class |\n";
  std::cout << "| ---: | --- | --- | --- | --- |\n";
  for (std::size_t row_count : rows) {
    std::cout << "| " << row_count
              << " | " << nemotron::SelectRoutedGemm1ProfileNameForTesting(row_count)
              << " | " << nemotron::ClassifyRoutedGemm1ProfileForTesting(row_count)
              << " | " << nemotron::SelectRoutedGemm2ProfileNameForTesting(row_count)
              << " | " << nemotron::ClassifyRoutedGemm2ProfileForTesting(row_count)
              << " |\n";
  }
  return 0;
}
