#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

int main() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cerr
        << "manifest_backed_test_gate: FAIL (NEMOTRON_FORWARD_MANIFEST is unset; "
           "manifest-backed tests would skip instead of executing)\n";
    return 1;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cerr << "manifest_backed_test_gate: FAIL (NEMOTRON_FORWARD_MANIFEST path does not exist: "
              << manifest_path << ")\n";
    return 1;
  }

  std::cout << "manifest_backed_test_gate: PASS (" << manifest_path << ")\n";
  return 0;
}
