#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t MiB(std::size_t value) {
  return value * 1024ull * 1024ull;
}

constexpr std::size_t kPerRequestFp32Bytes = 174325760ull;
constexpr std::size_t kPerRequestFp16Bytes = 87162880ull;
constexpr std::size_t kLogicalElementsPerRequest = kPerRequestFp32Bytes / sizeof(float);
constexpr std::size_t kInt8GroupSize = 128;
constexpr std::uint64_t kDefaultStochasticRoundingSeed = 0x4E454D4F54524F4Cull;
constexpr std::uint32_t kPhiloxRoundCount = 5;

enum class CacheFormat {
  kFp32,
  kFp16,
  kFp16Sr,
  kInt8Group,
};

enum class OperationKind {
  kClone,
  kUpdate,
  kMambaUpdate,
  kFixtureUpdate,
  kFixtureChain,
  kFixtureTrace,
};

const char* ToString(CacheFormat format) {
  switch (format) {
    case CacheFormat::kFp32:
      return "fp32";
    case CacheFormat::kFp16:
      return "fp16";
    case CacheFormat::kFp16Sr:
      return "fp16_sr";
    case CacheFormat::kInt8Group:
      return "int8_group";
  }
  return "unknown";
}

const char* ToString(OperationKind kind) {
  switch (kind) {
    case OperationKind::kClone:
      return "clone";
    case OperationKind::kUpdate:
      return "update";
    case OperationKind::kMambaUpdate:
      return "mamba_update";
    case OperationKind::kFixtureUpdate:
      return "fixture_update";
    case OperationKind::kFixtureChain:
      return "fixture_chain";
    case OperationKind::kFixtureTrace:
      return "fixture_trace";
  }
  return "unknown";
}

struct BenchmarkOptions {
  std::vector<std::size_t> concurrency_values = {1, 4, 8};
  std::vector<OperationKind> operations = {
      OperationKind::kClone,
      OperationKind::kUpdate,
      OperationKind::kMambaUpdate,
      OperationKind::kFixtureUpdate};
  std::size_t warmup_iterations = 1;
  std::size_t hot_iterations = 5;
  std::uint64_t stochastic_rounding_seed = kDefaultStochasticRoundingSeed;
  std::vector<std::uint64_t> stochastic_rounding_sweep_seeds = {
      kDefaultStochasticRoundingSeed,
      kDefaultStochasticRoundingSeed ^ 0x9E3779B97F4A7C15ull,
      kDefaultStochasticRoundingSeed + 0xD1B54A32D192ED03ull,
      kDefaultStochasticRoundingSeed + 0x94D049BB133111EBull};
  std::vector<std::size_t> fixture_chain_steps = {8, 32, 128};
  std::vector<std::size_t> fixture_trace_steps = {128, 320, 544, 704};
  float dt_gate_threshold = HUGE_VALF;
  std::optional<std::filesystem::path> json_output_path;
  std::filesystem::path fixture_root = std::filesystem::path(NEMOTRON_MAMBA_ORACLE_FIXTURE_ROOT);
  std::filesystem::path trace_fixture_root = std::filesystem::path(NEMOTRON_MAMBA_TARGET_TRACE_ROOT);
};

struct EnvironmentInfo {
  std::string device_name;
  int device_index = 0;
  int runtime_version = 0;
  int driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::size_t total_global_mem_bytes = 0;
  std::uint64_t stochastic_rounding_seed = kDefaultStochasticRoundingSeed;
  std::uint32_t stochastic_rounding_philox_rounds = kPhiloxRoundCount;
  float dt_gate_threshold = HUGE_VALF;
  std::vector<std::uint64_t> stochastic_rounding_sweep_seeds;
  std::vector<std::size_t> fixture_chain_steps;
  std::vector<std::size_t> fixture_trace_steps;
  std::vector<std::size_t> fixture_trace_phase_end_steps;
};

struct BenchmarkResult {
  std::string format;
  std::string operation;
  std::string fixture_name;
  std::string trace_phase_name;
  std::string trace_phase_kind;
  std::size_t fixture_steps = 1;
  std::size_t oracle_seed_count = 1;
  std::size_t active_requests = 0;
  std::size_t logical_elements_per_request = 0;
  std::size_t per_request_state_bytes = 0;
  std::size_t per_request_scale_bytes = 0;
  std::size_t total_state_bytes = 0;
  std::size_t total_scale_bytes = 0;
  std::size_t approximate_traffic_bytes = 0;
  std::size_t warmup_iterations = 0;
  std::size_t hot_iterations = 0;
  double hot_mean_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  double effective_gbps = 0.0;
  bool has_oracle_diffs = false;
  double max_next_state_abs_diff = 0.0;
  double mean_next_state_abs_diff = 0.0;
  double rms_next_state_abs_diff = 0.0;
  double stddev_mean_next_state_abs_diff = 0.0;
  double stddev_rms_next_state_abs_diff = 0.0;
  double max_output_abs_diff = 0.0;
  double mean_output_abs_diff = 0.0;
  double rms_output_abs_diff = 0.0;
  double stddev_mean_output_abs_diff = 0.0;
  double stddev_rms_output_abs_diff = 0.0;
};

struct TracePhase {
  std::string name;
  std::string kind;
  std::size_t length = 0;
  std::size_t end_step = 0;
};

struct DeviceBuffer {
  void* data = nullptr;
  std::size_t bytes = 0;

  bool Allocate(std::size_t nbytes) {
    bytes = nbytes;
    return cudaMalloc(&data, nbytes) == cudaSuccess;
  }

  void Reset() {
    if (data != nullptr) {
      cudaFree(data);
      data = nullptr;
      bytes = 0;
    }
  }

  ~DeviceBuffer() {
    Reset();
  }
};

bool CheckCuda(cudaError_t status, const char* where) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

std::string EscapeJson(std::string_view text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

bool ParseCsvUnsigned(std::string_view text, std::vector<std::size_t>* values) {
  if (values == nullptr || text.empty()) {
    return false;
  }
  std::vector<std::size_t> parsed;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    std::string token(text.substr(begin, end - begin));
    if (token.empty()) {
      return false;
    }
    try {
      parsed.push_back(std::stoull(token));
    } catch (...) {
      return false;
    }
    begin = end + 1;
  }
  if (parsed.empty()) {
    return false;
  }
  *values = std::move(parsed);
  return true;
}

bool ParseCsvUint64(std::string_view text, std::vector<std::uint64_t>* values) {
  if (values == nullptr || text.empty()) {
    return false;
  }
  std::vector<std::uint64_t> parsed;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    std::string token(text.substr(begin, end - begin));
    if (token.empty()) {
      return false;
    }
    try {
      parsed.push_back(std::stoull(token));
    } catch (...) {
      return false;
    }
    begin = end + 1;
  }
  if (parsed.empty()) {
    return false;
  }
  *values = std::move(parsed);
  return true;
}

bool ParseOperations(std::string_view text, std::vector<OperationKind>* operations) {
  if (operations == nullptr || text.empty()) {
    return false;
  }
  std::vector<OperationKind> parsed;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    const std::string token(text.substr(begin, end - begin));
    if (token == "clone") {
      parsed.push_back(OperationKind::kClone);
    } else if (token == "update") {
      parsed.push_back(OperationKind::kUpdate);
    } else if (token == "mamba_update") {
      parsed.push_back(OperationKind::kMambaUpdate);
    } else if (token == "fixture_update") {
      parsed.push_back(OperationKind::kFixtureUpdate);
    } else if (token == "fixture_chain") {
      parsed.push_back(OperationKind::kFixtureChain);
    } else if (token == "fixture_trace") {
      parsed.push_back(OperationKind::kFixtureTrace);
    } else {
      return false;
    }
    begin = end + 1;
  }
  if (parsed.empty()) {
    return false;
  }
  *operations = std::move(parsed);
  return true;
}

void PrintUsage(const char* argv0) {
  const BenchmarkOptions defaults;
  const auto join_unsigned_csv = [](const auto& values) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i) {
      oss << values[i];
      if (i + 1 != values.size()) {
        oss << ",";
      }
    }
    return oss.str();
  };
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --concurrency <csv>          Active request counts, default: 1,4,8\n"
      << "  --operations <csv>           clone,update,mamba_update,fixture_update,fixture_chain,fixture_trace default: clone,update,mamba_update,fixture_update\n"
      << "  --warmup <count>             Warmup iterations, default: 1\n"
      << "  --iterations <count>         Hot timed iterations, default: 5\n"
      << "  --sr-seed <uint64>           Seed for fp16_sr stochastic rounding, default: "
      << kDefaultStochasticRoundingSeed << "\n"
      << "  --sr-sweep-seeds <csv>       Seeds for fp16_sr fixture_chain sweep, default: "
      << join_unsigned_csv(defaults.stochastic_rounding_sweep_seeds) << "\n"
      << "  --fixture-chain-steps <csv>  Step counts for fixture_chain, default: "
      << join_unsigned_csv(defaults.fixture_chain_steps) << "\n"
      << "  --fixture-trace-steps <csv>  Step counts for fixture_trace; phase-end checkpoints are added automatically, default: "
      << join_unsigned_csv(defaults.fixture_trace_steps) << "\n"
      << "  --dt-threshold <float>       Gate SR to RN when |dt| exceeds this; default: disabled (inf)\n"
      << "  --fixture-root <path>        Layer-derived fixture root, default: "
      << defaults.fixture_root.string() << "\n"
      << "  --trace-fixture-root <path>  Target-shaped trace fixture root, default: "
      << defaults.trace_fixture_root.string() << "\n"
      << "  --json-output <path>         Write JSON results to a file\n";
}

bool ParseArgs(int argc, char** argv, BenchmarkOptions* options) {
  if (options == nullptr) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--concurrency") {
      if (i + 1 >= argc || !ParseCsvUnsigned(argv[++i], &options->concurrency_values)) {
        return false;
      }
    } else if (arg == "--operations") {
      if (i + 1 >= argc || !ParseOperations(argv[++i], &options->operations)) {
        return false;
      }
    } else if (arg == "--warmup") {
      if (i + 1 >= argc) {
        return false;
      }
      options->warmup_iterations = std::stoull(argv[++i]);
    } else if (arg == "--iterations") {
      if (i + 1 >= argc) {
        return false;
      }
      options->hot_iterations = std::stoull(argv[++i]);
    } else if (arg == "--sr-seed") {
      if (i + 1 >= argc) {
        return false;
      }
      options->stochastic_rounding_seed = std::stoull(argv[++i]);
    } else if (arg == "--sr-sweep-seeds") {
      if (i + 1 >= argc || !ParseCsvUint64(argv[++i], &options->stochastic_rounding_sweep_seeds)) {
        return false;
      }
    } else if (arg == "--fixture-chain-steps") {
      if (i + 1 >= argc || !ParseCsvUnsigned(argv[++i], &options->fixture_chain_steps)) {
        return false;
      }
    } else if (arg == "--fixture-trace-steps") {
      if (i + 1 >= argc || !ParseCsvUnsigned(argv[++i], &options->fixture_trace_steps)) {
        return false;
      }
    } else if (arg == "--dt-threshold") {
      if (i + 1 >= argc) {
        return false;
      }
      try {
        options->dt_gate_threshold = std::stof(argv[++i]);
      } catch (...) {
        return false;
      }
    } else if (arg == "--json-output") {
      if (i + 1 >= argc) {
        return false;
      }
      options->json_output_path = std::filesystem::path(argv[++i]);
    } else if (arg == "--fixture-root") {
      if (i + 1 >= argc) {
        return false;
      }
      options->fixture_root = std::filesystem::path(argv[++i]);
    } else if (arg == "--trace-fixture-root") {
      if (i + 1 >= argc) {
        return false;
      }
      options->trace_fixture_root = std::filesystem::path(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      return false;
    }
  }
  return options->hot_iterations > 0;
}

EnvironmentInfo CollectEnvironmentInfo() {
  EnvironmentInfo info;
  int device = 0;
  cudaGetDevice(&device);
  info.device_index = device;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    info.device_name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_global_mem_bytes = prop.totalGlobalMem;
  }
  cudaRuntimeGetVersion(&info.runtime_version);
  cudaDriverGetVersion(&info.driver_version);
  return info;
}

std::vector<float> LoadFloatTensor(const std::filesystem::path& path) {
  const std::uintmax_t bytes = std::filesystem::file_size(path);
  if (bytes % sizeof(float) != 0) {
    throw std::runtime_error("unexpected tensor byte size for " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::vector<float> values(bytes / sizeof(float), 0.0f);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  if (!input || input.gcount() != static_cast<std::streamsize>(bytes)) {
    throw std::runtime_error("unexpected tensor size for " + path.string());
  }
  return values;
}

std::size_t ExtractUnsignedField(std::string_view text, std::string_view key) {
  const std::string pattern = "\"" + std::string(key) + "\"";
  const std::size_t key_pos = text.find(pattern);
  if (key_pos == std::string_view::npos) {
    throw std::runtime_error("missing field " + std::string(key));
  }
  const std::size_t colon_pos = text.find(':', key_pos + pattern.size());
  if (colon_pos == std::string_view::npos) {
    throw std::runtime_error("missing colon for field " + std::string(key));
  }
  std::size_t start = colon_pos + 1;
  while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])) != 0) {
    ++start;
  }
  std::size_t end = start;
  while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end])) != 0) {
    ++end;
  }
  if (start == end) {
    throw std::runtime_error("invalid numeric field " + std::string(key));
  }
  return std::stoull(std::string(text.substr(start, end - start)));
}

std::string ExtractStringField(std::string_view text, std::string_view key) {
  const std::string pattern = "\"" + std::string(key) + "\"";
  const std::size_t key_pos = text.find(pattern);
  if (key_pos == std::string_view::npos) {
    throw std::runtime_error("missing field " + std::string(key));
  }
  const std::size_t colon_pos = text.find(':', key_pos + pattern.size());
  if (colon_pos == std::string_view::npos) {
    throw std::runtime_error("missing colon for field " + std::string(key));
  }
  const std::size_t first_quote = text.find('"', colon_pos + 1);
  if (first_quote == std::string_view::npos) {
    throw std::runtime_error("missing opening quote for field " + std::string(key));
  }
  const std::size_t second_quote = text.find('"', first_quote + 1);
  if (second_quote == std::string_view::npos) {
    throw std::runtime_error("missing closing quote for field " + std::string(key));
  }
  return std::string(text.substr(first_quote + 1, second_quote - first_quote - 1));
}

std::size_t FindMatchingDelimiter(
    std::string_view text,
    std::size_t open_pos,
    char open_char,
    char close_char) {
  if (open_pos >= text.size() || text[open_pos] != open_char) {
    return std::string_view::npos;
  }
  std::size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (std::size_t i = open_pos; i < text.size(); ++i) {
    const char ch = text[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
      continue;
    }
    if (ch == open_char) {
      ++depth;
      continue;
    }
    if (ch == close_char) {
      if (depth == 0) {
        return std::string_view::npos;
      }
      --depth;
      if (depth == 0) {
        return i;
      }
    }
  }
  return std::string_view::npos;
}

std::vector<TracePhase> ParseTracePhases(std::string_view text) {
  const std::string array_pattern = "\"trace_phases\"";
  const std::size_t array_pos = text.find(array_pattern);
  if (array_pos == std::string_view::npos) {
    throw std::runtime_error("missing trace_phases");
  }
  const std::size_t array_start = text.find('[', array_pos + array_pattern.size());
  if (array_start == std::string_view::npos) {
    throw std::runtime_error("missing trace_phases array");
  }
  const std::size_t array_end = FindMatchingDelimiter(text, array_start, '[', ']');
  if (array_end == std::string_view::npos) {
    throw std::runtime_error("unterminated trace_phases array");
  }

  std::vector<TracePhase> phases;
  std::size_t cumulative_steps = 0;
  std::size_t cursor = array_start;
  while (true) {
    const std::size_t object_start = text.find('{', cursor);
    if (object_start == std::string_view::npos || object_start >= array_end) {
      break;
    }
    const std::size_t object_end = FindMatchingDelimiter(text, object_start, '{', '}');
    if (object_end == std::string_view::npos || object_end > array_end) {
      throw std::runtime_error("unterminated trace phase object");
    }
    const std::string_view object = text.substr(object_start, object_end - object_start + 1);
    TracePhase phase;
    phase.name = ExtractStringField(object, "name");
    phase.length = ExtractUnsignedField(object, "length");
    phase.kind = ExtractStringField(object, "kind");
    cumulative_steps += phase.length;
    phase.end_step = cumulative_steps;
    phases.push_back(std::move(phase));
    cursor = object_end + 1;
  }
  if (phases.empty()) {
    throw std::runtime_error("missing trace phases");
  }
  return phases;
}

struct FixtureOracleData {
  std::filesystem::path root;
  std::string name;
  std::size_t batch_size = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;
  std::size_t state_size = 0;
  std::size_t hidden_count_per_request = 0;
  std::size_t state_count_per_request = 0;
  std::size_t bc_count_per_request = 0;
  std::vector<float> ssm_state;
  std::vector<float> hidden;
  std::vector<float> dt;
  std::vector<float> A;
  std::vector<float> B;
  std::vector<float> C;
  std::vector<float> D;
  std::vector<float> expected_next_state;
  std::vector<float> expected_output;
};

struct FixtureReferenceChain {
  std::size_t step_count = 0;
  std::vector<float> final_state;
  std::vector<float> final_output;
};

struct FixtureTraceData {
  std::filesystem::path root;
  std::string name;
  std::size_t batch_size = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;
  std::size_t state_size = 0;
  std::size_t trace_step_count = 0;
  std::size_t shared_system_root_tokens = 0;
  std::size_t committed_head_tokens = 0;
  std::size_t hidden_count_per_request = 0;
  std::size_t state_count_per_request = 0;
  std::size_t bc_count_per_request = 0;
  std::vector<TracePhase> phases;
  std::vector<float> initial_state;
  std::vector<float> A;
  std::vector<float> D;
  std::vector<float> hidden_trace;
  std::vector<float> dt_trace;
  std::vector<float> B_trace;
  std::vector<float> C_trace;
};

FixtureOracleData LoadFixtureOracleData(const std::filesystem::path& root) {
  const std::filesystem::path metadata_path = root / "metadata.json";
  std::ifstream input(metadata_path);
  if (!input) {
    throw std::runtime_error("failed to open " + metadata_path.string());
  }
  const std::string metadata((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

  FixtureOracleData fixture;
  fixture.root = root;
  fixture.name = root.filename().string();
  fixture.batch_size = ExtractUnsignedField(metadata, "batch_size");
  fixture.num_heads = ExtractUnsignedField(metadata, "num_heads");
  fixture.head_dim = ExtractUnsignedField(metadata, "head_dim");
  fixture.state_size = ExtractUnsignedField(metadata, "state_size");
  fixture.hidden_count_per_request = fixture.batch_size * fixture.num_heads * fixture.head_dim;
  fixture.state_count_per_request = fixture.hidden_count_per_request * fixture.state_size;
  fixture.bc_count_per_request = fixture.batch_size * fixture.num_heads * fixture.state_size;
  fixture.ssm_state = LoadFloatTensor(root / "ssm_state_fp32.bin");
  fixture.hidden = LoadFloatTensor(root / "hidden_fp32.bin");
  fixture.dt = LoadFloatTensor(root / "dt_fp32.bin");
  fixture.A = LoadFloatTensor(root / "A_fp32.bin");
  fixture.B = LoadFloatTensor(root / "B_fp32.bin");
  fixture.C = LoadFloatTensor(root / "C_fp32.bin");
  fixture.D = LoadFloatTensor(root / "D_fp32.bin");
  fixture.expected_next_state = LoadFloatTensor(root / "expected_next_state_fp32.bin");
  fixture.expected_output = LoadFloatTensor(root / "expected_output_fp32.bin");
  if (fixture.ssm_state.size() != fixture.state_count_per_request ||
      fixture.A.size() != fixture.state_count_per_request ||
      fixture.expected_next_state.size() != fixture.state_count_per_request ||
      fixture.hidden.size() != fixture.hidden_count_per_request ||
      fixture.dt.size() != fixture.hidden_count_per_request ||
      fixture.D.size() != fixture.hidden_count_per_request ||
      fixture.expected_output.size() != fixture.hidden_count_per_request ||
      fixture.B.size() != fixture.bc_count_per_request ||
      fixture.C.size() != fixture.bc_count_per_request) {
    throw std::runtime_error("fixture tensor sizes do not match metadata in " + root.string());
  }
  return fixture;
}

std::map<std::size_t, FixtureReferenceChain> BuildFixtureReferenceChains(
    const FixtureOracleData& fixture,
    std::vector<std::size_t> step_counts) {
  step_counts.erase(std::remove(step_counts.begin(), step_counts.end(), 0), step_counts.end());
  if (step_counts.empty()) {
    throw std::runtime_error("fixture chain step list must not be empty");
  }
  std::sort(step_counts.begin(), step_counts.end());
  step_counts.erase(std::unique(step_counts.begin(), step_counts.end()), step_counts.end());

  std::map<std::size_t, FixtureReferenceChain> references;
  std::vector<float> state = fixture.ssm_state;
  std::vector<float> next_state(fixture.state_count_per_request, 0.0f);
  std::vector<float> output(fixture.hidden_count_per_request, 0.0f);

  std::size_t next_capture_index = 0;
  const std::size_t max_steps = step_counts.back();
  for (std::size_t step = 1; step <= max_steps; ++step) {
    for (std::size_t hidden_index = 0; hidden_index < fixture.hidden_count_per_request; ++hidden_index) {
      const std::size_t head_index = hidden_index / fixture.head_dim;
      const std::size_t state_base = hidden_index * fixture.state_size;
      const std::size_t bc_base = head_index * fixture.state_size;
      const float dt_value = fixture.dt[hidden_index];
      const float hidden_value = fixture.hidden[hidden_index];
      float accum = 0.0f;
      for (std::size_t state_index = 0; state_index < fixture.state_size; ++state_index) {
        const std::size_t offset = state_base + state_index;
        const float next =
            state[offset] * std::exp(dt_value * fixture.A[offset]) +
            dt_value * fixture.B[bc_base + state_index] * hidden_value;
        next_state[offset] = next;
        accum += next * fixture.C[bc_base + state_index];
      }
      output[hidden_index] = accum + hidden_value * fixture.D[hidden_index];
    }
    state.swap(next_state);
    if (next_capture_index < step_counts.size() && step_counts[next_capture_index] == step) {
      references.emplace(
          step,
          FixtureReferenceChain{
              step,
              state,
              output,
          });
      while (next_capture_index < step_counts.size() && step_counts[next_capture_index] == step) {
        ++next_capture_index;
      }
    }
  }
  return references;
}

FixtureTraceData LoadFixtureTraceData(const std::filesystem::path& root) {
  const std::filesystem::path metadata_path = root / "metadata.json";
  std::ifstream input(metadata_path);
  if (!input) {
    throw std::runtime_error("failed to open " + metadata_path.string());
  }
  const std::string metadata((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

  FixtureTraceData trace;
  trace.root = root;
  trace.name = root.filename().string();
  trace.batch_size = ExtractUnsignedField(metadata, "batch_size");
  trace.num_heads = ExtractUnsignedField(metadata, "num_heads");
  trace.head_dim = ExtractUnsignedField(metadata, "head_dim");
  trace.state_size = ExtractUnsignedField(metadata, "state_size");
  trace.trace_step_count = ExtractUnsignedField(metadata, "trace_step_count");
  trace.shared_system_root_tokens = ExtractUnsignedField(metadata, "shared_system_root_tokens");
  trace.committed_head_tokens = ExtractUnsignedField(metadata, "committed_head_tokens");
  trace.hidden_count_per_request = trace.batch_size * trace.num_heads * trace.head_dim;
  trace.state_count_per_request = trace.hidden_count_per_request * trace.state_size;
  trace.bc_count_per_request = trace.batch_size * trace.num_heads * trace.state_size;
  trace.phases = ParseTracePhases(metadata);
  if (trace.phases.back().end_step != trace.trace_step_count) {
    throw std::runtime_error("trace phase lengths do not match trace_step_count in " + root.string());
  }

  trace.initial_state = LoadFloatTensor(root / "initial_state_fp32.bin");
  trace.A = LoadFloatTensor(root / "A_fp32.bin");
  trace.D = LoadFloatTensor(root / "D_fp32.bin");
  trace.hidden_trace = LoadFloatTensor(root / "hidden_trace_fp32.bin");
  trace.dt_trace = LoadFloatTensor(root / "dt_trace_fp32.bin");
  trace.B_trace = LoadFloatTensor(root / "B_trace_fp32.bin");
  trace.C_trace = LoadFloatTensor(root / "C_trace_fp32.bin");

  if (trace.initial_state.size() != trace.state_count_per_request ||
      trace.A.size() != trace.state_count_per_request ||
      trace.D.size() != trace.hidden_count_per_request ||
      trace.hidden_trace.size() != trace.trace_step_count * trace.hidden_count_per_request ||
      trace.dt_trace.size() != trace.trace_step_count * trace.hidden_count_per_request ||
      trace.B_trace.size() != trace.trace_step_count * trace.bc_count_per_request ||
      trace.C_trace.size() != trace.trace_step_count * trace.bc_count_per_request) {
    throw std::runtime_error("trace tensor sizes do not match metadata in " + root.string());
  }
  return trace;
}

std::vector<std::size_t> MergeTraceCheckpointSteps(
    const FixtureTraceData& trace,
    std::vector<std::size_t> step_counts) {
  for (const TracePhase& phase : trace.phases) {
    step_counts.push_back(phase.end_step);
  }
  step_counts.erase(std::remove(step_counts.begin(), step_counts.end(), 0), step_counts.end());
  if (step_counts.empty()) {
    throw std::runtime_error("fixture trace step list must not be empty");
  }
  std::sort(step_counts.begin(), step_counts.end());
  step_counts.erase(std::unique(step_counts.begin(), step_counts.end()), step_counts.end());
  if (step_counts.back() > trace.trace_step_count) {
    throw std::runtime_error("requested trace step count exceeds available trace length");
  }
  return step_counts;
}

const TracePhase* FindTracePhaseForStep(const FixtureTraceData* trace, std::size_t step_count) {
  if (trace == nullptr) {
    return nullptr;
  }
  for (const TracePhase& phase : trace->phases) {
    if (phase.end_step == step_count) {
      return &phase;
    }
  }
  return nullptr;
}

std::map<std::size_t, FixtureReferenceChain> BuildFixtureTraceReferences(
    const FixtureTraceData& trace,
    std::vector<std::size_t> step_counts) {
  step_counts = MergeTraceCheckpointSteps(trace, std::move(step_counts));

  std::map<std::size_t, FixtureReferenceChain> references;
  std::vector<float> state = trace.initial_state;
  std::vector<float> next_state(trace.state_count_per_request, 0.0f);
  std::vector<float> output(trace.hidden_count_per_request, 0.0f);

  std::size_t next_capture_index = 0;
  const std::size_t max_steps = step_counts.back();
  for (std::size_t step = 1; step <= max_steps; ++step) {
    const float* hidden = trace.hidden_trace.data() + (step - 1) * trace.hidden_count_per_request;
    const float* dt = trace.dt_trace.data() + (step - 1) * trace.hidden_count_per_request;
    const float* B = trace.B_trace.data() + (step - 1) * trace.bc_count_per_request;
    const float* C = trace.C_trace.data() + (step - 1) * trace.bc_count_per_request;
    for (std::size_t hidden_index = 0; hidden_index < trace.hidden_count_per_request; ++hidden_index) {
      const std::size_t head_index = hidden_index / trace.head_dim;
      const std::size_t state_base = hidden_index * trace.state_size;
      const std::size_t bc_base = head_index * trace.state_size;
      const float dt_value = dt[hidden_index];
      const float hidden_value = hidden[hidden_index];
      float accum = 0.0f;
      for (std::size_t state_index = 0; state_index < trace.state_size; ++state_index) {
        const std::size_t offset = state_base + state_index;
        const float next =
            state[offset] * std::exp(dt_value * trace.A[offset]) +
            dt_value * B[bc_base + state_index] * hidden_value;
        next_state[offset] = next;
        accum += next * C[bc_base + state_index];
      }
      output[hidden_index] = accum + hidden_value * trace.D[hidden_index];
    }
    state.swap(next_state);
    if (next_capture_index < step_counts.size() && step_counts[next_capture_index] == step) {
      references.emplace(
          step,
          FixtureReferenceChain{
              step,
              state,
              output,
          });
      while (next_capture_index < step_counts.size() && step_counts[next_capture_index] == step) {
        ++next_capture_index;
      }
    }
  }
  return references;
}

__device__ __forceinline__ uint2 PhiloxMulHiLo(std::uint32_t a, std::uint32_t b) {
  const std::uint64_t product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
  return make_uint2(static_cast<std::uint32_t>(product), static_cast<std::uint32_t>(product >> 32));
}

__device__ __forceinline__ uint4 Philox4x32(uint4 counter, uint2 key) {
  constexpr std::uint32_t kPhiloxM0 = 0xD2511F53u;
  constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57u;
  constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9u;
  constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85u;

  for (std::uint32_t round = 0; round < kPhiloxRoundCount; ++round) {
    const uint2 prod0 = PhiloxMulHiLo(kPhiloxM0, counter.x);
    const uint2 prod1 = PhiloxMulHiLo(kPhiloxM1, counter.z);
    counter = make_uint4(
        prod1.y ^ counter.y ^ key.x,
        prod1.x,
        prod0.y ^ counter.w ^ key.y,
        prod0.x);
    key.x += kPhiloxW0;
    key.y += kPhiloxW1;
  }
  return counter;
}

__device__ __forceinline__ float Uniform01FromBits(std::uint32_t bits) {
  return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ __half StochasticRoundToHalf(float value, std::uint32_t random_bits) {
  if (!isfinite(value)) {
    return __float2half_rn(value);
  }
  const __half lower = __float2half_rd(value);
  const __half upper = __float2half_ru(value);
  const float lower_f = __half2float(lower);
  const float upper_f = __half2float(upper);
  if (lower_f == upper_f) {
    return lower;
  }
  const float interval = upper_f - lower_f;
  if (!(interval > 0.0f)) {
    return __float2half_rn(value);
  }
  const float probability_up = fminf(fmaxf((value - lower_f) / interval, 0.0f), 1.0f);
  return Uniform01FromBits(random_bits) < probability_up ? upper : lower;
}

__global__ void InitFp32Kernel(float* data, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    data[i] = 0.001f * static_cast<float>((i % 97) - 48);
  }
}

__global__ void InitFp16Kernel(__half* data, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    data[i] = __float2half(0.001f * static_cast<float>((i % 97) - 48));
  }
}

__global__ void InitInt8Kernel(std::int8_t* data, float* scales, std::size_t count, std::size_t group_size) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    data[i] = static_cast<std::int8_t>((i % 31) - 15);
  }
  const std::size_t group_count = (count + group_size - 1) / group_size;
  for (std::size_t g = index; g < group_count; g += stride) {
    scales[g] = 1.0f / 127.0f;
  }
}

__global__ void UpdateFp32Kernel(const float* src, float* dst, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    const float value = src[i];
    dst[i] = value * 1.0001f + 0.00001f * static_cast<float>(i & 0xF);
  }
}

__global__ void UpdateFp16Kernel(const __half* src, __half* dst, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    const float value = __half2float(src[i]);
    dst[i] = __float2half_rn(value * 1.0001f + 0.00001f * static_cast<float>(i & 0xF));
  }
}

__global__ void UpdateFp16StochasticKernel(
    const __half* src,
    __half* dst,
    std::size_t count,
    std::uint64_t seed,
    std::uint32_t update_step) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const uint2 key = make_uint2(
      static_cast<std::uint32_t>(seed),
      static_cast<std::uint32_t>(seed >> 32));
  for (std::size_t i = index; i < count; i += stride) {
    const float value = __half2float(src[i]);
    const float updated = value * 1.0001f + 0.00001f * static_cast<float>(i & 0xF);
    const uint4 counter = make_uint4(
        static_cast<std::uint32_t>(i),
        static_cast<std::uint32_t>(i >> 32),
        update_step,
        0xA511E9B3u);
    const uint4 random = Philox4x32(counter, key);
    dst[i] = StochasticRoundToHalf(updated, random.x);
  }
}

__global__ void UpdateInt8GroupKernel(
    const std::int8_t* src,
    std::int8_t* dst,
    const float* src_scales,
    float* dst_scales,
    std::size_t count,
    std::size_t group_size) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    const std::size_t group = i / group_size;
    const float scale = src_scales[group];
    const float value = static_cast<float>(src[i]) * scale;
    const float updated = value * 1.0001f + 0.00001f * static_cast<float>(i & 0xF);
    const float quantized = updated / scale;
    const int clamped = max(-127, min(127, __float2int_rn(quantized)));
    dst[i] = static_cast<std::int8_t>(clamped);
    dst_scales[group] = scale;
  }
}

__device__ __forceinline__ std::size_t WrapNeighborIndex(
    std::size_t index,
    std::size_t count,
    std::uint32_t update_step) {
  const std::size_t offset = static_cast<std::size_t>((update_step & 0x7u) + 1u) * 32ull;
  const std::size_t neighbor = index + offset;
  return neighbor < count ? neighbor : (neighbor - count);
}

__global__ void MambaUpdateFp32Kernel(
    const float* src,
    float* dst,
    std::size_t count,
    std::uint32_t update_step) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    const std::size_t neighbor = WrapNeighborIndex(i, count, update_step);
    const float current = src[i];
    const float carry = src[neighbor];
    const float alpha = 0.985f + 0.00025f * static_cast<float>((i + update_step) & 0x1Fu);
    const float beta = 0.03125f * static_cast<float>(((i >> 3) + update_step) & 0x3u);
    const float drift = 0.00005f * static_cast<float>(static_cast<int>(update_step & 0xFu) - 7);
    dst[i] = alpha * current + beta * carry + drift;
  }
}

__global__ void MambaUpdateFp16Kernel(
    const __half* src,
    __half* dst,
    std::size_t count,
    std::uint32_t update_step) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    const std::size_t neighbor = WrapNeighborIndex(i, count, update_step);
    const float current = __half2float(src[i]);
    const float carry = __half2float(src[neighbor]);
    const float alpha = 0.985f + 0.00025f * static_cast<float>((i + update_step) & 0x1Fu);
    const float beta = 0.03125f * static_cast<float>(((i >> 3) + update_step) & 0x3u);
    const float drift = 0.00005f * static_cast<float>(static_cast<int>(update_step & 0xFu) - 7);
    dst[i] = __float2half_rn(alpha * current + beta * carry + drift);
  }
}

__global__ void MambaUpdateFp16StochasticKernel(
    const __half* src,
    __half* dst,
    std::size_t count,
    std::uint64_t seed,
    std::uint32_t update_step) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const uint2 key = make_uint2(
      static_cast<std::uint32_t>(seed),
      static_cast<std::uint32_t>(seed >> 32));
  for (std::size_t i = index; i < count; i += stride) {
    const std::size_t neighbor = WrapNeighborIndex(i, count, update_step);
    const float current = __half2float(src[i]);
    const float carry = __half2float(src[neighbor]);
    const float alpha = 0.985f + 0.00025f * static_cast<float>((i + update_step) & 0x1Fu);
    const float beta = 0.03125f * static_cast<float>(((i >> 3) + update_step) & 0x3u);
    const float drift = 0.00005f * static_cast<float>(static_cast<int>(update_step & 0xFu) - 7);
    const float updated = alpha * current + beta * carry + drift;
    const uint4 counter = make_uint4(
        static_cast<std::uint32_t>(i),
        static_cast<std::uint32_t>(i >> 32),
        update_step,
        0x4D414D42u);
    const uint4 random = Philox4x32(counter, key);
    dst[i] = StochasticRoundToHalf(updated, random.x);
  }
}

__global__ void MambaUpdateInt8GroupKernel(
    const std::int8_t* src,
    std::int8_t* dst,
    const float* src_scales,
    float* dst_scales,
    std::size_t count,
    std::size_t group_size,
    std::uint32_t update_step) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < count; i += stride) {
    const std::size_t neighbor = WrapNeighborIndex(i, count, update_step);
    const std::size_t group = i / group_size;
    const std::size_t neighbor_group = neighbor / group_size;
    const float current_scale = src_scales[group];
    const float neighbor_scale = src_scales[neighbor_group];
    const float current = static_cast<float>(src[i]) * current_scale;
    const float carry = static_cast<float>(src[neighbor]) * neighbor_scale;
    const float alpha = 0.985f + 0.00025f * static_cast<float>((i + update_step) & 0x1Fu);
    const float beta = 0.03125f * static_cast<float>(((i >> 3) + update_step) & 0x3u);
    const float drift = 0.00005f * static_cast<float>(static_cast<int>(update_step & 0xFu) - 7);
    const float updated = alpha * current + beta * carry + drift;
    const float quantized = updated / current_scale;
    const int clamped = max(-127, min(127, __float2int_rn(quantized)));
    dst[i] = static_cast<std::int8_t>(clamped);
    dst_scales[group] = current_scale;
  }
}

__global__ void FixtureUpdateFp32Kernel(
    const float* src_state,
    const float* hidden,
    const float* dt,
    const float* A,
    const float* B,
    float* dst_state,
    std::size_t total_state_count,
    std::size_t hidden_count_per_request,
    std::size_t state_size,
    std::size_t head_dim) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < total_state_count; i += stride) {
    const std::size_t state_index = i % state_size;
    const std::size_t hidden_index = (i / state_size) % hidden_count_per_request;
    const std::size_t head_index = hidden_index / head_dim;
    const std::size_t b_index = head_index * state_size + state_index;
    const float next_state = src_state[i] * expf(dt[hidden_index] * A[hidden_index * state_size + state_index]) +
                             dt[hidden_index] * B[b_index] * hidden[hidden_index];
    dst_state[i] = next_state;
  }
}

__global__ void FixtureUpdateFp16Kernel(
    const __half* src_state,
    const float* hidden,
    const float* dt,
    const float* A,
    const float* B,
    __half* dst_state,
    std::size_t total_state_count,
    std::size_t hidden_count_per_request,
    std::size_t state_size,
    std::size_t head_dim) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < total_state_count; i += stride) {
    const std::size_t state_index = i % state_size;
    const std::size_t hidden_index = (i / state_size) % hidden_count_per_request;
    const std::size_t head_index = hidden_index / head_dim;
    const std::size_t b_index = head_index * state_size + state_index;
    const float current = __half2float(src_state[i]);
    const float next_state = current * expf(dt[hidden_index] * A[hidden_index * state_size + state_index]) +
                             dt[hidden_index] * B[b_index] * hidden[hidden_index];
    dst_state[i] = __float2half_rn(next_state);
  }
}

__global__ void FixtureUpdateFp16StochasticKernel(
    const __half* src_state,
    const float* hidden,
    const float* dt,
    const float* A,
    const float* B,
    __half* dst_state,
    std::size_t total_state_count,
    std::size_t hidden_count_per_request,
    std::size_t state_size,
    std::size_t head_dim,
    std::uint64_t seed,
    std::uint32_t chain_step_index,
    float dt_gate_threshold) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const uint2 key = make_uint2(
      static_cast<std::uint32_t>(seed),
      static_cast<std::uint32_t>(seed >> 32));
  for (std::size_t i = index; i < total_state_count; i += stride) {
    const std::size_t state_index = i % state_size;
    const std::size_t hidden_index = (i / state_size) % hidden_count_per_request;
    const std::size_t head_index = hidden_index / head_dim;
    const std::size_t b_index = head_index * state_size + state_index;
    const float dt_value = dt[hidden_index];
    const float current = __half2float(src_state[i]);
    const float next_state = current * expf(dt_value * A[hidden_index * state_size + state_index]) +
                             dt_value * B[b_index] * hidden[hidden_index];
    if (fabsf(dt_value) > dt_gate_threshold) {
      dst_state[i] = __float2half_rn(next_state);
    } else {
      const uint4 counter = make_uint4(
          static_cast<std::uint32_t>(i),
          static_cast<std::uint32_t>(i >> 32),
          chain_step_index,
          0x46495854u);
      const uint4 random = Philox4x32(counter, key);
      dst_state[i] = StochasticRoundToHalf(next_state, random.x);
    }
  }
}

__global__ void FixtureOutputFp32Kernel(
    const float* next_state,
    const float* hidden,
    const float* C,
    const float* D,
    float* output,
    std::size_t total_hidden_count,
    std::size_t hidden_count_per_request,
    std::size_t state_size,
    std::size_t head_dim) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < total_hidden_count; i += stride) {
    const std::size_t local_hidden_index = i % hidden_count_per_request;
    const std::size_t head_index = local_hidden_index / head_dim;
    float accum = 0.0f;
    const std::size_t state_base = i * state_size;
    const std::size_t c_base = head_index * state_size;
    for (std::size_t s = 0; s < state_size; ++s) {
      accum += next_state[state_base + s] * C[c_base + s];
    }
    output[i] = accum + hidden[local_hidden_index] * D[local_hidden_index];
  }
}

__global__ void FixtureOutputFp16Kernel(
    const __half* next_state,
    const float* hidden,
    const float* C,
    const float* D,
    float* output,
    std::size_t total_hidden_count,
    std::size_t hidden_count_per_request,
    std::size_t state_size,
    std::size_t head_dim) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = index; i < total_hidden_count; i += stride) {
    const std::size_t local_hidden_index = i % hidden_count_per_request;
    const std::size_t head_index = local_hidden_index / head_dim;
    float accum = 0.0f;
    const std::size_t state_base = i * state_size;
    const std::size_t c_base = head_index * state_size;
    for (std::size_t s = 0; s < state_size; ++s) {
      accum += __half2float(next_state[state_base + s]) * C[c_base + s];
    }
    output[i] = accum + hidden[local_hidden_index] * D[local_hidden_index];
  }
}

bool MeasureGpuWorkMs(const std::function<bool(cudaStream_t)>& work, double* elapsed_ms) {
  if (elapsed_ms == nullptr) {
    return false;
  }
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ok = CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  ok &= CheckCuda(cudaEventCreate(&start), "cudaEventCreate start");
  ok &= CheckCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  if (ok) {
    ok &= CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    ok &= work(stream);
    ok &= CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    ok &= CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
  }
  float ms = 0.0f;
  if (ok) {
    ok &= CheckCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  }
  if (start != nullptr) {
    cudaEventDestroy(start);
  }
  if (stop != nullptr) {
    cudaEventDestroy(stop);
  }
  if (stream != nullptr) {
    cudaStreamDestroy(stream);
  }
  if (ok) {
    *elapsed_ms = static_cast<double>(ms);
  }
  return ok;
}

struct FormatRunBuffers {
  DeviceBuffer state_a;
  DeviceBuffer state_b;
  DeviceBuffer scales_a;
  DeviceBuffer scales_b;
  DeviceBuffer initial_state;
  DeviceBuffer fixture_hidden;
  DeviceBuffer fixture_dt;
  DeviceBuffer fixture_A;
  DeviceBuffer fixture_B;
  DeviceBuffer fixture_C;
  DeviceBuffer fixture_D;
  DeviceBuffer fixture_output;
  DeviceBuffer trace_hidden;
  DeviceBuffer trace_dt;
  DeviceBuffer trace_B;
  DeviceBuffer trace_C;
  DeviceBuffer trace_D;
  DeviceBuffer trace_output;
  std::size_t state_bytes = 0;
  std::size_t scale_bytes = 0;
  std::size_t element_count = 0;
  std::uint32_t update_step = 0;
  const FixtureOracleData* fixture = nullptr;
  const FixtureTraceData* trace_fixture = nullptr;
};

bool LaunchFixtureStep(
    CacheFormat format,
    FormatRunBuffers* buffers,
    std::uint64_t stochastic_rounding_seed,
    std::uint32_t chain_step_index,
    float dt_gate_threshold,
    cudaStream_t stream) {
  if (buffers == nullptr || buffers->fixture == nullptr) {
    return false;
  }
  const int block_size = 256;
  const int grid_size = static_cast<int>((buffers->element_count + block_size - 1) / block_size);
  const std::size_t hidden_count_total = buffers->fixture->hidden_count_per_request *
                                         (buffers->element_count / buffers->fixture->state_count_per_request);
  const int hidden_grid = static_cast<int>((hidden_count_total + block_size - 1) / block_size);
  if (format == CacheFormat::kFp32) {
    FixtureUpdateFp32Kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(buffers->state_a.data),
        reinterpret_cast<const float*>(buffers->fixture_hidden.data),
        reinterpret_cast<const float*>(buffers->fixture_dt.data),
        reinterpret_cast<const float*>(buffers->fixture_A.data),
        reinterpret_cast<const float*>(buffers->fixture_B.data),
        reinterpret_cast<float*>(buffers->state_b.data),
        buffers->element_count,
        buffers->fixture->hidden_count_per_request,
        buffers->fixture->state_size,
        buffers->fixture->head_dim);
    if (!CheckCuda(cudaGetLastError(), "FixtureUpdateFp32Kernel launch")) {
      return false;
    }
    FixtureOutputFp32Kernel<<<hidden_grid, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(buffers->state_b.data),
        reinterpret_cast<const float*>(buffers->fixture_hidden.data),
        reinterpret_cast<const float*>(buffers->fixture_C.data),
        reinterpret_cast<const float*>(buffers->fixture_D.data),
        reinterpret_cast<float*>(buffers->fixture_output.data),
        hidden_count_total,
        buffers->fixture->hidden_count_per_request,
        buffers->fixture->state_size,
        buffers->fixture->head_dim);
  } else if (format == CacheFormat::kFp16) {
    FixtureUpdateFp16Kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_a.data),
        reinterpret_cast<const float*>(buffers->fixture_hidden.data),
        reinterpret_cast<const float*>(buffers->fixture_dt.data),
        reinterpret_cast<const float*>(buffers->fixture_A.data),
        reinterpret_cast<const float*>(buffers->fixture_B.data),
        reinterpret_cast<__half*>(buffers->state_b.data),
        buffers->element_count,
        buffers->fixture->hidden_count_per_request,
        buffers->fixture->state_size,
        buffers->fixture->head_dim);
    if (!CheckCuda(cudaGetLastError(), "FixtureUpdateFp16Kernel launch")) {
      return false;
    }
    FixtureOutputFp16Kernel<<<hidden_grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_b.data),
        reinterpret_cast<const float*>(buffers->fixture_hidden.data),
        reinterpret_cast<const float*>(buffers->fixture_C.data),
        reinterpret_cast<const float*>(buffers->fixture_D.data),
        reinterpret_cast<float*>(buffers->fixture_output.data),
        hidden_count_total,
        buffers->fixture->hidden_count_per_request,
        buffers->fixture->state_size,
        buffers->fixture->head_dim);
  } else if (format == CacheFormat::kFp16Sr) {
    FixtureUpdateFp16StochasticKernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_a.data),
        reinterpret_cast<const float*>(buffers->fixture_hidden.data),
        reinterpret_cast<const float*>(buffers->fixture_dt.data),
        reinterpret_cast<const float*>(buffers->fixture_A.data),
        reinterpret_cast<const float*>(buffers->fixture_B.data),
        reinterpret_cast<__half*>(buffers->state_b.data),
        buffers->element_count,
        buffers->fixture->hidden_count_per_request,
        buffers->fixture->state_size,
        buffers->fixture->head_dim,
        stochastic_rounding_seed,
        chain_step_index,
        dt_gate_threshold);
    if (!CheckCuda(cudaGetLastError(), "FixtureUpdateFp16StochasticKernel launch")) {
      return false;
    }
    FixtureOutputFp16Kernel<<<hidden_grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_b.data),
        reinterpret_cast<const float*>(buffers->fixture_hidden.data),
        reinterpret_cast<const float*>(buffers->fixture_C.data),
        reinterpret_cast<const float*>(buffers->fixture_D.data),
        reinterpret_cast<float*>(buffers->fixture_output.data),
        hidden_count_total,
        buffers->fixture->hidden_count_per_request,
        buffers->fixture->state_size,
        buffers->fixture->head_dim);
  } else {
    return false;
  }
  if (!CheckCuda(cudaGetLastError(), "fixture output kernel launch")) {
    return false;
  }
  std::swap(buffers->state_a.data, buffers->state_b.data);
  return true;
}

bool LaunchTraceStep(
    CacheFormat format,
    FormatRunBuffers* buffers,
    std::uint64_t stochastic_rounding_seed,
    std::size_t trace_step_index,
    float dt_gate_threshold,
    cudaStream_t stream) {
  if (buffers == nullptr || buffers->trace_fixture == nullptr) {
    return false;
  }
  const FixtureTraceData& trace = *buffers->trace_fixture;
  const int block_size = 256;
  const int grid_size = static_cast<int>((buffers->element_count + block_size - 1) / block_size);
  const std::size_t hidden_count_total =
      trace.hidden_count_per_request * (buffers->element_count / trace.state_count_per_request);
  const int hidden_grid = static_cast<int>((hidden_count_total + block_size - 1) / block_size);
  const float* hidden_ptr = reinterpret_cast<const float*>(buffers->trace_hidden.data) +
                            trace_step_index * trace.hidden_count_per_request;
  const float* dt_ptr = reinterpret_cast<const float*>(buffers->trace_dt.data) +
                        trace_step_index * trace.hidden_count_per_request;
  const float* b_ptr = reinterpret_cast<const float*>(buffers->trace_B.data) +
                       trace_step_index * trace.bc_count_per_request;
  const float* c_ptr = reinterpret_cast<const float*>(buffers->trace_C.data) +
                       trace_step_index * trace.bc_count_per_request;

  if (format == CacheFormat::kFp32) {
    FixtureUpdateFp32Kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(buffers->state_a.data),
        hidden_ptr,
        dt_ptr,
        reinterpret_cast<const float*>(buffers->fixture_A.data),
        b_ptr,
        reinterpret_cast<float*>(buffers->state_b.data),
        buffers->element_count,
        trace.hidden_count_per_request,
        trace.state_size,
        trace.head_dim);
    if (!CheckCuda(cudaGetLastError(), "TraceUpdateFp32Kernel launch")) {
      return false;
    }
    FixtureOutputFp32Kernel<<<hidden_grid, block_size, 0, stream>>>(
        reinterpret_cast<const float*>(buffers->state_b.data),
        hidden_ptr,
        c_ptr,
        reinterpret_cast<const float*>(buffers->trace_D.data),
        reinterpret_cast<float*>(buffers->trace_output.data),
        hidden_count_total,
        trace.hidden_count_per_request,
        trace.state_size,
        trace.head_dim);
  } else if (format == CacheFormat::kFp16) {
    FixtureUpdateFp16Kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_a.data),
        hidden_ptr,
        dt_ptr,
        reinterpret_cast<const float*>(buffers->fixture_A.data),
        b_ptr,
        reinterpret_cast<__half*>(buffers->state_b.data),
        buffers->element_count,
        trace.hidden_count_per_request,
        trace.state_size,
        trace.head_dim);
    if (!CheckCuda(cudaGetLastError(), "TraceUpdateFp16Kernel launch")) {
      return false;
    }
    FixtureOutputFp16Kernel<<<hidden_grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_b.data),
        hidden_ptr,
        c_ptr,
        reinterpret_cast<const float*>(buffers->trace_D.data),
        reinterpret_cast<float*>(buffers->trace_output.data),
        hidden_count_total,
        trace.hidden_count_per_request,
        trace.state_size,
        trace.head_dim);
  } else if (format == CacheFormat::kFp16Sr) {
    FixtureUpdateFp16StochasticKernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_a.data),
        hidden_ptr,
        dt_ptr,
        reinterpret_cast<const float*>(buffers->fixture_A.data),
        b_ptr,
        reinterpret_cast<__half*>(buffers->state_b.data),
        buffers->element_count,
        trace.hidden_count_per_request,
        trace.state_size,
        trace.head_dim,
        stochastic_rounding_seed,
        static_cast<std::uint32_t>(trace_step_index + 1),
        dt_gate_threshold);
    if (!CheckCuda(cudaGetLastError(), "TraceUpdateFp16StochasticKernel launch")) {
      return false;
    }
    FixtureOutputFp16Kernel<<<hidden_grid, block_size, 0, stream>>>(
        reinterpret_cast<const __half*>(buffers->state_b.data),
        hidden_ptr,
        c_ptr,
        reinterpret_cast<const float*>(buffers->trace_D.data),
        reinterpret_cast<float*>(buffers->trace_output.data),
        hidden_count_total,
        trace.hidden_count_per_request,
        trace.state_size,
        trace.head_dim);
  } else {
    return false;
  }
  if (!CheckCuda(cudaGetLastError(), "trace output kernel launch")) {
    return false;
  }
  std::swap(buffers->state_a.data, buffers->state_b.data);
  return true;
}

bool UploadFloatTensor(const std::vector<float>& host, DeviceBuffer* device) {
  if (device == nullptr) {
    return false;
  }
  if (!device->Allocate(host.size() * sizeof(float))) {
    return false;
  }
  return CheckCuda(
      cudaMemcpy(device->data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
      "cudaMemcpy float tensor");
}

bool AllocateAndInitFixtureBuffers(
    CacheFormat format,
    std::size_t active_requests,
    const FixtureOracleData& fixture,
    FormatRunBuffers* buffers) {
  if (buffers == nullptr) {
    return false;
  }
  buffers->fixture = &fixture;
  buffers->element_count = fixture.state_count_per_request * active_requests;
  const std::size_t hidden_count_total = fixture.hidden_count_per_request * active_requests;
  std::vector<float> repeated_state(buffers->element_count, 0.0f);
  for (std::size_t request = 0; request < active_requests; ++request) {
    std::copy(
        fixture.ssm_state.begin(),
        fixture.ssm_state.end(),
        repeated_state.begin() + request * fixture.state_count_per_request);
  }

  if (format == CacheFormat::kFp32) {
    buffers->state_bytes = repeated_state.size() * sizeof(float);
    if (!buffers->initial_state.Allocate(buffers->state_bytes) ||
        !buffers->state_a.Allocate(buffers->state_bytes) ||
        !buffers->state_b.Allocate(buffers->state_bytes)) {
      return false;
    }
    if (!CheckCuda(
            cudaMemcpy(
                buffers->initial_state.data,
                repeated_state.data(),
                buffers->state_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy fixture initial_state fp32")) {
      return false;
    }
  } else {
    std::vector<__half> repeated_state_half(repeated_state.size());
    for (std::size_t i = 0; i < repeated_state.size(); ++i) {
      repeated_state_half[i] = __float2half_rn(repeated_state[i]);
    }
    buffers->state_bytes = repeated_state_half.size() * sizeof(__half);
    if (!buffers->initial_state.Allocate(buffers->state_bytes) ||
        !buffers->state_a.Allocate(buffers->state_bytes) ||
        !buffers->state_b.Allocate(buffers->state_bytes)) {
      return false;
    }
    if (!CheckCuda(
            cudaMemcpy(
                buffers->initial_state.data,
                repeated_state_half.data(),
                buffers->state_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy fixture initial_state fp16")) {
      return false;
    }
  }
  if (!UploadFloatTensor(fixture.hidden, &buffers->fixture_hidden) ||
      !UploadFloatTensor(fixture.dt, &buffers->fixture_dt) ||
      !UploadFloatTensor(fixture.A, &buffers->fixture_A) ||
      !UploadFloatTensor(fixture.B, &buffers->fixture_B) ||
      !UploadFloatTensor(fixture.C, &buffers->fixture_C) ||
      !UploadFloatTensor(fixture.D, &buffers->fixture_D) ||
      !buffers->fixture_output.Allocate(hidden_count_total * sizeof(float))) {
    return false;
  }
  return CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize fixture init");
}

bool AllocateAndInitTraceBuffers(
    CacheFormat format,
    std::size_t active_requests,
    const FixtureTraceData& trace,
    FormatRunBuffers* buffers) {
  if (buffers == nullptr) {
    return false;
  }
  buffers->trace_fixture = &trace;
  buffers->element_count = trace.state_count_per_request * active_requests;
  std::vector<float> repeated_state(buffers->element_count, 0.0f);
  for (std::size_t request = 0; request < active_requests; ++request) {
    std::copy(
        trace.initial_state.begin(),
        trace.initial_state.end(),
        repeated_state.begin() + request * trace.state_count_per_request);
  }

  if (format == CacheFormat::kFp32) {
    buffers->state_bytes = repeated_state.size() * sizeof(float);
    if (!buffers->initial_state.Allocate(buffers->state_bytes) ||
        !buffers->state_a.Allocate(buffers->state_bytes) ||
        !buffers->state_b.Allocate(buffers->state_bytes)) {
      return false;
    }
    if (!CheckCuda(
            cudaMemcpy(
                buffers->initial_state.data,
                repeated_state.data(),
                buffers->state_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy trace initial_state fp32")) {
      return false;
    }
  } else {
    std::vector<__half> repeated_state_half(repeated_state.size());
    for (std::size_t i = 0; i < repeated_state.size(); ++i) {
      repeated_state_half[i] = __float2half_rn(repeated_state[i]);
    }
    buffers->state_bytes = repeated_state_half.size() * sizeof(__half);
    if (!buffers->initial_state.Allocate(buffers->state_bytes) ||
        !buffers->state_a.Allocate(buffers->state_bytes) ||
        !buffers->state_b.Allocate(buffers->state_bytes)) {
      return false;
    }
    if (!CheckCuda(
            cudaMemcpy(
                buffers->initial_state.data,
                repeated_state_half.data(),
                buffers->state_bytes,
                cudaMemcpyHostToDevice),
            "cudaMemcpy trace initial_state fp16")) {
      return false;
    }
  }
  if (!UploadFloatTensor(trace.A, &buffers->fixture_A) ||
      !UploadFloatTensor(trace.hidden_trace, &buffers->trace_hidden) ||
      !UploadFloatTensor(trace.dt_trace, &buffers->trace_dt) ||
      !UploadFloatTensor(trace.B_trace, &buffers->trace_B) ||
      !UploadFloatTensor(trace.C_trace, &buffers->trace_C) ||
      !UploadFloatTensor(trace.D, &buffers->trace_D) ||
      !buffers->trace_output.Allocate(trace.hidden_count_per_request * active_requests * sizeof(float))) {
    return false;
  }
  return CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize trace init");
}

bool AllocateAndInitBuffers(
    CacheFormat format,
    std::size_t active_requests,
    OperationKind operation,
    const FixtureOracleData* fixture,
    const FixtureTraceData* trace_fixture,
    FormatRunBuffers* buffers) {
  if (operation == OperationKind::kFixtureUpdate || operation == OperationKind::kFixtureChain) {
    if (fixture == nullptr || format == CacheFormat::kInt8Group) {
      return false;
    }
    return AllocateAndInitFixtureBuffers(format, active_requests, *fixture, buffers);
  }
  if (operation == OperationKind::kFixtureTrace) {
    if (trace_fixture == nullptr || format == CacheFormat::kInt8Group) {
      return false;
    }
    return AllocateAndInitTraceBuffers(format, active_requests, *trace_fixture, buffers);
  }
  if (buffers == nullptr) {
    return false;
  }
  buffers->element_count = kLogicalElementsPerRequest * active_requests;

  const int block_size = 256;
  const int grid_size = static_cast<int>((buffers->element_count + block_size - 1) / block_size);

  if (format == CacheFormat::kFp32) {
    buffers->state_bytes = kPerRequestFp32Bytes * active_requests;
    if (!buffers->state_a.Allocate(buffers->state_bytes) || !buffers->state_b.Allocate(buffers->state_bytes)) {
      return false;
    }
    InitFp32Kernel<<<grid_size, block_size>>>(
        reinterpret_cast<float*>(buffers->state_a.data),
        buffers->element_count);
    InitFp32Kernel<<<grid_size, block_size>>>(
        reinterpret_cast<float*>(buffers->state_b.data),
        buffers->element_count);
  } else if (format == CacheFormat::kFp16 || format == CacheFormat::kFp16Sr) {
    buffers->state_bytes = kPerRequestFp16Bytes * active_requests;
    if (!buffers->state_a.Allocate(buffers->state_bytes) || !buffers->state_b.Allocate(buffers->state_bytes)) {
      return false;
    }
    InitFp16Kernel<<<grid_size, block_size>>>(
        reinterpret_cast<__half*>(buffers->state_a.data),
        buffers->element_count);
    InitFp16Kernel<<<grid_size, block_size>>>(
        reinterpret_cast<__half*>(buffers->state_b.data),
        buffers->element_count);
  } else {
    buffers->state_bytes = buffers->element_count * sizeof(std::int8_t);
    const std::size_t group_count = (buffers->element_count + kInt8GroupSize - 1) / kInt8GroupSize;
    buffers->scale_bytes = group_count * sizeof(float);
    if (!buffers->state_a.Allocate(buffers->state_bytes) ||
        !buffers->state_b.Allocate(buffers->state_bytes) ||
        !buffers->scales_a.Allocate(buffers->scale_bytes) ||
        !buffers->scales_b.Allocate(buffers->scale_bytes)) {
      return false;
    }
    InitInt8Kernel<<<grid_size, block_size>>>(
        reinterpret_cast<std::int8_t*>(buffers->state_a.data),
        reinterpret_cast<float*>(buffers->scales_a.data),
        buffers->element_count,
        kInt8GroupSize);
    InitInt8Kernel<<<grid_size, block_size>>>(
        reinterpret_cast<std::int8_t*>(buffers->state_b.data),
        reinterpret_cast<float*>(buffers->scales_b.data),
        buffers->element_count,
        kInt8GroupSize);
  }
  return CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize init");
}

bool ResetFixtureState(CacheFormat format, FormatRunBuffers* buffers) {
  if (buffers == nullptr) {
    return false;
  }
  bool ok = CheckCuda(
      cudaMemcpy(
          buffers->state_a.data,
          buffers->initial_state.data,
          buffers->state_bytes,
          cudaMemcpyDeviceToDevice),
      "cudaMemcpy fixture reset");
  if (format == CacheFormat::kInt8Group && buffers->scale_bytes > 0) {
    ok &= CheckCuda(
        cudaMemcpy(
            buffers->scales_a.data,
            buffers->scales_b.data,
            buffers->scale_bytes,
            cudaMemcpyDeviceToDevice),
        "cudaMemcpy fixture reset scales");
  }
  return ok && CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize fixture reset");
}

bool RunOneIteration(
    CacheFormat format,
    OperationKind operation,
    FormatRunBuffers* buffers,
    std::uint64_t stochastic_rounding_seed,
    float dt_gate_threshold,
    std::size_t fixture_steps,
    double* elapsed_ms) {
  if (buffers == nullptr) {
    return false;
  }
  if (operation == OperationKind::kFixtureUpdate ||
      operation == OperationKind::kFixtureChain ||
      operation == OperationKind::kFixtureTrace) {
    if (!ResetFixtureState(format, buffers)) {
      return false;
    }
  }
  auto work = [&](cudaStream_t stream) -> bool {
    const int block_size = 256;
    const int grid_size = static_cast<int>((buffers->element_count + block_size - 1) / block_size);
    if (operation == OperationKind::kClone) {
      bool ok = CheckCuda(
          cudaMemcpyAsync(buffers->state_b.data, buffers->state_a.data, buffers->state_bytes, cudaMemcpyDeviceToDevice, stream),
          "cudaMemcpyAsync state clone");
      if (format == CacheFormat::kInt8Group) {
        ok &= CheckCuda(
            cudaMemcpyAsync(buffers->scales_b.data, buffers->scales_a.data, buffers->scale_bytes, cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync scale clone");
      }
      return ok;
    }

    if (operation == OperationKind::kFixtureUpdate || operation == OperationKind::kFixtureChain) {
      if (buffers->fixture == nullptr) {
        return false;
      }
      const std::size_t step_count = operation == OperationKind::kFixtureUpdate ? 1 : fixture_steps;
      for (std::size_t step = 0; step < step_count; ++step) {
        if (!LaunchFixtureStep(
                format,
                buffers,
                stochastic_rounding_seed,
                static_cast<std::uint32_t>(step + 1),
                dt_gate_threshold,
                stream)) {
          return false;
        }
      }
      return true;
    }

    if (operation == OperationKind::kFixtureTrace) {
      if (buffers->trace_fixture == nullptr || fixture_steps == 0 ||
          fixture_steps > buffers->trace_fixture->trace_step_count) {
        return false;
      }
      for (std::size_t step = 0; step < fixture_steps; ++step) {
        if (!LaunchTraceStep(
                format,
                buffers,
                stochastic_rounding_seed,
                step,
                dt_gate_threshold,
                stream)) {
          return false;
        }
      }
      return true;
    }

    if (format == CacheFormat::kFp32) {
      if (operation == OperationKind::kUpdate) {
        UpdateFp32Kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(buffers->state_a.data),
            reinterpret_cast<float*>(buffers->state_b.data),
            buffers->element_count);
      } else {
        MambaUpdateFp32Kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const float*>(buffers->state_a.data),
            reinterpret_cast<float*>(buffers->state_b.data),
            buffers->element_count,
            buffers->update_step);
      }
    } else if (format == CacheFormat::kFp16) {
      if (operation == OperationKind::kUpdate) {
        UpdateFp16Kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(buffers->state_a.data),
            reinterpret_cast<__half*>(buffers->state_b.data),
            buffers->element_count);
      } else {
        MambaUpdateFp16Kernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(buffers->state_a.data),
            reinterpret_cast<__half*>(buffers->state_b.data),
            buffers->element_count,
            buffers->update_step);
      }
    } else if (format == CacheFormat::kFp16Sr) {
      if (operation == OperationKind::kUpdate) {
        UpdateFp16StochasticKernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(buffers->state_a.data),
            reinterpret_cast<__half*>(buffers->state_b.data),
            buffers->element_count,
            stochastic_rounding_seed,
            buffers->update_step);
      } else {
        MambaUpdateFp16StochasticKernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(buffers->state_a.data),
            reinterpret_cast<__half*>(buffers->state_b.data),
            buffers->element_count,
            stochastic_rounding_seed,
            buffers->update_step);
      }
    } else {
      if (operation == OperationKind::kUpdate) {
        UpdateInt8GroupKernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const std::int8_t*>(buffers->state_a.data),
            reinterpret_cast<std::int8_t*>(buffers->state_b.data),
            reinterpret_cast<const float*>(buffers->scales_a.data),
            reinterpret_cast<float*>(buffers->scales_b.data),
            buffers->element_count,
            kInt8GroupSize);
      } else {
        MambaUpdateInt8GroupKernel<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<const std::int8_t*>(buffers->state_a.data),
            reinterpret_cast<std::int8_t*>(buffers->state_b.data),
            reinterpret_cast<const float*>(buffers->scales_a.data),
            reinterpret_cast<float*>(buffers->scales_b.data),
            buffers->element_count,
            kInt8GroupSize,
            buffers->update_step);
      }
    }
    return CheckCuda(cudaGetLastError(), "update kernel launch");
  };

  const bool ok = MeasureGpuWorkMs(work, elapsed_ms);
  if (ok && operation != OperationKind::kClone &&
      operation != OperationKind::kFixtureUpdate &&
      operation != OperationKind::kFixtureChain &&
      operation != OperationKind::kFixtureTrace) {
    buffers->update_step += 1;
  }
  if (operation != OperationKind::kFixtureUpdate &&
      operation != OperationKind::kFixtureChain &&
      operation != OperationKind::kFixtureTrace) {
    std::swap(buffers->state_a.data, buffers->state_b.data);
    std::swap(buffers->scales_a.data, buffers->scales_b.data);
  }
  return ok;
}

std::size_t ApproximateTrafficBytes(
    CacheFormat format,
    OperationKind operation,
    const FormatRunBuffers& buffers,
    std::size_t fixture_steps) {
  if (operation == OperationKind::kFixtureUpdate ||
      operation == OperationKind::kFixtureChain ||
      operation == OperationKind::kFixtureTrace) {
    const std::size_t state_count_per_request =
        buffers.fixture != nullptr ? buffers.fixture->state_count_per_request
                                   : (buffers.trace_fixture != nullptr ? buffers.trace_fixture->state_count_per_request : 0);
    const std::size_t hidden_count_per_request =
        buffers.fixture != nullptr ? buffers.fixture->hidden_count_per_request
                                   : (buffers.trace_fixture != nullptr ? buffers.trace_fixture->hidden_count_per_request : 0);
    const std::size_t bc_count_per_request =
        buffers.fixture != nullptr ? buffers.fixture->bc_count_per_request
                                   : (buffers.trace_fixture != nullptr ? buffers.trace_fixture->bc_count_per_request : 0);
    if (state_count_per_request == 0 || hidden_count_per_request == 0 || bc_count_per_request == 0) {
      return 0;
    }
    const std::size_t active_requests = buffers.element_count / state_count_per_request;
    const std::size_t per_request_hidden_bytes = hidden_count_per_request * sizeof(float);
    const std::size_t per_request_state_fp32_bytes = state_count_per_request * sizeof(float);
    const std::size_t per_request_bc_bytes = bc_count_per_request * sizeof(float);
    const std::size_t per_step_bytes =
        3 * buffers.state_bytes +
        active_requests * (2 * per_request_hidden_bytes +
                           per_request_state_fp32_bytes +
                           2 * per_request_bc_bytes +
                           per_request_hidden_bytes +
                           per_request_hidden_bytes);
    return per_step_bytes * std::max<std::size_t>(fixture_steps, 1);
  }
  if (operation == OperationKind::kClone) {
    return 2 * buffers.state_bytes + 2 * buffers.scale_bytes;
  }
  if (operation == OperationKind::kMambaUpdate) {
    if (format == CacheFormat::kInt8Group) {
      return 3 * buffers.state_bytes + 3 * buffers.scale_bytes;
    }
    return 3 * buffers.state_bytes;
  }
  if (format == CacheFormat::kInt8Group) {
    return 2 * buffers.state_bytes + buffers.scale_bytes;
  }
  return 2 * buffers.state_bytes;
}

struct OracleDiffs {
  double max_next_state_abs_diff = 0.0;
  double mean_next_state_abs_diff = 0.0;
  double rms_next_state_abs_diff = 0.0;
  double max_output_abs_diff = 0.0;
  double mean_output_abs_diff = 0.0;
  double rms_output_abs_diff = 0.0;
};

double MeanOf(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double StddevOf(const std::vector<double>& values) {
  if (values.size() <= 1) {
    return 0.0;
  }
  const double mean = MeanOf(values);
  double sum_sq = 0.0;
  for (const double value : values) {
    const double centered = value - mean;
    sum_sq += centered * centered;
  }
  return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

bool CaptureFixtureOracleDiffs(
    CacheFormat format,
    const FormatRunBuffers& buffers,
    const std::vector<float>& expected_next_state,
    const std::vector<float>& expected_output,
    OracleDiffs* diffs) {
  const std::size_t state_count_per_request =
      buffers.fixture != nullptr ? buffers.fixture->state_count_per_request
                                 : (buffers.trace_fixture != nullptr ? buffers.trace_fixture->state_count_per_request : 0);
  const std::size_t hidden_count_per_request =
      buffers.fixture != nullptr ? buffers.fixture->hidden_count_per_request
                                 : (buffers.trace_fixture != nullptr ? buffers.trace_fixture->hidden_count_per_request : 0);
  if (diffs == nullptr || state_count_per_request == 0 || hidden_count_per_request == 0) {
    return false;
  }
  if (expected_next_state.size() != state_count_per_request ||
      expected_output.size() != hidden_count_per_request) {
    return false;
  }
  const std::size_t total_state_count = buffers.element_count;
  const std::size_t total_hidden_count =
      hidden_count_per_request * (buffers.element_count / state_count_per_request);

  std::vector<float> host_state(total_state_count, 0.0f);
  if (format == CacheFormat::kFp32) {
    if (!CheckCuda(
            cudaMemcpy(
                host_state.data(),
                buffers.state_a.data,
                total_state_count * sizeof(float),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy fixture state fp32")) {
      return false;
    }
  } else {
    std::vector<__half> host_state_half(total_state_count);
    if (!CheckCuda(
            cudaMemcpy(
                host_state_half.data(),
                buffers.state_a.data,
                total_state_count * sizeof(__half),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy fixture state fp16")) {
      return false;
    }
    for (std::size_t i = 0; i < total_state_count; ++i) {
      host_state[i] = __half2float(host_state_half[i]);
    }
  }

  std::vector<float> host_output(total_hidden_count, 0.0f);
  const void* output_buffer = buffers.trace_fixture != nullptr ? buffers.trace_output.data : buffers.fixture_output.data;
  if (!CheckCuda(
          cudaMemcpy(
              host_output.data(),
              output_buffer,
              total_hidden_count * sizeof(float),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy fixture output")) {
    return false;
  }

  double state_abs_sum = 0.0;
  double state_sq_sum = 0.0;
  double output_abs_sum = 0.0;
  double output_sq_sum = 0.0;
  for (std::size_t i = 0; i < total_state_count; ++i) {
    const double diff = std::fabs(static_cast<double>(host_state[i]) -
                                  static_cast<double>(expected_next_state[i % expected_next_state.size()]));
    diffs->max_next_state_abs_diff = std::max(diffs->max_next_state_abs_diff, diff);
    state_abs_sum += diff;
    state_sq_sum += diff * diff;
  }
  for (std::size_t i = 0; i < total_hidden_count; ++i) {
    const double diff = std::fabs(static_cast<double>(host_output[i]) -
                                  static_cast<double>(expected_output[i % expected_output.size()]));
    diffs->max_output_abs_diff = std::max(diffs->max_output_abs_diff, diff);
    output_abs_sum += diff;
    output_sq_sum += diff * diff;
  }
  diffs->mean_next_state_abs_diff = state_abs_sum / static_cast<double>(total_state_count);
  diffs->rms_next_state_abs_diff = std::sqrt(state_sq_sum / static_cast<double>(total_state_count));
  diffs->mean_output_abs_diff = output_abs_sum / static_cast<double>(total_hidden_count);
  diffs->rms_output_abs_diff = std::sqrt(output_sq_sum / static_cast<double>(total_hidden_count));
  return true;
}

bool SupportsBenchmarkCombination(CacheFormat format, OperationKind operation) {
  if ((operation == OperationKind::kFixtureUpdate ||
       operation == OperationKind::kFixtureChain ||
       operation == OperationKind::kFixtureTrace) &&
      format == CacheFormat::kInt8Group) {
    return false;
  }
  return true;
}

std::optional<BenchmarkResult> RunBenchmark(
    CacheFormat format,
    OperationKind operation,
    std::size_t active_requests,
    const BenchmarkOptions& options,
    const FixtureOracleData* fixture,
    const FixtureTraceData* trace_fixture,
    const std::map<std::size_t, FixtureReferenceChain>* fixture_chain_references,
    std::size_t fixture_steps) {
  FormatRunBuffers buffers;
  if (!AllocateAndInitBuffers(format, active_requests, operation, fixture, trace_fixture, &buffers)) {
    return std::nullopt;
  }

  std::vector<std::uint64_t> seeds = {options.stochastic_rounding_seed};
  if ((operation == OperationKind::kFixtureChain || operation == OperationKind::kFixtureTrace) &&
      format == CacheFormat::kFp16Sr) {
    seeds = options.stochastic_rounding_sweep_seeds;
  }
  std::vector<double> hot_ms;
  hot_ms.reserve(options.hot_iterations * seeds.size());
  std::vector<OracleDiffs> oracle_samples;
  oracle_samples.reserve(seeds.size());

  const std::vector<float>* expected_next_state = nullptr;
  const std::vector<float>* expected_output = nullptr;
  if ((operation == OperationKind::kFixtureUpdate || operation == OperationKind::kFixtureChain) && fixture != nullptr) {
    if (operation == OperationKind::kFixtureUpdate) {
      expected_next_state = &fixture->expected_next_state;
      expected_output = &fixture->expected_output;
    } else {
      if (fixture_chain_references == nullptr) {
        return std::nullopt;
      }
      const auto it = fixture_chain_references->find(fixture_steps);
      if (it == fixture_chain_references->end()) {
        return std::nullopt;
      }
      expected_next_state = &it->second.final_state;
      expected_output = &it->second.final_output;
    }
  } else if (operation == OperationKind::kFixtureTrace && trace_fixture != nullptr) {
    if (fixture_chain_references == nullptr) {
      return std::nullopt;
    }
    const auto it = fixture_chain_references->find(fixture_steps);
    if (it == fixture_chain_references->end()) {
      return std::nullopt;
    }
    expected_next_state = &it->second.final_state;
    expected_output = &it->second.final_output;
  }

  for (const std::uint64_t seed : seeds) {
    double discard_ms = 0.0;
    for (std::size_t i = 0; i < options.warmup_iterations; ++i) {
      if (!RunOneIteration(format, operation, &buffers, seed, options.dt_gate_threshold, fixture_steps, &discard_ms)) {
        return std::nullopt;
      }
    }

    for (std::size_t i = 0; i < options.hot_iterations; ++i) {
      double elapsed_ms = 0.0;
      if (!RunOneIteration(format, operation, &buffers, seed, options.dt_gate_threshold, fixture_steps, &elapsed_ms)) {
        return std::nullopt;
      }
      hot_ms.push_back(elapsed_ms);
    }

    if (expected_next_state != nullptr && expected_output != nullptr) {
      OracleDiffs diffs;
      if (!CaptureFixtureOracleDiffs(
              format,
              buffers,
              *expected_next_state,
              *expected_output,
              &diffs)) {
        return std::nullopt;
      }
      oracle_samples.push_back(diffs);
    }
  }

  const auto [min_it, max_it] = std::minmax_element(hot_ms.begin(), hot_ms.end());
  const double mean_ms = std::accumulate(hot_ms.begin(), hot_ms.end(), 0.0) / static_cast<double>(hot_ms.size());

  BenchmarkResult result;
  result.format = ToString(format);
  result.operation = ToString(operation);
  result.fixture_name =
      (operation == OperationKind::kFixtureUpdate || operation == OperationKind::kFixtureChain) && fixture != nullptr
          ? fixture->name
          : ((operation == OperationKind::kFixtureTrace && trace_fixture != nullptr) ? trace_fixture->name : "");
  if (operation == OperationKind::kFixtureTrace) {
    if (const TracePhase* phase = FindTracePhaseForStep(trace_fixture, fixture_steps)) {
      result.trace_phase_name = phase->name;
      result.trace_phase_kind = phase->kind;
    }
  }
  result.fixture_steps =
      (operation == OperationKind::kFixtureChain || operation == OperationKind::kFixtureTrace) ? fixture_steps : 1;
  result.oracle_seed_count = seeds.size();
  result.active_requests = active_requests;
  result.logical_elements_per_request =
      (operation == OperationKind::kFixtureUpdate || operation == OperationKind::kFixtureChain) && fixture != nullptr
          ? fixture->state_count_per_request
          : ((operation == OperationKind::kFixtureTrace && trace_fixture != nullptr)
                 ? trace_fixture->state_count_per_request
                 : kLogicalElementsPerRequest);
  result.per_request_state_bytes =
      (operation == OperationKind::kFixtureUpdate || operation == OperationKind::kFixtureChain) && fixture != nullptr
          ? (format == CacheFormat::kFp32
                 ? fixture->state_count_per_request * sizeof(float)
                 : fixture->state_count_per_request * sizeof(__half))
          : ((operation == OperationKind::kFixtureTrace && trace_fixture != nullptr)
                 ? (format == CacheFormat::kFp32
                        ? trace_fixture->state_count_per_request * sizeof(float)
                        : trace_fixture->state_count_per_request * sizeof(__half))
          : (format == CacheFormat::kFp32
                 ? kPerRequestFp32Bytes
                 : ((format == CacheFormat::kFp16 || format == CacheFormat::kFp16Sr)
                        ? kPerRequestFp16Bytes
                        : kLogicalElementsPerRequest * sizeof(std::int8_t))));
  result.per_request_scale_bytes = format == CacheFormat::kInt8Group
      ? ((kLogicalElementsPerRequest + kInt8GroupSize - 1) / kInt8GroupSize) * sizeof(float)
      : 0;
  result.total_state_bytes = buffers.state_bytes;
  result.total_scale_bytes = buffers.scale_bytes;
  result.approximate_traffic_bytes = ApproximateTrafficBytes(format, operation, buffers, fixture_steps);
  result.warmup_iterations = options.warmup_iterations;
  result.hot_iterations = options.hot_iterations;
  result.hot_mean_ms = mean_ms;
  result.hot_min_ms = *min_it;
  result.hot_max_ms = *max_it;
  result.effective_gbps =
      mean_ms > 0.0
          ? static_cast<double>(result.approximate_traffic_bytes) / (mean_ms / 1000.0) / 1.0e9
          : 0.0;
  if (!oracle_samples.empty()) {
    std::vector<double> next_mean_values;
    std::vector<double> next_rms_values;
    std::vector<double> output_mean_values;
    std::vector<double> output_rms_values;
    next_mean_values.reserve(oracle_samples.size());
    next_rms_values.reserve(oracle_samples.size());
    output_mean_values.reserve(oracle_samples.size());
    output_rms_values.reserve(oracle_samples.size());
    result.has_oracle_diffs = true;
    for (const OracleDiffs& diffs : oracle_samples) {
      result.max_next_state_abs_diff = std::max(result.max_next_state_abs_diff, diffs.max_next_state_abs_diff);
      result.max_output_abs_diff = std::max(result.max_output_abs_diff, diffs.max_output_abs_diff);
      next_mean_values.push_back(diffs.mean_next_state_abs_diff);
      next_rms_values.push_back(diffs.rms_next_state_abs_diff);
      output_mean_values.push_back(diffs.mean_output_abs_diff);
      output_rms_values.push_back(diffs.rms_output_abs_diff);
    }
    result.mean_next_state_abs_diff = MeanOf(next_mean_values);
    result.rms_next_state_abs_diff = MeanOf(next_rms_values);
    result.stddev_mean_next_state_abs_diff = StddevOf(next_mean_values);
    result.stddev_rms_next_state_abs_diff = StddevOf(next_rms_values);
    result.mean_output_abs_diff = MeanOf(output_mean_values);
    result.rms_output_abs_diff = MeanOf(output_rms_values);
    result.stddev_mean_output_abs_diff = StddevOf(output_mean_values);
    result.stddev_rms_output_abs_diff = StddevOf(output_rms_values);
  }
  return result;
}

std::string ResultsToJson(const EnvironmentInfo& env, const std::vector<BenchmarkResult>& results) {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"environment\": {\n";
  oss << "    \"device_name\": \"" << EscapeJson(env.device_name) << "\",\n";
  oss << "    \"device_index\": " << env.device_index << ",\n";
  oss << "    \"runtime_version\": " << env.runtime_version << ",\n";
  oss << "    \"driver_version\": " << env.driver_version << ",\n";
  oss << "    \"compute_capability\": \"" << env.compute_capability_major << "." << env.compute_capability_minor << "\",\n";
  oss << "    \"stochastic_rounding_seed\": " << env.stochastic_rounding_seed << ",\n";
  oss << "    \"stochastic_rounding_philox_rounds\": " << env.stochastic_rounding_philox_rounds << ",\n";
  oss << "    \"dt_gate_threshold\": ";
  if (std::isinf(env.dt_gate_threshold)) {
    oss << "null";
  } else {
    oss << std::fixed << std::setprecision(6) << env.dt_gate_threshold;
  }
  oss << ",\n";
  oss << "    \"total_global_mem_bytes\": " << env.total_global_mem_bytes << ",\n";
  oss << "    \"stochastic_rounding_sweep_seeds\": [";
  for (std::size_t i = 0; i < env.stochastic_rounding_sweep_seeds.size(); ++i) {
    oss << env.stochastic_rounding_sweep_seeds[i];
    if (i + 1 != env.stochastic_rounding_sweep_seeds.size()) {
      oss << ", ";
    }
  }
  oss << "],\n";
  oss << "    \"fixture_chain_steps\": [";
  for (std::size_t i = 0; i < env.fixture_chain_steps.size(); ++i) {
    oss << env.fixture_chain_steps[i];
    if (i + 1 != env.fixture_chain_steps.size()) {
      oss << ", ";
    }
  }
  oss << "],\n";
  oss << "    \"fixture_trace_steps\": [";
  for (std::size_t i = 0; i < env.fixture_trace_steps.size(); ++i) {
    oss << env.fixture_trace_steps[i];
    if (i + 1 != env.fixture_trace_steps.size()) {
      oss << ", ";
    }
  }
  oss << "],\n";
  oss << "    \"fixture_trace_phase_end_steps\": [";
  for (std::size_t i = 0; i < env.fixture_trace_phase_end_steps.size(); ++i) {
    oss << env.fixture_trace_phase_end_steps[i];
    if (i + 1 != env.fixture_trace_phase_end_steps.size()) {
      oss << ", ";
    }
  }
  oss << "]\n";
  oss << "  },\n";
  oss << "  \"results\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    oss << "    {\n";
    oss << "      \"format\": \"" << EscapeJson(r.format) << "\",\n";
    oss << "      \"operation\": \"" << EscapeJson(r.operation) << "\",\n";
    if (!r.fixture_name.empty()) {
      oss << "      \"fixture_name\": \"" << EscapeJson(r.fixture_name) << "\",\n";
    }
    if (r.fixture_steps != 1 || r.operation == "fixture_chain") {
      oss << "      \"fixture_steps\": " << r.fixture_steps << ",\n";
    }
    if (!r.trace_phase_name.empty()) {
      oss << "      \"trace_phase_name\": \"" << EscapeJson(r.trace_phase_name) << "\",\n";
      oss << "      \"trace_phase_kind\": \"" << EscapeJson(r.trace_phase_kind) << "\",\n";
    }
    if (r.has_oracle_diffs) {
      oss << "      \"oracle_seed_count\": " << r.oracle_seed_count << ",\n";
    }
    oss << "      \"active_requests\": " << r.active_requests << ",\n";
    oss << "      \"logical_elements_per_request\": " << r.logical_elements_per_request << ",\n";
    oss << "      \"per_request_state_bytes\": " << r.per_request_state_bytes << ",\n";
    oss << "      \"per_request_scale_bytes\": " << r.per_request_scale_bytes << ",\n";
    oss << "      \"total_state_bytes\": " << r.total_state_bytes << ",\n";
    oss << "      \"total_scale_bytes\": " << r.total_scale_bytes << ",\n";
    oss << "      \"approximate_traffic_bytes\": " << r.approximate_traffic_bytes << ",\n";
    oss << "      \"warmup_iterations\": " << r.warmup_iterations << ",\n";
    oss << "      \"hot_iterations\": " << r.hot_iterations << ",\n";
    oss << std::fixed << std::setprecision(6);
    oss << "      \"hot_mean_ms\": " << r.hot_mean_ms << ",\n";
    oss << "      \"hot_min_ms\": " << r.hot_min_ms << ",\n";
    oss << "      \"hot_max_ms\": " << r.hot_max_ms << ",\n";
    oss << "      \"effective_gbps\": " << r.effective_gbps;
    if (r.has_oracle_diffs) {
      oss << std::fixed << std::setprecision(9);
      oss << ",\n";
      oss << "      \"max_next_state_abs_diff\": " << r.max_next_state_abs_diff << ",\n";
      oss << "      \"mean_next_state_abs_diff\": " << r.mean_next_state_abs_diff << ",\n";
      oss << "      \"rms_next_state_abs_diff\": " << r.rms_next_state_abs_diff << ",\n";
      oss << "      \"stddev_mean_next_state_abs_diff\": " << r.stddev_mean_next_state_abs_diff << ",\n";
      oss << "      \"stddev_rms_next_state_abs_diff\": " << r.stddev_rms_next_state_abs_diff << ",\n";
      oss << "      \"max_output_abs_diff\": " << r.max_output_abs_diff << ",\n";
      oss << "      \"mean_output_abs_diff\": " << r.mean_output_abs_diff << ",\n";
      oss << "      \"rms_output_abs_diff\": " << r.rms_output_abs_diff << ",\n";
      oss << "      \"stddev_mean_output_abs_diff\": " << r.stddev_mean_output_abs_diff << ",\n";
      oss << "      \"stddev_rms_output_abs_diff\": " << r.stddev_rms_output_abs_diff << "\n";
    } else {
      oss << "\n";
    }
    oss << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

bool WriteTextFile(const std::filesystem::path& path, std::string_view contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return false;
  }
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  return output.good();
}

void PrintSummary(const BenchmarkResult& result) {
  std::cout << result.format
            << " op=" << result.operation;
  if (result.operation == "fixture_chain" || result.operation == "fixture_trace") {
    std::cout << " steps=" << result.fixture_steps;
  }
  if (!result.trace_phase_name.empty()) {
    std::cout << " phase=" << result.trace_phase_name;
  }
  std::cout
            << " requests=" << result.active_requests
            << " state_mib=" << std::fixed << std::setprecision(1)
            << static_cast<double>(result.total_state_bytes) / static_cast<double>(MiB(1))
            << " hot_mean_ms=" << std::setprecision(3) << result.hot_mean_ms
            << " gbps=" << std::setprecision(2) << result.effective_gbps;
  if (result.has_oracle_diffs) {
    std::cout << " out_max_abs_diff=" << std::setprecision(6) << result.max_output_abs_diff
              << " out_mean_abs_diff=" << result.mean_output_abs_diff
              << " state_max_abs_diff=" << result.max_next_state_abs_diff;
    if (result.oracle_seed_count > 1) {
      std::cout << " seed_count=" << result.oracle_seed_count
                << " out_mean_stddev=" << result.stddev_mean_output_abs_diff;
    }
  }
  std::cout
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 2;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::cerr << "error: no CUDA device available\n";
    return 1;
  }

  const EnvironmentInfo env = CollectEnvironmentInfo();
  EnvironmentInfo env_with_options = env;
  env_with_options.stochastic_rounding_seed = options.stochastic_rounding_seed;
  env_with_options.dt_gate_threshold = options.dt_gate_threshold;
  env_with_options.stochastic_rounding_sweep_seeds = options.stochastic_rounding_sweep_seeds;
  env_with_options.fixture_chain_steps = options.fixture_chain_steps;
  env_with_options.fixture_trace_steps = options.fixture_trace_steps;
  std::optional<FixtureOracleData> fixture;
  std::optional<FixtureTraceData> trace_fixture;
  const bool needs_fixture =
      std::find(options.operations.begin(), options.operations.end(), OperationKind::kFixtureUpdate) != options.operations.end() ||
      std::find(options.operations.begin(), options.operations.end(), OperationKind::kFixtureChain) != options.operations.end();
  if (needs_fixture) {
    try {
      fixture = LoadFixtureOracleData(options.fixture_root);
    } catch (const std::exception& error) {
      std::cerr << "error: failed to load fixture data: " << error.what() << "\n";
      return 1;
    }
  }
  if (std::find(options.operations.begin(), options.operations.end(), OperationKind::kFixtureTrace) !=
      options.operations.end()) {
    try {
      trace_fixture = LoadFixtureTraceData(options.trace_fixture_root);
      for (const TracePhase& phase : trace_fixture->phases) {
        env_with_options.fixture_trace_phase_end_steps.push_back(phase.end_step);
      }
      env_with_options.fixture_trace_steps =
          MergeTraceCheckpointSteps(*trace_fixture, env_with_options.fixture_trace_steps);
    } catch (const std::exception& error) {
      std::cerr << "error: failed to load trace fixture data: " << error.what() << "\n";
      return 1;
    }
  }
  std::map<std::size_t, FixtureReferenceChain> fixture_chain_references;
  if (std::find(options.operations.begin(), options.operations.end(), OperationKind::kFixtureChain) !=
      options.operations.end()) {
    try {
      fixture_chain_references = BuildFixtureReferenceChains(*fixture, options.fixture_chain_steps);
    } catch (const std::exception& error) {
      std::cerr << "error: failed to build fixture chain references: " << error.what() << "\n";
      return 1;
    }
  }
  std::map<std::size_t, FixtureReferenceChain> fixture_trace_references;
  if (std::find(options.operations.begin(), options.operations.end(), OperationKind::kFixtureTrace) !=
      options.operations.end()) {
    try {
      fixture_trace_references = BuildFixtureTraceReferences(*trace_fixture, options.fixture_trace_steps);
    } catch (const std::exception& error) {
      std::cerr << "error: failed to build fixture trace references: " << error.what() << "\n";
      return 1;
    }
  }
  const std::vector<CacheFormat> formats = {
      CacheFormat::kFp32,
      CacheFormat::kFp16,
      CacheFormat::kFp16Sr,
      CacheFormat::kInt8Group};
  std::vector<BenchmarkResult> results;

  for (const CacheFormat format : formats) {
    for (const OperationKind operation : options.operations) {
      if (!SupportsBenchmarkCombination(format, operation)) {
        continue;
      }
      for (const std::size_t active_requests : options.concurrency_values) {
        const std::vector<std::size_t> step_counts =
            operation == OperationKind::kFixtureChain
                ? options.fixture_chain_steps
                : (operation == OperationKind::kFixtureTrace
                       ? env_with_options.fixture_trace_steps
                       : std::vector<std::size_t>{1});
        for (const std::size_t fixture_steps : step_counts) {
          const auto result = RunBenchmark(
              format,
              operation,
              active_requests,
              options,
              fixture.has_value() ? &(*fixture) : nullptr,
              trace_fixture.has_value() ? &(*trace_fixture) : nullptr,
              operation == OperationKind::kFixtureTrace
                  ? (fixture_trace_references.empty() ? nullptr : &fixture_trace_references)
                  : (fixture_chain_references.empty() ? nullptr : &fixture_chain_references),
              fixture_steps);
          if (!result.has_value()) {
            std::cerr << "error: benchmark failed for format=" << ToString(format)
                      << " operation=" << ToString(operation)
                      << " requests=" << active_requests;
            if (operation == OperationKind::kFixtureChain) {
              std::cerr << " steps=" << fixture_steps;
            }
            std::cerr << "\n";
            return 1;
          }
          PrintSummary(*result);
          results.push_back(*result);
        }
      }
    }
  }

  const std::string json = ResultsToJson(env_with_options, results);
  if (options.json_output_path.has_value() && !WriteTextFile(*options.json_output_path, json)) {
    std::cerr << "error: failed to write JSON output\n";
    return 1;
  }

  std::cout << "\n" << json;
  return 0;
}
