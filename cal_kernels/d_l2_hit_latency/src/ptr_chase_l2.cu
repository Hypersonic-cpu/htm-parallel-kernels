#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "native_stats.h"

#if defined(HW_H100)
#pragma message("compile-time info: Hopper H100")
#elif defined(HW_V100)
#pragma message("compile-time info: Volta V100")
#else
#error Unsupported hardware target. Define HW_V100 or HW_H100.
#endif

namespace {

#ifndef GPGPU_SIM
constexpr int kMinBytes = 1U << 20;
#if defined(HW_H100)
// GH100 SXM5 exposes roughly 50 MiB L2; sweep well beyond it.
constexpr int kDefaultMaxBytes = 128U << 20;
#else
// GV100 exposes a 6 MiB unified L2. Sweep across and beyond that point.
constexpr int kDefaultMaxBytes = 32U << 20;
#endif
constexpr std::uint32_t kMeasureIters = 128U;
constexpr std::uint32_t kWarmPasses = 2U;  // warm the whole ring twice
#else
constexpr int kMinBytes = 4 * 1024 * 1024;
constexpr int kDefaultMaxBytes = 8 * 1024 * 1024;
constexpr std::uint32_t kMeasureIters = 64U;
constexpr std::uint32_t kWarmPasses = 2U;
#endif

constexpr std::uint32_t kUnrollFactor = 256U;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status__ = (call);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status__));                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

__device__ __forceinline__ std::uint64_t ld_global_cg_u64(std::uint64_t addr) {
  std::uint64_t v;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(addr));
  return v;
}

// Untimed warm-up path: runtime trip count is fine here.
// The warm phase is outside the clock64() window.
__device__ __forceinline__ std::uint64_t warm_steps_runtime(std::uint64_t ptr,
                                                            std::uint64_t steps) {
  while (steps >= kUnrollFactor) {
#pragma unroll 256
    for (std::uint32_t i = 0; i < kUnrollFactor; ++i) {
      ptr = ld_global_cg_u64(ptr);
    }
    steps -= kUnrollFactor;
  }

  for (std::uint64_t i = 0; i < steps; ++i) {
    ptr = ld_global_cg_u64(ptr);
  }
  return ptr;
}

template <std::uint32_t Tail>
__device__ __forceinline__ std::uint64_t
chase_tail(std::uint64_t ptr, std::integral_constant<bool, false>) {
#pragma unroll 256
  for (std::uint32_t i = 0; i < Tail; ++i) {
    ptr = ld_global_cg_u64(ptr);
  }
  return ptr;
}

template <std::uint32_t Tail>
__device__ __forceinline__ std::uint64_t
chase_tail(std::uint64_t ptr, std::integral_constant<bool, true>) {
  return ptr;
}

// Timed path: compile-time constant number of dependent loads.
template <std::uint32_t Steps>
__device__ __forceinline__ std::uint64_t chase_steps(std::uint64_t ptr) {
  constexpr std::uint32_t kFullChunks = Steps / kUnrollFactor;
  constexpr std::uint32_t kTail = Steps % kUnrollFactor;

#pragma unroll
  for (std::uint32_t chunk = 0; chunk < kFullChunks; ++chunk) {
#pragma unroll 256
    for (std::uint32_t i = 0; i < kUnrollFactor; ++i) {
      ptr = ld_global_cg_u64(ptr);
    }
  }

  return chase_tail<kTail>(ptr, std::integral_constant<bool, kTail == 0>{});
}

template <std::uint32_t Measure>
__global__ void ptr_chase_l2(std::uint64_t start_ptr, std::uint64_t warm_steps,
                             std::uint64_t *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  std::uint64_t ptr = start_ptr;

  // Warm the entire ring (possibly multiple passes) before timing.
  ptr = warm_steps_runtime(ptr, warm_steps);

#ifndef GPGPU_SIM
  std::uint64_t t0 = clock64();
#endif

  ptr = chase_steps<Measure>(ptr);

#ifndef GPGPU_SIM
  std::uint64_t t1 = clock64();
  out[0] = t1 - t0;
#else
  out[0] = 0;
#endif
  out[1] = ptr;  // keep the chain live
}

void build_random_cycle(std::vector<std::uint64_t> &host_arr,
                        uintptr_t device_base, std::uint64_t *start_ptr) {
  const std::uint32_t n = static_cast<std::uint32_t>(host_arr.size());
  std::vector<std::uint32_t> perm(n);
  std::iota(perm.begin(), perm.end(), 0);

  std::mt19937 rng(123456789u);
  std::shuffle(perm.begin(), perm.end(), rng);

  for (std::uint32_t i = 0; i + 1 < n; ++i) {
    host_arr[perm[i]] = static_cast<std::uint64_t>(
        device_base +
        static_cast<uintptr_t>(perm[i + 1]) * sizeof(std::uint64_t));
  }
  host_arr[perm[n - 1]] = static_cast<std::uint64_t>(
      device_base + static_cast<uintptr_t>(perm[0]) * sizeof(std::uint64_t));
  *start_ptr = static_cast<std::uint64_t>(
      device_base + static_cast<uintptr_t>(perm[0]) * sizeof(std::uint64_t));
}

bool parse_int_arg(const char *name, const char *value, int *out) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (value[0] == '\0' || end == nullptr || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "Invalid value for %s: %s\n", name, value);
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

bool parse_u32_arg(const char *name, const char *value, std::uint32_t *out) {
  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (value[0] == '\0' || end == nullptr || *end != '\0' ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    std::fprintf(stderr, "Invalid value for %s: %s\n", name, value);
    return false;
  }
  *out = static_cast<std::uint32_t>(parsed);
  return true;
}

bool parse_args(int argc, char **argv, int *min_bytes, int *max_bytes,
                std::uint32_t *warm_passes, bool *show_help) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (std::strcmp(arg, "--min-bytes") == 0) {
      if (i + 1 >= argc || !parse_int_arg(arg, argv[++i], min_bytes)) {
        return false;
      }
      continue;
    }
    if (std::strcmp(arg, "--max-bytes") == 0) {
      if (i + 1 >= argc || !parse_int_arg(arg, argv[++i], max_bytes)) {
        return false;
      }
      continue;
    }
    if (std::strcmp(arg, "--warm-passes") == 0) {
      if (i + 1 >= argc || !parse_u32_arg(arg, argv[++i], warm_passes) ||
          *warm_passes == 0) {
        return false;
      }
      continue;
    }
    if (std::strcmp(arg, "--help") == 0) {
      std::printf(
          "Usage: %s [--min-bytes N] [--max-bytes N] [--warm-passes N]\n",
          argv[0]);
      *show_help = true;
      return false;
    }
    std::fprintf(stderr, "Unknown argument: %s\n", arg);
    return false;
  }

  if (*min_bytes <= 0 || *max_bytes < *min_bytes || (*min_bytes % 8) != 0 ||
      (*max_bytes % 8) != 0 || *warm_passes == 0) {
    std::fprintf(stderr,
                 "Invalid sweep config: min_bytes=%d max_bytes=%d "
                 "warm_passes=%u\n",
                 *min_bytes, *max_bytes, *warm_passes);
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  int min_bytes = kMinBytes;
  int max_bytes = kDefaultMaxBytes;
  std::uint32_t warm_passes = kWarmPasses;
  bool show_help = false;

  if (!parse_args(argc, argv, &min_bytes, &max_bytes, &warm_passes,
                  &show_help)) {
    return show_help ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  std::uint64_t *d_arr = nullptr;
  std::uint64_t *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_arr, max_bytes));
  CHECK_CUDA(cudaMalloc(&d_out, 2 * sizeof(std::uint64_t)));

  std::printf("working_set_kB,\twarm_steps,\tmeasure_iters,\tpasses,\t"
              "min_latency_cycles,\tmedian_latency_cycles,\tavg_latency_cycles,\t"
              "max_latency_cycles,\tmin_total_cycles,\tmedian_total_cycles,\t"
              "avg_total_cycles,\tmax_total_cycles\n");

  for (int ws_bytes = min_bytes; ws_bytes <= max_bytes; ws_bytes *= 2) {
    const std::uint32_t n =
        static_cast<std::uint32_t>(ws_bytes / sizeof(std::uint64_t));
    if (n < 2) {
      continue;
    }

    std::vector<std::uint64_t> host_arr(n);
    std::uint64_t start_ptr = 0;
    build_random_cycle(host_arr, reinterpret_cast<uintptr_t>(d_arr),
                       &start_ptr);
    CHECK_CUDA(
        cudaMemcpy(d_arr, host_arr.data(), ws_bytes, cudaMemcpyHostToDevice));

    const std::uint64_t warm_steps =
        static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(warm_passes);

    std::uint64_t result[2] = {0, 0};
    std::vector<std::uint64_t> cycle_samples;
    cycle_samples.reserve(cal_kernels::kNativePasses);

    for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
      ptr_chase_l2<kMeasureIters><<<1, 1>>>(start_ptr, warm_steps, d_out);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(
          cudaMemcpy(result, d_out, sizeof(result), cudaMemcpyDeviceToHost));
      cycle_samples.push_back(result[0]);
    }

#ifdef GPGPU_SIM
    std::printf("%d,\t\t%llu,\t\t%u,\t\t%d,\t\tN/A,\t\tN/A,\t\tN/A,\t\tN/A,\t\t"
                "N/A,\t\tN/A,\t\tN/A,\t\tN/A\n",
                ws_bytes / 1024U,
                static_cast<unsigned long long>(warm_steps), kMeasureIters,
                cal_kernels::kNativePasses);
#else
    const auto total_stats = cal_kernels::summarize_numeric(cycle_samples);
    const auto per_access_stats = cal_kernels::summarize_samples(
        cal_kernels::normalize_samples(cycle_samples,
                                       static_cast<double>(kMeasureIters)));
    std::printf("%d,\t\t%llu,\t\t%u,\t\t%d,\t\t%.4f,\t\t%.4f,\t\t%.4f,\t\t%.4f,\t\t"
                "%.0f,\t\t%.0f,\t\t%.4f,\t\t%.0f\n",
                ws_bytes / 1024U,
                static_cast<unsigned long long>(warm_steps), kMeasureIters,
                cal_kernels::kNativePasses, per_access_stats.min,
                per_access_stats.median, per_access_stats.avg,
                per_access_stats.max, total_stats.min, total_stats.median,
                total_stats.avg, total_stats.max);
#endif
  }

#ifdef GPGPU_SIM
  std::printf(
      "GPGPU-Sim note: use L2 miss-rate and gpu_sim_cycle from report; "
      "clock64 timing is disabled in sim mode.\n");
#endif

  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_arr));
  return EXIT_SUCCESS;
}
