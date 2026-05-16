#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "l2_flush.h"
#include "native_stats.h"

using U32 = std::uint32_t;
using U64 = std::uint64_t;

#if defined(HW_H100)
#pragma message("compile-time info: Hopper H100")
#elif defined(HW_V100)
#pragma message("compile-time info: Volta V100")
#else
#error Unsupported hardware target. Define HW_V100 or HW_H100.
#endif

namespace {

constexpr std::size_t kNodeBytes = 128;
constexpr std::size_t kNodeWords = kNodeBytes / sizeof(U32);

struct CaseConfig {
  const char *name;
  std::size_t working_set_bytes;
};

#if defined(HW_H100)
const CaseConfig kCases[] = {
    {"lt_l2", 32ULL * 1024 * 1024},
    {"approx_l2", 50ULL * 1024 * 1024},
    {"ge_l2", 64ULL * 1024 * 1024},
};
#else
const CaseConfig kCases[] = {
    {"lt_l2", 4ULL * 1024 * 1024},
    {"approx_l2", 6ULL * 1024 * 1024},
    {"ge_l2", 16ULL * 1024 * 1024},
};
#endif

constexpr int kDefaultWarmupPasses = 4;
#if defined(GPGPU_SIM) || defined(PERF_RUN)
constexpr int kDefaultMeasureIters = 1;
#else
constexpr int kDefaultMeasureIters = 10;
#endif

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status__ = (call);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status__));                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

std::size_t bytes_to_nodes(std::size_t bytes) {
  std::size_t rounded = ((bytes + kNodeBytes - 1) / kNodeBytes) * kNodeBytes;
  return rounded / kNodeBytes;
}

__device__ __forceinline__ U32 ld_cg_u32(const U32 *ptr) {
  U32 value;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
  return value;
}

void build_random_ring(std::vector<U32> *host, std::uint32_t seed) {
  const std::size_t num_nodes = host->size() / kNodeWords;
  std::fill(host->begin(), host->end(), 0u);
  std::vector<U32> perm(num_nodes);
  for (std::size_t i = 0; i < num_nodes; ++i) {
    perm[i] = static_cast<U32>(i);
  }
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const U32 current = perm[i];
    const U32 next = perm[(i + 1) % num_nodes];
    (*host)[static_cast<std::size_t>(current) * kNodeWords] = next;
  }
}

bool alloc_and_copy(const std::vector<U32> &host, U32 **dev_ptr) {
  cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(dev_ptr), host.size() * sizeof(U32));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
    return false;
  }
  status = cudaMemcpy(*dev_ptr, host.data(), host.size() * sizeof(U32),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(status));
    cudaFree(*dev_ptr);
    *dev_ptr = nullptr;
    return false;
  }
  return true;
}

__global__ void warm_range_kernel(const U32 *next, std::size_t num_nodes,
                                  U32 *sink) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  U32 accum = 0;
#pragma unroll 128
  for (std::size_t node = tid; node < num_nodes; node += stride) {
    accum ^= ld_cg_u32(next + node * kNodeWords);
  }
  sink[tid] = accum;
}

__global__ void probe_kernel(const U32 *next, U32 start_node,
                             std::size_t num_nodes, int warmup_passes,
                             int measure_iters, U64 *cycles_out,
                             U32 *final_node_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  U32 current = start_node;
  const U64 warmup_steps =
      static_cast<U64>(num_nodes) * static_cast<U64>(warmup_passes);
  for (U64 i = 0; i < warmup_steps; ++i) {
    current = ld_cg_u32(next + static_cast<std::size_t>(current) * kNodeWords);
  }

#ifndef GPGPU_SIM
  const U64 t0 = clock64();
#endif
#pragma unroll 128
  for (int i = 0; i < measure_iters; ++i) {
    current = ld_cg_u32(next + static_cast<std::size_t>(current) * kNodeWords);
  }
#ifndef GPGPU_SIM
  const U64 t1 = clock64();
  cycles_out[0] = t1 - t0;
#else
  cycles_out[0] = 0;
#endif
  final_node_out[0] = current;
}

}  // namespace

int main(int argc, char **argv) {
  bool use_custom_size = false;
  std::size_t custom_working_set_bytes = 0;
  int warmup_passes = kDefaultWarmupPasses;
  int measure_iters = kDefaultMeasureIters;

  if (argc > 1) {
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(argv[1], &end, 0);
    if (argv[1][0] == '\0' || end == nullptr || *end != '\0' || parsed == 0) {
      std::fprintf(stderr, "Invalid working_set_bytes: %s\n", argv[1]);
      return EXIT_FAILURE;
    }
    use_custom_size = true;
    custom_working_set_bytes = static_cast<std::size_t>(parsed);
  }
  if (argc > 2) {
    warmup_passes = std::atoi(argv[2]);
  }
  if (argc > 3) {
    measure_iters = std::atoi(argv[3]);
  }
  if (warmup_passes < 0) {
    std::fprintf(stderr, "Invalid warmup_passes: %d\n", warmup_passes);
    return EXIT_FAILURE;
  }
  if (measure_iters <= 0) {
    std::fprintf(stderr, "Invalid measure_iters: %d\n", measure_iters);
    return EXIT_FAILURE;
  }

  std::printf("case,working_set_bytes,range_warm_nodes,chase_warm_steps,"
              "measure_iters,passes,min_cycles_per_access,"
              "median_cycles_per_access,avg_cycles_per_access,"
              "max_cycles_per_access,min_total_cycles,median_total_cycles,"
              "avg_total_cycles,max_total_cycles,final_node\n");

  const int case_count =
      use_custom_size ? 1 : static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
  for (int case_idx = 0; case_idx < case_count; ++case_idx) {
    const char *case_name = use_custom_size ? "custom" : kCases[case_idx].name;
    std::size_t working_set_bytes =
        use_custom_size ? custom_working_set_bytes
                        : kCases[case_idx].working_set_bytes;
    std::size_t num_nodes = bytes_to_nodes(working_set_bytes);
    if (num_nodes < 1024) {
      num_nodes = 1024;
    }
    working_set_bytes = num_nodes * kNodeBytes;

    std::vector<U32> host(num_nodes * kNodeWords);
    build_random_ring(&host, 0x7EEF1234u);

    U32 *dev_next = nullptr;
    if (!alloc_and_copy(host, &dev_next)) {
      return EXIT_FAILURE;
    }
    U64 *dev_cycles = nullptr;
    U32 *dev_final = nullptr;
    CHECK_CUDA(cudaMalloc(&dev_cycles, sizeof(U64)));
    CHECK_CUDA(cudaMalloc(&dev_final, sizeof(U32)));

#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
    float *dev_l2_flush = nullptr;
    const std::size_t l2_flush_elements =
        cal_kernels::kColdL2FlushBytes / sizeof(float);
    CHECK_CUDA(cudaMalloc(&dev_l2_flush, l2_flush_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(dev_l2_flush, 0,
                          l2_flush_elements * sizeof(float)));
#endif

    U32 *dev_warm_sink = nullptr;
    if (warmup_passes > 0) {
      const int warm_threads = 256;
#ifndef GPGPU_SIM
      const int warm_blocks = 256;
#else
      const int warm_blocks = 4;
#endif
      CHECK_CUDA(cudaMalloc(&dev_warm_sink,
                            static_cast<std::size_t>(warm_blocks) *
                                warm_threads * sizeof(U32)));
      warm_range_kernel<<<warm_blocks, warm_threads>>>(dev_next, num_nodes,
                                                       dev_warm_sink);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    U32 final_node = 0;
    std::vector<U64> cycle_samples;
    cycle_samples.reserve(cal_kernels::kNativePasses);
    for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
      CHECK_CUDA(
          cal_kernels::run_cold_l2_flush(dev_l2_flush, l2_flush_elements));
#endif
      probe_kernel<<<1, 1>>>(dev_next, 0, num_nodes, warmup_passes, measure_iters,
                             dev_cycles, dev_final);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      U64 cycles = 0;
      CHECK_CUDA(
          cudaMemcpy(&cycles, dev_cycles, sizeof(U64), cudaMemcpyDeviceToHost));
      CHECK_CUDA(
          cudaMemcpy(&final_node, dev_final, sizeof(U32), cudaMemcpyDeviceToHost));
      cycle_samples.push_back(cycles);
    }

    const U64 warmup_steps =
        static_cast<U64>(num_nodes) * static_cast<U64>(warmup_passes);
#ifdef GPGPU_SIM
    std::printf("%s,%zu,%zu,%" PRIu64 ",%d,%d,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,%u\n",
                case_name, working_set_bytes, num_nodes, warmup_steps,
                measure_iters, cal_kernels::kNativePasses, final_node);
#else
    const auto total_stats = cal_kernels::summarize_numeric(cycle_samples);
    const auto per_access_stats = cal_kernels::summarize_samples(
        cal_kernels::normalize_samples(cycle_samples,
                                       static_cast<double>(measure_iters)));
    std::printf("%s,%zu,%zu,%" PRIu64 ",%d,%d,%.4f,%.4f,%.4f,%.4f,%.0f,%.0f,%.4f,%.0f,%u\n",
                case_name, working_set_bytes, num_nodes, warmup_steps,
                measure_iters, cal_kernels::kNativePasses, per_access_stats.min,
                per_access_stats.median, per_access_stats.avg,
                per_access_stats.max, total_stats.min, total_stats.median,
                total_stats.avg, total_stats.max, final_node);
#endif

    if (dev_warm_sink != nullptr) {
      CHECK_CUDA(cudaFree(dev_warm_sink));
    }
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
    CHECK_CUDA(cudaFree(dev_l2_flush));
#endif
    CHECK_CUDA(cudaFree(dev_final));
    CHECK_CUDA(cudaFree(dev_cycles));
    CHECK_CUDA(cudaFree(dev_next));
  }

  return EXIT_SUCCESS;
}
