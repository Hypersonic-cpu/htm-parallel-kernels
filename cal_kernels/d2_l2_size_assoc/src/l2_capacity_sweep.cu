#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

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

// Use 64B so nominal working_set_bytes matches L2 line footprint much better.
constexpr std::size_t kNodeBytes = 64;
constexpr std::size_t kNodeQWords = kNodeBytes / sizeof(U64);

#ifndef GPGPU_SIM
constexpr int kDefaultWarmupPasses = 1;
constexpr int kDefaultMeasurePasses = 1;
#else
constexpr int kDefaultWarmupPasses = 1;
constexpr int kDefaultMeasurePasses = 1;
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

// Exactly one dependent global load.
// Load address is ptr itself: no shift/add/sub in the chase path.
__device__ __forceinline__ U64 ptr_chase_cg(U64 ptr) {
  U64 value;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
  return value;
}

void build_random_ring(std::vector<U64> *host, U64 dev_base, std::uint32_t seed) {
  const std::size_t num_nodes = host->size() / kNodeQWords;
  std::fill(host->begin(), host->end(), 0ULL);

  std::vector<U32> perm(num_nodes);
  for (std::size_t i = 0; i < num_nodes; ++i) {
    perm[i] = static_cast<U32>(i);
  }

  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);

  for (std::size_t i = 0; i < num_nodes; ++i) {
    const U32 current = perm[i];
    const U32 next = perm[(i + 1) % num_nodes];

    // Store the absolute device address of the next node at offset 0.
    (*host)[static_cast<std::size_t>(current) * kNodeQWords] =
        dev_base + static_cast<U64>(next) * kNodeBytes;
  }
}

bool alloc_and_copy(std::vector<U64> *host, std::uint32_t seed, U64 **dev_ptr) {
  cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(dev_ptr), host->size() * sizeof(U64));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
    return false;
  }

  const U64 dev_base = reinterpret_cast<U64>(*dev_ptr);
  build_random_ring(host, dev_base, seed);

  status = cudaMemcpy(*dev_ptr, host->data(), host->size() * sizeof(U64),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(status));
    cudaFree(*dev_ptr);
    *dev_ptr = nullptr;
    return false;
  }
  return true;
}

__global__ void probe_kernel(U64 start_ptr, U64 base_ptr, U32 num_nodes,
                             int warmup_passes, int measure_passes,
                             U64 *cycles_out, U32 *final_node_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  U64 current = start_ptr;

  const U64 warmup_steps =
      static_cast<U64>(num_nodes) * static_cast<U64>(warmup_passes);
  #pragma unroll 128
  for (U64 i = 0; i < warmup_steps; ++i) {
    current = ptr_chase_cg(current);
  }

#ifndef GPGPU_SIM
  const U64 t0 = clock64();
#endif

  const U64 measure_steps =
      static_cast<U64>(num_nodes) * static_cast<U64>(measure_passes);
  #pragma unroll 256
  for (U64 i = 0; i < measure_steps; ++i) {
    current = ptr_chase_cg(current);
  }

#ifndef GPGPU_SIM
  const U64 t1 = clock64();
  cycles_out[0] = t1 - t0;
#else
  cycles_out[0] = 0;
#endif

  // Convert back only after timing.
  final_node_out[0] = static_cast<U32>((current - base_ptr) / kNodeBytes);
}

}  // namespace

int main(int argc, char **argv) {
  int warmup_passes = kDefaultWarmupPasses;
  int measure_passes = kDefaultMeasurePasses;
  if (argc > 1) {
    warmup_passes = std::atoi(argv[1]);
  }
  if (argc > 2) {
    measure_passes = std::atoi(argv[2]);
  }

  // Keep the original sweep mostly unchanged.
  // With 64B nodes, these values now map much better to true L2 footprint.
#ifndef GPGPU_SIM
#if defined(HW_H100)
  const std::size_t sweep_bytes[] = {
      4ULL * 1024 * 1024,
      8ULL * 1024 * 1024,
      16ULL * 1024 * 1024,
      32ULL * 1024 * 1024,
      48ULL * 1024 * 1024,
      50ULL * 1024 * 1024,
      64ULL * 1024 * 1024,
      96ULL * 1024 * 1024,
      128ULL * 1024 * 1024,
  };
#else
  const std::size_t sweep_bytes[] = {
      256ULL * 1024,
      512ULL * 1024,
      1ULL * 1024 * 1024,
      2ULL * 1024 * 1024,
      4ULL * 1024 * 1024,
      6ULL * 1024 * 1024,
      8ULL * 1024 * 1024,
      12ULL * 1024 * 1024,
  };
#endif
#else
  const std::size_t sweep_bytes[] = {
      256ULL * 1024,
      6ULL * 1024 * 1024,
      8ULL * 1024 * 1024,
  };
#endif

  U64 *dev_cycles = nullptr;
  U32 *dev_final = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_cycles, sizeof(U64)));
  CHECK_CUDA(cudaMalloc(&dev_final, sizeof(U32)));

  std::printf("working_set_kB,nodes,warmup_passes,measure_passes,passes,"
              "measured_accesses,min_cycles_per_access,median_cycles_per_access,"
              "avg_cycles_per_access,max_cycles_per_access,min_total_cycles,"
              "median_total_cycles,avg_total_cycles,max_total_cycles,final_node\n");

  for (std::size_t working_set_bytes : sweep_bytes) {
    const std::size_t num_nodes = bytes_to_nodes(working_set_bytes);

    // One 64B node: [next_ptr][padding...]
    std::vector<U64> host(num_nodes * kNodeQWords);

    U64 *dev_next = nullptr;
    if (!alloc_and_copy(&host, static_cast<U32>(0x600Du + working_set_bytes),
                        &dev_next)) {
      return EXIT_FAILURE;
    }

    const U64 base_ptr = reinterpret_cast<U64>(dev_next);

    U32 final_node = 0;
    std::vector<U64> cycle_samples;
    cycle_samples.reserve(cal_kernels::kNativePasses);
    for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
      probe_kernel<<<1, 1>>>(base_ptr, base_ptr, static_cast<U32>(num_nodes),
                             warmup_passes, measure_passes, dev_cycles,
                             dev_final);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      U64 cycles = 0;
      CHECK_CUDA(
          cudaMemcpy(&cycles, dev_cycles, sizeof(U64), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(&final_node, dev_final, sizeof(U32),
                            cudaMemcpyDeviceToHost));
      cycle_samples.push_back(cycles);
    }

    const U64 measured_accesses =
        static_cast<U64>(num_nodes) * static_cast<U64>(measure_passes);

#ifdef GPGPU_SIM
    std::printf("%zu,\t%zu,\t%d,\t%d,\t%d,\t%" PRIu64 ",\tN/A,\tN/A,\tN/A,\tN/A,\t"
                "N/A,\tN/A,\tN/A,\tN/A,\t%u\n",
                working_set_bytes >> 10, num_nodes, warmup_passes, measure_passes,
                cal_kernels::kNativePasses, measured_accesses, final_node);
#else
    const auto total_stats = cal_kernels::summarize_numeric(cycle_samples);
    const auto per_access_stats = cal_kernels::summarize_samples(
        cal_kernels::normalize_samples(cycle_samples,
                                       static_cast<double>(measured_accesses)));
    std::printf("%zu,\t%zu,\t%d,\t%d,\t%d,\t%" PRIu64 ",\t%.4f,\t%.4f,\t%.4f,\t%.4f,\t"
                "%.0f,\t%.0f,\t%.4f,\t%.0f,\t%u\n",
                working_set_bytes >> 10, num_nodes, warmup_passes, measure_passes,
                cal_kernels::kNativePasses, measured_accesses,
                per_access_stats.min, per_access_stats.median,
                per_access_stats.avg, per_access_stats.max, total_stats.min,
                total_stats.median, total_stats.avg, total_stats.max,
                final_node);
#endif

    CHECK_CUDA(cudaFree(dev_next));
  }

#ifdef GPGPU_SIM
  std::printf(
      "GPGPU-Sim note: this benchmark now measures one full warmed pass per "
      "working set. Use your L2 counters together with the capacity sweep.\n");
#endif

  CHECK_CUDA(cudaFree(dev_final));
  CHECK_CUDA(cudaFree(dev_cycles));
  return EXIT_SUCCESS;
}
