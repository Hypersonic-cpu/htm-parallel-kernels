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

// Use 64B so the nominal hot-set size better matches actual L2 line footprint.
constexpr std::size_t kNodeBytes = 64;
constexpr std::size_t kNodeQWords = kNodeBytes / sizeof(U64);

#if defined(HW_H100) && !defined(GPGPU_SIM)
constexpr std::size_t kDefaultHotSetBytes = 16ULL * 1024 * 1024;
#else
constexpr std::size_t kDefaultHotSetBytes = 2ULL * 1024 * 1024;
#endif
constexpr int kDefaultMaxThreads = 128;

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
  const std::size_t rounded =
      ((bytes + kNodeBytes - 1) / kNodeBytes) * kNodeBytes;
  return rounded / kNodeBytes;
}

// Exactly one dependent global load.
// The load address is ptr itself: no shift/add/sub in the chase path.
__device__ __forceinline__ U64 ptr_chase_cg(U64 ptr) {
  U64 value;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
  return value;
}

void build_random_ring(std::vector<U64>* host, U64 dev_base,
                       std::size_t base_node, std::size_t num_nodes,
                       std::uint32_t seed) {
  std::vector<U32> perm(num_nodes);
  for (std::size_t i = 0; i < num_nodes; ++i) {
    perm[i] = static_cast<U32>(i);
  }

  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);

  for (std::size_t i = 0; i < num_nodes; ++i) {
    const U32 current = perm[i];
    const U32 next = perm[(i + 1) % num_nodes];

    (*host)[(base_node + static_cast<std::size_t>(current)) * kNodeQWords] =
        dev_base + static_cast<U64>(base_node + static_cast<std::size_t>(next)) *
                       kNodeBytes;
  }
}

void build_segmented_hot_ring(std::vector<U64>* host, U64 dev_base,
                              std::size_t total_nodes, int active_threads,
                              std::size_t nodes_per_thread) {
  std::fill(host->begin(), host->end(), 0ULL);

  for (int tid = 0; tid < active_threads; ++tid) {
    const std::size_t base =
        static_cast<std::size_t>(tid) * nodes_per_thread;
    build_random_ring(host, dev_base, base, nodes_per_thread,
                      static_cast<U32>(0xACE0u + tid));
  }
}

bool alloc_and_copy(std::vector<U64>* host, int active_threads,
                    std::size_t nodes_per_thread, U64** dev_ptr) {
  cudaError_t status =
      cudaMalloc(reinterpret_cast<void**>(dev_ptr), host->size() * sizeof(U64));
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
    return false;
  }

  const U64 dev_base = reinterpret_cast<U64>(*dev_ptr);
  build_segmented_hot_ring(host, dev_base,
                           (host->size() / kNodeQWords),
                           active_threads, nodes_per_thread);

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

__global__ void probe_kernel(U64 base_ptr, U32 nodes_per_thread,
                             int active_threads, int warmup_passes,
                             int measure_passes, U64* per_thread_cycles,
                             U32* per_thread_final) {
  const int tid = static_cast<int>(threadIdx.x);
  if (blockIdx.x != 0 || tid >= active_threads) {
    return;
  }

  U64 current = base_ptr +
                static_cast<U64>(tid) * static_cast<U64>(nodes_per_thread) *
                    kNodeBytes;

  const U64 warmup_steps =
      static_cast<U64>(nodes_per_thread) * static_cast<U64>(warmup_passes);
  for (U64 i = 0; i < warmup_steps; ++i) {
    current = ptr_chase_cg(current);
  }

  __syncthreads();

#ifndef GPGPU_SIM
  const U64 t0 = clock64();
#endif

  const U64 measure_steps =
      static_cast<U64>(nodes_per_thread) * static_cast<U64>(measure_passes);
  for (U64 i = 0; i < measure_steps; ++i) {
    current = ptr_chase_cg(current);
  }

#ifndef GPGPU_SIM
  const U64 t1 = clock64();
  per_thread_cycles[tid] = t1 - t0;
#else
  per_thread_cycles[tid] = 0;
#endif

  // Convert back only after timing.
  per_thread_final[tid] = static_cast<U32>((current - base_ptr) / kNodeBytes);
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t hot_set_bytes = kDefaultHotSetBytes;
  int warmup_passes = kDefaultWarmupPasses;
  int measure_passes = kDefaultMeasurePasses;
  int max_threads = kDefaultMaxThreads;

  if (argc > 1) {
    hot_set_bytes = std::strtoull(argv[1], nullptr, 0);
  }
  if (argc > 2) {
    warmup_passes = std::atoi(argv[2]);
  }
  if (argc > 3) {
    measure_passes = std::atoi(argv[3]);
  }
  if (argc > 4) {
    max_threads = std::atoi(argv[4]);
  }

  if (max_threads > 128) {
    std::fprintf(stderr,
                 "This benchmark currently supports up to 128 active threads "
                 "so they can be synchronized in one block.\n");
    return EXIT_FAILURE;
  }

  const int sweep_threads[] = {1, 2, 4, 8, 16, 32, 64, 128};

  std::printf("hot_set_kB,used_kB,active_threads,nodes_per_thread,"
              "warmup_passes,measure_passes,passes,measured_accesses_per_thread,"
              "min_cycles_per_access,median_cycles_per_access,"
              "avg_cycles_per_access,max_cycles_per_access\n");

  for (int active_threads : sweep_threads) {
    if (active_threads > max_threads) {
      continue;
    }

    std::size_t total_nodes = bytes_to_nodes(hot_set_bytes);

    // Ensure each active thread has a minimally meaningful private ring.
    if (total_nodes < static_cast<std::size_t>(active_threads) * 8) {
      total_nodes = static_cast<std::size_t>(active_threads) * 8;
    }

    // Make used capacity exact and explicit.
    const std::size_t nodes_per_thread =
        total_nodes / static_cast<std::size_t>(active_threads);
    total_nodes = nodes_per_thread * static_cast<std::size_t>(active_threads);

    const std::size_t used_hot_set_bytes = total_nodes * kNodeBytes;

    std::vector<U64> host(total_nodes * kNodeQWords);

    U64* dev_next = nullptr;
    if (!alloc_and_copy(&host, active_threads, nodes_per_thread, &dev_next)) {
      return EXIT_FAILURE;
    }

    U64* dev_cycles = nullptr;
    U32* dev_final = nullptr;
    CHECK_CUDA(cudaMalloc(&dev_cycles, active_threads * sizeof(U64)));
    CHECK_CUDA(cudaMalloc(&dev_final, active_threads * sizeof(U32)));

    const U64 measured_accesses =
        static_cast<U64>(nodes_per_thread) * static_cast<U64>(measure_passes);
    std::vector<double> pass_samples;
    pass_samples.reserve(cal_kernels::kNativePasses);
    for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
      // One block only, so all active threads can synchronize before timing.
      probe_kernel<<<1, 128>>>(reinterpret_cast<U64>(dev_next),
                               static_cast<U32>(nodes_per_thread),
                               active_threads, warmup_passes, measure_passes,
                               dev_cycles, dev_final);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      std::vector<U64> cycles(active_threads);
      CHECK_CUDA(cudaMemcpy(cycles.data(), dev_cycles,
                            active_threads * sizeof(U64),
                            cudaMemcpyDeviceToHost));

      double avg_value = 0.0;
      int active_count = 0;
      for (int i = 0; i < active_threads; ++i) {
        avg_value += static_cast<double>(cycles[i]) /
                     static_cast<double>(measured_accesses);
        ++active_count;
      }
      if (active_count > 0) {
        avg_value /= static_cast<double>(active_count);
      }
      pass_samples.push_back(avg_value);
    }

#ifdef GPGPU_SIM
    std::printf("%zu,%zu,%d,%zu,%d,%d,%d,%" PRIu64 ",N/A,N/A,N/A,N/A\n",
                hot_set_bytes >> 10, used_hot_set_bytes >> 10, active_threads,
                nodes_per_thread, warmup_passes, measure_passes,
                cal_kernels::kNativePasses, measured_accesses);
#else
    const auto stats = cal_kernels::summarize_samples(pass_samples);
    std::printf("%zu,%zu,%d,%zu,%d,%d,%d,%" PRIu64 ",%.4f,%.4f,%.4f,%.4f\n",
                hot_set_bytes >> 10, used_hot_set_bytes >> 10, active_threads,
                nodes_per_thread, warmup_passes, measure_passes,
                cal_kernels::kNativePasses, measured_accesses, stats.min,
                stats.median, stats.avg, stats.max);
#endif

    CHECK_CUDA(cudaFree(dev_final));
    CHECK_CUDA(cudaFree(dev_cycles));
    CHECK_CUDA(cudaFree(dev_next));
  }

#ifdef GPGPU_SIM
  std::printf(
      "GPGPU-Sim note: each thread now warms and measures a full private ring "
      "segment. The hot set is reported in kB.\n");
#endif

  return EXIT_SUCCESS;
}
