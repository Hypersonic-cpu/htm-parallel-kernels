#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

constexpr int kWarpSize = 32;
constexpr int kNodesPerThread = 32;
constexpr int kSmemElems = kWarpSize * kNodesPerThread;

#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
constexpr int kWarmPasses = 2;
constexpr int kMeasurePasses = 16;
#else
constexpr int kWarmPasses = 1;
constexpr int kMeasurePasses = 4;
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

__device__ __forceinline__ void init_private_ring(int *next, int active_threads) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= active_threads) {
    return;
  }
  const int base = tid * kNodesPerThread;
#pragma unroll
  for (int i = 0; i < kNodesPerThread; ++i) {
    next[base + i] = base + ((i + 1) & (kNodesPerThread - 1));
  }
}

__global__ void smem_parallel_warm(int active_threads, int warm_steps) {
  __shared__ int next[kSmemElems];
  if (blockIdx.x != 0 || threadIdx.x >= kWarpSize) {
    return;
  }

  init_private_ring(next, active_threads);
  __syncthreads();

  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= active_threads) {
    return;
  }
  int current = tid * kNodesPerThread;
#pragma unroll 128
  for (int i = 0; i < warm_steps; ++i) {
    current = next[current];
  }
  if (current == -1) {
    asm volatile("");
  }
}

__global__ void smem_parallel_measure(int active_threads, int measure_steps,
                                      std::uint64_t *cycles_out,
                                      int *sink_out) {
  __shared__ int next[kSmemElems];
  if (blockIdx.x != 0 || threadIdx.x >= kWarpSize) {
    return;
  }

  init_private_ring(next, active_threads);
  __syncthreads();

  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= active_threads) {
    cycles_out[tid] = 0;
    sink_out[tid] = 0;
    return;
  }

  int current = tid * kNodesPerThread;
  __syncwarp();

#ifndef GPGPU_SIM
  const std::uint64_t start = clock64();
#endif
#pragma unroll 128
  for (int i = 0; i < measure_steps; ++i) {
    current = next[current];
  }
#ifndef GPGPU_SIM
  const std::uint64_t stop = clock64();
  cycles_out[tid] = stop - start;
#else
  cycles_out[tid] = 0;
#endif
  sink_out[tid] = current;
}

}  // namespace

int main() {
  std::uint64_t *d_cycles = nullptr;
  int *d_sink = nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles, kWarpSize * sizeof(std::uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_sink, kWarpSize * sizeof(int)));

  const int sweep_threads[] = {1, 2, 4, 8, 16, 32};
  const int warm_steps = kNodesPerThread * kWarmPasses;
  const int measure_steps = kNodesPerThread * kMeasurePasses;
  std::printf("active_threads,working_set_bytes,ring_nodes_per_thread,warm_steps,"
              "measure_steps,passes,min_avg_cycles_per_access,"
              "median_avg_cycles_per_access,avg_avg_cycles_per_access,"
              "max_avg_cycles_per_access,min_max_cycles_per_access,"
              "median_max_cycles_per_access,avg_max_cycles_per_access,"
              "max_max_cycles_per_access,sink\n");

  for (int active_threads : sweep_threads) {
    int host_sink = 0;
    std::vector<double> avg_samples;
    std::vector<double> max_samples;
    avg_samples.reserve(cal_kernels::kNativePasses);
    max_samples.reserve(cal_kernels::kNativePasses);

    for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
      smem_parallel_warm<<<1, kWarpSize>>>(active_threads, warm_steps);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      smem_parallel_measure<<<1, kWarpSize>>>(active_threads, measure_steps,
                                              d_cycles, d_sink);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      std::vector<std::uint64_t> cycles(kWarpSize);
      std::vector<int> sinks(kWarpSize);
      CHECK_CUDA(cudaMemcpy(cycles.data(), d_cycles, kWarpSize * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(sinks.data(), d_sink, kWarpSize * sizeof(int),
                            cudaMemcpyDeviceToHost));
      host_sink = sinks[0];

      double avg_value = 0.0;
      double max_value = 0.0;
      for (int tid = 0; tid < active_threads; ++tid) {
        const double lane_cycles =
            static_cast<double>(cycles[tid]) / static_cast<double>(measure_steps);
        avg_value += lane_cycles;
        max_value = std::max(max_value, lane_cycles);
      }
      avg_value /= static_cast<double>(active_threads);
      avg_samples.push_back(avg_value);
      max_samples.push_back(max_value);
    }

#ifdef GPGPU_SIM
    std::printf("%d,%d,%d,%d,%d,%d,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,%d\n",
                active_threads, active_threads * kNodesPerThread *
                                    static_cast<int>(sizeof(int)),
                kNodesPerThread, warm_steps, measure_steps,
                cal_kernels::kNativePasses, host_sink);
#else
    const auto avg_stats = cal_kernels::summarize_samples(avg_samples);
    const auto max_stats = cal_kernels::summarize_samples(max_samples);
    std::printf("%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                active_threads,
                active_threads * kNodesPerThread * static_cast<int>(sizeof(int)),
                kNodesPerThread, warm_steps, measure_steps,
                cal_kernels::kNativePasses, avg_stats.min, avg_stats.median,
                avg_stats.avg, avg_stats.max, max_stats.min, max_stats.median,
                max_stats.avg, max_stats.max, host_sink);
#endif
  }

#ifdef GPGPU_SIM
  std::printf("GPGPU-Sim note: warm and measured launches are separate; timing "
              "excludes the dependent shared-memory setup path.\n");
#endif

  CHECK_CUDA(cudaFree(d_sink));
  CHECK_CUDA(cudaFree(d_cycles));
  return EXIT_SUCCESS;
}
