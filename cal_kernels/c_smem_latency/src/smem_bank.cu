#include <cuda_runtime.h>

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

constexpr int kSmemElems = 1024;
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
constexpr int kWarmPasses = 2;
constexpr int kMeasureIters = 512;
#else
constexpr int kWarmPasses = 1;
constexpr int kMeasureIters = 128;
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

__global__ void smem_latency_kernel(int stride, int warm_steps,
                                    std::uint64_t *cycles_out,
                                    int *sink_out) {
  __shared__ int next[kSmemElems];
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (int i = 0; i < kSmemElems; ++i) {
    next[i] = (i + stride) & (kSmemElems - 1);
  }
  __syncthreads();

  int current = 0;
#pragma unroll 128
  for (int i = 0; i < warm_steps; ++i) {
    current = next[current];
  }

#ifndef GPGPU_SIM
  const std::uint64_t t0 = clock64();
#endif
#pragma unroll 128
  for (int i = 0; i < kMeasureIters; ++i) {
    current = next[current];
  }
#ifndef GPGPU_SIM
  const std::uint64_t t1 = clock64();
  cycles_out[0] = t1 - t0;
#else
  cycles_out[0] = 0;
#endif
  sink_out[0] = current;
}

}  // namespace

int main() {
  std::uint64_t *d_cycles = nullptr;
  int *d_sink = nullptr;
  CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(std::uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_sink, sizeof(int)));

  const int strides[] = {1, 3, 5, 17, 33, 65};
  const int warm_steps = kSmemElems * kWarmPasses;
  std::printf("stride_words,working_set_bytes,warm_steps,measure_iters,passes,"
              "min_cycles_per_access,median_cycles_per_access,"
              "avg_cycles_per_access,max_cycles_per_access,min_total_cycles,"
              "median_total_cycles,avg_total_cycles,max_total_cycles,sink\n");

  for (int stride : strides) {
    int host_sink = 0;
    std::vector<std::uint64_t> cycle_samples;
    cycle_samples.reserve(cal_kernels::kNativePasses);

    for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
      smem_latency_kernel<<<1, 1>>>(stride, warm_steps, d_cycles, d_sink);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      std::uint64_t cycles = 0;
      CHECK_CUDA(cudaMemcpy(&cycles, d_cycles, sizeof(cycles),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(
          cudaMemcpy(&host_sink, d_sink, sizeof(host_sink), cudaMemcpyDeviceToHost));
      cycle_samples.push_back(cycles);
    }

#ifdef GPGPU_SIM
    std::printf("%d,%d,%d,%d,%d,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,%d\n", stride,
                kSmemElems * static_cast<int>(sizeof(int)), warm_steps,
                kMeasureIters, cal_kernels::kNativePasses, host_sink);
#else
    const auto total_stats = cal_kernels::summarize_numeric(cycle_samples);
    const auto per_access_stats = cal_kernels::summarize_samples(
        cal_kernels::normalize_samples(cycle_samples,
                                       static_cast<double>(kMeasureIters)));
    std::printf("%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.0f,%.0f,%.4f,%.0f,%d\n",
                stride, kSmemElems * static_cast<int>(sizeof(int)), warm_steps,
                kMeasureIters, cal_kernels::kNativePasses, per_access_stats.min,
                per_access_stats.median, per_access_stats.avg,
                per_access_stats.max, total_stats.min, total_stats.median,
                total_stats.avg, total_stats.max, host_sink);
#endif
  }

#ifdef GPGPU_SIM
  std::printf("GPGPU-Sim note: clock64 timing is disabled in sim mode.\n");
#endif

  CHECK_CUDA(cudaFree(d_sink));
  CHECK_CUDA(cudaFree(d_cycles));
  return EXIT_SUCCESS;
}
