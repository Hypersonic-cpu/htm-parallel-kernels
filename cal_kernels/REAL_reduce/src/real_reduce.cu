#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "l2_flush.h"
#include "native_stats.h"

#if defined(HW_H100)
#pragma message("compile-time info: Hopper H100")
#elif defined(HW_V100)
#pragma message("compile-time info: Volta V100")
#else
#error Unsupported hardware target. Define HW_V100 or HW_H100.
#endif

namespace {

constexpr int kThreads = 256;
constexpr int kBlocks = 256;
#if defined(GPGPU_SIM) || defined(PERF_RUN)
constexpr int kRepeats = 1;
#else
constexpr int kRepeats = 10;
#endif
constexpr int kWarmLaunches = 0;

struct CaseConfig {
  const char *name;
  int elements;
};

#if defined(HW_H100) && !defined(GPGPU_SIM)
const CaseConfig kCases[] = {
    {"le_l2", 8 * 1024 * 1024},
    {"approx_l2", 12 * 1024 * 1024},
    {"gt_l2", 64 * 1024 * 1024},
};
#else
const CaseConfig kCases[] = {
    {"le_l2", 1 * 1024 * 1024},
    {"approx_l2", 1536 * 1024},
    {"gt_l2", 16 * 1024 * 1024},
};
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

__global__ void reduce_blocks(const float *input, float *partials, int n) {
  __shared__ float scratch[kThreads];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int idx = blockIdx.x * blockDim.x + tid; idx < n;
       idx += gridDim.x * blockDim.x) {
    sum += input[idx];
  }
  scratch[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partials[blockIdx.x] = scratch[0];
  }
}

int run_case(const CaseConfig &cfg) {
  std::vector<float> host(cfg.elements);
  for (int i = 0; i < cfg.elements; ++i) {
    host[i] = (i & 1) ? -1.0f : 1.0f;
  }

  float *dev_input = nullptr;
  float *dev_partials = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_input, host.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_partials, kBlocks * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_input, host.data(), host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
  float *dev_l2_flush = nullptr;
  const std::size_t l2_flush_elements =
      cal_kernels::kColdL2FlushBytes / sizeof(float);
  CHECK_CUDA(cudaMalloc(&dev_l2_flush, l2_flush_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(dev_l2_flush, 0, l2_flush_elements * sizeof(float)));
#endif
#ifndef GPGPU_SIM
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
#endif
  std::vector<double> time_samples;
  time_samples.reserve(cal_kernels::kNativePasses);

  for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
    std::vector<double> launch_samples;
    launch_samples.reserve(kRepeats);
    for (int rep = 0; rep < kRepeats; ++rep) {
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
      CHECK_CUDA(cal_kernels::run_cold_l2_flush(dev_l2_flush, l2_flush_elements));
#endif
#ifdef GPGPU_SIM
      const auto start_time = std::chrono::steady_clock::now();
      reduce_blocks<<<kBlocks, kThreads>>>(dev_input, dev_partials, cfg.elements);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto stop_time = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(stop_time - start_time)
              .count();
#else
      CHECK_CUDA(cudaEventRecord(start));
      reduce_blocks<<<kBlocks, kThreads>>>(dev_input, dev_partials, cfg.elements);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
#endif
      launch_samples.push_back(static_cast<double>(elapsed_ms));
    }
    time_samples.push_back(
        cal_kernels::average_samples(launch_samples, kWarmLaunches));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> partials(kBlocks);
  CHECK_CUDA(cudaMemcpy(partials.data(), dev_partials,
                        partials.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double observed = 0.0;
  for (float value : partials) {
    observed += value;
  }

  const double expected = 0.0;
  const double error = std::fabs(observed - expected);
  const std::size_t bytes = host.size() * sizeof(float);
  const auto time_stats = cal_kernels::summarize_samples(time_samples);
  std::printf("REAL_reduce,%s,%d,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              cfg.name, cfg.elements, bytes, kRepeats,
              cal_kernels::kNativePasses, expected, observed, error,
              time_stats.min, time_stats.median, time_stats.avg,
              time_stats.max);

#ifndef GPGPU_SIM
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(start));
#endif
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
  CHECK_CUDA(cudaFree(dev_l2_flush));
#endif
  CHECK_CUDA(cudaFree(dev_partials));
  CHECK_CUDA(cudaFree(dev_input));
  if (error > 1e-2) {
    std::fprintf(stderr, "REAL_reduce checksum mismatch for %s\n", cfg.name);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

} // namespace

int main() {
  std::printf("workload,case,elements,bytes,repeats,passes,expected,observed,"
              "abs_error,min_ms,median_ms,avg_ms,max_ms\n");
  for (const CaseConfig &cfg : kCases) {
    const int rc = run_case(cfg);
    if (rc != EXIT_SUCCESS) {
      return rc;
    }
  }
  return EXIT_SUCCESS;
}
