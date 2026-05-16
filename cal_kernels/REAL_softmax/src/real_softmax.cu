#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
#if defined(GPGPU_SIM) || defined(PERF_RUN)
constexpr int kRepeats = 1;
#else
constexpr int kRepeats = 10;
#endif
constexpr int kWarmLaunches = 0;

struct CaseConfig {
  const char *name;
  int rows;
  int cols;
};

#if defined(HW_H100) && !defined(GPGPU_SIM)
const CaseConfig kCases[] = {
    {"le_l2", 4096, 1024},
    {"approx_l2", 6400, 1024},
    {"gt_l2", 32768, 1024},
};
#else
const CaseConfig kCases[] = {
    {"le_l2", 1024, 512},
    {"approx_l2", 1536, 512},
    {"gt_l2", 8192, 1024},
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

__global__ void softmax_kernel(const float *input, float *output, int rows,
                               int cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float scratch[];
  const int tid = threadIdx.x;
  const int base = row * cols;

  float local_max = -3.402823466e+38F;
  for (int col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, input[base + col]);
  }
  scratch[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = scratch[0];

  float local_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = expf(input[base + col] - row_max);
    output[base + col] = value;
    local_sum += value;
  }
  scratch[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float row_sum = scratch[0];

  for (int col = tid; col < cols; col += blockDim.x) {
    output[base + col] /= row_sum;
  }
}

std::size_t elements(const CaseConfig &cfg) {
  return static_cast<std::size_t>(cfg.rows) * static_cast<std::size_t>(cfg.cols);
}

int run_case(const CaseConfig &cfg) {
  const std::size_t n = elements(cfg);
  std::vector<float> input(n);
  std::vector<float> output(n);
  for (std::size_t i = 0; i < n; ++i) {
    input[i] = static_cast<float>((static_cast<int>(i % 257) - 128)) / 32.0f;
  }

  float *dev_input = nullptr;
  float *dev_output = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_input, input.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_output, output.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_input, input.data(), input.size() * sizeof(float),
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
      softmax_kernel<<<cfg.rows, kThreads, kThreads * sizeof(float)>>>(
          dev_input, dev_output, cfg.rows, cfg.cols);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto stop_time = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(stop_time - start_time)
              .count();
#else
      CHECK_CUDA(cudaEventRecord(start));
      softmax_kernel<<<cfg.rows, kThreads, kThreads * sizeof(float)>>>(
          dev_input, dev_output, cfg.rows, cfg.cols);
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
  CHECK_CUDA(cudaMemcpy(output.data(), dev_output, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  double max_row_sum_error = 0.0;
  for (int row = 0; row < cfg.rows; ++row) {
    double row_sum = 0.0;
    for (int col = 0; col < cfg.cols; ++col) {
      row_sum += output[static_cast<std::size_t>(row) * cfg.cols + col];
    }
    checksum += row_sum;
    max_row_sum_error =
        fmax(max_row_sum_error, std::fabs(row_sum - static_cast<double>(1.0)));
  }

  const std::size_t bytes = 2 * n * sizeof(float);
  const auto time_stats = cal_kernels::summarize_samples(time_samples);
  std::printf("REAL_softmax,%s,%d,%d,%zu,%zu,%d,%d,%.6f,%.6e,%.6f,%.6f,%.6f,%.6f\n",
              cfg.name, cfg.rows, cfg.cols, n, bytes, kRepeats,
              cal_kernels::kNativePasses, checksum, max_row_sum_error,
              time_stats.min, time_stats.median, time_stats.avg,
              time_stats.max);

#ifndef GPGPU_SIM
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(start));
#endif
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
  CHECK_CUDA(cudaFree(dev_l2_flush));
#endif
  CHECK_CUDA(cudaFree(dev_output));
  CHECK_CUDA(cudaFree(dev_input));
  if (max_row_sum_error > 1e-3) {
    std::fprintf(stderr, "REAL_softmax row sum mismatch for %s\n", cfg.name);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

} // namespace

int main() {
  std::printf("workload,case,rows,cols,elements,bytes,repeats,passes,checksum,"
              "max_row_sum_error,min_ms,median_ms,avg_ms,max_ms\n");
  for (const CaseConfig &cfg : kCases) {
    const int rc = run_case(cfg);
    if (rc != EXIT_SUCCESS) {
      return rc;
    }
  }
  return EXIT_SUCCESS;
}
