#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <cstring>
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
    {"le_l2", 2 * 1024 * 1024},
    {"approx_l2", 4 * 1024 * 1024},
    {"gt_l2", 16 * 1024 * 1024},
};
#else
const CaseConfig kCases[] = {
    {"le_l2", 256 * 1024},
    {"approx_l2", 512 * 1024},
    {"gt_l2", 4 * 1024 * 1024},
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

__global__ void scatter_kernel(const float *x, const int *idx, float *y,
                               int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
    y[idx[i]] = x[i];
  }
}

int run_case(const CaseConfig &cfg) {
  std::vector<float> x(cfg.elements), y(cfg.elements);
  std::vector<float> expected(cfg.elements, 0.0f);
  std::vector<int> idx(cfg.elements);
  for (int i = 0; i < cfg.elements; ++i) {
    x[i] = static_cast<float>((i * 7) % 2048) * 0.25f;
    idx[i] = (i * 33) & (cfg.elements - 1); // permutation because 33 is odd.
    expected[idx[i]] = x[i];
  }

  float *dev_x = nullptr;
  float *dev_y = nullptr;
  int *dev_idx = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_x, x.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_y, y.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_idx, idx.size() * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(dev_x, x.data(), x.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_idx, idx.data(), idx.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dev_y, 0, y.size() * sizeof(float)));
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
  float *dev_l2_flush = nullptr;
  const std::size_t l2_flush_elements =
      cal_kernels::kColdL2FlushBytes / sizeof(float);
  CHECK_CUDA(cudaMalloc(&dev_l2_flush, l2_flush_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(dev_l2_flush, 0, l2_flush_elements * sizeof(float)));
#endif

  const int blocks = 256;
  std::vector<double> time_samples;
  time_samples.reserve(cal_kernels::kNativePasses);
  for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
    CHECK_CUDA(cudaMemset(dev_y, 0, y.size() * sizeof(float)));
    std::vector<double> launch_samples;
    launch_samples.reserve(kRepeats);
    for (int rep = 0; rep < kRepeats; ++rep) {
#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
      CHECK_CUDA(cal_kernels::run_cold_l2_flush(dev_l2_flush, l2_flush_elements));
#endif
      const auto start_time = std::chrono::steady_clock::now();
      scatter_kernel<<<blocks, kThreads>>>(dev_x, dev_idx, dev_y, cfg.elements);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto stop_time = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(stop_time - start_time)
              .count();
      launch_samples.push_back(static_cast<double>(elapsed_ms));
    }
    time_samples.push_back(
        cal_kernels::average_samples(launch_samples, kWarmLaunches));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(y.data(), dev_y, y.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  double error = 0.0;
  for (int i = 0; i < cfg.elements; ++i) {
    checksum += y[i];
    error += std::fabs(static_cast<double>(y[i] - expected[i]));
  }

  const std::size_t bytes =
      x.size() * sizeof(float) + y.size() * sizeof(float) +
      idx.size() * sizeof(int);
  const auto time_stats = cal_kernels::summarize_samples(time_samples);
  std::printf("REAL_scatter,%s,%d,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              cfg.name, cfg.elements, bytes, kRepeats,
              cal_kernels::kNativePasses, checksum, error, time_stats.min,
              time_stats.median, time_stats.avg, time_stats.max);

#if !defined(GPGPU_SIM) && !defined(PERF_RUN)
  CHECK_CUDA(cudaFree(dev_l2_flush));
#endif
  CHECK_CUDA(cudaFree(dev_idx));
  CHECK_CUDA(cudaFree(dev_y));
  CHECK_CUDA(cudaFree(dev_x));
  if (error != 0.0) {
    std::fprintf(stderr, "REAL_scatter mismatch for %s\n", cfg.name);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <case>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const char *case_filter = argv[1];
  bool ran_case = false;
  std::printf("workload,case,elements,bytes,repeats,passes,checksum,abs_error,"
              "min_ms,median_ms,avg_ms,max_ms\n");
  for (const CaseConfig &cfg : kCases) {
    if (std::strcmp(case_filter, cfg.name) != 0) {
      continue;
    }
    ran_case = true;
    const int rc = run_case(cfg);
    if (rc != EXIT_SUCCESS) {
      return rc;
    }
  }
  if (!ran_case) {
    std::fprintf(stderr, "REAL_scatter case '%s' not found\n", case_filter);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
