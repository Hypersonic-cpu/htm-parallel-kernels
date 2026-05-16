#include <cassert>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "l2_flush.h"
#include "native_stats.h"

#if defined(HW_H100)
#pragma message("compile-time info: Hopper H100/A100")
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
  const char* name;
  int elements;
};

const CaseConfig kCases[] = {
    {"test", 1 << 16},
    {"small", 1 << 18},
};

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status__ = (call);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status__));                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

__global__ void vadd_kernel(const float* a, const float* b, float* c, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
    c[i] = a[i] + b[i];
  }
}

int run_case(const CaseConfig& cfg) {
  std::vector<float> a(cfg.elements), b(cfg.elements), c(cfg.elements),
      expected(cfg.elements);
  for (int i = 0; i < cfg.elements; ++i) {
    a[i] = static_cast<float>((i * 7) % 97) * 0.25f;
    b[i] = static_cast<float>((i * 11) % 89) * -0.5f;
    expected[i] = a[i] + b[i];
  }

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, c.size() * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(dev_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dev_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));

#if !defined(PERF_RUN)
  float* dev_l2_flush = nullptr;
  const std::size_t l2_flush_elements = cal_kernels::kColdL2FlushBytes / sizeof(float);
  CHECK_CUDA(cudaMalloc(&dev_l2_flush, l2_flush_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(dev_l2_flush, 0, l2_flush_elements * sizeof(float)));
#endif
#ifndef GPGPU_SIM
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
#endif

  const int blocks = (cfg.elements + kThreads - 1) / kThreads;
  std::vector<double> time_samples;
  time_samples.reserve(cal_kernels::kNativePasses);
  for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
    std::vector<double> launch_samples;
    launch_samples.reserve(kRepeats);
    for (int rep = 0; rep < kRepeats; ++rep) {
#if !defined(PERF_RUN)
      CHECK_CUDA(cal_kernels::run_cold_l2_flush(dev_l2_flush, l2_flush_elements));
#endif
#ifdef GPGPU_SIM
      const auto start_time = std::chrono::steady_clock::now();
      vadd_kernel<<<blocks, kThreads>>>(dev_a, dev_b, dev_c, cfg.elements);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto stop_time = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(stop_time - start_time).count();
#else
      CHECK_CUDA(cudaEventRecord(start));
      vadd_kernel<<<blocks, kThreads>>>(dev_a, dev_b, dev_c, cfg.elements);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaEventRecord(stop));
      CHECK_CUDA(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
#endif
      launch_samples.push_back(static_cast<double>(elapsed_ms));
    }
    time_samples.push_back(cal_kernels::average_samples(launch_samples, kWarmLaunches));
  }
  CHECK_CUDA(cudaMemcpy(
      c.data(), dev_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  double abs_error = 0.0;
  for (int i = 0; i < cfg.elements; ++i) {
    checksum += c[i];
    abs_error += std::fabs(static_cast<double>(c[i] - expected[i]));
  }

  const std::size_t bytes = (a.size() + b.size() + c.size()) * sizeof(float);
  const auto time_stats = cal_kernels::summarize_samples(time_samples);
  std::printf(
      "REAL_vadd,%s,%d,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
      cfg.name, cfg.elements, bytes, kRepeats, cal_kernels::kNativePasses,
      checksum, abs_error, time_stats.min, time_stats.median, time_stats.avg,
      time_stats.max);

#ifndef GPGPU_SIM
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(start));
#endif
#if !defined(PERF_RUN)
  CHECK_CUDA(cudaFree(dev_l2_flush));
#endif
  CHECK_CUDA(cudaFree(dev_c));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_a));
  if (abs_error > 1e-3) {
    std::fprintf(stderr, "REAL_vadd mismatch for %s\n", cfg.name);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) {
  assert(argc == 2);
  const char* case_filter = argv[1];
  bool ran_case = false;
  std::printf("workload,case,elements,bytes,repeats,passes,checksum,abs_error,"
              "min_ms,median_ms,avg_ms,max_ms\n");
  for (const CaseConfig& cfg : kCases) {
    if (case_filter != nullptr && std::strcmp(case_filter, cfg.name) != 0) {
      continue;
    }
    ran_case = true;
    const int rc = run_case(cfg);
    if (rc != EXIT_SUCCESS) {
      return rc;
    }
  }
  if (!ran_case) {
    std::fprintf(stderr, "REAL_vadd case '%s' not found\n", case_filter);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
