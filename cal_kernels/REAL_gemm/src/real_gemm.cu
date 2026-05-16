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

constexpr int kTile = 16;
constexpr int kThreadsX = kTile;
constexpr int kThreadsY = kTile;
#if defined(GPGPU_SIM) || defined(PERF_RUN)
constexpr int kRepeats = 1;
#else
constexpr int kRepeats = 10;
#endif
constexpr int kWarmLaunches = 0;

struct CaseConfig {
  const char* name;
  int m;
  int n;
  int k;
};

const CaseConfig kCases[] = {
    {"test", 128, 128, 128},
    {"small", 192, 192, 128},
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

__global__ void gemm_shared_kernel(const float* a, const float* b, float* c, int m,
                                   int n, int k) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  float sum = 0.0f;
  for (int tile = 0; tile < (k + kTile - 1) / kTile; ++tile) {
    const int a_col = tile * kTile + threadIdx.x;
    const int b_row = tile * kTile + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < kTile; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}

int run_case(const CaseConfig& cfg) {
  const std::size_t a_elems = static_cast<std::size_t>(cfg.m) * cfg.k;
  const std::size_t b_elems = static_cast<std::size_t>(cfg.k) * cfg.n;
  const std::size_t c_elems = static_cast<std::size_t>(cfg.m) * cfg.n;
  std::vector<float> a(a_elems), b(b_elems), c(c_elems), expected(c_elems, 0.0f);

  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>((i * 13) % 97 - 48) * 0.03125f;
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<float>((i * 17) % 89 - 44) * 0.03125f;
  }
  for (int row = 0; row < cfg.m; ++row) {
    for (int col = 0; col < cfg.n; ++col) {
      float sum = 0.0f;
      for (int p = 0; p < cfg.k; ++p) {
        sum += a[static_cast<std::size_t>(row) * cfg.k + p] *
               b[static_cast<std::size_t>(p) * cfg.n + col];
      }
      expected[static_cast<std::size_t>(row) * cfg.n + col] = sum;
    }
  }

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_a, a.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_b, b.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_c, c.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_a, a.data(), a.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, b.data(), b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

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

  const dim3 block(kThreadsX, kThreadsY);
  const dim3 grid((cfg.n + kTile - 1) / kTile, (cfg.m + kTile - 1) / kTile);
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
      gemm_shared_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto stop_time = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(stop_time - start_time).count();
#else
      CHECK_CUDA(cudaEventRecord(start));
      gemm_shared_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, cfg.m, cfg.n, cfg.k);
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
  for (std::size_t i = 0; i < c.size(); ++i) {
    checksum += c[i];
    abs_error += std::fabs(static_cast<double>(c[i] - expected[i]));
  }

  const std::size_t bytes = (a.size() + b.size() + c.size()) * sizeof(float);
  const auto time_stats = cal_kernels::summarize_samples(time_samples);
  std::printf(
      "REAL_gemm,%s,%d,%d,%d,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
      cfg.name, cfg.m, cfg.n, cfg.k, bytes, kRepeats, cal_kernels::kNativePasses,
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
  if (abs_error > 1e-2) {
    std::fprintf(stderr, "REAL_gemm mismatch for %s\n", cfg.name);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) {
  assert(argc == 2);
  const char* case_filter = argv[1];
  bool ran_case = false;
  std::printf("workload,case,m,n,k,bytes,repeats,passes,checksum,abs_error,min_ms,"
              "median_ms,avg_ms,max_ms\n");
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
    std::fprintf(stderr, "REAL_gemm case '%s' not found\n", case_filter);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
