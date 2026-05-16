#include <cassert>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
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

constexpr int kNnzPerRow = 8;
constexpr int kDenseCols = 4;
constexpr int kThreads = 128;
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

#if defined(HW_H100)
const CaseConfig kCases[] = {
    {"test", 4096, 4096},
    {"le_l2", 262144, 262144},
    {"approx_l2", 524288, 524288},
    {"gt_l2", 2097152, 2097152},
};
#else
const CaseConfig kCases[] = {
    {"le_l2", 32768, 32768},
    {"approx_l2", 65536, 65536},
    {"gt_l2", 262144, 262144},
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

__global__ void spmm_csr_kernel(const int *row_ptr, const int *col_idx,
                                const float *values, const float *dense,
                                float *out, int rows, int dense_cols) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * dense_cols;
  for (int item = linear; item < total; item += gridDim.x * blockDim.x) {
    const int row = item / dense_cols;
    const int col = item - row * dense_cols;
    float sum = 0.0f;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      sum += values[p] * dense[col_idx[p] * dense_cols + col];
    }
    out[item] = sum;
  }
}

void build_csr(const CaseConfig &cfg, std::vector<int> *row_ptr,
               std::vector<int> *col_idx, std::vector<float> *values) {
  row_ptr->resize(static_cast<std::size_t>(cfg.rows) + 1);
  col_idx->resize(static_cast<std::size_t>(cfg.rows) * kNnzPerRow);
  values->resize(static_cast<std::size_t>(cfg.rows) * kNnzPerRow);
  std::mt19937 rng(0x5eed1234u);
  for (int row = 0; row < cfg.rows; ++row) {
    (*row_ptr)[row] = row * kNnzPerRow;
    const int base = (row * 17) & (cfg.cols - 1);
    for (int j = 0; j < kNnzPerRow; ++j) {
      const int p = row * kNnzPerRow + j;
      (*col_idx)[p] =
          (base + j * 257 + static_cast<int>(rng() & 63)) & (cfg.cols - 1);
      (*values)[p] = 0.125f * static_cast<float>((row + j) % 11 - 5);
    }
  }
  (*row_ptr)[cfg.rows] = cfg.rows * kNnzPerRow;
}

std::size_t footprint_bytes(const CaseConfig &cfg) {
  return (static_cast<std::size_t>(cfg.rows) + 1) * sizeof(int) +
         static_cast<std::size_t>(cfg.rows) * kNnzPerRow *
             (sizeof(int) + sizeof(float)) +
         static_cast<std::size_t>(cfg.cols) * kDenseCols * sizeof(float) +
         static_cast<std::size_t>(cfg.rows) * kDenseCols * sizeof(float);
}

int run_case(const CaseConfig &cfg) {
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;
  build_csr(cfg, &row_ptr, &col_idx, &values);

  std::vector<float> dense(static_cast<std::size_t>(cfg.cols) * kDenseCols);
  for (int i = 0; i < static_cast<int>(dense.size()); ++i) {
    dense[i] = static_cast<float>((i * 19) % 31 - 15) / 16.0f;
  }

  std::vector<float> expected(static_cast<std::size_t>(cfg.rows) * kDenseCols,
                              0.0f);
  for (int row = 0; row < cfg.rows; ++row) {
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      for (int col = 0; col < kDenseCols; ++col) {
        expected[static_cast<std::size_t>(row) * kDenseCols + col] +=
            values[p] *
            dense[static_cast<std::size_t>(col_idx[p]) * kDenseCols + col];
      }
    }
  }

  int *dev_row_ptr = nullptr;
  int *dev_col_idx = nullptr;
  float *dev_values = nullptr;
  float *dev_dense = nullptr;
  float *dev_out = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_row_ptr, row_ptr.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dev_col_idx, col_idx.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dev_values, values.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_dense, dense.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_out, expected.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dev_row_ptr, row_ptr.data(),
                        row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_col_idx, col_idx.data(),
                        col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_values, values.data(),
                        values.size() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_dense, dense.data(), dense.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
#if !defined(PERF_RUN)
  float *dev_l2_flush = nullptr;
  const std::size_t l2_flush_elements =
      cal_kernels::kColdL2FlushBytes / sizeof(float);
  CHECK_CUDA(cudaMalloc(&dev_l2_flush, l2_flush_elements * sizeof(float)));
  CHECK_CUDA(cudaMemset(dev_l2_flush, 0, l2_flush_elements * sizeof(float)));
#endif

  const int total = cfg.rows * kDenseCols;
  const int blocks = (total + kThreads - 1) / kThreads;
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
#if !defined(PERF_RUN)
      CHECK_CUDA(
          cal_kernels::run_cold_l2_flush(dev_l2_flush, l2_flush_elements));
#endif
#ifdef GPGPU_SIM
      const auto start_time = std::chrono::steady_clock::now();
      spmm_csr_kernel<<<blocks, kThreads>>>(dev_row_ptr, dev_col_idx,
                                            dev_values, dev_dense, dev_out,
                                            cfg.rows, kDenseCols);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto stop_time = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(stop_time - start_time)
              .count();
#else
      CHECK_CUDA(cudaEventRecord(start));
      spmm_csr_kernel<<<blocks, kThreads>>>(dev_row_ptr, dev_col_idx,
                                            dev_values, dev_dense, dev_out,
                                            cfg.rows, kDenseCols);
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

  std::vector<float> observed(expected.size());
  CHECK_CUDA(cudaMemcpy(observed.data(), dev_out,
                        observed.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  double error = 0.0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    checksum += observed[i];
    error += std::fabs(static_cast<double>(observed[i] - expected[i]));
  }

  const auto time_stats = cal_kernels::summarize_samples(time_samples);
  std::printf(
      "REAL_spmm,%s,%d,%d,%d,%d,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
      cfg.name, cfg.rows, cfg.cols, static_cast<int>(values.size()), kDenseCols,
      footprint_bytes(cfg), kRepeats, cal_kernels::kNativePasses, checksum,
      error, time_stats.min, time_stats.median, time_stats.avg, time_stats.max);

#ifndef GPGPU_SIM
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaEventDestroy(start));
#endif
#if !defined(PERF_RUN)
  CHECK_CUDA(cudaFree(dev_l2_flush));
#endif
  CHECK_CUDA(cudaFree(dev_out));
  CHECK_CUDA(cudaFree(dev_dense));
  CHECK_CUDA(cudaFree(dev_values));
  CHECK_CUDA(cudaFree(dev_col_idx));
  CHECK_CUDA(cudaFree(dev_row_ptr));
  if (error > 1e-3) {
    std::fprintf(stderr, "REAL_spmm mismatch for %s\n", cfg.name);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char *argv[]) {
  std::printf("workload,case,rows,cols,nnz,dense_cols,bytes,repeats,passes,"
              "checksum,abs_error,min_ms,median_ms,avg_ms,max_ms\n");
  assert(argc == 2);
  std::string case_name{argv[1]};

  for (const CaseConfig &cfg : kCases) {
    if (case_name == cfg.name) {
      const int rc = run_case(cfg);
      if (rc != EXIT_SUCCESS) {
        return rc;
      }
      return EXIT_SUCCESS;
    }
  }
  
  std::printf("No such case name: %s\n", case_name.c_str());
  return EXIT_FAILURE;
}
