#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#include <cstddef>

namespace cal_kernels {

#ifndef GPGPU_SIM
constexpr std::size_t kColdL2FlushBytes = 128ULL * 1024 * 1024;
constexpr int kColdL2FlushThreads = 256;
constexpr int kColdL2FlushBlocks = 256;

__global__ void cold_l2_flush_kernel(float *data, std::size_t n) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = tid; i < n; i += stride) {
    data[i] = data[i] + 1.0f;
  }
}

inline cudaError_t run_cold_l2_flush(float *dev_flush,
                                     std::size_t flush_elements) {
  printf("run_cold_l2_flush, elem = %lu\n", flush_elements);
  fflush(stdout);
  cold_l2_flush_kernel<<<kColdL2FlushBlocks, kColdL2FlushThreads>>>(
      dev_flush, flush_elements);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceSynchronize();
}
#else
constexpr std::size_t kColdL2FlushBytes = 4 * sizeof(float);

__global__ inline void cold_l2_flush_kernel(float *data, std::size_t n) {
  const std::size_t tid = 0;
  data[tid] = data[tid] + 1.0f;
}

inline cudaError_t run_cold_l2_flush(float *dev_flush,
                                     std::size_t flush_elements) {
  printf("dummy l2 flush, should flushed by GPGPU-Sim\n");
  cold_l2_flush_kernel<<<1, 1>>>(dev_flush, 1);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceSynchronize();
}
#endif

} // namespace cal_kernels
