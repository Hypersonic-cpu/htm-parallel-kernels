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

#ifdef GPGPU_SIM
constexpr int kChainLen = 1024 * 4;
#else
constexpr int kChainLen = 1024 * 32;
#endif
constexpr int kThreadsPerBlock = 1;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status__ = (call);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status__));                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

__global__ void dep_chain_fp_warm(float *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float value = 1.0f;
#pragma unroll 128
  for (int i = 0; i < kChainLen; ++i) {
    value = value + 1.0f;
  }
  *out = value;
}

__global__ void dep_chain_fp_measure(float *out, std::uint64_t *cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
#ifndef GPGPU_SIM
  const std::uint64_t start = clock64();
#endif
  float value = 1.0f;
#pragma unroll 128
  for (int i = 0; i < kChainLen; ++i) {
    value = value + 1.0f;
  }
#ifndef GPGPU_SIM
  const std::uint64_t stop = clock64();
  *cycles = stop - start;
#else
  *cycles = 0;
#endif
  *out = value;
}

}  // namespace

int main() {
  float *d_out = nullptr;
  std::uint64_t *d_cycles = nullptr;
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(std::uint64_t)));

  std::vector<std::uint64_t> cycle_samples;
  cycle_samples.reserve(cal_kernels::kNativePasses);
  float host_out = 0.0f;

  for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
    dep_chain_fp_warm<<<1, kThreadsPerBlock>>>(d_out);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    dep_chain_fp_measure<<<1, kThreadsPerBlock>>>(d_out, d_cycles);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::uint64_t cycles = 0;
    CHECK_CUDA(cudaMemcpy(&cycles, d_cycles, sizeof(cycles),
                          cudaMemcpyDeviceToHost));
    cycle_samples.push_back(cycles);
  }

  CHECK_CUDA(cudaMemcpy(&host_out, d_out, sizeof(host_out), cudaMemcpyDeviceToHost));

  std::printf("VAL_fp_op,chain_len=%d\n", kChainLen);
#ifdef GPGPU_SIM
  std::printf("FADD dependent chain: separate warm and measured launches emitted "
              "for GPGPU-Sim report parsing (out=%.1f)\n",
              host_out);
#else
  const auto total_stats = cal_kernels::summarize_numeric(cycle_samples);
  const auto per_op_stats = cal_kernels::summarize_samples(
      cal_kernels::normalize_samples(cycle_samples, static_cast<double>(kChainLen)));
  std::printf(
      "FADD chain latency [passes=%d]: min/median/avg/max = %.4f / %.4f / "
      "%.4f / %.4f cycles/op; total cycles = %.0f / %.0f / %.4f / %.0f "
      "(out=%.1f)\n",
      cal_kernels::kNativePasses, per_op_stats.min, per_op_stats.median,
      per_op_stats.avg, per_op_stats.max, total_stats.min, total_stats.median,
      total_stats.avg, total_stats.max, host_out);
#endif

  CHECK_CUDA(cudaFree(d_cycles));
  CHECK_CUDA(cudaFree(d_out));
  return EXIT_SUCCESS;
}
