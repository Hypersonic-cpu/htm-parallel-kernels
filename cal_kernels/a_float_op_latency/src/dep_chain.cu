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
// Keep a true dependent chain, but bound simulator runtime to a practical
// smoke-test length.
constexpr int kChainLen = 1024 * 4;
#else
constexpr int kChainLen = 1024 * 32;
#endif
constexpr int threadsPerBlk = 1;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status__ = (call);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status__));                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

__global__ void dep_chain_fp(float *out, std::uint64_t *cycles) {
  if (blockIdx.x != 0) {
    return;
  }
#ifndef GPGPU_SIM
  std::uint64_t t0 = clock64();
#endif
  float value = static_cast<float>(threadIdx.x + 1);
#pragma unroll 128
  for (int i = 0; i < kChainLen; ++i) {
    value = value + 1.0f;
  }
#ifndef GPGPU_SIM
  std::uint64_t t1 = clock64();
#endif
  if (threadIdx.x == 0) {
    *out = value;
#ifdef GPGPU_SIM
    *cycles = 0;
#else
    *cycles = t1 - t0;
#endif
  }
}

__global__ void dep_chain_int(int *out, std::uint64_t *cycles) {
  if (blockIdx.x != 0) {
    return;
  }
#ifndef GPGPU_SIM
  std::uint64_t t0 = clock64();
#endif
  int value = threadIdx.x + 1;
#pragma unroll 128
  for (int i = 0; i < kChainLen; ++i) {
    // asm volatile("" : "+r"(value));
    // value = value + 1;
    asm volatile("add.s32 %0, %0, 1;" : "+r"(value));
  }
#ifndef GPGPU_SIM
  std::uint64_t t1 = clock64();
#endif
  if (threadIdx.x == 0) {
    *out = value;
#ifdef GPGPU_SIM
    *cycles = 0;
#else
    *cycles = t1 - t0;
#endif
  }
}

} // namespace

int main() {
  float *d_fp = nullptr;
  int *d_int = nullptr;
  std::uint64_t *d_cycles = nullptr;
  CHECK_CUDA(cudaMalloc(&d_fp, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_int, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(std::uint64_t)));

  std::vector<std::uint64_t> fp_cycle_samples;
  std::vector<std::uint64_t> int_cycle_samples;
  fp_cycle_samples.reserve(cal_kernels::kNativePasses);
  int_cycle_samples.reserve(cal_kernels::kNativePasses);

  for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
    dep_chain_fp<<<1, threadsPerBlk>>>(d_fp, d_cycles);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::uint64_t fp_cycles = 0;
    CHECK_CUDA(cudaMemcpy(&fp_cycles, d_cycles, sizeof(fp_cycles),
                          cudaMemcpyDeviceToHost));
    fp_cycle_samples.push_back(fp_cycles);

    dep_chain_int<<<1, threadsPerBlk>>>(d_int, d_cycles);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::uint64_t int_cycles = 0;
    CHECK_CUDA(cudaMemcpy(&int_cycles, d_cycles, sizeof(int_cycles),
                          cudaMemcpyDeviceToHost));
    int_cycle_samples.push_back(int_cycles);
  }

  int host_int = 0;
  float host_fp = 0.0f;
  CHECK_CUDA(
      cudaMemcpy(&host_int, d_int, sizeof(host_int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(&host_fp, d_fp, sizeof(host_fp), cudaMemcpyDeviceToHost));

  std::printf("dep_chain: chain_len=%d\n", kChainLen);
#ifdef GPGPU_SIM
  std::printf("FADD chain latency: N/A in GPGPU-Sim (out=%.1f)\n", host_fp);
  std::printf("IADD chain latency: N/A in GPGPU-Sim (out=%d)\n", host_int);
  std::printf("GPGPU-Sim note: use make ARCH=single report for per-kernel "
              "gpu_sim_cycle/gpu_sim_insn/gpu_ipc.\n");
#else
  const auto fp_total = cal_kernels::summarize_numeric(fp_cycle_samples);
  const auto fp_per_op = cal_kernels::summarize_samples(
      cal_kernels::normalize_samples(fp_cycle_samples,
                                     static_cast<double>(kChainLen)));
  const auto int_total = cal_kernels::summarize_numeric(int_cycle_samples);
  const auto int_per_op = cal_kernels::summarize_samples(
      cal_kernels::normalize_samples(int_cycle_samples,
                                     static_cast<double>(kChainLen)));
  std::printf(
      "FADD chain latency [passes=%d]: min/median/avg/max = %.4f / %.4f / "
      "%.4f / %.4f cycles/op; total cycles = %.0f / %.0f / %.4f / %.0f "
      "(out=%.1f)\n",
      cal_kernels::kNativePasses, fp_per_op.min, fp_per_op.median,
      fp_per_op.avg, fp_per_op.max, fp_total.min, fp_total.median,
      fp_total.avg, fp_total.max, host_fp);
  std::printf(
      "IADD chain latency [passes=%d]: min/median/avg/max = %.4f / %.4f / "
      "%.4f / %.4f cycles/op; total cycles = %.0f / %.0f / %.4f / %.0f "
      "(out=%d)\n",
      cal_kernels::kNativePasses, int_per_op.min, int_per_op.median,
      int_per_op.avg, int_per_op.max, int_total.min, int_total.median,
      int_total.avg, int_total.max, host_int);
  std::vector<double> adjusted_samples;
  adjusted_samples.reserve(fp_cycle_samples.size());
  for (std::size_t i = 0; i < fp_cycle_samples.size(); ++i) {
    adjusted_samples.push_back(
        static_cast<double>(fp_cycle_samples[i] - int_cycle_samples[i]) /
        static_cast<double>(kChainLen));
  }
  if (!adjusted_samples.empty()) {
    const auto adjusted = cal_kernels::summarize_samples(adjusted_samples);
    std::printf("FADD adjusted latency [passes=%d]: min/median/avg/max = "
                "%.4f / %.4f / %.4f / %.4f cycles/op\n",
                cal_kernels::kNativePasses, adjusted.min, adjusted.median,
                adjusted.avg, adjusted.max);
  }
#endif

  CHECK_CUDA(cudaFree(d_cycles));
  CHECK_CUDA(cudaFree(d_int));
  CHECK_CUDA(cudaFree(d_fp));
  return EXIT_SUCCESS;
}
