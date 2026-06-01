#include <cuda_runtime.h>

#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "native_stats.h"

#if defined(HW_H100)
#pragma message("compile-time info: Hopper H100")
#elif defined(HW_A100)
#pragma message("compile-time info: Ampere A100")
#elif defined(HW_V100)
#pragma message("compile-time info: Volta V100")
#else
#error Unsupported hardware target. Define HW_V100, HW_H100, or HW_A100.
#endif

namespace {

constexpr int kThreadsPerBlock = 256;
#if defined(GPGPU_SIM) || defined(PERF_RUN)
constexpr int kIters = 1;
#else
constexpr int kIters = 10;
#endif

#if defined(HTM_CONF_GV100)
constexpr std::size_t kL2Bytes = 6ULL * 1024 * 1024;
#elif defined(HTM_CONF_A100)
constexpr std::size_t kL2Bytes = 40ULL * 1024 * 1024;
#elif defined(HTM_CONF_GH100)
constexpr std::size_t kL2Bytes = 50ULL * 1024 * 1024;
#else
#error "Unknown HTM_CONF_*"
#endif

constexpr std::size_t kWorkingSetBytes = kL2Bytes / 3;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status__ = (call);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status__));                              \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

__device__ __forceinline__ float4 ld_global_cg_v4_f32(const float4 *ptr) {
  float4 value;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(ptr));
  return value;
}

__global__ void probe_kernel_measure(const float4 *in, float *out,
                                     std::uint64_t *accesses_out,
                                     std::size_t n_vec,
                                     int iters) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

  float sum = 0.0f;
  std::uint64_t accesses = 0;
  for (int iter = 0; iter < iters; ++iter) {
    for (std::size_t i = tid; i < n_vec; i += stride) {
      const float4 value = ld_global_cg_v4_f32(in + i);
      sum += value.x + value.y + value.z + value.w;
      ++accesses;
    }
  }

  out[tid] = sum;
  accesses_out[tid] = accesses;
}

double run_measure_kernel(int blocks, float *dev_out,
                          std::uint64_t *dev_accesses,
                          const float4 *dev_in, std::size_t n_vec) {
  CHECK_CUDA(cudaMemset(dev_out, 0,
                        static_cast<std::size_t>(blocks) * kThreadsPerBlock *
                            sizeof(float)));
  CHECK_CUDA(cudaMemset(dev_accesses, 0,
                        static_cast<std::size_t>(blocks) * kThreadsPerBlock *
                            sizeof(std::uint64_t)));
  const auto start_time = std::chrono::steady_clock::now();
  probe_kernel_measure<<<blocks, kThreadsPerBlock>>>(dev_in, dev_out,
                                                     dev_accesses, n_vec,
                                                     kIters);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  const auto stop_time = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop_time - start_time)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <blocks>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const int blocks = std::atoi(argv[1]);
  if (blocks <= 0) {
    std::fprintf(stderr, "blocks must be > 0\n");
    return EXIT_FAILURE;
  }

  const int total_threads = blocks * kThreadsPerBlock;
  const std::size_t n_vec = kWorkingSetBytes / sizeof(float4);
  const std::size_t working_set_bytes = n_vec * sizeof(float4);

  std::vector<float> host(working_set_bytes / sizeof(float));
  for (std::size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<float>((i % 251) + 1) * 0.5f;
  }

  float *dev_in = nullptr;
  float *dev_out = nullptr;
  std::uint64_t *dev_accesses = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_in, working_set_bytes));
  CHECK_CUDA(cudaMemcpy(dev_in, host.data(), working_set_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dev_out, static_cast<std::size_t>(total_threads) *
                                      sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dev_accesses, static_cast<std::size_t>(total_threads) *
                                           sizeof(std::uint64_t)));

  std::vector<double> first_ms_samples;
  std::vector<double> second_ms_samples;
  first_ms_samples.reserve(cal_kernels::kNativePasses);
  second_ms_samples.reserve(cal_kernels::kNativePasses);

  for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
    first_ms_samples.push_back(run_measure_kernel(
        blocks, dev_out, dev_accesses, reinterpret_cast<const float4 *>(dev_in),
        n_vec));
    second_ms_samples.push_back(run_measure_kernel(
        blocks, dev_out, dev_accesses, reinterpret_cast<const float4 *>(dev_in),
        n_vec));
  }

  std::vector<float> partial(total_threads);
  std::vector<std::uint64_t> accesses(total_threads);
  CHECK_CUDA(cudaMemcpy(partial.data(), dev_out,
                        static_cast<std::size_t>(total_threads) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(accesses.data(), dev_accesses,
                        static_cast<std::size_t>(total_threads) *
                            sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost));

  double checksum = 0.0;
  std::uint64_t total_accesses = 0;
  for (float value : partial) {
    checksum += static_cast<double>(value);
  }
  for (std::uint64_t value : accesses) {
    total_accesses += value;
  }

  const auto first_stats = cal_kernels::summarize_samples(first_ms_samples);
  const auto second_stats = cal_kernels::summarize_samples(second_ms_samples);
  std::printf(
      "benchmark,blocks,threads_per_block,total_threads,working_set_bytes,iters,"
      "passes,total_accesses,first_min_ms,first_median_ms,first_avg_ms,"
      "first_max_ms,second_min_ms,second_median_ms,second_avg_ms,"
      "second_max_ms,checksum\n");
  std::printf(
      "VAL_l2_parallel,%d,%d,%d,%zu,%d,%d,%" PRIu64 ",%.6f,%.6f,%.6f,%.6f,"
      "%.6f,%.6f,%.6f,%.6f,%.4f\n",
      blocks, kThreadsPerBlock, total_threads, working_set_bytes, kIters,
      cal_kernels::kNativePasses, total_accesses, first_stats.min,
      first_stats.median, first_stats.avg, first_stats.max, second_stats.min,
      second_stats.median, second_stats.avg, second_stats.max, checksum);

  CHECK_CUDA(cudaFree(dev_accesses));
  CHECK_CUDA(cudaFree(dev_out));
  CHECK_CUDA(cudaFree(dev_in));
  return EXIT_SUCCESS;
}
