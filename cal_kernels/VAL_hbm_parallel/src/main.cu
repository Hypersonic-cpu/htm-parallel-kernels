#include <cassert>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

using U64 = std::uint64_t;

struct CaseConfig {
  const char *name;
  std::size_t working_set_bytes;
};

#if defined(HW_H100)
const CaseConfig kCases[] = {
    // {"lt_l2", 32ULL * 1024 * 1024},
    // {"approx_l2", 50ULL * 1024 * 1024},
    {"ge_l2", 64ULL * 1024 * 1024},
};
#else
const CaseConfig kCases[] = {
    {"lt_l2", 4ULL * 1024 * 1024},
    {"approx_l2", 6ULL * 1024 * 1024},
    {"ge_l2", 16ULL * 1024 * 1024},
};
#endif

#if defined(HW_H100)
constexpr int kDefaultMaxBlocks = 1024;
#else
constexpr int kDefaultMaxBlocks = 1024;
#endif

#if defined(GPGPU_SIM) || defined(PERF_RUN)
constexpr int kDefaultIters = 1;
#else
constexpr int kDefaultIters = 10;
#endif
constexpr int kDefaultWarmIters = 0;

constexpr int kThreadsPerBlock = 256;
#if defined(HW_H100)
const int kSweepBlocks[] = {1, 4, 16, 32, 64, 132, 256, 512, 1024};
#else
const int kSweepBlocks[] = {1, 4, 16, 32, 64, 80, 160};
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

__device__ __forceinline__ float4 ld_global_cg_v4_f32(const float4 *ptr) {
  float4 value;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(ptr));
  return value;
}

__global__ void probe_kernel_warm(const float4 *in, float *out,
                                  std::size_t n_vec, int iters) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

  float sum = 0.0f;
  for (int iter = 0; iter < iters; ++iter) {
    for (std::size_t i = tid; i < n_vec; i += stride) {
      const float4 value = ld_global_cg_v4_f32(in + i);
      sum += value.x + value.y + value.z + value.w;
    }
  }
  out[tid] = sum;
}

__global__ void probe_kernel_measure(const float4 *in, float *out,
                                     U64 *cycles_out, U64 *accesses_out,
                                     std::size_t n_vec, int iters) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

  float sum = 0.0f;
  U64 accesses = 0;
#ifndef GPGPU_SIM
  const U64 start = clock64();
#endif
  for (int iter = 0; iter < iters; ++iter) {
    for (std::size_t i = tid; i < n_vec; i += stride) {
      const float4 value = ld_global_cg_v4_f32(in + i);
      sum += value.x + value.y + value.z + value.w;
      ++accesses;
    }
  }
#ifndef GPGPU_SIM
  const U64 stop = clock64();
  cycles_out[tid] = stop - start;
#else
  cycles_out[tid] = 0;
#endif
  accesses_out[tid] = accesses;
  out[tid] = sum;
}

bool parse_size_arg(const char *value, std::size_t *out) {
  char *end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 0);
  if (value[0] == '\0' || end == nullptr || *end != '\0' || parsed == 0) {
    return false;
  }
  *out = static_cast<std::size_t>(parsed);
  return true;
}

bool parse_int_arg(const char *value, int *out) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 0);
  if (value[0] == '\0' || end == nullptr || *end != '\0' || parsed <= 0) {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

bool parse_nonneg_int_arg(const char *value, int *out) {
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 0);
  if (value[0] == '\0' || end == nullptr || *end != '\0' || parsed < 0) {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  bool use_custom_size = false;
  std::size_t custom_working_set_bytes = 0;
  int iters = kDefaultIters;
  int max_blocks = kDefaultMaxBlocks;
  int warm_iters = kDefaultWarmIters;

  assert(argc == 2);
  unsigned blk_acq = std::atoi(argv[1]);
  // if (argc > 1) {
  //   if (!parse_size_arg(argv[1], &custom_working_set_bytes)) {
  //     std::fprintf(stderr, "Invalid working_set_bytes: %s\n", argv[1]);
  //     return EXIT_FAILURE;
  //   }
  //   use_custom_size = true;
  // }
  // if (argc > 2 && !parse_int_arg(argv[2], &iters)) {
  //   std::fprintf(stderr, "Invalid iters: %s\n", argv[2]);
  //   return EXIT_FAILURE;
  // }
  // if (argc > 3 && !parse_int_arg(argv[3], &max_blocks)) {
  //   std::fprintf(stderr, "Invalid max_blocks: %s\n", argv[3]);
  //   return EXIT_FAILURE;
  // }
  // if (argc > 4 && !parse_nonneg_int_arg(argv[4], &warm_iters)) {
  //   std::fprintf(stderr, "Invalid warm_iters: %s\n", argv[4]);
  //   return EXIT_FAILURE;
  // }

  std::printf("case,blocks,threads_per_block,total_threads,working_set_bytes,"
              "warm_iters,iters,passes,requested_read_bytes,min_event_ms,"
              "median_event_ms,avg_event_ms,max_event_ms,"
              "min_app_read_GiB_per_s,median_app_read_GiB_per_s,"
              "avg_app_read_GiB_per_s,max_app_read_GiB_per_s,"
              "min_avg_cycles_per_access,median_avg_cycles_per_access,"
              "avg_avg_cycles_per_access,max_avg_cycles_per_access,checksum\n");

  const int case_count =
      use_custom_size ? 1
                      : static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
  for (int case_idx = 0; case_idx < case_count; ++case_idx) {
    const char *case_name = use_custom_size ? "custom" : kCases[case_idx].name;
    std::size_t working_set_bytes = use_custom_size
                                        ? custom_working_set_bytes
                                        : kCases[case_idx].working_set_bytes;

    std::size_t n_vec = working_set_bytes / sizeof(float4);
    if (n_vec < static_cast<std::size_t>(kThreadsPerBlock)) {
      n_vec = static_cast<std::size_t>(kThreadsPerBlock);
    }
    working_set_bytes = n_vec * sizeof(float4);

    std::vector<float> host(working_set_bytes / sizeof(float));
    for (std::size_t i = 0; i < host.size(); ++i) {
      host[i] = static_cast<float>((i % 251) + 1) * 0.5f;
    }

    float *dev_in = nullptr;
    float *dev_out = nullptr;
    U64 *dev_cycles = nullptr;
    U64 *dev_accesses = nullptr;
    CHECK_CUDA(cudaMalloc(&dev_in, working_set_bytes));
    CHECK_CUDA(cudaMemcpy(dev_in, host.data(), working_set_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&dev_out, static_cast<std::size_t>(max_blocks) *
                                        kThreadsPerBlock * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_cycles, static_cast<std::size_t>(max_blocks) *
                                           kThreadsPerBlock * sizeof(U64)));
    CHECK_CUDA(cudaMalloc(&dev_accesses, static_cast<std::size_t>(max_blocks) *
                                             kThreadsPerBlock * sizeof(U64)));

#if !defined(PERF_RUN)
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

    bool hit_blk_num{false};
    for (int blocks : kSweepBlocks) {
      if (blocks > max_blocks) {
        assert(false);
      }
      if (blocks != blk_acq) {
        continue;
      }
      hit_blk_num = true;

      const int total_threads = blocks * kThreadsPerBlock;
      const double requested_read_bytes =
          static_cast<double>(working_set_bytes) * static_cast<double>(iters);
      std::vector<double> event_ms_samples;
      std::vector<double> gib_per_s_samples;
      std::vector<double> avg_cycles_samples;
      event_ms_samples.reserve(cal_kernels::kNativePasses);
      gib_per_s_samples.reserve(cal_kernels::kNativePasses);
      avg_cycles_samples.reserve(cal_kernels::kNativePasses);
      double checksum = 0.0;
      for (int pass = 0; pass < cal_kernels::kNativePasses; ++pass) {
#ifndef GPGPU_SIM
        double event_ms = 0.0;
#endif
        CHECK_CUDA(cudaMemset(dev_out, 0,
                              static_cast<std::size_t>(total_threads) *
                                  sizeof(float)));
        CHECK_CUDA(
            cudaMemset(dev_cycles, 0,
                       static_cast<std::size_t>(total_threads) * sizeof(U64)));
        CHECK_CUDA(
            cudaMemset(dev_accesses, 0,
                       static_cast<std::size_t>(total_threads) * sizeof(U64)));

        if (warm_iters > 0) {
          probe_kernel_warm<<<blocks, kThreadsPerBlock>>>(
              reinterpret_cast<const float4 *>(dev_in), dev_out, n_vec,
              warm_iters);
          CHECK_CUDA(cudaGetLastError());
          CHECK_CUDA(cudaDeviceSynchronize());
        }

#if !defined(PERF_RUN)
        if (warm_iters == 0) {
          CHECK_CUDA(
              cal_kernels::run_cold_l2_flush(dev_l2_flush, l2_flush_elements));
        }
#endif

#ifndef GPGPU_SIM
        CHECK_CUDA(cudaEventRecord(start));
#endif
        probe_kernel_measure<<<blocks, kThreadsPerBlock>>>(
            reinterpret_cast<const float4 *>(dev_in), dev_out, dev_cycles,
            dev_accesses, n_vec, iters);
        CHECK_CUDA(cudaGetLastError());
#ifndef GPGPU_SIM
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float event_ms_value = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&event_ms_value, start, stop));
        event_ms = static_cast<double>(event_ms_value);
#else
        CHECK_CUDA(cudaDeviceSynchronize());
#endif

        std::vector<float> partial(total_threads);
        std::vector<U64> cycles(total_threads);
        std::vector<U64> accesses(total_threads);
        CHECK_CUDA(
            cudaMemcpy(partial.data(), dev_out,
                       static_cast<std::size_t>(total_threads) * sizeof(float),
                       cudaMemcpyDeviceToHost));
        CHECK_CUDA(
            cudaMemcpy(cycles.data(), dev_cycles,
                       static_cast<std::size_t>(total_threads) * sizeof(U64),
                       cudaMemcpyDeviceToHost));
        CHECK_CUDA(
            cudaMemcpy(accesses.data(), dev_accesses,
                       static_cast<std::size_t>(total_threads) * sizeof(U64),
                       cudaMemcpyDeviceToHost));
        checksum = 0.0;
        for (float value : partial) {
          checksum += static_cast<double>(value);
        }
        double avg_cycles_per_access = 0.0;
        int active_count = 0;
        for (int tid = 0; tid < total_threads; ++tid) {
          if (accesses[tid] == 0) {
            continue;
          }
          avg_cycles_per_access += static_cast<double>(cycles[tid]) /
                                   static_cast<double>(accesses[tid]);
          ++active_count;
        }
        if (active_count > 0) {
          avg_cycles_per_access /= static_cast<double>(active_count);
        }
#ifndef GPGPU_SIM
        event_ms_samples.push_back(event_ms);
        gib_per_s_samples.push_back(requested_read_bytes / (event_ms * 1.0e-3) /
                                    static_cast<double>(1ULL << 30));
#endif
        avg_cycles_samples.push_back(avg_cycles_per_access);
      }

      assert(hit_blk_num);
#ifdef GPGPU_SIM
      std::printf("GPGPU-Sim Host >> "
                  "%s,%d,%d,%d,%zu,%d,%d,%d,%.0f,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/"
                  "A,N/A,N/A,N/A,N/A,%.4f\n",
                  case_name, blocks, kThreadsPerBlock, total_threads,
                  working_set_bytes, warm_iters, iters,
                  cal_kernels::kNativePasses, requested_read_bytes, checksum);
#else
      const auto event_stats = cal_kernels::summarize_samples(event_ms_samples);
      const auto gib_stats = cal_kernels::summarize_samples(gib_per_s_samples);
      const auto cycle_stats =
          cal_kernels::summarize_samples(avg_cycles_samples);
      std::printf(
          "%s,%d,%d,%d,%zu,%d,%d,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%."
          "4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
          case_name, blocks, kThreadsPerBlock, total_threads, working_set_bytes,
          warm_iters, iters, cal_kernels::kNativePasses, requested_read_bytes,
          event_stats.min, event_stats.median, event_stats.avg, event_stats.max,
          gib_stats.min, gib_stats.median, gib_stats.avg, gib_stats.max,
          cycle_stats.min, cycle_stats.median, cycle_stats.avg, cycle_stats.max,
          checksum);
#endif
    }

#ifndef GPGPU_SIM
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(start));
#endif
#if !defined(PERF_RUN)
    CHECK_CUDA(cudaFree(dev_l2_flush));
#endif
    CHECK_CUDA(cudaFree(dev_accesses));
    CHECK_CUDA(cudaFree(dev_cycles));
    CHECK_CUDA(cudaFree(dev_out));
    CHECK_CUDA(cudaFree(dev_in));
  }

  return EXIT_SUCCESS;
}
