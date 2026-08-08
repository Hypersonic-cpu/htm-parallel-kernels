// mini_nvlink_allgather.cu
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include <vector>

#include "l2_flush.h"

#ifndef MAX_GPUS
#define MAX_GPUS 4
#endif

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t st__ = (call);                                                 \
    if (st__ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(st__));                                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

static void CHECK_PEER(cudaError_t st) {
  if (st == cudaErrorPeerAccessAlreadyEnabled) {
    return;
  }
  if (st != cudaSuccess) {
    fprintf(stderr, "cudaDeviceEnablePeerAccess failed: %s\n",
            cudaGetErrorString(st));
    std::exit(EXIT_FAILURE);
  }
}

__device__ __forceinline__ float make_value_device(int src_gpu, std::size_t i) {
  // Deterministic, exactly representable small float.
  // GPU0 chunk: 1000 + i % 251
  // GPU1 chunk: 2000 + i % 251
  // GPU2 chunk: 3000 + i % 251
  // GPU3 chunk: 4000 + i % 251
  return static_cast<float>((src_gpu + 1) * 1000 + static_cast<int>(i % 251));
}

static float make_value_host(int src_gpu, std::size_t i) {
  return static_cast<float>((src_gpu + 1) * 1000 + static_cast<int>(i % 251));
}

static std::size_t chunk_elems_for_case(const char *case_name) {
  if (std::strcmp(case_name, "test") == 0) {
    return 4096;
  }
  // Weak-scaling profiles: this chunk size is per GPU and is independent of
  // the requested GPU count.
  if (std::strcmp(case_name, "small") == 0) {
    return 1u << 18;  // 1 MiB per GPU.
  }
  if (std::strcmp(case_name, "medium") == 0) {
    return 1u << 20;  // 4 MiB per GPU.
  }
  if (std::strcmp(case_name, "large") == 0) {
    return 1u << 22;  // 16 MiB per GPU.
  }
  // Preserve the established profile names for existing callers.
  if (std::strcmp(case_name, "le_l2") == 0) {
    return 1u << 18;
  }
  if (std::strcmp(case_name, "approx_l2") == 0) {
    return 1u << 20;
  }
  if (std::strcmp(case_name, "gt_l2") == 0) {
    return 1u << 22;
  }
  return 0;
}

__global__ void allgather_peer_kernel(float **srcs, float *dst, int n_gpu,
                                      std::size_t chunk_elems,
                                      std::size_t source_slot_count,
                                      std::size_t repeat) {
  std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t total = static_cast<std::size_t>(n_gpu) * chunk_elems;

  if (linear >= total)
    return;

  const int src_gpu = static_cast<int>(linear / chunk_elems);
  const std::size_t offset = linear % chunk_elems;

  // Each iteration reads a distinct rotating source slot.  All iterations
  // are independent: sources are read-only and every GPU writes only its
  // own output buffer, so a final cudaDeviceSynchronize after this kernel
  // completes every iteration without adding a relaunch per iteration.
  for (std::size_t iteration = 0; iteration < repeat; ++iteration) {
    const std::size_t slot = iteration % source_slot_count;
    float **slot_srcs = srcs + slot * n_gpu;

    // dst layout:
    //
    //   [ GPU0 chunk ][ GPU1 chunk ][ GPU2 chunk ][ GPU3 chunk ]
    //
    // If src_gpu is not the current GPU, this read is a peer load.
    dst[linear] = slot_srcs[src_gpu][offset];
  }
}

static void enable_full_peer_access(int n_gpu) {
  for (int i = 0; i < n_gpu; ++i) {
    CHECK_CUDA(cudaSetDevice(i));

    for (int j = 0; j < n_gpu; ++j) {
      if (i == j)
        continue;

      int can_access = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));

      if (!can_access) {
        fprintf(stderr,
                "ERROR: device %d cannot access peer device %d. "
                "No CUDA P2P path is available.\n",
                i, j);
        std::exit(EXIT_FAILURE);
      }

      CHECK_PEER(cudaDeviceEnablePeerAccess(j, 0));
    }
  }
}

int main(int argc, char **argv) {
  int n_gpu = 1;
  bool use_test_case = false;
  std::size_t repeat = 1;

#ifdef GPGPU_SIM
  // Small default for simulator.
  std::size_t chunk_elems = chunk_elems_for_case("test");
#else
  // Native default.
  std::size_t chunk_elems = chunk_elems_for_case("approx_l2");
#endif

  if (argc >= 2) {
    n_gpu = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    const std::size_t prof_size = chunk_elems_for_case(argv[2]);
    if (prof_size != 0) {
      chunk_elems = prof_size;
      use_test_case = (std::strcmp(argv[2], "test") == 0);
    } else {
      chunk_elems = std::strtoull(argv[2], nullptr, 10);
    }
  }
  if (argc >= 4) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(argv[3], &end, 10);
    if (errno == ERANGE || end == argv[3] || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
      fprintf(stderr, "ERROR: repeat must be an integer >= 1\n");
      return EXIT_FAILURE;
    }
    repeat = static_cast<std::size_t>(parsed);
  }

  if (!(n_gpu == 1 || n_gpu == 2 || n_gpu == 4)) {
    fprintf(stderr, "usage: %s [1|2|4 gpus] [chunk_elems|test|small|medium|large] [repeat]\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  if (n_gpu > MAX_GPUS) {
    fprintf(stderr, "ERROR: n_gpu=%d exceeds MAX_GPUS=%d\n", n_gpu, MAX_GPUS);
    return EXIT_FAILURE;
  }

  if (chunk_elems == 0) {
    fprintf(stderr, "ERROR: chunk_elems must be non-zero\n");
    return EXIT_FAILURE;
  }
  if (use_test_case && n_gpu < 2) {
    fprintf(stderr, "ERROR: test case requires multi-GPU (n_gpu >= 2)\n");
    return EXIT_FAILURE;
  }

  int dev_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&dev_count));

  if (dev_count < n_gpu) {
    fprintf(stderr, "ERROR: requested %d GPUs, but only %d visible\n", n_gpu,
            dev_count);
    return EXIT_FAILURE;
  }

  const std::size_t chunk_bytes = chunk_elems * sizeof(float);
  const std::size_t out_elems = static_cast<std::size_t>(n_gpu) * chunk_elems;
  const std::size_t out_bytes = out_elems * sizeof(float);

  std::size_t remote_read_bytes = static_cast<std::size_t>(n_gpu) *
                                  static_cast<std::size_t>(n_gpu - 1) *
                                  chunk_bytes;

  if (remote_read_bytes != 0 &&
      repeat > std::numeric_limits<std::size_t>::max() / remote_read_bytes) {
    fprintf(stderr, "ERROR: repeat makes expected traffic size overflow\n");
    return EXIT_FAILURE;
  }

  // Use distinct source allocations for a rotating working set larger than
  // each GPU's L2.  The device pointer table contains every slot up front;
  // the measured loop only selects a row, so it never performs a host-to-
  // device pointer-table update between iterations.
  std::size_t l2_cache_bytes = 0;
  for (int g = 0; g < n_gpu; ++g) {
    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, g));
    if (static_cast<std::size_t>(props.l2CacheSize) > l2_cache_bytes) {
      l2_cache_bytes = static_cast<std::size_t>(props.l2CacheSize);
    }
  }
  std::size_t source_slot_count = l2_cache_bytes / chunk_bytes + 1;
  if (source_slot_count < 2) {
    source_slot_count = 2;
  }
  const std::size_t rotating_source_bytes_per_gpu =
      source_slot_count * chunk_bytes;
  const std::size_t total_remote_read_bytes = remote_read_bytes * repeat;

  fprintf(
      stderr,
      "mini_nvlink_allgather: n_gpu=%d chunk_elems=%zu "
      "chunk_bytes=%zu out_bytes_per_gpu=%zu repeat=%zu "
      "l2_cache_bytes=%zu source_slots=%zu "
      "rotating_source_bytes_per_gpu=%zu "
      "expected_remote_read_bytes_per_iteration=%zu "
      "expected_remote_read_bytes_total=%zu\n",
      n_gpu, chunk_elems, chunk_bytes, out_bytes, repeat, l2_cache_bytes,
      source_slot_count, rotating_source_bytes_per_gpu, remote_read_bytes,
      total_remote_read_bytes);

  if (n_gpu > 1) {
    enable_full_peer_access(n_gpu);
  }

  std::vector<std::vector<float *>> d_in(
      n_gpu, std::vector<float *>(source_slot_count, nullptr));
  std::vector<float *> d_in_storage(n_gpu, nullptr);
  std::vector<float *> d_out(n_gpu, nullptr);
  std::vector<float **> d_src_table(n_gpu, nullptr);
  std::vector<float *> d_l2_flush(n_gpu, nullptr);
  const std::size_t l2_flush_elements = cal_kernels::kColdL2FlushBytes / sizeof(float);

  // Allocate the rotating input slots and one full gather output per GPU.
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));

    CHECK_CUDA(cudaMalloc(&d_out[g], out_bytes));
    CHECK_CUDA(cudaMalloc(&d_src_table[g],
                          source_slot_count * n_gpu * sizeof(float *)));
    CHECK_CUDA(cudaMalloc(&d_l2_flush[g], l2_flush_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_l2_flush[g], 0, l2_flush_elements * sizeof(float)));

    std::vector<float> h_in(chunk_elems);
    for (std::size_t i = 0; i < chunk_elems; ++i) {
      h_in[i] = make_value_host(g, i);
    }

    CHECK_CUDA(cudaMalloc(&d_in_storage[g], rotating_source_bytes_per_gpu));
    std::vector<float> h_source(source_slot_count * chunk_elems);
    for (std::size_t slot = 0; slot < source_slot_count; ++slot) {
      std::memcpy(h_source.data() + slot * chunk_elems, h_in.data(),
                  chunk_bytes);
      d_in[g][slot] = d_in_storage[g] + slot * chunk_elems;
    }
    CHECK_CUDA(cudaMemcpy(d_in_storage[g], h_source.data(),
                          rotating_source_bytes_per_gpu,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out[g], 0, out_bytes));
  }

  // Copy the table of UVA peer pointers to every GPU.
  //
  // d_src_table[g] lives on GPU g.
  // It contains pointer values:
  //
  //   d_in[0][slot], d_in[1][slot], d_in[2][slot], d_in[3][slot]
  //
  // With peer access enabled, GPU g can dereference peer pointers.  All
  // slots are copied once before the measured loop; later iterations only
  // select a different row in this device-resident table.
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));

    std::vector<float *> h_src_table(source_slot_count * n_gpu, nullptr);
    for (std::size_t slot = 0; slot < source_slot_count; ++slot) {
      for (int src = 0; src < n_gpu; ++src) {
        h_src_table[slot * n_gpu + src] = d_in[src][slot];
      }
    }
    CHECK_CUDA(cudaMemcpy(d_src_table[g], h_src_table.data(),
                          h_src_table.size() * sizeof(float *),
                          cudaMemcpyHostToDevice));
  }

  // Flush each GPU before the measured kernel launch.
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    CHECK_CUDA(cal_kernels::run_cold_l2_flush(d_l2_flush[g], l2_flush_elements));
  }
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((out_elems + kThreads - 1) / kThreads);

  // Repeat the actual gather region within the measured kernel.  A final
  // synchronization after each GPU has launched ensures that every device
  // has completed every iteration before verification.  Distinct rotating
  // source slots keep later iterations from merely rereading the same
  // cache-hot copies, while the one-launch structure keeps profiling traces
  // compact.
  const auto gather_start = std::chrono::steady_clock::now();
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));

    allgather_peer_kernel<<<blocks, kThreads>>>(
        d_src_table[g], d_out[g], n_gpu, chunk_elems, source_slot_count,
        repeat);

    CHECK_CUDA(cudaGetLastError());
  }
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  const auto gather_end = std::chrono::steady_clock::now();
  const double gather_duration_ms =
      std::chrono::duration<double, std::milli>(gather_end - gather_start)
          .count();
  fprintf(stderr,
          "gather_repeat_duration_ms=%.3f repeat=%zu "
          "source_slots=%zu\n",
          gather_duration_ms, repeat, source_slot_count);

  // Verify output on every GPU.
  int total_errors = 0;

  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));

    std::vector<float> h_out(out_elems);
    CHECK_CUDA(
        cudaMemcpy(h_out.data(), d_out[g], out_bytes, cudaMemcpyDeviceToHost));

    int errors = 0;

    for (int src = 0; src < n_gpu; ++src) {
      for (std::size_t i = 0; i < chunk_elems; ++i) {
        std::size_t idx = static_cast<std::size_t>(src) * chunk_elems + i;
        float got = h_out[idx];
        float ref = make_value_host(src, i);

        if (std::fabs(got - ref) > 1e-5f) {
          if (errors < 8) {
            fprintf(stderr,
                    "mismatch dst_gpu=%d src_gpu=%d i=%zu got=%f expected=%f\n",
                    g, src, i, got, ref);
          }
          ++errors;
        }
      }
    }

    total_errors += errors;

    fprintf(stderr,
            "GPU%d check: %s errors=%d samples: "
            "out[0]=%f out[chunk]=%f out[last]=%f\n",
            g, errors == 0 ? "PASS" : "FAIL", errors, h_out[0],
            n_gpu >= 2 ? h_out[chunk_elems] : -1.0f, h_out[out_elems - 1]);
  }

  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    CHECK_CUDA(cudaFree(d_src_table[g]));
    CHECK_CUDA(cudaFree(d_out[g]));
    CHECK_CUDA(cudaFree(d_in_storage[g]));
    CHECK_CUDA(cudaFree(d_l2_flush[g]));
  }

  if (total_errors == 0) {
    fprintf(stderr, "ALL PASS\n");
  } else {
    fprintf(stderr, "FAIL total_errors=%d\n", total_errors);
  }

#ifdef GPGPU_SIM
  std::fflush(stdout);
  std::fflush(stderr);
  _Exit(total_errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
#else
  return total_errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
