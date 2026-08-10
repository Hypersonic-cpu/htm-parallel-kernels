// mini_nvlink_allreduce.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
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

__global__ void allreduce_peer_kernel(float **srcs, float *dst, int n_gpu,
                                      std::size_t n_elem) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elem)
    return;

  float acc = 0.0f;

#pragma unroll
  for (int g = 0; g < MAX_GPUS; ++g) {
    if (g < n_gpu) {
      // If g != current GPU, this is a peer memory load.
      // With NVLink P2P enabled, this should exercise NVLink.
      acc += srcs[g][i];
    }
  }

  dst[i] = acc;
}

static float input_value(int gpu, std::size_t i) {
  // Small integer-like float, exactly representable for small i.
  // GPU0: 10 + i%17
  // GPU1: 20 + i%17
  // GPU2: 30 + i%17
  // GPU3: 40 + i%17
  return static_cast<float>((gpu + 1) * 10 + static_cast<int>(i % 17));
}

static float expected_value(int n_gpu, std::size_t i) {
  float acc = 0.0f;
  for (int g = 0; g < n_gpu; ++g) {
    acc += input_value(g, i);
  }
  return acc;
}

static std::size_t elem_count_for_case(const char *case_name) {
  if (std::strcmp(case_name, "test") == 0) {
    return 4096;
  }
  // Weak-scaling profiles: this input size is per GPU and is independent of
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
#ifdef GPGPU_SIM
  std::size_t n_elem = elem_count_for_case("test");
#else
  std::size_t n_elem = elem_count_for_case("approx_l2");
#endif

  if (argc >= 2) {
    n_gpu = std::atoi(argv[1]);
  }
  if (argc >= 3) {
    const std::size_t prof_size = elem_count_for_case(argv[2]);
    if (prof_size != 0) {
      n_elem = prof_size;
      use_test_case = (std::strcmp(argv[2], "test") == 0);
    } else {
      n_elem = std::strtoull(argv[2], nullptr, 10);
    }
  }

  if (!(n_gpu == 1 || n_gpu == 2 || n_gpu == 4)) {
    fprintf(stderr, "usage: %s [1|2|4 gpus] [n_elem]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (n_gpu > MAX_GPUS) {
    fprintf(stderr, "ERROR: n_gpu exceeds MAX_GPUS=%d\n", MAX_GPUS);
    return EXIT_FAILURE;
  }
  if (n_elem == 0) {
    fprintf(stderr, "ERROR: n_elem must be non-zero\n");
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

  fprintf(stderr, "mini_nvlink_allreduce: n_gpu=%d n_elem=%zu\n", n_gpu,
          n_elem);

  // Enable direct GPU-GPU access.
  if (n_gpu > 1) {
    enable_full_peer_access(n_gpu);
  }

  std::vector<float *> d_in(n_gpu, nullptr);
  std::vector<float *> d_out(n_gpu, nullptr);
  std::vector<float **> d_src_table(n_gpu, nullptr);
  std::vector<float *> d_l2_flush(n_gpu, nullptr);
  const std::size_t l2_flush_elements = cal_kernels::kColdL2FlushBytes / sizeof(float);

  const std::size_t bytes = n_elem * sizeof(float);

  // Allocate and initialize one input array per GPU.
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));

    CHECK_CUDA(cudaMalloc(&d_in[g], bytes));
    CHECK_CUDA(cudaMalloc(&d_out[g], bytes));
    CHECK_CUDA(cudaMalloc(&d_src_table[g], n_gpu * sizeof(float *)));
    CHECK_CUDA(cudaMalloc(&d_l2_flush[g], l2_flush_elements * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_l2_flush[g], 0, l2_flush_elements * sizeof(float)));

    std::vector<float> h_in(n_elem);
    for (std::size_t i = 0; i < n_elem; ++i) {
      h_in[i] = input_value(g, i);
    }

    CHECK_CUDA(cudaMemcpy(d_in[g], h_in.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out[g], 0, bytes));
  }

  // Copy the list of peer input pointers to every GPU.
  //
  // Because CUDA UVA is used, the pointer values d_in[0], d_in[1], ...
  // can be passed to kernels on peer GPUs once peer access is enabled.
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    CHECK_CUDA(cudaMemcpy(d_src_table[g], d_in.data(), n_gpu * sizeof(float *),
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
  int blocks = static_cast<int>((n_elem + kThreads - 1) / kThreads);

  // Launch one kernel per GPU.
  //
  // Each GPU writes its own d_out[g].
  // Each GPU reads all d_in[0..n_gpu-1].
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));

    allreduce_peer_kernel<<<blocks, kThreads>>>(d_src_table[g], d_out[g], n_gpu,
                                                n_elem);

    CHECK_CUDA(cudaGetLastError());
  }

  // Synchronize all GPUs.
  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // Verify: every GPU should contain the same all-reduced result.
  int total_errors = 0;

  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));

    std::vector<float> h_out(n_elem);
    CHECK_CUDA(
        cudaMemcpy(h_out.data(), d_out[g], bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (std::size_t i = 0; i < n_elem; ++i) {
      float ref = expected_value(n_gpu, i);
      float got = h_out[i];

      if (std::fabs(got - ref) > 1e-5f) {
        if (errors < 8) {
          fprintf(stderr, "mismatch gpu=%d i=%zu got=%f expected=%f\n", g, i,
                  got, ref);
        }
        ++errors;
      }
    }

    total_errors += errors;

    fprintf(
        stderr,
        "GPU%d check: %s, errors=%d, sample out[0]=%f out[1]=%f out[16]=%f\n",
        g, errors == 0 ? "PASS" : "FAIL", errors, h_out[0],
        n_elem > 1 ? h_out[1] : -1.0f, n_elem > 16 ? h_out[16] : -1.0f);
  }

  for (int g = 0; g < n_gpu; ++g) {
    CHECK_CUDA(cudaSetDevice(g));
    CHECK_CUDA(cudaFree(d_src_table[g]));
    CHECK_CUDA(cudaFree(d_out[g]));
    CHECK_CUDA(cudaFree(d_in[g]));
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
