#pragma once

#include <iostream>
#include <csignal>
#include <cstdlib>
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

#include "../../common.h"
#include "device_buffer.h"

// Macro for checking CUDA Errors
#define CUDA_CHECK(ans)                                                        \
  { cuda_check((ans), __FILE__, __LINE__); }

inline void cuda_check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file
              << ":" << line << std::endl;
    raise(SIGTRAP);
    std::exit(code);
  }
}

#define CUDA_CHECK_DEV(call)                                \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}

__host__ void sort_range_gpu(std::vector<element_t> &data, size_t range_start, size_t range_end);
