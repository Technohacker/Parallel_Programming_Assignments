#pragma once

#include <iostream>
#include <csignal>
#include <cstdlib>
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

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

#include "../../common.h"
#include "device_buffer.h"

__host__ void sort_blocks_gpu(std::vector<element_t> &data, size_t num_elements_per_block, size_t block_start, size_t block_end);