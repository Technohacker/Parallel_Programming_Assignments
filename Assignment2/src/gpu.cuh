#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

#include <assert.h>
#include <signal.h>

// Macro for checking CUDA Errors
#define CUDA_CHECK(ans)                                                        \
  { cuda_check((ans), __FILE__, __LINE__); }

inline void cuda_check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "cuda_check: %s @ %s:%d\n", cudaGetErrorString(code), file,
            line);
    raise(SIGTRAP);
    std::exit(code);
  }
}

// General purpose unified buffer
template<typename T>
class unified_buffer_t {
private:
    T *buf = nullptr;
    size_t num_elements = 0;

public:
    unified_buffer_t() {
        this->reallocate(0);
    }
    unified_buffer_t(size_t N) {
        this->reallocate(N);
    }
    unified_buffer_t(std::vector<T> vec) : unified_buffer_t(vec.size()) {
        CUDA_CHECK(cudaMemcpy(this->buf, vec.data(), this->num_elements * sizeof(T), cudaMemcpyHostToHost));
    }

    void reallocate(size_t N) {
        if (this->buf != nullptr) {
            std::cout << "Test " << this->buf << std::endl;
            CUDA_CHECK(cudaFreeHost(this->buf));
            this->buf = nullptr;
            this->num_elements = 0;
        }

        if (N > 0) {
            CUDA_CHECK(cudaMallocManaged(&this->buf, N * sizeof(T)));
            this->num_elements = N;
        }
    }
    void release() {
        this->reallocate(0);
    }

    std::vector<T> to_vector() {
        std::vector<T> vec(this->num_elements);
        CUDA_CHECK(cudaMemcpy(vec.data(), this->buf, vec.size() * sizeof(T), cudaMemcpyDefault));

        return vec;
    }

    __host__ __device__ T& operator[](int index) {
        assert(index < this->num_elements);
        return this->buf[index];
    }
    __host__ __device__ inline size_t size() {
        return this->num_elements;
    }
};
