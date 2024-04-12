#pragma once

#include <cstddef>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>

#include "gpu.h"

// General purpose buffer
template<typename T>
class device_buffer_t {
protected:
    T *buf = nullptr;
    size_t num_elements = 0;

public:
    device_buffer_t() {
        this->reallocate(0);
    }
    device_buffer_t(size_t N) {
        this->reallocate(N);
    }

    void reallocate(size_t N) {
        if (this->buf != nullptr) {
            CUDA_CHECK(cudaFree(this->buf));
            this->buf = nullptr;

            this->num_elements = 0;
        }

        if (N > 0) {
            CUDA_CHECK(cudaMalloc(&this->buf, N * sizeof(T)));
            this->num_elements = N;
        }
    }
    void release() {
        this->reallocate(0);
    }

    void copy_to_device(T* host_buf, size_t num_elements) {
        assert(this->num_elements == num_elements);
        CUDA_CHECK(cudaMemcpy(this->buf, host_buf, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_from_device(T* host_buf, size_t num_elements) {
        assert(this->num_elements == num_elements);
        CUDA_CHECK(cudaMemcpy(host_buf, this->buf, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    __device__ T& operator[](size_t index) {
        if (index >= this->num_elements) {
            printf("Index %ld out of bounds [0, %ld)\n", index, this->num_elements);
            assert(false);
        }
        return this->buf[index];
    }
    __host__ __device__ inline size_t size() {
        return this->num_elements;
    }
};
