#pragma once

#include <cstddef>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>

#include "gpu.h"

template<typename T>
class allocator {
public:
    static void mem_alloc(T** ptr, size_t num_elements);
    static void mem_free(T* ptr);
};

// General purpose buffer
template<typename T, typename A>
class buffer_t {
public:
    T *buf = nullptr;
    size_t num_elements = 0;

    buffer_t() {
        this->reallocate(0);
    }
    buffer_t(size_t N) {
        this->reallocate(N);
    }

    void reallocate(size_t N) {
        if (this->buf != nullptr) {
            A::mem_free(this->buf);
            this->buf = nullptr;
            this->num_elements = 0;
        }

        if (N > 0) {
            A::mem_alloc(&this->buf, N * sizeof(T));
            this->num_elements = N;
        }
    }
    void release() {
        this->reallocate(0);
    }
};

template<typename T>
class device_allocator : public allocator<T> {
public:
    static void mem_alloc(T** ptr, size_t num_elements) {
        CUDA_CHECK(cudaMalloc(ptr, num_elements * sizeof(T)));
    }
    static void mem_free(T* ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
};

template<typename T>
class device_buffer_t : public buffer_t<T, device_allocator<T>> {
public:
    device_buffer_t(): buffer_t<T, device_allocator<T>>() {}
    device_buffer_t(size_t N): buffer_t<T, device_allocator<T>>(N) {}

    void copy_to_device(T* host_buf, size_t num_elements) {
        assert(this->num_elements == num_elements);
        CUDA_CHECK(cudaMemcpy(this->buf, host_buf, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_from_device(T* host_buf, size_t num_elements) {
        assert(this->num_elements == num_elements);
        CUDA_CHECK(cudaMemcpy(host_buf, this->buf, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    __device__ T& operator[](int index) {
        assert(index < this->num_elements);
        return this->buf[index];
    }
    __host__ __device__ inline size_t size() {
        return this->num_elements;
    }
};
