#pragma once

#include <cassert>
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

// CUDA Parameters
const size_t BLOCK_SIZE = 256;
size_t grid_size(size_t domain_size) {
    return (domain_size / BLOCK_SIZE) + 1;
}

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

    void copy_to_device(device_buffer_t<T> &device) {
        assert(this->num_elements == device.num_elements);
        CUDA_CHECK(cudaMemcpy(device.buf, this->buf, this->num_elements * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    __device__ T& operator[](int index) {
        assert(index < this->num_elements);
        return this->buf[index];
    }
    __device__ inline size_t size() {
        return this->num_elements;
    }
};

template<typename T>
class host_allocator : public allocator<T> {
public:
    static void mem_alloc(T** ptr, size_t num_elements) {
        CUDA_CHECK(cudaMallocHost(ptr, num_elements * sizeof(T)));
    }
    static void mem_free(T* ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
};

template<typename T>
class host_buffer_t : public buffer_t<T, host_allocator<T>> {
public:
    host_buffer_t(): buffer_t<T, host_allocator<T>>() {}
    host_buffer_t(size_t N): buffer_t<T, host_allocator<T>>(N) {}

    host_buffer_t(std::vector<T> vec) : buffer_t<T, host_allocator<T>>(vec.size()) {
        CUDA_CHECK(cudaMemcpy(this->buf, vec.data(), this->num_elements * sizeof(T), cudaMemcpyHostToHost));
    }

    __host__ T& operator[](int index) {
        assert(index < this->num_elements);
        return this->buf[index];
    }
    __host__ inline size_t size() {
        return this->num_elements;
    }
    
    void copy_to_device(device_buffer_t<T> &device) {
        assert(this->num_elements == device.num_elements);
        CUDA_CHECK(cudaMemcpy(device.buf, this->buf, this->num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_from_device(device_buffer_t<T> &device) {
        assert(this->num_elements == device.num_elements);
        CUDA_CHECK(cudaMemcpy(this->buf, device.buf, this->num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    std::vector<T> to_vector() {
        std::vector<T> vec(this->num_elements);
        CUDA_CHECK(cudaMemcpy(vec.data(), this->buf, vec.size() * sizeof(T), cudaMemcpyDefault));

        return vec;
    }
};

// Adjacency list in a form more amenable for GPU manipulation
class gpu_adjacency_list_t {
private:
  size_t N;
  device_buffer_t<size_t> node_starts;
  device_buffer_t<node_t> node_neighbours;
  device_buffer_t<weight_t> out_edge_weights;

public:
  gpu_adjacency_list_t(adjacency_list_t &graph) {
    this->N = graph.size();
    // ======= Node Starts
    host_buffer_t<size_t> h_node_starts(N + 1);
    // Compute all start positions
    h_node_starts[0] = 0;
    for (size_t i = 0; i < N; i += 1) {
      h_node_starts[i + 1] = h_node_starts[i] + graph[i].size();
    }
    // Keep the number of neighbours handy for later
    size_t num_neighbours = h_node_starts[N];

    // ======= Node Neighbours
    host_buffer_t<node_t> h_node_neighbours(num_neighbours);
    host_buffer_t<weight_t> h_out_edge_weights(num_neighbours);

    int index = 0;
    for (auto src : graph) {
      for (auto neighbour : src) {
        h_node_neighbours[index] = neighbour.dest;
        h_out_edge_weights[index] = neighbour.weight;
        index += 1;
      }
    }

    this->node_starts.reallocate(N + 1);
    this->node_neighbours.reallocate(num_neighbours);
    this->out_edge_weights.reallocate(num_neighbours);

    h_node_starts.copy_to_device(this->node_starts);
    h_node_neighbours.copy_to_device(this->node_neighbours);
    h_out_edge_weights.copy_to_device(this->out_edge_weights);

    h_node_starts.release();
    h_node_neighbours.release();
    h_out_edge_weights.release();
  }

  void release() {
    this->node_starts.release();
    this->node_neighbours.release();
    this->out_edge_weights.release();
  }

  __device__ size_t num_nodes() {
    return this->N;
  }

  __device__ size_t get_neighbours(node_t source, node_t **neighours_start, weight_t** out_weights_start) {
    int start_index = this->node_starts[source];
    int end_index = this->node_starts[source + 1];

    *neighours_start = &this->node_neighbours[start_index];
    *out_weights_start = &this->out_edge_weights[start_index];

    return end_index - start_index;
  }
};
