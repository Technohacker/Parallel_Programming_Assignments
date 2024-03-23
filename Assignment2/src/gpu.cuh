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
            CUDA_CHECK(cudaFree(this->buf));
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

// Adjacency list in a form more amenable for GPU manipulation
class gpu_adjacency_list_t {
private:
  unified_buffer_t<size_t> node_starts;
  unified_buffer_t<node_t> node_neighbours;
  unified_buffer_t<weight_t> out_edge_weights;

public:
  gpu_adjacency_list_t(adjacency_list_t &graph) {
    size_t N = graph.size();
    // ======= Node Starts
    this->node_starts.reallocate(N + 1);

    // Compute all start positions
    this->node_starts[0] = 0;
    for (size_t i = 0; i < N; i += 1) {
      this->node_starts[i + 1] = this->node_starts[i] + graph[i].size();
    }
    // Keep the number of neighbours handy for later
    size_t num_neighbours = this->node_starts[N];

    // ======= Node Neighbours
    this->node_neighbours.reallocate(num_neighbours);
    this->out_edge_weights.reallocate(num_neighbours);

    int index = 0;
    for (auto src : graph) {
      for (auto neighbour : src) {
        this->node_neighbours[index] = neighbour.dest;
        this->out_edge_weights[index] = neighbour.weight;
        index += 1;
      }
    }
  }

  void release() {
    this->node_starts.release();
    this->node_neighbours.release();
    this->out_edge_weights.release();
  }

  __device__ size_t get_neighbours(node_t source, node_t **neighours_start, weight_t** out_weights_start) {
    int start_index = this->node_starts[source];
    int end_index = this->node_starts[source + 1];

    *neighours_start = &this->node_neighbours[start_index];
    *out_weights_start = &this->out_edge_weights[start_index];

    return end_index - start_index;
  }
};
