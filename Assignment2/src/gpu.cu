#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "common.h"
#include "impl.h"

#include "gpu.cuh"

// Computes the bucket number for a given total cost
__device__ inline int bucket_num_cuda(int total_cost, int delta) {
  return max(total_cost - 1, 0) / delta;
}

__global__ void initialize_buffers(gpu_adjacency_list_t d_graph,
                                   device_buffer_t<node_t> d_sources,
                                   device_buffer_t<int> d_settled,
                                   device_buffer_t<weight_t> d_total_costs,
                                   device_buffer_t<node_t> d_parents) {
  size_t self_node = blockIdx.x * blockDim.x + threadIdx.x;
  size_t self_source_index = blockIdx.y * blockDim.y + threadIdx.y;

  const size_t N_NODES = d_graph.num_nodes();
  const size_t N_SOURCES = d_sources.size();

  if (self_node >= N_NODES || self_source_index >= N_SOURCES) {
    return;
  }

  size_t source_node = d_sources[self_source_index];
  size_t source_offset = self_source_index * N_NODES;

  // All nodes start not settled
  d_settled[source_offset + self_node] = false;

  if (self_node == source_node) {
    d_total_costs[source_offset + self_node] = 0;
    d_parents[source_offset + self_node] = source_node;
  } else {
    d_total_costs[source_offset + self_node] = INFINITE_DIST;
    d_parents[source_offset + self_node] = INVALID_NODE;
  }
}

__global__ void find_min_buckets(gpu_adjacency_list_t d_graph,
                                 device_buffer_t<node_t> d_sources,
                                 device_buffer_t<int> d_settled,
                                 device_buffer_t<int> d_min_buckets,
                                 device_buffer_t<weight_t> d_total_costs) {
  size_t self_node = blockIdx.x * blockDim.x + threadIdx.x;
  size_t self_source_index = blockIdx.y * blockDim.y + threadIdx.y;

  const size_t N_NODES = d_graph.num_nodes();
  const size_t N_SOURCES = d_sources.size();

  if (self_node >= N_NODES || self_source_index >= N_SOURCES) {
    return;
  }

  size_t source_node = d_sources[self_source_index];
  size_t source_offset = self_source_index * N_NODES;

  // Settled nodes do not count towards min bucket
  if (d_settled[source_offset + self_node]) {
    return;
  }

  // Do an atomic min with this node's bucket number
  int self_bucket_num =
      bucket_num_cuda(d_total_costs[source_offset + self_node], DELTA);

  // Skip the atomic min if it's definitely larger
  if (d_min_buckets[self_source_index] < self_bucket_num) {
    return;
  }
  atomicMin(&d_min_buckets[self_source_index], self_bucket_num);
}

__global__ void phase_logic(gpu_adjacency_list_t d_graph,
                            device_buffer_t<node_t> d_sources,
                            device_buffer_t<int> d_settled,
                            device_buffer_t<int> d_min_buckets,
                            device_buffer_t<weight_t> d_total_costs,
                            device_buffer_t<node_t> d_parents) {
  size_t self_node = blockIdx.x * blockDim.x + threadIdx.x;
  size_t self_source_index = blockIdx.y * blockDim.y + threadIdx.y;

  const size_t N_NODES = d_graph.num_nodes();
  const size_t N_SOURCES = d_sources.size();

  if (self_node >= N_NODES || self_source_index >= N_SOURCES) {
    return;
  }

  // Skip this node if it has been deemed settled
  size_t source_offset = self_source_index * N_NODES;
  if (d_settled[source_offset + self_node]) {
    return;
  }

  // Compute all bucketing parameters
  int current_min_bucket = d_min_buckets[self_source_index];
  int self_bucket_num =
      bucket_num_cuda(d_total_costs[source_offset + self_node], DELTA);

  // Skip this node if it's not in the current bucket
  if (self_bucket_num != current_min_bucket) {
    return;
  }

  // Mark this node settled to begin with
  d_settled[source_offset + self_node] = true;

  // Get the total distance to this node
  weight_t self_total_cost = d_total_costs[source_offset + self_node];

  // Find this node's neighbours
  node_t *neighbours;
  weight_t *out_weights;
  int num_neighbours =
      d_graph.get_neighbours(self_node, &neighbours, &out_weights);

  for (size_t i = 0; i < num_neighbours; i += 1) {
    node_t neighbour = neighbours[i];
    weight_t new_cost = self_total_cost + out_weights[i];

    // Do an atomic min update on this neighbour's distance
    atomicMin(&d_total_costs[source_offset + neighbour], new_cost);

    // If our write was successful, update the parent and mark it not-settled
    if (d_total_costs[source_offset + neighbour] == new_cost) {
      d_settled[source_offset + neighbour] = false;

      // Update the parent only if the current parent has a larger total cost than us
      node_t current_parent = d_parents[source_offset + neighbour];
      if (current_parent == INVALID_NODE || d_total_costs[source_offset + current_parent] > self_total_cost) {
        d_parents[source_offset + neighbour] = self_node;
      }
    }
  }
}

std::unordered_map<node_t, std::vector<path_segment_t>>
delta_step(adjacency_list_t &graph, std::vector<node_t> sources, timer &t) {
  // Check if there are indeed any CUDA devices
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No CUDA Devices found!" << std::endl;
    std::exit(1);
  }

  // Prepare a GPU version of the input graph
  gpu_adjacency_list_t d_graph(graph);

  size_t num_nodes = graph.size();
  size_t num_nodes_across_sources = sources.size() * num_nodes;

  // Prepare all the buffers needed
  // All the sources to process for
  host_buffer_t<node_t> h_sources(sources);
  device_buffer_t<node_t> d_sources(h_sources.size());

  h_sources.copy_to_device(d_sources);

  // Total costs and parents for every graph node, per source
  host_buffer_t<weight_t> h_total_costs(num_nodes_across_sources);
  device_buffer_t<weight_t> d_total_costs(num_nodes_across_sources);
  host_buffer_t<node_t> h_parents(num_nodes_across_sources);
  device_buffer_t<node_t> d_parents(num_nodes_across_sources);

  // Whether each node has been settled
  device_buffer_t<int> d_settled(num_nodes_across_sources);

  // Current and next minimum bucket numbers for each source
  host_buffer_t<int> h_min_buckets(std::vector<int>(h_sources.size(), 0));
  host_buffer_t<int> B_INFS(std::vector<int>(h_sources.size(), B_INF));
  device_buffer_t<int> d_min_buckets(h_sources.size());

  // Start measuring time
  size_t grid_size_per_source = grid_size(num_nodes);
  dim3 grid_shape(grid_size_per_source, sources.size());
  dim3 block_shape(BLOCK_SIZE, 1);

  t.start();
  // Initialize all bucket numbers
  initialize_buffers<<<grid_shape, block_shape>>>(d_graph, d_sources, d_settled,
                                                  d_total_costs, d_parents);
  CUDA_CHECK(cudaDeviceSynchronize());

  h_min_buckets.copy_to_device(d_min_buckets);
  while (true) {
    // Run a phase
    phase_logic<<<grid_shape, block_shape>>>(
        d_graph, d_sources, d_settled, d_min_buckets, d_total_costs, d_parents);
    CUDA_CHECK(cudaDeviceSynchronize());

    B_INFS.copy_to_device(d_min_buckets);

    // Find the next minimum buckets
    find_min_buckets<<<grid_shape, block_shape>>>(d_graph, d_sources, d_settled,
                                                  d_min_buckets, d_total_costs);
    CUDA_CHECK(cudaDeviceSynchronize());

    h_min_buckets.copy_from_device(d_min_buckets);

    bool all_done = true;
    // std::cout << "New buckets: ";
    for (size_t i = 0; i < h_min_buckets.size(); i += 1) {
      // std::cout << h_min_buckets[i] << " ";

      if (h_min_buckets[i] < B_INF) {
        all_done = false;
      }
    }
    // std::cout << std::endl;
    if (all_done) {
      break;
    }
  }

  t.end();

  h_total_costs.copy_from_device(d_total_costs);
  h_parents.copy_from_device(d_parents);

  std::unordered_map<node_t, std::vector<path_segment_t>> paths;
  for (size_t i = 0; i < sources.size(); i += 1) {
    for (node_t n = 0; n < num_nodes; n += 1) {
      paths[sources[i]].push_back({
          .total_cost = h_total_costs[i * num_nodes + n],
          .parent = h_parents[i * num_nodes + n],
      });
    }
  }

  h_sources.release();
  d_graph.release();
  d_graph.release();
  d_sources.release();
  d_total_costs.release();
  d_parents.release();

  return paths;
}
