#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "common.h"
#include "impl.h"

#include "gpu.cuh"

// Host-side logic for a single source
std::vector<path_segment_t>
delta_step_single(adjacency_list_t &host_graph,
                  gpu_adjacency_list_t &device_graph, node_t source);
// GPU kernel for neighbour relaxation
__global__ void relax_nodes(gpu_adjacency_list_t graph,
                            unified_buffer_t<node_with_cost_t> bucket,
                            unified_buffer_t<path_segment_t> paths,
                            unified_buffer_t<size_t> device_return_starts,
                            unified_buffer_t<node_with_cost_t> device_return
);

std::unordered_map<node_t, std::vector<path_segment_t>>
delta_step(adjacency_list_t &graph, std::vector<node_t> sources) {
  // Check if there are indeed any CUDA devices
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "No CUDA Devices found!" << std::endl;
    std::exit(1);
  }

  // Prepare a GPU version of the input graph
  gpu_adjacency_list_t device_graph(graph);

  std::unordered_map<node_t, std::vector<path_segment_t>> paths;
  for (node_t src : sources) {
    paths[src] = delta_step_single(graph, device_graph, src);
  }

  device_graph.release();
  return paths;
}

template <typename T>
__global__ void set_all_but(unified_buffer_t<T> buf, T val, size_t except,
                            T replacement) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= buf.size()) {
    return;
  }

  buf[index] = (index == except) ? replacement : val;
}

std::vector<path_segment_t>
delta_step_single(adjacency_list_t &host_graph,
                  gpu_adjacency_list_t &device_graph, node_t source) {
  size_t N = host_graph.size();

  // Start with all nodes at infinite distance
  unified_buffer_t<path_segment_t> paths(N);
  const size_t block_size = 256;
  size_t grid_size = (N / block_size) + 1;

  set_all_but<<<grid_size, block_size>>>(paths,
                        {.total_cost = INFINITE_DIST, .parent = INVALID_NODE},
                        source, {.total_cost = 0, .parent = source});
  CUDA_CHECK(cudaDeviceSynchronize());
  auto get_distance = [&paths](auto node) { return paths[node].total_cost; };

  // Maintain a priority queue for dynamic bucketing
  weight_queue_t queue;

  // Process the source
  queue.push({
      .node = source,
      .total_cost = 0,
  });

  while (true) {
    // Repeat until we can't find any more buckets
    int current_bucket_num = next_bucket_num(queue, DELTA);
    if (current_bucket_num == -1) {
      break;
    }

    // Repeat until we can't get a bucket for this number
    while (true) {
      std::vector<node_with_cost_t> bucket =
          get_bucket_for_number(queue, get_distance, current_bucket_num, DELTA);
      if (bucket.empty()) {
        break;
      }

      // Allocate memory for the device-visible bucket
      unified_buffer_t<node_with_cost_t> device_bucket(bucket);

      // Prepare the return buffer
      unified_buffer_t<size_t> device_return_starts(bucket.size() + 1);

      device_return_starts[0] = 0;
      for (size_t i = 0; i < bucket.size(); i += 1) {
        device_return_starts[i + 1] = device_return_starts[i] + host_graph[bucket[i].node].size(); 
      }

      size_t num_neighbours = device_return_starts[bucket.size()];
      unified_buffer_t<node_with_cost_t> device_return(num_neighbours);

      // Relax all outgoing edges for each node in this bucket
      size_t grid_size = (bucket.size() / block_size) + 1;
      relax_nodes<<<grid_size, block_size>>>(
        device_graph,
        device_bucket,
        paths,
        device_return_starts,
        device_return
      );
      CUDA_CHECK(cudaDeviceSynchronize());

      // Read the return buffer to insert into the queue
      for (size_t i = 0; i < bucket.size(); i += 1) {
        for (size_t j = device_return_starts[i]; j < device_return_starts[i + 1]; j += 1) {
          auto elem = device_return[j];
          if (elem.node == INVALID_NODE) {
            continue;
          }

          if (elem.total_cost < paths[elem.node].total_cost) {
            // Update our paths list
            paths[elem.node] = {
                .total_cost = elem.total_cost,
                .parent = bucket[i].node,
            };

            // And queue it for bucketing
            queue.push(elem);
          }

        }
      }

      device_bucket.release();
      device_return_starts.release();
      device_return.release();
    }
  }

  auto cpu_paths = paths.to_vector();
  paths.release();

  return cpu_paths;
}

__global__ void relax_nodes(gpu_adjacency_list_t graph,
                            unified_buffer_t<node_with_cost_t> bucket,
                            unified_buffer_t<path_segment_t> paths,
                            unified_buffer_t<size_t> device_return_starts,
                            unified_buffer_t<node_with_cost_t> device_return
) {
  size_t source_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (source_index >= bucket.size()) {
    return;
  }

  node_with_cost_t source = bucket[source_index];

  node_t *neighbours;
  weight_t *out_weights;
  int num_neighbours = graph.get_neighbours(source.node, &neighbours, &out_weights);

  for (size_t i = 0; i < num_neighbours; i += 1) {
    node_t neighbour = neighbours[i];
    int new_cost = source.total_cost + out_weights[i];

    if (new_cost < paths[neighbour].total_cost) {
      size_t neighbour_pos = device_return_starts[source_index] + i;
      device_return[neighbour_pos] = {
        .node = neighbour,
        .total_cost = new_cost,
      };

      // Distance update is done by the CPU
      // Store the update
      // paths[outgoing.dest] = {
      //     .total_cost = new_cost,
      //     .parent = source.node,
      // };
    }
  }
}
