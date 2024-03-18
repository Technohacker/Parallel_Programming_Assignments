#include "common.h"
#include "impl.h"

std::vector<path_segment_t> delta_step_single(adjacency_list &graph, int source);

std::unordered_map<node_t, std::vector<path_segment_t>> delta_step(adjacency_list &graph, std::vector<node_t> sources) {
  std::unordered_map<node_t, std::vector<path_segment_t>> paths;

  // No extra preparation needed for CPU
  for (node_t src : sources) {
    paths[src] = delta_step_single(graph, src);
  }

  return paths;
}

std::vector<path_segment_t> delta_step_single(adjacency_list &graph, int source) {
  // Start with all nodes at infinite distance
  std::vector<path_segment_t> paths(
      graph.size(), {.total_cost = INFINITE_DIST, .parent = INVALID_NODE});

  // Maintain a priority queue for dynamic bucketing
  weight_queue_t queue;
  queue.emplace(std::vector<node_with_cost>());

  // Process the source
  queue.push({
      .node = source,
      .total_cost = 0,
  });
  paths[source] = {
    .total_cost = 0,
    .parent = source,
  };

  while (true) {
    // Repeat until we can't find any more buckets
    int current_bucket_num = next_bucket_num(queue, DELTA);
    if (current_bucket_num == -1) {
      break;
    }

    // std::cout << "Bucket " << current_bucket_num << std::endl;

    // Repeat until we can't get a bucket for this number
    while (true) {
      std::vector<node_with_cost> bucket = get_bucket_for_number(queue, paths, current_bucket_num, DELTA);
      if (bucket.empty()) {
        break;
      }

      // Relax all outgoing edges for each node in this bucket
      for (auto source : bucket) {
        // std::cout << "\tNode " << source.node << ". Cost " << source.total_cost << std::endl;
        for (auto outgoing : graph[source.node]) {
          // std::cout << "\t\tNeighbour " << outgoing.dest << std::endl;
          int new_cost = source.total_cost + outgoing.weight;

          if (new_cost < paths[outgoing.dest].total_cost) {
            // std::cout << "\t\tShorter path found. Was: " << paths[outgoing.dest].total_cost << ". Now: " << new_cost << std::endl;

            // Update our paths list
            paths[outgoing.dest] = {
              .total_cost = new_cost,
              .parent = source.node,
            };

            // And queue it for bucketing
            queue.push({
              .node = outgoing.dest,
              .total_cost = new_cost,
            });
          }
        }
      }

      // std::cout << "\tQueue Length " << queue.size() << std::endl;
    }
  }

  return paths;
}