#include "common.h"

bool operator<(node_with_cost_t a, node_with_cost_t b) {
    return a.total_cost < b.total_cost;
}

bool operator>(node_with_cost_t a, node_with_cost_t b) {
    return a.total_cost > b.total_cost;
}

bool operator==(node_with_cost_t a, node_with_cost_t b) {
    return a.total_cost == b.total_cost;
}

// Computes the bucket number for a given total cost
inline int bucket_num(int total_cost, int delta) {
    return std::max(total_cost - 1, 0) / delta;
}

int next_bucket_num(weight_queue_t &queue, int delta) {
  if (queue.empty()) {
    return -1;
  } else {
    return bucket_num(queue.top().total_cost, delta);
  }
}

std::vector<node_with_cost_t>
get_bucket_for_number(weight_queue_t &queue, weight_t distances[],
                      int current_bucket_num, int delta) {
  std::vector<node_with_cost_t> bucket;

  while (!queue.empty()) {
    node_with_cost_t pending_elem = queue.top();
    // Skip this node if we've got a better cost already
    if (pending_elem.total_cost > distances[pending_elem.node]) {
      queue.pop();
      continue;
    }

    // Otherwise take the node if it's in the same bucket
    if (bucket_num(pending_elem.total_cost, delta) == current_bucket_num) {
      queue.pop();
      bucket.push_back(pending_elem);
    } else {
      break;
    }
  }

  return bucket;
}
