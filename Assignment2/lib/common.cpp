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

int next_bucket_num(weight_queue_t &queue, int delta) {
  if (queue.empty()) {
    return -1;
  } else {
    return bucket_num(queue.top().total_cost, delta);
  }
}

std::vector<node_with_cost_t>
get_bucket_for_number(weight_queue_t &queue, std::function<weight_t (node_t&)> get_distance,
                      int current_bucket_num, int delta) {
  std::vector<node_with_cost_t> bucket;

  while (!queue.empty()) {
    node_with_cost_t pending_elem = queue.top();
    // Skip this node if we've got a better cost already
    if (pending_elem.total_cost > get_distance(pending_elem.node)) {
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

void timer::start() {
  m_begin = Time::now();
}

void timer::end() {
  m_end = Time::now();
}

double timer::elapsed() {
  return (std::chrono::duration_cast<std::chrono::milliseconds>((Time::now() - m_begin)).count()) / 1000.0;
}