#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <queue>
#include <vector>

// ===========================
// ========== Input ==========
// ===========================

typedef int node_t;
typedef int weight_t;

// Edge with an implicit source made clear from context
struct edge_t {
  node_t dest;
  weight_t weight;
};

// A Graph represented as an adjacency list
typedef std::vector<std::vector<edge_t>> adjacency_list_t;

// Values for special cases
const node_t INVALID_NODE = -1;
const weight_t INFINITE_DIST = 99999999;

// ===========================
// ========= Output ==========
// ===========================

// Represents a portion of a path for an implicit destination node
struct path_segment_t {
  weight_t total_cost;
  node_t parent;
};

// ===========================
// ======== Bucketing ========
// ===========================

// Value of delta used in both implementations
const int DELTA = 50;
const int B_INF = std::numeric_limits<int>::max();

// Represents a node with a total cost for use in bucketing
struct node_with_cost_t {
  node_t node;
  weight_t total_cost;
};

// Min-queue for bucketing
typedef std::priority_queue<node_with_cost_t, std::vector<node_with_cost_t>,
                            std::greater<node_with_cost_t>>
    weight_queue_t;
// Operators for bucketing
bool operator<(node_with_cost_t a, node_with_cost_t b);
bool operator>(node_with_cost_t a, node_with_cost_t b);
bool operator==(node_with_cost_t a, node_with_cost_t b);

// Computes the bucket number for a given total cost
inline int bucket_num(int total_cost, int delta) {
    return std::max(total_cost - 1, 0) / delta;
}

// Finds the bucket number of the top of the queue. -1 if no element remains
int next_bucket_num(weight_queue_t &queue, int delta);
// Constructs a bucket out of the topmost elements of the queue, all matching
// the same bucket number
//
// Discards elements that have a better total cost as stored in paths
std::vector<node_with_cost_t>
get_bucket_for_number(weight_queue_t &queue, std::function<weight_t (node_t&)> get_distance,
                      int current_bucket_num, int delta);

// ===========================
// ========= Timing ==========
// ===========================

using Time = std::chrono::steady_clock;
using double_sec = std::chrono::duration<double>;

class timer {
private:
    std::chrono::time_point<Time, double_sec> m_begin;
    std::chrono::time_point<Time, double_sec> m_end;
public:
    void start();
    void end();
    double elapsed();
};
