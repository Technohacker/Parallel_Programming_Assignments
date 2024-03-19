#pragma once

#include <unordered_map>
#include <vector>

#include "common.h"

// Each element in the returned vector contains the reverse-edge
// that connects the indexed node to its parent in the shortest path
std::unordered_map<node_t, std::vector<path_segment_t>> delta_step(adjacency_list_t &graph, std::vector<node_t> sources);