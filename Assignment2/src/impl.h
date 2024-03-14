#pragma once

#include <cstddef>
#include <vector>

struct dest_with_weight {
    int dest;
    int weight;
};

typedef std::vector<std::vector<dest_with_weight>> adjacency_list;