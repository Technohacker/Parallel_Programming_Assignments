#pragma once
#include <vector>

#include "../../common.h"

void sort_blocks_cpu(std::vector<element_t> &data, size_t num_elements_per_block, size_t block_start, size_t block_end);
void merge_sort_blocks_cpu(std::vector<element_t> &data, size_t num_elements_per_block, size_t block_start, size_t block_end);
