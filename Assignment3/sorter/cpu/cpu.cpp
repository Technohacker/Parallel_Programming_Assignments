#include <algorithm>
#include <cstddef>

#include "cpu.h"

void sort_blocks_cpu(std::vector<element_t> &data, size_t num_elements_per_block, size_t block_start, size_t block_end) {
    // First, sort each block individually
    #pragma omp taskloop shared(data)
    for (size_t i = block_start; i < block_end; i += 1) {
        std::sort(data.begin() + i * num_elements_per_block, data.begin() + (i + 1) * num_elements_per_block);
    }
    #pragma omp taskwait

    // Then merge the blocks
    merge_sort_blocks_cpu(data, num_elements_per_block, block_start, block_end);
}

// Merge sorts a list of blocks on the CPU using OpenMP tasks
void merge_sort_blocks_cpu(std::vector<element_t> &data, size_t num_elements_per_block, size_t block_start, size_t block_end) {
    // If we have only one block, we are done
    if (block_end - block_start <= 1) {
        return;
    }

    // Calculate the midpoint
    size_t block_mid = block_start + (block_end - block_start) / 2;

    // Recursively sort the two halves
    #pragma omp task shared(data)
    merge_sort_blocks_cpu(data, num_elements_per_block, block_start, block_mid);

    #pragma omp task shared(data)
    merge_sort_blocks_cpu(data, num_elements_per_block, block_mid, block_end);

    #pragma omp taskwait

    // Merge the two halves
    std::inplace_merge(data.begin() + block_start * num_elements_per_block,
                       data.begin() + block_mid * num_elements_per_block,
                       data.begin() + block_end * num_elements_per_block);
}