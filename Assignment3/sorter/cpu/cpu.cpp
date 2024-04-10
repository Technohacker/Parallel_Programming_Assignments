#include <algorithm>
#include <cstddef>
#include <vector>

#include "cpu.h"

// Performs a recursive merge sort on the given range of the data using OpenMP tasks
void sort_range_cpu(std::vector<int> &data, size_t range_start, size_t range_end) {
    // Base case: if the range is below block size, sort it using std::sort
    if (range_end - range_start <= CPU_BLOCK_SIZE) {
        std::sort(data.begin() + range_start, data.begin() + range_end);
        return;
    }

    // Recursively sort the two halves of the range
    size_t range_mid = (range_start + range_end) / 2;
    #pragma omp task shared(data)
    sort_range_cpu(data, range_start, range_mid);
    #pragma omp task shared(data)
    sort_range_cpu(data, range_mid, range_end);
    #pragma omp taskwait

    // Merge the two sorted halves
    std::inplace_merge(data.begin() + range_start, data.begin() + range_mid, data.begin() + range_end);
}