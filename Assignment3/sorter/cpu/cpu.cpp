#include <algorithm>
#include <cstddef>
#include <vector>

#include "cpu.h"

// Performs a recursive merge sort on the given range of the data using OpenMP tasks
void sort_range_cpu(std::vector<int> &data, range_t range) {
    size_t num_elements = range.end - range.start;

    // Base case: if the range is below block size, sort it using std::sort
    if (num_elements <= CPU_BLOCK_SIZE) {
        std::sort(data.begin() + range.start, data.begin() + range.end);
        return;
    }

    // Recursively sort the two halves of the range
    size_t range_mid = range.start + num_elements / 2;
    std::vector<range_t> left_ranges, right_ranges;

    range_t range_left = {
        .start = range.start,
        .end = range_mid,
    };
    range_t range_right = {
        .start = range_mid,
        .end = range.end,
    };

    #pragma omp task shared(data)
    sort_range_cpu(data, range_left);
    #pragma omp task shared(data)
    sort_range_cpu(data, range_right);

    #pragma omp taskwait

    // TODO: Merge the two sorted halves
    // merge_ranges_cpu(data, {range_left, range_right});
}
