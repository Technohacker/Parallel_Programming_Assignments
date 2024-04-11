#include <algorithm>
#include <cstddef>

#include "cpu.h"

// Performs a recursive merge sort on the given range of the data using OpenMP tasks
void sort_cpu(vector_view<element_t> data) {
    // Base case: if the range is below block size, sort it using std::sort
    if (data.size() <= CPU_BLOCK_SIZE) {
        std::sort(data.begin(), data.end());
        return;
    }

    // Recursively sort the two halves of the range
    size_t range_mid = data.size() / 2;

    vector_view<element_t> view_left = data.slice({0, range_mid});
    vector_view<element_t> view_right = data.slice({range_mid, data.size()});

    #pragma omp task shared(data)
    sort_cpu(view_left);
    #pragma omp task shared(data)
    sort_cpu(view_right);

    #pragma omp taskwait

    // TODO: Merge the two sorted halves
    // merge_ranges_cpu(data, {range_left, range_right});
}
