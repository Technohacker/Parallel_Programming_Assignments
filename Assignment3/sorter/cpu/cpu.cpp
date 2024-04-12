#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "cpu.h"

std::vector<element_t> sort_cpu(vector_view<element_t> data) {
    // Form views for each block of the data
    std::vector<vector_view<element_t>> block_views;
    for (size_t i = 0; i < data.size(); i += CPU_BLOCK_SIZE) {
        range_t block_range = {
            .start = i,
            .end = std::min(i + CPU_BLOCK_SIZE, data.size())
        };

        block_views.push_back(data.slice(block_range));
    }

    // Sort each block in parallel
    #pragma omp taskloop
    for (size_t i = 0; i < block_views.size(); i++) {
        std::sort(block_views[i].begin(), block_views[i].end());
    }

    #pragma omp taskwait

    // Then merge all the blocks into a single sorted range
    vector_view<vector_view<element_t>> block_views_view(block_views);

    return merge_multiple(block_views_view);
}

// Merges multiple sorted ranges of data into a single sorted range in parallel
std::vector<element_t> merge_multiple(vector_view<vector_view<element_t>> views) {
    size_t total_size = 0;
    for (size_t i = 0; i < views.size(); i++) {
        total_size += views[i].size();
    }

    if (total_size == 0) {
        return std::vector<element_t>();
    } else if (views.size() == 1) {
        // If there is only one range, return it
        return std::vector<element_t>(views[0].begin(), views[0].end());
    } else if (views.size() == 2) {
        // If there are two ranges, merge them
        std::vector<element_t> result(total_size);
        merge_pair(views[0], views[1], result);
        return result;
    } else {
        // Otherwise, split the views into two halves and merge them recursively
        size_t mid = views.size() / 2;

        vector_view<vector_view<element_t>> left_views = views.slice({0, mid});
        vector_view<vector_view<element_t>> right_views = views.slice({mid, views.size()});

        std::vector<element_t> left_result, right_result;

        #pragma omp task shared(left_result)
        left_result = std::move(merge_multiple(left_views));
        #pragma omp task shared(right_result)
        right_result = std::move(merge_multiple(right_views));

        #pragma omp taskwait

        std::vector<element_t> result(total_size);
        merge_pair(left_result, right_result, result);
        return result;
    }
}

// Merges two sorted ranges of data into a single sorted range in parallel
void merge_pair(vector_view<element_t> A, vector_view<element_t> B, vector_view<element_t> result) {
    // Ensure that the result is not from the same container as either input
    assert(!A.from_same_container(result));
    assert(!B.from_same_container(result));
    // And that the result is the correct size
    assert(A.size() + B.size() == result.size());

    vector_view<element_t> &smaller = A.size() < B.size() ? A : B;
    vector_view<element_t> &larger = A.size() < B.size() ? B : A;

    // If either range is smaller than the block size, merge them using std::merge
    if (smaller.size() <= CPU_BLOCK_SIZE) {
        std::merge(
            smaller.begin(), smaller.end(),
            larger.begin(), larger.end(),
            result.begin()
        );
        return;
    }

    // Split the larger range into two halves
    size_t range_mid = larger.size() / 2;

    // Find the mid point in the smaller range
    auto mid_iter = std::lower_bound(smaller.begin(), smaller.end(), larger[range_mid]);
    size_t mid = mid_iter - smaller.begin();

    // Create the views of both halves of the two ranges
    vector_view<element_t> smaller_left = smaller.slice({0, mid});
    vector_view<element_t> smaller_right = smaller.slice({mid, smaller.size()});
    vector_view<element_t> larger_left = larger.slice({0, range_mid});
    vector_view<element_t> larger_right = larger.slice({range_mid, larger.size()});

    // Recursively merge the two halves of the two ranges
    vector_view<element_t> result_left = result.slice({0, mid + range_mid});
    vector_view<element_t> result_right = result.slice({mid + range_mid, result.size()});

    #pragma omp task
    merge_pair(smaller_left, larger_left, result_left);
    #pragma omp task
    merge_pair(smaller_right, larger_right, result_right);

    #pragma omp taskwait
}
