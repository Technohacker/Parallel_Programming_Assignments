#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "cpu.h"

void sort_cpu(vector_view<element_t> &data, vector_view<element_t> &result) {
    // Form views for each block of the data
    std::cout << "CPU Block view start" << std::endl;
    std::vector<vector_view<element_t>> block_views;
    for (size_t i = 0; i < data.size(); i += CPU_BLOCK_SIZE) {
        range_t block_range = {
            .start = i,
            .end = std::min(i + CPU_BLOCK_SIZE, data.size())
        };

        block_views.push_back(data.slice(block_range));
    }
    std::cout << "CPU Block view end" << std::endl;

    // Sort each block in parallel
    std::cout << "CPU Block Sort start" << std::endl;
    #pragma omp taskloop shared(block_views)
    for (size_t i = 0; i < block_views.size(); i++) {
        std::sort(block_views[i].begin(), block_views[i].end());
    }
    #pragma omp taskwait
    std::cout << "CPU Block Sort end" << std::endl;

    // Then merge all the blocks into a single sorted range
    std::cout << "CPU Merge start" << std::endl;
    #pragma omp task shared(result)
    {
        vector_view<vector_view<element_t>> block_views_view(block_views);
        vector_view<element_t> result_view(result);

        merge_multiple(block_views_view, result_view);
    }
    #pragma omp taskwait
    std::cout << "CPU Merge end" << std::endl;
}

// Merges multiple sorted ranges of data into a single sorted range in parallel
void merge_multiple(vector_view<vector_view<element_t>> &views, vector_view<element_t> &result) {
    // Ensure that the result is not from the same container as any input
    // and that the views are contiguous
    size_t total_size = 0;
    for (size_t i = 0; i < views.size(); i++) {
        assert(!views[i].from_same_container(result));
        if (i > 0) {
            assert(views[i - 1].is_contiguous(views[i]));
        }

        total_size += views[i].size();
    }
    assert(total_size == result.size());

    if (total_size == 0) {
        // If there is no data, there's nothing to do
    } else if (views.size() == 1) {
        // If there is only one range, copy it into the result
        std::copy(views[0].begin(), views[0].end(), result.begin());
    } else if (views.size() == 2) {
        // If there are two ranges, merge them
        merge_pair(views[0], views[1], result);
    } else {
        // Otherwise, split the views into two halves and merge them recursively
        size_t mid = views.size() / 2;

        vector_view<vector_view<element_t>> left_views = views.slice({0, mid});
        vector_view<vector_view<element_t>> right_views = views.slice({mid, views.size()});

        vector_view<element_t> left_contiguous_view, right_contiguous_view;

        // Construct the consolidated views for the two halves of the result
        if (left_views.size() > 0) {
            left_contiguous_view = left_views[0];
            for (size_t i = 1; i < left_views.size(); i++) {
                left_contiguous_view = left_contiguous_view.merge(left_views[i]);
            }
        }

        if (right_views.size() > 0) {
            right_contiguous_view = right_views[0];
            for (size_t i = 1; i < right_views.size(); i++) {
                right_contiguous_view = right_contiguous_view.merge(right_views[i]);
            }
        }

        assert(left_contiguous_view.is_contiguous(right_contiguous_view));
        assert(left_contiguous_view.size() + right_contiguous_view.size() == result.size());

        // Create the views for the two halves of the result
        vector_view<element_t> left_result = result.slice({0, left_contiguous_view.size()});
        vector_view<element_t> right_result = result.slice({left_contiguous_view.size(), result.size()});

        #pragma omp task shared(left_result, left_views)
        {
            merge_multiple(left_views, left_result);
        }
        #pragma omp task shared(right_result, right_views)
        {
            merge_multiple(right_views, right_result);
        }
        #pragma omp taskwait

        // Copy the result of the halves into the contiguous views
        std::copy(left_result.begin(), left_result.end(), left_contiguous_view.begin());
        std::copy(right_result.begin(), right_result.end(), right_contiguous_view.begin());

        // Then merge the two halves of the result
        merge_pair(left_contiguous_view, right_contiguous_view, result);
    }
}

// Merges two sorted ranges of data into a single sorted range in parallel
void merge_pair(vector_view<element_t> &A, vector_view<element_t> &B, vector_view<element_t> &result) {
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
    {
        merge_pair(smaller_left, larger_left, result_left);
    }
    #pragma omp task
    {
        merge_pair(smaller_right, larger_right, result_right);
    }
    #pragma omp taskwait
}
