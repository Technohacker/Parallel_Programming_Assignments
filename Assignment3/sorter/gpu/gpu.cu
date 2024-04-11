#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "../cpu/cpu.h"

#include "device_buffer.h"
#include "gpu.h"

// Performs a bitonic sort of consecutive blocks of elements on the GPU
// Each block is sorted by a thread block
// Every sort_direction_stride blocks are sorted in the opposite direction
__global__ void bitonic_sort_blockwise(device_buffer_t<element_t> data, size_t sort_direction_stride = 1, bool only_merge = false) {
    assert(blockDim.x == GPU_BLOCK_SIZE / 2);

    // Find the start index of the block
    const size_t block_start = blockIdx.x * GPU_BLOCK_SIZE;
    // And the ID of this thread within the thread block
    const size_t worker_id = threadIdx.x;

    // Calculate the direction for this block
    bool block_ascending = (blockIdx.x / sort_direction_stride) % 2 == 0;

    // Load the block pair into shared memory
    __shared__ element_t shared_data[GPU_BLOCK_SIZE];
    shared_data[worker_id                     ] = data[block_start + worker_id                     ];
    shared_data[worker_id + GPU_BLOCK_SIZE / 2] = data[block_start + worker_id + GPU_BLOCK_SIZE / 2];
    __syncthreads();

    // If only_merge is set, we can start from the last iteration of the outer loop
    size_t sort_range_start = only_merge ? GPU_BLOCK_SIZE / 2 : 2;

    // Run an outer loop to sort a range of elements at a time
    // For each iteration, sort_range number of elements are sorted
    for (size_t sort_range = sort_range_start; sort_range <= GPU_BLOCK_SIZE; sort_range *= 2) {
        // Get the direction for this thread and iteration
        // The lower half of the threads sort one way, the upper half sort the other way
        bool ascending = (worker_id / (sort_range / 2) % 2) != block_ascending;

        // Run an inner loop to compare and swap elements
        for (size_t compare_range = sort_range / 2; compare_range > 0; compare_range /= 2) {
            // Find the element for this thread
            size_t element_id = 2 * worker_id - (worker_id & (compare_range - 1));
            // And the element to compare with
            size_t compare_id = element_id + compare_range;

            // Load the elements to compare
            element_t element = shared_data[element_id];
            element_t compare = shared_data[compare_id];

            // Compare and swap the elements if necessary
            if ((element <= compare) != ascending) {
                shared_data[element_id] = compare;
                shared_data[compare_id] = element;
            }

            // Wait for all comparisons to finish
            __syncthreads();
        }
    }

    // Wait for all threads to finish sorting
    __syncthreads();
    // Store the sorted blocks of elements back into global memory
    data[block_start + worker_id                     ] = shared_data[worker_id                     ];
    data[block_start + worker_id + GPU_BLOCK_SIZE / 2] = shared_data[worker_id + GPU_BLOCK_SIZE / 2];
}

// Performs a single bitonic merge stage on the GPU
// Each thread block handles a GPU_BLOCK_SIZE number of elements
__global__ void bitonic_merge_global(device_buffer_t<element_t> data, size_t sort_range, size_t compare_range) {
    assert(blockDim.x == GPU_BLOCK_SIZE / 2);

    // Get the position of this thread within the array
    size_t global_pos = blockIdx.x * blockDim.x + threadIdx.x;
    // And the position relative to the half of the array it is in
    size_t half_pos = global_pos & (data.size() / 2 - 1);

    // Check whether the elements should be sorted in ascending or descending order
    bool ascending = half_pos & (sort_range / 2) ? false : true;

    // Find the element for this thread
    size_t element_id = 2 * global_pos - (global_pos & (compare_range - 1));
    // And the element to compare with
    size_t compare_id = element_id + compare_range;

    // Load the elements to compare
    element_t element = data[element_id];
    element_t compare = data[compare_id];

    // Compare and swap the elements if necessary
    if ((element <= compare) != ascending) {
        data[element_id] = compare;
        data[compare_id] = element;
    }
}

bool is_power_of_two(size_t n) {
    return (n & (n - 1)) == 0;
}

__host__ void bitonic_sort(device_buffer_t<element_t> &data) {
    // Calculate the number of blocks of elements
    size_t num_blocks = data.size() / GPU_BLOCK_SIZE;

    // Ensure that it's a power of two
    assert(is_power_of_two(num_blocks));

    // First sort blockwise
    // Each block is sorted for use in later stages
    bitonic_sort_blockwise<<<num_blocks, GPU_BLOCK_SIZE / 2>>>(data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // If there is only one block, there is nothing more to do. The first block is already in the right order
    if (num_blocks == 1) {
        return;
    }

    // For larger numbers of blocks, start with a sort_range of 2 blocks until the buffer size is reached
    for (size_t sort_range = 2 * GPU_BLOCK_SIZE; sort_range <= data.size(); sort_range *= 2) {
        // Keep track of the comparison range only upto half a block
        for (size_t compare_range = sort_range / 2; compare_range >= GPU_BLOCK_SIZE; compare_range /= 2) {
            // Launch the kernel to perform a bitonic merge
            bitonic_merge_global<<<num_blocks, GPU_BLOCK_SIZE / 2>>>(data, sort_range, compare_range);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Once we're here, we can call the blockwise sort again, but only for merging
        bitonic_sort_blockwise<<<num_blocks, GPU_BLOCK_SIZE / 2>>>(data, sort_range / GPU_BLOCK_SIZE, true);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Returns the largest power of two that is less than or equal to the input
size_t largest_power_of_two(size_t n) {
    size_t power = 1;
    while (power * 2 <= n) {
        power *= 2;
    }
    return power;
}

__host__ void sort_range_gpu(std::vector<element_t> &data, range_t range) {
    // Get the number of elements to sort
    size_t num_elements_total = range.end - range.start;

    // Exit early if there are none
    if (num_elements_total == 0) {
        return;
    }

    // Find the number of blocks. This may not be a power of two
    assert(num_elements_total % GPU_BLOCK_SIZE == 0);

    size_t num_blocks_total = num_elements_total / GPU_BLOCK_SIZE;
    assert(num_blocks_total > 0);

    // Check if there are indeed any CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cout << "No CUDA Devices found!" << std::endl;
        std::exit(1);
    }

    // Find the largest power-of-2 number of blocks that can be sorted on the GPU at a time
    size_t total_gpu_memory;
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total_gpu_memory));

    // 80% Threshold for safe GPU memory usage
    size_t safe_gpu_memory = 0.8 * total_gpu_memory;
    size_t max_blocks = largest_power_of_two(safe_gpu_memory / (GPU_BLOCK_SIZE * sizeof(element_t)));

    if (max_blocks == 0) {
        std::cout << "Not enough GPU memory to sort any data" << std::endl;
        std::exit(1);
    }

    // Allocate memory on the GPU for the blocks of data assigned to the GPU
    device_buffer_t<element_t> device_data;
    // Also keep track of the ranges of the sorted blocks
    std::vector<range_t> sorted_block_ranges;

    // Find the buffer size to use
    size_t buffer_blocks = std::min(num_blocks_total, max_blocks);
    for (size_t i = 0; i < num_blocks_total; i += buffer_blocks) {
        // If we will overshoot the end of the range, reduce the buffer size to the next power of two
        if (i + buffer_blocks > num_blocks_total) {
            buffer_blocks = largest_power_of_two(num_blocks_total - i);
        }

        // Reallocate the device buffer if the number of elements is different
        if (device_data.size() != buffer_blocks * GPU_BLOCK_SIZE) {
            device_data.reallocate(buffer_blocks * GPU_BLOCK_SIZE);
        }

        // Copy the blocks of data to the GPU
        device_data.copy_to_device(&data[range.start + i * GPU_BLOCK_SIZE], buffer_blocks * GPU_BLOCK_SIZE);

        // Launch the kernel to sort the blocks of data
        std::cout << "GPU Block Sort Start" << std::endl;
        bitonic_sort(device_data);
        std::cout << "GPU Block Sort End" << std::endl;

        // Copy the sorted data back to the CPU
        device_data.copy_from_device(&data[range.start + i * GPU_BLOCK_SIZE], buffer_blocks * GPU_BLOCK_SIZE);

        // And mark the range of this block
        sorted_block_ranges.push_back({
            .start = range.start + i * GPU_BLOCK_SIZE,
            .end = range.start + (i + buffer_blocks) * GPU_BLOCK_SIZE
        });
    }

    return;
}
