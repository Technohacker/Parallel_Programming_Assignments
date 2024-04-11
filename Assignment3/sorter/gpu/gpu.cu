#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "device_buffer.h"
#include "gpu.h"

// Performs a bitonic sort of consecutive blocks of elements on the GPU
// Each block is sorted by a thread block
// Every other block of elements is sorted in the opposite direction
__global__ void bitonic_sort_pairwise(device_buffer_t<element_t> data) {
    assert(blockDim.x == GPU_BLOCK_SIZE / 2);

    // Find the start index of the block
    const size_t block_start = blockIdx.x * GPU_BLOCK_SIZE;
    // And the ID of this thread within the thread block
    const size_t worker_id = threadIdx.x;

    // Calculate the direction for this pair
    bool pair_ascending = blockIdx.x % 2 == 0;

    // Load the block pair into shared memory
    __shared__ element_t shared_data[GPU_BLOCK_SIZE];
    shared_data[worker_id                     ] = data[block_start + worker_id                     ];
    shared_data[worker_id + GPU_BLOCK_SIZE / 2] = data[block_start + worker_id + GPU_BLOCK_SIZE / 2];
    __syncthreads();

    // Run an outer loop to sort a range of elements at a time
    // For each iteration, sort_range number of elements are sorted
    for (size_t sort_range = 2; sort_range <= GPU_BLOCK_SIZE; sort_range *= 2) {
        // Get the direction for this thread and iteration
        // The lower half of the threads sort one way, the upper half sort the other way
        bool ascending = (worker_id / (sort_range / 2) % 2) != pair_ascending;

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
            if ((element > compare) == ascending) {
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

    size_t global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    size_t comparatorI = global_comparatorI & (data.size() / 2 - 1);

    // Bitonic merge
    bool ascending = comparatorI & (sort_range / 2) ? false : true;

    size_t element_id = 2 * global_comparatorI - (global_comparatorI & (compare_range - 1));
    size_t compare_id = element_id + compare_range;

    // Load the elements to compare
    element_t element = data[element_id];
    element_t compare = data[compare_id];

    // Compare and swap the elements if necessary
    if ((element > compare) == ascending) {
        data[element_id] = compare;
        data[compare_id] = element;
    }
}

__host__ void bitonic_sort(device_buffer_t<element_t> &data) {
    // Calculate the number of blocks of elements
    size_t num_blocks = data.size() / GPU_BLOCK_SIZE;

    // First sort blockwise
    // Each block is sorted for use in later stages
    bitonic_sort_pairwise<<<num_blocks, GPU_BLOCK_SIZE / 2>>>(data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // If there is only one block, there is nothing more to do. The first block is already in the right order
    if (num_blocks == 1) {
        return;
    }

    // For larger numbers of blocks, start with a sort_range of 2 blocks until the buffer size is reached
    for (size_t sort_range = 2 * GPU_BLOCK_SIZE; sort_range <= data.size(); sort_range *= 2) {
        // Keep track of the comparison range
        for (size_t compare_range = sort_range / 2; compare_range > 0; compare_range /= 2) {
            // Launch the kernel to perform the bitonic merge
            bitonic_merge_global<<<num_blocks, GPU_BLOCK_SIZE / 2>>>(data, sort_range, compare_range);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
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

bool is_power_of_two(size_t n) {
    return (n & (n - 1)) == 0;
}

__host__ std::vector<size_t> sort_range_gpu(std::vector<element_t> &data, size_t range_start, size_t range_end) {
    // Get the number of elements to sort
    size_t num_elements_total = range_end - range_start;

    // Exit early if there are none
    if (num_elements_total == 0) {
        return std::vector<size_t>();
    }

    // Find the number of blocks, and check that it is a power of two greater than zero
    assert(num_elements_total % GPU_BLOCK_SIZE == 0);

    size_t num_blocks_total = num_elements_total / GPU_BLOCK_SIZE;
    assert(num_blocks_total > 0);
    assert(is_power_of_two(num_blocks_total));

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
    // Also keep track of where the sorted blocks start
    std::vector<size_t> sorted_block_starts;

    // Find the buffer size to use
    size_t buffer_blocks = std::min(num_blocks_total, max_blocks);
    for (size_t i = 0; i < num_blocks_total; i += buffer_blocks) {
        // Reallocate the device buffer if the number of elements is different
        if (device_data.size() != buffer_blocks * GPU_BLOCK_SIZE) {
            device_data.reallocate(buffer_blocks * GPU_BLOCK_SIZE);
        }

        // Copy the blocks of data to the GPU
        device_data.copy_to_device(&data[range_start + i * GPU_BLOCK_SIZE], buffer_blocks * GPU_BLOCK_SIZE);

        // Launch the kernel to sort the blocks of data
        std::cout << "GPU Block Sort Start" << std::endl;
        bitonic_sort(device_data);
        std::cout << "GPU Block Sort End" << std::endl;

        // Copy the sorted data back to the CPU
        device_data.copy_from_device(&data[range_start + i * GPU_BLOCK_SIZE], buffer_blocks * GPU_BLOCK_SIZE);
        // And mark the start index of the sorted block
        sorted_block_starts.push_back(range_start + i * GPU_BLOCK_SIZE);
    }

    return sorted_block_starts;
}
