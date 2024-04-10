#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "device_buffer.h"
#include "gpu.h"

// Returns the largest power of two that is less than or equal to the input
__host__ __device__ size_t largest_power_of_two(size_t n) {
    size_t power = 1;
    while (power * 2 <= n) {
        power *= 2;
    }
    return power;
}

// Performs a bitonic sort on consecutive blocks of elements
// Each block of elements is sorted by a single thread block
// The direction of the sort for a block is determined by the direction bit of the block index
__global__ void bitonic_sort_blockwise(device_buffer_t<element_t> data, size_t block_direction_bit_pos) {
    assert(blockDim.x == GPU_BLOCK_SIZE / 2);

    // Find the start index of the block
    const size_t block_start = blockIdx.x * blockDim.x;
    // And the ID of this thread within the block
    const size_t worker_id = threadIdx.x;

    // Calculate the direction for this block
    // The masked bit of the thread block index determines the direction of the sort
    bool block_ascending = (blockIdx.x & (1 << block_direction_bit_pos)) == 0;

    // Load the block of elements into shared memory
    __shared__ element_t shared_data[GPU_BLOCK_SIZE / 2];
    shared_data[worker_id] = data[block_start + worker_id];
    __syncthreads();

    // Run an outer loop to sort a range of elements at a time
    // For each iteration, sort_range number of elements are sorted
    for (size_t sort_range = 2; sort_range <= GPU_BLOCK_SIZE / 2; sort_range *= 2) {
        // Find the sort group we are working on
        const size_t sort_group_num = worker_id / sort_range;
        // And the ID of this thread within the sort group
        size_t sort_group_id = worker_id % sort_range;

        // Every other group of elements is sorted in the opposite direction
        const bool ascending = sort_group_num % 2 == 0 ? block_ascending : !block_ascending;

        // Run an inner loop to compare and swap elements
        for (size_t compare_range = sort_range; compare_range > 1; compare_range /= 2) {
            // Find the comparison group we are working on
            const size_t compare_group_num = sort_group_id / compare_range;
            // And the ID of this thread within the comparison group
            const size_t compare_group_id = sort_group_id % compare_range;

            // Calculate the index of the element this thread is assigned to
            const size_t element_id = sort_group_num * sort_range + compare_group_num * compare_range + compare_group_id;
            // And the index of the element to compare with using bitwise XOR
            const size_t compare_id = element_id ^ (compare_range / 2);

            // Make sure the compare_id is not behind the element_id
            if (element_id < compare_id) {
                // Load the elements to compare
                const element_t element = shared_data[element_id];
                const element_t compare = shared_data[compare_id];

                // Compare and swap the elements if necessary
                if ((element > compare) == ascending) {
                    shared_data[element_id] = compare;
                    shared_data[compare_id] = element;
                }
            }

            __syncthreads();
        }

        __syncthreads();
    }

    // Store the sorted block of elements back into global memory
    data[block_start + worker_id] = shared_data[worker_id];
}

// __host__ void bitonic_sort_driver(device_buffer_t<element_t> &data, size_t range_start, size_t num_blocks) {
//     // If there is only one block of elements, launch the kernel to sort the block
//     if (num_blocks == 1) {
//         bitonic_sort_kernel<<<1, GPU_BLOCK_SIZE>>>(data, range_start, 0);
//         return;
//     }

//     // For larger numbers of blocks, start with a stride of 2 and go up to the number of blocks
//     for (size_t stride = 2; stride <= num_blocks; stride *= 2) {
//         // At each stride, perform a bitonic sort on the blocks of elements
//         bitonic_sort_kernel<<<num_blocks, GPU_BLOCK_SIZE>>>(data, range_start, stride - 1);

//         // Wait for the kernel to finish
//         CUDA_CHECK(cudaPeekAtLastError());
//         CUDA_CHECK(cudaDeviceSynchronize());

//         // Then launch a kernel to merge the blocks of elements
//         bitonic_merge_kernel<<<num_blocks, GPU_BLOCK_SIZE>>>(data, range_start, stride);

//     }
// }

// Merges the sorted blocks of data on the CPU recursively using OpenMP tasks
void do_final_merge(std::vector<element_t> &data, std::vector<size_t> &sorted_block_starts, size_t start, size_t end) {
    // If there is only one block of data, there is nothing to merge
    if (end - start <= 1) {
        return;
    }

    // Otherwise, split the blocks of data in half and merge them recursively
    size_t mid = (end - start) / 2;

    #pragma omp task shared(data, sorted_block_starts)
    do_final_merge(data, sorted_block_starts, start, start + mid);
    #pragma omp task shared(data, sorted_block_starts)
    do_final_merge(data, sorted_block_starts, start + mid, end);
    #pragma omp taskwait

    std::inplace_merge(
        data.begin() + sorted_block_starts[start],
        data.begin() + sorted_block_starts[start + mid],
        data.begin() + sorted_block_starts[end]);
}

__host__ void sort_range_gpu(std::vector<element_t> &data, size_t range_start, size_t range_end) {
    size_t num_elements_total = range_end - range_start;

    // Exit early if there are no elements to sort
    if (num_elements_total == 0) {
        return;
    }

    // Check if there are indeed any CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cout << "No CUDA Devices found!" << std::endl;
        std::exit(1);
    }

    // Find the largest power-of-2 number of elements that can fit on 80% of the GPU memory
    size_t total_gpu_memory;
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total_gpu_memory));

    size_t safe_gpu_memory = total_gpu_memory * 0.8;
    size_t max_elements = largest_power_of_two(safe_gpu_memory / sizeof(element_t));

    // Allocate memory on the GPU for the blocks of data assigned to the GPU
    device_buffer_t<element_t> device_data;
    // Also keep track of where the sorted blocks start
    std::vector<size_t> sorted_block_starts;

    size_t buffer_size = std::min(max_elements, num_elements_total);
    for (size_t i = range_start; i < range_end; i += buffer_size) {
        // If there are fewer elements than the buffer size, reduce the buffer size to the next power of two
        if (i + buffer_size > range_end) {
            buffer_size = largest_power_of_two(range_end - i);
        }

        // Reallocate the device buffer if the number of elements is different
        if (device_data.size() != buffer_size) {
            device_data.reallocate(buffer_size);
        }

        // Copy the blocks of data to the GPU
        device_data.copy_to_device(&data[i], buffer_size);

        // Launch the kernel to sort the blocks of data
        std::cout << "GPU Block Sort Start" << std::endl;
        bitonic_sort_blockwise<<<buffer_size / (GPU_BLOCK_SIZE / 2), GPU_BLOCK_SIZE / 2>>>(device_data, 0);
        // Wait for the kernel to finish
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "GPU Block Sort End" << std::endl;

        // Copy the sorted data back to the CPU
        device_data.copy_from_device(&data[i], buffer_size);
        // And mark the start of the sorted block
        sorted_block_starts.push_back(i);
    }

    // Do the final merge on the CPU
    do_final_merge(data, sorted_block_starts, 0, sorted_block_starts.size());
}
