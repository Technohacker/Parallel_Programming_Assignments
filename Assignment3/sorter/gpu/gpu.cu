#include <algorithm>
#include <cstddef>
#include <vector>

#include "../../common.h"
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

// Performs a bitonic sort on the individual blocks of data independently
// Each thread block is responsible for sorting a single block of data,
// and as many thread blocks are launched as there are blocks of data
__global__ void bitonic_sort_blockwise(device_buffer_t<element_t> data) {
    size_t block_start = blockIdx.x * blockDim.x;

    element_t *block_data = &data[block_start];

    // Load the block of data into shared memory
    extern __shared__ element_t shared_data[];
    shared_data[threadIdx.x] = block_data[threadIdx.x];

    __syncthreads();

    // Perform the bitonic sort on the block of data
    for (size_t size = 2; size <= blockDim.x; size *= 2) {
        for (size_t stride = size / 2; stride > 0; stride /= 2) {
            for (size_t i = 0; i < blockDim.x; i++) {
                size_t j = i ^ stride;

                if (j > i) {
                    if ((i & size) == 0 && shared_data[i] > shared_data[j]) {
                        element_t temp = shared_data[i];
                        shared_data[i] = shared_data[j];
                        shared_data[j] = temp;
                    }
                    if ((i & size) != 0 && shared_data[i] < shared_data[j]) {
                        element_t temp = shared_data[i];
                        shared_data[i] = shared_data[j];
                        shared_data[j] = temp;
                    }
                }
            }

            __syncthreads();
        }
    }

    // Store the sorted block of data back to global memory
    block_data[threadIdx.x] = shared_data[threadIdx.x];
}

// Merges two sorted arrays into a single sorted array
__device__ void merge_arrays(device_buffer_t<element_t> &data, device_buffer_t<element_t> &merge_data, size_t start, size_t mid, size_t end) {
    size_t i = start;
    size_t j = mid;
    size_t k = start;

    while (i < mid && j < end) {
        if (data[i] < data[j]) {
            merge_data[k++] = data[i++];
        } else {
            merge_data[k++] = data[j++];
        }
    }

    while (i < mid) {
        merge_data[k++] = data[i++];
    }

    while (j < end) {
        merge_data[k++] = data[j++];
    }

    memcpy(&data[start], &merge_data[start], (end - start) * sizeof(element_t));
}

// Merges a range of consecutive pairs of evenly sized blocks of data
// The number of blocks must be a power of two
// A thread block is launched to merge each pair of blocks
// The number of thread blocks is half the number of blocks
__global__ void merge_evenly_sized_blocks(device_buffer_t<element_t> data, device_buffer_t<element_t> merge_data, size_t num_elements_per_block, size_t range_start) {
    // Find the pair this thread block is responsible for
    size_t block_index = range_start + blockIdx.x * 2;

    // Find the starting and ending indices of the pair of blocks
    size_t pair_start = block_index * num_elements_per_block;
    size_t pair_mid = pair_start + num_elements_per_block;
    size_t pair_end = pair_mid + num_elements_per_block;

    // Merge the pair of blocks
    merge_arrays(data, merge_data, pair_start, pair_mid, pair_end);
}

// Recursively uses kernel launches to merge the blocks of data
__global__ void merge_blocks(device_buffer_t<element_t> data, device_buffer_t<element_t> merge_data, size_t num_elements_per_block, size_t block_start, size_t block_end) {
    size_t num_blocks = block_end - block_start;

    // If there is less than two blocks, there is nothing to merge
    if (num_blocks <= 1) {
        return;
    }

    // Find the largest power of two that fits the number of blocks
    // These can be merged in parallel
    size_t parallel_blocks = largest_power_of_two(num_blocks);
    // Check how many blocks are left over that are not a power of two
    size_t sequential_blocks = num_blocks - parallel_blocks;

    // Launch a kernel to merge the pairs of blocks in a tree-like fashion
    size_t num_remaining_pairs = parallel_blocks / 2;
    size_t merge_size = num_elements_per_block;
    while (num_remaining_pairs >= 1) {
        merge_evenly_sized_blocks<<<num_remaining_pairs, 1>>>(data, merge_data, merge_size, block_start);
        CUDA_CHECK_DEV(cudaPeekAtLastError());
        CUDA_CHECK_DEV(cudaDeviceSynchronize());

        num_remaining_pairs /= 2;
        merge_size *= 2;
    }

    // For the non-power of two blocks, recursively merge them
    if (sequential_blocks > 0) {
        size_t sequential_block_start = block_start + parallel_blocks;
        size_t sequential_block_end = sequential_block_start + sequential_blocks;

        merge_blocks<<<1, 1>>>(data, merge_data, num_elements_per_block, sequential_block_start, sequential_block_end);
        CUDA_CHECK_DEV(cudaPeekAtLastError());
        CUDA_CHECK_DEV(cudaDeviceSynchronize());
        
        // Wait for the kernel to finish
        __syncthreads();

        // Merge the two halves of the data
        merge_arrays(data, merge_data, block_start * num_elements_per_block, sequential_block_start * num_elements_per_block, block_end * num_elements_per_block);
    }

    merge_arrays(
        data,
        merge_data,
        block_start * num_elements_per_block,
        (block_start + parallel_blocks) * num_elements_per_block,
        block_end * num_elements_per_block
    );
}

// Merges the sorted blocks of data on the CPU
void do_final_merge(std::vector<element_t> &data, std::vector<size_t> &sorted_block_starts, size_t num_elements_per_block, size_t start, size_t end) {
    // If there is only one block, there is nothing to merge
    if (end - start <= 1) {
        return;
    }

    // Split the blocks into two halves
    size_t mid = start + (end - start) / 2;

    // Recursively merge the two halves using OpenMP tasks
    #pragma omp task shared(data, sorted_block_starts)
    do_final_merge(data, sorted_block_starts, num_elements_per_block, start, mid);
    #pragma omp task shared(data, sorted_block_starts)
    do_final_merge(data, sorted_block_starts, num_elements_per_block, mid, end);
    #pragma omp taskwait

    // Merge the two halves of the data
    std::inplace_merge(
        data.begin() + sorted_block_starts[start] * num_elements_per_block,
        data.begin() + sorted_block_starts[mid] * num_elements_per_block,
        data.begin() + sorted_block_starts[end] * num_elements_per_block
    );
}

__host__ void sort_blocks_gpu(std::vector<element_t> &data, size_t num_elements_per_block, size_t block_start, size_t block_end) {
    size_t num_blocks = (block_end - block_start);

    // Exit early if there are no blocks to sort
    if (num_blocks == 0) {
        return;
    }

    // Check if there are indeed any CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cout << "No CUDA Devices found!" << std::endl;
        std::exit(1);
    }

    // Print the compute capability of the first device
    cudaDeviceProp device_properties;
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));
    std::cout << "CUDA Device: " << device_properties.name << std::endl;
    std::cout << "Compute Capability: " << device_properties.major << "." << device_properties.minor << std::endl;

    // Find the largest power-of-2 number of blocks that can fit on 40% of the GPU memory
    // This is to ensure that there is enough memory for the merge step
    size_t total_gpu_memory;
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total_gpu_memory));
    size_t safe_gpu_memory = total_gpu_memory * 0.4;
    size_t max_blocks = largest_power_of_two(safe_gpu_memory / (num_elements_per_block * sizeof(element_t)));

    // Allocate memory on the GPU for the blocks of data assigned to the GPU
    device_buffer_t<element_t> device_data;
    // Also allocate memory for temporary storage in merge
    device_buffer_t<element_t> merge_data;

    // While there are more blocks of data than can fit on the GPU, sort the blocks in chunks
    for (size_t i = block_start; i < block_end; i += max_blocks) {
        size_t num_blocks = std::min(max_blocks, block_end - i);
        size_t num_elements = num_blocks * num_elements_per_block;

        // Reallocate the device buffer if the number of elements is different
        // This is likely to happen on the first and last iteration
        if (device_data.size() != num_elements) {
            device_data.reallocate(num_elements);
            merge_data.reallocate(num_elements);
        }

        // Copy the blocks of data to the GPU
        device_data.copy_to_device(data.data() + i * num_elements_per_block, num_elements);

        // Launch the kernel to sort the blocks of data
        std::cout << "GPU Block Sort Start" << std::endl;
        bitonic_sort_blockwise<<<num_blocks, num_elements_per_block, num_elements_per_block * sizeof(element_t)>>>(device_data);
        // Wait for the kernel to finish
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "GPU Block Sort End" << std::endl;

        // Launch the kernel to merge the blocks of data
        std::cout << "GPU Block Merge Start" << std::endl;
        merge_blocks<<<1, 1>>>(device_data, merge_data, num_elements_per_block, 0, num_blocks);
        // Wait for the kernel to finish
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "GPU Block Merge End" << std::endl;

        // Copy the sorted blocks back to the CPU
        device_data.copy_from_device(data.data() + i * num_elements_per_block, num_elements);
    }

    // Prepare a vector containing the indices of the sorted blocks
    std::vector<size_t> sorted_block_starts;
    for (size_t i = block_start; i < block_end; i += max_blocks) {
        sorted_block_starts.push_back(i);
    }
    sorted_block_starts.push_back(block_end);

    // Do the final merge on the CPU
    #pragma omp task shared(data, sorted_block_starts)
    do_final_merge(data, sorted_block_starts, num_elements_per_block, 0, sorted_block_starts.size() - 1);
}
