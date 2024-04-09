#include <algorithm>
#include <cstddef>
#include <vector>

#include "../../common.h"
#include "gpu.h"

// Performs a bitonic sort on the individual blocks of data independently
// Each thread block is responsible for sorting a single block of data,
// and as many thread blocks are launched as there are blocks of data
__global__ void bitonic_sort_blockwise(device_buffer_t<element_t> data) {
    size_t block_start = blockIdx.x * blockDim.x;

    element_t *block_data = data.buf + block_start;

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

// Performs a sequential merge of two consecutive unequal-sized blocks of data
__device__ void merge_blocks(element_t *data, size_t start, size_t mid, size_t end) {
    size_t i = start;
    size_t j = mid;
    size_t k = start;

    while (i < mid && j < end) {
        if (data[i] < data[j]) {
            data[k++] = data[i++];
        } else {
            data[k++] = data[j++];
        }
    }

    while (i < mid) {
        data[k++] = data[i++];
    }

    while (j < end) {
        data[k++] = data[j++];
    }
}

// Performs a sequential merge of several consecutive pairs of blocks of data simultaneously
// Each pair of blocks is merged by a single thread block, and as many thread blocks are launched as there are pairs of blocks
__global__ void merge_range_parallel(device_buffer_t<element_t> data, size_t num_elements_per_block, size_t range_start) {
    size_t element_start = range_start + blockIdx.x * 2 * num_elements_per_block;
    size_t element_mid = element_start + num_elements_per_block;
    size_t element_end = element_mid + num_elements_per_block;

    merge_blocks(data.buf, element_start, element_mid, element_end);
}

// Performs a sequential merge of a pair of unequal-sized subarrays of data
// The subarrays are merged by a single thread block
__global__ void merge_pair_sequential(device_buffer_t<element_t> data, size_t element_start, size_t element_mid, size_t element_end) {
    merge_blocks(data.buf, element_start, element_mid, element_end);
}

// Recursively merges the blocks of data until there is only one block left
// Uses the merge_range_parallel and merge_pair_sequential kernels to merge the blocks
__host__ void merge_blocks_gpu(device_buffer_t<element_t> &data, size_t num_elements_per_block, size_t block_start, size_t block_end) {
    std::cout << "Merge Entry: " << block_start << " " << block_end << std::endl;
    size_t num_blocks = (block_end - block_start);

    // Exit early if there are no blocks to merge, or if there is only one block left
    if (num_blocks == 0 || num_blocks == 1) {
        return;
    }

    // Find the largest power of 2 that is less than or equal to the number of blocks
    // This is the number of blocks that can be merged in parallel
    size_t num_parallel_merge_blocks = 1;
    while (num_parallel_merge_blocks * 2 <= num_blocks) {
        num_parallel_merge_blocks *= 2;
    }

    // Keep track of the start and end of the parallel and sequential merge ranges
    size_t parallel_start = block_start;
    size_t parallel_end = block_start + num_parallel_merge_blocks;
    size_t sequential_start = parallel_end;
    size_t sequential_end = block_end;

    size_t num_remaining_parallel_blocks = num_parallel_merge_blocks;
    size_t parallel_block_size = num_elements_per_block;
    while (num_remaining_parallel_blocks > 1) {
        std::cout << "Merge Parallel: " << num_remaining_parallel_blocks << " " << parallel_block_size << " " << parallel_start << std::endl;
        // Launch the kernel to merge the parallel range of blocks
        merge_range_parallel<<<num_remaining_parallel_blocks / 2, 1>>>(data, parallel_block_size, parallel_start);
        // Wait for the kernel to finish
        CUDA_CHECK(cudaDeviceSynchronize());

        // Halve the number of remaining parallel blocks and double the block size
        num_remaining_parallel_blocks /= 2;
        parallel_block_size *= 2;
    }

    // If there are sequential blocks left, recursively merge them
    if (sequential_end - sequential_start > 0) {
        std::cout << "Merge Sequential: " << sequential_start << " " << sequential_end << std::endl;
        merge_blocks_gpu(data, num_elements_per_block, sequential_start, sequential_end);

        // Then merge the parallel and sequential blocks
        merge_pair_sequential<<<1, 1>>>(
            data,
            parallel_start * num_elements_per_block,
            sequential_start * num_elements_per_block,
            sequential_end * num_elements_per_block
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::cout << "Merge Exit: " << block_start << " " << block_end << std::endl;
}

// Runs a CPU-side merge of the sorted blocks of data provided by the GPU
// The blocks are merged in a recursive manner until there is only one block left
__host__ void do_final_merge(std::vector<element_t> &data, std::vector<size_t> &sorted_block_starts, size_t start, size_t end) {
    // If there is only one block left, then we are done
    if (end - start <= 1) {
        return;
    }

    // Find the middle of the range
    size_t mid = start + (end - start) / 2;

    // Recursively merge the left and right halves
    #pragma omp task shared(data, sorted_block_starts)
    do_final_merge(data, sorted_block_starts, start, mid);
    #pragma omp task shared(data, sorted_block_starts)
    do_final_merge(data, sorted_block_starts, mid, end);

    // Wait for the tasks to finish
    #pragma omp taskwait

    // Merge the two halves
    size_t left = sorted_block_starts[start];
    size_t right = sorted_block_starts[mid];
    size_t left_end = sorted_block_starts[mid];
    size_t right_end = sorted_block_starts[end];

    std::inplace_merge(
        data.begin() + left,
        data.begin() + left_end,
        data.begin() + right_end
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

    // Find the largest power-of-2 number of blocks that can fit on 80% of the GPU memory
    size_t total_gpu_memory;
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total_gpu_memory));
    size_t safe_gpu_memory = total_gpu_memory * 0.8;
    size_t max_blocks = 1;
    while (max_blocks * num_elements_per_block * sizeof(element_t) * 2 < safe_gpu_memory) {
        max_blocks *= 2;
    }

    // Allocate memory on the GPU for the blocks of data assigned to the GPU
    device_buffer_t<element_t> device_data;

    // While there are more blocks of data than can fit on the GPU, sort the blocks in chunks
    for (size_t i = block_start; i < block_end; i += max_blocks) {
        size_t num_blocks = std::min(max_blocks, block_end - i);
        size_t num_elements = num_blocks * num_elements_per_block;

        // Reallocate the device buffer if the number of elements is different
        // This is likely to happen on the first and last iteration
        if (device_data.size() != num_elements) {
            device_data.reallocate(num_elements);
        }

        // Copy the blocks of data to the GPU
        device_data.copy_to_device(data.data() + i * num_elements_per_block, num_elements);

        // Launch the kernel to sort the blocks of data
        std::cout << "GPU Block Sort Start" << std::endl;
        bitonic_sort_blockwise<<<num_blocks, num_elements_per_block, num_elements_per_block * sizeof(element_t)>>>(device_data);
        // Wait for the kernel to finish
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "GPU Block Sort End" << std::endl;

        // Merge the sorted blocks of data
        std::cout << "GPU Block Merge Start" << std::endl;
        merge_blocks_gpu(device_data, num_elements_per_block, 0, num_blocks);
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
    do_final_merge(data, sorted_block_starts, 0, sorted_block_starts.size());
}
