// Loads a binary file containing 32-bit integers, sorts them, and writes them to a new file.
#include <cstddef>
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

#include "../common.h"
#include "timer.h"
#include "cpu/cpu.h"
#include "gpu/gpu.h"

struct config {
    // The path to the input file
    std::string in_file_path;
    // The path to the output file
    std::string out_file_path;

    // The fraction of the data to be sorted on the GPU
    float gpu_fraction;
};

config get_config(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <in file path> <out file path> <gpu fraction>" << std::endl;
        exit(1);
    }

    config cfg;
    cfg.in_file_path = argv[1];
    cfg.out_file_path = argv[2];
    cfg.gpu_fraction = std::stof(argv[3]);

    return cfg;
}

int main(int argc, char *argv[]) {
    config cfg = get_config(argc, argv);

    // Open the input file
    std::ifstream in_file(cfg.in_file_path, std::ios::binary);
    if (!in_file) {
        std::cerr << "Error: Could not open input file" << std::endl;
        exit(1);
    }

    // Get the file's size
    long num_elements;
    in_file.read(reinterpret_cast<char*>(&num_elements), sizeof(long));

    // Read the file into memory, while also allocating space for a merge buffer
    std::vector<element_t> data(num_elements);
    std::vector<element_t> merge_buffer(num_elements);

    std::cout << "File Read Start" << std::endl;
    in_file.read(reinterpret_cast<char*>(data.data()), num_elements * ELEMENT_SIZE);
    in_file.close();
    std::cout << "File Read End" << std::endl;

    range_t cpu_range = {
        .start = 0,
        .end = (size_t) num_elements
    };
    range_t gpu_range = {
        .start = 0,
        .end = 0
    };

    // Calculate the number of elements to be sorted on the GPU
    size_t num_gpu_elements = cfg.gpu_fraction * num_elements;
    std::cout << "GPU Fraction: " << cfg.gpu_fraction << std::endl;
    std::cout << "GPU Elements: " << num_gpu_elements << std::endl;

    // Round the number of elements to be sorted on the GPU to the smallest multiple of the GPU block size
    size_t num_gpu_blocks = num_gpu_elements / GPU_BLOCK_SIZE;
    num_gpu_elements = num_gpu_blocks * GPU_BLOCK_SIZE;

    // Adjust the ranges for the CPU and GPU, giving the GPU the first portion of the data
    cpu_range.start = num_gpu_elements;
    gpu_range.end = num_gpu_elements;

    // Make the views
    vector_view<element_t> data_view(data);
    vector_view<element_t> data_cpu_view(data, cpu_range);
    vector_view<element_t> data_gpu_view(data, gpu_range);

    vector_view<element_t> merge_buffer_cpu_view(merge_buffer, cpu_range);
    vector_view<element_t> merge_buffer_gpu_view(merge_buffer, gpu_range);

    std::cout << "Views Created" << std::endl;
    std::cout << "CPU Range: " << cpu_range.start << " - " << cpu_range.end << std::endl;
    std::cout << "GPU Range: " << gpu_range.start << " - " << gpu_range.end << std::endl;

    // Start the timer
    timer t;
    t.start();

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            // Launch two OpenMP tasks to blockwise sort the data on the CPU and GPU
            // Once each block is sorted, the CPU will merge sort the two parts
            #pragma omp task shared(data_cpu_view, merge_buffer_cpu_view)
            {
                std::cout << "CPU Start" << std::endl;
                sort_cpu(data_cpu_view, merge_buffer_cpu_view);
                std::cout << "CPU End" << std::endl;
            }

            #pragma omp task shared(data_gpu_view, merge_buffer_gpu_view)
            {
                std::cout << "GPU Start" << std::endl;
                sort_gpu(data_gpu_view, merge_buffer_gpu_view);
                std::cout << "GPU End" << std::endl;
            }

            #pragma omp taskwait

            // Then finally launch a task to merge the two sorted parts
            #pragma omp task shared(merge_buffer_cpu_view, merge_buffer_gpu_view, data_view)
            {
                std::cout << "Merge Start" << std::endl;
                merge_pair(merge_buffer_cpu_view, merge_buffer_gpu_view, data_view);
                std::cout << "Merge End" << std::endl;
            }
        }
    }

    // End the timer
    t.end();

    // Print the elapsed time
    std::cout << std::endl << "Elapsed time: " << t.elapsed() << " seconds" << std::endl;

    // Write the sorted data to the output file
    std::ofstream out_file(cfg.out_file_path, std::ios::binary);
    out_file.write(reinterpret_cast<char*>(&num_elements), sizeof(long));
    out_file.write(reinterpret_cast<char*>(data.data()), num_elements * ELEMENT_SIZE);
    out_file.close();

    return 0;
}