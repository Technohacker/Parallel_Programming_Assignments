// Loads a binary file containing 32-bit integers, sorts them, and writes them to a new file.
#include <algorithm>
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

    // Whether to use the GPU or not
    bool use_gpu;
};

config get_config(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <in file path> <out file path> <use gpu>" << std::endl;
        exit(1);
    }

    config cfg;
    cfg.in_file_path = argv[1];
    cfg.out_file_path = argv[2];
    cfg.use_gpu = std::stoi(argv[3]);

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

    // Ensure that the file is divisible into blocks
    in_file.seekg(0, std::ios::end);
    size_t file_size = in_file.tellg();

    // Read the file into memory
    in_file.seekg(0, std::ios::beg);
    std::vector<int> data(file_size / ELEMENT_SIZE);
    in_file.read(reinterpret_cast<char*>(data.data()), file_size);
    in_file.close();

    std::cout << "File Read" << std::endl;

    size_t num_elements = file_size / ELEMENT_SIZE;

    size_t cpu_range_start = 0;
    size_t cpu_range_end = num_elements;
    size_t gpu_range_start = 0;
    size_t gpu_range_end = 0;

    // Check if the GPU should be used
    if (cfg.use_gpu) {
        // Calculate the number of elements to be sorted on the GPU
        size_t num_gpu_elements = num_elements * GPU_FRACTION;

        // Round the number of elements to be sorted on the GPU to the smallest multiple of the GPU block size
        size_t num_gpu_blocks = num_gpu_elements / GPU_BLOCK_SIZE;
        num_gpu_elements = num_gpu_blocks * GPU_BLOCK_SIZE;

        // Adjust the ranges for the CPU and GPU, giving the GPU the first portion of the data
        cpu_range_start = num_gpu_elements;
        gpu_range_end = num_gpu_elements;
    }

    // Start the timer
    timer t;
    t.start();
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            // Launch two OpenMP tasks to blockwise sort the data on the CPU and GPU
            // Once each block is sorted, the CPU will merge sort the two parts
            #pragma omp task shared(data)
            {
                std::cout << "CPU Start" << std::endl;
                sort_range_cpu(data, cpu_range_start, cpu_range_end);
                std::cout << "CPU End" << std::endl;
            }

            #pragma omp task shared(data)
            {
                std::cout << "GPU Start" << std::endl;
                sort_range_gpu(data, gpu_range_start, gpu_range_end);
                std::cout << "GPU End" << std::endl;
            }
        }
    }

    // Then finally merge the CPU and GPU sorted arrays
    std::cout << "Merge Start" << std::endl;
    // std::inplace_merge(data.begin() + cpu_range_start, data.begin() + gpu_range_end, data.end());
    std::cout << "Merge End" << std::endl;

    // End the timer
    t.end();

    // Print the elapsed time
    std::cout << "Elapsed time: " << t.elapsed() << " seconds" << std::endl;

    // Write the sorted data to the output file
    std::ofstream out_file(cfg.out_file_path, std::ios::binary);
    out_file.write(reinterpret_cast<char*>(data.data()), file_size);
    out_file.close();

    return 0;
}