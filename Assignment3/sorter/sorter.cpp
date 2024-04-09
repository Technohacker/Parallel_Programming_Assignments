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

    // The number of elements in each block
    size_t num_elements_per_block;
    // The fraction of blocks to be sorted on the GPU
    float gpu_fraction;
};

config get_config(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <in file path> <out file path> <num bytes per block> <gpu fraction>" << std::endl;
        exit(1);
    }

    config cfg;
    cfg.in_file_path = argv[1];
    cfg.out_file_path = argv[2];
    cfg.num_elements_per_block = std::stoul(argv[3]) / ELEMENT_SIZE;
    cfg.gpu_fraction = std::stof(argv[4]);

    return cfg;
}

int main(int argc, char *argv[]) {
    config cfg = get_config(argc, argv);
    size_t num_bytes_per_block = cfg.num_elements_per_block * ELEMENT_SIZE;

    // Open the input file
    std::ifstream in_file(cfg.in_file_path, std::ios::binary);
    if (!in_file) {
        std::cerr << "Error: Could not open input file" << std::endl;
        exit(1);
    }

    // Ensure that the file is divisible into blocks
    in_file.seekg(0, std::ios::end);
    size_t file_size = in_file.tellg();
    if (file_size % num_bytes_per_block != 0) {
        std::cerr << "Error: File size is not divisible by the block size" << std::endl;
        exit(1);
    }

    // Read the file into memory
    in_file.seekg(0, std::ios::beg);
    std::vector<int> data(file_size / ELEMENT_SIZE);
    in_file.read(reinterpret_cast<char*>(data.data()), file_size);
    in_file.close();

    std::cout << "File Read" << std::endl;

    // Decide how many blocks to sort on the GPU
    size_t num_total_blocks = file_size / num_bytes_per_block;
    size_t num_gpu_blocks = num_total_blocks * cfg.gpu_fraction;
    size_t num_cpu_blocks = num_total_blocks - num_gpu_blocks;

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
                sort_blocks_cpu(data, cfg.num_elements_per_block, 0, num_cpu_blocks);
                std::cout << "CPU End" << std::endl;
            }

            #pragma omp task shared(data)
            {
                std::cout << "GPU Start" << std::endl;
                sort_blocks_gpu(data, cfg.num_elements_per_block, num_cpu_blocks, num_total_blocks);
                std::cout << "GPU End" << std::endl;
            }

            #pragma omp taskwait

            // Then finally merge the CPU and GPU sorted blocks
            std::cout << "Merge Start" << std::endl;
            std::inplace_merge(
                data.begin(),
                data.begin() + num_cpu_blocks * cfg.num_elements_per_block,
                data.begin() + num_total_blocks * cfg.num_elements_per_block
            );
            std::cout << "Merge End" << std::endl;
        }
    }

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