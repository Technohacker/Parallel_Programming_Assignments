// Generates a binary file containing random 32-bit integers from a seeded random number generator.
#include <cstddef>
#include <iostream>
#include <fstream>
#include <random>

#include "../common.h"

const size_t ONE_GIB = 1024 * 1024 * 1024;

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <output file> <size of file in GiB> <seed>" << std::endl;
        return 1;
    }

    std::string output_file = argv[1];
    size_t num_elems = (size_t) std::stoull(argv[2]) * ONE_GIB / ELEMENT_SIZE;
    unsigned int seed = (unsigned int) std::stoul(argv[3]);

    std::ofstream out(output_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    std::mt19937 gen(seed);
    std::uniform_int_distribution<element_t> dist;

    for (size_t i = 0; i < num_elems; i++) {
        element_t elem = dist(gen);
        out.write(reinterpret_cast<char*>(&elem), ELEMENT_SIZE);
    }

    out.close();
    return 0;
}
