// Generates a binary file containing random 32-bit integers from a seeded random number generator.
#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <output file> <size of file in GiB> <seed>" << std::endl;
        return 1;
    }

    std::string output_file = argv[1];
    size_t num_ints = (size_t) std::stoull(argv[2]) * 1024 * 1024 * 1024 / sizeof(int);
    unsigned int seed = (unsigned int) std::stoul(argv[3]);

    std::ofstream out(output_file, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist;

    for (size_t i = 0; i < num_ints; i++) {
        int num = dist(gen);
        out.write(reinterpret_cast<char*>(&num), sizeof(int));
    }

    out.close();
    return 0;
}
