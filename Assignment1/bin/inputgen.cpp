#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include <stdfloat>
#include <string>

#include "matrix.h"
#include "matrix_io_serial.h"

void make_and_write_matrix(float_matrix &matrix, std::mt19937 &rng, std::ofstream &file) {
  std::uniform_real_distribution<float_t> dis(-1.0, +1.0); 

  for (int i = 0; i < matrix.rows(); i += 1) {
    for (int j = 0; j < matrix.cols(); j += 1) {
      matrix[{i, j}] = dis(rng);
    }
  }

  matrix_serial_io io;
  io.write(matrix, file);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <matrix size> <seed> <out file prefix>"
              << std::endl;
    return 1;
  }

  // Parse args
  int matrix_size = std::stoi(argv[1]);
  int rng_seed = std::stoi(argv[2]);
  std::string file_prefix = argv[3];

  // Prepare the files
  std::ofstream file_a(file_prefix + "_a.bin", std::ios::out | std::ios::binary);
  std::ofstream file_b(file_prefix + "_b.bin", std::ios::out | std::ios::binary);

  // Make the RNG
  std::mt19937 rng(rng_seed);

  float_matrix matrix(matrix_size, matrix_size);

  // Build and write the two matrices
  make_and_write_matrix(matrix, rng, file_a);
  make_and_write_matrix(matrix, rng, file_b);
}
