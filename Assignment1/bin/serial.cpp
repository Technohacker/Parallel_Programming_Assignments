#include <fstream>
#include <iostream>

#include "matrix.h"
#include "matrix_io_serial.h"

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <matrix size> <in file prefix> <out file>"
              << std::endl;
    return 1;
  }

  // Parse args
  int matrix_size = std::stoi(argv[1]);
  std::string file_prefix = argv[2];
  std::string out_file = argv[3];

  // Prepare the files
  std::ifstream file_a(file_prefix + "_a.bin", std::ios::in | std::ios::binary);
  std::ifstream file_b(file_prefix + "_b.bin", std::ios::in | std::ios::binary);
  std::ofstream file_c(out_file, std::ios::out | std::ios::binary);

  // Read the two matrices, with one being transposed
  float_matrix A(matrix_size, matrix_size);
  float_matrix B_T(matrix_size, matrix_size);

  matrix_serial_io io;
  io.read_normal(A, file_a);
  io.read_transposed(B_T, file_b);

  // Reserve the output matrix
  float_matrix C(matrix_size, matrix_size);

  // Compute the result
  for (int i = 0; i < matrix_size; i += 1) {
    for (int j = 0; j < matrix_size; j += 1) {
      float sum = 0.0;
      for (int k = 0; k < matrix_size; k += 1) {
        sum += A[{i, k}] * B_T[{j, k}];
      }

      C[{i, j}] = sum;
    }
  }

  // And write
  io.write(C, file_c);
}