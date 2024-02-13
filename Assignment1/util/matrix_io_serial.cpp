#include <fstream>

#include "matrix.h"
#include "matrix_io_serial.h"

void matrix_serial_io::read_normal(float_matrix &matrix, std::ifstream &file) {
  for (int row = 0; row < matrix.rows(); row += 1) {
    for (int col = 0; col < matrix.cols(); col += 1) {
      file.read((char *)&matrix[{row, col}], sizeof(float));
    }
  }
}

void matrix_serial_io::read_transposed(float_matrix &matrix, std::ifstream &file) {
  for (int row = 0; row < matrix.rows(); row += 1) {
    for (int col = 0; col < matrix.cols(); col += 1) {
      file.read((char *)&matrix[{col, row}], sizeof(float));
    }
  }
}

// Row-major binary form
void matrix_serial_io::write(float_matrix &matrix, std::ofstream &file) {
  file.write((char *)matrix.buf_ptr(), matrix.buf_size() * sizeof(float));
}
