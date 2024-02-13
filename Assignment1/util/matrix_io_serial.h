#include "matrix.h"
#include <fstream>

#ifndef MATRIX_IO_SERIAL_H
#define MATRIX_IO_SERIAL_H 1

class matrix_serial_io {
public:
  void read_normal(float_matrix &matrix, std::ifstream &file);
  void read_transposed(float_matrix &matrix, std::ifstream &file);

  void write(float_matrix &matrix, std::ofstream &file);
};

#endif