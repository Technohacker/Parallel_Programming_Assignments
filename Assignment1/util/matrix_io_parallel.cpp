#include <mpi.h>

#include "matrix.h"
#include "matrix_io_parallel.h"

matrix_parallel_io::matrix_parallel_io(int matrix_size, int submatrix_size[], int cart_dims[], int cart_coords[]) {
  int file_element_dims[] = {matrix_size, matrix_size};
  int submatrix_starts[] = {cart_coords[0] * submatrix_size[0],
                            cart_coords[1] * submatrix_size[1]};

  MPI_Type_create_subarray(2, file_element_dims, submatrix_size,
                           submatrix_starts, MPI_ORDER_C, MPI_FLOAT,
                           &submatrix_type);
  MPI_Type_commit(&submatrix_type);
}

void matrix_parallel_io::read_start(float_matrix &matrix, MPI_File &file) {
  MPI_File_set_view(file, 0, MPI_FLOAT, submatrix_type, "native", MPI_INFO_NULL);

  MPI_File_read_at_all_begin(file, 0, matrix.buf_ptr(), matrix.buf_size(), MPI_FLOAT);
}

void matrix_parallel_io::read_end(float_matrix &matrix, MPI_File &file) {
  MPI_File_read_at_all_end(file, matrix.buf_ptr(), MPI_STATUS_IGNORE);
}

void matrix_parallel_io::write(float_matrix &matrix, MPI_File &file) {
  MPI_File_set_view(file, 0, MPI_FLOAT, submatrix_type, "native", MPI_INFO_NULL);

  MPI_File_write_at_all(file, 0, matrix.buf_ptr(), matrix.buf_size(), MPI_FLOAT, MPI_STATUS_IGNORE);
}
