#include "matrix.h"
#include <fstream>
#include <mpi.h>

#ifndef MATRIX_IO_PARALLEL_H
#define MATRIX_IO_PARALLEL_H 1

class matrix_parallel_io {
private:
    MPI_Datatype submatrix_type;

public:
    matrix_parallel_io(int matrix_size, int submatrix_size, int cart_dims[], int cart_coords[]);

    void read_start(float_matrix &matrix, MPI_File &file);
    void read_end(float_matrix &matrix, MPI_File &file);

    void write(float_matrix &matrix, MPI_File &file);
};

#endif