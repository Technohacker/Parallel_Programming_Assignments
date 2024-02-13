#include <cmath>
#include <iostream>
#include <mpi.h>
#include <ostream>

#include "matrix.h"
#include "matrix_io_parallel.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <matrix size> <in file prefix> <out file>" << std::endl;
    return 1;
  }

  // Parse args
  int matrix_size = std::stoi(argv[1]);
  std::string file_prefix = argv[2];
  std::string out_file = argv[3];

  // Get all relevant info for the current process
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Prepare a cartesian communicator
  double process_grid_side_double = std::sqrt(world_size);
  if (process_grid_side_double != std::floor(process_grid_side_double)) {
    std::cerr << "Number of processes must be a perfect square" << std::endl;
    return 1;
  }
  int process_grid_side = (int)process_grid_side_double;

  if (matrix_size % process_grid_side != 0) {
    std::cerr << "Number of processes must evenly divide matrix size"
              << std::endl;
    return 1;
  }

  MPI_Comm cart_comm;
  int cart_rank, cart_coords[] = {0, 0};
  int cart_dims[] = {process_grid_side, process_grid_side};
  int periodic[] = {true, true};

  MPI_Cart_create(MPI_COMM_WORLD, 2, cart_dims, periodic, true, &cart_comm);
  MPI_Comm_rank(cart_comm, &cart_rank);
  MPI_Cart_coords(cart_comm, cart_rank, 2, cart_coords);

  // Allocate space for the A, B and C submatrices
  int submatrix_size = matrix_size / process_grid_side;
  float_matrix A(submatrix_size, submatrix_size);
  float_matrix B(submatrix_size, submatrix_size);
  float_matrix C(submatrix_size, submatrix_size);

  // Open the three files
  MPI_File file_a, file_b, file_c;
  MPI_File_open(cart_comm, (file_prefix + "_a.bin").c_str(), MPI_MODE_RDONLY,
                MPI_INFO_NULL, &file_a);
  MPI_File_open(cart_comm, (file_prefix + "_b.bin").c_str(), MPI_MODE_RDONLY,
                MPI_INFO_NULL, &file_b);
  MPI_File_open(cart_comm, out_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &file_c);

  // Read the skewed version directly rather than shifting later
  int a_skew_rank, b_skew_rank;
  int a_skew[] = {0, 0};
  int b_skew[] = {0, 0};

  MPI_Cart_shift(cart_comm, 1, -cart_coords[0], &cart_rank, &a_skew_rank);
  MPI_Cart_shift(cart_comm, 0, -cart_coords[1], &cart_rank, &b_skew_rank);

  MPI_Cart_coords(cart_comm, a_skew_rank, 2, a_skew);
  MPI_Cart_coords(cart_comm, b_skew_rank, 2, b_skew);

  matrix_parallel_io a_start_io(matrix_size, submatrix_size, cart_dims,
                              a_skew);
  matrix_parallel_io b_start_io(matrix_size, submatrix_size, cart_dims,
                              b_skew);
  matrix_parallel_io self_io(matrix_size, submatrix_size, cart_dims,
                             cart_coords);

  MPI_Barrier(cart_comm);
  std::cout << "Self: (" << cart_coords[0] << ", " << cart_coords[1] << "), Skew A: (" << a_skew[0] << ", " << a_skew[1] << ")" << std::endl;
  MPI_Barrier(cart_comm);
  std::cout << "Self: (" << cart_coords[0] << ", " << cart_coords[1] << "), Skew B: (" << b_skew[0] << ", " << b_skew[1] << ")" << std::endl;
  MPI_Barrier(cart_comm);

  a_start_io.read_start(A, file_a);
  b_start_io.read_start(B, file_b);

  a_start_io.read_end(A, file_a);
  b_start_io.read_end(B, file_b);

  int send_neighbour_a, send_neighbour_b;
  int recv_neighbour_a, recv_neighbour_b;
  MPI_Cart_shift(cart_comm, 1, -1, &cart_rank, &send_neighbour_a);
  MPI_Cart_shift(cart_comm, 0, -1, &cart_rank, &send_neighbour_b);

  MPI_Cart_shift(cart_comm, 1, 1, &cart_rank, &recv_neighbour_a);
  MPI_Cart_shift(cart_comm, 0, 1, &cart_rank, &recv_neighbour_b);

  for (int step = 0; step < process_grid_side; step += 1) {
    // Local multiply
    for (int i = 0; i < submatrix_size; i += 1) {
      for (int j = 0; j < submatrix_size; j += 1) {
        // float sum = 0.0;
        for (int k = 0; k < submatrix_size; k += 1) {
          C[{i, j}] += A[{i, k}] * B[{k, j}];
        }
      }
    }
    MPI_Barrier(cart_comm);

    // Shift
    MPI_Sendrecv_replace(A.buf_ptr(), A.buf_size(), MPI_FLOAT, send_neighbour_a,
                         0, recv_neighbour_a, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(B.buf_ptr(), B.buf_size(), MPI_FLOAT, send_neighbour_b,
                         1, recv_neighbour_b, 1, cart_comm, MPI_STATUS_IGNORE);
  }

  self_io.write(C, file_c);
  MPI_Finalize();
}