#include <cmath>
#include <iostream>
#include <mpi.h>
#include <ostream>
#include <vector>

#include "matrix.h"
#include "matrix_io_parallel.h"
#include "timing_parallel.h"

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

  if (cart_rank == 0) {
    std::cout << "Matrix Size: " << matrix_size << std::endl;
  }

  // Allocate space for the A, B and C submatrices
  int sub_size = matrix_size / process_grid_side;
  int submatrix_size[] = {sub_size, sub_size};
  float_matrix A(sub_size, sub_size);
  float_matrix B(sub_size, sub_size);
  float_matrix C(sub_size, sub_size);

  // Open the three files
  MPI_File file_a, file_b, file_c;
  MPI_File_open(cart_comm, (file_prefix + "_a.bin").c_str(), MPI_MODE_RDONLY,
                MPI_INFO_NULL, &file_a);
  MPI_File_open(cart_comm, (file_prefix + "_b.bin").c_str(), MPI_MODE_RDONLY,
                MPI_INFO_NULL, &file_b);
  MPI_File_open(cart_comm, out_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &file_c);

  matrix_parallel_io self_io(matrix_size, submatrix_size, cart_dims,
                             cart_coords);

  self_io.read_start(A, file_a);
  self_io.read_start(B, file_b);

  self_io.read_end(A, file_a);
  self_io.read_end(B, file_b);

  float_matrix recv_A(sub_size, sub_size);
  float_matrix recv_B(sub_size, sub_size);

  MPI_Group cart_group;
  MPI_Comm_group(cart_comm, &cart_group);

  MPI_Group row_group, col_group;
  std::vector<int> row_ranks, col_ranks;
  for (int step = 0; step < process_grid_side; step += 1) {
    int row_member[] = {cart_coords[0], step};
    int col_member[] = {step, cart_coords[1]};

    int row_member_rank, col_member_rank;
    MPI_Cart_rank(cart_comm, row_member, &row_member_rank);
    MPI_Cart_rank(cart_comm, col_member, &col_member_rank);

    row_ranks.push_back(row_member_rank);
    col_ranks.push_back(col_member_rank);
  }
  MPI_Group_incl(cart_group, row_ranks.size(), row_ranks.data(), &row_group);
  MPI_Group_incl(cart_group, col_ranks.size(), col_ranks.data(), &col_group);

  MPI_Comm row_comm, col_comm;
  MPI_Comm_create(cart_comm, row_group, &row_comm);
  MPI_Comm_create(cart_comm, col_group, &col_comm);

  int row_rank, col_rank;
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_rank(col_comm, &col_rank);

  timer_parallel timer;
  timer.start(cart_comm);
  for (int step = 0; step < process_grid_side; step += 1) {
    float_matrix *A_src = nullptr, *B_src = nullptr;
    if (step == row_rank) {
      A_src = &A;
    } else {
      A_src = &recv_A;
    }

    if (step == col_rank) {
      B_src = &B;
    } else {
      B_src = &recv_B;
    }

    MPI_Request bcast_reqs[2];
    MPI_Ibcast(A_src->buf_ptr(), A_src->buf_size(), MPI_FLOAT, step, row_comm, &bcast_reqs[0]);
    MPI_Ibcast(B_src->buf_ptr(), B_src->buf_size(), MPI_FLOAT, step, col_comm, &bcast_reqs[1]);

    MPI_Waitall(2, bcast_reqs, MPI_STATUSES_IGNORE);

    // Local outer product
    for (int i = 0; i < sub_size; i += 1) {
      for (int j = 0; j < sub_size; j += 1) {
        for (int k = 0; k < sub_size; k += 1) {
          C[{i, j}] += (*A_src)[{i, k}] * (*B_src)[{k, j}];
        }
      }
    }
  }
  double elapsed_time = timer.end(cart_comm);
  if (cart_rank == 0) {
    std::cout << "Time taken: " << elapsed_time << std::endl;
  }

  self_io.write(C, file_c);
  MPI_Finalize();
}