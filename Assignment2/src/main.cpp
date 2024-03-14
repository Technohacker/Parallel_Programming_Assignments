#include <iostream>
#include <stdio.h>
#include <string>

#include "impl.h"

extern "C" {
#include "mmio.h"
}

int main(int argc, char **argv) {
  // Parse args
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <file>" << std::endl;
    return 1;
  }

  std::string file = argv[1];

  // Open the matrix file
  FILE *mm_file = fopen(file.c_str(), "r");
  if (mm_file == nullptr) {
    std::cout << "Matrix file not found" << std::endl;
    return 1;
  }

  MM_typecode mat_type;

  int res = mm_read_banner(mm_file, &mat_type);
  if (res != 0) {
    std::cout << "Error reading banner: " << res << std::endl;
    return 1;
  }

  if (!(mm_is_sparse(mat_type) &&
        // TODO: This condition should also be present
        // mm_is_integer(mat_type) &&
        mm_is_symmetric(mat_type))) {
    std::cout << "Wrong matrix type. Expected sparse, symmetric matrix. Found: "
              << mm_typecode_to_str(mat_type) << std::endl;
    return 1;
  }

  // HACK: Set the integer type just in case
  mm_set_integer(&mat_type);

  int M, N, num_edges;
  mm_read_mtx_crd_size(mm_file, &M, &N, &num_edges);

  if (M != N) {
    std::cout << "Matrix should be square" << std::endl;
    return 1;
  }

  // Read the matrix
  adjacency_list graph(M);

  for (int i = 0; i < num_edges; i += 1) {
    int src, dest, weight;
    int res = mm_read_mtx_crd_entry(mm_file, &src, &dest, &weight, nullptr,
                                    nullptr, mat_type);

    if (res != 0) {
      std::cout << "Error reading file: " << res << std::endl;
      return 1;
    }

    // Node numbers are considered to be 0-indexed for both src and dest
    // Add both edges to make it undirected
    graph.at(src - 1).push_back({
        .dest = dest - 1,
        .weight = weight,
    });
    graph.at(dest - 1).push_back({
        .dest = src - 1,
        .weight = weight,
    });
  }

  return 0;
}