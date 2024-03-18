#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include "common.h"
#include "impl.h"

extern "C" {
#include "mmio.h"
}

struct program_args {
  std::string file;
  std::vector<node_t> src_vertices;
};

bool parse_args(program_args &args, int argc, char **argv);
bool read_graph(program_args &args, adjacency_list &graph);

int main(int argc, char **argv) {
  program_args args;
  adjacency_list graph;

  if (!parse_args(args, argc, argv)) {
    return 1;
  }
  if (!read_graph(args, graph)) {
    return 1;
  }

  auto paths = delta_step(graph, args.src_vertices);

  return 0;
}

bool parse_args(program_args &args, int argc, char **argv) {
  // Parse args
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <file> [<source vertices>...]"
              << std::endl;
    return false;
  }

  args.file = argv[1];
  for (int i = 2; i < argc; i += 1) {
    args.src_vertices.push_back(std::stoi(argv[i]));
  }

  return true;
}

bool read_graph(program_args &args, adjacency_list &graph) {
  // Open the matrix file
  FILE *mm_file = fopen(args.file.c_str(), "r");
  if (mm_file == nullptr) {
    std::cout << "Matrix file not found" << std::endl;
    return false;
  }

  MM_typecode mat_type;

  int res = mm_read_banner(mm_file, &mat_type);
  if (res != 0) {
    std::cout << "Error reading banner: " << res << std::endl;
    return false;
  }

  if (!(mm_is_sparse(mat_type) &&
        // TODO: This condition should also be present
        // mm_is_integer(mat_type) &&
        mm_is_symmetric(mat_type))) {
    std::cout << "Wrong matrix type. Expected sparse, symmetric matrix. Found: "
              << mm_typecode_to_str(mat_type) << std::endl;
    return false;
  }

  // HACK: Set the integer type just in case
  mm_set_integer(&mat_type);

  int M, N, num_edges;
  mm_read_mtx_crd_size(mm_file, &M, &N, &num_edges);

  if (M != N) {
    std::cout << "Matrix should be square" << std::endl;
    return false;
  }

  // Read the matrix
  for (int i = 0; i < num_edges; i += 1) {
    int src, dest, weight;
    int res = mm_read_mtx_crd_entry(mm_file, &src, &dest, &weight, nullptr,
                                    nullptr, mat_type);

    if (res != 0) {
      std::cout << "Error reading file: " << res << std::endl;
      return false;
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

  return true;
}