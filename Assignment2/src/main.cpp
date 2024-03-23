#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#include "common.h"
#include "impl.h"

extern "C" {
#include "mmio.h"
}

struct program_args_t {
  std::string in_file;
  std::vector<node_t> src_vertices;
  std::vector<std::string> out_files;
};

struct final_path_t {
  weight_t total_cost;
  std::vector<node_t> path;
};

bool parse_args(program_args_t &args, int argc, char **argv);
bool read_graph(program_args_t &args, adjacency_list_t &graph);
std::vector<final_path_t> compute_paths(std::vector<path_segment_t> &path_segments);

using Time = std::chrono::steady_clock;
using double_sec = std::chrono::duration<double>;

auto time_start() {
    return Time::now();
}

double time_end(std::chrono::time_point<Time> start) {
    return (std::chrono::duration_cast<std::chrono::milliseconds>((Time::now() - start)).count()) / 1000.0;
}

int main(int argc, char **argv) {
  program_args_t args;
  adjacency_list_t graph;

  if (!parse_args(args, argc, argv)) {
    return 1;
  }
  if (!read_graph(args, graph)) {
    return 1;
  }

  auto start = time_start();
  auto paths = delta_step(graph, args.src_vertices);
  auto duration = time_end(start);

  std::cout << "Time taken: " << duration << std::endl;

  for (size_t i = 0; i < args.src_vertices.size(); i += 1) {
    node_t vertex = args.src_vertices[i];
    auto path_segments = paths[vertex];
    auto final_paths = compute_paths(path_segments);

    std::ofstream outfile(args.out_files[i], std::ios::out);
    for (auto path : final_paths) {
      // Total cost
      // outfile.write((char *) &path.total_cost, sizeof(path.total_cost));
      outfile << path.total_cost;
      // Num nodes
      size_t num_nodes = path.path.size();
      // outfile.write((char *) &num_nodes, sizeof(num_nodes));
      outfile << " " << num_nodes;
      // Nodes
      for (auto node : path.path) {
        node_t out_node = node + 1;
        // outfile.write((char *) &out_node, sizeof(out_node));
        outfile << " " << out_node;
      }
      outfile << std::endl;
    }

    outfile.close();
  }

  return 0;
}

bool parse_args(program_args_t &args, int argc, char **argv) {
  // Parse args
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <in file> [<source vertex> <out file>...]"
              << std::endl;
    return false;
  }

  args.in_file = argv[1];
  for (int i = 2; i < argc; i += 2) {
    args.src_vertices.push_back(std::stoi(argv[i]) - 1);
    args.out_files.push_back(argv[i + 1]);
  }

  return true;
}

bool read_graph(program_args_t &args, adjacency_list_t &graph) {
  // Open the matrix file
  FILE *mm_file = fopen(args.in_file.c_str(), "r");
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
    graph.resize(N);
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

void compute_full_path(std::vector<final_path_t> &final_paths, std::vector<path_segment_t> &path_segments, node_t for_node);

std::vector<final_path_t> compute_paths(std::vector<path_segment_t> &path_segments) {
  std::vector<final_path_t> final_paths(path_segments.size(), { .total_cost = INFINITE_DIST, .path = std::vector<node_t>() });

  for (node_t i = 0; i < path_segments.size(); i += 1) {
    compute_full_path(final_paths, path_segments, i);
  }

  return final_paths;
}

void compute_full_path(std::vector<final_path_t> &final_paths, std::vector<path_segment_t> &path_segments, node_t for_node) {
  auto &curr_segment = path_segments[for_node];
  auto &fin_path = final_paths[for_node];

  if (curr_segment.parent == INVALID_NODE) {
    // Disconnected node
    fin_path.total_cost = -1;
  } else if (path_segments[for_node].parent == for_node) {
    // Source node
    fin_path.total_cost = 0;
    fin_path.path.push_back(for_node);
  } else {
    // Parent path needed
    node_t parent = path_segments[for_node].parent;

    auto &fin_parent = final_paths[parent];

    if (fin_parent.total_cost == INFINITE_DIST) {
      // Parent path yet to be computed
      compute_full_path(final_paths, path_segments, parent);
    }
    
    // Use parent path and extend
    fin_path.total_cost = curr_segment.total_cost;
    fin_path.path.assign(fin_parent.path.begin(), fin_parent.path.end());
    fin_path.path.push_back(for_node);
  }
}