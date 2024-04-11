#pragma once
#include <vector>

#include "../../common.h"

void sort_cpu(vector_view<element_t> data);
void merge_pair(vector_view<element_t> A, vector_view<element_t> B, vector_view<element_t> result);
std::vector<element_t> merge_multiple(vector_view<vector_view<element_t>> views);
