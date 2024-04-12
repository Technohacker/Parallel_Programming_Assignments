#pragma once
#include <vector>

#include "../../common.h"

void sort_cpu(vector_view<element_t> &data, vector_view<element_t> &result);
void merge_pair(vector_view<element_t> &A, vector_view<element_t> &B, vector_view<element_t> &result);
void merge_multiple(vector_view<vector_view<element_t>> &views, vector_view<element_t> &result);
