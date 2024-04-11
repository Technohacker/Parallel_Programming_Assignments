#pragma once
#include <cstddef>

// The type of elements in the input file
typedef int element_t;

struct range_t {
    size_t start;
    size_t end;
};

const size_t ELEMENT_SIZE = sizeof(element_t);

const size_t CPU_BLOCK_SIZE = 1024;
const size_t GPU_BLOCK_SIZE = 2048;

const float GPU_FRACTION = 1;
