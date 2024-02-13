#include "matrix.h"

float_matrix::float_matrix(int rows, int cols) {
    m_rows = rows;
    m_cols = cols;

    m_buf_size = rows * cols;
    m_buffer = new float[m_buf_size];
}

float_matrix::~float_matrix() {
    delete m_buffer;
}

float& float_matrix::operator[](matrix_index index) {
    return m_buffer[(m_cols * index.first) + index.second];
}

size_t float_matrix::rows() {
    return m_rows;
}

size_t float_matrix::cols() {
    return m_cols;
}

float* float_matrix::buf_ptr() {
    return m_buffer;
}

size_t float_matrix::buf_size() {
    return m_buf_size;
}
