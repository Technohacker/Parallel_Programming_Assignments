#include <cstddef>
#include <utility>

#ifndef MATRIX_H
#define MATRIX_H 1

typedef std::pair<size_t, size_t> matrix_index;

class float_matrix {
private:
    size_t m_rows;
    size_t m_cols;

    float *m_buffer;
    size_t m_buf_size;
public:
    float_matrix(int rows, int cols);
    ~float_matrix();

    float& operator[] (matrix_index index);

    size_t rows();
    size_t cols();

    float * buf_ptr();
    size_t buf_size();
};

#endif