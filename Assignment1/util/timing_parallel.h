#include <mpi.h>

#ifndef TIMING_PARALLEL_H
#define TIMING_PARALLEL_H 1

class timer_parallel {
private:
    double m_begin;
public:
    void start(MPI_Comm &comm);
    double end(MPI_Comm &comm);
};

#endif