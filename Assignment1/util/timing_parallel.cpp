#include "timing_parallel.h"

#include <mpi.h>

void timer_parallel::start(MPI_Comm &comm) {
    MPI_Barrier(comm);
    m_begin = MPI_Wtime();
}

double timer_parallel::end(MPI_Comm &comm) {
    MPI_Barrier(comm);
    return (MPI_Wtime() - m_begin);
}