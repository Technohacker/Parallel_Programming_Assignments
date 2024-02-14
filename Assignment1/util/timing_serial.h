#include <chrono>
#include <ratio>

#ifndef TIMING_SERIAL_H
#define TIMING_SERIAL_H 1

using Time = std::chrono::steady_clock;
using double_sec = std::chrono::duration<double>;

class timer_serial {
private:
    std::chrono::time_point<Time, double_sec> m_begin;
public:
    void start();
    double end();
};

#endif