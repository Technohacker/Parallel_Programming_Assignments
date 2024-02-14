#include "timing_serial.h"
#include <chrono>

void timer_serial::start() {
    m_begin = Time::now();
}

double timer_serial::end() {
    return (std::chrono::duration_cast<std::chrono::milliseconds>((Time::now() - m_begin)).count()) / 1000.0;
}