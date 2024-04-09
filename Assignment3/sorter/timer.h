#pragma once

#include <chrono>

using Time = std::chrono::steady_clock;
using double_sec = std::chrono::duration<double>;

class timer {
private:
    std::chrono::time_point<Time, double_sec> m_begin;
    std::chrono::time_point<Time, double_sec> m_end;
public:
    void start() {
        m_begin = Time::now();
    }

    void end() {
        m_end = Time::now();
    }

    double elapsed() {
        return (std::chrono::duration_cast<std::chrono::milliseconds>((Time::now() - m_begin)).count()) / 1000.0;
    }
};