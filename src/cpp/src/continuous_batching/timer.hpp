// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <chrono>
#include <iostream>

class ManualTimer {
    double m_total;
    std::chrono::steady_clock::time_point m_start, m_end;
    std::string m_title;
public:
    ManualTimer(const std::string& title) :
        m_total(0.),
        m_start(),
        m_end(),
        m_title(title) {
    }

    void start() {
        m_start = std::chrono::steady_clock::now();
    }

    void end() {
        m_end = std::chrono::steady_clock::now();
        m_total += std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start).count();
    }

    std::chrono::steady_clock::time_point get_start_time() {
        return m_start;
    }

    std::chrono::steady_clock::time_point get_end_time() {
        return m_end;
    }

    float get_duration() const {
        return m_total / 1e6;
    }

    float get_duration_microsec() const {
        return m_total;
    }

    void clear() {
        m_total = 0.0;
        m_start = std::chrono::steady_clock::time_point();
        m_end = std::chrono::steady_clock::time_point();
    }

    ~ManualTimer() {
        // std::cout << m_title << ": " << m_total / 1e6 << " secs" << std::endl;
    }
};
