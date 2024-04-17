// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <chrono>
#include <iostream>

class Timer {
    const decltype(std::chrono::steady_clock::now()) m_start;
public:
    Timer() :
        m_start(std::chrono::steady_clock::now()) {
    }

    double current_in_milli() const {
        auto m_end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(m_end - m_start).count();
    }
};

class ScopedTimer {
    double m_total;
    decltype(std::chrono::steady_clock::now()) m_start;
    std::string m_title;
public:
    ScopedTimer(const std::string& title) :
        m_total(0.),
        m_title(title) {
    }

    void start() {
        m_start = std::chrono::steady_clock::now();
    }

    void end() {
        auto m_end = std::chrono::steady_clock::now();
        m_total += std::chrono::duration<double, std::milli>(m_end - m_start).count();
    }

    ~ScopedTimer() {
        std::cout << m_title << ": " << m_total / 1000. << " secs" << std::endl;
    }
};
