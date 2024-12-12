// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <chrono>
#include <iostream>

class ManualTimer {
    double m_total;
    decltype(std::chrono::steady_clock::now()) m_start;
    std::string m_title;
public:
    ManualTimer(const std::string& title) :
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

    float get_duration() const {
        return m_total / 1000.;
    }

    ~ManualTimer() {
        std::cout << m_title << ": " << m_total / 1000. << " secs" << std::endl;
    }
};
