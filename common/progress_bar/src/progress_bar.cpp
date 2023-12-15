// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "progress_bar.hpp"

#include <iomanip>
#include <iostream>

ProgressBar::ProgressBar(int total, int width) : total(total), width(width) {
    std::cout << "[";
    print_progress(0);
    std::cout << "]" << std::flush;
}

void ProgressBar::progress(int current) {
    if (current >= 0 && current <= total) {
        int new_progress = static_cast<int>((current * 100.0) / total);
        if (new_progress != last_progress) {
            print_progress(new_progress);
        }
    }
}

void ProgressBar::finish() {
    print_progress(100);
    std::cout << std::endl;
}

void ProgressBar::print_progress(int progress) {
    int pos = (width * progress) / 100;
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << progress << "%" << std::flush;
    last_progress = progress;
}
