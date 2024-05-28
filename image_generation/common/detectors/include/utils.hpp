// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <iostream>
#include <vector>

std::vector<uint8_t> read_bgr_from_txt(const std::string& file_name) {
    std::ifstream input_data(file_name, std::ifstream::in);

    std::vector<uint8_t> res;
    std::string line;
    while (std::getline(input_data, line)) {
        try {
            int value = std::stoi(line);  // 将每行的字符串转换为整数
            if (value < 0 || value > 255) {
                std::cerr << "invalid uint8: " << value << std::endl;
                continue;
            }
            res.push_back(static_cast<uint8_t>(value));
        } catch (const std::invalid_argument& e) {
            std::cerr << "invalid line: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "out of range: " << line << std::endl;
        }
    }

    return res;
}