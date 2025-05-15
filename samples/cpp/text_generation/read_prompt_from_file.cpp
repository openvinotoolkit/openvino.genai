// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include "read_prompt_from_file.h"

std::string utils::read_prompt(const std::string& file_path) {
    std::string prompt;
    std::ifstream file(file_path);
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        prompt = buffer.str();
        file.close();        
    }
    return prompt;
}