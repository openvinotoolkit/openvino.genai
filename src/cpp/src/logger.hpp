// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <iostream>
#include <string>

namespace ov::genai {

class Logger {
public:
    static void warn(const std::string& message) {
        std::cout << "[WARN] " << message << '\n';
    };
};

}  // namespace ov::genai
