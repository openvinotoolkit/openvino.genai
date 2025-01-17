// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>
#include <optional>
#include <regex>
#include <vector>

#include "openvino/op/constant.hpp"

namespace ov {
namespace genai {
namespace utils {

template <typename T>
struct LoRAParts {
    T alpha, A, B;

    LoRAParts() = default;
    LoRAParts(const T& alpha, const T& A, const T& B) : alpha(alpha), A(A), B(B) {}

    template <typename Other>
    LoRAParts(const LoRAParts<Other>& other) : alpha(other.alpha), A(other.A), B(other.B) {}
};


// Holds a compiled regex pattern and an index to a particular capture group
// operator() takes a string, parses it with that regex pattern and returns the capture group value
struct RegexParser {
    std::regex pattern;
    std::vector<size_t> capture_indices;
    RegexParser (const std::string& pattern, size_t capture_index) : pattern(pattern), capture_indices(1, capture_index) {}
    RegexParser (const std::string& pattern, const std::vector<size_t>& capture_indices) : pattern(pattern), capture_indices(capture_indices) {}
    std::optional<std::string> operator() (const std::string& name) const {
        std::smatch match;
        if(std::regex_match(name, match, pattern)) {
            for(auto capture_index: capture_indices) {
                // check if a given capture group exists (really matched) and return the first matched group
                if(capture_index < match.size() && match[capture_index].matched) {
                    return match[capture_index];
                }
            }
        }
        return std::nullopt;
    }
};


using LoRAWeight = LoRAParts<std::shared_ptr<ov::op::v0::Constant>>;
using LoRATensors = std::map<std::string, LoRAWeight>;

}
}
}
