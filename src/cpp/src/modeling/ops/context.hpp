// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {

class OpContext {
public:
    ov::Output<ov::Node> scalar_f32(float v);
    ov::Output<ov::Node> scalar_i64(int64_t v);
    ov::Output<ov::Node> scalar_i32(int32_t v);
    ov::Output<ov::Node> scalar_bool(bool v);
    ov::Output<ov::Node> const_i64_vec(const std::vector<int64_t>& values);
    ov::Output<ov::Node> const_i32_vec(const std::vector<int32_t>& values);

private:
    std::unordered_map<float, ov::Output<ov::Node>> f32_cache_;
    std::unordered_map<int64_t, ov::Output<ov::Node>> i64_cache_;
    std::unordered_map<int32_t, ov::Output<ov::Node>> i32_cache_;
    std::unordered_map<bool, ov::Output<ov::Node>> bool_cache_;
    std::unordered_map<std::string, ov::Output<ov::Node>> i64_vec_cache_;
    std::unordered_map<std::string, ov::Output<ov::Node>> i32_vec_cache_;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
