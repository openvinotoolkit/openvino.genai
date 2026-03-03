// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/context.hpp"

#include <sstream>

#include <openvino/opsets/opset13.hpp>

namespace {

std::string join_ints(const std::vector<int64_t>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

std::string join_ints(const std::vector<int32_t>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {

ov::Output<ov::Node> OpContext::scalar_f32(float v) {
    auto it = f32_cache_.find(v);
    if (it != f32_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {v});
    f32_cache_.emplace(v, node);
    return node;
}

ov::Output<ov::Node> OpContext::scalar_i64(int64_t v) {
    auto it = i64_cache_.find(v);
    if (it != i64_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {v});
    i64_cache_.emplace(v, node);
    return node;
}

ov::Output<ov::Node> OpContext::scalar_i32(int32_t v) {
    auto it = i32_cache_.find(v);
    if (it != i32_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {v});
    i32_cache_.emplace(v, node);
    return node;
}

ov::Output<ov::Node> OpContext::scalar_bool(bool v) {
    auto it = bool_cache_.find(v);
    if (it != bool_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{}, {v});
    bool_cache_.emplace(v, node);
    return node;
}

ov::Output<ov::Node> OpContext::const_i64_vec(const std::vector<int64_t>& values) {
    auto key = join_ints(values);
    auto it = i64_vec_cache_.find(key);
    if (it != i64_vec_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
    i64_vec_cache_.emplace(std::move(key), node);
    return node;
}

ov::Output<ov::Node> OpContext::const_i32_vec(const std::vector<int32_t>& values) {
    auto key = join_ints(values);
    auto it = i32_vec_cache_.find(key);
    if (it != i32_vec_cache_.end()) {
        return it->second;
    }
    auto node = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{values.size()}, values);
    i32_vec_cache_.emplace(std::move(key), node);
    return node;
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
