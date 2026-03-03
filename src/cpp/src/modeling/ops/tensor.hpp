// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"

namespace ov {
namespace genai {
namespace modeling {

class Tensor {
public:
    Tensor() = default;
    Tensor(const ov::Output<ov::Node>& value, OpContext* ctx = nullptr);

    const ov::Output<ov::Node>& output() const;
    OpContext* context() const;

    ov::element::Type dtype() const;
    Tensor to(const ov::element::Type& type) const;

    Tensor pow(float exp) const;
    Tensor mean(int64_t axis, bool keepdim = true) const;
    Tensor rsqrt() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor tanh() const;
    Tensor softmax(int64_t axis) const;
    Tensor reshape(const ov::Output<ov::Node>& shape, bool special_zero = true) const;
    Tensor reshape(const std::vector<int64_t>& shape, bool special_zero = true) const;
    Tensor reshape(std::initializer_list<int64_t> shape, bool special_zero = true) const;
    Tensor permute(const std::vector<int32_t>& order) const;
    Tensor permute(std::initializer_list<int32_t> order) const;
    Tensor transpose(const std::vector<int32_t>& order) const;
    Tensor transpose(std::initializer_list<int32_t> order) const;
    Tensor unsqueeze(int64_t axis) const;
    Tensor unsqueeze(const std::vector<int64_t>& axes) const;
    Tensor unsqueeze(std::initializer_list<int64_t> axes) const;
    Tensor squeeze(int64_t axis) const;
    Tensor squeeze(const std::vector<int64_t>& axes) const;
    Tensor squeeze(std::initializer_list<int64_t> axes) const;

private:
    ov::Output<ov::Node> value_;
    OpContext* ctx_ = nullptr;
};

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float b);
Tensor operator+(float a, const Tensor& b);

Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, float b);
Tensor operator-(float a, const Tensor& b);
Tensor operator-(const Tensor& a);

Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, float b);
Tensor operator*(float a, const Tensor& b);

Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, float b);
Tensor operator/(float a, const Tensor& b);

}  // namespace modeling
}  // namespace genai
}  // namespace ov
