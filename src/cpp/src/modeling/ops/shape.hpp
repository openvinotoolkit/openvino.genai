// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace shape {

ov::Output<ov::Node> axis_i64(int64_t axis);
ov::Output<ov::Node> of(const Tensor& x);
ov::Output<ov::Node> dim(const Tensor& x, int64_t axis);
ov::Output<ov::Node> make(const std::vector<ov::Output<ov::Node>>& dims);
ov::Output<ov::Node> make(std::initializer_list<ov::Output<ov::Node>> dims);
Tensor broadcast_to(const Tensor& x, const ov::Output<ov::Node>& shape);

}  // namespace shape
}  // namespace modeling
}  // namespace genai
}  // namespace ov
