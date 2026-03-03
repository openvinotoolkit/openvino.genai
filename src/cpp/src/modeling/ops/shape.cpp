// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/shape.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace shape {

ov::Output<ov::Node> axis_i64(int64_t axis) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
}

ov::Output<ov::Node> of(const Tensor& x) {
    return std::make_shared<ov::op::v3::ShapeOf>(x.output(), ov::element::i64);
}

ov::Output<ov::Node> dim(const Tensor& x, int64_t axis) {
    auto shape = of(x);
    auto idx = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    return std::make_shared<ov::op::v8::Gather>(shape, idx, axis_node);
}

ov::Output<ov::Node> make(const std::vector<ov::Output<ov::Node>>& dims) {
    if (dims.empty()) {
        OPENVINO_THROW("shape::make requires at least one dimension");
    }
    ov::OutputVector outputs;
    outputs.reserve(dims.size());
    for (const auto& dim : dims) {
        outputs.push_back(dim);
    }
    return std::make_shared<ov::op::v0::Concat>(outputs, 0);
}

ov::Output<ov::Node> make(std::initializer_list<ov::Output<ov::Node>> dims) {
    return make(std::vector<ov::Output<ov::Node>>(dims));
}

Tensor broadcast_to(const Tensor& x, const ov::Output<ov::Node>& shape) {
    auto node = std::make_shared<ov::op::v3::Broadcast>(x.output(), shape, ov::op::BroadcastType::NUMPY);
    return Tensor(node, x.context());
}

}  // namespace shape
}  // namespace modeling
}  // namespace genai
}  // namespace ov
