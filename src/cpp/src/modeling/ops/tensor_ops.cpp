// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/tensor_ops.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"

namespace {

ov::genai::modeling::Tensor ensure_bool(const ov::genai::modeling::Tensor& mask) {
    if (mask.dtype() == ov::element::boolean) {
        return mask;
    }
    return mask.to(ov::element::boolean);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace tensor {

Tensor repeat(const Tensor& x, const std::vector<int64_t>& repeats) {
    auto* ctx = x.context();
    auto repeats_node = ops::const_vec(ctx, repeats);
    auto node = std::make_shared<ov::op::v0::Tile>(x.output(), repeats_node);
    return Tensor(node, ctx);
}

Tensor repeat(const Tensor& x, std::initializer_list<int64_t> repeats) {
    return repeat(x, std::vector<int64_t>(repeats));
}

Tensor tile(const Tensor& x, const std::vector<int64_t>& repeats) {
    return repeat(x, repeats);
}

Tensor tile(const Tensor& x, std::initializer_list<int64_t> repeats) {
    return repeat(x, repeats);
}

Tensor stack(const std::vector<Tensor>& xs, int64_t axis) {
    if (xs.empty()) {
        OPENVINO_THROW("stack requires at least one tensor");
    }
    std::vector<Tensor> expanded;
    expanded.reserve(xs.size());
    for (const auto& x : xs) {
        expanded.push_back(x.unsqueeze(axis));
    }
    return ops::concat(expanded, axis);
}

Tensor masked_scatter(const Tensor& input, const Tensor& mask, const Tensor& updates) {
    auto mask_bool = ensure_bool(mask);
    return ops::where(mask_bool, updates, input);
}

Tensor masked_add(const Tensor& input, const Tensor& mask, const Tensor& updates) {
    auto mask_bool = ensure_bool(mask);
    auto added = input + updates;
    return ops::where(mask_bool, added, input);
}

Tensor masked_fill(const Tensor& input, const Tensor& mask, float value) {
    auto mask_bool = ensure_bool(mask);
    auto* ctx = input.context();
    auto fill = std::make_shared<ov::op::v0::Constant>(input.dtype(), ov::Shape{}, std::vector<float>{value});
    return ops::where(mask_bool, Tensor(fill, ctx), input);
}

Tensor pad(const Tensor& input,
           const std::vector<int64_t>& pads_begin,
           const std::vector<int64_t>& pads_end,
           float value) {
    auto* ctx = input.context();
    auto pads_begin_node = ops::const_vec(ctx, pads_begin);
    auto pads_end_node = ops::const_vec(ctx, pads_end);
    auto pad_value = std::make_shared<ov::op::v0::Constant>(input.dtype(), ov::Shape{}, std::vector<float>{value});
    auto node = std::make_shared<ov::op::v1::Pad>(input.output(),
                                                  pads_begin_node,
                                                  pads_end_node,
                                                  pad_value,
                                                  ov::op::PadMode::CONSTANT);
    return Tensor(node, ctx);
}

std::vector<Tensor> split(const Tensor& input, int64_t num_splits, int64_t axis) {
    auto* ctx = input.context();
    auto axis_node = ops::const_scalar(ctx, axis);
    auto node = std::make_shared<ov::op::v1::Split>(input.output(), axis_node, static_cast<size_t>(num_splits));
    
    std::vector<Tensor> results;
    results.reserve(num_splits);
    for (size_t i = 0; i < static_cast<size_t>(num_splits); ++i) {
        results.push_back(Tensor(node->output(i), ctx));
    }
    return results;
}

std::vector<Tensor> split(const Tensor& input, const std::vector<int64_t>& split_sizes, int64_t axis) {
    auto* ctx = input.context();
    auto axis_node = ops::const_scalar(ctx, axis);
    auto split_lengths = ops::const_vec(ctx, split_sizes);
    auto node = std::make_shared<ov::op::v1::VariadicSplit>(input.output(), axis_node, split_lengths);
    
    std::vector<Tensor> results;
    results.reserve(split_sizes.size());
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        results.push_back(Tensor(node->output(i), ctx));
    }
    return results;
}

Tensor flip(const Tensor& input, int64_t axis) {
    auto* ctx = input.context();
    auto axes_node = ops::const_vec(ctx, std::vector<int64_t>{axis});
    // mode="reflect" with begin=0 and end=0 effectively reverses the tensor
    // Use Reverse op for flipping
    auto node = std::make_shared<ov::op::v1::Reverse>(input.output(), axes_node, ov::op::v1::Reverse::Mode::INDEX);
    return Tensor(node, ctx);
}

}  // namespace tensor
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
