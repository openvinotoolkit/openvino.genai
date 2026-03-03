// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/tensor.hpp"

#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

namespace {

ov::genai::modeling::OpContext* resolve_context(const ov::genai::modeling::Tensor& a,
                                                const ov::genai::modeling::Tensor& b) {
    auto* a_ctx = a.context();
    auto* b_ctx = b.context();
    if (a_ctx && b_ctx && a_ctx != b_ctx) {
        OPENVINO_THROW("Tensor contexts do not match");
    }
    return a_ctx ? a_ctx : b_ctx;
}

ov::Output<ov::Node> scalar_f32(ov::genai::modeling::OpContext* ctx, float v) {
    if (ctx) {
        return ctx->scalar_f32(v);
    }
    return ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {v});
}

ov::Output<ov::Node> i64_vec(ov::genai::modeling::OpContext* ctx, const std::vector<int64_t>& values) {
    if (ctx) {
        return ctx->const_i64_vec(values);
    }
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> i32_vec(ov::genai::modeling::OpContext* ctx, const std::vector<int32_t>& values) {
    if (ctx) {
        return ctx->const_i32_vec(values);
    }
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{values.size()}, values);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {

Tensor::Tensor(const ov::Output<ov::Node>& value, OpContext* ctx) : value_(value), ctx_(ctx) {}

const ov::Output<ov::Node>& Tensor::output() const {
    return value_;
}

OpContext* Tensor::context() const {
    return ctx_;
}

ov::element::Type Tensor::dtype() const {
    return value_.get_element_type();
}

Tensor Tensor::to(const ov::element::Type& type) const {
    if (dtype() == type) {
        return *this;
    }
    auto node = std::make_shared<ov::op::v0::Convert>(value_, type);
    return Tensor(node, ctx_);
}

Tensor Tensor::pow(float exp) const {
    auto exp_node = scalar_f32(ctx_, exp);
    auto node = std::make_shared<ov::op::v1::Power>(value_, exp_node, ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx_);
}

Tensor Tensor::mean(int64_t axis, bool keepdim) const {
    auto axis_node = i64_vec(ctx_, {axis});
    auto node = std::make_shared<ov::op::v1::ReduceMean>(value_, axis_node, keepdim);
    return Tensor(node, ctx_);
}

Tensor Tensor::rsqrt() const {
    auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(value_);
    auto one = scalar_f32(ctx_, 1.0f);
    auto node = std::make_shared<ov::op::v1::Divide>(one, sqrt_node, ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx_);
}

Tensor Tensor::sin() const {
    auto node = std::make_shared<ov::op::v0::Sin>(value_);
    return Tensor(node, ctx_);
}

Tensor Tensor::cos() const {
    auto node = std::make_shared<ov::op::v0::Cos>(value_);
    return Tensor(node, ctx_);
}

Tensor Tensor::exp() const {
    auto node = std::make_shared<ov::op::v0::Exp>(value_);
    return Tensor(node, ctx_);
}

Tensor Tensor::log() const {
    auto node = std::make_shared<ov::op::v0::Log>(value_);
    return Tensor(node, ctx_);
}

Tensor Tensor::tanh() const {
    auto node = std::make_shared<ov::op::v0::Tanh>(value_);
    return Tensor(node, ctx_);
}

Tensor Tensor::softmax(int64_t axis) const {
    auto node = std::make_shared<ov::op::v1::Softmax>(value_, axis);
    return Tensor(node, ctx_);
}

Tensor Tensor::reshape(const ov::Output<ov::Node>& shape, bool special_zero) const {
    auto node = std::make_shared<ov::op::v1::Reshape>(value_, shape, special_zero);
    return Tensor(node, ctx_);
}

Tensor Tensor::reshape(const std::vector<int64_t>& shape, bool special_zero) const {
    return reshape(i64_vec(ctx_, shape), special_zero);
}

Tensor Tensor::reshape(std::initializer_list<int64_t> shape, bool special_zero) const {
    return reshape(std::vector<int64_t>(shape), special_zero);
}

Tensor Tensor::permute(const std::vector<int32_t>& order) const {
    auto node = std::make_shared<ov::op::v1::Transpose>(value_, i32_vec(ctx_, order));
    return Tensor(node, ctx_);
}

Tensor Tensor::permute(std::initializer_list<int32_t> order) const {
    return permute(std::vector<int32_t>(order));
}

Tensor Tensor::transpose(const std::vector<int32_t>& order) const {
    return permute(order);
}

Tensor Tensor::transpose(std::initializer_list<int32_t> order) const {
    return permute(order);
}

Tensor Tensor::unsqueeze(int64_t axis) const {
    return unsqueeze(std::vector<int64_t>{axis});
}

Tensor Tensor::unsqueeze(const std::vector<int64_t>& axes) const {
    auto node = std::make_shared<ov::op::v0::Unsqueeze>(value_, i64_vec(ctx_, axes));
    return Tensor(node, ctx_);
}

Tensor Tensor::unsqueeze(std::initializer_list<int64_t> axes) const {
    return unsqueeze(std::vector<int64_t>(axes));
}

Tensor Tensor::squeeze(int64_t axis) const {
    return squeeze(std::vector<int64_t>{axis});
}

Tensor Tensor::squeeze(const std::vector<int64_t>& axes) const {
    auto node = std::make_shared<ov::op::v0::Squeeze>(value_, i64_vec(ctx_, axes));
    return Tensor(node, ctx_);
}

Tensor Tensor::squeeze(std::initializer_list<int64_t> axes) const {
    return squeeze(std::vector<int64_t>(axes));
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    auto* ctx = resolve_context(a, b);
    auto node = std::make_shared<ov::op::v1::Add>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor operator+(const Tensor& a, float b) {
    auto* ctx = a.context();
    auto b_node = scalar_f32(ctx, b);
    return a + Tensor(b_node, ctx);
}

Tensor operator+(float a, const Tensor& b) {
    return b + a;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    auto* ctx = resolve_context(a, b);
    auto node = std::make_shared<ov::op::v1::Subtract>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor operator-(const Tensor& a, float b) {
    auto* ctx = a.context();
    auto b_node = scalar_f32(ctx, b);
    return a - Tensor(b_node, ctx);
}

Tensor operator-(float a, const Tensor& b) {
    auto* ctx = b.context();
    auto a_node = scalar_f32(ctx, a);
    auto node = std::make_shared<ov::op::v1::Subtract>(a_node, b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor operator-(const Tensor& a) {
    auto node = std::make_shared<ov::op::v0::Negative>(a.output());
    return Tensor(node, a.context());
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    auto* ctx = resolve_context(a, b);
    auto node = std::make_shared<ov::op::v1::Multiply>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor operator*(const Tensor& a, float b) {
    auto* ctx = a.context();
    auto b_node = scalar_f32(ctx, b);
    return a * Tensor(b_node, ctx);
}

Tensor operator*(float a, const Tensor& b) {
    return b * a;
}

Tensor operator/(const Tensor& a, const Tensor& b) {
    auto* ctx = resolve_context(a, b);
    auto node = std::make_shared<ov::op::v1::Divide>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor operator/(const Tensor& a, float b) {
    auto* ctx = a.context();
    auto b_node = scalar_f32(ctx, b);
    auto node = std::make_shared<ov::op::v1::Divide>(a.output(), b_node, ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor operator/(float a, const Tensor& b) {
    auto* ctx = b.context();
    auto a_node = scalar_f32(ctx, a);
    auto node = std::make_shared<ov::op::v1::Divide>(a_node, b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
