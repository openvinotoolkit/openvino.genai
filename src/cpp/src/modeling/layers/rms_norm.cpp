// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layers/rms_norm.hpp"

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {

RMSNorm::RMSNorm(const Tensor& weight, float eps) : Module(), weight_(weight), eps_(eps) {}

RMSNorm::RMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent)
    : Module(name, ctx, parent), eps_(eps) {
    weight_param_ = &register_parameter("weight");
}

WeightParameter& RMSNorm::weight_param() {
    if (!weight_param_) {
        OPENVINO_THROW("RMSNorm has no registered parameter");
    }
    return *weight_param_;
}

const WeightParameter& RMSNorm::weight_param() const {
    if (!weight_param_) {
        OPENVINO_THROW("RMSNorm has no registered parameter");
    }
    return *weight_param_;
}

const Tensor& RMSNorm::weight() const {
    if (weight_param_) {
        return weight_param_->value();
    }
    return weight_;
}

Tensor RMSNorm::forward(const Tensor& x) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto var = xf.pow(2.0f).mean(-1, true);
    auto norm = xf * (var + eps_).rsqrt();
    auto w = weight().to(orig_dtype);
    return norm.to(orig_dtype) * w;
}

std::pair<Tensor, Tensor> RMSNorm::forward(const Tensor& x, const Tensor& residual) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto rf = residual.to(ov::element::f32);
    auto sum = xf + rf;
    auto residual_out = sum.to(orig_dtype);
    auto var = sum.pow(2.0f).mean(-1, true);
    auto norm = sum * (var + eps_).rsqrt();
    auto w = weight().to(orig_dtype);
    auto out = norm.to(orig_dtype) * w;
    return {out, residual_out};
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov

