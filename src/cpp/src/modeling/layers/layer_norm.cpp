// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layers/layer_norm.hpp"

#include <openvino/core/except.hpp>

namespace ov {
namespace genai {
namespace modeling {

FP32LayerNorm::FP32LayerNorm(const Tensor& weight, const Tensor* bias, float eps)
    : Module(),
      weight_(weight),
      eps_(eps),
      elementwise_affine_(true),
      has_const_bias_(bias != nullptr) {
    if (bias) {
        bias_ = *bias;
    }
}

FP32LayerNorm::FP32LayerNorm(BuilderContext& ctx,
                             const std::string& name,
                             float eps,
                             bool elementwise_affine,
                             bool bias,
                             Module* parent)
    : Module(name, ctx, parent),
      eps_(eps),
      elementwise_affine_(elementwise_affine) {
    if (elementwise_affine_) {
        weight_param_ = &register_parameter("weight");
        if (bias) {
            bias_param_ = &register_parameter("bias");
        }
    }
}

WeightParameter& FP32LayerNorm::weight_param() {
    if (!weight_param_) {
        OPENVINO_THROW("FP32LayerNorm has no weight parameter");
    }
    return *weight_param_;
}

WeightParameter& FP32LayerNorm::bias_param() {
    if (!bias_param_) {
        OPENVINO_THROW("FP32LayerNorm has no bias parameter");
    }
    return *bias_param_;
}

const WeightParameter& FP32LayerNorm::weight_param() const {
    if (!weight_param_) {
        OPENVINO_THROW("FP32LayerNorm has no weight parameter");
    }
    return *weight_param_;
}

const WeightParameter& FP32LayerNorm::bias_param() const {
    if (!bias_param_) {
        OPENVINO_THROW("FP32LayerNorm has no bias parameter");
    }
    return *bias_param_;
}

const Tensor& FP32LayerNorm::weight() const {
    if (weight_param_) {
        return weight_param_->value();
    }
    return weight_;
}

const Tensor* FP32LayerNorm::bias() const {
    if (bias_param_) {
        return &bias_param_->value();
    }
    if (has_const_bias_) {
        return &bias_;
    }
    return nullptr;
}

Tensor FP32LayerNorm::forward(const Tensor& x) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto mean = xf.mean(-1, true);
    auto diff = xf - mean;
    auto var = diff.pow(2.0f).mean(-1, true);
    auto norm = diff * (var + eps_).rsqrt();

    if (elementwise_affine_) {
        norm = norm * weight().to(ov::element::f32);
        if (const Tensor* b = bias()) {
            norm = norm + b->to(ov::element::f32);
        }
    }
    return norm.to(orig_dtype);
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov
