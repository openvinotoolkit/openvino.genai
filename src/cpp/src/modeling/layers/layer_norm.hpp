// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class FP32LayerNorm : public Module {
public:
    FP32LayerNorm(const Tensor& weight, const Tensor* bias, float eps);
    FP32LayerNorm(BuilderContext& ctx,
                  const std::string& name,
                  float eps,
                  bool elementwise_affine = true,
                  bool bias = true,
                  Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

    WeightParameter& weight_param();
    WeightParameter& bias_param();
    const WeightParameter& weight_param() const;
    const WeightParameter& bias_param() const;

private:
    const Tensor& weight() const;
    const Tensor* bias() const;

    Tensor weight_;
    Tensor bias_;
    WeightParameter* weight_param_ = nullptr;
    WeightParameter* bias_param_ = nullptr;
    float eps_ = 1e-6f;
    bool elementwise_affine_ = true;
    bool has_const_bias_ = false;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
