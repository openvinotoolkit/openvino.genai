// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class RMSNorm : public Module {
public:
    RMSNorm(const Tensor& weight, float eps);
    RMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;
    std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& residual) const;
    WeightParameter& weight_param();
    const WeightParameter& weight_param() const;

private:
    const Tensor& weight() const;

    Tensor weight_;
    WeightParameter* weight_param_ = nullptr;
    float eps_ = 1e-6f;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
