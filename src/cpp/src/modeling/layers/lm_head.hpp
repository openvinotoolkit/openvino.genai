// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {

class LMHead : public Module {
public:
    explicit LMHead(const Tensor& weight);
    LMHead(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    // Decode path: compute logits for all tokens in `x`.
    Tensor forward(const Tensor& x) const;

    // Prefill path: `x` is expected to be packed as [total_tokens, hidden],
    // `cu_seqlens_q` is expected to be [batch + 1] (prefix sums).
    // The layer selects the last token per sequence, then computes logits.
    Tensor forward(const Tensor& x, const Tensor& cu_seqlens_q) const;

    void tie_to(WeightParameter& other);
    WeightParameter& weight_param();
    const WeightParameter& weight_param() const;

private:
    const Tensor& weight() const;

    Tensor weight_;
    WeightParameter* weight_param_ = nullptr;
};

}  // namespace modeling
}  // namespace genai
}  // namespace ov
