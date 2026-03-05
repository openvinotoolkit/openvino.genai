// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::genai {
class WhisperScaledDotProductAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("WhisperScaledDotProductAttentionDecomposition");
    WhisperScaledDotProductAttentionDecomposition();
    std::shared_ptr<ov::Node> decompose(std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node);
};
}  // namespace ov::genai
