// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/core/model.hpp"

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "utils.hpp"

namespace {

// Parameter -> MatMul -> Result : contains no ScaledDotProductAttention op.
std::shared_ptr<ov::Model> make_model_without_sdpa() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 8});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{8, 8}, 0.0f);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weights);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

// Parameter(q, k, v) -> ScaledDotProductAttention -> Result.
std::shared_ptr<ov::Model> make_model_with_sdpa() {
    const ov::PartialShape qkv_shape{1, 4, 8, 16};
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, qkv_shape);
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, false);
    auto result = std::make_shared<ov::op::v0::Result>(sdpa);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{query, key, value});
}

}  // namespace

// A ScaledDotProductAttention op is necessary but NOT sufficient: a bare SDPA without the
// stateful KV-cache / attention_mask / beam_idx structure of an LLM does not pass
// SDPAToPagedAttention, so the model is correctly reported as not supporting paged attention.
TEST(SupportsPagedAttention, FalseForBareScaledDotProductAttention) {
    auto model = make_model_with_sdpa();
    EXPECT_FALSE(ov::genai::supports_paged_attention(model));
    EXPECT_FALSE(ov::genai::utils::supports_paged_attention(model));
}

// A model without any ScaledDotProductAttention op (e.g. some Gemma exports) cannot be
// converted to paged attention and must fall back to a non-paged pipeline.
TEST(SupportsPagedAttention, FalseWhenModelHasNoScaledDotProductAttention) {
    auto model = make_model_without_sdpa();
    EXPECT_FALSE(ov::genai::supports_paged_attention(model));
    EXPECT_FALSE(ov::genai::utils::supports_paged_attention(model));
}
