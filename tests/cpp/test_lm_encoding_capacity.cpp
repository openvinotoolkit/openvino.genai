// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "lm_encoding.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

namespace {

ov::InferRequest make_request() {
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    mask->output(0).set_names({"attention_mask"});
    auto beams = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    beams->output(0).set_names({"beam_idx"});
    auto logits = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 4}, {0.f, 1.f, 0.f, 0.f});
    logits->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{logits}, ov::ParameterVector{ids, mask, beams});
    ov::Core core;
    return core.compile_model(model, "CPU").create_infer_request();
}

void check_capacity(size_t max_new_tokens, bool expected_finished) {
    auto request = make_request();
    ov::genai::GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.ignore_eos = true;
    auto group = std::make_shared<ov::genai::SequenceGroup>(0, ov::genai::TokenIds{0}, config);
    ov::Tensor ids{ov::element::i64, {1, 1}};
    ov::Tensor mask{ov::element::i64, {1, 1}};
    ids.data<int64_t>()[0] = 0;
    mask.data<int64_t>()[0] = 1;
    ov::genai::Sampler sampler;
    ov::genai::utils::CacheState cache_state;
    auto result = ov::genai::get_lm_encoded_results(request,
                                                    ids,
                                                    mask,
                                                    nullptr,
                                                    sampler,
                                                    {group},
                                                    std::nullopt,
                                                    cache_state,
                                                    nullptr,
                                                    std::nullopt,
                                                    1);
    ASSERT_EQ(result.results.tokens.size(), 1u);
    ASSERT_EQ(result.results.tokens[0].size(), 1u);
    const auto sequence = group->get_sequences().front();
    EXPECT_EQ(sequence->has_finished(), expected_finished);
    EXPECT_EQ(sequence->out_of_memory(), !expected_finished);
}

TEST(LMEncodingCapacityTest, KeepsFinishedSequenceAtCapacity) {
    check_capacity(1, true);
}

TEST(LMEncodingCapacityTest, StopsRunningSequenceAtCapacity) {
    check_capacity(2, false);
}

}  // namespace
