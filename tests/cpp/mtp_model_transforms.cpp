// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

#include "speculative_decoding/mtp_model_transforms.hpp"

namespace {

using ov::genai::utils::mtp::apply_mtp_rt_info;
using ov::genai::utils::mtp::extract_mtp_info_from_config;
using ov::genai::utils::mtp::extract_tied_lm_head_weight;
using ov::genai::utils::mtp::graft_lm_head_on_mtp;

constexpr size_t HIDDEN = 8;
constexpr size_t VOCAB = 16;

// Main-model stand-in: inputs_embeds -> (identity hidden) -> logits = MatMul(hidden, W^T) + last_hidden_state.
// Mirrors the real Qwen3.5 language_model which outputs both `logits` and `last_hidden_state`.
std::shared_ptr<ov::Model> make_main_model(const std::vector<float>& weights_data) {
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, HIDDEN});
    embeds->output(0).set_names({"inputs_embeds"});

    // A trivial transform standing in for the decoder stack.
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HIDDEN}, 0.5f);
    auto hidden = std::make_shared<ov::op::v1::Add>(embeds, bias);

    // tied lm_head weight [VOCAB, HIDDEN], applied with transpose_b=true.
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{VOCAB, HIDDEN}, weights_data.data());
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(hidden, weights, false, true);

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    logits_result->set_friendly_name("logits");

    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden);
    hidden_result->output(0).set_names({"last_hidden_state"});
    hidden_result->set_friendly_name("last_hidden_state");

    return std::make_shared<ov::Model>(ov::ResultVector{logits_result, hidden_result},
                                       ov::ParameterVector{embeds});
}

// MTP-model stand-in: hidden_states -> last_hidden_state only (no lm_head), like openvino_mtp_model.xml.
std::shared_ptr<ov::Model> make_mtp_model() {
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, HIDDEN});
    hidden_states->output(0).set_names({"hidden_states"});

    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HIDDEN}, 1.0f);
    auto out = std::make_shared<ov::op::v1::Add>(hidden_states, bias);

    auto result = std::make_shared<ov::op::v0::Result>(out);
    result->output(0).set_names({"last_hidden_state"});
    result->set_friendly_name("last_hidden_state");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden_states});
}

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::as_type_ptr<T>(op)) {
            count++;
        }
    }
    return count;
}

}  // namespace

TEST(MtpModelTransforms, ExtractMtpInfoFromConfig) {
    {
        ov::AnyMap config{{"mtp_mode", true}, {"device", std::string("CPU")}};
        auto info = extract_mtp_info_from_config(config);
        EXPECT_TRUE(info.mtp_mode);
        EXPECT_EQ(config.find("mtp_mode"), config.end());  // consumed key erased
        EXPECT_NE(config.find("device"), config.end());    // other keys preserved
    }
    {
        ov::AnyMap config;
        auto info = extract_mtp_info_from_config(config);
        EXPECT_FALSE(info.mtp_mode);
    }
}

TEST(MtpModelTransforms, ApplyMtpRtInfo) {
    auto model = make_mtp_model();
    model->set_rt_info(true, "mtp_mode");
    ov::AnyMap properties;
    apply_mtp_rt_info(model, properties);
    ASSERT_NE(properties.find("mtp_mode"), properties.end());
    EXPECT_TRUE(properties.at("mtp_mode").as<bool>());
}

TEST(MtpModelTransforms, ExtractTiedLmHeadWeight) {
    std::vector<float> weights_data(VOCAB * HIDDEN);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = static_cast<float>(i % 5) * 0.1f;
    }
    auto main_model = make_main_model(weights_data);

    bool transpose_weight = false;
    auto weight = extract_tied_lm_head_weight(main_model, transpose_weight);
    ASSERT_TRUE(weight.get_node());
    EXPECT_TRUE(transpose_weight);
    EXPECT_EQ(weight.get_partial_shape(), (ov::PartialShape{VOCAB, HIDDEN}));
}

TEST(MtpModelTransforms, GraftLmHeadAddsLogitsResult) {
    std::vector<float> weights_data(VOCAB * HIDDEN, 0.1f);
    auto main_model = make_main_model(weights_data);
    auto mtp_model = make_mtp_model();

    ASSERT_EQ(mtp_model->get_results().size(), 1u);
    graft_lm_head_on_mtp(mtp_model, main_model);

    // A `logits` result is added while `last_hidden_state` is preserved.
    ASSERT_EQ(mtp_model->get_results().size(), 2u);
    bool has_logits = false, has_hidden = false;
    for (const auto& result : mtp_model->get_results()) {
        const auto& names = result->input_value(0).get_names();
        has_logits |= names.count("logits") > 0;
        has_hidden |= names.count("last_hidden_state") > 0;
    }
    EXPECT_TRUE(has_logits);
    EXPECT_TRUE(has_hidden);

    // The cloned weight lives in the MTP model (deep copy, no cross-model reference).
    EXPECT_GE(count_ops<ov::op::v0::Constant>(mtp_model), 2u);
}

// The grafted MTP logits must equal main_hidden @ W^T for the same hidden input the main model
// produces, i.e. logits = MTP(hidden) @ tied_weight.
TEST(MtpModelTransforms, GraftedLogitsMatchTiedWeightMatmul) {
    std::vector<float> weights_data(VOCAB * HIDDEN);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = static_cast<float>((i * 7) % 11) * 0.05f - 0.2f;
    }
    auto main_model = make_main_model(weights_data);
    auto mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(mtp_model, main_model);

    ov::Core core;
    auto mtp_req = core.compile_model(mtp_model, "CPU").create_infer_request();

    const size_t seq_len = 3;
    std::vector<float> hidden_in(seq_len * HIDDEN);
    for (size_t i = 0; i < hidden_in.size(); ++i) {
        hidden_in[i] = static_cast<float>(i) * 0.03f - 0.1f;
    }
    mtp_req.set_tensor("hidden_states", ov::Tensor(ov::element::f32, ov::Shape{1, seq_len, HIDDEN}, hidden_in.data()));
    mtp_req.infer();

    auto logits = mtp_req.get_tensor("logits");
    auto mtp_hidden = mtp_req.get_tensor("last_hidden_state");
    ASSERT_EQ(logits.get_shape(), (ov::Shape{1, seq_len, VOCAB}));

    // Reference: logits[t, v] = sum_h mtp_hidden[t, h] * W[v, h]  (transpose_b).
    const float* hidden_data = mtp_hidden.data<float>();
    const float* logits_data = logits.data<float>();
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t v = 0; v < VOCAB; ++v) {
            float expected = 0.0f;
            for (size_t h = 0; h < HIDDEN; ++h) {
                expected += hidden_data[t * HIDDEN + h] * weights_data[v * HIDDEN + h];
            }
            EXPECT_NEAR(logits_data[t * VOCAB + v], expected, 1e-4f)
                << "Mismatch at token " << t << ", vocab " << v;
        }
    }
}
