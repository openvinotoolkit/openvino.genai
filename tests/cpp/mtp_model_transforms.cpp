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
using ov::genai::utils::mtp::expose_last_hidden_state;
using ov::genai::utils::mtp::extract_mtp_info_from_config;
using ov::genai::utils::mtp::extract_tied_lm_head_weight;
using ov::genai::utils::mtp::graft_lm_head_on_mtp;

constexpr size_t HIDDEN = 8;
constexpr size_t VOCAB = 16;
// Qwen3.8 (Qwen4Exp) hyper-connection stream count.
constexpr size_t HC_COUNT = 4;
constexpr const char* HC_STREAMS_PRODUCER = "hc_streams";

// Main model stand-in with logits and last_hidden_state outputs.
std::shared_ptr<ov::Model> make_main_model(const std::vector<float>& weights_data,
                                           bool matching_result_friendly_names = true) {
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, HIDDEN});
    embeds->output(0).set_names({"inputs_embeds"});

    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HIDDEN}, 0.5f);
    auto hidden = std::make_shared<ov::op::v1::Add>(embeds, bias);

    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{VOCAB, HIDDEN}, weights_data.data());
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(hidden, weights, false, true);

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    logits_result->set_friendly_name(matching_result_friendly_names ? "logits" : "main_logits_result");

    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden);
    hidden_result->output(0).set_names({"last_hidden_state"});
    hidden_result->set_friendly_name(matching_result_friendly_names ? "last_hidden_state" : "main_hidden_result");

    return std::make_shared<ov::Model>(ov::ResultVector{logits_result, hidden_result},
                                       ov::ParameterVector{embeds});
}

// MTP stand-in: hidden_states -> last_hidden_state only.
std::shared_ptr<ov::Model> make_mtp_model(bool matching_result_friendly_name = true) {
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, HIDDEN});
    hidden_states->output(0).set_names({"hidden_states"});

    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HIDDEN}, 1.0f);
    auto out = std::make_shared<ov::op::v1::Add>(hidden_states, bias);

    auto result = std::make_shared<ov::op::v0::Result>(out);
    result->output(0).set_names({"last_hidden_state"});
    result->set_friendly_name(matching_result_friendly_name ? "last_hidden_state" : "mtp_hidden_result");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden_states});
}

// Qwen3.8 (Qwen4Exp) main model stand-in: lm_head consumes a reduction of the wider hyper-connection streams,
// the logits output is the only result.
std::shared_ptr<ov::Model> make_hyper_connection_main_model() {
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, HC_COUNT * HIDDEN});
    embeds->output(0).set_names({"inputs_embeds"});

    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HC_COUNT * HIDDEN}, 0.5f);
    auto streams = std::make_shared<ov::op::v1::Add>(embeds, bias);
    streams->set_friendly_name(HC_STREAMS_PRODUCER);

    auto reduce_weights =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HC_COUNT * HIDDEN, HIDDEN}, 0.25f);
    auto hidden = std::make_shared<ov::op::v0::MatMul>(streams, reduce_weights);

    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{VOCAB, HIDDEN}, 0.1f);
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(hidden, weights, false, true);

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});

    return std::make_shared<ov::Model>(ov::ResultVector{logits_result}, ov::ParameterVector{embeds});
}

// Qwen3.8 (Qwen4Exp) MTP stand-in: hyper-connection hidden_states -> next_hidden_states and last_hidden_state.
std::shared_ptr<ov::Model> make_hyper_connection_mtp_model() {
    auto hidden_states =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, HC_COUNT * HIDDEN});
    hidden_states->output(0).set_names({"hidden_states"});

    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HC_COUNT * HIDDEN}, 1.0f);
    auto next_streams = std::make_shared<ov::op::v1::Add>(hidden_states, bias);

    auto reduce_weights =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{HC_COUNT * HIDDEN, HIDDEN}, 0.25f);
    auto hidden = std::make_shared<ov::op::v0::MatMul>(next_streams, reduce_weights);

    auto next_result = std::make_shared<ov::op::v0::Result>(next_streams);
    next_result->output(0).set_names({"next_hidden_states"});
    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden);
    hidden_result->output(0).set_names({"last_hidden_state"});

    return std::make_shared<ov::Model>(ov::ResultVector{next_result, hidden_result},
                                       ov::ParameterVector{hidden_states});
}

void set_mtp_hidden_states_annotation(const std::shared_ptr<ov::Model>& model,
                                      const std::string& producer,
                                      size_t hidden_size) {
    model->set_rt_info("{\"producer\": \"" + producer + "\", \"output_index\": 0, \"hidden_size\": " +
                           std::to_string(hidden_size) + "}",
                       "mtp_hidden_states");
}

std::shared_ptr<ov::op::v0::Result> find_result(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& result : model->get_results()) {
        if (result->output(0).get_names().count(name) != 0) {
            return result;
        }
    }
    return nullptr;
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

TEST(MtpModelTransforms, ExposeLastHiddenUsesLLMHeadWithoutAnnotation) {
    std::vector<float> weights_data(VOCAB * HIDDEN, 0.1f);
    auto main_model = make_main_model(weights_data);
    expose_last_hidden_state(main_model);
    EXPECT_EQ(main_model->get_results().size(), 2u);

    const auto hidden_result = find_result(main_model, "last_hidden_state");
    ASSERT_TRUE(hidden_result);
    EXPECT_TRUE(ov::as_type_ptr<ov::op::v0::MatMul>(hidden_result->get_input_node_shared_ptr(0)));
    EXPECT_EQ(hidden_result->get_input_partial_shape(0), (ov::PartialShape{1, -1, HIDDEN}));
}


TEST(MtpModelTransforms, Qwen38ExposeLastHiddenStateUsesAnnotatedProducer) {
    auto main_model = make_hyper_connection_main_model();
    set_mtp_hidden_states_annotation(main_model, HC_STREAMS_PRODUCER, HC_COUNT * HIDDEN);
    expose_last_hidden_state(main_model);

    const auto hidden_result = find_result(main_model, "last_hidden_state");
    ASSERT_TRUE(hidden_result);
    EXPECT_EQ(hidden_result->get_input_node_shared_ptr(0)->get_friendly_name(), HC_STREAMS_PRODUCER);
    EXPECT_EQ(hidden_result->get_input_partial_shape(0), (ov::PartialShape{1, -1, HC_COUNT * HIDDEN}));
    EXPECT_TRUE(hidden_result->get_rt_info().at("manually_added_output").as<bool>());

    ov::Core core;
    auto request = core.compile_model(main_model, "CPU").create_infer_request();
    const size_t seq_len = 2;
    std::vector<float> embeds(seq_len * HC_COUNT * HIDDEN);
    for (size_t i = 0; i < embeds.size(); ++i) {
        embeds[i] = static_cast<float>(i) * 0.01f;
    }
    request.set_tensor("inputs_embeds",
                       ov::Tensor(ov::element::f32, ov::Shape{1, seq_len, HC_COUNT * HIDDEN}, embeds.data()));
    request.infer();

    const auto hidden = request.get_tensor("last_hidden_state");
    ASSERT_EQ(hidden.get_shape(), (ov::Shape{1, seq_len, HC_COUNT * HIDDEN}));
    const float* hidden_data = hidden.data<float>();
    for (size_t i = 0; i < embeds.size(); ++i) {
        EXPECT_NEAR(hidden_data[i], embeds[i] + 0.5f, 1e-5f) << "Mismatch at element " << i;
    }
}

TEST(MtpModelTransforms, Qwen38ExposeLastHiddenStateRejectsHiddenSizeMismatch) {
    auto main_model = make_hyper_connection_main_model();
    set_mtp_hidden_states_annotation(main_model, HC_STREAMS_PRODUCER, HIDDEN);
    EXPECT_THROW(expose_last_hidden_state(main_model), ov::Exception);
}

TEST(MtpModelTransforms, Qwen38GraftLmHeadOnHyperConnectionMtp) {
    auto main_model = make_hyper_connection_main_model();
    auto mtp_model = make_hyper_connection_mtp_model();
    graft_lm_head_on_mtp(mtp_model, main_model);

    ASSERT_EQ(mtp_model->get_results().size(), 3u);
    const auto logits_result = find_result(mtp_model, "logits");
    ASSERT_TRUE(logits_result);
    EXPECT_EQ(logits_result->get_input_partial_shape(0), (ov::PartialShape{1, -1, VOCAB}));
    const auto next_hidden_result = find_result(mtp_model, "next_hidden_states");
    ASSERT_TRUE(next_hidden_result);
    EXPECT_EQ(next_hidden_result->get_input_partial_shape(0), (ov::PartialShape{1, -1, HC_COUNT * HIDDEN}));
}

TEST(MtpModelTransforms, FindsOutputsByResultTensorName) {
    std::vector<float> weights_data(VOCAB * HIDDEN, 0.1f);
    auto main_model = make_main_model(weights_data, false);
    auto mtp_model = make_mtp_model(false);

    bool transpose_weight = false;
    EXPECT_TRUE(extract_tied_lm_head_weight(main_model, transpose_weight).get_node());
    EXPECT_NO_THROW(graft_lm_head_on_mtp(mtp_model, main_model));
}

TEST(MtpModelTransforms, GraftLmHeadAddsLogitsResult) {
    std::vector<float> weights_data(VOCAB * HIDDEN, 0.1f);
    auto main_model = make_main_model(weights_data);
    auto mtp_model = make_mtp_model();

    ASSERT_EQ(mtp_model->get_results().size(), 1u);
    graft_lm_head_on_mtp(mtp_model, main_model);

    ASSERT_EQ(mtp_model->get_results().size(), 2u);
    bool has_logits = false, has_hidden = false;
    for (const auto& result : mtp_model->get_results()) {
        const auto& names = result->input_value(0).get_names();
        has_logits |= names.count("logits") > 0;
        has_hidden |= names.count("last_hidden_state") > 0;
    }
    EXPECT_TRUE(has_logits);
    EXPECT_TRUE(has_hidden);

    EXPECT_GE(count_ops<ov::op::v0::Constant>(mtp_model), 2u);
}

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
