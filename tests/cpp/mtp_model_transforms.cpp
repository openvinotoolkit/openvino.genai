// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding/mtp_model_transforms.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using ov::genai::utils::mtp::apply_mtp_rt_info;
using ov::genai::utils::mtp::expose_last_hidden_state;
using ov::genai::utils::mtp::extract_mtp_info_from_config;
using ov::genai::utils::mtp::extract_tied_lm_head_weight;
using ov::genai::utils::mtp::graft_lm_head_on_mtp;
using ov::genai::utils::mtp::restrict_mtp_draft_vocab;

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

// Main model stand-in whose lm_head weight is u8 grouped-quantized like an int8 optimum-intel export:
// Constant u8 [VOCAB, GROUPS, GROUP_SIZE] -> Convert -> Subtract(zero point) -> Multiply(scale) -> Reshape -> MatMul.
std::shared_ptr<ov::Model> make_compressed_main_model() {
    constexpr size_t GROUPS = 2;
    constexpr size_t GROUP_SIZE = HIDDEN / GROUPS;
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, HIDDEN});
    embeds->output(0).set_names({"inputs_embeds"});

    std::vector<uint8_t> weight_data(VOCAB * HIDDEN);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<uint8_t>((i * 37) % 251);
    }
    std::vector<uint8_t> zero_point_data(VOCAB * GROUPS);
    std::vector<float> scale_data(VOCAB * GROUPS);
    for (size_t i = 0; i < zero_point_data.size(); ++i) {
        zero_point_data[i] = static_cast<uint8_t>(120 + i % 9);
        scale_data[i] = 0.001f * static_cast<float>(1 + i % 13);
    }
    auto weight =
        std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{VOCAB, GROUPS, GROUP_SIZE}, weight_data);
    auto zero_point =
        std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{VOCAB, GROUPS, 1}, zero_point_data);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{VOCAB, GROUPS, 1}, scale_data);
    auto weight_f32 = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32);
    auto zero_point_f32 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f32);
    auto shifted = std::make_shared<ov::op::v1::Subtract>(weight_f32, zero_point_f32);
    auto scaled = std::make_shared<ov::op::v1::Multiply>(shifted, scale);
    auto pattern = ov::op::v0::Constant::create(ov::element::i64,
                                                ov::Shape{2},
                                                {static_cast<int64_t>(VOCAB), static_cast<int64_t>(HIDDEN)});
    auto decompressed = std::make_shared<ov::op::v1::Reshape>(scaled, pattern, false);
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(embeds, decompressed, false, true);

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{logits_result}, ov::ParameterVector{embeds});
}

// Runs the MTP model on fixed hidden states and returns the logits as [seq_len * vocab_size].
std::vector<float> infer_mtp_logits(const std::shared_ptr<ov::Model>& mtp_model, size_t seq_len) {
    std::vector<float> hidden_in(seq_len * HIDDEN);
    for (size_t i = 0; i < hidden_in.size(); ++i) {
        hidden_in[i] = static_cast<float>((i * 5) % 17) * 0.07f - 0.5f;
    }
    ov::Core core;
    auto request = core.compile_model(mtp_model, "CPU").create_infer_request();
    request.set_tensor("hidden_states", ov::Tensor(ov::element::f32, ov::Shape{1, seq_len, HIDDEN}, hidden_in.data()));
    request.infer();
    const auto logits = request.get_tensor("logits");
    EXPECT_EQ(logits.get_shape(), (ov::Shape{1, seq_len, VOCAB}));
    return {logits.data<float>(), logits.data<float>() + logits.get_size()};
}

void expect_restricted_logits(const std::vector<float>& full,
                              const std::vector<float>& restricted,
                              const std::vector<int64_t>& draft_vocab,
                              size_t seq_len) {
    ASSERT_EQ(full.size(), restricted.size());
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t v = 0; v < VOCAB; ++v) {
            const float value = restricted[t * VOCAB + v];
            if (std::find(draft_vocab.begin(), draft_vocab.end(), static_cast<int64_t>(v)) != draft_vocab.end()) {
                EXPECT_NEAR(value, full[t * VOCAB + v], 1e-4f) << "token " << t << ", vocab " << v;
            } else {
                EXPECT_TRUE(std::isinf(value) && value < 0.0f) << "token " << t << ", vocab " << v << ": " << value;
            }
        }
    }
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

TEST(MtpModelTransforms, ExtractMtpDraftVocab) {
    {
        ov::AnyMap config{{"mtp_mode", true}, {"mtp_draft_vocab", std::vector<int64_t>{9, 2, 5, 2}}};
        auto info = extract_mtp_info_from_config(config);
        EXPECT_EQ(info.draft_vocab, (std::vector<int64_t>{2, 5, 9}));  // sorted, duplicates removed
        EXPECT_EQ(config.find("mtp_draft_vocab"), config.end());
    }
    {
        ov::AnyMap config{{"mtp_mode", true}, {"mtp_draft_vocab", std::vector<int32_t>{3, 1}}};
        EXPECT_EQ(extract_mtp_info_from_config(config).draft_vocab, (std::vector<int64_t>{1, 3}));
    }
    {
        ov::AnyMap config{{"mtp_mode", true}};
        EXPECT_TRUE(extract_mtp_info_from_config(config).draft_vocab.empty());
    }
    {
        ov::AnyMap config{{"mtp_draft_vocab", std::vector<int64_t>{1}}};  // not an MTP draft
        EXPECT_THROW(extract_mtp_info_from_config(config), ov::Exception);
    }
    {
        ov::AnyMap config{{"mtp_mode", true}, {"mtp_draft_vocab", std::vector<int64_t>{}}};
        EXPECT_THROW(extract_mtp_info_from_config(config), ov::Exception);
    }
    {
        ov::AnyMap config{{"mtp_mode", true}, {"mtp_draft_vocab", std::vector<int64_t>{-1, 3}}};
        EXPECT_THROW(extract_mtp_info_from_config(config), ov::Exception);
    }
    {
        ov::AnyMap config{{"mtp_mode", true}, {"mtp_draft_vocab", std::string("1 2 3")}};
        EXPECT_THROW(extract_mtp_info_from_config(config), ov::Exception);
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
    auto main_model = make_hyper_connection_main_model();
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

TEST(MtpModelTransforms, RestrictDraftVocabKeepsSubsetLogits) {
    std::vector<float> weights_data(VOCAB * HIDDEN);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = static_cast<float>((i * 7) % 11) * 0.05f - 0.2f;
    }
    auto main_model = make_main_model(weights_data);
    auto full_mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(full_mtp_model, main_model);
    auto restricted_mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(restricted_mtp_model, main_model);

    const std::vector<int64_t> draft_vocab{0, 3, 4, 10, VOCAB - 1};
    restrict_mtp_draft_vocab(restricted_mtp_model, draft_vocab);

    const auto logits_result = find_result(restricted_mtp_model, "logits");
    ASSERT_TRUE(logits_result);
    EXPECT_EQ(logits_result->get_input_partial_shape(0), (ov::PartialShape{1, -1, VOCAB}));
    // The main model keeps its full lm_head.
    EXPECT_EQ(main_model->get_results().front()->get_input_partial_shape(0), (ov::PartialShape{1, -1, VOCAB}));
    for (const auto& op : main_model->get_ordered_ops()) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op); constant && constant->get_shape().size() == 2) {
            EXPECT_EQ(constant->get_shape(), (ov::Shape{VOCAB, HIDDEN}));
        }
    }

    const size_t seq_len = 3;
    expect_restricted_logits(infer_mtp_logits(full_mtp_model, seq_len),
                             infer_mtp_logits(restricted_mtp_model, seq_len),
                             draft_vocab,
                             seq_len);
}

TEST(MtpModelTransforms, RestrictDraftVocabSlicesCompressedWeight) {
    auto main_model = make_compressed_main_model();
    auto full_mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(full_mtp_model, main_model);
    auto restricted_mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(restricted_mtp_model, main_model);

    const std::vector<int64_t> draft_vocab{1, 2, 7, 8, 13};
    restrict_mtp_draft_vocab(restricted_mtp_model, draft_vocab);

    // Weight, zero point and scale are sliced to the subset rows; no constant keeps the full vocabulary.
    size_t num_subset_constants = 0;
    for (const auto& op : restricted_mtp_model->get_ordered_ops()) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op); constant && constant->get_shape().size() == 3) {
            EXPECT_EQ(constant->get_shape()[0], draft_vocab.size());
            ++num_subset_constants;
        }
    }
    EXPECT_EQ(num_subset_constants, 3u);

    const size_t seq_len = 2;
    expect_restricted_logits(infer_mtp_logits(full_mtp_model, seq_len),
                             infer_mtp_logits(restricted_mtp_model, seq_len),
                             draft_vocab,
                             seq_len);
}

TEST(MtpModelTransforms, RestrictDraftVocabRejectsOutOfRangeId) {
    std::vector<float> weights_data(VOCAB * HIDDEN, 0.1f);
    auto main_model = make_main_model(weights_data);
    auto mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(mtp_model, main_model);
    EXPECT_THROW(restrict_mtp_draft_vocab(mtp_model, {1, VOCAB}), ov::Exception);
}

TEST(MtpModelTransforms, RestrictDraftVocabEmptyIsNoOp) {
    std::vector<float> weights_data(VOCAB * HIDDEN, 0.1f);
    auto main_model = make_main_model(weights_data);
    auto mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(mtp_model, main_model);
    const size_t num_ops = mtp_model->get_ordered_ops().size();
    restrict_mtp_draft_vocab(mtp_model, {});
    EXPECT_EQ(mtp_model->get_ordered_ops().size(), num_ops);
}
