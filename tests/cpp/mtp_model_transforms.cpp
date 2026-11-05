// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_set>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
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

TEST(MtpModelTransforms, GraftPreservesCheckpointDraftHead) {
    std::vector<float> main_weights(VOCAB * HIDDEN, 0.1f);
    std::vector<float> draft_weights(VOCAB * HIDDEN, 0.2f);
    auto main_model = make_main_model(main_weights);
    auto draft_model = make_main_model(draft_weights);
    const auto output = draft_model->output("logits").get_node_shared_ptr();
    graft_lm_head_on_mtp(draft_model, main_model);
    EXPECT_EQ(draft_model->get_results().size(), 2);
    EXPECT_EQ(draft_model->output("logits").get_node_shared_ptr(), output);
}

TEST(MtpModelTransforms, GraftSharesPackedWeightStorageAndRetainsLifetime) {
    auto mtp_model = make_mtp_model();
    std::weak_ptr<ov::Node> source_packed_weight;
    std::weak_ptr<ov::Node> source_scales;
    std::shared_ptr<ov::op::v0::Constant> cloned_packed_weight;
    std::shared_ptr<ov::op::v0::Constant> cloned_scales;
    {
        auto main_model = make_main_model(std::vector<float>(VOCAB * HIDDEN, 0.1f));
        std::vector<uint8_t> packed_data(VOCAB * HIDDEN / 4, 0xe4);
        auto packed_weight =
            std::make_shared<ov::op::v0::Constant>(ov::element::u2, ov::Shape{VOCAB, HIDDEN}, packed_data.data());
        auto scales = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{VOCAB, 1}, 0.25f);
        packed_weight->set_friendly_name("packed_lm_head");
        packed_weight->get_rt_info()["storage_tag"] = std::string("packed_source");
        packed_weight->output(0).set_names({"packed_lm_head_data"});
        packed_weight->output(0).get_rt_info()["tensor_tag"] = std::string("packed_tensor");
        scales->set_friendly_name("lm_head_scales");
        scales->get_rt_info()["storage_tag"] = std::string("scales_source");
        scales->output(0).set_names({"lm_head_scales_data"});
        auto converted_weight = std::make_shared<ov::op::v0::Convert>(packed_weight, ov::element::f32);
        auto decompressed_weight = std::make_shared<ov::op::v1::Multiply>(converted_weight, scales);
        decompressed_weight->get_rt_info()["decompression_tag"] = std::string("source");
        decompressed_weight->input(0).get_rt_info()["input_tag"] = std::string("packed_input");
        auto main_matmul =
            ov::as_type_ptr<ov::op::v0::MatMul>(main_model->get_results()[0]->input_value(0).get_node_shared_ptr());
        ASSERT_TRUE(main_matmul);
        main_matmul->input(1).replace_source_output(decompressed_weight);
        main_model->validate_nodes_and_infer_types();
        source_packed_weight = packed_weight;
        source_scales = scales;

        graft_lm_head_on_mtp(mtp_model, main_model);
        bool transpose_weight = false;
        auto cloned_weight = extract_tied_lm_head_weight(mtp_model, transpose_weight).get_node_shared_ptr();
        auto cloned_multiply = ov::as_type_ptr<ov::op::v1::Multiply>(cloned_weight);
        ASSERT_TRUE(cloned_multiply);
        auto cloned_convert = ov::as_type_ptr<ov::op::v0::Convert>(cloned_multiply->get_input_node_shared_ptr(0));
        ASSERT_TRUE(cloned_convert);
        cloned_packed_weight = ov::as_type_ptr<ov::op::v0::Constant>(cloned_convert->get_input_node_shared_ptr(0));
        cloned_scales = ov::as_type_ptr<ov::op::v0::Constant>(cloned_multiply->get_input_node_shared_ptr(1));
        ASSERT_TRUE(cloned_packed_weight);
        ASSERT_TRUE(cloned_scales);
        EXPECT_NE(cloned_multiply, decompressed_weight);
        EXPECT_NE(cloned_convert, converted_weight);
        EXPECT_NE(cloned_packed_weight, packed_weight);
        EXPECT_NE(cloned_scales, scales);
        EXPECT_EQ(cloned_packed_weight->get_data_ptr(), packed_weight->get_data_ptr());
        EXPECT_EQ(cloned_scales->get_data_ptr(), scales->get_data_ptr());
        EXPECT_EQ(cloned_packed_weight->get_element_type(), ov::element::u2);
        EXPECT_EQ(cloned_packed_weight->get_shape(), (ov::Shape{VOCAB, HIDDEN}));
        EXPECT_EQ(cloned_packed_weight->get_rt_info().at("storage_tag").as<std::string>(), "packed_source");
        EXPECT_EQ(cloned_packed_weight->output(0).get_names(), packed_weight->output(0).get_names());
        EXPECT_EQ(cloned_packed_weight->output(0).get_rt_info().at("tensor_tag").as<std::string>(), "packed_tensor");
        EXPECT_EQ(cloned_multiply->get_rt_info().at("decompression_tag").as<std::string>(), "source");
        EXPECT_EQ(cloned_multiply->input(0).get_rt_info().at("input_tag").as<std::string>(), "packed_input");

        cloned_packed_weight->set_friendly_name("draft_packed_weight");
        cloned_packed_weight->get_rt_info()["storage_tag"] = std::string("draft");
        cloned_packed_weight->output(0).set_names({"draft_packed_data"});
        cloned_packed_weight->output(0).get_rt_info()["tensor_tag"] = std::string("draft_tensor");
        EXPECT_EQ(packed_weight->get_friendly_name(), "packed_lm_head");
        EXPECT_EQ(packed_weight->get_rt_info().at("storage_tag").as<std::string>(), "packed_source");
        EXPECT_EQ(packed_weight->output(0).get_names(), (std::unordered_set<std::string>{"packed_lm_head_data"}));
        EXPECT_EQ(packed_weight->output(0).get_rt_info().at("tensor_tag").as<std::string>(), "packed_tensor");
        EXPECT_EQ(packed_weight->output(0).get_target_inputs().size(), 1u);
        EXPECT_EQ(scales->get_friendly_name(), "lm_head_scales");
        EXPECT_EQ(scales->get_rt_info().at("storage_tag").as<std::string>(), "scales_source");
        EXPECT_EQ(scales->output(0).get_names(), (std::unordered_set<std::string>{"lm_head_scales_data"}));
        EXPECT_EQ(scales->output(0).get_target_inputs().size(), 1u);
    }
    EXPECT_TRUE(source_packed_weight.expired());
    EXPECT_TRUE(source_scales.expired());
    const auto* packed_bytes = cloned_packed_weight->get_data_ptr<uint8_t>();
    for (size_t i = 0; i < cloned_packed_weight->get_byte_size(); ++i) {
        EXPECT_EQ(packed_bytes[i], 0xe4);
    }
    for (const auto scale : cloned_scales->cast_vector<float>()) {
        EXPECT_EQ(scale, 0.25f);
    }
    EXPECT_NO_THROW(mtp_model->validate_nodes_and_infer_types());
}

TEST(MtpModelTransforms, GraftedLogitsMatchTiedWeightMatmul) {
    std::vector<float> weights_data(VOCAB * HIDDEN);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = static_cast<float>((i * 7) % 11) * 0.05f - 0.2f;
    }
    auto main_model = make_main_model(weights_data);
    bool transpose_weight = false;
    auto source_weight = extract_tied_lm_head_weight(main_model, transpose_weight).get_node_shared_ptr();
    source_weight->set_friendly_name("source_lm_head");
    source_weight->get_rt_info()["source_tag"] = std::string("source");
    source_weight->output(0).set_names({"source_lm_head_data"});
    source_weight->output(0).get_rt_info()["tensor_tag"] = std::string("source_tensor");
    auto mtp_model = make_mtp_model();
    graft_lm_head_on_mtp(mtp_model, main_model);

    ov::Core core;
    auto mtp_req = core.compile_model(mtp_model, "CPU").create_infer_request();
    EXPECT_EQ(source_weight->get_friendly_name(), "source_lm_head");
    EXPECT_EQ(source_weight->get_rt_info().at("source_tag").as<std::string>(), "source");
    EXPECT_EQ(source_weight->output(0).get_names(), (std::unordered_set<std::string>{"source_lm_head_data"}));
    EXPECT_EQ(source_weight->output(0).get_rt_info().at("tensor_tag").as<std::string>(), "source_tensor");
    EXPECT_EQ(source_weight->output(0).get_target_inputs().size(), 1u);

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
            EXPECT_NEAR(logits_data[t * VOCAB + v], expected, 1e-4f) << "Mismatch at token " << t << ", vocab " << v;
        }
    }
}
