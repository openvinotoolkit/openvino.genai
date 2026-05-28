// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <numeric>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "speculative_decoding/stateful/dflash_strategy.hpp"

namespace {

std::shared_ptr<ov::Model> make_annotated_hidden_state_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 2, 4}, 1.0f);

    auto layer0 = std::make_shared<ov::op::v1::Add>(input, bias);
    layer0->output(0).set_names({"ov.hidden_states.decoder_layer_0"});
    auto layer1 = std::make_shared<ov::op::v1::Add>(layer0, bias);
    layer1->output(0).set_names({"ov.hidden_states.decoder_layer_1"});
    auto layer2 = std::make_shared<ov::op::v1::Add>(layer1, bias);
    layer2->output(0).set_names({"ov.hidden_states.decoder_layer_2"});
    auto layer3 = std::make_shared<ov::op::v1::Add>(layer2, bias);
    layer3->output(0).set_names({"ov.hidden_states.decoder_layer_3"});
    auto layer4 = std::make_shared<ov::op::v1::Add>(layer3, bias);
    layer4->output(0).set_names({"ov.hidden_states.decoder_layer_4"});

    auto result = std::make_shared<ov::op::v0::Result>(layer4);
    result->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    model->set_rt_info(
        std::string(R"({"version":1,"layers":{"0":"ov.hidden_states.decoder_layer_0","1":"ov.hidden_states.decoder_layer_1","2":"ov.hidden_states.decoder_layer_2","3":"ov.hidden_states.decoder_layer_3","4":"ov.hidden_states.decoder_layer_4"}})"),
        "hidden_states_decoder_layers");
    return model;
}

std::shared_ptr<ov::Model> make_dflash_draft_hidden_states_model(const ov::PartialShape& hidden_states_shape) {
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, hidden_states_shape);
    hidden_states->set_friendly_name("hidden_states");
    hidden_states->output(0).set_names({"hidden_states"});

    const auto hidden_size = static_cast<size_t>(hidden_states_shape[2].get_length());
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, hidden_size}, {1.0f});
    auto consumer = std::make_shared<ov::op::v1::Add>(hidden_states, bias);
    consumer->set_friendly_name("draft_hidden_states_consumer");
    auto result = std::make_shared<ov::op::v0::Result>(consumer);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden_states});
}

std::shared_ptr<ov::op::v1::Add> find_dflash_draft_hidden_states_consumer(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "draft_hidden_states_consumer") {
            return ov::as_type_ptr<ov::op::v1::Add>(node);
        }
    }
    return nullptr;
}

size_t count_outputs_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    size_t count = 0;
    for (const auto& output : model->outputs()) {
        if (output.get_names().count(name) != 0) {
            ++count;
        }
    }
    return count;
}

ov::Tensor make_logits(const std::vector<std::vector<float>>& rows) {
    ov::Tensor logits(ov::element::f32, ov::Shape{1, rows.size(), rows.front().size()});
    auto* data = logits.data<float>();
    for (const auto& row : rows) {
        std::copy(row.begin(), row.end(), data);
        data += row.size();
    }
    return logits;
}

}  // namespace

TEST(Eagle3ModelTransforms, AddsSelectedAnnotatedHiddenStatesAsOutput) {
    auto model = make_annotated_hidden_state_model();

    ov::genai::utils::eagle3::transform_hidden_state(model, {0, 1, 2});

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    const auto hidden_state = model->output("last_hidden_state");
    ASSERT_EQ(hidden_state.get_partial_shape(), ov::PartialShape({1, 2, 12}));
    ASSERT_EQ(count_outputs_with_name(model, "ov.hidden_states.decoder_layer_0"), 0);
}

TEST(Eagle3ModelTransforms, ThrowsForMissingAnnotatedHiddenStateLayer) {
    auto model = make_annotated_hidden_state_model();

    EXPECT_THROW(ov::genai::utils::eagle3::transform_hidden_state(model, {0, 1, 99}), ov::Exception);
}

TEST(DFlashModelTransforms, AppliesAndExtractsDraftRtInfo) {
    auto model = make_annotated_hidden_state_model();
    model->set_rt_info(true, "dflash_mode");
    model->set_rt_info(std::string("16"), {"dflash", "block_size"});
    model->set_rt_info(std::string("151669"), {"dflash", "mask_token_id"});
    model->set_rt_info(std::string("1,12,23,34,45"), {"dflash", "target_layer_ids"});

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);

    ASSERT_TRUE(properties.at("dflash_mode").as<bool>());
    ASSERT_EQ(properties.at("dflash_block_size").as<int64_t>(), 16);
    ASSERT_EQ(properties.at("dflash_mask_token_id").as<int64_t>(), 151669);
    ASSERT_EQ(properties.at("dflash_target_layer_ids").as<std::vector<int32_t>>(),
              (std::vector<int32_t>{1, 12, 23, 34, 45}));

    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_TRUE(rt_info.dflash_mode);
    ASSERT_EQ(rt_info.block_size, 16);
    ASSERT_EQ(rt_info.mask_token_id, 151669);
    ASSERT_EQ(rt_info.target_layer_ids, (std::vector<int32_t>{1, 12, 23, 34, 45}));
    ASSERT_TRUE(properties.empty());
}

TEST(DFlashModelTransforms, AddsArbitraryAnnotatedHiddenStatesAsOutput) {
    auto model = make_annotated_hidden_state_model();

    ov::genai::utils::dflash::expose_target_hidden_states(model, {0, 1, 2, 3, 4});

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    const auto hidden_state = model->output("last_hidden_state");
    ASSERT_EQ(hidden_state.get_partial_shape(), ov::PartialShape({1, 2, 20}));
}

TEST(DFlashModelTransforms, ThrowsForMissingAnnotatedHiddenStateLayer) {
    auto model = make_annotated_hidden_state_model();

    EXPECT_THROW(ov::genai::utils::dflash::expose_target_hidden_states(model, {0, 99}), ov::Exception);
}

TEST(DFlashModelTransforms, ReshapesStaticDraftHiddenStatesInputForCB) {
    auto model = make_dflash_draft_hidden_states_model(ov::PartialShape({1, 2, 4}));

    ov::genai::utils::dflash::reshape_draft_hidden_states_input_for_cb(model);

    ASSERT_EQ(model->input("hidden_states").get_partial_shape(), ov::PartialShape({2, 1, 4}));
    auto consumer = find_dflash_draft_hidden_states_consumer(model);
    ASSERT_TRUE(consumer);
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(consumer->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(reshape);
    ASSERT_EQ(reshape->get_output_partial_shape(0), ov::PartialShape({1, 2, 4}));
}

TEST(DFlashModelTransforms, ReshapesDynamicDraftHiddenStatesInputForCB) {
    auto model = make_dflash_draft_hidden_states_model(
        ov::PartialShape({ov::Dimension(1), ov::Dimension::dynamic(), ov::Dimension(4)}));

    ov::genai::utils::dflash::reshape_draft_hidden_states_input_for_cb(model);

    ASSERT_EQ(model->input("hidden_states").get_partial_shape(),
              ov::PartialShape({ov::Dimension::dynamic(), ov::Dimension(1), ov::Dimension(4)}));
    auto consumer = find_dflash_draft_hidden_states_consumer(model);
    ASSERT_TRUE(consumer);
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(consumer->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(reshape);
    ASSERT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension(1), ov::Dimension::dynamic(), ov::Dimension(4)}));
}

TEST(DFlashModelTransforms, ReshapesFullyDynamicDraftHiddenStatesInputForCB) {
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape::dynamic(3));
    hidden_states->set_friendly_name("hidden_states");
    hidden_states->output(0).set_names({"hidden_states"});
    auto result = std::make_shared<ov::op::v0::Result>(hidden_states);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden_states});

    ov::genai::utils::dflash::reshape_draft_hidden_states_input_for_cb(model);

    ASSERT_EQ(model->input("hidden_states").get_partial_shape(),
              ov::PartialShape({ov::Dimension::dynamic(), ov::Dimension(1), ov::Dimension::dynamic()}));
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(model->get_results().front()->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(reshape);
    ASSERT_EQ(reshape->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension(1), ov::Dimension::dynamic(), ov::Dimension::dynamic()}));
}

TEST(DFlashModelTransforms, ThrowsForMissingDraftHiddenStatesInput) {
    auto model = make_annotated_hidden_state_model();

    EXPECT_THROW(ov::genai::utils::dflash::reshape_draft_hidden_states_input_for_cb(model), ov::Exception);
}

TEST(DFlashModelTransforms, ThrowsForIncompatibleDraftHiddenStatesInput) {
    auto model = make_dflash_draft_hidden_states_model(ov::PartialShape({2, 2, 4}));

    EXPECT_THROW(ov::genai::utils::dflash::reshape_draft_hidden_states_input_for_cb(model), ov::Exception);
}

TEST(DFlashHiddenStateProvider, AppendsAndTruncatesFullContext) {
    ov::genai::DFlashHiddenStateProvider provider;
    ov::Tensor first(ov::element::f32, ov::Shape{1, 2, 3});
    std::iota(first.data<float>(), first.data<float>() + first.get_size(), 0.0f);
    ov::Tensor second(ov::element::f32, ov::Shape{1, 2, 3});
    std::iota(second.data<float>(), second.data<float>() + second.get_size(), 6.0f);

    provider.append(first, 2);
    provider.append(second, 1);
    ASSERT_EQ(provider.context_length(), 3);
    ASSERT_EQ(provider.tensor().get_shape(), ov::Shape({1, 3, 3}));

    provider.truncate(2);
    ASSERT_EQ(provider.context_length(), 2);
    ASSERT_EQ(provider.tensor().get_shape(), ov::Shape({1, 2, 3}));
}

TEST(DFlashSamplerAdapter, GreedySamplesTargetSeedToken) {
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 4;
    config.do_sample = false;
    ov::genai::DFlashSamplerAdapter adapter{ov::genai::Tokenizer()};
    auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(0, ov::genai::TokenIds{0}, config);

    auto sampled = adapter.sample(sequence_group, make_logits({{0.0f, 1.0f, 5.0f, 2.0f}}), 1, 1);

    ASSERT_EQ(sampled, (std::vector<int64_t>{2}));
    ASSERT_EQ((*sequence_group)[0]->get_generated_ids(), (std::vector<int64_t>{2}));
}

TEST(DFlashSamplerAdapter, GreedyValidationReturnsAcceptedPrefixAndFallback) {
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 4;
    config.do_sample = false;
    ov::genai::DFlashSamplerAdapter adapter{ov::genai::Tokenizer()};
    auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(0, ov::genai::TokenIds{0}, config);
    (*sequence_group)[0]->append_token(2, 0.0f);

    auto logits = make_logits({
        {0.0f, 1.0f, 5.0f, 2.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 2.0f, 5.0f},
    });
    auto validated = adapter.sample(sequence_group, logits, 2, 2, 1, true);

    ASSERT_EQ(validated, (std::vector<int64_t>{2, 4}));
    ASSERT_EQ((*sequence_group)[0]->get_generated_ids(), (std::vector<int64_t>{2, 4}));
}
