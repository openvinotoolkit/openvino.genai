// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

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

ov::Tensor make_token_major_hidden_delta(size_t seq_len, size_t hidden_size, float start = 0.0f) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{seq_len, 1, hidden_size});
    std::iota(tensor.data<float>(), tensor.data<float>() + tensor.get_size(), start);
    return tensor;
}

std::vector<float> tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const float>();
    return std::vector<float>(data, data + tensor.get_size());
}

std::vector<int64_t> int64_tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + tensor.get_size());
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

}  // namespace

TEST(DFlashCBHiddenDeltaBuffer, AppendsAndMaterializesSingleChunk) {
    ov::genai::dflash_cb::HiddenDeltaBuffer buffer;
    auto hidden_delta = make_token_major_hidden_delta(2, 3);

    buffer.append(hidden_delta);
    auto materialized = buffer.materialize();

    ASSERT_FALSE(buffer.empty());
    ASSERT_EQ(buffer.token_count(), 2);
    ASSERT_EQ(materialized.get_shape(), ov::Shape({2, 1, 3}));
    ASSERT_EQ(tensor_values(materialized), tensor_values(hidden_delta));
}

TEST(DFlashCBHiddenDeltaBuffer, MergesChunksInOrder) {
    ov::genai::dflash_cb::HiddenDeltaBuffer buffer;
    auto first = make_token_major_hidden_delta(2, 2, 0.0f);
    auto second = make_token_major_hidden_delta(1, 2, 4.0f);

    buffer.append(first);
    buffer.append(second);
    auto materialized = buffer.materialize();

    ASSERT_EQ(buffer.token_count(), 3);
    ASSERT_EQ(materialized.get_shape(), ov::Shape({3, 1, 2}));
    ASSERT_EQ(tensor_values(materialized), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
}

TEST(DFlashModelTransforms, AppliesAndExtractsDraftRtInfo) {
    auto model = make_annotated_hidden_state_model();
    model->set_rt_info(true, "dflash_mode");
    model->set_rt_info(std::string("151669"), {"dflash", "mask_token_id"});
    model->set_rt_info(std::string("1,12,23,34,45"), {"dflash", "target_layer_ids"});

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);

    ASSERT_TRUE(properties.at("dflash_mode").as<bool>());
    ASSERT_EQ(properties.at("dflash_mask_token_id").as<int64_t>(), 151669);
    ASSERT_EQ(properties.at("dflash_target_layer_ids").as<std::vector<int32_t>>(),
              (std::vector<int32_t>{1, 12, 23, 34, 45}));

    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_TRUE(rt_info.dflash_mode);
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

TEST(DFlashCBHiddenState, TruncatesRejectedTail) {
    auto hidden_delta = make_token_major_hidden_delta(4, 2);

    auto unchanged = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 0);
    ASSERT_EQ(unchanged.get_shape(), ov::Shape({4, 1, 2}));
    ASSERT_EQ(tensor_values(unchanged), tensor_values(hidden_delta));

    auto truncated = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 1);
    ASSERT_EQ(truncated.get_shape(), ov::Shape({3, 1, 2}));
    ASSERT_EQ(tensor_values(truncated), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));

    auto empty = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 4);
    ASSERT_EQ(empty.get_shape(), ov::Shape({0, 1, 2}));
    ASSERT_EQ(empty.get_size(), 0);
}

TEST(DFlashCBDraftInputs, BuildsSeedMaskBlock) {
    auto input_ids = ov::genai::dflash_cb::build_draft_input_ids(42, 99, 3);

    ASSERT_EQ(input_ids.get_shape(), ov::Shape({1, 4}));
    ASSERT_EQ(int64_tensor_values(input_ids), (std::vector<int64_t>{42, 99, 99, 99}));
}

TEST(DFlashCBDraftInputs, BuildsPositionIdsFromCommittedLength) {
    auto position_ids = ov::genai::dflash_cb::build_draft_position_ids(5, 2, 3);

    ASSERT_EQ(position_ids.get_shape(), ov::Shape({1, 6}));
    ASSERT_EQ(int64_tensor_values(position_ids), (std::vector<int64_t>{5, 6, 7, 8, 9, 10}));
}

TEST(DFlashCBGenerationConfig, DefaultsAssistantTokensToSeven) {
    ov::genai::GenerationConfig config;

    ov::genai::dflash_cb::ensure_num_assistant_tokens_is_set(config);

    ASSERT_EQ(config.num_assistant_tokens, ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS);
}

TEST(DFlashCBGenerationConfig, PreservesExplicitAssistantTokens) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 2;

    ov::genai::dflash_cb::ensure_num_assistant_tokens_is_set(config);

    ASSERT_EQ(config.num_assistant_tokens, 2);
}

TEST(DFlashCBLinearAttentionCheckpointing, ComputesRequiredBlockCount) {
    ASSERT_EQ(ov::genai::dflash_cb::linear_attention_checkpoint_block_count(5), 7);
    ASSERT_EQ(
        ov::genai::dflash_cb::linear_attention_checkpoint_block_count(
            ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS),
        ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS + 2);
}

TEST(DFlashCBLinearAttentionCheckpointing, AdjustsBlockCountOnlyForLinearAttentionTargets) {
    ASSERT_EQ(ov::genai::dflash_cb::adjusted_linear_attention_block_count(
                  /*current_block_count=*/0,
                  /*num_assistant_tokens=*/5,
                  /*target_has_linear_attention=*/false),
              0);
    ASSERT_EQ(ov::genai::dflash_cb::adjusted_linear_attention_block_count(
                  /*current_block_count=*/0,
                  /*num_assistant_tokens=*/5,
                  /*target_has_linear_attention=*/true),
              7);
    ASSERT_EQ(ov::genai::dflash_cb::adjusted_linear_attention_block_count(
                  /*current_block_count=*/9,
                  /*num_assistant_tokens=*/5,
                  /*target_has_linear_attention=*/true),
              9);
}

TEST(DFlashCBCandidatePlanning, KeepsDraftWindowStableUntilGenerationEnds) {
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 0, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 8, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 9, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 10, 10), 0);
}

TEST(DFlashCBCandidatePlanning, ClampsValidationWindowToTargetLength) {
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 0, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 9, 10), 0);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 10, 10), 0);
}

TEST(DFlashCBCandidatePlanning, SupportsSingleAssistantToken) {
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 0, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 9, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(1, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(1, 9, 10), 0);
}

TEST(DFlashCBValidationAccounting, ComputesAcceptedAndRejected) {
    auto full_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 5);
    ASSERT_TRUE(full_accept.target_extended);
    ASSERT_EQ(full_accept.accepted, 3);
    ASSERT_EQ(full_accept.rejected, 0);

    auto partial_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 3);
    ASSERT_TRUE(partial_accept.target_extended);
    ASSERT_EQ(partial_accept.accepted, 1);
    ASSERT_EQ(partial_accept.rejected, 2);

    auto full_reject = ov::genai::dflash_cb::validation_accounting(3, 1, 2);
    ASSERT_TRUE(full_reject.target_extended);
    ASSERT_EQ(full_reject.accepted, 0);
    ASSERT_EQ(full_reject.rejected, 3);

    auto no_target_extension = ov::genai::dflash_cb::validation_accounting(3, 1, 1);
    ASSERT_FALSE(no_target_extension.target_extended);
    ASSERT_EQ(no_target_extension.accepted, 0);
    ASSERT_EQ(no_target_extension.rejected, 0);
}

TEST(DFlashCBLinearAttentionCheckpointing, UsesSeedAwarePromotionSlot) {
    const auto full_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 5);
    ASSERT_EQ(ov::genai::dflash_cb::linear_attention_checkpoint_slot_for_validation(
                  full_accept,
                  /*validation_input_includes_seed_token=*/true),
              4);

    const auto partial_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 3);
    ASSERT_EQ(ov::genai::dflash_cb::linear_attention_checkpoint_slot_for_validation(
                  partial_accept,
                  /*validation_input_includes_seed_token=*/true),
              2);

    const auto full_reject = ov::genai::dflash_cb::validation_accounting(3, 1, 2);
    ASSERT_EQ(ov::genai::dflash_cb::linear_attention_checkpoint_slot_for_validation(
                  full_reject,
                  /*validation_input_includes_seed_token=*/true),
              1);
}
