// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "model_desc.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_lattice.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy.hpp"
#include "speculative_decoding/continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"
#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> make_annotated_stateful_sdpa_model() {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});
    auto embedding_weights =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 4}, std::vector<float>(16 * 4, 1.0f));
    auto embeddings =
        std::make_shared<ov::op::v8::Gather>(embedding_weights,
                                             input_ids,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->output(0).set_names({"beam_idx"});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 1, -1});
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    auto make_cache = [&](const std::string& variable_id) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(4), ov::element::f32, variable_id});
        auto initial = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 4}, {0.0f});
        auto read = std::make_shared<ov::op::v6::ReadValue>(initial, variable);
        auto past = std::make_shared<ov::op::v8::Gather>(read, beam_idx, gather_axis);
        auto current = std::make_shared<ov::op::v1::Reshape>(embeddings, reshape_pattern, true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, current}, 1);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto transposed = std::make_shared<ov::op::v1::Transpose>(concat, transpose_order);
        return std::make_tuple(transposed, assign);
    };

    auto [key, key_assign] = make_cache("dflash_test_key_cache");
    auto [value, value_assign] = make_cache("dflash_test_value_cache");
    auto query =
        std::make_shared<ov::op::v0::Unsqueeze>(embeddings,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto mask = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 2, 3}, {0.0f});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, false);

    auto attention_transposed = std::make_shared<ov::op::v1::Transpose>(sdpa, transpose_order);
    auto attention_hidden =
        std::make_shared<ov::op::v1::Reshape>(attention_transposed,
                                              ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, -1}),
                                              true);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 4}, {1.0f});
    auto decoder_layer = std::make_shared<ov::op::v1::Add>(embeddings, attention_hidden);
    decoder_layer->set_friendly_name("decoder_layer_0");
    auto final_norm = std::make_shared<ov::op::v1::Multiply>(decoder_layer, bias);
    final_norm->set_friendly_name("final_norm");
    auto lm_head_weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 8}, {1.0f});
    auto lm_head = std::make_shared<ov::op::v0::MatMul>(final_norm, lm_head_weights);
    lm_head->set_friendly_name("lm_head");

    auto logits_result = std::make_shared<ov::op::v0::Result>(lm_head);
    logits_result->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{logits_result},
                                             ov::SinkVector{key_assign, value_assign},
                                             ov::ParameterVector{input_ids, beam_idx});
    model->set_rt_info(
        std::string(
            R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0},"1":{"producer":"final_norm","output_index":0}}})"),
        "hidden_states_decoder_layers");
    return model;
}

std::shared_ptr<ov::Model> make_eagle3_pattern_hidden_state_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    ov::Output<ov::Node> current = input->output(0);

    for (size_t layer_idx = 0; layer_idx < 5; ++layer_idx) {
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 2, 4}, 1.0f);
        auto multiplied = std::make_shared<ov::op::v1::Multiply>(current, scale);
        auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4, 4}, 1.0f);
        auto projection = std::make_shared<ov::op::v0::MatMul>(multiplied, weight);
        auto layer = std::make_shared<ov::op::v1::Add>(current, projection);
        layer->set_friendly_name("model.layers." + std::to_string(layer_idx) + "/residual_add");
        current = layer->output(0);
    }

    auto result = std::make_shared<ov::op::v0::Result>(current);
    result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
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

TEST(DFlashModelTransforms, AppliesAndExtractsDraftRtInfo) {
    auto model = make_annotated_stateful_sdpa_model();
    model->set_rt_info(true, "dflash_mode");
    model->set_rt_info(std::string("151669"), {"dflash", "mask_token_id"});
    model->set_rt_info(std::string("1,12,23,34,45"), {"dflash", "target_layer_ids"});

    // Verify default legacy candidate_position_offset when omitted
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
    ASSERT_EQ(rt_info.candidate_position_offset, 1);
    ASSERT_TRUE(properties.empty());

    // Verify explicit candidate_position_offset
    model->set_rt_info(std::string("0"), {"dflash", "candidate_position_offset"});
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);
    ASSERT_EQ(properties.at("dflash_candidate_position_offset").as<size_t>(), 0);

    auto deepspec_rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(deepspec_rt_info.candidate_position_offset, 0);
    ASSERT_TRUE(properties.empty());

    // Verify invalid values throw in extraction
    properties["dflash_mode"] = true;
    properties["dflash_mask_token_id"] = int64_t(151669);
    properties["dflash_target_layer_ids"] = std::vector<int32_t>{1, 12, 23, 34, 45};
    properties["dflash_candidate_position_offset"] = size_t(2);
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(properties), ov::Exception);
}

TEST(DFlashModelTransforms, FallsBackToEagle3LayerPatternWithoutAnnotations) {
    auto model = make_eagle3_pattern_hidden_state_model();
    const std::vector<int32_t> target_layer_ids = {0, 2, 4};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);

    ASSERT_FALSE(retained_locators);
    ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    const auto hidden_state = model->output("last_hidden_state");
    ASSERT_EQ(hidden_state.get_partial_shape(), ov::PartialShape({1, 2, 12}));
}

TEST(DFlashModelTransforms, ValidatesLocatorsAcrossPagedAttentionAndGatherBeforeAddingResult) {
    auto model = make_annotated_stateful_sdpa_model();
    const auto original_result_count = model->get_results().size();
    const std::vector<int32_t> target_layer_ids = {0, 1};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);
    ASSERT_TRUE(retained_locators);
    const auto& locators = *retained_locators;
    ASSERT_EQ(locators.size(), 2);
    ASSERT_EQ(locators.front().producer, "decoder_layer_0");
    ASSERT_EQ(locators.back().producer, "final_norm");

    ov::pass::SDPAToPagedAttention(/*use_per_layer_block_indices_inputs=*/false,
                                   /*use_score_outputs=*/false,
                                   /*allow_score_aggregation=*/true,
                                   /*allow_cache_rotation=*/false)
        .run_on_model(model);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 0);
    ASSERT_EQ(model->get_results().size(), original_result_count);
    const auto transformed_ops = model->get_ordered_ops();
    ASSERT_TRUE(std::any_of(transformed_ops.begin(), transformed_ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return std::string(node->get_type_name()).find("PagedAttention") != std::string::npos;
    }));
    for (const auto& locator : locators) {
        const auto shape = locator.output.get_partial_shape();
        ASSERT_TRUE(shape[1].is_static());
        ASSERT_EQ(shape[1].get_length(), 1);
    }

    ov::genai::utils::apply_gather_before_matmul_transformation(model);
    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 0);
    ASSERT_EQ(model->get_results().size(), original_result_count);
    const auto parameters = model->get_parameters();
    ASSERT_TRUE(
        std::any_of(parameters.begin(), parameters.end(), [](const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
            return parameter->get_friendly_name() == "sampled_tokens_indices";
        }));
    const auto transformed_lm_head = std::get<0>(ov::genai::utils::find_llm_matmul(model));
    ASSERT_TRUE(transformed_lm_head);
    ASSERT_TRUE(ov::as_type_ptr<ov::op::v8::Gather>(transformed_lm_head->input_value(0).get_node_shared_ptr()));

    ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    ASSERT_EQ(model->get_results().size(), original_result_count + 1);
    ASSERT_EQ(model->output("last_hidden_state").get_partial_shape()[2].get_length(), 8);
}

TEST(DFlashModelTransforms, RejectsMalformedRequestedLocators) {
    const std::vector<std::pair<std::string, std::vector<int32_t>>> invalid_annotations = {
        {R"({"layers":{"0":{"producer":1,"output_index":0}}})", {0}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":-1}}})", {0}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0}}})", {1}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0},"1":{"producer":"decoder_layer_0","output_index":0}}})",
         {0, 1}},
    };
    for (const auto& [annotation, layer_ids] : invalid_annotations) {
        SCOPED_TRACE(annotation);
        auto model = make_annotated_stateful_sdpa_model();
        model->set_rt_info(annotation, "hidden_states_decoder_layers");
        EXPECT_THROW(ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, layer_ids), ov::Exception);
    }
}

TEST(DFlashModelTransforms, RejectsReplacedRetainedLocatorWhileOriginalRemainsLive) {
    auto model = make_annotated_stateful_sdpa_model();
    const std::vector<int32_t> target_layer_ids = {0};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);
    ASSERT_TRUE(retained_locators);
    const auto& locators = *retained_locators;
    const auto original_output = locators.front().output;
    const auto original = original_output.get_node_shared_ptr();

    auto keep_original_live = std::make_shared<ov::op::v0::Result>(original_output);
    keep_original_live->set_friendly_name("original_decoder_layer_0_result");
    model->add_results({keep_original_live});

    auto replacement = std::make_shared<ov::op::v1::Add>(original->input_value(0), original->input_value(1));
    const auto original_name = original->get_friendly_name();
    for (auto consumer : original_output.get_target_inputs()) {
        if (consumer.get_node() != keep_original_live.get()) {
            consumer.replace_source_output(replacement->output(0));
        }
    }
    original->set_friendly_name("replaced_decoder_layer_0");
    replacement->set_friendly_name(original_name);
    model->validate_nodes_and_infer_types();

    EXPECT_THROW(
        ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids),
        ov::Exception);
}

namespace {

// A target that exports [tokens, 1, 8] hidden states: two copies of its [tokens, 1, 4] input.
std::shared_ptr<ov::Model> make_fc_move_target_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 4});
    input->output(0).set_names({"target_input"});
    auto hidden = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{input, input}, 2);
    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden);
    hidden_result->output(0).set_names({"last_hidden_state"});
    return std::make_shared<ov::Model>(ov::ResultVector{hidden_result}, ov::ParameterVector{input});
}

std::shared_ptr<ov::op::v0::Parameter> make_draft_hidden_states(size_t width) {
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                                 ov::PartialShape{-1, -1, static_cast<int64_t>(width)});
    hidden_states->output(0).set_names({"hidden_states"});
    return hidden_states;
}

// f16 [4, in_width] weights behind a Convert, as an exported compressed FC with transpose_b.
std::shared_ptr<ov::op::v0::Convert> make_fc_weights(size_t in_width, float first_value = -16.0f) {
    std::vector<float> values(4 * in_width);
    std::iota(values.begin(), values.end(), first_value);
    auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4, in_width}, values);
    return std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
}

// A draft whose FC output feeds two consumers, as the exported hidden_norm does.
std::shared_ptr<ov::Model> make_fc_move_draft_model(const std::shared_ptr<ov::op::v0::Parameter>& hidden_states,
                                                    const ov::Output<ov::Node>& fc_weights) {
    auto fc = std::make_shared<ov::op::v0::MatMul>(hidden_states, fc_weights, false, true);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, {0.5f, -1.0f, 2.0f, -0.5f});
    auto draft_add = std::make_shared<ov::op::v1::Add>(fc, bias);
    auto draft_multiply = std::make_shared<ov::op::v1::Multiply>(fc, bias);
    return std::make_shared<ov::Model>(ov::OutputVector{draft_add, draft_multiply}, ov::ParameterVector{hidden_states});
}

}  // namespace

TEST(DFlashModelTransforms, MovesTheDraftFCToTheTargetAndKeepsTheDraftOutputs) {
    auto target_model = make_fc_move_target_model();
    auto fc_weights = make_fc_weights(8);
    auto draft_model = make_fc_move_draft_model(make_draft_hidden_states(8), fc_weights);
    const auto original_target = target_model->clone();
    const auto original_draft = draft_model->clone();

    ov::genai::utils::dflash::move_fc_from_draft_to_main(draft_model, target_model);

    ASSERT_EQ(target_model->output("last_hidden_state").get_partial_shape(), ov::PartialShape({-1, 1, 4}));
    ASSERT_EQ(draft_model->input("hidden_states").get_partial_shape(), ov::PartialShape({-1, -1, 4}));
    // the weight subgraph itself moves, so its decompression markers are kept and no weights are copied
    ASSERT_EQ(target_model->output("last_hidden_state").get_node()->get_input_node_ptr(0)->get_input_node_ptr(1),
              fc_weights.get());

    ov::Core core;
    const ov::AnyMap properties{{ov::hint::inference_precision.name(), ov::element::f32}};
    auto original_target_request = core.compile_model(original_target, "CPU", properties).create_infer_request();
    auto original_draft_request = core.compile_model(original_draft, "CPU", properties).create_infer_request();
    auto target_request = core.compile_model(target_model, "CPU", properties).create_infer_request();
    auto draft_request = core.compile_model(draft_model, "CPU", properties).create_infer_request();
    for (const size_t tokens : {size_t{1}, size_t{3}}) {
        SCOPED_TRACE(tokens);
        ov::Tensor input(ov::element::f32, {tokens, 1, 4});
        std::iota(input.data<float>(), input.data<float>() + input.get_size(), -2.0f);
        original_target_request.set_tensor("target_input", input);
        original_target_request.infer();
        original_draft_request.set_tensor("hidden_states", original_target_request.get_tensor("last_hidden_state"));
        original_draft_request.infer();
        target_request.set_tensor("target_input", input);
        target_request.infer();
        draft_request.set_tensor("hidden_states", target_request.get_tensor("last_hidden_state"));
        draft_request.infer();
        for (size_t output_idx = 0; output_idx < 2; ++output_idx) {
            const auto expected = original_draft_request.get_output_tensor(output_idx);
            const auto actual = draft_request.get_output_tensor(output_idx);
            ASSERT_EQ(actual.get_shape(), expected.get_shape());
            for (size_t i = 0; i < expected.get_size(); ++i) {
                EXPECT_NEAR(actual.data<const float>()[i], expected.data<const float>()[i], 1e-5f);
            }
        }
    }
}

TEST(DFlashModelTransforms, KeysTheCachedTargetOnTheMovedFCWeights) {
    // GenAI compiles the target with ov::cache_model_path, which leaves weight values out of the model cache key.
    const auto cache_dir =
        std::filesystem::temp_directory_path() /
        ("dflash_fc_cache_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(cache_dir);
    ov::Core core;
    const ov::AnyMap properties{{ov::hint::inference_precision.name(), ov::element::f32},
                                {ov::cache_dir.name(), cache_dir.string()},
                                {ov::cache_model_path.name(), cache_dir}};
    std::vector<std::vector<float>> outputs;
    for (const float first_weight : {-16.0f, 3.0f}) {
        auto target_model = make_fc_move_target_model();
        auto draft_model = make_fc_move_draft_model(make_draft_hidden_states(8), make_fc_weights(8, first_weight));
        ov::genai::utils::dflash::move_fc_from_draft_to_main(draft_model, target_model);
        auto request = core.compile_model(target_model, "CPU", properties).create_infer_request();
        ov::Tensor input(ov::element::f32, {1, 1, 4});
        std::iota(input.data<float>(), input.data<float>() + input.get_size(), -2.0f);
        request.set_tensor("target_input", input);
        request.infer();
        const auto hidden = request.get_tensor("last_hidden_state");
        outputs.emplace_back(hidden.data<const float>(), hidden.data<const float>() + hidden.get_size());
    }
    const auto blobs = std::count_if(std::filesystem::directory_iterator(cache_dir),
                                     std::filesystem::directory_iterator(),
                                     [](const auto& entry) {
                                         return entry.path().extension() == ".blob";
                                     });
    std::filesystem::remove_all(cache_dir);

    EXPECT_EQ(blobs, 2);
    EXPECT_NE(outputs[0], outputs[1]);
}

TEST(DFlashModelTransforms, RejectsADraftWhoseHiddenStatesDoNotFeedOnlyAMatchingFC) {
    // hidden_states also feeds a second consumer, so it cannot be resized
    auto shared_hidden_states = make_draft_hidden_states(8);
    auto shared_fc = std::make_shared<ov::op::v0::MatMul>(shared_hidden_states, make_fc_weights(8), false, true);
    auto other_consumer = std::make_shared<ov::op::v1::Multiply>(
        shared_hidden_states,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f}));
    auto shared_input_draft = std::make_shared<ov::Model>(ov::OutputVector{shared_fc, other_consumer},
                                                          ov::ParameterVector{shared_hidden_states});

    // hidden_states is the MatMul's second input, not its data
    auto weight_side_hidden_states = make_draft_hidden_states(8);
    auto data = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 2, 4}, std::vector<float>(8, 1.0f));
    auto weight_side_matmul = std::make_shared<ov::op::v0::MatMul>(data, weight_side_hidden_states);
    auto weight_side_draft = std::make_shared<ov::Model>(ov::OutputVector{weight_side_matmul},
                                                         ov::ParameterVector{weight_side_hidden_states});

    // the FC expects 6-wide rows, but the target exports 8
    auto narrow_draft = make_fc_move_draft_model(make_draft_hidden_states(6), make_fc_weights(6));

    const std::vector<std::pair<std::shared_ptr<ov::Model>, std::string>> cases = {
        {shared_input_draft, "must feed only the data input"},
        {weight_side_draft, "must feed only the data input"},
        {narrow_draft, "exports hidden states of width 8"},
    };
    for (const auto& [draft_model, expected_message] : cases) {
        SCOPED_TRACE(expected_message);
        try {
            ov::genai::utils::dflash::move_fc_from_draft_to_main(draft_model, make_fc_move_target_model());
            FAIL() << "Expected the DFlash FC move to be rejected";
        } catch (const ov::Exception& exception) {
            EXPECT_NE(std::string(exception.what()).find(expected_message), std::string::npos) << exception.what();
        }
    }
}

namespace {

constexpr int64_t MASK_TOKEN = 99;
constexpr size_t HIDDEN_SIZE = 2;

std::shared_ptr<ov::Model> make_dflash_draft_inputs_model() {
    ov::ParameterVector params;
    auto add_input = [&](const std::string& name, const ov::element::Type& type, const ov::PartialShape& shape) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
        parameter->output(0).get_tensor().set_names({name});
        params.push_back(parameter);
    };
    add_input("token_type_ids", ov::element::i64, ov::PartialShape{-1, 1});
    add_input("hidden_states", ov::element::f32, ov::PartialShape{-1, 1, static_cast<int64_t>(HIDDEN_SIZE)});
    add_input("sampled_tokens_indices", ov::element::i64, ov::PartialShape{-1});
    auto logits = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 8}, {0.0f});
    logits->output(0).get_tensor().set_names({"logits"});
    return std::make_shared<ov::Model>(ov::OutputVector{logits}, params);
}

// Hidden-state rows [rows, 1, HIDDEN_SIZE] whose values count up from `first`.
ov::Tensor make_hidden_rows(size_t rows, float first) {
    ov::Tensor tensor(ov::element::f32, {rows, 1, HIDDEN_SIZE});
    std::iota(tensor.data<float>(), tensor.data<float>() + tensor.get_size(), first);
    return tensor;
}

ov::genai::GeneratedSequences target_step(const std::vector<int64_t>& tokens,
                                          size_t num_processed_tokens,
                                          const ov::Tensor& hidden_states) {
    ov::genai::GeneratedSequences sequences;
    sequences.emplace(0,
                      ov::genai::GeneratedSequence(tokens,
                                                   std::vector<float>(tokens.size(), 0.0f),
                                                   num_processed_tokens,
                                                   hidden_states));
    return sequences;
}

}  // namespace

class DFlashCBPipelineTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class DraftInstance : public ContinuousBatchingForDFlashDecodingImpl {
    public:
        explicit DraftInstance(size_t candidate_position_offset = 1) {
            m_sampler = std::make_shared<ov::genai::Sampler>();
            dflash_mode_enabled = true;
            m_dflash_mask_token_id = MASK_TOKEN;
            m_dflash_candidate_position_offset = candidate_position_offset;
            ov::Core core;
            auto request = core.compile_model(make_dflash_draft_inputs_model(), "CPU").create_infer_request();
            m_model_runner = std::make_shared<ov::genai::ModelRunner>(request, /*block_size=*/4);
            m_model_runner->enable_dflash_draft(candidate_position_offset);
        }

        void use_dynamic_split_fuse() {
            m_dflash_block_may_follow_prompt_rows = true;
        }

        ov::genai::SequenceGroup::Ptr add(uint64_t request_id,
                                          const std::vector<int64_t>& prompt,
                                          size_t num_assistant_tokens,
                                          size_t max_new_tokens = 100) {
            auto config = ov::genai::utils::get_greedy_config();
            config.num_assistant_tokens = num_assistant_tokens;
            config.max_new_tokens = max_new_tokens;
            auto group = std::make_shared<ov::genai::SequenceGroup>(request_id, prompt, config);
            group->pause_generation(true);
            m_requests.push_back(group);
            return group;
        }

        void remove(uint64_t request_id) {
            m_requests.erase(std::remove_if(m_requests.begin(), m_requests.end(),
                                            [&](const ov::genai::SequenceGroup::Ptr& group) {
                                                return group->get_request_id() == request_id;
                                            }),
                             m_requests.end());
        }

        bool has_window(const ov::genai::SequenceGroup::Ptr& group) const {
            return m_dflash_draft_windows.count(sequence_id(group)) != 0;
        }

        const ov::genai::detail::DFlashDraftWindow& window(const ov::genai::SequenceGroup::Ptr& group) const {
            return m_dflash_draft_windows.at(sequence_id(group));
        }

        // Leaves the request as the draft multistep does after its window ran whole and sampled `candidates`.
        void finish_window(const ov::genai::SequenceGroup::Ptr& group, const std::vector<int64_t>& candidates) {
            const auto planned = window(group);
            auto sequence = group->get_running_sequences().front();
            if (planned.block_rows > 0) {
                sequence->remove_last_tokens(planned.block_rows - 1);
            }
            auto& logit_processor = m_sampler->get_logit_processor(group->get_request_id());
            for (const int64_t token : candidates) {
                sequence->append_token(token, 0.0f);
                logit_processor.register_new_generated_token(token);
            }
            group->update_processed_tokens_num(planned.committed_rows + planned.context_rows);
            group->set_num_validated_tokens(0);
            m_dflash_draft_windows.clear();
        }

        // Leaves the request as the draft multistep does after the scheduler ran only `forwarded_rows` of its window,
        // stopping before the first candidate row.
        void drop_window(const ov::genai::SequenceGroup::Ptr& group, size_t forwarded_rows = 0) {
            const auto planned = window(group);
            group->update_processed_tokens_num(planned.committed_rows + forwarded_rows);
            if (planned.block_rows > 0) {
                group->get_running_sequences().front()->remove_last_tokens(planned.block_rows - 1);
            }
            group->update_processed_tokens_num(
                std::min(group->get_num_processed_tokens(), planned.committed_rows + planned.context_rows));
            group->set_num_validated_tokens(0);
            m_dflash_draft_windows.clear();
        }

    private:
        static uint64_t sequence_id(const ov::genai::SequenceGroup::Ptr& group) {
            return group->get_sequences().front()->get_id();
        }
    };

    // Prompt of 8 tokens processed by the target in one chunk; the target sampled 7, and the draft forwarded the
    // prompt rows. The target then generated seed 17 alone and exported the row of 7, so the first block follows.
    ov::genai::SequenceGroup::Ptr start_with_block(DraftInstance& draft, size_t num_assistant_tokens = 3) {
        auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, num_assistant_tokens);
        draft.update_request(0, target_step({7}, 8, make_hidden_rows(8, 0.0f)), true);
        draft.finish_window(group, {});
        draft.update_request(0, target_step({7, 17}, 9, make_hidden_rows(1, 50.0f)), true);
        return group;
    }

    static std::vector<int64_t> generated(const ov::genai::SequenceGroup::Ptr& group) {
        return group->get_sequences().front()->get_generated_ids();
    }
};

TEST_F(DFlashCBPipelineTest, PrefillChunkOffersOnlyTheExportedRows) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, 3);

    auto result = draft.update_request(0, target_step({}, 4, make_hidden_rows(4, 0.0f)), true);

    ASSERT_EQ(result.inserted_tokens_cnt, 0);
    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_FALSE(group->is_waiting());
    // a window inside the prompt is scheduled in the prompt phase, capped to the exported rows
    ASSERT_FALSE(group->can_generate_tokens());
    ASSERT_EQ(group->get_num_processed_tokens(), 0);
    ASSERT_EQ(draft.window(group).context_rows, 4);
    ASSERT_EQ(draft.window(group).block_rows, 0);
}

TEST_F(DFlashCBPipelineTest, WindowStartingInThePromptHasNoBlock) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, 3);

    // the target processed the whole prompt and sampled 7
    draft.update_request(0, target_step({7}, 8, make_hidden_rows(8, 0.0f)), true);

    // the window carries only the prompt rows, so it is scheduled in the prompt phase like any prefill chunk
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7}));
    ASSERT_EQ(draft.window(group).context_rows, 8);
    ASSERT_EQ(draft.window(group).block_rows, 0);
    ASSERT_FALSE(group->can_generate_tokens());
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 8);
}

TEST_F(DFlashCBPipelineTest, FirstBlockFollowsThePromptRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);

    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 17, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    // the window starts at the prompt end, so it is scheduled in the generation phase
    ASSERT_TRUE(group->can_generate_tokens());
    ASSERT_EQ(group->get_num_processed_tokens(), 8);
    // the row of 7 + seed + 3 masks
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 1 + 4);
    ASSERT_FALSE(group->requires_sampling());
}

TEST_F(DFlashCBPipelineTest, DynamicSplitFuseDraftsTheFirstBlockWithTheLastPromptRows) {
    DraftInstance draft;
    draft.use_dynamic_split_fuse();
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, 3);
    // the target ran the first prompt chunk, which has no block
    draft.update_request(0, target_step({}, 4, make_hidden_rows(4, 0.0f)), true);
    ASSERT_EQ(draft.window(group).block_rows, 0);
    draft.finish_window(group, {});

    // the second chunk ended the prompt and the target sampled 7
    draft.update_request(0, target_step({7}, 8, make_hidden_rows(4, 4.0f)), true);

    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_EQ(draft.window(group).committed_rows, 4);
    ASSERT_EQ(draft.window(group).context_rows, 4);
    ASSERT_EQ(draft.window(group).block_rows, 4);
    // the 4 remaining prompt rows, then seed 7 and 3 masks
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 4 + 4);
}

TEST_F(DFlashCBPipelineTest, DeepSpecBlockStartsCandidatesAtTheSeed) {
    DraftInstance draft(/*candidate_position_offset=*/0);
    auto group = start_with_block(draft);

    // the row of 7 + seed + 2 masks
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 1 + 3);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 17, MASK_TOKEN, MASK_TOKEN}));
}

TEST_F(DFlashCBPipelineTest, PartialRejectionKeepsAcceptedRowsAndRewindsTheBlock) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    // the target accepted 20, rejected 21 and corrected it to 30; it exported rows for [17, 20, 21, 22]
    auto result = draft.update_request(0, target_step({7, 17, 20, 30}, 11, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 2);
    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 17, 20, 30, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_EQ(group->get_num_processed_tokens(), 9);
    // rows of the seed and the accepted candidate, then the next block
    ASSERT_EQ(draft.window(group).committed_rows, 9);
    ASSERT_EQ(draft.window(group).context_rows, 2);
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 2 + 4);
}

TEST_F(DFlashCBPipelineTest, FullAcceptanceTakesAllValidatedRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    auto result =
        draft.update_request(0, target_step({7, 17, 20, 21, 22, 23}, 13, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(draft.window(group).context_rows, 4);
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 4 + 4);
}

TEST_F(DFlashCBPipelineTest, FullRejectionTakesOnlyTheSeedRow) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    auto result = draft.update_request(0, target_step({7, 17, 30}, 10, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 3);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 17, 30, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_EQ(draft.window(group).context_rows, 1);
}

TEST_F(DFlashCBPipelineTest, DeferredTargetPublishesNoNewRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    // the target did not run the request: it still reports 9 processed positions and stale rows
    auto result = draft.update_request(0, target_step({7, 17, 20, 21, 22}, 9, make_hidden_rows(8, 0.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_FALSE(draft.has_window(group));
    ASSERT_TRUE(group->is_waiting());
}

TEST_F(DFlashCBPipelineTest, TargetRecomputeReusesTheCommittedRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    // the target was preempted and recomputed positions [0, 4) before validating the candidates
    draft.update_request(0, target_step({7, 17, 20, 21, 22}, 4, make_hidden_rows(4, 0.0f)), true);
    ASSERT_FALSE(draft.has_window(group));
    ASSERT_EQ(group->get_num_processed_tokens(), 9);

    // it recomputed [4, 11) and validated the candidates: 20 accepted, 21 corrected to 30
    draft.update_request(0, target_step({7, 17, 20, 30}, 11, make_hidden_rows(7, 100.0f)), true);
    ASSERT_EQ(draft.window(group).committed_rows, 9);
    ASSERT_EQ(draft.window(group).context_rows, 2);
}

TEST_F(DFlashCBPipelineTest, MissedWindowLeavesTheRequestToTheTarget) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.drop_window(group);

    // the target generated 30 alone and exported only the seed row; the row of 7 is gone
    auto result = draft.update_request(0, target_step({7, 17, 30}, 10, make_hidden_rows(1, 100.0f)), true);

    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 17, 30}));
    ASSERT_FALSE(draft.has_window(group));
    // the draft request stops, which the speculative step turns into target-only decoding
    ASSERT_EQ(group->get_generation_stream()->get_status(), ov::genai::GenerationStatus::STOP);
}

TEST_F(DFlashCBPipelineTest, MissedWindowRunsWithTheRowsOfADeferredTarget) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.drop_window(group);

    // the target did not run the request either, and still publishes the rows of its last forward
    draft.update_request(0, target_step({7, 17}, 9, make_hidden_rows(1, 50.0f)), true);

    ASSERT_EQ(group->get_generation_stream()->get_status(), ov::genai::GenerationStatus::RUNNING);
    ASSERT_EQ(draft.window(group).committed_rows, 8);
    ASSERT_EQ(draft.window(group).context_rows, 1);
}

TEST_F(DFlashCBPipelineTest, WindowCutAtTheSeedKeepsDrafting) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    // the scheduler ran the row of 7 and the seed, but no candidate row
    draft.drop_window(group, /*forwarded_rows=*/2);

    // the target generated 30 alone and exported the seed row, which the draft has not committed
    draft.update_request(0, target_step({7, 17, 30}, 10, make_hidden_rows(1, 100.0f)), true);

    ASSERT_EQ(group->get_generation_stream()->get_status(), ov::genai::GenerationStatus::RUNNING);
    ASSERT_EQ(draft.window(group).committed_rows, 9);
    ASSERT_EQ(draft.window(group).context_rows, 1);
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 1 + 4);
}

TEST_F(DFlashCBPipelineTest, StaleExtraRowsAreReplaced) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});
    // a request whose window was deferred keeps the extra rows of its last plan
    group->set_num_validated_tokens(7);

    draft.update_request(0, target_step({7, 17, 20, 30}, 11, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 2 + 4);
}

TEST_F(DFlashCBPipelineTest, NearMaxNewTokensForwardsOnlyTheContext) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3}, 3, /*max_new_tokens=*/3);
    draft.update_request(0, target_step({3}, 4, make_hidden_rows(4, 0.0f)), true);
    draft.finish_window(group, {});
    draft.update_request(0, target_step({3, 9}, 5, make_hidden_rows(1, 0.0f)), true);

    // one token left for the target: no block, only the context rows are forwarded
    ASSERT_EQ(draft.window(group).block_rows, 0);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{3, 9}));
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 1);
}

TEST_F(DFlashCBPipelineTest, RestoredPrefixIsAlreadyCommitted) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, 3);
    // prefix caching restored 4 positions of both models before the first step
    group->update_processed_tokens_num(4);

    draft.update_request(0, target_step({}, 6, make_hidden_rows(2, 0.0f)), true);

    ASSERT_EQ(draft.window(group).committed_rows, 4);
    ASSERT_EQ(draft.window(group).context_rows, 2);
    ASSERT_EQ(group->get_num_processed_tokens(), 4);
}

TEST_F(DFlashCBPipelineTest, ReusedRequestIdStartsFromAFreshWindow) {
    DraftInstance draft;
    auto first = start_with_block(draft);
    draft.remove(0);

    auto second = draft.add(0, {5, 6, 7, 8}, 3);
    draft.update_request(0, target_step({}, 2, make_hidden_rows(2, 0.0f)), true);

    ASSERT_EQ(draft.window(second).committed_rows, 0);
    ASSERT_EQ(draft.window(second).context_rows, 2);
    ASSERT_EQ(draft.window(second).block_rows, 0);
}

class DFlashStrategyTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class StrategyInstance : public DFlashDecodingImpl {
    public:
        explicit StrategyInstance(ov::genai::ModelInputType input_type = ov::genai::ModelInputType::TOKENS) {
            m_model_input_type = input_type;
        }
        using DFlashDecodingImpl::make_draft_generation_config;
        using DFlashDecodingImpl::validate_request_config;
    };

    static void expect_rejected(const ov::genai::GenerationConfig& config,
                                const std::string& expected_message,
                                ov::genai::ModelInputType input_type = ov::genai::ModelInputType::TOKENS) {
        // Unsupported configurations must be rejected before add_request touches the pipelines.
        StrategyInstance strategy(input_type);
        try {
            strategy.add_request(0, ov::Tensor{}, config);
            FAIL() << "Expected the DFlash request to be rejected";
        } catch (const ov::Exception& exception) {
            EXPECT_NE(std::string(exception.what()).find(expected_message), std::string::npos) << exception.what();
        }
    }
};

TEST_F(DFlashStrategyTest, RejectsUnsupportedRequests) {
    const std::vector<std::pair<std::function<void(ov::genai::GenerationConfig&)>, std::string>> cases = {
        {[](ov::genai::GenerationConfig& config) { config.do_sample = true; }, "greedy decoding only"},
        {[](ov::genai::GenerationConfig& config) { config.assistant_confidence_threshold = 0.5f; },
         "assistant_confidence_threshold must be 0.f"},
        {[](ov::genai::GenerationConfig& config) { config.adapters = ov::genai::AdapterConfig{}; },
         "does not support adapters"},
    };
    for (const auto& [make_unsupported, expected_message] : cases) {
        SCOPED_TRACE(expected_message);
        auto config = ov::genai::utils::get_greedy_config();
        make_unsupported(config);
        expect_rejected(config, expected_message);
    }
}

TEST_F(DFlashStrategyTest, ValidatesVlmRequests) {
    // The target runs these modes as it does without a draft.
    auto config = ov::genai::utils::get_greedy_config();
    config.pruning_ratio = 1;
    config.echo = true;
    config.logprobs = 1;
    EXPECT_NO_THROW(StrategyInstance(ov::genai::ModelInputType::EMBEDDINGS).validate_request_config(config));

    config.return_omni_outputs = true;
    expect_rejected(config, "Omni intermediate outputs", ov::genai::ModelInputType::EMBEDDINGS);
}

TEST_F(DFlashStrategyTest, DraftConfigOnlyProposesCandidates) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 3;
    config.stop_strings = {"stop"};
    config.stop_token_ids = {5};
    // an explicit EOS that may differ from the pipeline default
    config.set_eos_token_id(7);
    config.echo = true;

    const auto draft_config = StrategyInstance::make_draft_generation_config(config);

    EXPECT_TRUE(draft_config.ignore_eos);
    EXPECT_TRUE(draft_config.stop_strings.empty());
    EXPECT_EQ(draft_config.stop_token_ids, (std::set<int64_t>{5, 7}));
    EXPECT_FALSE(draft_config.echo);
    EXPECT_EQ(draft_config.max_new_tokens, SIZE_MAX - 1);
    EXPECT_NO_THROW(draft_config.validate());
}

TEST_F(DFlashStrategyTest, RejectsADraftWithoutTokenTypeIds) {
    auto make_model = [] {
        auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
        input_ids->output(0).set_names({"input_ids"});
        auto result = std::make_shared<ov::op::v0::Result>(input_ids);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_ids});
    };
    ov::genai::utils::dflash::DFlashRTInfo rt_info;
    rt_info.dflash_mode = true;
    rt_info.mask_token_id = MASK_TOKEN;
    rt_info.target_layer_ids = {0};
    const ov::genai::ModelDesc main_desc(make_model(), ov::genai::Tokenizer{});
    const ov::genai::ModelDesc draft_desc(make_model(), ov::genai::Tokenizer{});

    try {
        DFlashDecodingImpl strategy(main_desc, draft_desc, rt_info);
        FAIL() << "Expected the stateful DFlash draft to be rejected";
    } catch (const ov::Exception& exception) {
        EXPECT_NE(std::string(exception.what()).find("re-export the draft"), std::string::npos) << exception.what();
    }
}

namespace {

constexpr size_t DSPARK_TEST_VOCAB_SIZE = 16;
constexpr size_t DSPARK_TEST_MARKOV_RANK = 4;
constexpr int64_t DSPARK_TEST_TOP_K = 2;

void set_dspark_markov_head_rt_info(const std::shared_ptr<ov::Model>& model, const std::string& omitted_key = "") {
    const std::vector<std::pair<std::string, std::string>> rt_info = {
        {"top_k", std::to_string(DSPARK_TEST_TOP_K)},
        {"vocab_size", std::to_string(DSPARK_TEST_VOCAB_SIZE)},
        {"markov_rank", std::to_string(DSPARK_TEST_MARKOV_RANK)},
        {"markov_head_type", "vanilla"},
    };
    for (const auto& [name, value] : rt_info) {
        if (name != omitted_key) {
            model->set_rt_info(value, {"dflash", name});
        }
    }
}

// The dense step biases token j after token p by codebook[p] . codebook[j]. The candidate scorer keeps the Top-K
// draft logits and scores candidate i followed by candidate j with the sum of their logits. As in the optimum-intel
// export, every K-sized shape follows the TopK.
std::shared_ptr<ov::Model> make_dspark_markov_head_model(const std::string& omitted_key = "") {
    auto prev_token_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    prev_token_ids->output(0).set_names({"prev_token_ids"});
    const ov::Shape codebook_shape{DSPARK_TEST_VOCAB_SIZE, DSPARK_TEST_MARKOV_RANK};
    std::vector<float> codebook_values(ov::shape_size(codebook_shape));
    std::iota(codebook_values.begin(), codebook_values.end(), 0.0f);
    auto codebook = ov::op::v0::Constant::create(ov::element::f32, codebook_shape, codebook_values);
    auto codes = std::make_shared<ov::op::v8::Gather>(codebook,
                                                      prev_token_ids,
                                                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto markov_bias = std::make_shared<ov::op::v0::MatMul>(codes, codebook, false, true);
    auto markov_bias_result = std::make_shared<ov::op::v0::Result>(markov_bias);
    markov_bias_result->output(0).set_names({"markov_bias"});

    auto draft_logits =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{-1, static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE)});
    draft_logits->output(0).set_names({"draft_logits"});
    auto anchor_token_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    anchor_token_ids->output(0).set_names({"anchor_token_ids"});
    auto subsequence_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    subsequence_begins->output(0).set_names({"subsequence_begins"});
    auto top_k = std::make_shared<ov::op::v11::TopK>(
        draft_logits,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {DSPARK_TEST_TOP_K}),
        -1,
        ov::op::v11::TopK::Mode::MAX,
        ov::op::v11::TopK::SortType::SORT_VALUES);
    auto candidate_ids = std::make_shared<ov::op::v0::Convert>(top_k->output(1), ov::element::i64);
    auto predecessor_logits =
        std::make_shared<ov::op::v0::Unsqueeze>(top_k->output(0),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    auto successor_logits =
        std::make_shared<ov::op::v0::Unsqueeze>(top_k->output(0),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto edge_scores = std::make_shared<ov::op::v1::Add>(predecessor_logits, successor_logits);
    auto edge_scores_result = std::make_shared<ov::op::v0::Result>(edge_scores);
    edge_scores_result->output(0).set_names({"edge_scores"});
    auto candidate_ids_result = std::make_shared<ov::op::v0::Result>(candidate_ids);
    candidate_ids_result->output(0).set_names({"candidate_ids"});

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{markov_bias_result, edge_scores_result, candidate_ids_result},
        ov::ParameterVector{prev_token_ids, draft_logits, anchor_token_ids, subsequence_begins});
    set_dspark_markov_head_rt_info(model, omitted_key);
    return model;
}

ov::AnyMap select_markov_computation(std::optional<int64_t> top_k,
                                     const std::shared_ptr<ov::Model>& markov_head_model) {
    ov::AnyMap properties;
    ov::genai::utils::dflash::select_markov_head_computation(markov_head_model, top_k, properties);
    return properties;
}

std::shared_ptr<ov::Model> selected_computation(const ov::AnyMap& properties) {
    return properties.at("dflash_markov_head_computation").as<std::shared_ptr<ov::Model>>();
}

// The K of the selected candidate scorer, or TOP_K_FOR_MARKOV_STEP for the dense step.
int64_t selected_top_k(const ov::AnyMap& properties) {
    const auto computation = selected_computation(properties);
    if (ov::genai::utils::dflash::is_markov_step(computation)) {
        return ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP;
    }
    return std::stoll(computation->get_rt_info<std::string>(std::vector<std::string>{"dflash", "top_k"}));
}

void set_markov_head_computation(ov::AnyMap& properties, const std::shared_ptr<ov::Model>& computation) {
    properties["dflash_markov_head_computation"] = computation;
}

std::shared_ptr<ov::Model> make_dspark_markov_step() {
    return selected_computation(select_markov_computation(int64_t{0}, make_dspark_markov_head_model()));
}

std::shared_ptr<ov::Model> make_dspark_candidate_scorer(bool with_extra_input = false,
                                                        const std::string& omitted_key = "") {
    auto model = selected_computation(select_markov_computation(std::nullopt, make_dspark_markov_head_model()));
    if (with_extra_input) {
        auto draft_hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
        draft_hidden_states->output(0).set_names({"draft_hidden_states"});
        model->add_parameters({draft_hidden_states});
    }
    if (!omitted_key.empty()) {
        model->get_rt_info()["dflash"].as<ov::AnyMap>().erase(omitted_key);
    }
    return model;
}

ov::genai::utils::dflash::DFlashRTInfo make_dspark_backbone_info() {
    ov::genai::utils::dflash::DFlashRTInfo info;
    info.markov_rank = DSPARK_TEST_MARKOV_RANK;
    info.markov_head_type = "vanilla";
    return info;
}

ov::AnyMap make_dspark_draft_properties(size_t markov_rank, const std::string& head_type) {
    ov::AnyMap properties;
    properties["dflash_mode"] = true;
    properties["dflash_mask_token_id"] = int64_t{15};
    properties["dflash_target_layer_ids"] = std::vector<int32_t>{0};
    properties["dflash_candidate_position_offset"] = size_t{0};
    properties["dflash_markov_rank"] = markov_rank;
    properties["dflash_markov_head_type"] = head_type;
    return properties;
}

std::shared_ptr<ov::Model> make_dflash_export_backbone_model() {
    auto inputs_embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 4});
    inputs_embeds->output(0).set_names({"inputs_embeds"});
    auto identity =
        std::make_shared<ov::op::v1::Add>(inputs_embeds,
                                          ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}));
    auto result = std::make_shared<ov::op::v0::Result>(identity);
    result->output(0).set_names({"last_hidden_state"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{inputs_embeds});
}

std::vector<float> tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const float>();
    return std::vector<float>(data, data + tensor.get_size());
}

std::vector<int64_t> int64_tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + tensor.get_size());
}

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& prefix)
        : m_path(std::filesystem::temp_directory_path() /
                 (prefix + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::remove_all(m_path);
        std::filesystem::create_directories(m_path);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    const std::filesystem::path& path() const {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

}  // namespace

TEST(DSparkModelTransforms, AppliesAndExtractsMarkovHeadRtInfo) {
    auto backbone = make_dflash_export_backbone_model();
    backbone->set_rt_info(true, "dflash_mode");
    backbone->set_rt_info(std::string("15"), {"dflash", "mask_token_id"});
    backbone->set_rt_info(std::string("0,1"), {"dflash", "target_layer_ids"});
    backbone->set_rt_info(std::string("0"), {"dflash", "candidate_position_offset"});
    backbone->set_rt_info(std::string("4"), {"dflash", "markov_rank"});
    backbone->set_rt_info(std::string("vanilla"), {"dflash", "markov_head_type"});

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(backbone, properties);
    ASSERT_EQ(properties.at("dflash_markov_rank").as<size_t>(), 4);
    ASSERT_EQ(properties.at("dflash_markov_head_type").as<std::string>(), "vanilla");

    auto markov_step = make_dspark_markov_step();
    set_markov_head_computation(properties, markov_step);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(rt_info.candidate_position_offset, 0);
    ASSERT_EQ(rt_info.markov_rank, 4);
    ASSERT_EQ(rt_info.markov_head_type, "vanilla");
    ASSERT_EQ(rt_info.markov_head_computation, markov_step);
    ASSERT_TRUE(ov::genai::utils::dflash::is_markov_step(rt_info.markov_head_computation));
    ASSERT_TRUE(properties.empty());
}

TEST(DSparkModelTransforms, TreatsMarkovRankZeroAsPlainDFlash) {
    // DeepSpec publishes its DFlash drafts as DSpark checkpoints with markov_rank=0; their export has no Markov keys.
    auto backbone = make_dflash_export_backbone_model();
    backbone->set_rt_info(true, "dflash_mode");
    backbone->set_rt_info(std::string("15"), {"dflash", "mask_token_id"});
    backbone->set_rt_info(std::string("0,1"), {"dflash", "target_layer_ids"});
    backbone->set_rt_info(std::string("0"), {"dflash", "candidate_position_offset"});

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(backbone, properties);
    ASSERT_EQ(properties.count("dflash_markov_rank"), 0);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_TRUE(rt_info.dflash_mode);
    ASSERT_EQ(rt_info.candidate_position_offset, 0);
    ASSERT_EQ(rt_info.markov_rank, 0);
    ASSERT_EQ(rt_info.markov_head_computation, nullptr);
    ASSERT_TRUE(properties.empty());

    auto explicit_rank_zero = make_dspark_draft_properties(0, "vanilla");
    rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(explicit_rank_zero);
    ASSERT_EQ(rt_info.markov_rank, 0);
    ASSERT_EQ(rt_info.markov_head_computation, nullptr);
    ASSERT_TRUE(explicit_rank_zero.empty());
}

TEST(DSparkModelTransforms, ExtractsSelectedCandidateScorer) {
    auto candidate_scorer = make_dspark_candidate_scorer();
    auto properties = make_dspark_draft_properties(DSPARK_TEST_MARKOV_RANK, "vanilla");
    set_markov_head_computation(properties, candidate_scorer);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(rt_info.markov_head_computation, candidate_scorer);
    ASSERT_FALSE(ov::genai::utils::dflash::is_markov_step(rt_info.markov_head_computation));
    ASSERT_TRUE(properties.empty());
}

TEST(DSparkModelTransforms, RejectsInconsistentDraftProperties) {
    const auto with_computation = [](ov::AnyMap properties, const std::shared_ptr<ov::Model>& computation) {
        set_markov_head_computation(properties, computation);
        return properties;
    };
    const auto vanilla = [](size_t markov_rank) {
        return make_dspark_draft_properties(markov_rank, "vanilla");
    };
    auto gated_candidate_scorer = make_dspark_candidate_scorer();
    gated_candidate_scorer->set_rt_info(std::string("gated"), {"dflash", "markov_head_type"});
    auto top_k_for_a_draft_model = vanilla(0);
    top_k_for_a_draft_model["markov_head_top_k"] = int64_t{8};
    const std::vector<std::pair<std::string, ov::AnyMap>> cases = {
        {"Markov head without its computation", vanilla(DSPARK_TEST_MARKOV_RANK)},
        {"gated Markov head type",
         with_computation(make_dspark_draft_properties(DSPARK_TEST_MARKOV_RANK, "gated"), make_dspark_markov_step())},
        {"DFlash-2 Markov head type",
         with_computation(make_dspark_draft_properties(DSPARK_TEST_MARKOV_RANK, "dflash2"),
                          make_dspark_candidate_scorer())},
        {"computation of another head type",
         with_computation(vanilla(DSPARK_TEST_MARKOV_RANK), gated_candidate_scorer)},
        {"computation of another rank", with_computation(vanilla(8), make_dspark_candidate_scorer())},
        {"markov_head_top_k without draft_model(<draft folder>)", top_k_for_a_draft_model},
    };
    for (auto [name, properties] : cases) {
        SCOPED_TRACE(name);
        EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(properties), ov::Exception);
    }
}

TEST(DSparkModelTransforms, RejectsMarkovStepThatDoesNotMatchTheBackbone) {
    auto info = make_dspark_backbone_info();
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_markov_step(make_dspark_markov_step(), info));

    auto mismatched_rank = info;
    mismatched_rank.markov_rank = 8;
    EXPECT_THROW(ov::genai::utils::dflash::validate_markov_step(make_dspark_markov_step(), mismatched_rank),
                 ov::Exception);

    auto wrong_vocab = make_dspark_markov_step();
    wrong_vocab->set_rt_info(std::string("32"), {"dflash", "vocab_size"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_markov_step(wrong_vocab, info), ov::Exception);

    EXPECT_THROW(ov::genai::utils::dflash::validate_markov_step(make_dflash_export_backbone_model(), info),
                 ov::Exception);
}

TEST(DSparkModelTransforms, ReadsTheMarkovHeadModelNextToTheDraft) {
    const TemporaryDirectory temporary_directory("genai_dspark_markov_");
    const auto& draft_dir = temporary_directory.path();
    const auto info = make_dspark_backbone_info();

    ov::AnyMap without_markov_head{{"dflash_markov_rank", size_t{0}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, without_markov_head);
    ASSERT_EQ(without_markov_head.count("dflash_markov_head_computation"), 0);

    ov::AnyMap top_k_without_markov_head{{"dflash_markov_rank", size_t{0}}, {"markov_head_top_k", 16}};
    EXPECT_THROW(ov::genai::utils::dflash::read_markov_head(draft_dir, top_k_without_markov_head), ov::Exception);

    ov::AnyMap missing_markov_head_model{{"dflash_markov_rank", size_t{4}}};
    EXPECT_THROW(ov::genai::utils::dflash::read_markov_head(draft_dir, missing_markov_head_model), ov::Exception);

    ov::save_model(make_dspark_markov_head_model(), draft_dir / "openvino_markov_head_model.xml");
    ov::AnyMap rank_zero_next_to_markov_head{{"dflash_markov_rank", size_t{0}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, rank_zero_next_to_markov_head);
    ASSERT_EQ(rank_zero_next_to_markov_head.count("dflash_markov_head_computation"), 0);

    ov::AnyMap exported_top_k{{"dflash_markov_rank", size_t{4}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, exported_top_k);
    ASSERT_EQ(selected_top_k(exported_top_k), DSPARK_TEST_TOP_K);
    const auto candidate_scorer = selected_computation(exported_top_k);
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_candidate_scorer(candidate_scorer, info));
    EXPECT_EQ(candidate_scorer->output("candidate_ids").get_partial_shape()[1].get_length(), DSPARK_TEST_TOP_K);

    // markov_head_top_k is consumed here so that it never reaches compile_model.
    ov::AnyMap dense_walk{{"dflash_markov_rank", size_t{4}}, {"markov_head_top_k", 0}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, dense_walk);
    ASSERT_EQ(dense_walk.count("markov_head_top_k"), 0);
    ASSERT_EQ(selected_top_k(dense_walk), ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP);
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_markov_step(selected_computation(dense_walk), info));

    ov::AnyMap replaced_top_k{{"dflash_markov_rank", size_t{4}}, {"markov_head_top_k", int64_t{3}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, replaced_top_k);
    ASSERT_EQ(replaced_top_k.count("markov_head_top_k"), 0);
    ASSERT_EQ(selected_top_k(replaced_top_k), 3);
}

TEST(DSparkModelTransforms, SelectsDenseStepOrCandidateScorerAndReplacesTopK) {
    const auto markov_head_model = make_dspark_markov_head_model();
    auto& core = ov::genai::utils::singleton_core();

    const auto markov_step = selected_computation(select_markov_computation(int64_t{0}, markov_head_model));
    ASSERT_EQ(markov_step->inputs().size(), 1);
    ASSERT_EQ(markov_step->outputs().size(), 1);
    EXPECT_FALSE(markov_step->has_rt_info(std::vector<std::string>{"dflash", "top_k"}));
    auto markov_step_request = core.compile_model(markov_step, "CPU").create_infer_request();
    int64_t previous_token = 1;
    markov_step_request.set_tensor("prev_token_ids", ov::Tensor(ov::element::i64, ov::Shape{1}, &previous_token));
    markov_step_request.infer();
    const auto markov_bias = tensor_values(markov_step_request.get_tensor("markov_bias"));
    ASSERT_EQ(markov_bias.size(), DSPARK_TEST_VOCAB_SIZE);
    // codebook[1] = [4, 5, 6, 7], codebook[0] = [0, 1, 2, 3], codebook[2] = [8, 9, 10, 11].
    EXPECT_FLOAT_EQ(markov_bias[0], 38.0f);
    EXPECT_FLOAT_EQ(markov_bias[2], 214.0f);

    constexpr size_t num_positions = 2;
    std::vector<float> draft_logits(num_positions * DSPARK_TEST_VOCAB_SIZE);
    for (size_t position = 0; position < num_positions; ++position) {
        for (size_t token = 0; token < DSPARK_TEST_VOCAB_SIZE; ++token) {
            draft_logits[position * DSPARK_TEST_VOCAB_SIZE + token] =
                static_cast<float>((token * 5 + position * 3) % DSPARK_TEST_VOCAB_SIZE);
        }
    }
    int64_t anchor_token = 0;
    int64_t subsequence_begins[] = {0, static_cast<int64_t>(num_positions)};
    for (const auto& [top_k, expected_top_k] : std::vector<std::pair<std::optional<int64_t>, size_t>>{
             {std::nullopt, DSPARK_TEST_TOP_K},
             {int64_t{3}, 3},
             {static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE), DSPARK_TEST_VOCAB_SIZE}}) {
        SCOPED_TRACE(expected_top_k);
        const auto selected = select_markov_computation(top_k, markov_head_model);
        ASSERT_EQ(selected_top_k(selected), static_cast<int64_t>(expected_top_k));
        const auto candidate_scorer = selected_computation(selected);
        EXPECT_EQ(candidate_scorer->get_rt_info<std::string>(std::vector<std::string>{"dflash", "top_k"}),
                  std::to_string(expected_top_k));
        auto request = core.compile_model(candidate_scorer, "CPU").create_infer_request();
        request.set_tensor(
            "draft_logits",
            ov::Tensor(ov::element::f32, ov::Shape{num_positions, DSPARK_TEST_VOCAB_SIZE}, draft_logits.data()));
        request.set_tensor("anchor_token_ids", ov::Tensor(ov::element::i64, ov::Shape{1}, &anchor_token));
        request.set_tensor("subsequence_begins", ov::Tensor(ov::element::i64, ov::Shape{2}, subsequence_begins));
        request.infer();
        const auto candidate_ids = int64_tensor_values(request.get_tensor("candidate_ids"));
        const auto edge_scores = tensor_values(request.get_tensor("edge_scores"));
        ASSERT_EQ(candidate_ids.size(), num_positions * expected_top_k);
        ASSERT_EQ(edge_scores.size(), num_positions * expected_top_k * expected_top_k);
        for (size_t position = 0; position < num_positions; ++position) {
            const auto* logits = draft_logits.data() + position * DSPARK_TEST_VOCAB_SIZE;
            std::vector<int64_t> ranked_tokens(DSPARK_TEST_VOCAB_SIZE);
            std::iota(ranked_tokens.begin(), ranked_tokens.end(), 0);
            std::sort(ranked_tokens.begin(), ranked_tokens.end(), [&](int64_t lhs, int64_t rhs) {
                return logits[lhs] > logits[rhs];
            });
            for (size_t i = 0; i < expected_top_k; ++i) {
                ASSERT_EQ(candidate_ids[position * expected_top_k + i], ranked_tokens[i]);
                for (size_t j = 0; j < expected_top_k; ++j) {
                    EXPECT_FLOAT_EQ(edge_scores[(position * expected_top_k + i) * expected_top_k + j],
                                    logits[ranked_tokens[i]] + logits[ranked_tokens[j]]);
                }
            }
        }
    }

    // Selection works on a copy: the Markov head model keeps its exported K.
    EXPECT_EQ(markov_head_model->output("candidate_ids").get_partial_shape()[1].get_length(), DSPARK_TEST_TOP_K);
    EXPECT_EQ(markov_head_model->get_rt_info<std::string>(std::vector<std::string>{"dflash", "top_k"}),
              std::to_string(DSPARK_TEST_TOP_K));

    for (const int64_t invalid_top_k : {int64_t{-1}, static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE) + 1}) {
        SCOPED_TRACE(invalid_top_k);
        EXPECT_THROW(select_markov_computation(invalid_top_k, markov_head_model), ov::Exception);
    }
    for (const std::string key : {"top_k", "vocab_size"}) {
        SCOPED_TRACE(key);
        const auto incomplete = make_dspark_markov_head_model(key);
        EXPECT_THROW(select_markov_computation(std::nullopt, incomplete), ov::Exception);
        EXPECT_THROW(select_markov_computation(int64_t{0}, incomplete), ov::Exception);
    }
}

TEST(DSparkModelTransforms, RejectsCandidateScorerThatDoesNotMatchTheBackbone) {
    const auto info = make_dspark_backbone_info();
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(), info));

    auto mismatched_rank = info;
    mismatched_rank.markov_rank = 8;
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(), mismatched_rank),
                 ov::Exception);

    auto wrong_top_k = make_dspark_candidate_scorer();
    wrong_top_k->set_rt_info(std::string("3"), {"dflash", "top_k"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(wrong_top_k, info), ov::Exception);

    // the gated candidate scorer of a DFlash-2 head also reads draft_hidden_states
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(true), info),
                 ov::Exception);

    auto stateful = make_annotated_stateful_sdpa_model();
    set_dspark_markov_head_rt_info(stateful);
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(stateful, info), ov::Exception);

    for (const std::string key : {"top_k", "vocab_size", "markov_rank", "markov_head_type"}) {
        SCOPED_TRACE(key);
        EXPECT_THROW(
            ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(false, key), info),
            ov::Exception);
    }
}

namespace {

using ov::genai::dflash_cb::CandidateSelector;

constexpr size_t SELECTOR_VOCAB_SIZE = 32;
constexpr int64_t SELECTOR_MASK_TOKEN = 31;

// [rows, 1, vocab] logits whose row r has the value values[r] (1 by default) at argmax_tokens[r] and 0 elsewhere.
ov::Tensor make_argmax_logits(const std::vector<int64_t>& argmax_tokens, const std::vector<float>& values = {}) {
    ov::Tensor logits(ov::element::f32, {argmax_tokens.size(), 1, SELECTOR_VOCAB_SIZE});
    std::fill_n(logits.data<float>(), logits.get_size(), 0.0f);
    for (size_t row = 0; row < argmax_tokens.size(); ++row) {
        logits.data<float>()[row * SELECTOR_VOCAB_SIZE + argmax_tokens[row]] = values.empty() ? 1.0f : values[row];
    }
    return logits;
}

ov::Tensor make_zero_logits(size_t rows) {
    ov::Tensor logits(ov::element::f32, {rows, 1, SELECTOR_VOCAB_SIZE});
    std::fill_n(logits.data<float>(), logits.get_size(), 0.0f);
    return logits;
}

// A two-block lattice with K = 2: edge_scores[s, i, j] scores candidate j at row s after candidate i at row s - 1.
std::pair<ov::Tensor, ov::Tensor> make_two_block_lattice() {
    ov::Tensor edge_scores(ov::element::f32, {4, 2, 2});
    ov::Tensor candidate_ids(ov::element::i64, {4, 2});
    const std::vector<float> scores = {
        0.0f, 1.0f, /*unused*/ 5.0f, 5.0f,  // row 0 from the anchor: candidate 1
        3.0f, 2.0f, 0.0f, 4.0f,             // row 1 after candidate 1: candidate 1
        2.0f, 1.0f, /*unused*/ 0.0f, 9.0f,  // row 2 from the anchor: candidate 0
        1.0f, 1.0f, 0.0f, 0.0f,             // row 3 after candidate 0: a tie takes candidate 0
    };
    std::copy(scores.begin(), scores.end(), edge_scores.data<float>());
    const std::vector<int64_t> ids = {10, 11, 12, 13, 20, 21, 22, 23};
    std::copy(ids.begin(), ids.end(), candidate_ids.data<int64_t>());
    return {edge_scores, candidate_ids};
}

// A one-block lattice over two positions with K = 2: position 0 keeps tokens {1, 3}, position 1 tokens {0, 4}. The
// best path drafts 3 then 4; after token 1, token 0 is the better candidate at position 1.
std::pair<ov::Tensor, ov::Tensor> make_one_block_lattice(
    const std::vector<float>& scores = {2.0f, 3.0f, 2.0f, 3.0f, 8.0f, 1.0f, 0.0f, 9.0f},
    const std::vector<int64_t>& ids = {1, 3, 0, 4}) {
    ov::Tensor edge_scores(ov::element::f32, {2, 2, 2});
    ov::Tensor candidate_ids(ov::element::i64, {2, 2});
    std::copy(scores.begin(), scores.end(), edge_scores.data<float>());
    std::copy(ids.begin(), ids.end(), candidate_ids.data<int64_t>());
    return {edge_scores, candidate_ids};
}

std::shared_ptr<ov::op::v0::Parameter> add_named_input(ov::ParameterVector& params,
                                                       const std::string& name,
                                                       const ov::element::Type& type,
                                                       const ov::PartialShape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
    parameter->output(0).set_names({name});
    params.push_back(parameter);
    return parameter;
}

// A candidate scorer that scores any rows with the given lattice.
std::shared_ptr<ov::Model> make_fixed_candidate_scorer(const ov::Tensor& edge_scores, const ov::Tensor& candidate_ids) {
    ov::ParameterVector params;
    add_named_input(params,
                    "draft_logits",
                    ov::element::f32,
                    ov::PartialShape{-1, static_cast<int64_t>(SELECTOR_VOCAB_SIZE)});
    add_named_input(params, "anchor_token_ids", ov::element::i64, ov::PartialShape{-1});
    add_named_input(params, "subsequence_begins", ov::element::i64, ov::PartialShape{-1});
    auto scores_result = std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Constant>(edge_scores));
    scores_result->output(0).set_names({"edge_scores"});
    auto ids_result = std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Constant>(candidate_ids));
    ids_result->output(0).set_names({"candidate_ids"});
    return std::make_shared<ov::Model>(ov::ResultVector{scores_result, ids_result}, params);
}

// A dense Markov step whose bias after token p is 10 at token p + 1 and 0 elsewhere.
std::shared_ptr<ov::Model> make_successor_markov_step() {
    ov::ParameterVector params;
    const auto prev_token_ids = add_named_input(params, "prev_token_ids", ov::element::i64, ov::PartialShape{-1});
    auto successors =
        std::make_shared<ov::op::v1::Add>(prev_token_ids,
                                          ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto markov_bias = std::make_shared<ov::op::v1::OneHot>(
        successors,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(SELECTOR_VOCAB_SIZE)}),
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {10.0f}),
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}),
        -1);
    auto result = std::make_shared<ov::op::v0::Result>(markov_bias);
    result->output(0).set_names({"markov_bias"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, params);
}

std::shared_ptr<CandidateSelector> make_selector(const std::shared_ptr<ov::Model>& markov_head) {
    ov::genai::utils::dflash::DFlashRTInfo rt_info;
    rt_info.markov_head_computation = markov_head;
    return std::make_shared<CandidateSelector>(rt_info, "CPU", ov::AnyMap{});
}

ov::genai::GenerationConfig block_config(size_t max_new_tokens = 100) {
    auto config = ov::genai::utils::get_greedy_config();
    config.max_new_tokens = max_new_tokens;
    return config;
}

// A DFlash draft request after its forward: the prompt {3}, then the anchor and num_rows mask placeholders, all of
// whose rows were gathered.
ov::genai::SequenceGroup::Ptr make_block_request(uint64_t request_id,
                                                 int64_t anchor,
                                                 size_t num_rows,
                                                 const ov::genai::GenerationConfig& config) {
    auto group = std::make_shared<ov::genai::SequenceGroup>(request_id, ov::genai::TokenIds{3}, config);
    const auto sequence = group->get_running_sequences().front();
    sequence->append_token(anchor, 0.0f);
    for (size_t row = 0; row < num_rows; ++row) {
        sequence->append_token(SELECTOR_MASK_TOKEN, 0.0f);
    }
    group->schedule_tokens(2 + num_rows);
    group->set_output_seq_len(num_rows);
    return group;
}

// Drafts the blocks of the requests, whose gathered rows the logits hold in order, from the source's rows.
void draft_blocks(const std::shared_ptr<ov::genai::BlockRowSource>& source,
                  const std::vector<ov::genai::SequenceGroup::Ptr>& requests,
                  const ov::Tensor& logits) {
    ov::genai::Sampler sampler(2);
    sampler.set_block_sampling(1, source);
    sampler.sample(requests, logits, false, false);
}

// The rows of a selector without its faster draws, so the sampler samples every row.
class SelectorRows final : public ov::genai::BlockRowSource {
public:
    explicit SelectorRows(std::shared_ptr<CandidateSelector> selector) : m_selector(std::move(selector)) {}

    void prepare(const std::vector<ov::genai::SampledBlock>& blocks, const ov::Tensor& logits) override {
        m_selector->prepare(blocks, logits);
    }

    bool is_position_major() const override {
        return m_selector->is_position_major();
    }

    void step(size_t position,
              const std::vector<size_t>& blocks,
              const std::vector<int64_t>& previous_tokens) override {
        m_selector->step(position, blocks, previous_tokens);
    }

    bool fill_row(size_t block, size_t position, int64_t previous_token, float* row) const override {
        return m_selector->fill_row(block, position, previous_token, row);
    }

private:
    std::shared_ptr<CandidateSelector> m_selector;
};

// The candidates a request drafted after its anchor.
std::vector<int64_t> drafted(const ov::genai::SequenceGroup::Ptr& request) {
    const auto& ids = request->get_running_sequences().front()->get_generated_ids();
    return std::vector<int64_t>(ids.begin() + 1, ids.end());
}

}  // namespace

TEST(DFlashCBCandidateSelection, RequiresAMarkovHead) {
    EXPECT_THROW(make_selector(nullptr), ov::Exception);
}

TEST(DFlashCBCandidateSelection, RejectsBlockRowsPastTheOutput) {
    auto selector = make_selector(make_successor_markov_step());
    const auto logits = make_zero_logits(4);
    using ov::genai::SampledBlock;
    EXPECT_THROW(selector->prepare({SampledBlock{7, 2, 3, 3}}, logits), ov::Exception);
    EXPECT_THROW(selector->prepare({SampledBlock{7, 0, 2, 3}}, logits), ov::Exception);
    EXPECT_NO_THROW(selector->prepare({SampledBlock{7, 1, 3, 3}}, logits));
}

TEST(DFlashCBCandidateSelection, RejectsDraftOutputsWithEmptyRows) {
    auto selector = make_selector(make_successor_markov_step());
    const ov::Tensor logits(ov::element::f32, {3, 1, 0});
    EXPECT_THROW(selector->prepare({ov::genai::SampledBlock{7, 0, 3, 3}}, logits), ov::Exception);
}

TEST(DFlashCBCandidateSelection, CandidateScorerDraftsTheGreedyLatticePathOfEveryBlock) {
    const auto [edge_scores, candidate_ids] = make_two_block_lattice();
    auto first = make_block_request(0, 7, 2, block_config());
    auto second = make_block_request(1, 8, 2, block_config());

    draft_blocks(make_selector(make_fixed_candidate_scorer(edge_scores, candidate_ids)),
                 {first, second},
                 make_zero_logits(4));

    EXPECT_EQ(drafted(first), (std::vector<int64_t>{11, 13}));
    EXPECT_EQ(drafted(second), (std::vector<int64_t>{20, 22}));
}

TEST(DFlashCBCandidateSelection, CandidateScorerSkipsARequestWithoutCandidatesLeft) {
    // Rows 0-1 and 3-4 hold the blocks; row 2 belongs to a request that has no candidate left to draft, so the
    // scorer packs the rows of the other two blocks back to back.
    const auto [edge_scores, candidate_ids] = make_two_block_lattice();
    auto first = make_block_request(0, 7, 2, block_config());
    auto done = make_block_request(1, 9, 1, block_config(2));
    auto second = make_block_request(2, 8, 2, block_config());

    draft_blocks(make_selector(make_fixed_candidate_scorer(edge_scores, candidate_ids)),
                 {first, done, second},
                 make_zero_logits(5));

    EXPECT_EQ(drafted(first), (std::vector<int64_t>{11, 13}));
    EXPECT_TRUE(drafted(done).empty());
    EXPECT_EQ(drafted(second), (std::vector<int64_t>{20, 22}));
}

TEST(DFlashCBCandidateSelection, DraftSamplerAppliesTheRequestLogitProcessorsToTheLatticeRows) {
    const auto [edge_scores, candidate_ids] = make_one_block_lattice();
    const auto selector = make_selector(make_fixed_candidate_scorer(edge_scores, candidate_ids));
    std::vector<float> backbone = {1.0f, 5.0f, 0.0f, 4.0f, 2.0f, 3.0f};
    backbone.resize(SELECTOR_VOCAB_SIZE, 0.0f);
    std::vector<float> second_row = {6.0f, 0.0f, 2.0f, 1.0f, 5.0f, 3.0f};
    second_row.resize(SELECTOR_VOCAB_SIZE, 0.0f);
    backbone.insert(backbone.end(), second_row.begin(), second_row.end());
    ov::Tensor logits(ov::element::f32, {2, 1, SELECTOR_VOCAB_SIZE});
    std::copy(backbone.begin(), backbone.end(), logits.data<float>());

    auto greedy = make_block_request(0, 2, 2, block_config());
    draft_blocks(selector, {greedy}, logits);
    EXPECT_EQ(drafted(greedy), (std::vector<int64_t>{3, 4}));

    // The penalty halves the score of the prompt token 3 (3 -> 1.5), so token 1 (2) wins and token 0 follows it.
    auto penalized_config = block_config();
    penalized_config.repetition_penalty = 2.0f;
    auto penalized = make_block_request(0, 2, 2, penalized_config);
    draft_blocks(selector, {penalized}, logits);
    EXPECT_EQ(drafted(penalized), (std::vector<int64_t>{1, 0}));
}

TEST(DFlashCBCandidateSelection, DraftSamplerBreaksTiesByTheSmallerTokenIdLikeTheGreedyPath) {
    // Position 0 ties tokens 3 and 1, listed in that order; token 1 then leads to token 4, token 3 to token 0.
    const auto [edge_scores, candidate_ids] =
        make_one_block_lattice({5.0f, 5.0f, 5.0f, 5.0f, 9.0f, 0.0f, 0.0f, 9.0f}, {3, 1, 0, 4});
    ASSERT_EQ(ov::genai::dflash_cb::greedy_candidate_path(edge_scores, candidate_ids, 0, 2),
              (std::vector<int64_t>{1, 4}));

    // min_new_tokens without stop tokens changes no logit but sends the request through the draft sampler
    auto config = block_config();
    config.min_new_tokens = 1;
    ASSERT_FALSE(ov::genai::dflash_cb::logit_processors_keep_argmax(config));
    auto request = make_block_request(0, 2, 2, config);
    draft_blocks(make_selector(make_fixed_candidate_scorer(edge_scores, candidate_ids)),
                 {request},
                 make_zero_logits(2));
    EXPECT_EQ(drafted(request), (std::vector<int64_t>{1, 4}));
}

TEST(DFlashCBCandidateSelection, TokenOutsideTheCandidatesEndsTheBlock) {
    // Far below 1, the repetition penalty lifts the prompt token 3 from below the candidates above them, and the
    // lattice has no row after it.
    const auto [edge_scores, candidate_ids] =
        make_one_block_lattice({-50.0f, -40.0f, -50.0f, -40.0f, 0.0f, 1.0f, 0.0f, 1.0f}, {1, 4, 0, 5});
    auto logits = make_argmax_logits({3, 3});
    auto config = block_config();
    config.repetition_penalty = 0.001f;
    auto request = make_block_request(0, 2, 2, config);

    draft_blocks(make_selector(make_fixed_candidate_scorer(edge_scores, candidate_ids)), {request}, logits);

    EXPECT_EQ(drafted(request), (std::vector<int64_t>{3}));
}

TEST(DFlashCBCandidateSelection, DenseMarkovStepBiasesEveryPositionByTheTokenDraftedBefore) {
    // The backbone's best token is 30 at every row; the Markov bias after token p prefers p + 1.
    const auto logits = make_argmax_logits({30, 30, 30, 30, 30});
    const auto selector = make_selector(make_successor_markov_step());
    ASSERT_TRUE(selector->is_position_major());
    auto first = make_block_request(0, 7, 3, block_config());
    auto second = make_block_request(1, 20, 2, block_config());

    draft_blocks(selector, {first, second}, logits);

    EXPECT_EQ(drafted(first), (std::vector<int64_t>{8, 9, 10}));
    EXPECT_EQ(drafted(second), (std::vector<int64_t>{21, 22}));
}

TEST(DFlashCBCandidateSelection, DenseMarkovStepDraftsAGreedyBlockAsTheSamplerDrawsItsRows) {
    // The backbone's best token beats the Markov bias of 10 at rows 0 and 2 and loses to it at rows 1 and 3.
    const auto logits = make_argmax_logits({20, 3, 29, 12}, {15.0f, 5.0f, 15.0f, 5.0f});
    const auto selector = make_selector(make_successor_markov_step());
    auto drawn = make_block_request(0, 7, 4, block_config());
    auto sampled = make_block_request(0, 7, 4, block_config());

    draft_blocks(selector, {drawn}, logits);
    draft_blocks(std::make_shared<SelectorRows>(make_selector(make_successor_markov_step())), {sampled}, logits);

    EXPECT_EQ(drafted(drawn), (std::vector<int64_t>{20, 21, 29, 30}));
    EXPECT_EQ(drafted(drawn), drafted(sampled));
    auto penalized = block_config();
    penalized.repetition_penalty = 2.0f;
    selector->prepare({ov::genai::SampledBlock{7, 0, 4, 4}}, logits);
    selector->step(0, {0}, {7});
    EXPECT_TRUE(selector->draw_token(0, 0, block_config()));
    EXPECT_FALSE(selector->draw_token(0, 0, penalized));
}

TEST(DFlashCBCandidateSelection, GreedyPathWalksEachBlockFromItsAnchor) {
    const auto [edge_scores, candidate_ids] = make_two_block_lattice();
    using ov::genai::dflash_cb::greedy_candidate_path;
    ASSERT_EQ(greedy_candidate_path(edge_scores, candidate_ids, 0, 2), (std::vector<int64_t>{11, 13}));
    ASSERT_EQ(greedy_candidate_path(edge_scores, candidate_ids, 2, 2), (std::vector<int64_t>{20, 22}));
    ASSERT_EQ(greedy_candidate_path(edge_scores, candidate_ids, 2, 1), (std::vector<int64_t>{20}));
}

TEST(DFlashCBCandidateSelection, LogitProcessorsKeepTheArgmaxOnlyWithoutHistoryOrSampling) {
    using ov::genai::dflash_cb::logit_processors_keep_argmax;
    auto config = ov::genai::utils::get_greedy_config();
    ASSERT_TRUE(logit_processors_keep_argmax(config));
    config.logprobs = 2;
    ASSERT_TRUE(logit_processors_keep_argmax(config));

    for (const auto& change : std::vector<std::function<void(ov::genai::GenerationConfig&)>>{
             [](auto& c) { c.repetition_penalty = 1.1f; },
             [](auto& c) { c.presence_penalty = 0.1f; },
             [](auto& c) { c.frequency_penalty = 0.1f; },
             [](auto& c) { c.min_new_tokens = 1; },
             [](auto& c) { c.structured_output_config = ov::genai::StructuredOutputConfig(); },
             [](auto& c) { c.do_sample = true; },
         }) {
        auto changed = ov::genai::utils::get_greedy_config();
        change(changed);
        ASSERT_FALSE(logit_processors_keep_argmax(changed));
    }
}

TEST(DFlashCBCandidateSelection, ValidatesSubsequenceBegins) {
    const auto begins = [](std::vector<int64_t> values) {
        ov::Tensor tensor(ov::element::i64, ov::Shape{values.size()});
        std::copy(values.begin(), values.end(), tensor.data<int64_t>());
        return tensor;
    };
    using ov::genai::dflash_cb::validate_subsequence_begins;
    EXPECT_NO_THROW(validate_subsequence_begins(begins({0, 3}), 1, 3));
    EXPECT_NO_THROW(validate_subsequence_begins(begins({0, 3, 4, 8}), 3, 8));

    EXPECT_THROW(validate_subsequence_begins(begins({0}), 0, 0), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 3, 8}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({1, 3, 4, 8}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 3, 4, 7}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 3, 3, 8}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 5, 4, 8}), 3, 8), ov::Exception);
    int32_t i32_values[] = {0, 3};
    EXPECT_THROW(validate_subsequence_begins(ov::Tensor(ov::element::i32, ov::Shape{2}, i32_values), 1, 3),
                 ov::Exception);
}
