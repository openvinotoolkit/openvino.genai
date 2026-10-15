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
#include <vector>

#include "logger.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
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
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
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

constexpr size_t DSPARK_TEST_VOCAB_SIZE = 16;
constexpr size_t DSPARK_TEST_MARKOV_RANK = 4;
constexpr int64_t DSPARK_TEST_TOP_K = 2;

void set_dspark_markov_head_rt_info(const std::shared_ptr<ov::Model>& model, const std::string& omitted_key = "") {
    const std::vector<std::pair<std::string, std::string>> rt_info = {
        {"interface", "markov_head_v1"},
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

int64_t selected_top_k(const ov::AnyMap& properties) {
    return properties.at("dflash_markov_head_top_k").as<int64_t>();
}

void set_markov_head_computation(ov::AnyMap& properties, const std::shared_ptr<ov::Model>& computation, int64_t top_k) {
    properties["dflash_markov_head_computation"] = computation;
    properties["dflash_markov_head_top_k"] = top_k;
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
    auto input_ids_default = ov::genai::dflash_cb::build_draft_input_ids(42, 99, 3);
    ASSERT_EQ(input_ids_default.get_shape(), ov::Shape({1, 4}));
    ASSERT_EQ(int64_tensor_values(input_ids_default), (std::vector<int64_t>{42, 99, 99, 99}));

    auto input_ids_deepspec = ov::genai::dflash_cb::build_draft_input_ids(42, 99, 3, /*candidate_position_offset=*/0);
    ASSERT_EQ(input_ids_deepspec.get_shape(), ov::Shape({1, 3}));
    ASSERT_EQ(int64_tensor_values(input_ids_deepspec), (std::vector<int64_t>{42, 99, 99}));
}

TEST(DFlashCBDraftInputs, BuildsPositionIdsFromCommittedLength) {
    auto position_ids_default = ov::genai::dflash_cb::build_draft_position_ids(5, 2, 3);
    ASSERT_EQ(position_ids_default.get_shape(), ov::Shape({1, 6}));
    ASSERT_EQ(int64_tensor_values(position_ids_default), (std::vector<int64_t>{5, 6, 7, 8, 9, 10}));

    auto position_ids_deepspec =
        ov::genai::dflash_cb::build_draft_position_ids(5, 2, 3, /*candidate_position_offset=*/0);
    ASSERT_EQ(position_ids_deepspec.get_shape(), ov::Shape({1, 5}));
    ASSERT_EQ(int64_tensor_values(position_ids_deepspec), (std::vector<int64_t>{5, 6, 7, 8, 9}));
}

TEST(DFlashCBDraftInputs, BuildsAttentionMaskForFullDraftContext) {
    auto attention_mask_default = ov::genai::dflash_cb::build_draft_attention_mask(5, 2, 3);
    ASSERT_EQ(attention_mask_default.get_shape(), ov::Shape({1, 11}));
    ASSERT_EQ(int64_tensor_values(attention_mask_default),
              (std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

    auto attention_mask_deepspec =
        ov::genai::dflash_cb::build_draft_attention_mask(5, 2, 3, /*candidate_position_offset=*/0);
    ASSERT_EQ(attention_mask_deepspec.get_shape(), ov::Shape({1, 10}));
    ASSERT_EQ(int64_tensor_values(attention_mask_deepspec),
              (std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(DFlashCBVlmPromptIds, BuildsPlaceholderPromptIds) {
    ASSERT_EQ(ov::genai::dflash_cb::build_placeholder_prompt_ids(3, 42), (std::vector<int64_t>{42, 42, 42}));
}

TEST(DFlashCBVlmPromptIds, RejectsInvalidPromptLengthAndPlaceholderId) {
    EXPECT_THROW(ov::genai::dflash_cb::build_placeholder_prompt_ids(0, 42), ov::Exception);
    EXPECT_THROW(ov::genai::dflash_cb::build_placeholder_prompt_ids(1, -1), ov::Exception);
}

TEST(DFlashCBGenerationConfig, DefaultsAssistantTokensToFive) {
    ov::genai::GenerationConfig config;

    ov::genai::dflash_cb::ensure_num_assistant_tokens_is_set(config);

    ASSERT_EQ(config.num_assistant_tokens, ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS);
}

TEST(DFlashCBGenerationConfig, PreservesUnsetMaxNewTokensForDraft) {
    ov::genai::GenerationConfig config;
    ov::genai::dflash_cb::ensure_num_assistant_tokens_is_set(config);
    auto draft_config = config;

    ov::genai::dflash_cb::add_draft_assistant_token_headroom(draft_config, config);

    ASSERT_EQ(draft_config.max_new_tokens, SIZE_MAX);
}

TEST(DFlashCBGenerationConfig, ExtendsFiniteActiveLimitForDraft) {
    constexpr size_t assistant_tokens = ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS;

    ov::genai::GenerationConfig max_new_tokens_config;
    max_new_tokens_config.max_new_tokens = 10;
    max_new_tokens_config.num_assistant_tokens = assistant_tokens;
    auto max_new_tokens_draft_config = max_new_tokens_config;
    ov::genai::dflash_cb::add_draft_assistant_token_headroom(max_new_tokens_draft_config,
                                                              max_new_tokens_config);
    ASSERT_EQ(max_new_tokens_draft_config.max_new_tokens, 10 + assistant_tokens);

    ov::genai::GenerationConfig max_length_config;
    max_length_config.max_length = 100;
    max_length_config.num_assistant_tokens = assistant_tokens;
    auto max_length_draft_config = max_length_config;
    ov::genai::dflash_cb::add_draft_assistant_token_headroom(max_length_draft_config, max_length_config);
    ASSERT_EQ(max_length_draft_config.max_length, 100 + assistant_tokens);
}

TEST(DFlashCBGenerationConfig, PreservesFiniteLimitWhenHeadroomWouldOverflow) {
    constexpr size_t assistant_tokens = ov::genai::dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS;

    ov::genai::GenerationConfig max_new_tokens_config;
    max_new_tokens_config.max_new_tokens = SIZE_MAX - 2;
    max_new_tokens_config.num_assistant_tokens = assistant_tokens;
    auto max_new_tokens_draft_config = max_new_tokens_config;
    ov::genai::dflash_cb::add_draft_assistant_token_headroom(max_new_tokens_draft_config,
                                                              max_new_tokens_config);
    ASSERT_EQ(max_new_tokens_draft_config.max_new_tokens, max_new_tokens_config.max_new_tokens);

    ov::genai::GenerationConfig max_length_config;
    max_length_config.max_length = SIZE_MAX - 2;
    max_length_config.num_assistant_tokens = assistant_tokens;
    auto max_length_draft_config = max_length_config;
    ov::genai::dflash_cb::add_draft_assistant_token_headroom(max_length_draft_config, max_length_config);
    ASSERT_EQ(max_length_draft_config.max_length, max_length_config.max_length);
}

TEST(DFlashCBGenerationConfig, PreservesExplicitAssistantTokens) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 2;

    ov::genai::dflash_cb::ensure_num_assistant_tokens_is_set(config);

    ASSERT_EQ(config.num_assistant_tokens, 2);
}

TEST(DFlashCBGenerationConfig, AcceptsSupportedVlmOptions) {
    ov::genai::GenerationConfig config;

    EXPECT_NO_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config));
}

TEST(DFlashCBGenerationConfig, RejectsUnsupportedVlmOptions) {
    {
        ov::genai::GenerationConfig config;
        config.pruning_ratio = 1;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.echo = true;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.logprobs = 1;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.return_omni_outputs = true;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
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
    set_markov_head_computation(properties, markov_step, ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(rt_info.candidate_position_offset, 0);
    ASSERT_EQ(rt_info.markov_rank, 4);
    ASSERT_EQ(rt_info.markov_head_type, "vanilla");
    ASSERT_EQ(rt_info.markov_head_computation, markov_step);
    ASSERT_EQ(rt_info.markov_head_top_k, ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP);
    ASSERT_TRUE(properties.empty());
}

TEST(DSparkModelTransforms, KeepsDFlashDraftWithoutMarkovHeadUnchanged) {
    auto properties = make_dspark_draft_properties(0, "vanilla");
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(rt_info.markov_rank, 0);
    ASSERT_EQ(rt_info.markov_head_computation, nullptr);
    ASSERT_TRUE(properties.empty());
}

TEST(DSparkModelTransforms, AppliesMarkovRankZeroBackboneRtInfoAsPlainDFlash) {
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
}

TEST(DSparkModelTransforms, RejectsMarkovHeadWithoutSelectedComputation) {
    auto properties = make_dspark_draft_properties(4, "vanilla");
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(properties), ov::Exception);
}

TEST(DSparkModelTransforms, ExtractsSelectedCandidateScorer) {
    auto candidate_scorer = make_dspark_candidate_scorer();
    auto properties = make_dspark_draft_properties(4, "vanilla");
    set_markov_head_computation(properties, candidate_scorer, DSPARK_TEST_TOP_K);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(rt_info.markov_head_computation, candidate_scorer);
    ASSERT_EQ(rt_info.markov_head_top_k, DSPARK_TEST_TOP_K);
    ASSERT_TRUE(properties.empty());

    // The selected top_k decides how the computation is validated and run.
    auto scorer_as_dense_step = make_dspark_draft_properties(4, "vanilla");
    set_markov_head_computation(scorer_as_dense_step,
                                candidate_scorer,
                                ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP);
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(scorer_as_dense_step), ov::Exception);

    auto dense_step_as_scorer = make_dspark_draft_properties(4, "vanilla");
    set_markov_head_computation(dense_step_as_scorer, make_dspark_markov_step(), DSPARK_TEST_TOP_K);
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(dense_step_as_scorer), ov::Exception);

    auto gated_candidate_scorer = make_dspark_candidate_scorer();
    gated_candidate_scorer->set_rt_info(std::string("gated"), {"dflash", "markov_head_type"});
    auto mismatched_head = make_dspark_draft_properties(4, "vanilla");
    set_markov_head_computation(mismatched_head, gated_candidate_scorer, DSPARK_TEST_TOP_K);
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(mismatched_head), ov::Exception);

    auto mismatched_rank = make_dspark_draft_properties(8, "vanilla");
    set_markov_head_computation(mismatched_rank, make_dspark_candidate_scorer(), DSPARK_TEST_TOP_K);
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(mismatched_rank), ov::Exception);
}

TEST(DSparkModelTransforms, RejectsUnsupportedMarkovHeadType) {
    for (const std::string head_type : {"gated", "rnn"}) {
        auto properties = make_dspark_draft_properties(4, head_type);
        set_markov_head_computation(properties,
                                    make_dspark_markov_step(),
                                    ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP);
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

    auto wrong_interface = make_dspark_markov_step();
    wrong_interface->set_rt_info(std::string("markov_step_v1"), {"dflash", "interface"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_markov_step(wrong_interface, info), ov::Exception);

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
    const auto top_3_scorer = selected_computation(replaced_top_k);
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_candidate_scorer(top_3_scorer, info));
    EXPECT_EQ(top_3_scorer->output("candidate_ids").get_partial_shape()[1].get_length(), 3);
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
}

TEST(DSparkModelTransforms, RejectsMarkovHeadModelWithoutSelectionRtInfo) {
    for (const std::string key : {"interface", "top_k", "vocab_size"}) {
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

    auto wrong_interface = make_dspark_candidate_scorer();
    wrong_interface->set_rt_info(std::string("topk_lattice_v1"), {"dflash", "interface"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(wrong_interface, info), ov::Exception);

    auto wrong_top_k = make_dspark_candidate_scorer();
    wrong_top_k->set_rt_info(std::string("3"), {"dflash", "top_k"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(wrong_top_k, info), ov::Exception);

    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(true), info),
                 ov::Exception);

    auto stateful = make_annotated_stateful_sdpa_model();
    set_dspark_markov_head_rt_info(stateful);
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(stateful, info), ov::Exception);
}

TEST(DSparkModelTransforms, RejectsCandidateScorerWithIncompleteRtInfo) {
    const auto info = make_dspark_backbone_info();
    for (const std::string key : {"interface", "top_k", "vocab_size", "markov_rank", "markov_head_type"}) {
        SCOPED_TRACE(key);
        const auto incomplete = make_dspark_candidate_scorer(false, key);
        EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(incomplete, info), ov::Exception);
    }
}

TEST(DSparkModelTransforms, WarnsAboutCandidateScorerWithFewCandidates) {
    const auto info = make_dspark_backbone_info();
    const auto markov_head_model = make_dspark_markov_head_model();
    auto& logger = ov::genai::Logger::get_instance();
    const auto validation_warnings = [&](int64_t top_k) {
        const auto candidate_scorer = selected_computation(select_markov_computation(top_k, markov_head_model));
        logger.set_log_level(ov::log::Level::WARNING);
        testing::internal::CaptureStdout();
        ov::genai::utils::dflash::validate_candidate_scorer(candidate_scorer, info);
        const auto output = testing::internal::GetCapturedStdout();
        logger.set_log_level(ov::genai::get_cur_log_level());
        return output;
    };

    EXPECT_NE(validation_warnings(1).find("the Markov head has no effect"), std::string::npos);
    EXPECT_NE(validation_warnings(4).find("keeps only 4 candidates"), std::string::npos);
    EXPECT_TRUE(validation_warnings(8).empty());
    EXPECT_TRUE(validation_warnings(static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE)).empty());
}

TEST(DSparkCandidateScorer, ValidatesSubsequenceBegins) {
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

namespace {

// A two-position lattice over a vocabulary of 6 with K = 2. Position 0 keeps tokens {1, 3}, position 1 tokens {0, 4}.
// The best path drafts 3 then 4; after token 1, token 0 is the better candidate at position 1.
struct CandidateLattice {
    ov::Tensor backbone_logits{ov::element::f32, ov::Shape{1, 2, 6}};
    ov::Tensor edge_scores{ov::element::f32, ov::Shape{2, 2, 2}};
    ov::Tensor candidate_ids{ov::element::i64, ov::Shape{2, 2}};
    ov::Tensor row{ov::element::f32, ov::Shape{1, 1, 6}};

    CandidateLattice() {
        const std::vector<float> logits = {
            1.0f, 5.0f, 0.0f, 4.0f, 2.0f, 3.0f,
            6.0f, 0.0f, 2.0f, 1.0f, 5.0f, 3.0f,
        };
        const std::vector<float> scores = {
            2.0f, 3.0f,
            2.0f, 3.0f,
            8.0f, 1.0f,
            0.0f, 9.0f,
        };
        const std::vector<int64_t> ids = {1, 3, 0, 4};
        std::copy(logits.begin(), logits.end(), backbone_logits.data<float>());
        std::copy(scores.begin(), scores.end(), edge_scores.data<float>());
        std::copy(ids.begin(), ids.end(), candidate_ids.data<int64_t>());
    }

    template <typename SampleToken>
    std::vector<int64_t> walk(SampleToken&& sample_token) {
        std::vector<int64_t> drafted;
        ov::genai::dflash_cb::walk_candidate_lattice(backbone_logits,
                                                     edge_scores,
                                                     candidate_ids,
                                                     2,
                                                     row,
                                                     [&](const ov::Tensor& candidate_row) -> std::optional<int64_t> {
                                                         const auto token = sample_token(candidate_row);
                                                         if (token) {
                                                             drafted.push_back(*token);
                                                         }
                                                         return token;
                                                     });
        return drafted;
    }
};

std::optional<int64_t> argmax_token(const ov::Tensor& row) {
    const auto* data = row.data<const float>();
    return std::distance(data, std::max_element(data, data + row.get_size()));
}

// Drafts through a greedy sampler the way the draft runner does, with the prompt {3} and the anchor token 2.
std::vector<int64_t> draft_through_sampler(CandidateLattice& lattice, float repetition_penalty) {
    ov::genai::GenerationConfig config = ov::genai::utils::get_greedy_config();
    config.max_new_tokens = 8;
    config.repetition_penalty = repetition_penalty;
    ov::genai::Sampler sampler;
    auto group = std::make_shared<ov::genai::SequenceGroup>(0, ov::genai::TokenIds{3}, config);
    auto sequence = (*group)[0];
    sequence->append_token(2, 0.0f);
    sequence->set_status(ov::genai::SequenceStatus::RUNNING);
    group->update_processed_tokens_num(2);
    return lattice.walk([&](const ov::Tensor& row) -> std::optional<int64_t> {
        group->schedule_tokens(1);
        group->set_output_seq_len(1);
        group->set_num_validated_tokens(0);
        sampler.sample({group}, row, false);
        group->finish_iteration();
        return sequence->get_generated_ids().back();
    });
}

}  // namespace

TEST(DSparkCandidateScorer, SpreadsTheLatticeRowOverTheVocabulary) {
    CandidateLattice lattice;
    std::vector<float> row(6);
    ov::genai::dflash_cb::fill_candidate_row(lattice.backbone_logits.data<const float>() + 6,
                                             6,
                                             lattice.edge_scores,
                                             lattice.candidate_ids,
                                             1,
                                             1,
                                             row.data());

    // Candidates 0 and 4 get the scores after candidate 1 (token 3). The other tokens keep their backbone order,
    // CANDIDATE_ROW_GAP below the worst candidate score for the best backbone logit.
    const float shift = 0.0f - ov::genai::dflash_cb::CANDIDATE_ROW_GAP - 6.0f;
    const std::vector<float> expected = {0.0f, 0.0f + shift, 2.0f + shift, 1.0f + shift, 9.0f, 3.0f + shift};
    for (size_t token = 0; token < row.size(); ++token) {
        EXPECT_FLOAT_EQ(row[token], expected[token]) << "token " << token;
    }
}

TEST(DSparkCandidateScorer, WalksTheRowOfTheDraftedCandidate) {
    CandidateLattice lattice;
    EXPECT_EQ(lattice.walk(argmax_token), (std::vector<int64_t>{3, 4}));

    // A processor that masks token 3 moves the walk to the row of token 1.
    const auto mask_token_3 = [](ov::Tensor row) {
        row.data<float>()[3] = -std::numeric_limits<float>::infinity();
        return argmax_token(row);
    };
    EXPECT_EQ(lattice.walk(mask_token_3), (std::vector<int64_t>{1, 0}));

    // With both candidates masked, the token comes from the remaining backbone logits and ends the block.
    const auto mask_candidates_1_and_3 = [](ov::Tensor row) {
        row.data<float>()[1] = -std::numeric_limits<float>::infinity();
        row.data<float>()[3] = -std::numeric_limits<float>::infinity();
        return argmax_token(row);
    };
    EXPECT_EQ(lattice.walk(mask_candidates_1_and_3), (std::vector<int64_t>{5}));

    size_t calls = 0;
    const auto stop_after_one = [&](const ov::Tensor& row) -> std::optional<int64_t> {
        return calls++ == 0 ? argmax_token(row) : std::nullopt;
    };
    EXPECT_EQ(lattice.walk(stop_after_one), (std::vector<int64_t>{3}));
}

TEST(DSparkCandidateScorer, TakesTheBestCandidateInTheRowOfThePreviousOne) {
    CandidateLattice lattice;
    EXPECT_EQ(ov::genai::dflash_cb::greedy_candidate_path(lattice.edge_scores, lattice.candidate_ids),
              (std::vector<int64_t>{3, 4}));
}

TEST(DSparkCandidateScorer, BreaksTiesByTheSmallerTokenIdLikeTheSampler) {
    // Position 0 ties tokens 3 and 1, listed in that order; token 1 then leads to token 4, token 3 to token 0.
    CandidateLattice lattice;
    const std::vector<int64_t> ids = {3, 1, 0, 4};
    const std::vector<float> scores = {
        5.0f, 5.0f,
        5.0f, 5.0f,
        9.0f, 0.0f,
        0.0f, 9.0f,
    };
    std::copy(ids.begin(), ids.end(), lattice.candidate_ids.data<int64_t>());
    std::copy(scores.begin(), scores.end(), lattice.edge_scores.data<float>());

    EXPECT_EQ(ov::genai::dflash_cb::greedy_candidate_path(lattice.edge_scores, lattice.candidate_ids),
              (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(draft_through_sampler(lattice, 1.0f), (std::vector<int64_t>{1, 4}));
}

TEST(DSparkCandidateScorer, SkipsTheSamplerOnlyWhenTheLogitProcessorsKeepTheArgmax) {
    using ov::genai::dflash_cb::logit_processors_keep_argmax;
    const auto keeps_argmax = [](const std::function<void(ov::genai::GenerationConfig&)>& change) {
        ov::genai::GenerationConfig config = ov::genai::utils::get_greedy_config();
        change(config);
        return logit_processors_keep_argmax(config);
    };
    EXPECT_TRUE(keeps_argmax([](ov::genai::GenerationConfig&) {}));
    EXPECT_TRUE(keeps_argmax([](ov::genai::GenerationConfig& config) {
        config.logprobs = 2;
    }));

    EXPECT_FALSE(keeps_argmax([](ov::genai::GenerationConfig& config) {
        config.repetition_penalty = 1.2f;
    }));
    EXPECT_FALSE(keeps_argmax([](ov::genai::GenerationConfig& config) {
        config.presence_penalty = 1.5f;
    }));
    EXPECT_FALSE(keeps_argmax([](ov::genai::GenerationConfig& config) {
        config.frequency_penalty = 0.5f;
    }));
    EXPECT_FALSE(keeps_argmax([](ov::genai::GenerationConfig& config) {
        config.min_new_tokens = 4;
    }));
    EXPECT_FALSE(keeps_argmax([](ov::genai::GenerationConfig& config) {
        config.structured_output_config = ov::genai::StructuredOutputConfig();
    }));
    EXPECT_FALSE(keeps_argmax([](ov::genai::GenerationConfig& config) {
        config.do_sample = true;
    }));
}

TEST(DSparkCandidateScorer, AppliesTheRequestLogitProcessors) {
    CandidateLattice lattice;
    EXPECT_EQ(draft_through_sampler(lattice, 1.0f), (std::vector<int64_t>{3, 4}));
    // The penalty halves the score of the prompt token 3 (3 -> 1.5), so token 1 (2) wins and token 0 follows it.
    EXPECT_EQ(draft_through_sampler(lattice, 2.0f), (std::vector<int64_t>{1, 0}));
}
