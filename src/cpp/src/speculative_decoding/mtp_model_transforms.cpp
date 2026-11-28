// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mtp_model_transforms.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace mtp {

namespace {

ov::Output<ov::Node> find_result_source(const std::shared_ptr<ov::Model>& model, const std::string& tensor_name) {
    for (const auto& result : model->get_results()) {
        const auto source = result->input_value(0);
        if (result->output(0).get_names().count(tensor_name) != 0 ||
            source.get_names().count(tensor_name) != 0 ||
            result->get_friendly_name() == tensor_name) {
            return source;
        }
    }
    return {};
}

// Clone into the MTP model without sharing source-model constants.
std::shared_ptr<ov::Node> clone_subgraph(const std::shared_ptr<ov::Node>& node,
                                         std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>& cloned_nodes) {
    const auto it = cloned_nodes.find(node.get());
    if (it != cloned_nodes.end()) {
        return it->second;
    }

    std::shared_ptr<ov::Node> cloned;
    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        cloned = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(),
                                                        constant->get_shape(),
                                                        constant->get_data_ptr());
    } else {
        ov::OutputVector cloned_inputs;
        cloned_inputs.reserve(node->get_input_size());
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            auto cloned_input = clone_subgraph(node->get_input_node_shared_ptr(i), cloned_nodes);
            cloned_inputs.push_back(cloned_input->output(node->get_input_source_output(i).get_index()));
        }
        cloned = node->clone_with_new_inputs(cloned_inputs);
    }

    cloned->set_friendly_name(node->get_friendly_name() + "_mtp_lm_head");
    cloned_nodes[node.get()] = cloned;
    return cloned;
}

// The MTP head input when it is not the lm_head input (Qwen4Exp: the pre-final-mixer hyper-connection streams),
// located by the `mtp_hidden_states` rt_info JSON {"producer", "output_index", "hidden_size"} of the main model.
ov::Output<ov::Node> find_mtp_hidden_states_producer(const std::shared_ptr<ov::Model>& main_model) {
    constexpr const char* MTP_HIDDEN_STATES_RT_INFO_KEY = "mtp_hidden_states";
    if (!main_model->has_rt_info(MTP_HIDDEN_STATES_RT_INFO_KEY)) {
        return {};
    }
    const auto annotation = nlohmann::json::parse(main_model->get_rt_info<std::string>(MTP_HIDDEN_STATES_RT_INFO_KEY));
    const auto producer = annotation.at("producer").get<std::string>();
    const auto output_index = annotation.at("output_index").get<size_t>();
    std::shared_ptr<ov::Node> node;
    for (const auto& op : main_model->get_ordered_ops()) {
        if (op->get_friendly_name() == producer) {
            OPENVINO_ASSERT(!node, "MTP hidden-state producer ", producer, " is not unique in the main model.");
            node = op;
        }
    }
    OPENVINO_ASSERT(node && output_index < node->get_output_size(),
                    "Failed to locate MTP hidden-state producer ", producer, ":", output_index, " in the main model.");
    const auto& shape = node->get_output_partial_shape(output_index);
    if (annotation.contains("hidden_size")) {
        OPENVINO_ASSERT(shape.rank().is_static() && shape[shape.size() - 1].compatible(annotation.at("hidden_size").get<int64_t>()),
                        "MTP hidden-state producer ", producer, " has shape ", shape, ", expected hidden size ",
                        annotation.at("hidden_size").get<int64_t>());
    }
    return node->output(output_index);
}

std::vector<int64_t> parse_draft_vocab(const ov::Any& value) {
    std::vector<int64_t> draft_vocab;
    if (value.is<std::vector<int64_t>>()) {
        draft_vocab = value.as<std::vector<int64_t>>();
    } else if (value.is<std::vector<int32_t>>()) {
        const auto ids = value.as<std::vector<int32_t>>();
        draft_vocab.assign(ids.begin(), ids.end());
    } else {
        OPENVINO_THROW(MTP_DRAFT_VOCAB_PROPERTY, " must be a list of token ids (std::vector<int64_t>)");
    }
    OPENVINO_ASSERT(!draft_vocab.empty(), MTP_DRAFT_VOCAB_PROPERTY, " must not be empty");
    std::sort(draft_vocab.begin(), draft_vocab.end());
    draft_vocab.erase(std::unique(draft_vocab.begin(), draft_vocab.end()), draft_vocab.end());
    OPENVINO_ASSERT(draft_vocab.front() >= 0,
                    MTP_DRAFT_VOCAB_PROPERTY,
                    " contains a negative token id ",
                    draft_vocab.front());
    return draft_vocab;
}

// Copies rows `ids` of a constant whose first dimension is the vocabulary.
std::shared_ptr<ov::op::v0::Constant> slice_vocab_rows(const std::shared_ptr<ov::op::v0::Constant>& constant,
                                                       const std::vector<int64_t>& ids) {
    const auto& shape = constant->get_shape();
    const size_t vocab_size = shape[0];
    const size_t row_bits = ov::shape_size(shape) / vocab_size * constant->get_element_type().bitwidth();
    OPENVINO_ASSERT(row_bits % 8 == 0,
                    "MTP draft vocabulary: the rows of lm_head constant ",
                    constant->get_friendly_name(),
                    " are not byte-aligned");
    const size_t row_bytes = row_bits / 8;

    auto sliced_shape = shape;
    sliced_shape[0] = ids.size();
    std::vector<uint8_t> rows(ids.size() * row_bytes);
    const auto* src = static_cast<const uint8_t*>(constant->get_data_ptr());
    for (size_t row = 0; row < ids.size(); ++row) {
        std::memcpy(rows.data() + row * row_bytes, src + static_cast<size_t>(ids[row]) * row_bytes, row_bytes);
    }
    auto sliced = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(), sliced_shape, rows.data());
    sliced->set_friendly_name(constant->get_friendly_name() + "_draft_vocab");
    return sliced;
}

}  // namespace

MtpRTInfo extract_mtp_info_from_config(ov::AnyMap& config) {
    MtpRTInfo mtp_rt_info;
    const auto it = config.find("mtp_mode");
    if (it != config.end()) {
        mtp_rt_info.mtp_mode = it->second.as<bool>();
        config.erase(it);
    }
    const auto vocab_it = config.find(MTP_DRAFT_VOCAB_PROPERTY);
    if (vocab_it != config.end()) {
        OPENVINO_ASSERT(mtp_rt_info.mtp_mode, MTP_DRAFT_VOCAB_PROPERTY, " is supported for MTP draft models only");
        mtp_rt_info.draft_vocab = parse_draft_vocab(vocab_it->second);
        config.erase(vocab_it);
    }
    return mtp_rt_info;
}

void apply_mtp_rt_info(const std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (model->has_rt_info("mtp_mode") && model->get_rt_info<bool>("mtp_mode")) {
        properties["mtp_mode"] = true;
    }
}

ov::Output<ov::Node> extract_tied_lm_head_weight(const std::shared_ptr<ov::Model>& main_model,
                                                 bool& transpose_weight) {
    const auto logits_source = find_result_source(main_model, "logits");
    OPENVINO_ASSERT(logits_source.get_node(), "Failed to locate `logits` output in the main model for MTP lm_head graft.");

    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(logits_source.get_node_shared_ptr());
    OPENVINO_ASSERT(matmul, "Expected the main model `logits` output to be produced by a MatMul (lm_head).");

    transpose_weight = matmul->get_transpose_b();
    return matmul->input_value(1);
}

void expose_last_hidden_state(const std::shared_ptr<ov::Model>& main_model) {
    if (find_result_source(main_model, "last_hidden_state").get_node()) {
        return;
    }

    const auto logits_source = find_result_source(main_model, "logits");
    OPENVINO_ASSERT(logits_source.get_node(),
                    "Failed to locate `logits` output in the main model for MTP hidden-state graft.");

    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(logits_source.get_node_shared_ptr());
    OPENVINO_ASSERT(matmul, "Expected the main model `logits` output to be produced by a MatMul (lm_head).");

    // The lm_head MatMul consumes the model's last hidden state as its first input.
    auto hidden_state = find_mtp_hidden_states_producer(main_model);
    if (!hidden_state.get_node()) {
        hidden_state = matmul->input_value(0);
    }

    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden_state);
    hidden_result->output(0).set_names({"last_hidden_state"});
    hidden_result->set_friendly_name("last_hidden_state");
    // NPUW uses this info to identify manually added outputs.
    hidden_result->get_rt_info()["manually_added_output"] = true;

    main_model->add_results({hidden_result});
    main_model->validate_nodes_and_infer_types();
}

void graft_lm_head_on_mtp(std::shared_ptr<ov::Model>& mtp_model, const std::shared_ptr<ov::Model>& main_model) {
    const auto mtp_hidden_state = find_result_source(mtp_model, "last_hidden_state");
    OPENVINO_ASSERT(mtp_hidden_state.get_node(),
                    "Failed to locate `last_hidden_state` output in the MTP draft model for lm_head graft.");

    bool transpose_weight = false;
    const auto main_weight = extract_tied_lm_head_weight(main_model, transpose_weight);

    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = clone_subgraph(main_weight.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "Failed to clone the tied lm_head weight into the MTP draft model.");

    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(mtp_hidden_state,
                                                              cloned_weight->output(main_weight.get_index()),
                                                              false,
                                                              transpose_weight);
    logits_matmul->set_friendly_name("mtp_lm_head_matmul");

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    logits_result->set_friendly_name("logits");
    // NPUW uses this info to identify manually added outputs.
    logits_result->get_rt_info()["manually_added_output"] = true;

    mtp_model->add_results({logits_result});
    mtp_model->validate_nodes_and_infer_types();
}

void restrict_mtp_draft_vocab(const std::shared_ptr<ov::Model>& mtp_model, const std::vector<int64_t>& draft_vocab) {
    if (draft_vocab.empty()) {
        return;
    }
    const auto logits_source = find_result_source(mtp_model, "logits");
    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(logits_source.get_node_shared_ptr());
    OPENVINO_ASSERT(matmul,
                    "MTP draft vocabulary: expected the draft `logits` output to be produced by the lm_head MatMul");
    OPENVINO_ASSERT(matmul->get_transpose_b(),
                    "MTP draft vocabulary: expected an lm_head weight laid out as [vocab_size, hidden_size]");
    const auto& weight_shape = matmul->get_input_partial_shape(1);
    OPENVINO_ASSERT(weight_shape.rank().is_static() && weight_shape.size() == 2 && weight_shape[0].is_static(),
                    "MTP draft vocabulary: unexpected lm_head weight shape ",
                    weight_shape);
    const auto vocab_size = static_cast<int64_t>(weight_shape[0].get_length());
    OPENVINO_ASSERT(draft_vocab.back() < vocab_size,
                    "MTP draft vocabulary: token id ",
                    draft_vocab.back(),
                    " is out of the vocabulary range [0, ",
                    vocab_size,
                    ")");
    const auto subset_size = static_cast<int64_t>(draft_vocab.size());

    // Slice the vocabulary-major constants of the weight subgraph (compressed weight, scale, zero point) and the
    // reshape patterns that name the vocabulary size (grouped quantization).
    std::vector<std::shared_ptr<ov::Node>> stack{matmul->get_input_node_shared_ptr(1)};
    std::unordered_set<ov::Node*> visited;
    size_t num_sliced = 0;
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            const auto& shape = constant->get_shape();
            if (!shape.empty() && shape[0] == static_cast<size_t>(vocab_size)) {
                ov::replace_node(constant, slice_vocab_rows(constant, draft_vocab));
                ++num_sliced;
            }
            continue;
        }
        if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
            auto pattern = ov::as_type_ptr<ov::op::v0::Constant>(reshape->get_input_node_shared_ptr(1));
            OPENVINO_ASSERT(pattern, "MTP draft vocabulary: unsupported dynamic reshape in the lm_head weight");
            auto target_shape = pattern->cast_vector<int64_t>();
            if (!target_shape.empty() && target_shape[0] == vocab_size) {
                target_shape[0] = subset_size;
                reshape->input(1).replace_source_output(
                    ov::op::v0::Constant::create(pattern->get_element_type(), pattern->get_shape(), target_shape));
            }
            stack.push_back(reshape->get_input_node_shared_ptr(0));
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            stack.push_back(node->get_input_node_shared_ptr(i));
        }
    }
    OPENVINO_ASSERT(num_sliced > 0,
                    "MTP draft vocabulary: no constant with ",
                    vocab_size,
                    " rows found under the draft lm_head weight");
    mtp_model->validate_nodes_and_infer_types();  // propagate the subset size to the lm_head MatMul

    // logits = ScatterUpdate(Broadcast(-inf, [shape(subset_logits)[:-1], vocab_size]), draft_vocab, subset_logits, -1)
    const auto logits_consumers = matmul->output(0).get_target_inputs();
    const auto logits_type = matmul->get_output_element_type(0);
    const auto rank = static_cast<int64_t>(matmul->get_output_partial_shape(0).size());
    auto i64_scalar_vector = [](int64_t value) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {value});
    };
    auto subset_logits_shape = std::make_shared<ov::op::v3::ShapeOf>(matmul, ov::element::i64);
    auto leading_dims = std::make_shared<ov::op::v8::Slice>(subset_logits_shape,
                                                            i64_scalar_vector(0),
                                                            i64_scalar_vector(-1),
                                                            i64_scalar_vector(1),
                                                            i64_scalar_vector(0));
    auto full_logits_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{leading_dims, i64_scalar_vector(vocab_size)}, 0);
    auto masked = std::make_shared<ov::op::v3::Broadcast>(
        ov::op::v0::Constant::create(logits_type, ov::Shape{}, {-std::numeric_limits<float>::infinity()}),
        full_logits_shape);
    auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{draft_vocab.size()}, draft_vocab);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {rank - 1});
    auto full_logits = std::make_shared<ov::op::v3::ScatterUpdate>(masked, indices, matmul, axis);
    full_logits->set_friendly_name("mtp_lm_head_draft_vocab_scatter");
    for (auto& consumer : logits_consumers) {
        consumer.replace_source_output(full_logits);
    }
    mtp_model->validate_nodes_and_infer_types();
}

}  // namespace mtp
}  // namespace utils
}  // namespace genai
}  // namespace ov
