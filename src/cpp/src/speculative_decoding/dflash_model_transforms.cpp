// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"

#include "eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

namespace {

constexpr const char* DFLASH_HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers";
constexpr const char* LAST_HIDDEN_STATE_OUTPUT_NAME = "last_hidden_state";

std::optional<ov::Output<ov::Node>> find_dflash_output_by_tensor_name(const std::shared_ptr<ov::Model>& model,
                                                                      const std::string& tensor_name) {
    for (const auto& node : model->get_ordered_ops()) {
        for (const auto& output : node->outputs()) {
            if (output.get_names().count(tensor_name) != 0) {
                return output;
            }
        }
    }
    return std::nullopt;
}

void add_dflash_hidden_state_result(std::shared_ptr<ov::Model>& model,
                                    const std::vector<ov::Output<ov::Node>>& hidden_state_outputs) {
    ov::Output<ov::Node> output_to_operate;
    if (hidden_state_outputs.size() > 1) {
        auto concat = std::make_shared<ov::op::v0::Concat>(hidden_state_outputs, -1);
        concat->set_friendly_name("dflash_hidden_states_concat");
        output_to_operate = concat->output(0);
    } else {
        output_to_operate = hidden_state_outputs[0];
    }

    auto result = std::make_shared<ov::op::v0::Result>(output_to_operate);
    result->output(0).set_names({LAST_HIDDEN_STATE_OUTPUT_NAME});
    result->set_friendly_name(LAST_HIDDEN_STATE_OUTPUT_NAME);
    result->get_rt_info()["manually_added_output"] = true;
    model->add_results({result});
}

std::vector<ov::Output<ov::Node>> get_dflash_annotated_hidden_state_outputs(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<int32_t>& target_layer_ids) {
    if (!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY)) {
        return {};
    }

    const auto annotation = nlohmann::json::parse(model->get_rt_info<std::string>(DFLASH_HIDDEN_STATES_RT_INFO_KEY));
    OPENVINO_ASSERT(annotation.contains("layers") && annotation["layers"].is_object(),
                    "Invalid hidden-state annotation metadata in model rt_info.");

    std::vector<ov::Output<ov::Node>> outputs;
    outputs.reserve(target_layer_ids.size());
    for (const auto layer_idx : target_layer_ids) {
        const auto key = std::to_string(layer_idx);
        OPENVINO_ASSERT(annotation["layers"].contains(key),
                        "Missing hidden-state annotation for decoder layer ",
                        layer_idx,
                        ".");
        const auto tensor_name = annotation["layers"].at(key).get<std::string>();
        auto output = find_dflash_output_by_tensor_name(model, tensor_name);
        OPENVINO_ASSERT(output.has_value(),
                        "Hidden-state tensor annotated for decoder layer ",
                        layer_idx,
                        " was not found in the model graph: ",
                        tensor_name);
        outputs.push_back(*output);
    }
    return outputs;
}

template <typename T>
std::optional<T> get_rt_info_value(const std::shared_ptr<ov::Model>& model, const std::vector<std::string>& path) {
    if (!model->has_rt_info(path)) {
        return std::nullopt;
    }
    return model->get_rt_info<T>(path);
}

std::vector<int32_t> parse_layer_ids(const std::string& raw) {
    std::vector<int32_t> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            result.push_back(static_cast<int32_t>(std::stoi(item)));
        }
    }
    return result;
}

std::shared_ptr<ov::op::v0::Parameter> find_hidden_states_parameter(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "hidden_states" ||
            parameter->output(0).get_names().count("hidden_states") != 0) {
            return parameter;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Parameter> find_inputs_embeds_parameter(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "inputs_embeds" ||
            parameter->output(0).get_names().count("inputs_embeds") != 0) {
            return parameter;
        }
    }
    return nullptr;
}

}  // namespace

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (!model->has_rt_info("dflash_mode") || !model->get_rt_info<bool>("dflash_mode")) {
        return;
    }

    properties["dflash_mode"] = true;
    if (auto mask_token_id = get_rt_info_value<std::string>(model, {"dflash", "mask_token_id"})) {
        properties["dflash_mask_token_id"] = static_cast<int64_t>(std::stoll(*mask_token_id));
    }
    if (auto target_layer_ids = get_rt_info_value<std::string>(model, {"dflash", "target_layer_ids"})) {
        properties["dflash_target_layer_ids"] = parse_layer_ids(*target_layer_ids);
    }
}

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config) {
    DFlashRTInfo info;
    auto mode_it = config.find("dflash_mode");
    if (mode_it == config.end()) {
        return info;
    }

    info.dflash_mode = mode_it->second.as<bool>();
    config.erase(mode_it);
    if (!info.dflash_mode) {
        return info;
    }

    auto mask_it = config.find("dflash_mask_token_id");
    OPENVINO_ASSERT(mask_it != config.end(), "DFlash draft model is missing dflash_mask_token_id RT info.");
    info.mask_token_id = mask_it->second.as<int64_t>();
    config.erase(mask_it);

    auto layers_it = config.find("dflash_target_layer_ids");
    OPENVINO_ASSERT(layers_it != config.end(), "DFlash draft model is missing dflash_target_layer_ids RT info.");
    info.target_layer_ids = layers_it->second.as<std::vector<int32_t>>();
    config.erase(layers_it);
    OPENVINO_ASSERT(!info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");

    return info;
}

void expose_target_hidden_states(std::shared_ptr<ov::Model>& model, const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    auto annotated_outputs = get_dflash_annotated_hidden_state_outputs(model, target_layer_ids);
    OPENVINO_ASSERT(!annotated_outputs.empty(),
                    "DFlash requires hidden-state annotations in the target model. "
                    "Export the target model with Optimum hidden-state annotations.");
    add_dflash_hidden_state_result(model, annotated_outputs);
}

void reshape_draft_hidden_states_input_for_cb(std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash draft model cannot be null.");

    auto hidden_states = find_hidden_states_parameter(model);
    OPENVINO_ASSERT(hidden_states, "DFlash draft model must have 'hidden_states' input.");

    const auto draft_shape = hidden_states->get_partial_shape();
    OPENVINO_ASSERT(draft_shape.rank().is_static() && draft_shape.rank().get_length() == 3,
                    "DFlash draft hidden_states input must have rank 3.");
    OPENVINO_ASSERT(draft_shape[0].is_dynamic() || draft_shape[0].get_length() == 1,
                    "DFlash draft hidden_states input must use exported shape [1, seq_len, hidden].");

    std::unordered_set<const ov::Node*> live_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        live_nodes.insert(node.get());
    }
    std::vector<ov::Input<ov::Node>> original_consumers;
    for (auto consumer : hidden_states->output(0).get_target_inputs()) {
        if (live_nodes.count(consumer.get_node()) != 0) {
            original_consumers.push_back(consumer);
        }
    }
    OPENVINO_ASSERT(!original_consumers.empty(),
                    "DFlash draft hidden_states input has no live consumers.");
    hidden_states->set_partial_shape(ov::PartialShape({draft_shape[1], ov::Dimension(1), draft_shape[2]}));

    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                     ov::Shape{3},
                                                     std::vector<int64_t>{1, -1, 0});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, reshape_shape, true);
    reshape->set_friendly_name("dflash_hidden_states_cb_to_draft_layout");

    for (auto consumer : original_consumers) {
        consumer.replace_source_output(reshape->output(0));
    }
    model->validate_nodes_and_infer_types();
}

void attach_target_lm_head_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                    const std::shared_ptr<ov::Model>& draft_model) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash lm_head graft requires both target and draft models.");

    auto target_head = std::get<0>(ov::genai::utils::find_llm_matmul(main_model));
    auto target_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(target_head);
    OPENVINO_ASSERT(target_matmul,
                    "DFlash: could not locate the target lm_head MatMul to graft onto the draft.");

    // Clone ONLY the weight side (input(1)); input(0) is the target activation and is not reused.
    // Cloning recursively carries any INT4 decompression subgraph intact.
    auto weight_source = target_matmul->input_value(1);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target lm_head weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    std::shared_ptr<ov::op::v0::Result> hidden_result;
    for (const auto& result : draft_model->get_results()) {
        if (result->output(0).get_names().count(LAST_HIDDEN_STATE_OUTPUT_NAME) != 0 ||
            result->get_friendly_name() == LAST_HIDDEN_STATE_OUTPUT_NAME) {
            hidden_result = result;
            break;
        }
    }
    OPENVINO_ASSERT(hidden_result,
                    "DFlash draft model must expose a 'last_hidden_state' output to graft the target lm_head.");

    // Feed the draft post-norm hidden into a bare MatMul with the target head weight + transpose flags.
    auto draft_hidden = hidden_result->input_value(0);
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(draft_hidden,
                                                              cloned_weight_out,
                                                              target_matmul->get_transpose_a(),
                                                              target_matmul->get_transpose_b());
    logits_matmul->set_friendly_name("dflash_grafted_lm_head");

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    logits_result->set_friendly_name("logits");
    logits_result->get_rt_info()["manually_added_output"] = true;

    draft_model->add_results({logits_result});
    draft_model->remove_result(hidden_result);
    draft_model->validate_nodes_and_infer_types();
}

void attach_target_embedding_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                      const std::shared_ptr<ov::Model>& draft_model) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash embedding attach requires both target and draft models.");

    auto gather = eagle3::find_embedding_gather(main_model);
    OPENVINO_ASSERT(gather,
                    "DFlash: could not find the target token-embedding Gather to attach onto the draft.");

    // Clone the embedding weight (input(0)). The recursive clone is precision-agnostic: it copies a
    // plain f16/f32 Constant or an int8/int4 dequant chain intact, so the draft owns its embedding.
    auto weight_source = gather->input_value(0);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target embedding weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    auto inputs_embeds_param = find_inputs_embeds_parameter(draft_model);
    OPENVINO_ASSERT(inputs_embeds_param,
                    "DFlash draft model must expose an 'inputs_embeds' input to attach the target embedding.");

    // Snapshot the live consumers of inputs_embeds before rewiring them onto the new Gather.
    std::unordered_set<const ov::Node*> live_nodes;
    for (const auto& node : draft_model->get_ordered_ops()) {
        live_nodes.insert(node.get());
    }
    std::vector<ov::Input<ov::Node>> original_consumers;
    for (auto consumer : inputs_embeds_param->output(0).get_target_inputs()) {
        if (live_nodes.count(consumer.get_node()) != 0) {
            original_consumers.push_back(consumer);
        }
    }
    OPENVINO_ASSERT(!original_consumers.empty(), "DFlash draft inputs_embeds input has no live consumers.");

    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto embed_gather = std::make_shared<ov::op::v8::Gather>(cloned_weight_out, input_ids, axis);
    embed_gather->set_friendly_name("dflash_draft_embedding");

    for (auto consumer : original_consumers) {
        consumer.replace_source_output(embed_gather->output(0));
    }
    draft_model->add_parameters({input_ids});
    draft_model->remove_parameter(inputs_embeds_param);
    draft_model->validate_nodes_and_infer_types();
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
