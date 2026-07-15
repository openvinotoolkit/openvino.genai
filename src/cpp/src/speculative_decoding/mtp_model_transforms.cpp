// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mtp_model_transforms.hpp"

#include <functional>
#include <unordered_map>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace mtp {

namespace {

ov::Output<ov::Node> find_result_source(const std::shared_ptr<ov::Model>& model, const std::string& tensor_name) {
    for (const auto& result : model->get_results()) {
        const auto source = result->input_value(0);
        const auto& names = source.get_names();
        if (names.find(tensor_name) != names.end()) {
            return source;
        }
        if (result->get_friendly_name() == tensor_name) {
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

}  // namespace

MtpRTInfo extract_mtp_info_from_config(ov::AnyMap& config) {
    MtpRTInfo mtp_rt_info;
    const auto it = config.find("mtp_mode");
    if (it != config.end()) {
        mtp_rt_info.mtp_mode = it->second.as<bool>();
        config.erase(it);
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

void remove_roundtrip_converts(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        auto outer = ov::as_type_ptr<ov::op::v0::Convert>(node);
        if (!outer) {
            continue;
        }
        auto inner = ov::as_type_ptr<ov::op::v0::Convert>(outer->get_input_node_shared_ptr(0));
        if (!inner) {
            continue;
        }
        const auto source = inner->input_value(0);
        if (source.get_element_type() != outer->get_output_element_type(0)) {
            continue;
        }
        for (auto& target : outer->output(0).get_target_inputs()) {
            target.replace_source_output(source);
        }
    }
    model->validate_nodes_and_infer_types();
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
    auto hidden_state = matmul->input_value(0);

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

}  // namespace mtp
}  // namespace utils
}  // namespace genai
}  // namespace ov
