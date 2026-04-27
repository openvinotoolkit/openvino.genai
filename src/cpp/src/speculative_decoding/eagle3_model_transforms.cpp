// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_model_transforms.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "logger.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace eagle3 {

Eagle3RTInfo extract_eagle3_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    Eagle3RTInfo eagle_rt_info;
    if (config.find("eagle3_mode") != config.end()) {
        eagle_rt_info.eagle3_mode = config.at("eagle3_mode").as<bool>();
        config.erase("eagle3_mode");
        auto it = config.find("hidden_layers_list");
        if (it != config.end()) {
            OPENVINO_ASSERT(it->second.is<std::vector<int32_t>>(),
                            "hidden_layers_list must be a vector of int32_t values");
            eagle_rt_info.hidden_layers_list = it->second.as<std::vector<int32_t>>();
            config.erase("hidden_layers_list");
        } else {
            // compute the layers from number of hidden layers
            auto config_file_path = models_path / "config.json";
            OPENVINO_ASSERT(std::filesystem::exists(config_file_path), "Cannot deduce layers for hidden layer extraction because the file is missing: ", config_file_path);
            std::ifstream file(config_file_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;
            int num_decoder_layers = 0;
            read_json_param(data, "num_hidden_layers", num_decoder_layers);

            // Ensure sufficient layers for meaningful feature extraction
            // Minimum of 10 layers is based on practical LLM architectures (e.g., small GPT-2 has 12 layers)
            OPENVINO_ASSERT(
                num_decoder_layers >= 10,
                "num_decoder_layers must be at least 10 for automatic hidden layer selection, got: ",
                num_decoder_layers,
                ". For models with fewer layers, please explicitly specify 'hidden_layers_list' in the configuration.");

            // The following default hidden layer selection corresponds to the EAGLE reference implementation:
            // https://github.com/SafeAILab/EAGLE/blob/0ea94696/eagle/model/modeling_llama_kv.py#L1138
            // These layers (2, num_decoder_layers / 2, num_decoder_layers - 3) are chosen to capture features from
            // early, middle, and late stages of the decoder, as recommended by the EAGLE authors.
            // Note: Integer division (num_decoder_layers / 2) is intentional and produces the desired behavior
            // for typical LLM layer counts (e.g., 12→6, 24→12, 32→16).
            // If you wish to use different layers, provide the "hidden_layers_list" parameter in the config.
            eagle_rt_info.hidden_layers_list = { 2, num_decoder_layers / 2, num_decoder_layers - 3 };
        }
        OPENVINO_ASSERT(eagle_rt_info.hidden_layers_list.size() == 3, "Eagle3 is expected to provide exactly three layers for extraction");
    }
    return eagle_rt_info;
}

void apply_eagle3_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (model->has_rt_info("eagle3_mode") && model->get_rt_info<bool>("eagle3_mode")) {
        properties["eagle3_mode"] = true;
        if (model->has_rt_info("hidden_layers_list")) {
            properties["hidden_layers_list"] = model->get_rt_info<std::vector<int>>("hidden_layers_list");
        }
    }
}

void share_vocabulary(const std::shared_ptr<ov::Model>& main_model, const std::shared_ptr<ov::Model>& draft_model) {
    // extract embedding weight from main model
    auto find_embedding_gather = [](const std::shared_ptr<ov::Model>& model)
        -> std::shared_ptr<ov::Node> {
        constexpr size_t MIN_VOCAB_SIZE_THRESHOLD = 1000;
        for (const auto& node : model->get_ordered_ops()) {
            auto gather = std::dynamic_pointer_cast<ov::op::util::GatherBase>(node);
            if (!gather) continue;
            // [vocab, hidden_size] * [batch, seq_len] -> [batch, seq_len, hidden_size]
            auto data_node = gather->input_value(0).get_node_shared_ptr();
            auto indices_node = gather->input_value(1).get_node_shared_ptr();
            if (!data_node || !indices_node) continue;
            // indices_node should be on parameter path, maybe this is better rule
            ov::PartialShape ps = data_node->get_output_partial_shape(0);
            if (ps.rank().is_static() && ps.rank().get_length() >= 2) {
                if (ps[0].is_static() && ps[0].get_length() > MIN_VOCAB_SIZE_THRESHOLD) { // Heuristic: vocab size > 1000
                    return gather;
                }
            }
            std::string fname = data_node->get_friendly_name();
            if (fname.find("embed_tokens") != std::string::npos ||
                fname.find("embedding") != std::string::npos) {
                return gather;
            }
        }
        return nullptr;
    };
    auto main_gather  = find_embedding_gather(main_model);
    auto draft_gather = find_embedding_gather(draft_model);
    if (!main_gather || !draft_gather) {
        return;
    }
    auto main_weight_node = main_gather->input_value(0).get_node_shared_ptr();
    auto draft_weight_node = draft_gather->input_value(0).get_node_shared_ptr();

    if (main_weight_node.get() == draft_weight_node.get()) {
        return;
    }

    GENAI_INFO("Copying embedding weights from main to draft model for eagle3 speculative decoding.");

    // Helper function to recursively clone a node and its inputs
    // This handles cases where embedding has intermediate ops (Convert, FakeQuantize, etc.)
    std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&,
                                            std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>&)>
        clone_node_recursive =
            [&](const std::shared_ptr<ov::Node>& node,
                std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>& cloned_nodes) -> std::shared_ptr<ov::Node> {

        auto it = cloned_nodes.find(node.get());
        if (it != cloned_nodes.end()) {
            return it->second;
        }

        std::shared_ptr<ov::Node> cloned;

        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            // For Constant nodes, create a deep copy with new data
            cloned = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(),
                                                            constant->get_shape(),
                                                            constant->get_data_ptr());
        } else {
            // For other nodes, clone recursively with cloned inputs
            ov::OutputVector cloned_inputs;
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                auto input_node = node->get_input_node_shared_ptr(i);
                auto cloned_input = clone_node_recursive(input_node, cloned_nodes);
                cloned_inputs.push_back(cloned_input->output(node->get_input_source_output(i).get_index()));
            }
            cloned = node->clone_with_new_inputs(cloned_inputs);
        }

        cloned->set_friendly_name(node->get_friendly_name() + "_cloned_for_draft");
        cloned_nodes[node.get()] = cloned;
        return cloned;
    };

    // Clone the entire subgraph from main model
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight_node = clone_node_recursive(main_weight_node, cloned_nodes);

    OPENVINO_ASSERT(cloned_weight_node,
                    "Failed to clone embedding weight node from main model to draft model. "
                    "This is required for Eagle3 speculative decoding.");

    // Replace draft model's weight node with the cloned subgraph
    // This avoids cross-model references by duplicating the vocabulary weights
    draft_weight_node->output(0).replace(cloned_weight_node->output(0));
}

void move_fc_from_draft_to_main(std::shared_ptr<ov::Model>& draft_model, std::shared_ptr<ov::Model>& main_model) {
    // extract the FC transform weight from draft model
    auto remove_fc_and_rewire = [](const std::shared_ptr<ov::Model>& model) -> std::shared_ptr<ov::Node> {
        for (const auto& node : model->get_ordered_ops()) {
            auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(node);
            if (!matmul_node) continue;
            auto input_node = matmul_node->get_input_node_shared_ptr(0);
            auto param_node = ov::as_type_ptr<ov::op::v0::Parameter>(input_node);
            if (!param_node || input_node->get_friendly_name().find("hidden_states") == std::string::npos) continue;
            // Rewire all outputs of this MatMul to use the input_node directly
            for (auto& output : matmul_node->outputs()) {
                for (auto& target : output.get_target_inputs()) {
                    target.replace_source_output(input_node);
                }
            }
            return matmul_node->input_value(1).get_node_shared_ptr();
        }
        return nullptr;
    };
    auto fc_weights = remove_fc_and_rewire(draft_model);
    if (!fc_weights)
        OPENVINO_THROW("Failed to locate FC weights in eagle3 draft model for shifting to main model.");
    // now we create the fc into main model
    for (const auto& result : main_model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        if (input_node && input_node->get_friendly_name().find("eagle3_hidden_states_concat") != std::string::npos) {
            auto matmul = std::make_shared<ov::op::v0::MatMul>(input_node, fc_weights, false, true);
            matmul->set_friendly_name("eagle3_hidden_state_fc");
            result->input(0).replace_source_output(matmul);
            break;
        }
    }
}

// Helper function to find d2t result node in the model
static std::shared_ptr<ov::op::v0::Result> find_d2t_result_node(const std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input_node);
        if (constant && constant->get_friendly_name().find("d2t") != std::string::npos) {
            return result;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Constant> extract_d2t_mapping_table(const std::shared_ptr<ov::Model>& model) {
    // extract result nodes from model
    auto d2t_result = find_d2t_result_node(model);
    if (d2t_result) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(d2t_result->input_value(0).get_node_shared_ptr());
        model->remove_result(d2t_result);
        model->validate_nodes_and_infer_types();
        return constant;
    }
    return nullptr;
}

void transform_hidden_state(std::shared_ptr<ov::Model>& model, const std::vector<int32_t>& hidden_layers_to_abstract) {
    if (hidden_layers_to_abstract.empty()) {
        return;
    }
    OPENVINO_ASSERT(
        hidden_layers_to_abstract.size() == 3 || hidden_layers_to_abstract.size() == 1,
        "Expected exactly 1 or 3 hidden layers for extraction: 1 for draft model, 3 for main model (early/middle/late stages)."
    );

    std::vector<std::string> patterns;
    if (hidden_layers_to_abstract.size() > 1) {
        patterns.reserve(hidden_layers_to_abstract.size());
        for (int32_t idx : hidden_layers_to_abstract) {
            patterns.emplace_back("layers." + std::to_string(idx) + "/"); // main description
        }
    } else {
        patterns.emplace_back("midlayer"); // draft description
    }

    // Helper: check if node is a residual Add node with expected structure
    auto is_residual_node = [](const std::shared_ptr<ov::Node>& node) -> bool {
        if (const auto& add = ov::as_type_ptr<ov::op::v1::Add>(node)) {
            auto input1 = add->get_input_node_shared_ptr(1);
            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(input1);
            if (!matmul) return false;
            auto matmul_input = matmul->get_input_node_shared_ptr(0);
            return matmul_input && ov::is_type<ov::op::v1::Multiply>(matmul_input);
        }
        return false;
    };

    std::vector<ov::Output<ov::Node>> residual_outputs;
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_residual_node(node)) continue;
        const std::string& name = node->get_friendly_name();
        for (const auto& pattern : patterns) {
            if (name.find(pattern) != std::string::npos) {
                residual_outputs.push_back(node->output(0));
                break;
            }
        }
    }

    if (!residual_outputs.empty()) {
        OPENVINO_ASSERT(residual_outputs.size() == patterns.size(),
                        "Number of extracted hidden states does not match the requested number.");
        std::shared_ptr<ov::Node> node_to_operate;
        if (residual_outputs.size() > 1) {
            auto concat = std::make_shared<ov::op::v0::Concat>(residual_outputs, -1);
            concat->set_friendly_name("eagle3_hidden_states_concat");
            node_to_operate = concat;
        } else {
            node_to_operate = residual_outputs[0].get_node_shared_ptr();
        }
        auto result = std::make_shared<ov::op::v0::Result>(node_to_operate);
        const std::string output_name = "last_hidden_state";
        result->output(0).set_names({output_name});
        result->set_friendly_name(output_name);
        // NPUW use this info to identify manually added outputs
        result->get_rt_info()["manually_added_output"] = true;
        model->add_results({result});
    }
}

ov::Tensor slice_hidden_state_for_last_token(const ov::Tensor& hidden_features) {
    OPENVINO_ASSERT(hidden_features.get_size() > 0, "Hidden features tensor is empty");

    const auto shape = hidden_features.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 0, "Expected shape [1, seq_len, hidden_size]");

    const size_t seq_len = shape[1];

    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, seq_len - 1, seq_len);
    return ov::Tensor(hidden_features, start_coord, end_coord);
}

std::shared_ptr<ov::Model> create_eagle3_kv_update_model(const std::shared_ptr<ov::Model>& main_model) {
    // the kv update model acceptes all kv cache inputs from main_model
    // extra inputs for updating kv cache: block_indices, block_indices_begins, block_update_indices, block_update_indices_begins， all with element::i32, PartialShape{-1}
    using namespace ov;
    ParameterVector inputs;
    // clone the kv cache parameters from the main model
    auto params = main_model->get_parameters();
    std::vector<Output<Node>> key_caches;
    std::vector<Output<Node>> value_caches;
    for (const auto& param : params) {
        const std::string& name = param->get_friendly_name();
        // Find paged_attention op connected to this param
        std::shared_ptr<ov::Node> paged_attention_op = nullptr;
        for (const auto& node : main_model->get_ordered_ops()) {
            // Typical paged_attention op is custom, so check op type and input
            if (node->get_friendly_name().find("PagedAttentionExtension") != std::string::npos) {
                for (size_t idx = 0; idx < node->get_input_size(); ++idx) {
                    if (node->get_input_node_shared_ptr(idx).get() == param.get()) {
                        paged_attention_op = node;
                        break;
                    }
                }
                if (paged_attention_op) break;
            }
        }
        if (name.find("key_cache") != std::string::npos) {
            auto cloned_param = std::make_shared<ov::op::v0::Parameter>(param->get_element_type(), param->get_partial_shape());
            cloned_param->set_friendly_name(name);
            cloned_param->output(0).set_names({name});
            // Clone runtime info from paged_attention op if found
            if (paged_attention_op) {
                for (const auto& [key, value] : paged_attention_op->get_rt_info()) {
                    cloned_param->get_rt_info()[key] = value;
                }
            }
            inputs.push_back(cloned_param);
            key_caches.push_back(cloned_param);
        } else if (name.find("value_cache") != std::string::npos) {
            auto cloned_param = std::make_shared<ov::op::v0::Parameter>(param->get_element_type(), param->get_partial_shape());
            cloned_param->set_friendly_name(name);
            cloned_param->output(0).set_names({name});
            // Clone runtime info from paged_attention op if found
            if (paged_attention_op) {
                for (const auto& [key, value] : paged_attention_op->get_rt_info()) {
                    cloned_param->get_rt_info()[key] = value;
                }
            }
            inputs.push_back(cloned_param);
            value_caches.push_back(cloned_param);
        }
    }

    auto block_indices_begins = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_indices_begins->set_friendly_name("block_indices_begins");
    block_indices_begins->output(0).set_names({"block_indices_begins"});
    inputs.push_back(block_indices_begins);

    auto block_indices = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_indices->set_friendly_name("block_indices");
    block_indices->output(0).set_names({"block_indices"});
    inputs.push_back(block_indices);

    auto block_update_indices = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_update_indices->set_friendly_name("block_update_indices");
    block_update_indices->output(0).set_names({"block_update_indices"});
    inputs.push_back(block_update_indices);

    auto block_update_indices_begins = std::make_shared<op::v0::Parameter>(
        element::i32, PartialShape{-1});
    block_update_indices_begins->set_friendly_name("block_update_indices_begins");
    block_update_indices_begins->output(0).set_names({"block_update_indices_begins"});
    inputs.push_back(block_update_indices_begins);

    ResultVector results;
    size_t pair_count = std::min(key_caches.size(), value_caches.size());
    for (size_t i = 0; i < pair_count; ++i) {
        auto key_gather = std::make_shared<op::v8::Gather>(
            key_caches[i], block_update_indices, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, 0));
        key_gather->set_friendly_name("reordered_key_cache_" + std::to_string(i));
        auto key_scatter = std::make_shared<op::v3::ScatterUpdate>(
            key_caches[i], block_indices, key_gather, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, 0));
        key_scatter->set_friendly_name("updated_key_cache_" + std::to_string(i));

        auto value_gather = std::make_shared<op::v8::Gather>(
            value_caches[i], block_update_indices, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, 0));
        value_gather->set_friendly_name("reordered_value_cache_" + std::to_string(i));
        auto value_scatter = std::make_shared<op::v3::ScatterUpdate>(
            value_caches[i], block_indices, value_gather, std::make_shared<op::v0::Constant>(element::i32, ov::Shape{1}, 0));
        value_scatter->set_friendly_name("updated_value_cache_" + std::to_string(i));

        // Concat key and value scatter outputs along last axis
        auto concat = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{key_scatter->output(0), value_scatter->output(0)}, -1);
        concat->set_friendly_name("kv_cache_pair_concat_" + std::to_string(i));
        results.push_back(std::make_shared<op::v0::Result>(concat));
    }

    auto model = std::make_shared<Model>(results, inputs, "kv_cache_reorder_model");
    // addition runtime info for identification
    // in GPU, we need to sync kv precision with main model, which already been assigned default values based on PA ops
    model->get_rt_info()["auxiliary_kv_update_model"] = true;
    return model;
}
}  // namespace eagle3
}  // namespace utils
}  // namespace genai
}  // namespace ov
