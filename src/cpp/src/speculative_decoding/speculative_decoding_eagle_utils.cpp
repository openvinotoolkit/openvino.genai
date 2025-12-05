// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_eagle_utils.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "logger.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/concat.hpp"

namespace ov {
namespace genai {
namespace speculative_decoding {

void ensure_num_assistant_tokens_is_set(ov::genai::GenerationConfig& config) {
    // Only num_assistant_tokens is supported, not assistant_confidence_threshold
    OPENVINO_ASSERT(
        config.assistant_confidence_threshold == 0.f,
        "Speculative Decoding only supports num_assistant_tokens, not assistant_confidence_threshold. Set it to 0.f.");
    if (config.num_assistant_tokens == 0) {
        config.num_assistant_tokens = DEFAULT_NUM_ASSISTANT_TOKENS;
    }
}

Eagle3RTInfo
extract_eagle_mode_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    Eagle3RTInfo eagle_rt_info;
    if (config.find("eagle3_mode") != config.end()) {
        eagle_rt_info.eagle3_mode = config.at("eagle3_mode").as<bool>();
        config.erase("eagle3_mode");
        if (config.find("hidden_layers_list") != config.end()) {
            eagle_rt_info.hidden_layers_list = config.at("hidden_layers_list").as<std::vector<int>>();
            config.erase("hidden_layers_list");
        } else {
            // compute the layers from number of hidden layers
            auto config_file_path = models_path / "config.json";
            if (!std::filesystem::exists(config_file_path))
                OPENVINO_THROW("cannot deduce layers for hidden layer extraction");
            std::ifstream file(config_file_path);

            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;
            int num_decoder_layers = 0;
            read_json_param(data, "num_hidden_layers", num_decoder_layers);
            OPENVINO_ASSERT(num_decoder_layers > 3, "num_decoder_layers is too small to deduce hidden layers for extraction");
            // The following default hidden layer selection corresponds to the EAGLE reference implementation:
            // https://github.com/SafeAILab/EAGLE/blob/0ea94696/eagle/model/modeling_llama_kv.py#L1138
            // These layers (2, num_decoder_layers / 2, num_decoder_layers - 3) are chosen to capture features from
            // early, middle, and late stages of the decoder, as recommended by the EAGLE authors.
            // If you wish to use different layers, provide the "hidden_layers_list" parameter in the config.
            eagle_rt_info.hidden_layers_list = { 2, num_decoder_layers / 2, num_decoder_layers - 3 };
        }
    }
    return eagle_rt_info;
}

void share_embedding_weights(std::shared_ptr<ov::Model>& main_model, std::shared_ptr<ov::Model>& draft_model) {
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

    try {
        draft_weight_node->output(0).replace(main_weight_node->output(0));
    } catch (const std::exception& e) {
        GENAI_WARN(std::string("Error: failed to import embedding weights from main model to draft model. Exception: ") + e.what());
    } catch (...) {
        GENAI_WARN("Error: failed to import embedding weights from main model to draft model due to unknown exception.");
    }
}

void shift_fc_from_draft_to_main(std::shared_ptr<ov::Model>& main_model, std::shared_ptr<ov::Model>& draft_model) {
    // extract the FC transform weight from draft model
    auto remove_fc_and_rewire = [](const std::shared_ptr<ov::Model>& model) -> std::shared_ptr<ov::Node> {
        for (const auto& node : model->get_ordered_ops()) {
            auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(node);
            if (!matmul_node) continue;
            auto input_node = matmul_node->get_input_node_shared_ptr(0);
            auto param_node = ov::as_type_ptr<ov::op::v0::Parameter>(input_node);
            if (!param_node) continue;
            if (input_node->get_friendly_name().find("hidden_states") == std::string::npos) continue;
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
    if (!fc_weights) {
        GENAI_WARN("Error: failed to retrieve and remove FC matmul from draft model.");
        return;
    }
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

std::shared_ptr<ov::op::v0::Constant> extract_d2t_mapping_table(std::shared_ptr<ov::Model>& model) {
    // extract result nodes from model
    for (const auto& result : model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input_node);
        if (constant && constant->get_friendly_name().find("d2t") != std::string::npos) {
            return constant;
        }
    }
    return nullptr;
}

void remove_d2t_result_node(std::shared_ptr<ov::Model>& model) {
    // Find and remove the d2t Result node
    std::shared_ptr<ov::op::v0::Result> d2t_result_to_remove = nullptr;

    for (const auto& result : model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input_node);
        if (constant && constant->get_friendly_name().find("d2t") != std::string::npos) {
            d2t_result_to_remove = result;
            break;
        }
    }

    if (d2t_result_to_remove) {
        model->remove_result(d2t_result_to_remove);
        model->validate_nodes_and_infer_types();
    }
}

void hidden_state_transform(std::shared_ptr<ov::Model>& model,
                            const std::vector<int>& hidden_layers_to_abstract) {
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
        for (int idx : hidden_layers_to_abstract) {
            patterns.emplace_back("layers." + std::to_string(idx) + "/"); // main description
        }
    } else {
        patterns.emplace_back("midlayer"); // draft description
    }

    // Helper: check if node is a residual Add node with expected structure
    auto is_residual_node = [](const std::shared_ptr<ov::Node>& node) -> bool {
        if (auto add = ov::as_type_ptr<ov::op::v1::Add>(node)) {
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

        result->get_rt_info()["manually_added_output"] = true;

        model->add_results({result});
    }
}

}  // namespace speculative_decoding
}  // namespace genai
}  // namespace ov
