// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "paged_attention_transformations.hpp"
#include "cache_manager.hpp"

namespace ov::genai {
inline ov::PartialShape to_partial_with_dyn_0_dim(const ov::Shape& static_shape) {
    ov::PartialShape partial_shape = static_shape;
    partial_shape[0] = ov::Dimension::dynamic();
    return partial_shape;
}

/** Applies transformations to the ov::Model to enable paged attention inference.
 * @param model Pointer to the ov::Model representing one of the supported LLM architectures.
 * @param device_config Configuration struct for inferencing device specifics.
 * @param per_layer_cache_control If true, then the transformations will enable per-layer control of KV cache blocks, allowing to specify
 * different sets of KV cache blocks for different attention layers. If false, then the KV cache block structure will be identical across all
 * decoder layers.
 */
void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, DeviceConfig& device_config, bool per_layer_cache_control) {
    const ov::op::util::VariableVector& variables = model->get_variables();
    OPENVINO_ASSERT(!variables.empty(), "Model is supposed to be stateful");

    bool use_block_indices_inputs = per_layer_cache_control;
    bool use_score_outputs = per_layer_cache_control;
    ov::pass::SDPAToPagedAttention(use_block_indices_inputs, use_score_outputs).run_on_model(model);

    const ov::ParameterVector& parameters = model->get_parameters();

    std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>> key_cache_params;
    std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>> value_cache_params;
    for (const auto& param_ptr : parameters) {
        const auto& name = param_ptr->get_friendly_name();
        if (name.find("key_cache.") == 0) {
            key_cache_params[name] = param_ptr;
        }
        else if (name.find("value_cache.") == 0) {
            value_cache_params[name] = param_ptr;
        }
    }

    OPENVINO_ASSERT(key_cache_params.size() == value_cache_params.size());
    OPENVINO_ASSERT(key_cache_params.size() > 0);

    size_t num_layers = key_cache_params.size();
    // extract num_kv_heads and head_size
    std::string key_cache_param_name = "key_cache.0";
    OPENVINO_ASSERT(key_cache_params.count(key_cache_param_name) != 0, "key_cache.0 tensor not found among model parameters");
    ov::PartialShape k_shape = key_cache_params[key_cache_param_name]->get_partial_shape();
    OPENVINO_ASSERT(k_shape.rank().get_length() == 3, "KV cache shape is expected to have rank 3, while shape is ", k_shape);
    size_t num_kv_heads = k_shape[1].get_length(), head_size = k_shape[2].get_length();

    device_config.set_model_params(num_kv_heads, head_size, num_layers);

    for (auto it_k = key_cache_params.begin(), it_v = value_cache_params.begin(); it_k != key_cache_params.end();++it_k, ++it_v) {
        it_k->second->set_element_type(device_config.get_cache_precision());
        it_v->second->set_element_type(device_config.get_cache_precision());
        // TODO: CVS-145270
        it_k->second->set_partial_shape(to_partial_with_dyn_0_dim(device_config.get_key_cache_shape()));
        it_v->second->set_partial_shape(to_partial_with_dyn_0_dim(device_config.get_value_cache_shape()));
    }

    model->validate_nodes_and_infer_types();
}
}
