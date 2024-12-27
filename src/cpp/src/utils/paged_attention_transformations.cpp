// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils/paged_attention_transformations.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

namespace ov {
namespace genai {
namespace utils {


size_t get_hidden_size(const std::shared_ptr<ov::Model> model) {
    const auto& parameters = model->get_parameters();
    // extract num_kv_heads and head_size
    size_t kv_caches_inputs_offset = 2;
    ov::PartialShape k_shape = parameters[kv_caches_inputs_offset]->get_partial_shape();
    OPENVINO_ASSERT(k_shape.rank().get_length() == 3, "KV cache shape is expected to have rank 3, while shape is ", k_shape);
    size_t num_kv_heads = k_shape[1].get_length(), head_size = k_shape[2].get_length();
    return num_kv_heads * head_size;
}

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, bool per_layer_cache_control) {
    const ov::op::util::VariableVector& variables = model->get_variables();
    OPENVINO_ASSERT(!variables.empty(), "Model is supposed to be stateful");

    bool use_block_indices_inputs = per_layer_cache_control;
    bool use_score_outputs = per_layer_cache_control;
    ov::pass::SDPAToPagedAttention(use_block_indices_inputs, use_score_outputs).run_on_model(model);
}

void set_kv_cache_type_and_shape(std::shared_ptr<ov::Model> model, DeviceConfig& device_config) {
    const ov::ParameterVector& parameters = model->get_parameters();

    std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>> key_cache_params, value_cache_params;
    for (const auto& param_ptr : parameters) {
        const auto& name = param_ptr->get_friendly_name();
        if (name.find("key_cache.") == 0) {
            key_cache_params[name] = param_ptr;
        }
        else if (name.find("value_cache.") == 0) {
            value_cache_params[name] = param_ptr;
        }
    }

    OPENVINO_ASSERT(key_cache_params.size() > 0);
    OPENVINO_ASSERT(key_cache_params.size() == value_cache_params.size());

    size_t num_layers = key_cache_params.size();
    // extract num_kv_heads and head_size
    std::string key_cache_param_name = "key_cache.0";
    OPENVINO_ASSERT(key_cache_params.count(key_cache_param_name) != 0, "key_cache.0 tensor not found among model parameters");
    ov::PartialShape k_shape = key_cache_params[key_cache_param_name]->get_partial_shape();
    OPENVINO_ASSERT(k_shape.rank().get_length() == 3, "KV cache shape is expected to have rank 3, while shape is ", k_shape);
    size_t head_size = k_shape[2].get_length();
    std::vector<size_t> num_kv_heads(num_layers);
    for (size_t idx = 0; idx < num_layers; idx++) {
        size_t num_heads = key_cache_params[std::string("key_cache.") + std::to_string(idx)]->get_partial_shape()[1].get_length();
        num_kv_heads[idx] = num_heads;
    }
    device_config.set_model_params(num_kv_heads, head_size, num_layers);

    for (size_t idx = 0; idx < num_layers; idx++) {
        auto k = key_cache_params[std::string("key_cache.") + std::to_string(idx)];
        auto v = value_cache_params[std::string("value_cache.") + std::to_string(idx)];
        k->set_element_type(device_config.get_cache_precision());
        v->set_element_type(device_config.get_cache_precision());
        k->set_partial_shape(device_config.get_key_cache_shape(idx));
        v->set_partial_shape(device_config.get_value_cache_shape(idx));
    }

    model->validate_nodes_and_infer_types();
}

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, DeviceConfig& device_config, bool per_layer_cache_control) {
    apply_paged_attention_transformations(model, per_layer_cache_control);
    set_kv_cache_type_and_shape(model, device_config);
}

}  // namespace utils
}  // namespace genai
}  // namespace ov