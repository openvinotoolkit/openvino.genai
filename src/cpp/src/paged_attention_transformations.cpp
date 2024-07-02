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

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, DeviceConfig& device_config) {
    const ov::op::util::VariableVector& variables = model->get_variables();
    OPENVINO_ASSERT(!variables.empty(), "Model is supposed to be stateful");

    ov::pass::SDPAToPagedAttention().run_on_model(model);

    const ov::ParameterVector& parameters = model->get_parameters();

    size_t num_layers = std::count_if(parameters.begin(), parameters.end(), [](std::shared_ptr<ov::op::v0::Parameter> parameter) {
        return parameter->get_friendly_name().find("key_cache.") == 0;
    });

    // extract num_kv_heads and head_size
    ov::PartialShape k_shape = parameters[KV_CACHES_INPUTS_OFFSET]->get_partial_shape();
    OPENVINO_ASSERT(k_shape.rank().get_length() == 3, "KV cache shape is expected to have rank 3, while shape is ", k_shape);
    size_t num_kv_heads = k_shape[1].get_length(), head_size = k_shape[2].get_length();

    device_config.set_model_params(num_kv_heads, head_size, num_layers);

    for (size_t decoder_layer_id = 0; decoder_layer_id < num_layers; ++decoder_layer_id) {
        auto key_cache_id = get_key_cache_tensor_id_for_decoder_layer(decoder_layer_id);
        auto value_cache_id = get_value_cache_tensor_id_for_decoder_layer(decoder_layer_id);
        parameters[key_cache_id]->set_element_type(device_config.get_cache_precision());
        parameters[value_cache_id]->set_element_type(device_config.get_cache_precision());
        // TODO: CVS-145270
        parameters[key_cache_id]->set_partial_shape(to_partial_with_dyn_0_dim(device_config.get_key_cache_shape()));
        parameters[value_cache_id]->set_partial_shape(to_partial_with_dyn_0_dim(device_config.get_value_cache_shape()));
    }
    model->validate_nodes_and_infer_types();
}
}
