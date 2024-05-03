// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/core/model.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "model_config.hpp"
#include "device_config.hpp"

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model, const ModelConfig& model_config, const DeviceConfig& device_config) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SDPAToPagedAttention>();
    manager.run_passes(model);

    const ov::ParameterVector& parameters = model->get_parameters();
    for (size_t decoder_layer_id = 0; decoder_layer_id < model_config.get_num_layers(); ++decoder_layer_id) {
        parameters[2 + 2 * decoder_layer_id]->set_element_type(device_config.get_cache_precision());
        parameters[2 + 2 * decoder_layer_id + 1]->set_element_type(device_config.get_cache_precision());
        parameters[2 + 2 * decoder_layer_id]->set_partial_shape(device_config.get_key_cache_shape());
        parameters[2 + 2 * decoder_layer_id + 1]->set_partial_shape(device_config.get_value_cache_shape());
    }
    model->validate_nodes_and_infer_types();
}
