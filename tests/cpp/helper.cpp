// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "helper.hpp"
#include "openvino/op/concat.hpp"

std::shared_ptr<ov::Model> get_dummy_model(ov::Core core, size_t num_layers) {
    ov::NodeVector keys, values;
    ov::ParameterVector params;
    ov::element::Type kv_cache_type = core.get_property("CPU", ov::hint::kv_cache_precision);

    auto shape = ov::PartialShape::dynamic(4);
    shape[1] = 12;
    shape[2] = 64;
    shape[3] = 64;

    for (size_t i = 0; i < num_layers; i++) {
        auto key = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, shape);
        key->get_output_tensor(0).set_names({"key_cache." + std::to_string(i)});
        value->get_output_tensor(0).set_names({"value_cache." + std::to_string(i)});
        keys.push_back(key);
        values.push_back(value);
        params.push_back(key);
        params.push_back(value);
    }
    const auto& concat1 = std::make_shared<ov::op::v0::Concat>(keys, 1);
    const auto& concat2 = std::make_shared<ov::op::v0::Concat>(values, 1);
    return std::make_shared<ov::Model>(ov::OutputVector{concat1, concat2}, params);
}

std::shared_ptr<ov::Model> get_dummy_hybrid_model(ov::Core core, size_t kv_num_layers, size_t la_num_layers) {
    ov::OutputVector outputs;
    ov::ParameterVector params;
    ov::element::Type kv_cache_type = core.get_property("CPU", ov::hint::kv_cache_precision);
    
    // KV cache inputs
    ov::NodeVector keys, values;
    auto kv_shape = ov::PartialShape::dynamic(4);
    kv_shape[1] = 12;
    kv_shape[2] = 64;
    kv_shape[3] = 64;
    
    for (size_t i = 0; i < kv_num_layers; i++) {
        auto key = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, kv_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, kv_shape);
        key->get_output_tensor(0).set_names({"key_cache." + std::to_string(i)});
        value->get_output_tensor(0).set_names({"value_cache." + std::to_string(i)});
        keys.push_back(key);
        values.push_back(value);
        params.push_back(key);
        params.push_back(value);
    }
    if (!keys.empty()) {
        outputs.push_back(std::make_shared<ov::op::v0::Concat>(keys, 1));
        outputs.push_back(std::make_shared<ov::op::v0::Concat>(values, 1));
    }
    
    // Linear attention state table inputs (one per layer)
    auto state_shape = ov::PartialShape::dynamic(3);
    state_shape[1] = 64;   // feature dimension
    state_shape[2] = 64;   // state size
    
    for (size_t i = 0; i < la_num_layers; i++) {
        auto conv_state = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, state_shape);
        conv_state->get_output_tensor(0).set_names({"conv_state_table." + std::to_string(i)});
        params.push_back(conv_state);
        outputs.push_back(conv_state->output(0));
        
        auto gated_state = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, state_shape);
        gated_state->get_output_tensor(0).set_names({"gated_delta_state_table." + std::to_string(i)});
        params.push_back(gated_state);
        outputs.push_back(gated_state->output(0));
    }
    
    return std::make_shared<ov::Model>(outputs, params);
}
