// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen3_moe_mlp.hpp"
#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

using namespace ov;
using namespace ov::op;

namespace ov {
namespace genai {

ov::Output<ov::Node> Qwen3MoeMLPBuilder::build(
    const ov::Output<ov::Node>& hidden_states,
    const std::string& layer_prefix,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    int intermediate_size,
    const std::string& activation) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::build: hidden_states node is null");
    }
    
    if (layer_prefix.empty()) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::build: layer_prefix is empty");
    }
    
    if (intermediate_size <= 0) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::build: intermediate_size must be positive, got " + 
                                 std::to_string(intermediate_size));
    }

    // Step 1: Gate projection
    // gate = linear(hidden_states, gate_proj.weight)
    auto gate = make_linear(
        hidden_states,
        layer_prefix + ".gate_proj",
        weights);

    // Step 2: Apply activation function
    // gate_act = activation_fn(gate)
    auto gate_act = apply_activation(gate, activation);

    // Step 3: Up projection
    // up = linear(hidden_states, up_proj.weight)
    auto up = make_linear(
        hidden_states,
        layer_prefix + ".up_proj",
        weights);

    // Step 4: Element-wise multiply gate_act and up
    // gate_up = gate_act * up
    auto gate_up = std::make_shared<v1::Multiply>(
        gate_act, up, AutoBroadcastType::NUMPY);

    // Step 5: Down projection
    // output = linear(gate_up, down_proj.weight)
    auto output = make_linear(
        gate_up,
        layer_prefix + ".down_proj",
        weights);

    return output;
}

ov::Output<ov::Node> Qwen3MoeMLPBuilder::make_linear(
    const ov::Output<ov::Node>& input,
    const std::string& weight_key,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    bool transpose_weight) {
    
    // Input validation
    if (!input.get_node()) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::make_linear: input node is null");
    }
    
    if (weight_key.empty()) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::make_linear: weight_key is empty");
    }

    // Load weight tensor
    std::string full_weight_key = weight_key + ".weight";
    if (weights.count(full_weight_key) == 0) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::make_linear: weight tensor not found for key: " + 
                                 full_weight_key);
    }

    auto weight_tensor = weights.at(full_weight_key);
    
    // Validate weight tensor shape
    auto weight_shape = weight_tensor.get_shape();
    if (weight_shape.size() != 2) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::make_linear: weight tensor must be 2D, got " + 
                                 std::to_string(weight_shape.size()) + "D for key: " + full_weight_key);
    }

    // Create weight constant node
    auto weight_const = std::make_shared<v0::Constant>(weight_tensor);

    // Convert weight to f32 for computation
    auto weight_f32 = std::make_shared<v0::Convert>(
        weight_const, element::f32);

    // Perform matrix multiplication
    // output = input @ weight^T (if transpose_weight=true)
    // output = input @ weight (if transpose_weight=false)
    auto matmul = std::make_shared<v0::MatMul>(
        input, weight_f32, false, transpose_weight);

    return matmul;
}

ov::Output<ov::Node> Qwen3MoeMLPBuilder::apply_activation(
    const ov::Output<ov::Node>& input,
    const std::string& activation_type) {
    
    // Input validation
    if (!input.get_node()) {
        throw std::runtime_error("Qwen3MoeMLPBuilder::apply_activation: input node is null");
    }

    // Parse activation type
    ActivationType act_type = parse_activation_type(activation_type);

    // Apply corresponding activation function
    switch (act_type) {
        case ActivationType::SILU:
        case ActivationType::SWISH: {
            // SiLU/Swish: x * sigmoid(x)
            auto swish = std::make_shared<v4::Swish>(input);
            return swish;
        }
        
        case ActivationType::GELU: {
            // GELU: Gaussian Error Linear Unit
            auto gelu = std::make_shared<v7::Gelu>(input);
            return gelu;
        }
        
        case ActivationType::RELU: {
            // ReLU: max(0, x)
            auto relu = std::make_shared<v0::Relu>(input);
            return relu;
        }
        
        default: {
            // Default to SiLU if unknown activation type
            auto swish = std::make_shared<v4::Swish>(input);
            return swish;
        }
    }
}

Qwen3MoeMLPBuilder::ActivationType Qwen3MoeMLPBuilder::parse_activation_type(
    const std::string& activation_str) {
    
    // Convert to lowercase for case-insensitive comparison
    std::string lower_str = activation_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Parse activation type
    if (lower_str == "silu") {
        return ActivationType::SILU;
    } else if (lower_str == "swish") {
        return ActivationType::SWISH;
    } else if (lower_str == "gelu") {
        return ActivationType::GELU;
    } else if (lower_str == "relu") {
        return ActivationType::RELU;
    } else {
        // Default to SILU for unknown types
        return ActivationType::SILU;
    }
}

} // namespace genai
} // namespace ov