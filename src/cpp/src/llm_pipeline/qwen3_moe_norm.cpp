// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_norm.hpp"
#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include <stdexcept>

using namespace ov;
using namespace ov::op;

namespace ov {
namespace genai {

ov::Output<ov::Node> Qwen3MoeRMSNormBuilder::build(
    const ov::Output<ov::Node>& input,
    const std::string& weight_key,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    float epsilon) {
    
    // Input validation
    if (!input.get_node()) {
        throw std::runtime_error("Qwen3MoeRMSNormBuilder::build: input node is null");
    }
    
    if (epsilon <= 0.0f) {
        throw std::runtime_error("Qwen3MoeRMSNormBuilder::build: epsilon must be positive");
    }

    // Step 1: Create epsilon constant node
    auto eps_node = std::make_shared<v0::Constant>(
        element::f32, Shape{1, 1, 1}, epsilon);

    // Step 2: Square the input (input^2)
    auto constant_2 = std::make_shared<v0::Constant>(
        element::f32, Shape{1, 1, 1}, 2.0f);
    auto squared = std::make_shared<v1::Power>(input, constant_2);

    // Step 3: Compute mean along last dimension
    // ReduceMean with axis=-1 and keep_dims=true
    auto axis_last = std::make_shared<v0::Constant>(
        element::i32, Shape{1}, -1);
    auto variance = std::make_shared<v1::ReduceMean>(
        squared, axis_last, true);

    // Step 4: Add epsilon for numerical stability
    auto add_eps = std::make_shared<v1::Add>(variance, eps_node);

    // Step 5: Compute square root
    auto sqrt_node = std::make_shared<v0::Sqrt>(add_eps);

    // Step 6: Compute reciprocal (1 / sqrt(variance + epsilon))
    auto constant_1 = std::make_shared<v0::Constant>(
        element::f32, Shape{1, 1, 1}, 1.0f);
    auto reciprocal = std::make_shared<v1::Divide>(constant_1, sqrt_node);

    // Step 7: Multiply input by reciprocal
    auto normalized = std::make_shared<v1::Multiply>(
        input, reciprocal, AutoBroadcastType::NUMPY);

    // Step 8: Load and apply weight if exists
    auto weight_node = load_norm_weights(weight_key + ".weight", weights);
    
    if (weight_node.get_node()) {
        // Weight exists and is not all ones, apply it
        auto weighted_output = std::make_shared<v1::Multiply>(
            normalized, weight_node, AutoBroadcastType::NUMPY);
        return weighted_output;
    }

    // No weight or all-ones weight, return normalized output
    return normalized;
}

ov::Output<ov::Node> Qwen3MoeRMSNormBuilder::compute_rms_norm_no_weight(
    const ov::Output<ov::Node>& input,
    float epsilon) {
    
    // Input validation
    if (!input.get_node()) {
        throw std::runtime_error("Qwen3MoeRMSNormBuilder::compute_rms_norm_no_weight: input node is null");
    }
    
    if (epsilon <= 0.0f) {
        throw std::runtime_error("Qwen3MoeRMSNormBuilder::compute_rms_norm_no_weight: epsilon must be positive");
    }

    // Create epsilon constant
    auto eps_node = std::make_shared<v0::Constant>(
        element::f32, Shape{1, 1, 1}, epsilon);

    // Square the input
    auto constant_2 = std::make_shared<v0::Constant>(
        element::f32, Shape{1, 1, 1}, 2.0f);
    auto squared = std::make_shared<v1::Power>(input, constant_2);

    // Compute mean along last dimension
    auto axis_last = std::make_shared<v0::Constant>(
        element::i32, Shape{1}, -1);
    auto variance = std::make_shared<v1::ReduceMean>(
        squared, axis_last, true);

    // Add epsilon
    auto add_eps = std::make_shared<v1::Add>(variance, eps_node);

    // Compute square root
    auto sqrt_node = std::make_shared<v0::Sqrt>(add_eps);

    // Compute reciprocal
    auto constant_1 = std::make_shared<v0::Constant>(
        element::f32, Shape{1, 1, 1}, 1.0f);
    auto reciprocal = std::make_shared<v1::Divide>(constant_1, sqrt_node);

    // Multiply input by reciprocal
    auto normalized = std::make_shared<v1::Multiply>(
        input, reciprocal, AutoBroadcastType::NUMPY);

    return normalized;
}

ov::Output<ov::Node> Qwen3MoeRMSNormBuilder::apply_weight(
    const ov::Output<ov::Node>& normalized,
    const ov::Output<ov::Node>& weight) {
    
    // Input validation
    if (!normalized.get_node()) {
        throw std::runtime_error("Qwen3MoeRMSNormBuilder::apply_weight: normalized node is null");
    }
    
    if (!weight.get_node()) {
        throw std::runtime_error("Qwen3MoeRMSNormBuilder::apply_weight: weight node is null");
    }

    // Simple multiplication with NUMPY broadcasting
    auto weighted = std::make_shared<v1::Multiply>(
        normalized, weight, AutoBroadcastType::NUMPY);

    return weighted;
}

ov::Output<ov::Node> Qwen3MoeRMSNormBuilder::load_norm_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    // Check if weight exists
    if (weights.count(key) == 0) {
        // Weight doesn't exist, return empty output
        return ov::Output<ov::Node>();
    }

    auto weight_tensor = weights.at(key);
    
    // Check if all elements are 1.0 (optimization from building_blocks.cpp)
    bool all_ones = true;
    
    if (weight_tensor.get_element_type() == element::f32) {
        const float* data = weight_tensor.data<float>();
        for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
            if (data[i] != 1.0f) {
                all_ones = false;
                break;
            }
        }
    } else if (weight_tensor.get_element_type() == element::f16) {
        const uint16_t* data = weight_tensor.data<uint16_t>();
        const uint16_t one_in_fp16 = 0x3C00;  // FP16 representation of 1.0
        for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
            if (data[i] != one_in_fp16) {
                all_ones = false;
                break;
            }
        }
    } else {
        throw std::runtime_error(
            "Qwen3MoeRMSNormBuilder::load_norm_weights: Unsupported weight type " + 
            weight_tensor.get_element_type().get_type_name());
    }

    // If all weights are 1.0, skip multiplication (optimization)
    if (all_ones) {
        return ov::Output<ov::Node>();
    }

    // Reshape weight tensor for broadcasting: [hidden_size] -> [1, 1, hidden_size]
    auto original_shape = weight_tensor.get_shape();
    weight_tensor.set_shape(Shape{1, 1, original_shape[0]});

    // Create constant node from weight tensor
    auto weight_const = std::make_shared<v0::Constant>(weight_tensor);

    // Convert to f32 for computation
    auto weight_f32 = std::make_shared<v0::Convert>(
        weight_const, element::f32);

    return weight_f32;
}

} // namespace genai
} // namespace ov