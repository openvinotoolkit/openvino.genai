// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_experts.hpp"
#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include <stdexcept>
#include <algorithm>
#include <cctype>

using namespace ov;
using namespace ov::op;

namespace ov {
namespace genai {

Qwen3MoeExpertsBuilder::Qwen3MoeExpertsBuilder(const MoELayerConfig& config)
    : config_(config) {
    // Validate configuration
    if (config_.num_experts <= 0) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder: num_experts must be positive, got " + 
                                 std::to_string(config_.num_experts));
    }
    if (config_.top_k <= 0) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder: top_k must be positive, got " + 
                                 std::to_string(config_.top_k));
    }
    if (config_.intermediate_size <= 0) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder: intermediate_size must be positive, got " + 
                                 std::to_string(config_.intermediate_size));
    }
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::build(
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& selected_experts,
    const ov::Output<ov::Node>& routing_weights,
    const std::string& weight_prefix,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    const std::string& activation) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build: hidden_states node is null");
    }
    if (!selected_experts.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build: selected_experts node is null");
    }
    if (!routing_weights.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build: routing_weights node is null");
    }
    if (weight_prefix.empty()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build: weight_prefix is empty");
    }

    // Validate hidden_states shape: [batch, seq_len, hidden_dim]
    auto hidden_shape = hidden_states.get_partial_shape();
    if (hidden_shape.rank().is_static() && hidden_shape.rank().get_length() != 3) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build: hidden_states must be 3D [batch, seq_len, hidden_dim], got rank " + 
                                 std::to_string(hidden_shape.rank().get_length()));
    }

    // Step 1: Get original shape dimensions for later reshaping
    auto shape_node = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);
    auto index_0 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto index_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_size = std::make_shared<v8::Gather>(shape_node, index_0, axis_0);
    auto seq_len = std::make_shared<v8::Gather>(shape_node, index_1, axis_0);

    // Step 2: Reshape hidden_states to 2D [batch*seq_len, hidden_dim]
    auto hidden_states_2d = reshape_to_2d(hidden_states);

    // Step 3: Load 3D expert weight tensors
    auto gate_up_weights = load_expert_weight(
        weight_prefix + ".gate_up_proj.weight", 
        weights);
    auto down_weights = load_expert_weight(
        weight_prefix + ".down_proj.weight", 
        weights);

    // Step 4: Aggregate expert outputs
    auto aggregated_output = aggregate_expert_outputs(
        hidden_states_2d,
        selected_experts,
        routing_weights,
        gate_up_weights,
        down_weights,
        activation);

    // Step 5: Reshape output back to 3D [batch, seq_len, hidden_dim]
    auto output_3d = reshape_to_3d(aggregated_output, batch_size, seq_len);

    return output_3d;
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::build_expert_loop_body(
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& expert_idx,
    const ov::Output<ov::Node>& gate_up_weights,
    const ov::Output<ov::Node>& down_weights,
    const std::string& activation) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build_expert_loop_body: hidden_states node is null");
    }
    if (!expert_idx.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build_expert_loop_body: expert_idx node is null");
    }
    if (!gate_up_weights.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build_expert_loop_body: gate_up_weights node is null");
    }
    if (!down_weights.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::build_expert_loop_body: down_weights node is null");
    }

    // Step 1: Gather gate_up weights for this expert from 3D tensor
    // gate_up_weights shape: [num_experts, 2*intermediate_size, hidden_dim]
    // After gather: [2*intermediate_size, hidden_dim]
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto gate_up_slice = std::make_shared<v8::Gather>(
        gate_up_weights, expert_idx, axis_0);

    // Step 2: Compute gate_up projection
    // hidden_states @ gate_up_slice^T
    // Output shape: [num_tokens, 2*intermediate_size]
    auto gate_up_matmul = std::make_shared<v0::MatMul>(
        hidden_states, gate_up_slice, false, true);

    // Step 3: Split into gate and up projections
    // Split along last dimension into 2 equal parts
    auto axis_neg1 = std::make_shared<v0::Constant>(element::i64, Shape{}, -1);
    auto num_splits = std::make_shared<v0::Constant>(element::i64, Shape{}, 2);
    auto split = std::make_shared<v1::Split>(gate_up_matmul, axis_neg1, 2);
    auto gate_proj = split->output(0);  // [num_tokens, intermediate_size]
    auto up_proj = split->output(1);    // [num_tokens, intermediate_size]

    // Step 4: Apply activation to gate projection
    auto gate_act = apply_activation(gate_proj, activation);

    // Step 5: Element-wise multiply gate_act and up_proj
    // intermediate = gate_act * up_proj
    auto intermediate = std::make_shared<v1::Multiply>(
        gate_act, up_proj, AutoBroadcastType::NUMPY);

    // Step 6: Gather down weights for this expert from 3D tensor
    // down_weights shape: [num_experts, hidden_dim, intermediate_size]
    // After gather: [hidden_dim, intermediate_size]
    auto down_slice = std::make_shared<v8::Gather>(
        down_weights, expert_idx, axis_0);

    // Step 7: Compute down projection
    // intermediate @ down_slice^T
    // Output shape: [num_tokens, hidden_dim]
    auto output = std::make_shared<v0::MatMul>(
        intermediate, down_slice, false, true);

    return output;
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::aggregate_expert_outputs(
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& selected_experts,
    const ov::Output<ov::Node>& routing_weights,
    const ov::Output<ov::Node>& gate_up_weights,
    const ov::Output<ov::Node>& down_weights,
    const std::string& activation) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::aggregate_expert_outputs: hidden_states node is null");
    }
    if (!selected_experts.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::aggregate_expert_outputs: selected_experts node is null");
    }
    if (!routing_weights.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::aggregate_expert_outputs: routing_weights node is null");
    }

    // Step 1: Create expert mask using one-hot encoding
    // selected_experts shape: [batch*seq_len, top_k]
    // one_hot output shape: [batch*seq_len, top_k, num_experts]
    auto depth = std::make_shared<v0::Constant>(element::i64, Shape{}, config_.num_experts);
    auto on_value = std::make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto off_value = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto axis_neg1 = std::make_shared<v0::Constant>(element::i64, Shape{}, -1);
    
    auto expert_mask = std::make_shared<v9::OneHot>(
        selected_experts, depth, on_value, off_value, -1);

    // Step 2: Permute expert_mask to [num_experts, top_k, batch*seq_len]
    auto perm = std::make_shared<v0::Constant>(
        element::i64, Shape{3}, std::vector<int64_t>{2, 1, 0});
    auto expert_mask_permuted = std::make_shared<v1::Transpose>(expert_mask, perm);

    // Step 3: Initialize final output tensor with zeros
    auto final_output = create_zeros_like(hidden_states);

    // Step 4: Process each expert
    // For simplicity and efficiency, we'll use a loop-based approach
    // In a production implementation, this could be optimized with Loop operator
    
    // Get shape information
    auto hidden_shape = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);
    auto index_0 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto index_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto num_tokens = std::make_shared<v8::Gather>(hidden_shape, index_0, axis_0);
    auto hidden_dim = std::make_shared<v8::Gather>(hidden_shape, index_1, axis_0);

    // Create a loop to process each expert
    // Loop structure: for expert_idx in range(num_experts)
    auto trip_count = std::make_shared<v0::Constant>(element::i64, Shape{}, config_.num_experts);
    auto condition = std::make_shared<v0::Constant>(element::boolean, Shape{}, true);
    
    // Create Loop operation
    auto loop = std::make_shared<v5::Loop>(trip_count, condition);
    
    // Loop body inputs: (iteration, final_output_in)
    auto body_param_iter = std::make_shared<v0::Parameter>(element::i64, Shape{});
    auto body_param_output = std::make_shared<v0::Parameter>(
        hidden_states.get_element_type(), 
        hidden_states.get_partial_shape());
    
    // Get expert mask for current expert
    auto expert_hit = std::make_shared<v8::Gather>(
        expert_mask_permuted, body_param_iter, axis_0);
    
    // Find tokens using this expert: sum over top_k dimension
    auto reduce_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto expert_hit_sum = std::make_shared<v1::ReduceSum>(expert_hit, reduce_axis, false);
    
    // Convert to boolean mask
    auto zero_const = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto token_mask = std::make_shared<v1::Greater>(expert_hit_sum, zero_const);
    
    // Find non-zero positions (tokens using this expert)
    auto nonzero = std::make_shared<v3::NonZero>(token_mask, element::i64);
    auto squeeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto token_indices = std::make_shared<v0::Squeeze>(nonzero, squeeze_axis);
    
    // Gather hidden states for tokens using this expert
    auto current_states = std::make_shared<v8::Gather>(
        hidden_states, token_indices, axis_0);
    
    // Compute expert output
    auto expert_output = build_expert_loop_body(
        current_states,
        body_param_iter,
        gate_up_weights,
        down_weights,
        activation);
    
    // Get routing weights for these tokens
    // Need to find which top_k position each token uses for this expert
    // This is complex, so we'll use a simplified approach:
    // Gather routing weights based on expert_hit mask
    
    // For each token, find its top_k position for this expert
    auto expert_hit_expanded = std::make_shared<v0::Unsqueeze>(
        expert_hit, 
        std::make_shared<v0::Constant>(element::i64, Shape{1}, -1));
    
    // Multiply expert_hit with routing_weights to get weights for this expert
    auto expert_routing_weights = std::make_shared<v1::Multiply>(
        expert_hit_expanded, routing_weights, AutoBroadcastType::NUMPY);
    
    // Sum over top_k dimension to get final weight per token
    auto weight_sum = std::make_shared<v1::ReduceSum>(
        expert_routing_weights, reduce_axis, false);
    
    // Gather weights for tokens using this expert
    auto current_weights = std::make_shared<v8::Gather>(
        weight_sum, token_indices, axis_0);
    
    // Expand weights to match expert_output shape
    auto weights_expanded = std::make_shared<v0::Unsqueeze>(
        current_weights,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, -1));
    
    // Weight expert output
    auto weighted_output = std::make_shared<v1::Multiply>(
        expert_output, weights_expanded, AutoBroadcastType::NUMPY);
    
    // Scatter-add weighted output to final result
    // Create indices for ScatterNDUpdate
    auto indices_expanded = std::make_shared<v0::Unsqueeze>(
        token_indices,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, -1));
    
    auto updated_output = std::make_shared<v15::ScatterNDUpdate>(
        body_param_output, indices_expanded, weighted_output);
    
    // Loop body outputs: (condition, updated_output)
    auto body_condition = std::make_shared<v0::Constant>(element::boolean, Shape{}, true);
    
    auto body_result = std::make_shared<ov::Model>(
        OutputVector{body_condition, updated_output},
        ParameterVector{body_param_iter, body_param_output});
    
    loop->set_function(body_result);
    loop->set_special_body_ports({-1, 0});
    loop->set_merged_input(body_param_output, final_output, updated_output);
    
    // Get loop output
    auto loop_output = loop->output(0);
    
    return loop_output;
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::reshape_to_2d(
    const ov::Output<ov::Node>& hidden_states) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::reshape_to_2d: hidden_states node is null");
    }

    // Get shape of hidden_states: [batch, seq_len, hidden_dim]
    auto shape_node = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);

    // Extract dimensions
    auto index_0 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto index_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto index_2 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    
    auto batch_dim = std::make_shared<v8::Gather>(shape_node, index_0, axis_0);
    auto seq_len_dim = std::make_shared<v8::Gather>(shape_node, index_1, axis_0);
    auto hidden_dim = std::make_shared<v8::Gather>(shape_node, index_2, axis_0);

    // Compute total tokens: batch * seq_len
    auto total_tokens = std::make_shared<v1::Multiply>(
        batch_dim, seq_len_dim, AutoBroadcastType::NUMPY);

    // Create target shape: [total_tokens, hidden_dim]
    auto target_shape = std::make_shared<v0::Concat>(
        OutputVector{total_tokens, hidden_dim}, 0);

    // Reshape hidden_states to 2D
    auto hidden_states_2d = std::make_shared<v1::Reshape>(
        hidden_states, target_shape, false);

    return hidden_states_2d;
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::reshape_to_3d(
    const ov::Output<ov::Node>& output_2d,
    const ov::Output<ov::Node>& batch_size,
    const ov::Output<ov::Node>& seq_len) {
    
    // Input validation
    if (!output_2d.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::reshape_to_3d: output_2d node is null");
    }
    if (!batch_size.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::reshape_to_3d: batch_size node is null");
    }
    if (!seq_len.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::reshape_to_3d: seq_len node is null");
    }

    // Get hidden_dim from output_2d shape
    auto shape_node = std::make_shared<v3::ShapeOf>(output_2d, element::i64);
    auto index_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto hidden_dim = std::make_shared<v8::Gather>(shape_node, index_1, axis_0);

    // Create target shape: [batch, seq_len, hidden_dim]
    auto target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_size, seq_len, hidden_dim}, 0);

    // Reshape output to 3D
    auto output_3d = std::make_shared<v1::Reshape>(
        output_2d, target_shape, false);

    return output_3d;
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::load_expert_weight(
    const std::string& weight_key,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    const std::vector<size_t>& expected_shape) {
    
    // Input validation
    if (weight_key.empty()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::load_expert_weight: weight_key is empty");
    }

    // Check if weight exists
    if (weights.count(weight_key) == 0) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::load_expert_weight: weight tensor not found for key: " + 
                                 weight_key);
    }

    auto weight_tensor = weights.at(weight_key);
    
    // Validate weight tensor shape: should be 3D
    auto weight_shape = weight_tensor.get_shape();
    if (weight_shape.size() != 3) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::load_expert_weight: weight tensor must be 3D, got " + 
                                 std::to_string(weight_shape.size()) + "D for key: " + weight_key);
    }

    // Validate num_experts dimension matches configuration
    if (config_.num_experts > 0 && weight_shape[0] != static_cast<size_t>(config_.num_experts)) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::load_expert_weight: weight shape[0] (" + 
                                 std::to_string(weight_shape[0]) + 
                                 ") does not match num_experts (" + 
                                 std::to_string(config_.num_experts) + ")");
    }

    // Optional: validate expected shape if provided
    if (!expected_shape.empty() && weight_shape != expected_shape) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::load_expert_weight: weight shape mismatch for key: " + 
                                 weight_key);
    }

    // Create weight constant node
    auto weight_const = std::make_shared<v0::Constant>(weight_tensor);

    // Convert weight to f32 for computation
    auto weight_f32 = std::make_shared<v0::Convert>(
        weight_const, element::f32);

    return weight_f32;
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::apply_activation(
    const ov::Output<ov::Node>& input,
    const std::string& activation_type) {
    
    // Input validation
    if (!input.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::apply_activation: input node is null");
    }

    // Convert to lowercase for case-insensitive comparison
    std::string lower_str = activation_type;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Apply corresponding activation function
    if (lower_str == "silu" || lower_str == "swish") {
        // SiLU/Swish: x * sigmoid(x)
        return std::make_shared<v4::Swish>(input);
    } else if (lower_str == "gelu") {
        // GELU: Gaussian Error Linear Unit
        return std::make_shared<v7::Gelu>(input);
    } else if (lower_str == "relu") {
        // ReLU: max(0, x)
        return std::make_shared<v0::Relu>(input);
    } else {
        // Default to SiLU if unknown activation type
        return std::make_shared<v4::Swish>(input);
    }
}

ov::Output<ov::Node> Qwen3MoeExpertsBuilder::create_zeros_like(
    const ov::Output<ov::Node>& reference) {
    
    // Input validation
    if (!reference.get_node()) {
        throw std::runtime_error("Qwen3MoeExpertsBuilder::create_zeros_like: reference node is null");
    }

    // Get shape of reference tensor
    auto shape_node = std::make_shared<v3::ShapeOf>(reference, element::i64);

    // Create zeros constant with shape [1]
    auto zero_scalar = std::make_shared<v0::Constant>(
        reference.get_element_type(), Shape{}, 0.0f);

    // Broadcast zero to match reference shape
    auto zeros = std::make_shared<v3::Broadcast>(
        zero_scalar, shape_node, BroadcastModeSpec(BroadcastType::NUMPY));

    return zeros;
}

} // namespace genai
} // namespace ov