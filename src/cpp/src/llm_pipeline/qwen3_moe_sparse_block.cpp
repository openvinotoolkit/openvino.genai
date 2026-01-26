// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_sparse_block.hpp"
#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include <stdexcept>

using namespace ov;
using namespace ov::op;

namespace ov {
namespace genai {

Qwen3MoeSparseMoeBlockBuilder::Qwen3MoeSparseMoeBlockBuilder(
    const MoELayerConfig& config,
    std::shared_ptr<Qwen3MoeTopKRouterBuilder> router_builder,
    std::shared_ptr<Qwen3MoeExpertsBuilder> experts_builder)
    : config_(config),
      router_builder_(router_builder),
      experts_builder_(experts_builder) {
    
    // Validate configuration
    if (config_.num_experts <= 0) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder: num_experts must be positive, got " + 
                                 std::to_string(config_.num_experts));
    }
    if (config_.top_k <= 0) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder: top_k must be positive, got " + 
                                 std::to_string(config_.top_k));
    }
    if (config_.top_k > config_.num_experts) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder: top_k (" + 
                                 std::to_string(config_.top_k) + 
                                 ") cannot exceed num_experts (" + 
                                 std::to_string(config_.num_experts) + ")");
    }
    
    // Validate builder dependencies
    if (!router_builder_) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder: router_builder cannot be null");
    }
    if (!experts_builder_) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder: experts_builder cannot be null");
    }
}

ov::Output<ov::Node> Qwen3MoeSparseMoeBlockBuilder::build(
    const ov::Output<ov::Node>& hidden_states,
    const std::string& layer_prefix,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    // Step 1: Validate inputs
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder::build: hidden_states node is null");
    }
    
    if (layer_prefix.empty()) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder::build: layer_prefix is empty");
    }
    
    // Validate hidden_states shape: [batch, seq_len, hidden_dim]
    auto hidden_shape = hidden_states.get_partial_shape();
    if (hidden_shape.rank().is_static() && hidden_shape.rank().get_length() != 3) {
        throw std::runtime_error("Qwen3MoeSparseMoeBlockBuilder::build: hidden_states must be 3D [batch, seq_len, hidden_dim], got rank " + 
                                 std::to_string(hidden_shape.rank().get_length()));
    }
    
    // Step 2: Get input shape components for later reshaping
    // Extract batch_size, seq_len, hidden_dim from hidden_states shape
    auto shape_node = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);
    
    auto index_0 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto index_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto index_2 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    
    auto batch_size = std::make_shared<v8::Gather>(shape_node, index_0, axis_0);
    auto seq_len = std::make_shared<v8::Gather>(shape_node, index_1, axis_0);
    auto hidden_dim = std::make_shared<v8::Gather>(shape_node, index_2, axis_0);
    
    // Step 3: Call router to select experts
    // Router returns: (router_logits, routing_weights, selected_experts)
    // - router_logits: [batch*seq_len, num_experts] - for load balancing loss
    // - routing_weights: [batch*seq_len, top_k] - weights for expert aggregation
    // - selected_experts: [batch*seq_len, top_k] - indices of selected experts
    auto [router_logits, routing_weights, selected_experts] = router_builder_->build(
        hidden_states,
        layer_prefix + ".gate",
        weights);
    
    // Step 4: Call experts computation
    // Experts builder handles:
    // - Reshaping hidden_states to 2D [batch*seq_len, hidden_dim]
    // - Computing expert outputs for selected experts
    // - Aggregating weighted expert outputs
    // - Reshaping back to 3D [batch, seq_len, hidden_dim]
    auto expert_output = experts_builder_->build(
        hidden_states,
        selected_experts,
        routing_weights,
        layer_prefix + ".experts",
        weights,
        "silu");  // Default activation for Qwen3-MoE
    
    // Step 5: Return final output
    // The experts_builder already handles reshaping back to 3D,
    // so expert_output has shape [batch, seq_len, hidden_dim]
    return expert_output;
}

} // namespace genai
} // namespace ov