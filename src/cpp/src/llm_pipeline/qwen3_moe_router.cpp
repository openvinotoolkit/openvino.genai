// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_router.hpp"
#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include <stdexcept>

using namespace ov;
using namespace ov::op;

namespace ov {
namespace genai {

Qwen3MoeTopKRouterBuilder::Qwen3MoeTopKRouterBuilder(const MoELayerConfig& config)
    : config_(config) {
    // Validate configuration
    if (config_.num_experts <= 0) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder: num_experts must be positive, got " + 
                                 std::to_string(config_.num_experts));
    }
    if (config_.top_k <= 0) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder: top_k must be positive, got " + 
                                 std::to_string(config_.top_k));
    }
    if (config_.top_k > config_.num_experts) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder: top_k (" + 
                                 std::to_string(config_.top_k) + 
                                 ") cannot exceed num_experts (" + 
                                 std::to_string(config_.num_experts) + ")");
    }
}

std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>> 
Qwen3MoeTopKRouterBuilder::build(
    const ov::Output<ov::Node>& hidden_states,
    const std::string& weight_key,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::build: hidden_states node is null");
    }
    
    if (weight_key.empty()) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::build: weight_key is empty");
    }
    
    // Validate hidden_states rank is 3 [batch, seq_len, hidden_dim]
    auto hidden_shape = hidden_states.get_partial_shape();
    if (hidden_shape.rank().is_static() && hidden_shape.rank().get_length() != 3) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::build: hidden_states must be 3D [batch, seq_len, hidden_dim], got rank " + 
                                 std::to_string(hidden_shape.rank().get_length()));
    }

    // Step 1: Load router weight tensor
    auto router_weight = load_router_weight(weight_key + ".weight", weights);

    // Step 2: Compute router logits
    auto router_logits = compute_router_logits(hidden_states, router_weight);

    // Step 3: Select top-k experts
    auto [routing_weights, selected_experts] = select_topk_experts(
        router_logits, 
        config_.top_k, 
        config_.normalize_topk);

    // Return tuple of (router_logits, routing_weights, selected_experts)
    return std::make_tuple(router_logits, routing_weights, selected_experts);
}

ov::Output<ov::Node> Qwen3MoeTopKRouterBuilder::compute_router_logits(
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& router_weight) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::compute_router_logits: hidden_states node is null");
    }
    
    if (!router_weight.get_node()) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::compute_router_logits: router_weight node is null");
    }

    // Step 1: Reshape hidden_states to 2D [batch*seq_len, hidden_dim]
    auto hidden_states_2d = reshape_to_2d(hidden_states);

    // Step 2: Linear projection: router_logits = hidden_states_2d @ router_weight^T
    // router_weight shape: [num_experts, hidden_dim]
    // hidden_states_2d shape: [batch*seq_len, hidden_dim]
    // output shape: [batch*seq_len, num_experts]
    auto router_logits = std::make_shared<v0::MatMul>(
        hidden_states_2d, router_weight, false, true);

    return router_logits;
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> 
Qwen3MoeTopKRouterBuilder::select_topk_experts(
    const ov::Output<ov::Node>& router_logits,
    int top_k,
    bool normalize) {
    
    // Input validation
    if (!router_logits.get_node()) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::select_topk_experts: router_logits node is null");
    }
    
    if (top_k <= 0) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::select_topk_experts: top_k must be positive, got " + 
                                 std::to_string(top_k));
    }

    // Step 1: Apply softmax along the expert dimension (axis=-1)
    auto axis_last = std::make_shared<v0::Constant>(element::i64, Shape{}, -1);
    auto router_probs = std::make_shared<v8::Softmax>(router_logits, -1);

    // Step 2: TopK selection
    // Create constant for k value
    auto k_const = std::make_shared<v0::Constant>(element::i64, Shape{}, top_k);
    
    // TopK operation: returns (values, indices)
    // mode="max" selects largest values
    // sort="value" sorts by value in descending order
    // axis=-1 operates on the last dimension (expert dimension)
    auto topk = std::make_shared<v11::TopK>(
        router_probs,
        k_const,
        -1,  // axis
        v11::TopK::Mode::MAX,
        v11::TopK::SortType::SORT_VALUES,
        element::i64);  // index element type

    // Extract outputs from TopK
    auto routing_weights = topk->output(0);  // Top-k values
    auto selected_experts = topk->output(1);  // Top-k indices

    // Step 3: Normalize routing weights if requested
    if (normalize) {
        // Compute sum along the top-k dimension (axis=-1, keep_dims=true)
        auto reduce_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, -1);
        auto sum = std::make_shared<v1::ReduceSum>(routing_weights, reduce_axis, true);
        
        // Normalize: routing_weights = routing_weights / sum
        routing_weights = std::make_shared<v1::Divide>(
            routing_weights, sum, AutoBroadcastType::NUMPY);
    }

    return std::make_pair(routing_weights, selected_experts);
}

ov::Output<ov::Node> Qwen3MoeTopKRouterBuilder::reshape_to_2d(
    const ov::Output<ov::Node>& hidden_states) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::reshape_to_2d: hidden_states node is null");
    }

    // Get shape of hidden_states: [batch, seq_len, hidden_dim]
    auto shape_node = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);

    // Extract dimensions
    // Get batch dimension (index 0)
    auto index_0 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_dim = std::make_shared<v8::Gather>(shape_node, index_0, axis_0);

    // Get seq_len dimension (index 1)
    auto index_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto seq_len_dim = std::make_shared<v8::Gather>(shape_node, index_1, axis_0);

    // Get hidden_dim dimension (index 2)
    auto index_2 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
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

ov::Output<ov::Node> Qwen3MoeTopKRouterBuilder::load_router_weight(
    const std::string& weight_key,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    // Input validation
    if (weight_key.empty()) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::load_router_weight: weight_key is empty");
    }

    // Check if weight exists
    if (weights.count(weight_key) == 0) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::load_router_weight: weight tensor not found for key: " + 
                                 weight_key);
    }

    auto weight_tensor = weights.at(weight_key);
    
    // Validate weight tensor shape: should be [num_experts, hidden_dim]
    auto weight_shape = weight_tensor.get_shape();
    if (weight_shape.size() != 2) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::load_router_weight: weight tensor must be 2D [num_experts, hidden_dim], got " + 
                                 std::to_string(weight_shape.size()) + "D for key: " + weight_key);
    }

    // Validate num_experts dimension matches configuration
    if (config_.num_experts > 0 && weight_shape[0] != static_cast<size_t>(config_.num_experts)) {
        throw std::runtime_error("Qwen3MoeTopKRouterBuilder::load_router_weight: weight shape[0] (" + 
                                 std::to_string(weight_shape[0]) + 
                                 ") does not match num_experts (" + 
                                 std::to_string(config_.num_experts) + ")");
    }

    // Create weight constant node
    auto weight_const = std::make_shared<v0::Constant>(weight_tensor);

    // Convert weight to f32 for computation
    auto weight_f32 = std::make_shared<v0::Convert>(
        weight_const, element::f32);

    return weight_f32;
}

} // namespace genai
} // namespace ov