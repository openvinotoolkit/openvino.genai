// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>
#include <unordered_map>
#include <openvino/openvino.hpp>
#include "qwen3_moe_config.hpp"

namespace ov {
namespace genai {

/**
 * @brief Builder class for TopK routing mechanism in Qwen3-MoE.
 * 
 * This class provides methods to construct the TopK routing computation graph
 * using OpenVINO operators. The router is responsible for:
 * 1. Computing router logits for all experts
 * 2. Applying softmax to get routing probabilities
 * 3. Selecting top-k experts per token
 * 4. Optionally normalizing the routing weights
 * 
 * The router takes hidden states as input and produces:
 * - router_logits: Used for load balancing loss computation
 * - routing_weights: Weights for aggregating expert outputs
 * - selected_experts: Indices of the selected top-k experts
 * 
 * Formula:
 *   router_logits = hidden_states @ router_weight^T
 *   router_probs = softmax(router_logits, dim=-1)
 *   routing_weights, selected_experts = topk(router_probs, k=top_k)
 *   if normalize_topk:
 *       routing_weights = routing_weights / sum(routing_weights)
 */
class Qwen3MoeTopKRouterBuilder {
public:
    /**
     * @brief Constructs a new Qwen3MoeTopKRouterBuilder object.
     * 
     * @param config MoE layer configuration containing num_experts, top_k, and normalize_topk
     */
    explicit Qwen3MoeTopKRouterBuilder(const MoELayerConfig& config);

    /**
     * @brief Default constructor.
     */
    Qwen3MoeTopKRouterBuilder() = default;

    /**
     * @brief Builds the complete TopK routing computation graph.
     * 
     * Constructs the full routing mechanism:
     * 1. Loads router weight tensor from weights map
     * 2. Computes router logits via linear projection
     * 3. Applies softmax to get routing probabilities
     * 4. Selects top-k experts per token
     * 5. Optionally normalizes routing weights to sum to 1
     * 
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_dim]
     * @param weight_key Prefix for weight keys (e.g., "model.layers.0.mlp.gate")
     * @param weights Map containing all model weight tensors
     * @return std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>>
     *         Returns (router_logits, routing_weights, selected_experts):
     *         - router_logits: [batch*seq_len, num_experts] - for load balancing loss
     *         - routing_weights: [batch*seq_len, top_k] - weights for expert aggregation
     *         - selected_experts: [batch*seq_len, top_k] - indices of selected experts
     * @throws std::runtime_error if inputs are invalid or weights are missing
     */
    std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>> build(
        const ov::Output<ov::Node>& hidden_states,
        const std::string& weight_key,
        const std::unordered_map<std::string, ov::Tensor>& weights);

    /**
     * @brief Computes router logits via linear projection.
     * 
     * Performs the following operations:
     * 1. Reshapes hidden_states from [batch, seq_len, hidden_dim] to [batch*seq_len, hidden_dim]
     * 2. Applies linear projection: router_logits = hidden_states @ router_weight^T
     * 
     * The router weight has shape [num_experts, hidden_dim], so the output
     * router_logits has shape [batch*seq_len, num_experts].
     * 
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_dim]
     * @param router_weight Router weight tensor of shape [num_experts, hidden_dim]
     * @return ov::Output<ov::Node> Router logits of shape [batch*seq_len, num_experts]
     * @throws std::runtime_error if inputs are invalid
     */
    ov::Output<ov::Node> compute_router_logits(
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& router_weight);

    /**
     * @brief Selects top-k experts based on routing probabilities.
     * 
     * Performs the following operations:
     * 1. Applies softmax to router_logits along the expert dimension
     * 2. Selects top-k experts using TopK operation
     * 3. Optionally normalizes the routing weights to sum to 1
     * 
     * The normalization step ensures that the weighted sum of expert outputs
     * maintains proper scale. This is controlled by the normalize parameter.
     * 
     * @param router_logits Router logits of shape [batch*seq_len, num_experts]
     * @param top_k Number of experts to select per token
     * @param normalize Whether to normalize routing weights to sum to 1
     * @return std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
     *         Returns (routing_weights, selected_experts):
     *         - routing_weights: [batch*seq_len, top_k] - normalized or unnormalized weights
     *         - selected_experts: [batch*seq_len, top_k] - indices of selected experts
     * @throws std::runtime_error if inputs are invalid or top_k is invalid
     */
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> select_topk_experts(
        const ov::Output<ov::Node>& router_logits,
        int top_k,
        bool normalize);

private:
    /**
     * @brief Reshapes 3D hidden states to 2D for routing computation.
     * 
     * Converts [batch, seq_len, hidden_dim] to [batch*seq_len, hidden_dim]
     * by computing the total number of tokens and creating a new shape.
     * 
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_dim]
     * @return ov::Output<ov::Node> Reshaped tensor of shape [batch*seq_len, hidden_dim]
     */
    ov::Output<ov::Node> reshape_to_2d(const ov::Output<ov::Node>& hidden_states);

    /**
     * @brief Loads router weight tensor from weights map.
     * 
     * Looks up the router weight tensor, validates its shape, and converts
     * it to f32 for computation. Expected shape: [num_experts, hidden_dim].
     * 
     * @param weight_key Key to lookup weight tensor (e.g., "model.layers.0.mlp.gate.weight")
     * @param weights Map containing all model weight tensors
     * @return ov::Output<ov::Node> Router weight constant node in f32
     * @throws std::runtime_error if weight is missing or has invalid shape
     */
    ov::Output<ov::Node> load_router_weight(
        const std::string& weight_key,
        const std::unordered_map<std::string, ov::Tensor>& weights);

    MoELayerConfig config_;  ///< MoE layer configuration
};

} // namespace genai
} // namespace ov