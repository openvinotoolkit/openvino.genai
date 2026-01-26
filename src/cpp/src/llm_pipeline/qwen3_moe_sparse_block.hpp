// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <openvino/openvino.hpp>
#include "qwen3_moe_config.hpp"
#include "qwen3_moe_router.hpp"
#include "qwen3_moe_experts.hpp"

namespace ov {
namespace genai {

/**
 * @brief Builder class for sparse MoE block orchestration in Qwen3-MoE.
 * 
 * This class orchestrates the complete sparse mixture-of-experts block by:
 * 1. Using the router to select top-k experts per token
 * 2. Computing expert outputs for selected experts
 * 3. Aggregating weighted expert outputs
 * 
 * The sparse MoE block workflow:
 *   hidden_states [batch, seq_len, hidden_dim]
 *   -> router: compute routing logits and select top-k experts
 *   -> experts: compute outputs for selected experts
 *   -> aggregate: weighted sum of expert outputs
 *   -> output [batch, seq_len, hidden_dim]
 * 
 * Integration:
 * This builder is used in decoder layers based on the layer selection strategy
 * (determined by decoder_sparse_step and mlp_only_layers configuration).
 * 
 * Load Balancing:
 * The router_logits output can be used for computing auxiliary load balancing
 * loss to encourage balanced expert utilization during training.
 */
class Qwen3MoeSparseMoeBlockBuilder {
public:
    /**
     * @brief Constructs a new Qwen3MoeSparseMoeBlockBuilder object.
     * 
     * @param config MoE layer configuration containing num_experts, top_k, and intermediate_size
     * @param router_builder Shared pointer to router builder for expert selection
     * @param experts_builder Shared pointer to experts builder for expert computation
     */
    Qwen3MoeSparseMoeBlockBuilder(
        const MoELayerConfig& config,
        std::shared_ptr<Qwen3MoeTopKRouterBuilder> router_builder,
        std::shared_ptr<Qwen3MoeExpertsBuilder> experts_builder);

    /**
     * @brief Default constructor.
     */
    Qwen3MoeSparseMoeBlockBuilder() = default;

    /**
     * @brief Builds the complete sparse MoE block computation graph.
     * 
     * Orchestrates the full sparse MoE workflow:
     * 1. Validates input shapes and dimensions
     * 2. Calls router to compute routing logits and select top-k experts
     * 3. Calls experts builder to compute expert outputs
     * 4. Returns the aggregated output
     * 
     * Input/Output Shapes:
     * - Input hidden_states: [batch, seq_len, hidden_dim]
     * - Router outputs:
     *   - router_logits: [batch*seq_len, num_experts] (for load balancing loss)
     *   - routing_weights: [batch*seq_len, top_k] (for expert aggregation)
     *   - selected_experts: [batch*seq_len, top_k] (expert indices)
     * - Expert output: [batch*seq_len, hidden_dim]
     * - Final output: [batch, seq_len, hidden_dim] (reshaped from expert output)
     * 
     * Error Handling:
     * - Validates hidden_states is 3D tensor [batch, seq_len, hidden_dim]
     * - Checks weight tensors exist for router and all experts
     * - Validates shape compatibility throughout the pipeline
     * 
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_dim]
     * @param layer_prefix Prefix for weight keys (e.g., "model.layers.0.mlp")
     * @param weights Map containing all model weight tensors
     * @return ov::Output<ov::Node> Output tensor of shape [batch, seq_len, hidden_dim]
     * @throws std::runtime_error if inputs are invalid or weights are missing
     */
    ov::Output<ov::Node> build(
        const ov::Output<ov::Node>& hidden_states,
        const std::string& layer_prefix,
        const std::unordered_map<std::string, ov::Tensor>& weights);

private:
    MoELayerConfig config_;  ///< MoE layer configuration
    std::shared_ptr<Qwen3MoeTopKRouterBuilder> router_builder_;  ///< Router builder for expert selection
    std::shared_ptr<Qwen3MoeExpertsBuilder> experts_builder_;  ///< Experts builder for expert computation
};

} // namespace genai
} // namespace ov