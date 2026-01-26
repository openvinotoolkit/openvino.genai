// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <openvino/openvino.hpp>
#include "qwen3_moe_config.hpp"

namespace ov {
namespace genai {

/**
 * @brief Builder class for expert computation in Qwen3-MoE sparse blocks.
 * 
 * This class provides methods to construct the expert computation graph using
 * OpenVINO operators. The experts use 3D weight tensors for efficient storage
 * and computation of multiple expert networks.
 * 
 * Expert Computation Structure:
 * For each selected expert e:
 *   gate, up = split(hidden_states @ gate_up_proj[e]^T)
 *   intermediate = activation(gate) * up
 *   output = intermediate @ down_proj[e]^T
 * 
 * The expert weights are stored as 3D tensors:
 * - gate_up_proj: [num_experts, 2*intermediate_size, hidden_dim]
 * - down_proj: [num_experts, hidden_dim, intermediate_size]
 * 
 * Each token may be routed to different experts based on the router's top-k
 * selection. The final output is a weighted aggregation of expert outputs,
 * where weights come from the router's routing_weights.
 * 
 * Token-Level Expert Selection:
 * - Each token in the batch can be processed by different experts
 * - Expert selection is determined by the router's top-k mechanism
 * - Tokens are grouped by expert for efficient batch processing
 * - Expert outputs are weighted by routing scores before aggregation
 * 
 * Implementation Strategy:
 * 1. Create expert mask using one-hot encoding of selected_experts
 * 2. For each expert, identify which tokens use that expert
 * 3. Gather hidden states for tokens assigned to current expert
 * 4. Compute expert forward pass (gate/up/down projections)
 * 5. Weight expert output by routing weights
 * 6. Scatter-add weighted outputs to final result tensor
 */
class Qwen3MoeExpertsBuilder {
public:
    /**
     * @brief Constructs a new Qwen3MoeExpertsBuilder object.
     * 
     * @param config MoE layer configuration containing num_experts, top_k, and intermediate_size
     */
    explicit Qwen3MoeExpertsBuilder(const MoELayerConfig& config);

    /**
     * @brief Default constructor.
     */
    Qwen3MoeExpertsBuilder() = default;

    /**
     * @brief Builds the complete expert computation and aggregation graph.
     * 
     * Constructs the full expert processing pipeline:
     * 1. Validates input shapes and dimensions
     * 2. Reshapes hidden_states to 2D [batch*seq_len, hidden_dim]
     * 3. Loads 3D expert weight tensors (gate_up_proj, down_proj)
     * 4. Creates expert mask from selected_experts using one-hot encoding
     * 5. Iterates through experts, processing tokens assigned to each
     * 6. Aggregates weighted expert outputs into final result
     * 7. Reshapes output back to 3D [batch, seq_len, hidden_dim]
     * 
     * Weight Tensor Structure:
     * - gate_up_proj: [num_experts, 2*intermediate_size, hidden_dim]
     *   Contains concatenated gate and up projection weights for all experts
     * - down_proj: [num_experts, hidden_dim, intermediate_size]
     *   Contains down projection weights for all experts
     * 
     * Expert Computation for expert e:
     *   gate_up = hidden_states @ gate_up_proj[e]^T  # [tokens, 2*intermediate_size]
     *   gate, up = split(gate_up, axis=-1)           # Each [tokens, intermediate_size]
     *   intermediate = activation(gate) * up          # [tokens, intermediate_size]
     *   output = intermediate @ down_proj[e]^T        # [tokens, hidden_dim]
     * 
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_dim]
     * @param selected_experts Indices of selected experts, shape [batch*seq_len, top_k]
     * @param routing_weights Weights for expert aggregation, shape [batch*seq_len, top_k]
     * @param weight_prefix Prefix for weight keys (e.g., "model.layers.0.mlp.experts")
     * @param weights Map containing all model weight tensors
     * @param activation Activation function name (default: "silu")
     * @return ov::Output<ov::Node> Output tensor of shape [batch, seq_len, hidden_dim]
     * @throws std::runtime_error if inputs are invalid or weights are missing
     */
    ov::Output<ov::Node> build(
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& selected_experts,
        const ov::Output<ov::Node>& routing_weights,
        const std::string& weight_prefix,
        const std::unordered_map<std::string, ov::Tensor>& weights,
        const std::string& activation = "silu");

    /**
     * @brief Builds the computation graph for a single expert's forward pass.
     * 
     * Implements the expert MLP computation:
     * 1. Gather gate_up weights for expert_idx from 3D tensor
     * 2. Compute gate_up projection: hidden_states @ gate_up_weights^T
     * 3. Split into gate and up projections along last dimension
     * 4. Apply activation to gate projection
     * 5. Element-wise multiply: intermediate = activation(gate) * up
     * 6. Gather down weights for expert_idx from 3D tensor
     * 7. Compute down projection: intermediate @ down_weights^T
     * 
     * This method processes a batch of tokens assigned to a single expert.
     * The expert_idx is used to select the appropriate weight slices from
     * the 3D weight tensors.
     * 
     * @param hidden_states Input tensor for tokens assigned to this expert, shape [num_tokens, hidden_dim]
     * @param expert_idx Index of the expert to use (scalar or 1D tensor)
     * @param gate_up_weights 3D tensor of gate_up weights, shape [num_experts, 2*intermediate_size, hidden_dim]
     * @param down_weights 3D tensor of down weights, shape [num_experts, hidden_dim, intermediate_size]
     * @param activation Activation function name (e.g., "silu")
     * @return ov::Output<ov::Node> Expert output tensor, shape [num_tokens, hidden_dim]
     * @throws std::runtime_error if inputs are invalid
     */
    ov::Output<ov::Node> build_expert_loop_body(
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& expert_idx,
        const ov::Output<ov::Node>& gate_up_weights,
        const ov::Output<ov::Node>& down_weights,
        const std::string& activation);

    /**
     * @brief Aggregates expert outputs with routing weights.
     * 
     * Implements the weighted aggregation of expert outputs:
     * 1. Creates expert mask using one-hot encoding of selected_experts
     * 2. Permutes mask to [num_experts, top_k, batch*seq_len]
     * 3. Initializes final output tensor with zeros
     * 4. For each expert:
     *    a. Extract expert hit mask (which tokens use this expert)
     *    b. Find non-zero positions (tokens and top_k positions)
     *    c. Gather hidden states for tokens using this expert
     *    d. Compute expert output using build_expert_loop_body
     *    e. Weight expert output by corresponding routing_weights
     *    f. Scatter-add weighted output to final result
     * 5. Return aggregated final output
     * 
     * The aggregation ensures that each token's output is the weighted sum
     * of its top-k selected experts' outputs, where weights come from the
     * router's softmax probabilities.
     * 
     * Expert Mask Structure:
     * - selected_experts: [batch*seq_len, top_k] contains expert indices
     * - one_hot encoding: [batch*seq_len, top_k, num_experts]
     * - permuted mask: [num_experts, top_k, batch*seq_len]
     * - expert_hit[e]: [top_k, batch*seq_len] indicates which tokens use expert e
     * 
     * @param hidden_states Input tensor, shape [batch*seq_len, hidden_dim]
     * @param selected_experts Expert indices, shape [batch*seq_len, top_k]
     * @param routing_weights Routing weights, shape [batch*seq_len, top_k]
     * @param gate_up_weights 3D gate_up weight tensor
     * @param down_weights 3D down weight tensor
     * @param activation Activation function name
     * @return ov::Output<ov::Node> Aggregated output, shape [batch*seq_len, hidden_dim]
     * @throws std::runtime_error if inputs are invalid
     */
    ov::Output<ov::Node> aggregate_expert_outputs(
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& selected_experts,
        const ov::Output<ov::Node>& routing_weights,
        const ov::Output<ov::Node>& gate_up_weights,
        const ov::Output<ov::Node>& down_weights,
        const std::string& activation);

private:
    /**
     * @brief Reshapes 3D hidden states to 2D for expert computation.
     * 
     * Converts [batch, seq_len, hidden_dim] to [batch*seq_len, hidden_dim]
     * by computing the total number of tokens and creating a new shape.
     * 
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_dim]
     * @return ov::Output<ov::Node> Reshaped tensor of shape [batch*seq_len, hidden_dim]
     */
    ov::Output<ov::Node> reshape_to_2d(const ov::Output<ov::Node>& hidden_states);

    /**
     * @brief Reshapes 2D output back to 3D.
     * 
     * Converts [batch*seq_len, hidden_dim] back to [batch, seq_len, hidden_dim]
     * using the original batch and seq_len dimensions.
     * 
     * @param output_2d Output tensor of shape [batch*seq_len, hidden_dim]
     * @param batch_size Batch dimension
     * @param seq_len Sequence length dimension
     * @return ov::Output<ov::Node> Reshaped tensor of shape [batch, seq_len, hidden_dim]
     */
    ov::Output<ov::Node> reshape_to_3d(
        const ov::Output<ov::Node>& output_2d,
        const ov::Output<ov::Node>& batch_size,
        const ov::Output<ov::Node>& seq_len);

    /**
     * @brief Loads 3D expert weight tensor from weights map.
     * 
     * Looks up the weight tensor, validates its shape, and converts it to f32.
     * Expected shapes:
     * - gate_up_proj: [num_experts, 2*intermediate_size, hidden_dim]
     * - down_proj: [num_experts, hidden_dim, intermediate_size]
     * 
     * @param weight_key Key to lookup weight tensor
     * @param weights Map containing all model weight tensors
     * @param expected_shape Expected shape for validation (empty to skip validation)
     * @return ov::Output<ov::Node> Weight constant node in f32
     * @throws std::runtime_error if weight is missing or has invalid shape
     */
    ov::Output<ov::Node> load_expert_weight(
        const std::string& weight_key,
        const std::unordered_map<std::string, ov::Tensor>& weights,
        const std::vector<size_t>& expected_shape = {});

    /**
     * @brief Applies activation function to input tensor.
     * 
     * Supports activation functions: "silu", "swish", "gelu", "relu"
     * Defaults to SiLU if activation type is not recognized.
     * 
     * @param input Input tensor to apply activation to
     * @param activation_type Name of activation function
     * @return ov::Output<ov::Node> Output tensor after activation
     */
    ov::Output<ov::Node> apply_activation(
        const ov::Output<ov::Node>& input,
        const std::string& activation_type);

    /**
     * @brief Creates a zeros tensor with the same shape as the input.
     * 
     * @param reference Reference tensor to match shape
     * @return ov::Output<ov::Node> Zeros tensor with same shape as reference
     */
    ov::Output<ov::Node> create_zeros_like(const ov::Output<ov::Node>& reference);

    MoELayerConfig config_;  ///< MoE layer configuration
};

} // namespace genai
} // namespace ov