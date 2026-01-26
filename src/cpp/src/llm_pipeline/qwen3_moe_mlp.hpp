// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <openvino/openvino.hpp>

namespace ov {
namespace genai {

/**
 * @brief Builder class for standard MLP (Multi-Layer Perceptron) layers.
 * 
 * This class provides methods to construct standard MLP computation graphs
 * using OpenVINO operators. The MLP is used in non-MoE decoder layers of
 * Qwen3-MoE architecture.
 * 
 * MLP Structure: down_proj(act_fn(gate_proj(x)) * up_proj(x))
 * 
 * The MLP consists of three linear projections:
 * 1. gate_proj: projects hidden_size -> intermediate_size
 * 2. up_proj: projects hidden_size -> intermediate_size
 * 3. down_proj: projects intermediate_size -> hidden_size
 * 
 * The gate projection is passed through an activation function (default SiLU),
 * then element-wise multiplied with the up projection, and finally projected
 * back to hidden size through the down projection.
 * 
 * Note: No bias terms are used in any of the projections.
 */
class Qwen3MoeMLPBuilder {
public:
    /**
     * @brief Activation function types supported by the MLP.
     */
    enum class ActivationType {
        SILU,   // Sigmoid Linear Unit (also called Swish)
        GELU,   // Gaussian Error Linear Unit
        RELU,   // Rectified Linear Unit
        SWISH   // Alias for SILU
    };

    /**
     * @brief Constructs a new Qwen3MoeMLPBuilder object.
     */
    Qwen3MoeMLPBuilder() = default;

    /**
     * @brief Builds the complete MLP computation graph.
     * 
     * Constructs the full MLP forward pass:
     * 1. Gate projection: gate = linear(hidden_states, gate_proj.weight)
     * 2. Activation: gate_act = activation_fn(gate)
     * 3. Up projection: up = linear(hidden_states, up_proj.weight)
     * 4. Element-wise multiply: gate_up = gate_act * up
     * 5. Down projection: output = linear(gate_up, down_proj.weight)
     * 
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @param layer_prefix Prefix for weight keys (e.g., "model.layers.0.mlp")
     * @param weights Map containing all model weight tensors
     * @param intermediate_size Size of the intermediate hidden dimension
     * @param activation Activation function name (default: "silu")
     * @return ov::Output<ov::Node> Output tensor of shape [batch, seq_len, hidden_size]
     * @throws std::runtime_error if inputs are invalid or weights are missing
     */
    ov::Output<ov::Node> build(
        const ov::Output<ov::Node>& hidden_states,
        const std::string& layer_prefix,
        const std::unordered_map<std::string, ov::Tensor>& weights,
        int intermediate_size,
        const std::string& activation = "silu");

private:
    /**
     * @brief Creates a linear projection layer (matrix multiplication).
     * 
     * Performs: output = input @ weight^T
     * 
     * The weight matrix is transposed during multiplication (transpose_b=true).
     * Weights are converted to f32 for computation. No bias is added.
     * 
     * @param input Input tensor to project
     * @param weight_key Key to lookup weight tensor in weights map
     * @param weights Map containing all model weight tensors
     * @param transpose_weight Whether to transpose weight matrix (default: true)
     * @return ov::Output<ov::Node> Output of linear projection
     * @throws std::runtime_error if weight tensor is missing or invalid
     */
    ov::Output<ov::Node> make_linear(
        const ov::Output<ov::Node>& input,
        const std::string& weight_key,
        const std::unordered_map<std::string, ov::Tensor>& weights,
        bool transpose_weight = true);

    /**
     * @brief Applies activation function to input tensor.
     * 
     * Supports the following activation functions:
     * - "silu" or "swish": Sigmoid Linear Unit (x * sigmoid(x))
     * - "gelu": Gaussian Error Linear Unit
     * - "relu": Rectified Linear Unit (max(0, x))
     * 
     * If the activation type is not recognized, defaults to SiLU.
     * 
     * @param input Input tensor to apply activation to
     * @param activation_type Name of activation function
     * @return ov::Output<ov::Node> Output tensor after activation
     */
    ov::Output<ov::Node> apply_activation(
        const ov::Output<ov::Node>& input,
        const std::string& activation_type);

    /**
     * @brief Converts activation string to ActivationType enum.
     * 
     * @param activation_str Activation function name string
     * @return ActivationType Corresponding enum value
     */
    ActivationType parse_activation_type(const std::string& activation_str);
};

} // namespace genai
} // namespace ov