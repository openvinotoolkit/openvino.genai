// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <openvino/openvino.hpp>

namespace ov {
namespace genai {

/**
 * @brief Builder class for RMS (Root Mean Square) normalization operations.
 * 
 * This class provides methods to construct RMS normalization computation graphs
 * using OpenVINO operators. RMS normalization is used in three places in Qwen3-MoE:
 * 1. input_layernorm - normalizes input to each decoder layer
 * 2. post_attention_layernorm - normalizes after attention before MLP/MoE
 * 3. Q/K head normalization - normalizes query and key tensors on head dimension only
 * 
 * Formula: output = input * rsqrt(mean(input^2) + epsilon) * weight
 * 
 * For Q/K normalization, the operation is performed on the head dimension only,
 * unlike full hidden dimension normalization used in layer norms.
 */
class Qwen3MoeRMSNormBuilder {
public:
    /**
     * @brief Constructs a new Qwen3MoeRMSNormBuilder object.
     */
    Qwen3MoeRMSNormBuilder() = default;

    /**
     * @brief Builds RMS normalization graph with learned weights.
     * 
     * Constructs the complete RMS normalization operation including:
     * 1. Computing variance: mean(input^2)
     * 2. Adding epsilon for numerical stability
     * 3. Computing reciprocal square root
     * 4. Multiplying input by reciprocal
     * 5. Applying learned weight parameters (if available)
     * 
     * @param input Input tensor to normalize
     * @param weight_key Key to lookup weight tensor in weights map (e.g., "model.layers.0.input_layernorm")
     * @param weights Map containing all model weight tensors
     * @param epsilon Small constant for numerical stability (default: 1e-6)
     * @return ov::Output<ov::Node> Normalized output tensor
     */
    ov::Output<ov::Node> build(
        const ov::Output<ov::Node>& input,
        const std::string& weight_key,
        const std::unordered_map<std::string, ov::Tensor>& weights,
        float epsilon = 1e-6f);

    /**
     * @brief Computes RMS normalization without learned weights.
     * 
     * This is useful for Q/K head normalization where weights might not exist
     * or when only the normalization operation is needed without scaling.
     * 
     * Formula: output = input * rsqrt(mean(input^2) + epsilon)
     * 
     * @param input Input tensor to normalize
     * @param epsilon Small constant for numerical stability (default: 1e-6)
     * @return ov::Output<ov::Node> Normalized output tensor (without weight scaling)
     */
    ov::Output<ov::Node> compute_rms_norm_no_weight(
        const ov::Output<ov::Node>& input,
        float epsilon = 1e-6f);

    /**
     * @brief Applies learned weight parameters to normalized tensor.
     * 
     * Multiplies the normalized tensor by learned weight parameters using
     * NUMPY broadcasting rules. This is separated from normalization to allow
     * flexible composition.
     * 
     * @param normalized Normalized input tensor
     * @param weight Weight tensor to multiply with
     * @return ov::Output<ov::Node> Weighted output tensor
     */
    ov::Output<ov::Node> apply_weight(
        const ov::Output<ov::Node>& normalized,
        const ov::Output<ov::Node>& weight);

private:
    /**
     * @brief Loads normalization weight tensor from weights map.
     * 
     * Looks up the weight tensor using the provided key, handles type conversion
     * to f32, and reshapes for proper broadcasting. Optimizes for the case where
     * all weights are 1.0 (returns nullptr to skip multiplication).
     * 
     * @param key Weight key to lookup (e.g., "model.layers.0.input_layernorm.weight")
     * @param weights Map containing all model weight tensors
     * @return ov::Output<ov::Node> Weight constant node, or nullptr if weights are all ones
     */
    ov::Output<ov::Node> load_norm_weights(
        const std::string& key,
        const std::unordered_map<std::string, ov::Tensor>& weights);
};

} // namespace genai
} // namespace ov