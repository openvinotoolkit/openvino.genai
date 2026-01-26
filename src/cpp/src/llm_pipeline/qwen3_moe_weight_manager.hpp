// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "openvino/openvino.hpp"
#include "qwen3_moe_config.hpp"

namespace ov {
namespace genai {

/**
 * @brief Enum for supported weight file formats.
 */
enum class WeightFormat {
    SAFETENSORS,  // .safetensors format (default for Qwen3-MoE)
    PYTORCH,      // .pt or .bin format
    GGUF,         // .gguf format
    UNKNOWN       // Unknown or unsupported format
};

/**
 * @brief Weight manager for Qwen3-MoE model.
 * 
 * Handles loading, validation, and management of model weights from checkpoint files.
 * Supports multiple weight formats (safetensors, PyTorch, GGUF) and handles 3D expert
 * weight tensors efficiently.
 * 
 * Weight Key Naming Convention:
 * - Embeddings: "model.embed_tokens.weight"
 * - Layer weights: "model.layers.{layer_idx}.{component}.{weight_name}"
 * - Final norm: "model.norm.weight"
 * - LM head: "lm_head.weight"
 * 
 * Expert Weight Tensor Structure (3D):
 * - gate_up_proj: [num_experts, 2*intermediate_size, hidden_dim]
 * - down_proj: [num_experts, hidden_dim, intermediate_size]
 */
class Qwen3MoeWeightManager {
public:
    /**
     * @brief Default constructor.
     */
    Qwen3MoeWeightManager() = default;

    /**
     * @brief Load weights from checkpoint file.
     * 
     * Loads all model weights from a checkpoint file into a weight map.
     * Automatically detects weight format from file extension.
     * 
     * @param checkpoint_path Path to checkpoint file or directory containing weights
     * @param config Model configuration for validation
     * @return Map of weight names to OpenVINO tensors
     * @throws std::runtime_error if checkpoint file not found or format unsupported
     */
    std::unordered_map<std::string, ov::Tensor> load_weights(
        const std::filesystem::path& checkpoint_path,
        const Qwen3MoeConfig& config);

    /**
     * @brief Load 3D expert weight tensor from checkpoint.
     * 
     * Loads expert weights as a 3D tensor with shape [num_experts, dim1, dim2].
     * If weights are stored per-expert, stacks them into a single 3D tensor.
     * 
     * @param checkpoint_path Path to checkpoint file
     * @param weight_key Base key for expert weights (e.g., "model.layers.0.mlp.experts.gate_up_proj")
     * @param num_experts Number of experts
     * @param dim1 First dimension size (e.g., 2*intermediate_size for gate_up_proj)
     * @param dim2 Second dimension size (e.g., hidden_dim)
     * @return 3D tensor with shape [num_experts, dim1, dim2]
     * @throws std::runtime_error if weights not found or shape mismatch
     */
    ov::Tensor load_expert_weights_3d(
        const std::filesystem::path& checkpoint_path,
        const std::string& weight_key,
        int num_experts,
        int dim1,
        int dim2);

    /**
     * @brief Reshape expert weights from list to 3D tensor.
     * 
     * Takes a vector of per-expert weight tensors and stacks them into a single
     * 3D tensor along the first dimension.
     * 
     * @param expert_weights Vector of expert weight tensors, each with shape [dim1, dim2]
     * @return 3D tensor with shape [num_experts, dim1, dim2]
     * @throws std::runtime_error if expert weights have inconsistent shapes
     */
    ov::Tensor reshape_expert_weights_to_3d(
        const std::vector<ov::Tensor>& expert_weights);

    /**
     * @brief Validate loaded weights against configuration.
     * 
     * Checks that all expected weights are present and have correct shapes.
     * 
     * @param weights Map of loaded weights
     * @param config Model configuration
     * @return true if all weights are valid, false otherwise
     */
    bool validate_weight_shapes(
        const std::unordered_map<std::string, ov::Tensor>& weights,
        const Qwen3MoeConfig& config);

    /**
     * @brief Get list of expected weight keys for a configuration.
     * 
     * Generates the complete list of weight keys that should be present
     * for the given model configuration.
     * 
     * @param config Model configuration
     * @return Vector of expected weight key names
     */
    std::vector<std::string> get_expected_weight_keys(
        const Qwen3MoeConfig& config);

    /**
     * @brief Detect weight format from file path.
     * 
     * Determines the weight file format based on file extension.
     * 
     * @param path Path to weight file
     * @return Detected weight format
     */
    static WeightFormat detect_format(const std::filesystem::path& path);

private:
    /**
     * @brief Load a single tensor from checkpoint file.
     * 
     * @param file_path Path to checkpoint file
     * @param tensor_name Name of tensor to load
     * @return Loaded tensor
     * @throws std::runtime_error if tensor not found
     */
    ov::Tensor load_single_tensor(
        const std::filesystem::path& file_path,
        const std::string& tensor_name);

    /**
     * @brief Load weights from safetensors format.
     * 
     * @param checkpoint_path Path to .safetensors file
     * @param config Model configuration
     * @return Map of weight names to tensors
     */
    std::unordered_map<std::string, ov::Tensor> load_safetensors(
        const std::filesystem::path& checkpoint_path,
        const Qwen3MoeConfig& config);

    /**
     * @brief Load weights from PyTorch format.
     * 
     * @param checkpoint_path Path to .pt or .bin file
     * @param config Model configuration
     * @return Map of weight names to tensors
     */
    std::unordered_map<std::string, ov::Tensor> load_pytorch(
        const std::filesystem::path& checkpoint_path,
        const Qwen3MoeConfig& config);

    /**
     * @brief Load weights from GGUF format.
     * 
     * @param checkpoint_path Path to .gguf file
     * @param config Model configuration
     * @return Map of weight names to tensors
     */
    std::unordered_map<std::string, ov::Tensor> load_gguf(
        const std::filesystem::path& checkpoint_path,
        const Qwen3MoeConfig& config);

    /**
     * @brief Validate shape of a specific weight tensor.
     * 
     * @param weight_name Name of the weight
     * @param tensor Weight tensor
     * @param expected_shape Expected shape
     * @return true if shape matches, false otherwise
     */
    bool validate_tensor_shape(
        const std::string& weight_name,
        const ov::Tensor& tensor,
        const std::vector<size_t>& expected_shape);

    /**
     * @brief Generate weight keys for embedding layer.
     * 
     * @return Vector of embedding weight keys
     */
    std::vector<std::string> get_embedding_keys();

    /**
     * @brief Generate weight keys for a decoder layer.
     * 
     * @param layer_idx Layer index
     * @param is_moe_layer Whether this layer uses MoE
     * @return Vector of weight keys for this layer
     */
    std::vector<std::string> get_layer_keys(int layer_idx, bool is_moe_layer);

    /**
     * @brief Generate weight keys for final norm and LM head.
     * 
     * @return Vector of output layer weight keys
     */
    std::vector<std::string> get_output_keys();
};

} // namespace genai
} // namespace ov