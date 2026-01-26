// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <string>

namespace ov {
namespace genai {

/**
 * @brief Configuration for MoE (Mixture of Experts) layer parameters.
 */
struct MoELayerConfig {
    int num_experts = 128;
    int top_k = 8;  // num_experts_per_tok
    int intermediate_size = 768;  // moe_intermediate_size
    bool normalize_topk = false;  // norm_topk_prob
};

/**
 * @brief Configuration for attention mechanism parameters.
 */
struct AttentionConfig {
    int num_heads = 32;
    int num_kv_heads = 4;
    int head_dim = 64;
    int sliding_window = 4096;
    float dropout = 0.0f;
    float rms_norm_eps = 1e-6f;
    bool attention_bias = false;
    bool use_sliding_window = false;
};

/**
 * @brief Configuration for Rotary Position Embeddings (RoPE).
 */
struct RoPEConfig {
    int max_position_embeddings = 32768;
    float rope_theta = 10000.0f;
    int head_dim = 64;
};

/**
 * @brief Configuration class for Qwen3-MoE model parameters.
 * 
 * This configuration corresponds to the Qwen3-MoE model architecture,
 * including sparse mixture-of-experts layers, multi-head attention with
 * Q/K normalization, sliding window attention, and rotary position embeddings.
 */
class Qwen3MoeConfig {
public:
    // Basic model parameters
    int vocab_size = 151936;
    int hidden_size = 2048;
    int num_hidden_layers = 24;
    int num_attention_heads = 32;
    int num_key_value_heads = 4;
    int intermediate_size = 6144;  // For standard MLP layers
    std::string hidden_act = "silu";

    // MoE parameters
    int num_experts = 128;
    int num_experts_per_tok = 8;
    int moe_intermediate_size = 768;
    int decoder_sparse_step = 1;
    bool norm_topk_prob = false;
    bool output_router_logits = false;
    float router_aux_loss_coef = 0.001f;
    std::vector<int> mlp_only_layers;  // Layer indices that use standard MLP instead of MoE

    // Attention parameters
    bool attention_bias = false;
    float attention_dropout = 0.0f;
    bool use_sliding_window = false;
    int sliding_window = 4096;

    // RoPE parameters
    int max_position_embeddings = 32768;
    float rope_theta = 10000.0f;

    // Normalization
    float rms_norm_eps = 1e-6f;

    // Other parameters
    float initializer_range = 0.02f;
    bool use_cache = true;
    bool tie_word_embeddings = false;

    /**
     * @brief Default constructor with default values.
     */
    Qwen3MoeConfig() = default;

    /**
     * @brief Construct Qwen3MoeConfig from a JSON configuration file.
     * @param config_path Path to the JSON configuration file.
     */
    explicit Qwen3MoeConfig(const std::filesystem::path& config_path);

    /**
     * @brief Get MoE layer configuration for a specific layer.
     * @param layer_idx Index of the layer (0-based).
     * @return MoELayerConfig structure with MoE parameters.
     */
    MoELayerConfig get_moe_layer_config(int layer_idx) const;

    /**
     * @brief Get attention configuration.
     * @return AttentionConfig structure with attention parameters.
     */
    AttentionConfig get_attention_config() const;

    /**
     * @brief Get RoPE configuration.
     * @return RoPEConfig structure with RoPE parameters.
     */
    RoPEConfig get_rope_config() const;

    /**
     * @brief Determine if a layer should use MoE block.
     * @param layer_idx Index of the layer (0-based).
     * @return true if the layer uses MoE, false if it uses standard MLP.
     */
    bool is_moe_layer(int layer_idx) const;

    /**
     * @brief Validate configuration consistency.
     * Throws exception if configuration is invalid.
     */
    void validate() const;
};

/**
 * @brief Parse Qwen3-MoE configuration from a JSON file.
 * @param config_path Path to the JSON configuration file.
 * @return Qwen3MoeConfig object with parsed parameters.
 */
Qwen3MoeConfig parse_qwen3_moe_config_from_json(const std::filesystem::path& config_path);

/**
 * @brief Get default Qwen3-MoE configuration for testing.
 * @return Qwen3MoeConfig object with default values.
 */
Qwen3MoeConfig get_default_qwen3_moe_config();

}  // namespace genai
}  // namespace ov