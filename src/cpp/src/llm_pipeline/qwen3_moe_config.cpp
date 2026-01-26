// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_config.hpp"
#include "json_utils.hpp"

#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <openvino/core/except.hpp>

namespace ov {
namespace genai {

namespace {

/**
 * @brief Validate that hidden_size is divisible by num_attention_heads.
 */
void validate_attention_heads(int hidden_size, int num_attention_heads) {
    OPENVINO_ASSERT(
        hidden_size % num_attention_heads == 0,
        "hidden_size (", hidden_size, ") must be divisible by num_attention_heads (", num_attention_heads, ")"
    );
}

/**
 * @brief Validate that num_attention_heads is divisible by num_key_value_heads.
 */
void validate_kv_heads(int num_attention_heads, int num_key_value_heads) {
    OPENVINO_ASSERT(
        num_attention_heads % num_key_value_heads == 0,
        "num_attention_heads (", num_attention_heads, ") must be divisible by num_key_value_heads (", num_key_value_heads, ")"
    );
}

/**
 * @brief Validate MoE parameters.
 */
void validate_moe_params(int num_experts, int num_experts_per_tok, int moe_intermediate_size, int decoder_sparse_step) {
    OPENVINO_ASSERT(num_experts > 0, "num_experts must be positive, got: ", num_experts);
    OPENVINO_ASSERT(num_experts_per_tok > 0, "num_experts_per_tok must be positive, got: ", num_experts_per_tok);
    OPENVINO_ASSERT(
        num_experts_per_tok <= num_experts,
        "num_experts_per_tok (", num_experts_per_tok, ") must be <= num_experts (", num_experts, ")"
    );
    OPENVINO_ASSERT(moe_intermediate_size > 0, "moe_intermediate_size must be positive, got: ", moe_intermediate_size);
    OPENVINO_ASSERT(decoder_sparse_step > 0, "decoder_sparse_step must be positive, got: ", decoder_sparse_step);
}

/**
 * @brief Validate mlp_only_layers indices.
 */
void validate_mlp_only_layers(const std::vector<int>& mlp_only_layers, int num_hidden_layers) {
    for (int layer_idx : mlp_only_layers) {
        OPENVINO_ASSERT(
            layer_idx >= 0 && layer_idx < num_hidden_layers,
            "mlp_only_layers contains invalid layer index: ", layer_idx,
            " (must be in range [0, ", num_hidden_layers, "))"
        );
    }
}

}  // namespace

Qwen3MoeConfig::Qwen3MoeConfig(const std::filesystem::path& config_path) {
    std::ifstream stream(config_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", config_path, "' with Qwen3-MoE config");
    
    nlohmann::json data = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;

    // Parse basic parameters
    read_json_param(data, "vocab_size", vocab_size);
    read_json_param(data, "hidden_size", hidden_size);
    read_json_param(data, "num_hidden_layers", num_hidden_layers);
    read_json_param(data, "num_attention_heads", num_attention_heads);
    read_json_param(data, "num_key_value_heads", num_key_value_heads);
    read_json_param(data, "intermediate_size", intermediate_size);
    read_json_param(data, "hidden_act", hidden_act);

    // Parse MoE parameters
    read_json_param(data, "num_experts", num_experts);
    read_json_param(data, "num_experts_per_tok", num_experts_per_tok);
    read_json_param(data, "moe_intermediate_size", moe_intermediate_size);
    read_json_param(data, "decoder_sparse_step", decoder_sparse_step);
    read_json_param(data, "norm_topk_prob", norm_topk_prob);
    read_json_param(data, "output_router_logits", output_router_logits);
    read_json_param(data, "router_aux_loss_coef", router_aux_loss_coef);
    read_json_param(data, "mlp_only_layers", mlp_only_layers);

    // Parse attention parameters
    read_json_param(data, "attention_bias", attention_bias);
    read_json_param(data, "attention_dropout", attention_dropout);
    read_json_param(data, "use_sliding_window", use_sliding_window);
    read_json_param(data, "sliding_window", sliding_window);

    // Parse RoPE parameters
    read_json_param(data, "max_position_embeddings", max_position_embeddings);
    
    // Handle rope_theta which might be in rope_parameters dict
    if (data.contains("rope_parameters") && data["rope_parameters"].is_object()) {
        read_json_param(data["rope_parameters"], "rope_theta", rope_theta);
    } else {
        read_json_param(data, "rope_theta", rope_theta);
    }

    // Parse normalization
    read_json_param(data, "rms_norm_eps", rms_norm_eps);

    // Parse other parameters
    read_json_param(data, "initializer_range", initializer_range);
    read_json_param(data, "use_cache", use_cache);
    read_json_param(data, "tie_word_embeddings", tie_word_embeddings);

    // Validate configuration
    validate();
}

MoELayerConfig Qwen3MoeConfig::get_moe_layer_config(int layer_idx) const {
    MoELayerConfig config;
    config.num_experts = num_experts;
    config.top_k = num_experts_per_tok;
    config.intermediate_size = moe_intermediate_size;
    config.normalize_topk = norm_topk_prob;
    return config;
}

AttentionConfig Qwen3MoeConfig::get_attention_config() const {
    AttentionConfig config;
    config.num_heads = num_attention_heads;
    config.num_kv_heads = num_key_value_heads;
    config.head_dim = hidden_size / num_attention_heads;
    config.sliding_window = sliding_window;
    config.dropout = attention_dropout;
    config.rms_norm_eps = rms_norm_eps;
    config.attention_bias = attention_bias;
    config.use_sliding_window = use_sliding_window;
    return config;
}

RoPEConfig Qwen3MoeConfig::get_rope_config() const {
    RoPEConfig config;
    config.max_position_embeddings = max_position_embeddings;
    config.rope_theta = rope_theta;
    config.head_dim = hidden_size / num_attention_heads;
    return config;
}

bool Qwen3MoeConfig::is_moe_layer(int layer_idx) const {
    // Check if layer is explicitly marked as MLP-only
    if (std::find(mlp_only_layers.begin(), mlp_only_layers.end(), layer_idx) != mlp_only_layers.end()) {
        return false;
    }

    // If mlp_only_layers is empty and decoder_sparse_step > 0,
    // use decoder_sparse_step to determine MoE layers
    if (mlp_only_layers.empty() && decoder_sparse_step > 0 && num_experts > 0) {
        // Layer uses MoE if (layer_idx + 1) is divisible by decoder_sparse_step
        return ((layer_idx + 1) % decoder_sparse_step) == 0;
    }

    // Otherwise, not a MoE layer
    return false;
}

void Qwen3MoeConfig::validate() const {
    // Validate attention configuration
    validate_attention_heads(hidden_size, num_attention_heads);
    validate_kv_heads(num_attention_heads, num_key_value_heads);

    // Validate MoE parameters
    validate_moe_params(num_experts, num_experts_per_tok, moe_intermediate_size, decoder_sparse_step);

    // Validate intermediate_size
    OPENVINO_ASSERT(intermediate_size > 0, "intermediate_size must be positive, got: ", intermediate_size);

    // Validate mlp_only_layers
    validate_mlp_only_layers(mlp_only_layers, num_hidden_layers);

    // Validate other parameters
    OPENVINO_ASSERT(vocab_size > 0, "vocab_size must be positive, got: ", vocab_size);
    OPENVINO_ASSERT(num_hidden_layers > 0, "num_hidden_layers must be positive, got: ", num_hidden_layers);
    OPENVINO_ASSERT(max_position_embeddings > 0, "max_position_embeddings must be positive, got: ", max_position_embeddings);
    OPENVINO_ASSERT(rope_theta > 0.0f, "rope_theta must be positive, got: ", rope_theta);
    OPENVINO_ASSERT(rms_norm_eps > 0.0f, "rms_norm_eps must be positive, got: ", rms_norm_eps);
}

Qwen3MoeConfig parse_qwen3_moe_config_from_json(const std::filesystem::path& config_path) {
    return Qwen3MoeConfig(config_path);
}

Qwen3MoeConfig get_default_qwen3_moe_config() {
    Qwen3MoeConfig config;
    // Default values are already set in the class definition
    // This function is useful for testing and as a reference
    return config;
}

}  // namespace genai
}  // namespace ov