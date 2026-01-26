// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <utility>
#include <tuple>
#include <openvino/openvino.hpp>
#include "qwen3_moe_config.hpp"
#include "qwen3_moe_norm.hpp"
#include "qwen3_moe_rope.hpp"

namespace ov {
namespace genai {

/**
 * @brief Builder class for Qwen3-MoE multi-head attention with Q/K normalization.
 * 
 * This class constructs the multi-head attention computation graph for Qwen3-MoE,
 * which includes several unique features:
 * 1. Q/K normalization on head dimension (not full hidden dimension)
 * 2. Sliding window attention support for local attention patterns
 * 3. Grouped Query Attention (GQA) with KV head expansion
 * 4. Rotary Position Embeddings (RoPE) application
 * 5. KV cache management with ReadValue/Assign operations
 * 
 * The attention mechanism follows the standard scaled dot-product attention:
 *   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + mask) * V
 * 
 * Reference: modeling_qwen3_moe.py lines 130-201
 */
class Qwen3MoeAttentionBuilder {
public:
    /**
     * @brief Constructs a new Qwen3MoeAttentionBuilder object.
     * 
     * @param config Attention configuration parameters
     * @param norm_builder Shared pointer to RMS normalization builder for Q/K norm
     * @param rope_builder Shared pointer to RoPE builder for position encoding
     */
    Qwen3MoeAttentionBuilder(
        const AttentionConfig& config,
        std::shared_ptr<Qwen3MoeRMSNormBuilder> norm_builder,
        std::shared_ptr<Qwen3MoeRotaryEmbeddingBuilder> rope_builder);

    /**
     * @brief Builds the complete multi-head attention computation graph.
     * 
     * This method constructs the full attention mechanism including:
     * 1. Q/K/V projections with Q/K normalization
     * 2. Head splitting and transposition
     * 3. RoPE application to Q and K
     * 4. KV cache handling (ReadValue, concatenation, Assign)
     * 5. Sliding window mask construction
     * 6. Scaled dot-product attention computation
     * 7. Output projection
     * 
     * @param hidden_states Input hidden states of shape [batch, seq_len, hidden_size]
     * @param attention_mask Attention mask of shape [batch, seq_len] or [batch, 1, seq_len, seq_len]
     * @param position_ids Position indices of shape [batch, seq_len]
     * @param position_embeddings Pair of (cos, sin) embeddings from RoPE builder
     * @param layer_prefix Weight key prefix (e.g., "model.layers.0.self_attn")
     * @param weights Map containing all model weight tensors
     * @return Pair of (attention_output, cache_sinks) where:
     *         - attention_output: shape [batch, seq_len, hidden_size]
     *         - cache_sinks: vector of Assign operations for KV cache
     */
    std::pair<ov::Output<ov::Node>, ov::SinkVector> build(
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& attention_mask,
        const ov::Output<ov::Node>& position_ids,
        const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& position_embeddings,
        const std::string& layer_prefix,
        const std::unordered_map<std::string, ov::Tensor>& weights);

    /**
     * @brief Builds Q/K/V projections with Q/K normalization on head dimension.
     * 
     * This method performs:
     * 1. Linear projections: Q = hidden @ W_q, K = hidden @ W_k, V = hidden @ W_v
     * 2. Reshape to [batch, seq_len, num_heads, head_dim]
     * 3. Apply RMS norm to Q and K on head dimension only (unlike full layer norm)
     * 4. Transpose to [batch, num_heads, seq_len, head_dim]
     * 
     * Note: Q/K normalization is applied AFTER reshape but BEFORE transpose,
     * operating only on the head_dim dimension (last dimension after reshape).
     * 
     * Reference: modeling_qwen3_moe.py lines 171-173
     * 
     * @param hidden_states Input hidden states of shape [batch, seq_len, hidden_size]
     * @param layer_prefix Weight key prefix for loading projection weights
     * @param weights Map containing all model weight tensors
     * @return Tuple of (query, key, value) tensors, each of shape:
     *         - query: [batch, num_heads, seq_len, head_dim]
     *         - key: [batch, num_kv_heads, seq_len, head_dim]
     *         - value: [batch, num_kv_heads, seq_len, head_dim]
     */
    std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>>
    build_qkv_projections(
        const ov::Output<ov::Node>& hidden_states,
        const std::string& layer_prefix,
        const std::unordered_map<std::string, ov::Tensor>& weights);

    /**
     * @brief Splits and transposes tensor for multi-head attention.
     * 
     * Transforms input from [batch, seq_len, num_heads * head_dim] to
     * [batch, num_heads, seq_len, head_dim] through:
     * 1. Reshape to [batch, seq_len, num_heads, head_dim]
     * 2. Transpose to [batch, num_heads, seq_len, head_dim]
     * 
     * @param x Input tensor to split
     * @param num_heads Number of attention heads
     * @param head_dim Dimension of each head
     * @return Transposed tensor of shape [batch, num_heads, seq_len, head_dim]
     */
    ov::Output<ov::Node> split_heads(
        const ov::Output<ov::Node>& x,
        int num_heads,
        int head_dim);

    /**
     * @brief Builds sliding window attention mask.
     * 
     * Creates a mask that restricts attention to a local window around each position.
     * The mask allows attending to positions within window_size on both sides.
     * 
     * If window_size is None or < 0, returns the original attention_mask unchanged.
     * 
     * The sliding window mask is combined with the causal mask to ensure:
     * 1. Causal constraint: position i can only attend to positions <= i
     * 2. Window constraint: position i can only attend to positions in [i-window_size, i]
     * 
     * Reference: modeling_qwen3_moe.py line 508 (create_sliding_window_causal_mask)
     * 
     * @param attention_mask Base attention mask (typically causal mask)
     * @param key_states Key tensor to determine sequence length
     * @param window_size Size of the sliding window (positions on each side)
     * @return Combined attention mask with sliding window applied
     */
    ov::Output<ov::Node> build_sliding_window_mask(
        const ov::Output<ov::Node>& attention_mask,
        const ov::Output<ov::Node>& key_states,
        int window_size);

    /**
     * @brief Computes scaled dot-product attention.
     * 
     * Implements the core attention mechanism:
     * 1. Expand KV heads if using GQA (num_kv_heads < num_heads)
     * 2. Compute attention scores: Q @ K^T
     * 3. Scale by 1/sqrt(head_dim)
     * 4. Apply attention mask (additive mask with large negative values)
     * 5. Apply softmax to get attention weights
     * 6. Compute weighted sum: attention_weights @ V
     * 
     * Reference: modeling_qwen3_moe.py lines 103-126 (eager_attention_forward)
     * 
     * @param q Query tensor of shape [batch, num_heads, seq_len, head_dim]
     * @param k Key tensor of shape [batch, num_kv_heads, seq_len_k, head_dim]
     * @param v Value tensor of shape [batch, num_kv_heads, seq_len_k, head_dim]
     * @param mask Attention mask (additive, with -inf for masked positions)
     * @param scaling Scaling factor (typically 1/sqrt(head_dim))
     * @return Attention output of shape [batch, num_heads, seq_len, head_dim]
     */
    ov::Output<ov::Node> compute_attention(
        const ov::Output<ov::Node>& q,
        const ov::Output<ov::Node>& k,
        const ov::Output<ov::Node>& v,
        const ov::Output<ov::Node>& mask,
        float scaling);

private:
    AttentionConfig config_;                                    ///< Attention configuration
    std::shared_ptr<Qwen3MoeRMSNormBuilder> norm_builder_;     ///< RMS norm builder for Q/K norm
    std::shared_ptr<Qwen3MoeRotaryEmbeddingBuilder> rope_builder_; ///< RoPE builder for position encoding

    /**
     * @brief Expands KV heads to match query heads for Grouped Query Attention.
     * 
     * Repeats each KV head n_rep times where n_rep = num_heads / num_kv_heads.
     * This is used when num_kv_heads < num_heads (GQA).
     * 
     * Reference: modeling_qwen3_moe.py lines 91-100 (repeat_kv)
     * 
     * @param kv Input KV tensor of shape [batch, num_kv_heads, seq_len, head_dim]
     * @param n_rep Number of times to repeat each KV head
     * @return Expanded tensor of shape [batch, num_heads, seq_len, head_dim]
     */
    ov::Output<ov::Node> repeat_kv(
        const ov::Output<ov::Node>& kv,
        int n_rep);

    /**
     * @brief Loads linear projection weight and bias from weights map.
     * 
     * @param key Weight key (e.g., "model.layers.0.self_attn.q_proj")
     * @param weights Map containing all model weight tensors
     * @return Pair of (weight_node, bias_node), bias_node may be nullptr if no bias
     */
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> load_projection_weights(
        const std::string& key,
        const std::unordered_map<std::string, ov::Tensor>& weights);

    /**
     * @brief Builds linear projection operation (MatMul + optional bias).
     * 
     * @param input Input tensor
     * @param weight Weight tensor
     * @param bias Optional bias tensor (may be nullptr)
     * @return Output of linear projection
     */
    ov::Output<ov::Node> build_linear(
        const ov::Output<ov::Node>& input,
        const ov::Output<ov::Node>& weight,
        const ov::Output<ov::Node>& bias);
};

} // namespace genai
} // namespace ov