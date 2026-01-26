// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_attention.hpp"
#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include "openvino/op/util/variable.hpp"
#include <stdexcept>
#include <cmath>

using namespace ov;
using namespace ov::op;

namespace ov {
namespace genai {

// Constant for masked positions in attention mask (large negative value)
constexpr float ATTENTION_MASK_VALUE = -65504.0f;

Qwen3MoeAttentionBuilder::Qwen3MoeAttentionBuilder(
    const AttentionConfig& config,
    std::shared_ptr<Qwen3MoeRMSNormBuilder> norm_builder,
    std::shared_ptr<Qwen3MoeRotaryEmbeddingBuilder> rope_builder)
    : config_(config), norm_builder_(norm_builder), rope_builder_(rope_builder) {
    
    // Validate configuration
    if (config_.num_heads <= 0) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder: num_heads must be positive");
    }
    
    if (config_.num_kv_heads <= 0) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder: num_kv_heads must be positive");
    }
    
    if (config_.head_dim <= 0) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder: head_dim must be positive");
    }
    
    if (config_.num_heads % config_.num_kv_heads != 0) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder: num_heads must be divisible by num_kv_heads");
    }
    
    if (!norm_builder_) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder: norm_builder cannot be null");
    }
    
    if (!rope_builder_) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder: rope_builder cannot be null");
    }
}

ov::Output<ov::Node> Qwen3MoeAttentionBuilder::split_heads(
    const ov::Output<ov::Node>& x,
    int num_heads,
    int head_dim) {
    
    // Input validation
    if (!x.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::split_heads: input x is null");
    }
    
    // Step 1: Get input shape using ShapeOf
    auto input_shape = std::make_shared<v3::ShapeOf>(x, element::i64);
    
    // Step 2: Create target shape [batch, seq_len, num_heads, head_dim]
    // We use Gather to extract batch and seq_len from input shape
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_dim = std::make_shared<v8::Gather>(input_shape, 
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 0), axis_0);
    
    auto seq_len_dim = std::make_shared<v8::Gather>(input_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 1), axis_0);
    
    auto num_heads_const = std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads);
    auto head_dim_const = std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim);
    
    // Concatenate to form target shape
    auto target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_dim, seq_len_dim, num_heads_const, head_dim_const}, 0);
    
    // Step 3: Reshape to [batch, seq_len, num_heads, head_dim]
    auto reshaped = std::make_shared<v1::Reshape>(x, target_shape, false);
    
    // Step 4: Transpose to [batch, num_heads, seq_len, head_dim]
    auto perm = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto transposed = std::make_shared<v1::Transpose>(reshaped, perm);
    
    return transposed;
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> 
Qwen3MoeAttentionBuilder::load_projection_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    // Load weight tensor
    std::string weight_key = key + ".weight";
    if (weights.count(weight_key) == 0) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::load_projection_weights: weight not found: " + weight_key);
    }
    
    auto weight_tensor = weights.at(weight_key);
    auto weight_const = std::make_shared<v0::Constant>(weight_tensor);
    auto weight_f32 = std::make_shared<v0::Convert>(weight_const, element::f32);
    
    // Load bias tensor if exists
    ov::Output<ov::Node> bias_node;
    std::string bias_key = key + ".bias";
    if (weights.count(bias_key) > 0) {
        auto bias_tensor = weights.at(bias_key);
        auto bias_const = std::make_shared<v0::Constant>(bias_tensor);
        bias_node = std::make_shared<v0::Convert>(bias_const, element::f32);
    }
    
    return {weight_f32, bias_node};
}

ov::Output<ov::Node> Qwen3MoeAttentionBuilder::build_linear(
    const ov::Output<ov::Node>& input,
    const ov::Output<ov::Node>& weight,
    const ov::Output<ov::Node>& bias) {
    
    // MatMul: input @ weight^T
    auto matmul = std::make_shared<v0::MatMul>(input, weight, false, true);
    
    // Add bias if exists
    if (bias.get_node()) {
        return std::make_shared<v1::Add>(matmul, bias, AutoBroadcastType::NUMPY);
    }
    
    return matmul;
}

std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>>
Qwen3MoeAttentionBuilder::build_qkv_projections(
    const ov::Output<ov::Node>& hidden_states,
    const std::string& layer_prefix,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::build_qkv_projections: hidden_states is null");
    }
    
    // Step 1: Q projection
    auto [q_weight, q_bias] = load_projection_weights(layer_prefix + ".q_proj", weights);
    auto q_proj = build_linear(hidden_states, q_weight, q_bias);
    
    // Step 2: Reshape Q to [batch, seq_len, num_heads, head_dim]
    auto input_shape = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_dim = std::make_shared<v8::Gather>(input_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 0), axis_0);
    auto seq_len_dim = std::make_shared<v8::Gather>(input_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 1), axis_0);
    
    auto q_num_heads = std::make_shared<v0::Constant>(element::i64, Shape{1}, config_.num_heads);
    auto q_head_dim = std::make_shared<v0::Constant>(element::i64, Shape{1}, config_.head_dim);
    auto q_target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_dim, seq_len_dim, q_num_heads, q_head_dim}, 0);
    auto q_reshaped = std::make_shared<v1::Reshape>(q_proj, q_target_shape, false);
    
    // Step 3: Apply RMS norm to Q on head dimension (last dimension)
    // Note: We need to apply norm on the last dimension only
    auto q_normalized = norm_builder_->build(q_reshaped, layer_prefix + ".q_norm", weights, config_.rms_norm_eps);
    
    // Step 4: Transpose Q to [batch, num_heads, seq_len, head_dim]
    auto q_perm = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto q_transposed = std::make_shared<v1::Transpose>(q_normalized, q_perm);
    
    // Step 5: K projection
    auto [k_weight, k_bias] = load_projection_weights(layer_prefix + ".k_proj", weights);
    auto k_proj = build_linear(hidden_states, k_weight, k_bias);
    
    // Step 6: Reshape K to [batch, seq_len, num_kv_heads, head_dim]
    auto k_num_heads = std::make_shared<v0::Constant>(element::i64, Shape{1}, config_.num_kv_heads);
    auto k_target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_dim, seq_len_dim, k_num_heads, q_head_dim}, 0);
    auto k_reshaped = std::make_shared<v1::Reshape>(k_proj, k_target_shape, false);
    
    // Step 7: Apply RMS norm to K on head dimension
    auto k_normalized = norm_builder_->build(k_reshaped, layer_prefix + ".k_norm", weights, config_.rms_norm_eps);
    
    // Step 8: Transpose K to [batch, num_kv_heads, seq_len, head_dim]
    auto k_perm = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto k_transposed = std::make_shared<v1::Transpose>(k_normalized, k_perm);
    
    // Step 9: V projection
    auto [v_weight, v_bias] = load_projection_weights(layer_prefix + ".v_proj", weights);
    auto v_proj = build_linear(hidden_states, v_weight, v_bias);
    
    // Step 10: Reshape V to [batch, seq_len, num_kv_heads, head_dim]
    auto v_target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_dim, seq_len_dim, k_num_heads, q_head_dim}, 0);
    auto v_reshaped = std::make_shared<v1::Reshape>(v_proj, v_target_shape, false);
    
    // Step 11: Transpose V to [batch, num_kv_heads, seq_len, head_dim]
    // Note: V does NOT get normalized, only Q and K
    auto v_perm = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto v_transposed = std::make_shared<v1::Transpose>(v_reshaped, v_perm);
    
    return {q_transposed, k_transposed, v_transposed};
}

ov::Output<ov::Node> Qwen3MoeAttentionBuilder::build_sliding_window_mask(
    const ov::Output<ov::Node>& attention_mask,
    const ov::Output<ov::Node>& key_states,
    int window_size) {
    
    // If window_size is invalid, return original mask
    if (window_size < 0 || !config_.use_sliding_window) {
        return attention_mask;
    }
    
    // Input validation
    if (!attention_mask.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::build_sliding_window_mask: attention_mask is null");
    }
    
    if (!key_states.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::build_sliding_window_mask: key_states is null");
    }
    
    // Step 1: Get sequence length from key_states shape
    // key_states shape: [batch, num_kv_heads, seq_len, head_dim]
    auto key_shape = std::make_shared<v3::ShapeOf>(key_states, element::i64);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto seq_len = std::make_shared<v8::Gather>(key_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 2), axis_0);
    
    // Step 2: Create position ranges for query and key
    // query_positions: [0, 1, 2, ..., seq_len-1]
    auto start = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto step = std::make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto query_positions = std::make_shared<v4::Range>(start, seq_len, step, element::i64);
    
    // key_positions: same as query_positions
    auto key_positions = query_positions;
    
    // Step 3: Reshape for broadcasting
    // query_positions: [seq_len] -> [seq_len, 1]
    auto unsqueeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto query_pos_unsqueezed = std::make_shared<v0::Unsqueeze>(query_positions, unsqueeze_axis);
    
    // key_positions: [seq_len] -> [1, seq_len]
    auto unsqueeze_axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto key_pos_unsqueezed = std::make_shared<v0::Unsqueeze>(key_positions, unsqueeze_axis_0);
    
    // Step 4: Compute distance: query_pos - key_pos
    // Result shape: [seq_len, seq_len]
    auto distance = std::make_shared<v1::Subtract>(query_pos_unsqueezed, key_pos_unsqueezed);
    
    // Step 5: Create window condition: distance <= window_size AND distance >= 0
    // This ensures we only attend to positions within the window and not in the future
    auto window_size_const = std::make_shared<v0::Constant>(element::i64, Shape{}, window_size);
    auto zero_const = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    
    // distance <= window_size
    auto within_window = std::make_shared<v1::LessEqual>(distance, window_size_const);
    
    // distance >= 0 (causal constraint)
    auto causal_constraint = std::make_shared<v1::GreaterEqual>(distance, zero_const);
    
    // Combine: within_window AND causal_constraint
    auto window_mask = std::make_shared<v1::LogicalAnd>(within_window, causal_constraint);
    
    // Step 6: Convert boolean mask to additive mask
    // True -> 0.0, False -> ATTENTION_MASK_VALUE (large negative value for masking)
    auto zero_f32 = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.0f);
    auto neg_inf = std::make_shared<v0::Constant>(element::f32, Shape{}, ATTENTION_MASK_VALUE);
    auto additive_window_mask = std::make_shared<v1::Select>(window_mask, zero_f32, neg_inf);
    
    // Step 7: Combine with original attention mask
    // Both masks are additive, so we can add them
    auto combined_mask = std::make_shared<v1::Add>(attention_mask, additive_window_mask, AutoBroadcastType::NUMPY);
    
    return combined_mask;
}

ov::Output<ov::Node> Qwen3MoeAttentionBuilder::repeat_kv(
    const ov::Output<ov::Node>& kv,
    int n_rep) {
    
    // If n_rep is 1, no need to repeat
    if (n_rep == 1) {
        return kv;
    }
    
    // Input validation
    if (!kv.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::repeat_kv: kv is null");
    }
    
    if (n_rep <= 0) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::repeat_kv: n_rep must be positive");
    }
    
    // kv shape: [batch, num_kv_heads, seq_len, head_dim]
    // Step 1: Unsqueeze to add repeat dimension
    // [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_kv_heads, 1, seq_len, head_dim]
    auto unsqueeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto kv_unsqueezed = std::make_shared<v0::Unsqueeze>(kv, unsqueeze_axis);
    
    // Step 2: Create tile repeats: [1, 1, n_rep, 1, 1]
    auto repeats = std::make_shared<v0::Constant>(element::i64, Shape{5}, 
        std::vector<int64_t>{1, 1, n_rep, 1, 1});
    auto kv_tiled = std::make_shared<v0::Tile>(kv_unsqueezed, repeats);
    
    // Step 3: Reshape to merge num_kv_heads and n_rep dimensions
    // [batch, num_kv_heads, n_rep, seq_len, head_dim] -> [batch, num_kv_heads * n_rep, seq_len, head_dim]
    auto kv_shape = std::make_shared<v3::ShapeOf>(kv, element::i64);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    
    auto batch_dim = std::make_shared<v8::Gather>(kv_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 0), axis_0);
    auto num_kv_heads = std::make_shared<v8::Gather>(kv_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 1), axis_0);
    auto seq_len = std::make_shared<v8::Gather>(kv_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 2), axis_0);
    auto head_dim = std::make_shared<v8::Gather>(kv_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 3), axis_0);
    
    // num_heads = num_kv_heads * n_rep
    auto n_rep_const = std::make_shared<v0::Constant>(element::i64, Shape{1}, n_rep);
    auto num_heads = std::make_shared<v1::Multiply>(num_kv_heads, n_rep_const);
    
    auto target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_dim, num_heads, seq_len, head_dim}, 0);
    
    auto kv_reshaped = std::make_shared<v1::Reshape>(kv_tiled, target_shape, false);
    
    return kv_reshaped;
}

ov::Output<ov::Node> Qwen3MoeAttentionBuilder::compute_attention(
    const ov::Output<ov::Node>& q,
    const ov::Output<ov::Node>& k,
    const ov::Output<ov::Node>& v,
    const ov::Output<ov::Node>& mask,
    float scaling) {
    
    // Input validation
    if (!q.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::compute_attention: q is null");
    }
    
    if (!k.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::compute_attention: k is null");
    }
    
    if (!v.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::compute_attention: v is null");
    }
    
    // Step 1: Repeat K/V if using GQA (num_kv_heads < num_heads)
    int n_rep = config_.num_heads / config_.num_kv_heads;
    auto k_expanded = repeat_kv(k, n_rep);
    auto v_expanded = repeat_kv(v, n_rep);
    
    // Step 2: Compute attention scores: Q @ K^T
    // q shape: [batch, num_heads, seq_len_q, head_dim]
    // k shape: [batch, num_heads, seq_len_k, head_dim]
    // scores shape: [batch, num_heads, seq_len_q, seq_len_k]
    auto scores = std::make_shared<v0::MatMul>(q, k_expanded, false, true);
    
    // Step 3: Scale by scaling factor (1/sqrt(head_dim))
    auto scaling_const = std::make_shared<v0::Constant>(element::f32, Shape{}, scaling);
    auto scaled_scores = std::make_shared<v1::Multiply>(scores, scaling_const);
    
    // Step 4: Apply attention mask (additive mask)
    ov::Output<ov::Node> masked_scores = scaled_scores;
    if (mask.get_node()) {
        masked_scores = std::make_shared<v1::Add>(scaled_scores, mask, AutoBroadcastType::NUMPY);
    }
    
    // Step 5: Apply softmax along last dimension
    auto softmax_output = std::make_shared<v8::Softmax>(masked_scores, -1);
    
    // Step 6: Apply dropout (only during training, but we build the graph for inference)
    // For inference, dropout is a no-op, so we skip it
    
    // Step 7: Compute attention output: softmax @ V
    // softmax shape: [batch, num_heads, seq_len_q, seq_len_k]
    // v shape: [batch, num_heads, seq_len_k, head_dim]
    // output shape: [batch, num_heads, seq_len_q, head_dim]
    auto attn_output = std::make_shared<v0::MatMul>(softmax_output, v_expanded);
    
    return attn_output;
}

std::pair<ov::Output<ov::Node>, ov::SinkVector> Qwen3MoeAttentionBuilder::build(
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& attention_mask,
    const ov::Output<ov::Node>& position_ids,
    const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& position_embeddings,
    const std::string& layer_prefix,
    const std::unordered_map<std::string, ov::Tensor>& weights) {
    
    // Input validation
    if (!hidden_states.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::build: hidden_states is null");
    }
    
    // Step 1: Build Q/K/V projections with normalization
    auto [q, k, v] = build_qkv_projections(hidden_states, layer_prefix, weights);
    
    // Step 2: Apply RoPE to Q and K
    auto [cos, sin] = position_embeddings;
    if (!cos.get_node() || !sin.get_node()) {
        throw std::runtime_error("Qwen3MoeAttentionBuilder::build: position_embeddings are null");
    }
    
    // Create hidden_dim constant for RoPE
    auto hidden_dim_const = std::make_shared<v0::Constant>(element::i64, Shape{}, config_.head_dim);
    auto [q_rot, k_rot] = rope_builder_->apply_rotary_pos_emb(q, k, cos, sin, hidden_dim_const);
    
    // Step 3: Handle KV cache
    ov::SinkVector sinks;
    
    // Get batch dimension from hidden_states
    auto input_shape = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);
    auto axis_0 = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto batch_dim = std::make_shared<v8::Gather>(input_shape,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 0), axis_0);
    
    // Create initial cache tensors (zeros)
    auto zero_const = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.0f);
    
    // K cache shape: [batch, num_kv_heads, 0, head_dim]
    auto k_cache_shape = std::make_shared<v0::Concat>(OutputVector{
        batch_dim,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, config_.num_kv_heads),
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
        std::make_shared<v0::Constant>(element::i64, Shape{1}, config_.head_dim)
    }, 0);
    auto k_cache_default = std::make_shared<v3::Broadcast>(zero_const, k_cache_shape);
    
    // V cache shape: same as K cache
    auto v_cache_default = std::make_shared<v3::Broadcast>(zero_const, k_cache_shape);
    
    // Create cache variables
    auto k_var_info = util::VariableInfo{
        PartialShape{-1, config_.num_kv_heads, -1, config_.head_dim},
        element::f32,
        layer_prefix + ".k_cache"
    };
    auto k_var = std::make_shared<util::Variable>(k_var_info);
    
    auto v_var_info = util::VariableInfo{
        PartialShape{-1, config_.num_kv_heads, -1, config_.head_dim},
        element::f32,
        layer_prefix + ".v_cache"
    };
    auto v_var = std::make_shared<util::Variable>(v_var_info);
    
    // ReadValue operations
    auto k_cache_read = std::make_shared<v6::ReadValue>(k_cache_default, k_var);
    auto v_cache_read = std::make_shared<v6::ReadValue>(v_cache_default, v_var);
    
    // Concatenate with new K/V along sequence dimension (axis=2)
    auto concat_axis = std::make_shared<v0::Constant>(element::i64, Shape{}, 2);
    auto k_updated = std::make_shared<v0::Concat>(OutputVector{k_cache_read, k_rot}, 2);
    auto v_updated = std::make_shared<v0::Concat>(OutputVector{v_cache_read, v}, 2);
    
    // Assign operations
    auto k_assign = std::make_shared<v6::Assign>(k_updated, k_var);
    auto v_assign = std::make_shared<v6::Assign>(v_updated, v_var);
    
    // Add assigns to sinks
    sinks.push_back(k_assign);
    sinks.push_back(v_assign);
    
    // Step 4: Build sliding window mask
    auto final_mask = build_sliding_window_mask(attention_mask, k_updated, config_.sliding_window);
    
    // Step 5: Compute attention
    // Scaling factor: 1.0 / sqrt(head_dim)
    float scaling = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
    auto attn_output = compute_attention(q_rot, k_updated, v_updated, final_mask, scaling);
    
    // Step 6: Output projection
    // Transpose back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    auto perm_back = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto attn_transposed = std::make_shared<v1::Transpose>(attn_output, perm_back);
    
    // Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_size]
    auto attn_shape = std::make_shared<v3::ShapeOf>(hidden_states, element::i64);
    auto attn_reshaped = std::make_shared<v1::Reshape>(attn_transposed, attn_shape, false);
    
    // Output projection
    auto [o_weight, o_bias] = load_projection_weights(layer_prefix + ".o_proj", weights);
    auto o_proj = build_linear(attn_reshaped, o_weight, o_bias);
    
    return {o_proj, sinks};
}

} // namespace genai
} // namespace ov