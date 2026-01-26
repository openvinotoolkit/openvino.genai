// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen3_moe_rope.hpp"
#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"
#include <stdexcept>
#include <cmath>

using namespace ov;
using namespace ov::op;

namespace ov {
namespace genai {

Qwen3MoeRotaryEmbeddingBuilder::Qwen3MoeRotaryEmbeddingBuilder(const RoPEConfig& config)
    : config_(config), constants_built_(false) {
    
    // Validate configuration
    if (config_.head_dim <= 0) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder: head_dim must be positive");
    }
    
    if (config_.head_dim % 2 != 0) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder: head_dim must be even");
    }
    
    if (config_.rope_theta <= 0.0f) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder: rope_theta must be positive");
    }
    
    if (config_.max_position_embeddings <= 0) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder: max_position_embeddings must be positive");
    }
}

ov::Output<ov::Node> Qwen3MoeRotaryEmbeddingBuilder::build_rope_constants() {
    // Return cached constants if already built
    if (constants_built_ && cached_rope_constants_.get_node()) {
        return cached_rope_constants_;
    }
    
    // Step 1: Create dimension range: arange(0, head_dim, 2)
    // This creates [0, 2, 4, 6, ..., head_dim-2]
    auto start = make_constant<int64_t>(0);
    auto stop = make_constant<int64_t>(config_.head_dim);
    auto step = make_constant<int64_t>(2);
    auto range_node = std::make_shared<v4::Range>(start, stop, step, element::i64);
    
    // Step 2: Convert to float for division
    auto range_f32 = std::make_shared<v0::Convert>(range_node, element::f32);
    
    // Step 3: Divide by head_dim to get ratio: arange(0, head_dim, 2) / head_dim
    auto constant_head_dim = make_constant<float>(static_cast<float>(config_.head_dim), element::f32);
    auto ratio = std::make_shared<v1::Divide>(range_f32, constant_head_dim);
    
    // Step 4: Negate the ratio: -(arange(0, head_dim, 2) / head_dim)
    auto constant_neg_one = make_constant<float>(-1.0f, element::f32);
    auto negated_ratio = std::make_shared<v1::Multiply>(ratio, constant_neg_one);
    
    // Step 5: Compute base^(-ratio): rope_theta ^ (-(arange(0, head_dim, 2) / head_dim))
    // This is equivalent to: 1.0 / (rope_theta ^ (arange(0, head_dim, 2) / head_dim))
    auto constant_rope_theta = make_constant<float>(config_.rope_theta, element::f32);
    auto inv_freq = std::make_shared<v1::Power>(constant_rope_theta, negated_ratio);
    
    // Cache the result
    cached_rope_constants_ = inv_freq;
    constants_built_ = true;
    
    return inv_freq;
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> 
Qwen3MoeRotaryEmbeddingBuilder::build_position_embeddings(
    const ov::Output<ov::Node>& position_ids,
    const ov::Output<ov::Node>& batch_dim) {
    
    // Input validation
    if (!position_ids.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::build_position_embeddings: position_ids is null");
    }
    
    if (!batch_dim.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::build_position_embeddings: batch_dim is null");
    }
    
    // Get or build rope constants (inv_freq)
    auto inv_freq = build_rope_constants();
    
    // Step 1: Expand position_ids from [batch, seq_len] to [batch, 1, seq_len]
    auto axis_1 = make_constant<int64_t>(1);
    auto position_expanded = std::make_shared<v0::Unsqueeze>(position_ids, axis_1);
    
    // Step 2: Convert position_ids to float
    auto position_f32 = std::make_shared<v0::Convert>(position_expanded, element::f32);
    
    // Step 3: Create target shape for broadcasting inv_freq: [batch, 1, 1]
    auto const_1 = make_constant<int64_t>(1);
    auto target_shape = std::make_shared<v0::Concat>(
        OutputVector{batch_dim, const_1, const_1}, 0);
    
    // Step 4: Broadcast inv_freq to match batch dimension
    // inv_freq shape: [head_dim/2] -> [batch, 1, head_dim/2]
    auto inv_freq_expanded = std::make_shared<v3::Broadcast>(
        inv_freq, target_shape, BroadcastType::BIDIRECTIONAL);
    
    // Step 5: MatMul to compute frequencies
    // [batch, 1, head_dim/2] @ [batch, 1, seq_len] -> [batch, head_dim/2, seq_len]
    auto freqs = std::make_shared<v0::MatMul>(
        inv_freq_expanded, position_f32, false, false);
    
    // Step 6: Transpose to [batch, seq_len, head_dim/2]
    auto perm = make_constant_vector<int32_t>({0, 2, 1}, element::i32);
    auto freqs_transposed = std::make_shared<v1::Transpose>(freqs, perm);
    
    // Step 7: Concatenate along last dimension to get full head_dim
    // [batch, seq_len, head_dim/2] + [batch, seq_len, head_dim/2] -> [batch, seq_len, head_dim]
    auto emb = std::make_shared<v0::Concat>(
        OutputVector{freqs_transposed, freqs_transposed}, -1);
    
    // Step 8: Compute cos and sin
    auto cos = std::make_shared<ov::opset13::Cos>(emb);
    auto sin = std::make_shared<ov::opset13::Sin>(emb);
    
    return {cos, sin};
}

ov::Output<ov::Node> Qwen3MoeRotaryEmbeddingBuilder::rotate_half(
    const ov::Output<ov::Node>& x,
    int64_t head_size,
    const ov::Output<ov::Node>& axis) {
    
    // Input validation
    if (!x.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::rotate_half: input x is null");
    }
    
    if (!axis.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::rotate_half: axis is null");
    }
    
    if (head_size % 2 != 0) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::rotate_half: head_size must be even");
    }
    
    // Step 1: Define slice parameters for second half [head_size/2 : head_size]
    auto start_half = make_constant_vector<int64_t>({head_size / 2});
    auto end = make_constant_vector<int64_t>({head_size});
    auto step = make_constant_vector<int64_t>({1});
    
    // Step 2: Slice second half of the tensor
    auto second_half = std::make_shared<ov::opset13::Slice>(x, start_half, end, step, axis);
    
    // Step 3: Negate the second half
    auto constant_neg_one = make_constant<float>(-1.0f, element::f32);
    auto negated_second_half = std::make_shared<v1::Multiply>(second_half, constant_neg_one);
    
    // Step 4: Define slice parameters for first half [0 : head_size/2]
    auto start_zero = make_constant_vector<int64_t>({0});
    auto end_half = make_constant_vector<int64_t>({head_size / 2});
    
    // Step 5: Slice first half of the tensor
    auto first_half = std::make_shared<ov::opset13::Slice>(x, start_zero, end_half, step, axis);
    
    // Step 6: Concatenate [-second_half, first_half] along last dimension
    auto rotated = std::make_shared<v0::Concat>(
        OutputVector{negated_second_half, first_half}, -1);
    
    return rotated;
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> 
Qwen3MoeRotaryEmbeddingBuilder::apply_rotary_pos_emb(
    const ov::Output<ov::Node>& q,
    const ov::Output<ov::Node>& k,
    const ov::Output<ov::Node>& cos,
    const ov::Output<ov::Node>& sin,
    const ov::Output<ov::Node>& hidden_dim) {
    
    // Input validation
    if (!q.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::apply_rotary_pos_emb: q is null");
    }
    
    if (!k.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::apply_rotary_pos_emb: k is null");
    }
    
    if (!cos.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::apply_rotary_pos_emb: cos is null");
    }
    
    if (!sin.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::apply_rotary_pos_emb: sin is null");
    }
    
    if (!hidden_dim.get_node()) {
        throw std::runtime_error("Qwen3MoeRotaryEmbeddingBuilder::apply_rotary_pos_emb: hidden_dim is null");
    }
    
    // Step 1: Unsqueeze cos and sin to add dimension for num_heads broadcasting
    // cos/sin shape: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    auto axis_1 = make_constant<int64_t>(1);
    auto cos_unsqueezed = std::make_shared<v0::Unsqueeze>(cos, axis_1);
    auto sin_unsqueezed = std::make_shared<v0::Unsqueeze>(sin, axis_1);
    
    // Step 2: Apply rotation to query
    // q_embed = (q * cos) + (rotate_half(q) * sin)
    auto q_cos = std::make_shared<v1::Multiply>(q, cos_unsqueezed, AutoBroadcastType::NUMPY);
    auto q_rotated_half = rotate_half(q, config_.head_dim, hidden_dim);
    auto q_sin = std::make_shared<v1::Multiply>(q_rotated_half, sin_unsqueezed, AutoBroadcastType::NUMPY);
    auto q_embed = std::make_shared<v1::Add>(q_cos, q_sin, AutoBroadcastType::NUMPY);
    
    // Step 3: Apply rotation to key (same process as query)
    // k_embed = (k * cos) + (rotate_half(k) * sin)
    auto k_cos = std::make_shared<v1::Multiply>(k, cos_unsqueezed, AutoBroadcastType::NUMPY);
    auto k_rotated_half = rotate_half(k, config_.head_dim, hidden_dim);
    auto k_sin = std::make_shared<v1::Multiply>(k_rotated_half, sin_unsqueezed, AutoBroadcastType::NUMPY);
    auto k_embed = std::make_shared<v1::Add>(k_cos, k_sin, AutoBroadcastType::NUMPY);
    
    return {q_embed, k_embed};
}

} // namespace genai
} // namespace ov