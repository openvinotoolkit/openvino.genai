// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>
#include <openvino/openvino.hpp>

namespace ov {
namespace genai {

/**
 * @brief Configuration structure for Rotary Position Embeddings (RoPE).
 * 
 * Contains parameters needed to compute rotary position embeddings
 * according to the RoFormer paper (https://arxiv.org/abs/2104.09864).
 */
struct RoPEConfig {
    int64_t max_position_embeddings;  ///< Maximum sequence length supported
    float rope_theta;                  ///< Base value for frequency computation (default: 10000.0)
    int64_t head_dim;                  ///< Dimension of each attention head
    
    /**
     * @brief Constructs RoPEConfig with default values.
     */
    RoPEConfig(int64_t max_pos = 32768, float theta = 10000.0f, int64_t h_dim = 128)
        : max_position_embeddings(max_pos), rope_theta(theta), head_dim(h_dim) {}
};

/**
 * @brief Builder class for Rotary Position Embedding operations.
 * 
 * This class provides methods to construct RoPE computation graphs using OpenVINO operators.
 * RoPE applies rotational transformations to query and key tensors based on their positions,
 * allowing the model to encode relative positional information.
 * 
 * The rotation is applied using the formula:
 *   q_rotated = q * cos(m*theta) + rotate_half(q) * sin(m*theta)
 *   k_rotated = k * cos(m*theta) + rotate_half(k) * sin(m*theta)
 * 
 * where:
 *   - m is the position index
 *   - theta is computed from rope_theta and head dimension
 *   - rotate_half concatenates (-x[..., head_dim/2:], x[..., :head_dim/2])
 * 
 * Reference implementation: modeling_qwen3_moe.py lines 56-88, 395-457
 */
class Qwen3MoeRotaryEmbeddingBuilder {
public:
    /**
     * @brief Constructs a new Qwen3MoeRotaryEmbeddingBuilder object.
     * 
     * @param config RoPE configuration parameters
     */
    explicit Qwen3MoeRotaryEmbeddingBuilder(const RoPEConfig& config);

    /**
     * @brief Builds the inverse frequency constants for RoPE computation.
     * 
     * Computes inv_freq = 1.0 / (rope_theta ^ (arange(0, head_dim, 2) / head_dim))
     * This creates the base frequencies used to generate position-dependent rotations.
     * 
     * The result is cached to avoid recomputation across multiple calls.
     * 
     * @return ov::Output<ov::Node> Inverse frequency tensor of shape [head_dim/2]
     */
    ov::Output<ov::Node> build_rope_constants();

    /**
     * @brief Builds position embeddings (cos and sin) for given position IDs.
     * 
     * Computes the cosine and sine embeddings for rotary position encoding:
     * 1. Expands position_ids to [batch, 1, seq_len]
     * 2. Broadcasts inv_freq to match batch dimension
     * 3. Computes frequencies via matrix multiplication
     * 4. Transposes and concatenates to get full embedding dimension
     * 5. Computes cos and sin of the frequencies
     * 
     * @param position_ids Position indices tensor of shape [batch, seq_len]
     * @param batch_dim Batch dimension size as a scalar tensor
     * @return std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> Pair of (cos, sin) embeddings,
     *         each of shape [batch, seq_len, head_dim]
     */
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_position_embeddings(
        const ov::Output<ov::Node>& position_ids,
        const ov::Output<ov::Node>& batch_dim);

    /**
     * @brief Applies rotary position embeddings to query and key tensors.
     * 
     * Implements the core RoPE transformation:
     *   q_embed = (q * cos) + (rotate_half(q) * sin)
     *   k_embed = (k * cos) + (rotate_half(k) * sin)
     * 
     * The cos and sin tensors are unsqueezed to add a dimension for num_heads broadcasting.
     * 
     * @param q Query tensor of shape [batch, num_heads, seq_len, head_dim]
     * @param k Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
     * @param cos Cosine embeddings of shape [batch, seq_len, head_dim]
     * @param sin Sine embeddings of shape [batch, seq_len, head_dim]
     * @param hidden_dim Hidden dimension axis for slicing operations
     * @return std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> Pair of (q_embed, k_embed)
     *         with rotary embeddings applied
     */
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> apply_rotary_pos_emb(
        const ov::Output<ov::Node>& q,
        const ov::Output<ov::Node>& k,
        const ov::Output<ov::Node>& cos,
        const ov::Output<ov::Node>& sin,
        const ov::Output<ov::Node>& hidden_dim);

    /**
     * @brief Rotates half of the hidden dimensions of the input tensor.
     * 
     * Implements the rotate_half operation used in RoPE:
     *   rotate_half(x) = concatenate([-x[..., head_dim/2:], x[..., :head_dim/2]], dim=-1)
     * 
     * This operation is essential for applying the rotary transformation, as it creates
     * the orthogonal component needed for 2D rotation in the complex plane.
     * 
     * @param x Input tensor to rotate
     * @param head_size Size of the head dimension (must be even)
     * @param axis Axis along which to perform slicing (typically the last dimension)
     * @return ov::Output<ov::Node> Rotated tensor with same shape as input
     */
    ov::Output<ov::Node> rotate_half(
        const ov::Output<ov::Node>& x,
        int64_t head_size,
        const ov::Output<ov::Node>& axis);

private:
    RoPEConfig config_;                           ///< RoPE configuration parameters
    ov::Output<ov::Node> cached_rope_constants_;  ///< Cached inverse frequency constants
    bool constants_built_;                        ///< Flag indicating if constants are cached

    /**
     * @brief Helper to create constant scalar tensors.
     * 
     * @param value Scalar value
     * @param dtype Element type
     * @return ov::Output<ov::Node> Constant node
     */
    template<typename T>
    ov::Output<ov::Node> make_constant(T value, ov::element::Type dtype = ov::element::i64) {
        return std::make_shared<ov::op::v0::Constant>(dtype, ov::Shape{}, value);
    }

    /**
     * @brief Helper to create constant vector tensors.
     * 
     * @param values Vector of values
     * @param dtype Element type
     * @return ov::Output<ov::Node> Constant node
     */
    template<typename T>
    ov::Output<ov::Node> make_constant_vector(const std::vector<T>& values, ov::element::Type dtype = ov::element::i64) {
        return std::make_shared<ov::op::v0::Constant>(dtype, ov::Shape{values.size()}, values);
    }
};

} // namespace genai
} // namespace ov