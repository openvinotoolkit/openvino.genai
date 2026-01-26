// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <openvino/openvino.hpp>
#include "qwen3_moe_config.hpp"
#include "layer_selection_strategy.hpp"
#include "qwen3_moe_norm.hpp"
#include "qwen3_moe_rope.hpp"
#include "qwen3_moe_attention.hpp"
#include "qwen3_moe_mlp.hpp"
#include "qwen3_moe_router.hpp"
#include "qwen3_moe_experts.hpp"
#include "qwen3_moe_sparse_block.hpp"

namespace ov {
namespace genai {

/**
 * @brief Main graph builder class for Qwen3-MoE model.
 * 
 * This class orchestrates the construction of the complete Qwen3-MoE computation graph
 * using OpenVINO operator APIs. The model structure follows:
 * 
 * Input Flow:
 *   input_ids -> embeddings -> N decoder layers -> final_norm -> lm_head -> logits
 * 
 * Each Decoder Layer:
 *   1. Input RMS normalization
 *   2. Multi-head attention with Q/K normalization and RoPE
 *   3. First residual connection
 *   4. Post-attention RMS normalization
 *   5. MLP or Sparse MoE block (based on layer configuration)
 *   6. Second residual connection
 * 
 * Key Features:
 * - Conditional layer selection between standard MLP and sparse MoE blocks
 * - Sliding window attention support
 * - Q/K normalization on head dimension
 * - Rotary position embeddings (RoPE)
 * - KV cache management through ReadValue/Assign operations
 * - 3D expert weight tensor management for MoE layers
 * 
 * The builder uses specialized component builders for each major component:
 * - Qwen3MoeRMSNormBuilder: RMS normalization layers
 * - Qwen3MoeRotaryEmbeddingBuilder: Rotary position embeddings
 * - Qwen3MoeAttentionBuilder: Multi-head attention mechanism
 * - Qwen3MoeMLPBuilder: Standard MLP layers
 * - Qwen3MoeTopKRouterBuilder: Expert routing for MoE
 * - Qwen3MoeExpertsBuilder: Expert computation for MoE
 * - Qwen3MoeSparseMoeBlockBuilder: Complete MoE block orchestration
 * 
 * Reference: modeling_qwen3_moe.py for PyTorch implementation details
 */
class Qwen3MoeGraphBuilder {
public:
    /**
     * @brief Constructs a new Qwen3MoeGraphBuilder object.
     * 
     * Initializes all component builders and prepares for graph construction.
     * 
     * @param config Qwen3-MoE model configuration containing all architecture parameters
     */
    explicit Qwen3MoeGraphBuilder(const Qwen3MoeConfig& config);

    /**
     * @brief Builds the complete Qwen3-MoE computation graph.
     * 
     * Constructs the full model graph including:
     * 1. Input parameters (input_ids, attention_mask, position_ids)
     * 2. Token embedding layer
     * 3. N decoder layers with conditional MLP/MoE selection
     * 4. Final RMS normalization
     * 5. LM head projection (with optional weight tying)
     * 6. Output result node
     * 
     * The method also handles:
     * - RoPE constant initialization
     * - Position embeddings computation
     * - KV cache operations (ReadValue/Assign) collected in sinks
     * - Attention mask preparation
     * 
     * @return std::shared_ptr<ov::Model> Complete OpenVINO model ready for compilation
     * @throws std::runtime_error if configuration is invalid or weights are missing
     */
    std::shared_ptr<ov::Model> build_graph();

    /**
     * @brief Sets the model weights for graph construction.
     * 
     * Loads pretrained weights that will be used during graph construction.
     * Weights should be provided as a map from weight key to tensor.
     * 
     * Weight keys follow the pattern:
     * - "model.embed_tokens.weight": embedding weights
     * - "model.layers.{i}.{component}.{param}": layer weights
     * - "model.norm.weight": final normalization weights
     * - "lm_head.weight": output projection weights (optional if weight tying)
     * 
     * @param weights Map from weight key to OpenVINO tensor
     */
    void set_weights(const std::unordered_map<std::string, ov::Tensor>& weights);

private:
    const Qwen3MoeConfig& config_;  ///< Model configuration reference

    // Component builders
    std::shared_ptr<LayerSelectionStrategy> layer_selector_;  ///< Determines layer types (MLP vs MoE)
    std::shared_ptr<Qwen3MoeRMSNormBuilder> norm_builder_;  ///< RMS normalization builder
    std::shared_ptr<Qwen3MoeRotaryEmbeddingBuilder> rope_builder_;  ///< RoPE builder
    std::shared_ptr<Qwen3MoeAttentionBuilder> attention_builder_;  ///< Attention builder
    std::shared_ptr<Qwen3MoeMLPBuilder> mlp_builder_;  ///< Standard MLP builder
    std::shared_ptr<Qwen3MoeTopKRouterBuilder> router_builder_;  ///< MoE router builder
    std::shared_ptr<Qwen3MoeExpertsBuilder> experts_builder_;  ///< MoE experts builder
    std::shared_ptr<Qwen3MoeSparseMoeBlockBuilder> moe_builder_;  ///< MoE block builder

    // Model weights
    std::unordered_map<std::string, ov::Tensor> weights_;  ///< Loaded model weights

    /**
     * @brief Initializes all component builders.
     * 
     * Creates and configures all specialized builders needed for graph construction.
     * Called during constructor initialization.
     */
    void initialize_builders();

    /**
     * @brief Creates input parameter nodes for the model.
     * 
     * Creates three input parameters:
     * 1. input_ids: [batch, seq_len] - Token IDs (i64)
     * 2. attention_mask: [batch, seq_len] - Attention mask (i64)
     * 3. position_ids: [batch, seq_len] - Position indices (i64)
     * 
     * All parameters use dynamic shapes (-1) for batch and sequence dimensions.
     * 
     * @return Vector of input parameter nodes
     */
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> create_input_parameters();

    /**
     * @brief Builds the token embedding layer.
     * 
     * Constructs the embedding lookup operation:
     * 1. Loads embedding weights from "model.embed_tokens.weight"
     * 2. Converts to f32 for computation
     * 3. Converts input_ids to i32 for indexing
     * 4. Performs Gather operation along axis 0
     * 
     * Returns both the embedded tokens and the embedding weights tensor.
     * The embedding weights are returned separately to support weight tying
     * with the LM head if lm_head.weight is not present.
     * 
     * @param input_ids Input token IDs of shape [batch, seq_len]
     * @return Pair of (embeddings, embedding_weights):
     *         - embeddings: [batch, seq_len, hidden_size]
     *         - embedding_weights: [vocab_size, hidden_size] for potential weight tying
     * @throws std::runtime_error if embedding weights are missing
     */
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_embedding(
        const ov::Output<ov::Node>& input_ids);

    /**
     * @brief Builds a single decoder layer.
     * 
     * Constructs one complete decoder layer with the following structure:
     * 1. Input layer normalization
     * 2. Self-attention with Q/K normalization and RoPE
     * 3. First residual connection
     * 4. Post-attention layer normalization
     * 5. MLP or MoE block (based on layer_selector_)
     * 6. Second residual connection
     * 
     * The layer type (MLP vs MoE) is determined by LayerSelectionStrategy based on:
     * - decoder_sparse_step configuration
     * - mlp_only_layers list
     * 
     * KV cache operations (ReadValue/Assign) from attention are collected in sinks.
     * 
     * @param layer_idx Layer index (0-based, range: [0, num_hidden_layers))
     * @param hidden_states Input hidden states of shape [batch, seq_len, hidden_size]
     * @param attention_mask Attention mask for masking padded tokens
     * @param position_ids Position indices for RoPE
     * @param position_embeddings Pair of (cos, sin) embeddings from RoPE builder
     * @param sinks Vector to collect KV cache Assign operations
     * @return Output hidden states of shape [batch, seq_len, hidden_size]
     * @throws std::runtime_error if layer weights are missing or layer_idx is invalid
     */
    ov::Output<ov::Node> build_decoder_layer(
        int layer_idx,
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& attention_mask,
        const ov::Output<ov::Node>& position_ids,
        const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& position_embeddings,
        ov::SinkVector& sinks);

    /**
     * @brief Builds the LM head output projection.
     * 
     * Constructs the final projection from hidden states to vocabulary logits:
     * 1. Checks for "lm_head.weight" in weights
     * 2. If present: loads and uses lm_head weights
     * 3. If absent: uses embedding weights (weight tying)
     * 4. Performs MatMul with transpose_b=true
     * 
     * Weight tying is a common technique where the embedding and output projection
     * share the same weights to reduce model parameters.
     * 
     * @param hidden_states Final hidden states of shape [batch, seq_len, hidden_size]
     * @param embeddings Embedding weights for potential weight tying [vocab_size, hidden_size]
     * @return Logits of shape [batch, seq_len, vocab_size]
     */
    ov::Output<ov::Node> build_lm_head(
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& embeddings);

    /**
     * @brief Gets the list of input parameters.
     * 
     * @return Vector of input parameter nodes
     */
    ov::ParameterVector get_model_inputs() const;

    /**
     * @brief Gets the list of output results.
     * 
     * @return Vector of output result nodes
     */
    ov::OutputVector get_model_outputs() const;
};

} // namespace genai
} // namespace ov