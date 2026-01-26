// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_graph_builder.hpp"
#include <stdexcept>
#include <sstream>

namespace ov {
namespace genai {

Qwen3MoeGraphBuilder::Qwen3MoeGraphBuilder(const Qwen3MoeConfig& config)
    : config_(config) {
    // Validate configuration
    config_.validate();
    
    // Initialize all component builders
    initialize_builders();
}

void Qwen3MoeGraphBuilder::initialize_builders() {
    // Create layer selection strategy
    layer_selector_ = std::make_shared<LayerSelectionStrategy>(config_);
    
    // Create normalization builder
    norm_builder_ = std::make_shared<Qwen3MoeRMSNormBuilder>();
    
    // Create RoPE builder with configuration
    rope_builder_ = std::make_shared<Qwen3MoeRotaryEmbeddingBuilder>(config_.get_rope_config());
    
    // Create attention builder with configuration and dependencies
    attention_builder_ = std::make_shared<Qwen3MoeAttentionBuilder>(
        config_.get_attention_config(),
        norm_builder_,
        rope_builder_
    );
    
    // Create MLP builder
    mlp_builder_ = std::make_shared<Qwen3MoeMLPBuilder>();
    
    // Create MoE component builders (using first MoE layer config as template)
    auto moe_config = config_.get_moe_layer_config(0);
    router_builder_ = std::make_shared<Qwen3MoeTopKRouterBuilder>(moe_config);
    experts_builder_ = std::make_shared<Qwen3MoeExpertsBuilder>(moe_config);
    
    // Create sparse MoE block builder
    moe_builder_ = std::make_shared<Qwen3MoeSparseMoeBlockBuilder>(
        moe_config,
        router_builder_,
        experts_builder_
    );
}

void Qwen3MoeGraphBuilder::set_weights(const std::unordered_map<std::string, ov::Tensor>& weights) {
    weights_ = weights;
}

std::vector<std::shared_ptr<ov::op::v0::Parameter>> Qwen3MoeGraphBuilder::create_input_parameters() {
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> params;
    
    // Create input_ids parameter: [batch, seq_len]
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, 
        ov::PartialShape{-1, -1}
    );
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});
    params.push_back(input_ids);
    
    // Create attention_mask parameter: [batch, seq_len]
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64,
        ov::PartialShape{-1, -1}
    );
    attention_mask->set_friendly_name("attention_mask");
    attention_mask->output(0).set_names({"attention_mask"});
    params.push_back(attention_mask);
    
    // Create position_ids parameter: [batch, seq_len]
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64,
        ov::PartialShape{-1, -1}
    );
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).set_names({"position_ids"});
    params.push_back(position_ids);
    
    return params;
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> Qwen3MoeGraphBuilder::build_embedding(
    const ov::Output<ov::Node>& input_ids) {
    
    // Check if embedding weights exist
    const std::string embed_key = "model.embed_tokens.weight";
    if (weights_.count(embed_key) == 0) {
        throw std::runtime_error("Missing embedding weights: " + embed_key);
    }
    
    // Load embedding weights
    const auto& embed_tensor = weights_.at(embed_key);
    
    // Create embedding constant
    auto embed_const = std::make_shared<ov::op::v0::Constant>(embed_tensor);
    
    // Convert to f32 for computation
    auto embed_f32 = std::make_shared<ov::op::v0::Convert>(embed_const, ov::element::f32);
    
    // Convert input_ids to i32 for Gather operation
    auto input_ids_i32 = std::make_shared<ov::op::v0::Convert>(input_ids, ov::element::i32);
    
    // Create axis constant for Gather (axis=0 for vocabulary dimension)
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    
    // Gather embeddings: [batch, seq_len, hidden_size]
    auto embeddings = std::make_shared<ov::op::v8::Gather>(embed_f32, input_ids_i32, axis);
    
    // Return both embeddings and embedding weights (for potential weight tying)
    return {embeddings->output(0), embed_f32->output(0)};
}

ov::Output<ov::Node> Qwen3MoeGraphBuilder::build_decoder_layer(
    int layer_idx,
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& attention_mask,
    const ov::Output<ov::Node>& position_ids,
    const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& position_embeddings,
    ov::SinkVector& sinks) {
    
    // Validate layer index
    if (layer_idx < 0 || layer_idx >= config_.num_hidden_layers) {
        std::ostringstream oss;
        oss << "Invalid layer index: " << layer_idx 
            << " (valid range: [0, " << config_.num_hidden_layers << "))";
        throw std::runtime_error(oss.str());
    }
    
    // Determine if this layer uses MoE or standard MLP
    bool is_moe = layer_selector_->is_moe_layer(layer_idx);
    
    // Create layer prefix for weight keys
    std::ostringstream layer_prefix_stream;
    layer_prefix_stream << "model.layers." << layer_idx;
    std::string layer_prefix = layer_prefix_stream.str();
    
    // 1. Input layer normalization
    auto input_norm = norm_builder_->build(
        hidden_states,
        layer_prefix + ".input_layernorm",
        weights_,
        config_.rms_norm_eps
    );
    
    // 2. Self-attention with Q/K normalization and RoPE
    auto [attn_output, attn_sinks] = attention_builder_->build(
        input_norm,
        attention_mask,
        position_ids,
        position_embeddings,
        layer_prefix + ".self_attn",
        weights_
    );
    
    // Collect attention KV cache sinks
    sinks.insert(sinks.end(), attn_sinks.begin(), attn_sinks.end());
    
    // 3. First residual connection
    auto residual_1 = std::make_shared<ov::op::v1::Add>(
        hidden_states,
        attn_output,
        ov::op::AutoBroadcastType::NUMPY
    );
    
    // 4. Post-attention layer normalization
    auto post_attn_norm = norm_builder_->build(
        residual_1->output(0),
        layer_prefix + ".post_attention_layernorm",
        weights_,
        config_.rms_norm_eps
    );
    
    // 5. MLP or MoE block based on layer configuration
    ov::Output<ov::Node> mlp_output;
    if (is_moe) {
        // Use sparse MoE block
        mlp_output = moe_builder_->build(
            post_attn_norm,
            layer_prefix + ".mlp",
            weights_
        );
    } else {
        // Use standard MLP
        mlp_output = mlp_builder_->build(
            post_attn_norm,
            layer_prefix + ".mlp",
            weights_,
            config_.intermediate_size,
            config_.hidden_act
        );
    }
    
    // 6. Second residual connection
    auto residual_2 = std::make_shared<ov::op::v1::Add>(
        residual_1->output(0),
        mlp_output,
        ov::op::AutoBroadcastType::NUMPY
    );
    
    return residual_2->output(0);
}

ov::Output<ov::Node> Qwen3MoeGraphBuilder::build_lm_head(
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& embeddings) {
    
    ov::Output<ov::Node> lm_head_weights;
    
    // Check for weight tying
    const std::string lm_head_key = "lm_head.weight";
    if (weights_.count(lm_head_key) > 0) {
        // Use separate LM head weights
        const auto& lm_head_tensor = weights_.at(lm_head_key);
        auto lm_head_const = std::make_shared<ov::op::v0::Constant>(lm_head_tensor);
        lm_head_weights = std::make_shared<ov::op::v0::Convert>(
            lm_head_const, 
            ov::element::f32
        )->output(0);
    } else {
        // Use embedding weights (weight tying)
        lm_head_weights = embeddings;
    }
    
    // MatMul: [batch, seq_len, hidden_size] @ [vocab_size, hidden_size]^T
    // Result: [batch, seq_len, vocab_size]
    auto logits = std::make_shared<ov::op::v0::MatMul>(
        hidden_states,
        lm_head_weights,
        false,  // transpose_a
        true    // transpose_b
    );
    
    return logits->output(0);
}

std::shared_ptr<ov::Model> Qwen3MoeGraphBuilder::build_graph() {
    // Validate that weights have been set
    if (weights_.empty()) {
        throw std::runtime_error("Model weights must be set before building graph. Call set_weights() first.");
    }
    
    // 1. Create input parameters
    auto params = create_input_parameters();
    auto input_ids = params[0]->output(0);
    auto attention_mask = params[1]->output(0);
    auto position_ids = params[2]->output(0);
    
    // 2. Build embedding layer
    auto [inputs_embeds, embeddings] = build_embedding(input_ids);
    auto hidden_states = inputs_embeds;
    
    // 3. Get input shape components for RoPE
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_ids);
    auto batch_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, 
        ov::Shape{1}, 
        0
    );
    auto batch_size = std::make_shared<ov::op::v8::Gather>(
        input_shape,
        batch_axis,
        batch_axis
    );
    
    // 4. Build RoPE constants (cached in rope_builder_)
    auto rope_const = rope_builder_->build_rope_constants();
    
    // 5. Build position embeddings (cos, sin)
    auto position_embeddings = rope_builder_->build_position_embeddings(
        position_ids,
        batch_size->output(0)
    );
    
    // 6. Process all decoder layers
    ov::SinkVector sinks;
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        hidden_states = build_decoder_layer(
            i,
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings,
            sinks
        );
    }
    
    // 7. Final layer normalization
    auto final_norm = norm_builder_->build(
        hidden_states,
        "model.norm",
        weights_,
        config_.rms_norm_eps
    );
    
    // 8. LM head projection
    auto logits = build_lm_head(final_norm, embeddings);
    
    // 9. Create result node
    auto result = std::make_shared<ov::op::v0::Result>(logits);
    result->set_friendly_name("logits");
    result->output(0).set_names({"logits"});
    
    // 10. Create model with results, sinks (KV cache operations), and parameters
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{result->output(0)},
        sinks,
        ov::ParameterVector{params.begin(), params.end()}
    );
    
    // 11. Set model metadata
    model->set_friendly_name("qwen3_moe");
    
    // 12. Set runtime options for KV cache precision
    // Use f16 for KV cache to reduce memory usage
    model->set_rt_info(ov::element::f16, {"runtime_options", "kv_cache_precision"});
    
    // Set activations scale factor for quantization-aware inference
    model->set_rt_info(8.0f, {"runtime_options", "activations_scale_factor"});
    
    return model;
}

ov::ParameterVector Qwen3MoeGraphBuilder::get_model_inputs() const {
    // This method would be called after build_graph() to retrieve inputs
    // For now, return empty as inputs are created during build_graph()
    return ov::ParameterVector{};
}

ov::OutputVector Qwen3MoeGraphBuilder::get_model_outputs() const {
    // This method would be called after build_graph() to retrieve outputs
    // For now, return empty as outputs are created during build_graph()
    return ov::OutputVector{};
}

} // namespace genai
} // namespace ov