// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "modeling/builder_context.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_moe.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3_5TextModelConfig {
    std::string architecture = "qwen3_5";
    int32_t hidden_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t vocab_size = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    float partial_rotary_factor = 0.25f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool tie_word_embeddings = false;
    std::vector<std::string> layer_types;
    int32_t full_attention_interval = 4;

    int32_t linear_conv_kernel_dim = 4;
    int32_t linear_key_head_dim = 128;
    int32_t linear_value_head_dim = 128;
    int32_t linear_num_key_heads = 16;
    int32_t linear_num_value_heads = 32;
    int32_t moe_intermediate_size = 0;
    int32_t shared_expert_intermediate_size = 0;
    int32_t num_experts = 0;
    int32_t num_experts_per_tok = 0;
    bool norm_topk_prob = true;
    bool output_router_logits = false;
    float router_aux_loss_coef = 0.0f;

    bool mrope_interleaved = false;
    std::vector<int32_t> mrope_section = {11, 11, 10};

    bool is_moe_enabled() const {
        return num_experts > 0 && moe_intermediate_size > 0 && shared_expert_intermediate_size > 0;
    }
};

class Qwen3_5EmbeddingInjector : public Module {
public:
    Qwen3_5EmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    Tensor forward(const Tensor& inputs_embeds,
                   const Tensor& visual_embeds,
                   const Tensor& visual_pos_mask) const;
};

class Qwen3_5RMSNorm : public Module {
public:
    Qwen3_5RMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;
    std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& residual) const;

private:
    const Tensor& weight() const;

    WeightParameter* weight_param_ = nullptr;
    float eps_ = 1e-6f;
};

class Qwen3_5Attention : public Module {
public:
    Qwen3_5Attention(BuilderContext& ctx, const std::string& name, const Qwen3_5TextModelConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor* attention_mask) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    int32_t rotary_dim_ = 0;
    float scaling_ = 1.0f;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    Qwen3_5RMSNorm q_norm_;
    Qwen3_5RMSNorm k_norm_;
};

class Qwen3_5GatedDeltaNet : public Module {
public:
    Qwen3_5GatedDeltaNet(BuilderContext& ctx, const std::string& name, const Qwen3_5TextModelConfig& cfg, int32_t layer_idx, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor* attention_mask,
                   const Tensor* cache_position) const;

private:
    const Tensor& in_proj_qkv_weight() const;
    const Tensor& in_proj_z_weight() const;
    const Tensor& in_proj_b_weight() const;
    const Tensor& in_proj_a_weight() const;
    const Tensor& conv1d_weight() const;
    const Tensor& a_log() const;
    const Tensor& dt_bias() const;
    const Tensor& out_proj_weight() const;

    Tensor apply_depthwise_causal_conv(const Tensor& mixed_qkv,
                                       const Tensor& prev_conv_state,
                                       Tensor* next_conv_state) const;

    Tensor rms_norm_gated(const Tensor& x, const Tensor& z) const;

    int32_t layer_idx_ = 0;
    int32_t hidden_size_ = 0;
    int32_t num_v_heads_ = 0;
    int32_t num_k_heads_ = 0;
    int32_t head_k_dim_ = 0;
    int32_t head_v_dim_ = 0;
    int32_t key_dim_ = 0;
    int32_t value_dim_ = 0;
    int32_t conv_dim_ = 0;
    int32_t conv_kernel_size_ = 4;
    int32_t conv_state_size_ = 0;
    float eps_ = 1e-6f;

    WeightParameter* in_proj_qkv_param_ = nullptr;
    WeightParameter* in_proj_z_param_ = nullptr;
    WeightParameter* in_proj_b_param_ = nullptr;
    WeightParameter* in_proj_a_param_ = nullptr;
    WeightParameter* conv1d_param_ = nullptr;
    WeightParameter* a_log_param_ = nullptr;
    WeightParameter* dt_bias_param_ = nullptr;
    WeightParameter* norm_param_ = nullptr;
    WeightParameter* out_proj_param_ = nullptr;
};

class Qwen3_5MLP : public Module {
public:
    Qwen3_5MLP(BuilderContext& ctx,
               const std::string& name,
               const Qwen3_5TextModelConfig& cfg,
               int32_t intermediate_size,
               Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class Qwen3_5DecoderLayer : public Module {
public:
    Qwen3_5DecoderLayer(BuilderContext& ctx,
                        const std::string& name,
                        const Qwen3_5TextModelConfig& cfg,
                        int32_t layer_idx,
                        Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor& beam_idx,
                                      const Tensor& rope_cos,
                                      const Tensor& rope_sin,
                                      const Tensor* full_attention_mask,
                                      const Tensor* linear_attention_mask,
                                      const Tensor* cache_position,
                                      const std::optional<Tensor>& residual) const;

private:
    std::string layer_type_;
    std::unique_ptr<Qwen3_5Attention> self_attn_;
    std::unique_ptr<Qwen3_5GatedDeltaNet> linear_attn_;
    std::unique_ptr<Qwen3_5MLP> dense_mlp_;
    std::unique_ptr<Qwen3_5SparseMoeBlock> moe_mlp_;
    Qwen3_5RMSNorm input_layernorm_;
    Qwen3_5RMSNorm post_attention_layernorm_;
};

class Qwen3_5Model : public Module {
public:
    Qwen3_5Model(BuilderContext& ctx, const Qwen3_5TextModelConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor& full_attention_mask,
                   const Tensor* linear_attention_mask,
                   const Tensor* cache_position,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr);
    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor& full_attention_mask,
                          const Tensor* linear_attention_mask,
                          const Tensor* cache_position,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* visual_pos_mask = nullptr);

    VocabEmbedding& embed_tokens();

private:
    Tensor forward_impl(const Tensor* input_ids,
                        const Tensor* inputs_embeds,
                        const Tensor& position_ids,
                        const Tensor& beam_idx,
                        const Tensor& full_attention_mask,
                        const Tensor* linear_attention_mask,
                        const Tensor* cache_position,
                        const Tensor* visual_embeds,
                        const Tensor* visual_pos_mask);
    std::pair<Tensor, Tensor> build_mrope_cos_sin(const Tensor& position_ids) const;

    Qwen3_5TextModelConfig cfg_;
    VocabEmbedding embed_tokens_;
    Qwen3_5EmbeddingInjector embedding_injector_;
    std::vector<Qwen3_5DecoderLayer> layers_;
    Qwen3_5RMSNorm norm_;
    int32_t head_dim_ = 0;
    int32_t rotary_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

class Qwen3_5ForCausalLM : public Module {
public:
    Qwen3_5ForCausalLM(BuilderContext& ctx, const Qwen3_5TextModelConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor& full_attention_mask,
                   const Tensor* linear_attention_mask,
                   const Tensor* cache_position,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr);
    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor& full_attention_mask,
                          const Tensor* linear_attention_mask,
                          const Tensor* cache_position,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* visual_pos_mask = nullptr);

private:
    Qwen3_5TextModelConfig cfg_;
    Qwen3_5Model model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_qwen3_5_text_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds = false,
    bool enable_visual_inputs = true);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
