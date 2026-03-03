// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/rope.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

std::vector<std::string> build_default_layer_types(int32_t num_layers, int32_t interval) {
    const int32_t safe_interval = interval > 0 ? interval : 4;
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        out.push_back(((i + 1) % safe_interval) == 0 ? "full_attention" : "linear_attention");
    }
    return out;
}

std::optional<int32_t> get_qwen3_5_layer_limit_from_env() {
    static constexpr const char* kEnvName = "OV_GENAI_qwen3_5_NUM_LAYERS";
    const char* raw = std::getenv(kEnvName);
    if (!raw || raw[0] == '\0') {
        return std::nullopt;
    }

    try {
        const int64_t parsed = std::stoll(raw);
        if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
            OPENVINO_THROW(kEnvName, " must be an integer in [0, INT32_MAX], got: ", raw);
        }
        return static_cast<int32_t>(parsed);
    } catch (const std::exception&) {
        OPENVINO_THROW(kEnvName, " must be an integer in [0, INT32_MAX], got: ", raw);
    }
}

bool use_linear_attention_op() {
    const char* raw = std::getenv("OV_GENAI_USE_LINEAR_ATTENTION_OP");
    if (!raw || raw[0] == '\0')
        return true;  // enabled by default
    return std::string(raw) != "0";
}

ov::genai::modeling::models::Qwen3_5TextModelConfig apply_qwen3_5_layer_limit(
    const ov::genai::modeling::models::Qwen3_5TextModelConfig& input_cfg) {
    auto cfg = input_cfg;
    const auto env_limit = get_qwen3_5_layer_limit_from_env();
    if (!env_limit.has_value()) {
        return cfg;
    }

    const int32_t limited_layers = std::clamp(*env_limit, 0, cfg.num_hidden_layers);
    cfg.num_hidden_layers = limited_layers;

    if (!cfg.layer_types.empty() && cfg.layer_types.size() >= static_cast<size_t>(limited_layers)) {
        cfg.layer_types.resize(static_cast<size_t>(limited_layers));
    }

    return cfg;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3_5EmbeddingInjector::Qwen3_5EmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {}

Tensor Qwen3_5EmbeddingInjector::forward(const Tensor& inputs_embeds,
                                         const Tensor& visual_embeds,
                                         const Tensor& visual_pos_mask) const {
    auto mask = visual_pos_mask.unsqueeze(2);
    auto updates = visual_embeds.to(inputs_embeds.dtype());
    return ops::tensor::masked_scatter(inputs_embeds, mask, updates);
}

Qwen3_5RMSNorm::Qwen3_5RMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent)
    : Module(name, ctx, parent),
      eps_(eps) {
    weight_param_ = &register_parameter("weight");
}

const Tensor& Qwen3_5RMSNorm::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("Qwen3_5RMSNorm weight parameter is not registered");
    }
    return weight_param_->value();
}

Tensor Qwen3_5RMSNorm::forward(const Tensor& x) const {
    auto x_f32 = x.to(ov::element::f32);
    auto var = x_f32.pow(2.0f).mean(-1, true);
    auto norm = x_f32 * (var + eps_).rsqrt();
    auto scaled = norm * (1.0f + weight().to(ov::element::f32));
    return scaled.to(x.dtype());
}

std::pair<Tensor, Tensor> Qwen3_5RMSNorm::forward(const Tensor& x, const Tensor& residual) const {
    auto sum = x + residual;
    return {forward(sum), sum};
}

Qwen3_5Attention::Qwen3_5Attention(BuilderContext& ctx,
                                       const std::string& name,
                                       const Qwen3_5TextModelConfig& cfg,
                                       Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      q_norm_(ctx, "q_norm", cfg.rms_norm_eps, this),
      k_norm_(ctx, "k_norm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0 || num_kv_heads_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 attention configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }
    rotary_dim_ = static_cast<int32_t>(std::floor(static_cast<float>(head_dim_) * cfg.partial_rotary_factor));
    rotary_dim_ = std::max<int32_t>(0, std::min<int32_t>(rotary_dim_, head_dim_));
    if ((rotary_dim_ % 2) != 0) {
        rotary_dim_ -= 1;
    }

    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");
}

const Tensor& Qwen3_5Attention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention q_proj parameter is not registered");
    }
    return q_proj_param_->value();
}

const Tensor& Qwen3_5Attention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention k_proj parameter is not registered");
    }
    return k_proj_param_->value();
}

const Tensor& Qwen3_5Attention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention v_proj parameter is not registered");
    }
    return v_proj_param_->value();
}

const Tensor& Qwen3_5Attention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("Qwen3_5Attention o_proj parameter is not registered");
    }
    return o_proj_param_->value();
}

Tensor Qwen3_5Attention::forward(const Tensor& hidden_states,
                                   const Tensor& beam_idx,
                                   const Tensor& rope_cos,
                                   const Tensor& rope_sin,
                                   const Tensor* attention_mask) const {
    auto* policy = &ctx().op_policy();
    auto* op_ctx = hidden_states.context();

    auto q_linear = ops::linear(hidden_states, q_proj_weight()).reshape({0, 0, num_heads_, head_dim_ * 2});
    auto q_states = ops::slice(q_linear, 0, head_dim_, 1, 3);
    auto gate = ops::slice(q_linear, head_dim_, head_dim_ * 2, 1, 3).reshape({0, 0, num_heads_ * head_dim_});

    auto k_states = ops::linear(hidden_states, k_proj_weight()).reshape({0, 0, num_kv_heads_, head_dim_});
    auto v_states = ops::linear(hidden_states, v_proj_weight()).reshape({0, 0, num_kv_heads_, head_dim_});

    auto q_heads = q_norm_.forward(q_states).permute({0, 2, 1, 3});
    auto k_heads = k_norm_.forward(k_states).permute({0, 2, 1, 3});
    auto v_heads = v_states.permute({0, 2, 1, 3});

    if (rotary_dim_ > 0) {
        auto q_rot = ops::slice(q_heads, 0, rotary_dim_, 1, 3);
        auto k_rot = ops::slice(k_heads, 0, rotary_dim_, 1, 3);
        auto q_rotated = ops::llm::apply_rope(q_rot, rope_cos, rope_sin, rotary_dim_, policy);
        auto k_rotated = ops::llm::apply_rope(k_rot, rope_cos, rope_sin, rotary_dim_, policy);
        if (rotary_dim_ < head_dim_) {
            auto q_pass = ops::slice(q_heads, rotary_dim_, head_dim_, 1, 3);
            auto k_pass = ops::slice(k_heads, rotary_dim_, head_dim_, 1, 3);
            q_heads = ops::concat({q_rotated, q_pass}, 3);
            k_heads = ops::concat({k_rotated, k_pass}, 3);
        } else {
            q_heads = q_rotated;
            k_heads = k_rotated;
        }
    }

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    auto cached = ops::append_kv_cache(k_heads, v_heads, beam_idx, num_kv_heads_, head_dim_, cache_prefix, ctx());
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    Tensor mask = attention_mask
                      ? ops::llm::build_kv_causal_mask_with_attention(q_heads, k_expanded, *attention_mask)
                      : ops::llm::build_kv_causal_mask(q_heads, k_expanded);
    auto attn = ops::llm::sdpa(q_heads, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);

    const int64_t attn_hidden = static_cast<int64_t>(num_heads_) * static_cast<int64_t>(head_dim_);
    auto merged = attn.permute({0, 2, 1, 3}).reshape({0, 0, attn_hidden});
    auto gate_sigmoid = Tensor(std::make_shared<ov::op::v0::Sigmoid>(gate.output()), op_ctx);
    auto gated = merged * gate_sigmoid;
    return ops::linear(gated, o_proj_weight());
}

Qwen3_5MLP::Qwen3_5MLP(BuilderContext& ctx,
                           const std::string& name,
                           const Qwen3_5TextModelConfig& cfg,
                           int32_t intermediate_size,
                           Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3_5 MLP activation: ", cfg.hidden_act);
    }
    if (intermediate_size <= 0) {
        OPENVINO_THROW("Qwen3_5MLP intermediate size must be > 0");
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen3_5MLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("Qwen3_5MLP gate_proj parameter is not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& Qwen3_5MLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("Qwen3_5MLP up_proj parameter is not registered");
    }
    return up_proj_param_->value();
}

const Tensor& Qwen3_5MLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("Qwen3_5MLP down_proj parameter is not registered");
    }
    return down_proj_param_->value();
}

Tensor Qwen3_5MLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

Qwen3_5GatedDeltaNet::Qwen3_5GatedDeltaNet(BuilderContext& ctx,
                                               const std::string& name,
                                               const Qwen3_5TextModelConfig& cfg,
                                               int32_t layer_idx,
                                               Module* parent)
    : Module(name, ctx, parent),
      layer_idx_(layer_idx),
      hidden_size_(cfg.hidden_size),
      num_v_heads_(cfg.linear_num_value_heads),
      num_k_heads_(cfg.linear_num_key_heads),
      head_k_dim_(cfg.linear_key_head_dim),
      head_v_dim_(cfg.linear_value_head_dim),
      key_dim_(head_k_dim_ * num_k_heads_),
      value_dim_(head_v_dim_ * num_v_heads_),
      conv_dim_(key_dim_ * 2 + value_dim_),
      conv_kernel_size_(cfg.linear_conv_kernel_dim),
      conv_state_size_(cfg.linear_conv_kernel_dim),
      eps_(cfg.rms_norm_eps) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3_5 linear attention activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || num_v_heads_ <= 0 || num_k_heads_ <= 0 || head_k_dim_ <= 0 || head_v_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 linear attention configuration");
    }
    if (conv_kernel_size_ <= 0) {
        OPENVINO_THROW("Qwen3_5 linear_conv_kernel_dim must be > 0");
    }
    if ((num_v_heads_ % num_k_heads_) != 0) {
        OPENVINO_THROW("Qwen3_5 linear_num_value_heads must be divisible by linear_num_key_heads");
    }

    in_proj_qkv_param_ = &register_parameter("in_proj_qkv.weight");
    in_proj_z_param_ = &register_parameter("in_proj_z.weight");
    in_proj_b_param_ = &register_parameter("in_proj_b.weight");
    in_proj_a_param_ = &register_parameter("in_proj_a.weight");
    conv1d_param_ = &register_parameter("conv1d.weight");
    a_log_param_ = &register_parameter("A_log");
    dt_bias_param_ = &register_parameter("dt_bias");
    norm_param_ = &register_parameter("norm.weight");
    out_proj_param_ = &register_parameter("out_proj.weight");
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_qkv_weight() const {
    if (!in_proj_qkv_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_qkv parameter is not registered");
    }
    return in_proj_qkv_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_z_weight() const {
    if (!in_proj_z_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_z parameter is not registered");
    }
    return in_proj_z_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_b_weight() const {
    if (!in_proj_b_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_b parameter is not registered");
    }
    return in_proj_b_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::in_proj_a_weight() const {
    if (!in_proj_a_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet in_proj_a parameter is not registered");
    }
    return in_proj_a_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::conv1d_weight() const {
    if (!conv1d_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet conv1d parameter is not registered");
    }
    return conv1d_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::a_log() const {
    if (!a_log_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet A_log parameter is not registered");
    }
    return a_log_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::dt_bias() const {
    if (!dt_bias_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet dt_bias parameter is not registered");
    }
    return dt_bias_param_->value();
}

const Tensor& Qwen3_5GatedDeltaNet::out_proj_weight() const {
    if (!out_proj_param_) {
        OPENVINO_THROW("Qwen3_5GatedDeltaNet out_proj parameter is not registered");
    }
    return out_proj_param_->value();
}

Tensor Qwen3_5GatedDeltaNet::apply_depthwise_causal_conv(const Tensor& mixed_qkv,
                                                           const Tensor& prev_conv_state,
                                                           Tensor* next_conv_state) const {
    auto* op_ctx = mixed_qkv.context();
    auto input_with_state = ops::concat({prev_conv_state, mixed_qkv}, 2);

    auto conv_weight = conv1d_weight().reshape({conv_dim_, 1, 1, conv_kernel_size_}, false);
    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(input_with_state.output(),
                                                                conv_weight.output(),
                                                                ov::Strides{1},
                                                                ov::CoordinateDiff{0},
                                                                ov::CoordinateDiff{0},
                                                                ov::Strides{1});
    auto conv_act = ops::silu(Tensor(conv, op_ctx));

    auto seq_len = shape::dim(mixed_qkv, 2);
    auto conv_len = shape::dim(conv_act, 2);
    auto out_start = std::make_shared<ov::op::v1::Subtract>(conv_len, seq_len);
    auto out_slice = std::make_shared<ov::op::v8::Slice>(conv_act.output(),
                                                         out_start,
                                                         conv_len,
                                                         ops::const_vec(op_ctx, std::vector<int64_t>{1}),
                                                         ops::const_vec(op_ctx, std::vector<int64_t>{2}));

    auto total_len = shape::dim(input_with_state, 2);
    auto kernel = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_state_size_)});
    auto state_start = std::make_shared<ov::op::v1::Subtract>(total_len, kernel);
    auto state_slice = std::make_shared<ov::op::v8::Slice>(input_with_state.output(),
                                                           state_start,
                                                           total_len,
                                                           ops::const_vec(op_ctx, std::vector<int64_t>{1}),
                                                           ops::const_vec(op_ctx, std::vector<int64_t>{2}));
    if (next_conv_state) {
        *next_conv_state = Tensor(state_slice, op_ctx);
    }
    return Tensor(out_slice, op_ctx);
}

Tensor Qwen3_5GatedDeltaNet::rms_norm_gated(const Tensor& x, const Tensor& z) const {
    auto x_f32 = x.to(ov::element::f32);
    auto z_f32 = z.to(ov::element::f32);
    auto var = x_f32.pow(2.0f).mean(-1, true);
    auto norm = x_f32 * (var + eps_).rsqrt();
    auto gate = ops::silu(z_f32);
    auto weighted = norm * norm_param_->value().to(ov::element::f32);
    return (weighted * gate).to(x.dtype());
}

Tensor Qwen3_5GatedDeltaNet::forward(const Tensor& hidden_states,
                                       const Tensor& beam_idx,
                                       const Tensor* attention_mask,
                                       const Tensor* cache_position) const {
    (void)cache_position;
    auto* op_ctx = hidden_states.context();

    Tensor masked_hidden = hidden_states;
    if (attention_mask) {
        masked_hidden = hidden_states * attention_mask->to(hidden_states.dtype()).unsqueeze(2);
    }

    const int32_t ratio = num_v_heads_ / num_k_heads_;

    auto projected_qkv = ops::linear(masked_hidden, in_proj_qkv_weight());
    auto projected_z = ops::linear(masked_hidden, in_proj_z_weight());
    auto projected_b = ops::linear(masked_hidden, in_proj_b_weight());
    auto projected_a = ops::linear(masked_hidden, in_proj_a_weight());

    // Keep Z in head layout; b/a are already [B, S, num_v_heads].
    auto z = projected_z.reshape({0, 0, num_v_heads_, head_v_dim_});
    auto b = projected_b;
    auto a = projected_a;

    auto mixed_qkv = projected_qkv.permute({0, 2, 1});

    auto batch = shape::dim(masked_hidden, 0);
    auto conv_shape = shape::make({batch,
                                   ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_dim_)}),
                                   ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(conv_state_size_)})});
    auto conv_init = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(masked_hidden.dtype()), conv_shape);

    auto state_prefix = "linear_states." + std::to_string(layer_idx_);
    ov::op::util::VariableInfo conv_info{ov::PartialShape{-1, conv_dim_, conv_state_size_},
                                         masked_hidden.dtype(),
                                         state_prefix + ".conv"};
    auto conv_var = std::make_shared<ov::op::util::Variable>(conv_info);
    auto conv_read = std::make_shared<ov::op::v6::ReadValue>(conv_init.output(), conv_var);
    auto conv_cached = ops::gather(Tensor(conv_read->output(0), op_ctx), beam_idx, 0);

    Tensor next_conv_state;
    auto mixed_after_conv = apply_depthwise_causal_conv(mixed_qkv, conv_cached, &next_conv_state);
    auto conv_assign = std::make_shared<ov::opset13::Assign>(next_conv_state.output(), conv_var);
    ctx().register_sink(conv_assign);

    auto mixed_bt = mixed_after_conv.permute({0, 2, 1});
    auto q_conv = ops::slice(mixed_bt, 0, key_dim_, 1, 2);
    auto k_conv = ops::slice(mixed_bt, key_dim_, key_dim_ * 2, 1, 2);
    auto v_conv = ops::slice(mixed_bt, key_dim_ * 2, key_dim_ * 2 + value_dim_, 1, 2);

    auto q_heads = q_conv.reshape({0, 0, num_k_heads_, head_k_dim_});
    auto k_heads = k_conv.reshape({0, 0, num_k_heads_, head_k_dim_});
    auto v_heads = v_conv.reshape({0, 0, num_v_heads_, head_v_dim_});

    if (ratio > 1 && !use_linear_attention_op()) {
        q_heads = ops::llm::repeat_kv(q_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
                      .permute({0, 2, 1, 3});
        k_heads = ops::llm::repeat_kv(k_heads.permute({0, 2, 1, 3}), num_v_heads_, num_k_heads_, head_k_dim_)
                      .permute({0, 2, 1, 3});
    }

    auto reduce_kdim = ops::const_vec(op_ctx, std::vector<int64_t>{-1});
    auto q_f32 = q_heads.to(ov::element::f32);
    auto k_f32 = k_heads.to(ov::element::f32);
    auto v_f32 = v_heads.to(ov::element::f32);

    auto beta = Tensor(std::make_shared<ov::op::v0::Sigmoid>(b.to(ov::element::f32).output()), op_ctx);
    auto softplus_in = a.to(ov::element::f32) + dt_bias().to(ov::element::f32);
    auto softplus = Tensor(std::make_shared<ov::op::v4::SoftPlus>(softplus_in.output()), op_ctx);
    auto g = -(a_log().to(ov::element::f32).exp() * softplus);

    auto recurrent_shape = shape::make({batch,
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_v_heads_)}),
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_k_dim_)}),
                                        ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_v_dim_)})});
    auto recurrent_init = shape::broadcast_to(Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx), recurrent_shape);

    ov::op::util::VariableInfo recurrent_info{ov::PartialShape{-1, num_v_heads_, head_k_dim_, head_v_dim_},
                                              ov::element::f32,
                                              state_prefix + ".recurrent"};
    auto recurrent_var = std::make_shared<ov::op::util::Variable>(recurrent_info);
    auto recurrent_read = std::make_shared<ov::op::v6::ReadValue>(recurrent_init.output(), recurrent_var);
    auto recurrent_cached = ops::gather(Tensor(recurrent_read->output(0), op_ctx), beam_idx, 0);

    Tensor core_attn_tensor;  // [B, S, num_v_heads, head_v_dim]

    if (use_linear_attention_op()) {
        // ── Fused LinearAttention op path ──
        auto la_result = ops::linear_attention(q_f32, k_f32, v_f32, beta, g, recurrent_cached);
        core_attn_tensor = la_result.first;   // [B, S, num_v_heads, head_v_dim]
        auto recurrent_final = la_result.second;  // [B, num_v_heads, head_k_dim, head_v_dim]
        auto recurrent_assign = std::make_shared<ov::opset13::Assign>(recurrent_final.output(), recurrent_var);
        ctx().register_sink(recurrent_assign);
    } else {
        // ── TensorIterator path (default) ──
        auto q_ss = Tensor(std::make_shared<ov::op::v1::ReduceSum>(q_f32.pow(2.0f).output(), reduce_kdim, true), op_ctx);
        auto k_ss = Tensor(std::make_shared<ov::op::v1::ReduceSum>(k_f32.pow(2.0f).output(), reduce_kdim, true), op_ctx);
        auto q_normed = q_f32 * (q_ss + 1e-6f).rsqrt();
        auto k_normed = k_f32 * (k_ss + 1e-6f).rsqrt();
        auto q_scaled = q_normed * (1.0f / std::sqrt(static_cast<float>(head_k_dim_)));

        auto q_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_, head_k_dim_});
        auto k_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_, head_k_dim_});
        auto v_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_, head_v_dim_});
        auto g_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_});
        auto b_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, num_v_heads_});
        auto state_t = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, num_v_heads_, head_k_dim_, head_v_dim_});

        auto axis_seq = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto q_s = std::make_shared<ov::op::v0::Squeeze>(q_t, axis_seq);
        auto k_s = std::make_shared<ov::op::v0::Squeeze>(k_t, axis_seq);
        auto v_s = std::make_shared<ov::op::v0::Squeeze>(v_t, axis_seq);
        auto g_s = std::make_shared<ov::op::v0::Squeeze>(g_t, axis_seq);
        auto b_s = std::make_shared<ov::op::v0::Squeeze>(b_t, axis_seq);

        auto g_exp = std::make_shared<ov::op::v0::Exp>(g_s);
        auto g_exp_u1 = std::make_shared<ov::op::v0::Unsqueeze>(g_exp, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        auto g_exp_e = std::make_shared<ov::op::v0::Unsqueeze>(g_exp_u1, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        auto state_decay = std::make_shared<ov::op::v1::Multiply>(state_t, g_exp_e, ov::op::AutoBroadcastType::NUMPY);

        auto k_uns = std::make_shared<ov::op::v0::Unsqueeze>(k_s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        auto state_k = std::make_shared<ov::op::v1::Multiply>(state_decay, k_uns, ov::op::AutoBroadcastType::NUMPY);
        auto kv_mem = std::make_shared<ov::op::v1::ReduceSum>(state_k,
                                                               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                               false);

        auto b_uns = std::make_shared<ov::op::v0::Unsqueeze>(b_s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        auto delta = std::make_shared<ov::op::v1::Multiply>(
            std::make_shared<ov::op::v1::Subtract>(v_s, kv_mem, ov::op::AutoBroadcastType::NUMPY),
            b_uns,
            ov::op::AutoBroadcastType::NUMPY);

        auto delta_uns = std::make_shared<ov::op::v0::Unsqueeze>(delta, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-2}));
        auto outer = std::make_shared<ov::op::v1::Multiply>(k_uns, delta_uns, ov::op::AutoBroadcastType::NUMPY);
        auto state_new = std::make_shared<ov::op::v1::Add>(state_decay, outer, ov::op::AutoBroadcastType::NUMPY);

        auto q_uns = std::make_shared<ov::op::v0::Unsqueeze>(q_s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        auto y_state = std::make_shared<ov::op::v1::Multiply>(state_new, q_uns, ov::op::AutoBroadcastType::NUMPY);
        auto y_t_step = std::make_shared<ov::op::v1::ReduceSum>(y_state,
                                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                                false);
        auto y_t_unsq = std::make_shared<ov::op::v0::Unsqueeze>(y_t_step, axis_seq);

        auto state_result = std::make_shared<ov::op::v0::Result>(state_new);
        auto y_result = std::make_shared<ov::op::v0::Result>(y_t_unsq);
        auto body = std::make_shared<ov::Model>(ov::OutputVector{state_result->output(0), y_result->output(0)},
                                                ov::ParameterVector{q_t, k_t, v_t, g_t, b_t, state_t});

        auto ti = std::make_shared<ov::op::v0::TensorIterator>();
        ti->set_body(body);
        ti->set_sliced_input(q_t, q_scaled.output(), 0, 1, 1, -1, 1);
        ti->set_sliced_input(k_t, k_normed.output(), 0, 1, 1, -1, 1);
        ti->set_sliced_input(v_t, v_f32.output(), 0, 1, 1, -1, 1);
        ti->set_sliced_input(g_t, g.output(), 0, 1, 1, -1, 1);
        ti->set_sliced_input(b_t, beta.output(), 0, 1, 1, -1, 1);
        ti->set_merged_input(state_t, recurrent_cached.output(), state_result);

        auto recurrent_final = ti->get_iter_value(state_result, -1);
        auto core_attn = ti->get_concatenated_slices(y_result, 0, 1, 1, -1, 1);
        auto recurrent_assign = std::make_shared<ov::opset13::Assign>(recurrent_final, recurrent_var);
        ctx().register_sink(recurrent_assign);

        core_attn_tensor = Tensor(core_attn, op_ctx);
    }
    auto gated_4d = rms_norm_gated(core_attn_tensor, z);
    auto merged = gated_4d.reshape({0, 0, value_dim_});
    return ops::linear(merged, out_proj_weight()).to(masked_hidden.dtype());
}

Qwen3_5DecoderLayer::Qwen3_5DecoderLayer(BuilderContext& ctx,
                                             const std::string& name,
                                             const Qwen3_5TextModelConfig& cfg,
                                             int32_t layer_idx,
                                             Module* parent)
    : Module(name, ctx, parent),
      layer_type_(cfg.layer_types.at(static_cast<size_t>(layer_idx))),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {
    if (layer_type_ == "full_attention") {
        self_attn_ = std::make_unique<Qwen3_5Attention>(ctx, "self_attn", cfg, this);
    } else if (layer_type_ == "linear_attention") {
        linear_attn_ = std::make_unique<Qwen3_5GatedDeltaNet>(ctx, "linear_attn", cfg, layer_idx, this);
    } else {
        OPENVINO_THROW("Unsupported Qwen3_5 layer type: ", layer_type_);
    }

    if (cfg.is_moe_enabled()) {
        moe_mlp_ = std::make_unique<Qwen3_5SparseMoeBlock>(ctx, "mlp", cfg, this);
    } else {
        dense_mlp_ = std::make_unique<Qwen3_5MLP>(ctx, "mlp", cfg, cfg.intermediate_size, this);
    }
}

std::pair<Tensor, Tensor> Qwen3_5DecoderLayer::forward(const Tensor& hidden_states,
                                                         const Tensor& beam_idx,
                                                         const Tensor& rope_cos,
                                                         const Tensor& rope_sin,
                                                         const Tensor* full_attention_mask,
                                                         const Tensor* linear_attention_mask,
                                                         const Tensor* cache_position,
                                                         const std::optional<Tensor>& residual) const {
    Tensor normed;
    Tensor next_residual;
    if (residual) {
        auto norm_out = input_layernorm_.forward(hidden_states, *residual);
        normed = norm_out.first;
        next_residual = norm_out.second;
    } else {
        normed = input_layernorm_.forward(hidden_states);
        next_residual = hidden_states;
    }

    Tensor mixed;
    if (layer_type_ == "full_attention") {
        mixed = self_attn_->forward(normed, beam_idx, rope_cos, rope_sin, full_attention_mask);
    } else {
        mixed = linear_attn_->forward(normed, beam_idx, linear_attention_mask, cache_position);
    }

    auto post = post_attention_layernorm_.forward(mixed, next_residual);
    Tensor mlp_out = dense_mlp_ ? dense_mlp_->forward(post.first) : moe_mlp_->forward(post.first);
    return {mlp_out, post.second};
}

Qwen3_5Model::Qwen3_5Model(BuilderContext& ctx, const Qwen3_5TextModelConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      cfg_(cfg),
      embed_tokens_(ctx, "embed_tokens", this),
      embedding_injector_(ctx, "embedding_injector", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0
                    ? cfg.head_dim
                    : (cfg.num_attention_heads > 0 ? (cfg.hidden_size / cfg.num_attention_heads) : 0)),
      rope_theta_(cfg.rope_theta) {
    if (head_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3_5 head dimension");
    }
    rotary_dim_ = static_cast<int32_t>(std::floor(static_cast<float>(head_dim_) * cfg.partial_rotary_factor));
    rotary_dim_ = std::max<int32_t>(0, std::min<int32_t>(rotary_dim_, head_dim_));
    if ((rotary_dim_ % 2) != 0) {
        rotary_dim_ -= 1;
    }
    OPENVINO_ASSERT(rotary_dim_ > 0, "Qwen3_5 rotary dimension must be > 0");

    Qwen3_5TextModelConfig layer_cfg = cfg;
    if (layer_cfg.layer_types.empty()) {
        layer_cfg.layer_types = build_default_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
    }
    if (layer_cfg.layer_types.size() != static_cast<size_t>(cfg.num_hidden_layers)) {
        OPENVINO_THROW("Qwen3_5 layer_types size mismatch with num_hidden_layers");
    }

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", layer_cfg, i, this);
    }
}

std::pair<Tensor, Tensor> Qwen3_5Model::build_mrope_cos_sin(const Tensor& position_ids) const {
    auto* ctx = position_ids.context();
    const int32_t half_dim = rotary_dim_ / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int32_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim_);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta_, exponent);
    }

    auto inv_freq_const = ops::const_vec(ctx, inv_freq);
    Tensor inv_freq_tensor(inv_freq_const, ctx);
    auto inv_freq_reshaped =
        inv_freq_tensor.reshape({1, 1, static_cast<int64_t>(half_dim)}, false);

    const auto pos_rank = position_ids.output().get_partial_shape().rank();
    if (pos_rank.is_static() && pos_rank.get_length() == 2) {
        auto pos_f = position_ids.to(ov::element::f32);
        auto freqs = pos_f.unsqueeze(2) * inv_freq_reshaped;
        return {freqs.cos(), freqs.sin()};
    }

    auto pos_t = ops::slice(position_ids, 0, 1, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_h = ops::slice(position_ids, 1, 2, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_w = ops::slice(position_ids, 2, 3, 1, 0).squeeze(0).to(ov::element::f32);

    auto freqs_t = pos_t.unsqueeze(2) * inv_freq_reshaped;
    if (!cfg_.mrope_interleaved) {
        return {freqs_t.cos(), freqs_t.sin()};
    }

    auto freqs_h = pos_h.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_w = pos_w.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_all = ops::tensor::stack({freqs_t, freqs_h, freqs_w}, 0);
    auto freqs = ops::rope::mrope_interleaved(freqs_all, cfg_.mrope_section);
    return {freqs.cos(), freqs.sin()};
}

Tensor Qwen3_5Model::forward_impl(const Tensor* input_ids,
                                  const Tensor* inputs_embeds,
                                  const Tensor& position_ids,
                                  const Tensor& beam_idx,
                                  const Tensor& full_attention_mask,
                                  const Tensor* linear_attention_mask,
                                  const Tensor* cache_position,
                                  const Tensor* visual_embeds,
                                  const Tensor* visual_pos_mask) {
    OPENVINO_ASSERT((input_ids != nullptr) || (inputs_embeds != nullptr),
                    "Either input_ids or inputs_embeds must be provided");
    Tensor hidden_states = inputs_embeds ? *inputs_embeds : embed_tokens_.forward(*input_ids);
    if (visual_embeds && visual_pos_mask) {
        hidden_states = embedding_injector_.forward(hidden_states, *visual_embeds, *visual_pos_mask);
    }
    auto cos_sin = build_mrope_cos_sin(position_ids);

    std::optional<Tensor> linear_mask_view;
    const Tensor* linear_mask = nullptr;
    if (linear_attention_mask) {
        const Tensor& seq_source = inputs_embeds ? *inputs_embeds : *input_ids;
        auto* op_ctx = seq_source.context();
        auto q_len = shape::dim(seq_source, 1);
        auto mask_len = shape::dim(*linear_attention_mask, 1);
        auto start = std::make_shared<ov::op::v1::Subtract>(mask_len, q_len);
        auto sliced = std::make_shared<ov::op::v8::Slice>(
            linear_attention_mask->output(),
            start,
            mask_len,
            ops::const_vec(op_ctx, std::vector<int64_t>{1}),
            ops::const_vec(op_ctx, std::vector<int64_t>{1}));
        linear_mask_view = Tensor(sliced, op_ctx);
        linear_mask = &(*linear_mask_view);
    }

    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto out = layer.forward(hidden_states,
                                 beam_idx,
                                 cos_sin.first,
                                 cos_sin.second,
                                 &full_attention_mask,
                                 linear_mask,
                                 cache_position,
                                 residual);
        hidden_states = out.first;
        residual = out.second;
    }

    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

Tensor Qwen3_5Model::forward(const Tensor& input_ids,
                             const Tensor& position_ids,
                             const Tensor& beam_idx,
                             const Tensor& full_attention_mask,
                             const Tensor* linear_attention_mask,
                             const Tensor* cache_position,
                             const Tensor* visual_embeds,
                             const Tensor* visual_pos_mask) {
    return forward_impl(&input_ids,
                        nullptr,
                        position_ids,
                        beam_idx,
                        full_attention_mask,
                        linear_attention_mask,
                        cache_position,
                        visual_embeds,
                        visual_pos_mask);
}

Tensor Qwen3_5Model::forward_embeds(const Tensor& inputs_embeds,
                                    const Tensor& position_ids,
                                    const Tensor& beam_idx,
                                    const Tensor& full_attention_mask,
                                    const Tensor* linear_attention_mask,
                                    const Tensor* cache_position,
                                    const Tensor* visual_embeds,
                                    const Tensor* visual_pos_mask) {
    return forward_impl(nullptr,
                        &inputs_embeds,
                        position_ids,
                        beam_idx,
                        full_attention_mask,
                        linear_attention_mask,
                        cache_position,
                        visual_embeds,
                        visual_pos_mask);
}

VocabEmbedding& Qwen3_5Model::embed_tokens() {
    return embed_tokens_;
}

Qwen3_5ForCausalLM::Qwen3_5ForCausalLM(BuilderContext& ctx, const Qwen3_5TextModelConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor Qwen3_5ForCausalLM::forward(const Tensor& input_ids,
                                   const Tensor& position_ids,
                                   const Tensor& beam_idx,
                                   const Tensor& full_attention_mask,
                                   const Tensor* linear_attention_mask,
                                   const Tensor* cache_position,
                                   const Tensor* visual_embeds,
                                   const Tensor* visual_pos_mask) {
    auto hidden = model_.forward(input_ids,
                                 position_ids,
                                 beam_idx,
                                 full_attention_mask,
                                 linear_attention_mask,
                                 cache_position,
                                 visual_embeds,
                                 visual_pos_mask);
    return lm_head_.forward(hidden);
}

Tensor Qwen3_5ForCausalLM::forward_embeds(const Tensor& inputs_embeds,
                                          const Tensor& position_ids,
                                          const Tensor& beam_idx,
                                          const Tensor& full_attention_mask,
                                          const Tensor* linear_attention_mask,
                                          const Tensor* cache_position,
                                          const Tensor* visual_embeds,
                                          const Tensor* visual_pos_mask) {
    auto hidden = model_.forward_embeds(inputs_embeds,
                                        position_ids,
                                        beam_idx,
                                        full_attention_mask,
                                        linear_attention_mask,
                                        cache_position,
                                        visual_embeds,
                                        visual_pos_mask);
    return lm_head_.forward(hidden);
}

std::shared_ptr<ov::Model> create_qwen3_5_text_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds,
    bool enable_visual_inputs) {
    Qwen3_5TextModelConfig text_cfg;
    text_cfg.architecture = "qwen3_5";
    text_cfg.hidden_size = cfg.text.hidden_size;
    text_cfg.num_attention_heads = cfg.text.num_attention_heads;
    text_cfg.num_key_value_heads = cfg.text.num_key_value_heads > 0 ? cfg.text.num_key_value_heads : cfg.text.num_attention_heads;
    text_cfg.head_dim = cfg.text.resolved_head_dim();
    text_cfg.intermediate_size = cfg.text.intermediate_size;
    text_cfg.num_hidden_layers = cfg.text.num_hidden_layers;
    text_cfg.vocab_size = cfg.text.vocab_size;
    text_cfg.max_position_embeddings = cfg.text.max_position_embeddings;
    text_cfg.rms_norm_eps = cfg.text.rms_norm_eps;
    text_cfg.rope_theta = cfg.text.rope_theta;
    text_cfg.partial_rotary_factor = cfg.text.partial_rotary_factor;
    text_cfg.hidden_act = cfg.text.hidden_act;
    text_cfg.attention_bias = cfg.text.attention_bias;
    text_cfg.tie_word_embeddings = cfg.text.tie_word_embeddings;
    text_cfg.layer_types = cfg.text.layer_types;
    text_cfg.full_attention_interval = cfg.text.full_attention_interval;
    text_cfg.linear_conv_kernel_dim = cfg.text.linear_conv_kernel_dim;
    text_cfg.linear_key_head_dim = cfg.text.linear_key_head_dim;
    text_cfg.linear_value_head_dim = cfg.text.linear_value_head_dim;
    text_cfg.linear_num_key_heads = cfg.text.linear_num_key_heads;
    text_cfg.linear_num_value_heads = cfg.text.linear_num_value_heads;
    text_cfg.moe_intermediate_size = cfg.text.moe_intermediate_size;
    text_cfg.shared_expert_intermediate_size = cfg.text.shared_expert_intermediate_size;
    text_cfg.num_experts = cfg.text.num_experts;
    text_cfg.num_experts_per_tok = cfg.text.num_experts_per_tok;
    text_cfg.norm_topk_prob = cfg.text.norm_topk_prob;
    text_cfg.output_router_logits = cfg.text.output_router_logits;
    text_cfg.router_aux_loss_coef = cfg.text.router_aux_loss_coef;
    text_cfg.mrope_interleaved = cfg.text.rope.mrope_interleaved;
    text_cfg.mrope_section = cfg.text.rope.mrope_section;

    const auto effective_cfg = apply_qwen3_5_layer_limit(text_cfg);

    BuilderContext ctx;
    Qwen3_5ForCausalLM model(ctx, effective_cfg);
    // HF Qwen3.5-MoE checkpoints store text weights under
    // model.language_model.layers.N.*, while this model registers
    // model.layers[N].* parameters.
    // Add per-layer rules first so generic prefix rules don't consume them.
    for (int32_t i = 0; i < effective_cfg.num_hidden_layers; ++i) {
        const std::string idx = std::to_string(i);
        model.packed_mapping().rules.push_back(
            {"model.language_model.layers." + idx + ".", "model.layers[" + idx + "].", 0});
        model.packed_mapping().rules.push_back(
            {"language_model.layers." + idx + ".", "model.layers[" + idx + "].", 0});
    }
    model.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    model.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing = false;
    options.allow_unmatched = true;
    options.report_missing = true;
    options.report_unmatched = false;
    (void)ov::genai::modeling::weights::load_model(model, source, finalizer, options);

    const auto float_type = ov::element::f32;
    auto attention_mask = ctx.parameter(Qwen3_5TextIO::kAttentionMask, ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(Qwen3_5TextIO::kPositionIds, ov::element::i64, ov::PartialShape{3, -1, -1});
    auto beam_idx = ctx.parameter(Qwen3_5TextIO::kBeamIdx, ov::element::i32, ov::PartialShape{-1});

    Tensor input_ids;
    Tensor inputs_embeds;
    if (use_inputs_embeds) {
        inputs_embeds = ctx.parameter(Qwen3_5TextIO::kInputsEmbeds, float_type, ov::PartialShape{-1, -1, cfg.text.hidden_size});
    } else {
        input_ids = ctx.parameter(Qwen3_5TextIO::kInputIds, ov::element::i64, ov::PartialShape{-1, -1});
    }

    const Tensor* visual_embeds_ptr = nullptr;
    const Tensor* visual_pos_mask_ptr = nullptr;
    Tensor visual_embeds;
    Tensor visual_pos_mask;
    if (enable_visual_inputs) {
        visual_embeds = ctx.parameter(Qwen3_5TextIO::kVisualEmbeds, float_type, ov::PartialShape{-1, -1, cfg.text.hidden_size});
        visual_pos_mask = ctx.parameter(Qwen3_5TextIO::kVisualPosMask, ov::element::boolean, ov::PartialShape{-1, -1});
        visual_embeds_ptr = &visual_embeds;
        visual_pos_mask_ptr = &visual_pos_mask;
    }

    Tensor logits;
    if (use_inputs_embeds) {
        logits = model.forward_embeds(inputs_embeds,
                                      position_ids,
                                      beam_idx,
                                      attention_mask,
                                      &attention_mask,
                                      nullptr,
                                      visual_embeds_ptr,
                                      visual_pos_mask_ptr);
    } else {
        logits = model.forward(input_ids, position_ids, beam_idx, attention_mask, &attention_mask, nullptr, visual_embeds_ptr, visual_pos_mask_ptr);
    }

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, Qwen3_5TextIO::kLogits);
    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

