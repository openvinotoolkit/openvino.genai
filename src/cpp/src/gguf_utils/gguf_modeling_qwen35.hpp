// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>

#include <openvino/openvino.hpp>
#include "openvino/opsets/opset13.hpp"

#include "gguf_utils/building_blocks.hpp"
#include "gguf_utils/gguf.hpp"

using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

namespace {

auto set_name_q35 = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

// L2 normalization along the last axis: x / max(||x||, eps)
ov::Output<ov::Node> l2_normalize(const ov::Output<ov::Node>& input, float eps) {
    auto square = std::make_shared<v1::Multiply>(input, input, AutoBroadcastType::NUMPY);
    auto sum_sq = std::make_shared<v1::ReduceSum>(
        square,
        std::make_shared<v0::Constant>(element::i32, Shape{1}, -1),
        true);
    auto norm = std::make_shared<v0::Sqrt>(sum_sq);
    auto eps_node = std::make_shared<v0::Constant>(element::f32, Shape{}, eps);
    auto norm_clamped = std::make_shared<v1::Maximum>(norm, eps_node, AutoBroadcastType::NUMPY);
    return std::make_shared<v1::Divide>(input, norm_clamped, AutoBroadcastType::NUMPY);
}

// Softplus: log(1 + exp(x))
ov::Output<ov::Node> softplus(const ov::Output<ov::Node>& input) {
    auto exp_x = std::make_shared<v0::Exp>(input);
    auto one = std::make_shared<v0::Constant>(element::f32, Shape{}, 1.0f);
    auto one_plus_exp = std::make_shared<v1::Add>(exp_x, one, AutoBroadcastType::NUMPY);
    return std::make_shared<v0::Log>(one_plus_exp);
}

// Sigmoid: 1 / (1 + exp(-x))
ov::Output<ov::Node> sigmoid_op(const ov::Output<ov::Node>& input) {
    return std::make_shared<v0::Sigmoid>(input);
}

// 1D causal convolution using stored state
// conv_input: [batch, channels, seq_len]
// conv_kernel: [channels, kernel_size] (depthwise)
// conv_state: ReadValue with shape [batch, channels, kernel_size - 1]
// Returns: (conv_output [batch, channels, seq_len], updated_state, assign_sink)
std::tuple<ov::Output<ov::Node>, ov::SinkVector>
build_conv1d_with_state(
    const ov::Output<ov::Node>& conv_input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::string& conv_weight_key,
    int conv_channels,
    int kernel_size,
    int layer_idx,
    const ov::Output<ov::Node>& batch_dim,
    const ov::Output<ov::Node>& beam_idx) {

    // Create conv state variable
    std::string state_name = "conv_state." + std::to_string(layer_idx);
    auto var_info = ov::op::util::VariableInfo{
        ov::PartialShape{-1, conv_channels, kernel_size - 1},
        ov::element::f32,
        state_name
    };
    auto var = std::make_shared<ov::op::util::Variable>(var_info);

    // Default state: zeros
    auto zero_const = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.0f);
    auto state_shape = std::make_shared<ov::opset13::Concat>(OutputVector{
        batch_dim,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, conv_channels),
        std::make_shared<v0::Constant>(element::i64, Shape{1}, kernel_size - 1)
    }, 0);
    auto state_default = std::make_shared<v3::Broadcast>(zero_const, state_shape, BroadcastType::NUMPY);

    auto read_state = std::make_shared<v6::ReadValue>(state_default, var);
    auto gathered_state = std::make_shared<v8::Gather>(read_state, beam_idx,
        std::make_shared<v0::Constant>(element::i64, Shape{}, 0), 0);

    // Concatenate state with input: [batch, channels, kernel_size-1 + seq_len]
    auto padded_input = std::make_shared<ov::opset13::Concat>(OutputVector{gathered_state, conv_input}, 2);

    // Get conv kernel weights
    // After GGUF loading (dimension reversal), shape is [conv_channels, kernel_size]
    // For depthwise GroupConvolution we need [groups=conv_channels, 1, 1, kernel_size]
    auto weight_tensor = consts.at(conv_weight_key + ".weight");
    auto kernel_shape = weight_tensor.get_shape();
    int64_t n_channels = kernel_shape[0];
    int64_t k_size = kernel_shape[1];

    // Reshape weight for GroupConvolution: [groups=conv_channels, 1, 1, 1, kernel_size]
    ov::Tensor reshaped_weight(element::f32, ov::Shape{(size_t)n_channels, 1, 1, 1, (size_t)k_size});
    if (weight_tensor.get_element_type() == element::f16) {
        const uint16_t* f16_data = weight_tensor.data<uint16_t>();
        float* dst = reshaped_weight.data<float>();
        for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
            ov::float16 val;
            std::memcpy(&val, &f16_data[i], sizeof(uint16_t));
            dst[i] = static_cast<float>(val);
        }
    } else {
        const float* src_data = weight_tensor.data<float>();
        float* dst = reshaped_weight.data<float>();
        std::memcpy(dst, src_data, weight_tensor.get_byte_size());
    }

    auto kernel_const = std::make_shared<v0::Constant>(reshaped_weight);

    // Add height dimension for 2D conv: [batch, channels, 1, seq_len+kernel-1]
    auto unsqueeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto padded_4d = std::make_shared<v0::Unsqueeze>(padded_input, unsqueeze_axis);

    // GroupConvolution (depthwise): groups = conv_channels
    auto conv = std::make_shared<v1::GroupConvolution>(
        padded_4d, kernel_const,
        ov::Strides{1, 1},
        ov::CoordinateDiff{0, 0},
        ov::CoordinateDiff{0, 0},
        ov::Strides{1, 1});

    // Remove height dimension
    auto squeeze_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto conv_output = std::make_shared<v0::Squeeze>(conv, squeeze_axis);

    // Update state: take the last (kernel_size-1) elements from padded_input along dim 2
    auto neg_k_minus_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, -(kernel_size - 1));
    auto int_max = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::numeric_limits<int64_t>::max());
    auto step = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto axis_2 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto new_state = std::make_shared<ov::opset13::Slice>(padded_input, neg_k_minus_1, int_max, step, axis_2);

    auto state_assign = std::make_shared<ov::opset13::Assign>(new_state, var);

    return {conv_output, {state_assign}};
}

// Build a single full-attention layer for Qwen3.5
// Q projection outputs [query, gate] concatenated; gate is sigmoid-applied to attention output
std::tuple<ov::Output<ov::Node>, ov::SinkVector, ov::Output<ov::Node>,
           std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>, std::shared_ptr<ov::Node>>
qwen35_full_attn_layer(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    int layer_idx,
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& attn_mask,
    const ov::Output<ov::Node>& causal_mask_in,
    const ov::Output<ov::Node>& position_ids,
    const ov::Output<ov::Node>& rope_const,
    const ov::Output<ov::Node>& beam_idx,
    const ov::Output<ov::Node>& batch_dim,
    const ov::Output<ov::Node>& hidden_dim,
    const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& cos_sin_cached,
    const std::shared_ptr<ov::Node>& output_shape) {

    int num_heads = std::get<int>(configs.at("head_num"));
    int head_dim = std::get<int>(configs.at("head_size"));
    int num_heads_kv = std::get<int>(configs.at("head_num_kv"));
    float rms_norm_eps = std::get<float>(configs.at("rms_norm_eps"));

    std::string layer_prefix = format("model.layers[%d]", layer_idx);

    // Input LayerNorm
    auto input_layernorm = make_rms_norm(layer_prefix + ".input_layernorm",
                                         hidden_states,
                                         consts,
                                         rms_norm_eps);

    // Q projection: outputs [n_embd_head * 2 * n_head] = query + gate concatenated
    // In Qwen3.5, Q proj output is 2x normal size (query + gate)
    auto q_full = make_fc(
        layer_prefix + ".self_attn.q_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.q_proj.qtype"));

    // Split q_full into query and gate along last dimension
    // q_full shape: [batch, seq, num_heads * head_dim * 2]
    // Reshape to [batch, seq, num_heads, head_dim * 2]
    auto reshape_4d = std::make_shared<v0::Constant>(
        element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_heads, head_dim * 2});
    auto q_full_4d = std::make_shared<v1::Reshape>(q_full, reshape_4d, true);

    // Slice query part: [..., :head_dim]
    auto slice_start_0 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto slice_end_hd = std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim);
    auto slice_step = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto slice_axis_3 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 3);
    auto q_part = std::make_shared<ov::opset13::Slice>(q_full_4d, slice_start_0, slice_end_hd, slice_step, slice_axis_3);

    // Slice gate part: [..., head_dim:]
    auto slice_start_hd = std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim);
    auto slice_end_max = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::numeric_limits<int64_t>::max());
    auto gate_part = std::make_shared<ov::opset13::Slice>(q_full_4d, slice_start_hd, slice_end_max, slice_step, slice_axis_3);

    // Apply Q norm (RMS norm on the query part)
    auto q_normed = make_rms_norm_qwen3(layer_prefix + ".self_attn.q_norm", q_part, consts, rms_norm_eps);

    // Transpose Q: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto q_transposed = std::make_shared<v1::Transpose>(q_normed, transpose_order);

    // K projection
    auto k_proj = make_fc(
        layer_prefix + ".self_attn.k_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.k_proj.qtype"));

    // Reshape K to [batch, seq, num_heads_kv, head_dim]
    auto k_reshape = std::make_shared<v0::Constant>(
        element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_heads_kv, head_dim});
    auto k_reshaped = std::make_shared<v1::Reshape>(k_proj, k_reshape, true);

    // Apply K norm
    auto k_normed = make_rms_norm_qwen3(layer_prefix + ".self_attn.k_norm", k_reshaped, consts, rms_norm_eps);

    // Transpose K
    auto k_transposed = std::make_shared<v1::Transpose>(k_normed, transpose_order);

    // V projection
    auto v_proj = make_fc(
        layer_prefix + ".self_attn.v_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.v_proj.qtype"));

    // Reshape and transpose V
    auto v_reshape = std::make_shared<v0::Constant>(
        element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_heads_kv, head_dim});
    auto v_reshaped = std::make_shared<v1::Reshape>(v_proj, v_reshape, true);
    auto v_transposed = std::make_shared<v1::Transpose>(v_reshaped, transpose_order);

    // Apply RoPE
    Output<Node> cos, sin;
    if (cos_sin_cached.first.get_node() == nullptr) {
        std::tie(cos, sin) = rope_emb(v_transposed, rope_const, position_ids, batch_dim);
    }
    auto [q_rot, k_rot, new_cos_sin] = apply_rotary_pos_emb(
        q_transposed, k_transposed, cos, sin, head_dim, hidden_dim, cos_sin_cached);

    // KV Cache
    auto create_cache = [&](const std::string& name, int n_kv_heads) {
        auto var_info = ov::op::util::VariableInfo{
            ov::PartialShape{-1, n_kv_heads, -1, head_dim},
            ov::element::f32,
            name
        };
        auto var = std::make_shared<ov::op::util::Variable>(var_info);
        auto zero_c = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.0f);
        auto cache_shape = std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, n_kv_heads),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0);
        auto cache_default = std::make_shared<v3::Broadcast>(zero_c, cache_shape, BroadcastType::NUMPY);
        auto read_value = std::make_shared<v6::ReadValue>(cache_default, var);
        auto gathered = std::make_shared<v8::Gather>(read_value, beam_idx,
            std::make_shared<v0::Constant>(element::i64, Shape{}, 0), 0);
        return std::make_pair(var, gathered);
    };

    auto k_cache = create_cache(
        "past_key_values." + std::to_string(layer_idx) + ".keypresent." + std::to_string(layer_idx) + ".key",
        num_heads_kv);
    auto v_cache = create_cache(
        "past_key_values." + std::to_string(layer_idx) + ".valuepresent." + std::to_string(layer_idx) + ".key",
        num_heads_kv);

    auto k_combined = std::make_shared<ov::opset13::Concat>(OutputVector{k_cache.second, k_rot}, 2);
    auto v_combined = std::make_shared<ov::opset13::Concat>(OutputVector{v_cache.second, v_transposed}, 2);

    auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined, k_cache.first);
    auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined, v_cache.first);

    // GQA: expand K/V if needed
    Output<Node> k_for_attn = k_combined;
    Output<Node> v_for_attn = v_combined;
    if (num_heads != num_heads_kv) {
        int kv_per_head = num_heads / num_heads_kv;
        auto unsqueeze_ax = std::make_shared<v0::Constant>(element::i64, Shape{}, 2);
        auto k_unsq = std::make_shared<v0::Unsqueeze>(k_combined, unsqueeze_ax);
        auto v_unsq = std::make_shared<v0::Unsqueeze>(v_combined, unsqueeze_ax);

        auto bcast_shape = std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, kv_per_head),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0);

        k_for_attn = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(k_unsq, bcast_shape, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true);

        v_for_attn = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(v_unsq, bcast_shape, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true);
    }

    // Causal mask
    Output<Node> final_mask = causal_mask_in;
    if (causal_mask_in.get_node() == nullptr) {
        auto input_shape_node = std::make_shared<v3::ShapeOf>(input_layernorm);
        final_mask = causal_mask(attn_mask, k_cache.second, hidden_dim, input_shape_node);
    }

    // Scaled Dot Product Attention
    auto attention = std::make_shared<ScaledDotProductAttention>(
        q_rot, k_for_attn, v_for_attn, final_mask, false);

    // Transpose attention output: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
    auto attn_transpose = std::make_shared<v1::Transpose>(attention,
        std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3}));

    // Apply sigmoid gate
    // gate_part shape: [batch, seq, num_heads, head_dim] - same layout as attn_transpose after transpose
    auto gate_sigmoid = std::make_shared<v0::Sigmoid>(gate_part);
    auto gated_attn = std::make_shared<v1::Multiply>(attn_transpose, gate_sigmoid, AutoBroadcastType::NUMPY);

    // Handle output shape
    std::shared_ptr<ov::Node> final_output_shape = output_shape;
    if (!output_shape) {
        auto input_shape_for_out = std::make_shared<v3::ShapeOf>(input_layernorm);
        auto indices = std::make_shared<v0::Constant>(
            element::i64, Shape{2}, std::vector<int64_t>{0, 1});
        auto gathered = std::make_shared<v8::Gather>(
            input_shape_for_out, indices,
            std::make_shared<v0::Constant>(element::i64, Shape{}, 0));
        auto minus_one = std::make_shared<v0::Constant>(element::i64, Shape{1}, -1);
        final_output_shape = std::make_shared<v0::Concat>(OutputVector{gathered, minus_one}, 0);
    }

    // Reshape to [batch, seq, hidden_size]
    auto attn_output = std::make_shared<v1::Reshape>(gated_attn, final_output_shape, false);

    // Output projection
    auto o_proj = make_fc(
        layer_prefix + ".self_attn.o_proj",
        attn_output,
        consts,
        qtypes.at(layer_prefix + ".self_attn.o_proj.qtype"));

    // Residual connection
    auto attn_residual = std::make_shared<v1::Add>(hidden_states, o_proj, AutoBroadcastType::NUMPY);

    // Post-attention LayerNorm
    auto post_attn_norm = make_rms_norm(
        layer_prefix + ".post_attention_layernorm",
        attn_residual,
        consts,
        rms_norm_eps);

    // MLP (SwiGLU)
    auto gate_proj_mlp = make_fc(
        layer_prefix + ".mlp.gate_proj",
        post_attn_norm,
        consts,
        qtypes.at(layer_prefix + ".mlp.gate_proj.qtype"));
    auto silu = std::make_shared<v4::Swish>(gate_proj_mlp);
    auto up_proj = make_fc(
        layer_prefix + ".mlp.up_proj",
        post_attn_norm,
        consts,
        qtypes.at(layer_prefix + ".mlp.up_proj.qtype"));
    auto mul = std::make_shared<v1::Multiply>(silu, up_proj, AutoBroadcastType::NUMPY);
    auto down_proj = make_fc(
        layer_prefix + ".mlp.down_proj",
        mul,
        consts,
        qtypes.at(layer_prefix + ".mlp.down_proj.qtype"));

    // Final residual
    auto output = std::make_shared<v1::Add>(attn_residual, down_proj, AutoBroadcastType::NUMPY);

    ov::SinkVector sinks = {k_assign, v_assign};

    return {output, sinks, final_mask, new_cos_sin, final_output_shape};
}

// Build a single linear attention (gated delta net) layer for Qwen3.5
std::tuple<ov::Output<ov::Node>, ov::SinkVector>
qwen35_linear_attn_layer(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    int layer_idx,
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& beam_idx,
    const ov::Output<ov::Node>& batch_dim) {

    float rms_norm_eps = std::get<float>(configs.at("rms_norm_eps"));
    int ssm_d_conv = std::get<int>(configs.at("ssm_d_conv"));
    int ssm_d_state = std::get<int>(configs.at("ssm_d_state"));
    int ssm_dt_rank = std::get<int>(configs.at("ssm_dt_rank"));
    int ssm_n_group = std::get<int>(configs.at("ssm_n_group"));
    int hidden_size = std::get<int>(configs.at("hidden_size"));

    int head_k_dim = ssm_d_state;
    int num_k_heads = ssm_n_group;
    int num_v_heads = ssm_dt_rank;
    int head_v_dim = ssm_d_state;
    int key_dim = head_k_dim * num_k_heads;
    int value_dim = head_v_dim * num_v_heads;
    int conv_dim = key_dim * 2 + value_dim;

    std::string layer_prefix = format("model.layers[%d]", layer_idx);

    // Input LayerNorm
    auto input_layernorm = make_rms_norm(layer_prefix + ".input_layernorm",
                                         hidden_states,
                                         consts,
                                         rms_norm_eps);

    // QKV projection: [batch, seq, conv_dim]
    auto qkv_proj = make_fc(
        layer_prefix + ".linear_attn.qkv_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".linear_attn.qkv_proj.qtype"));

    // Gate Z projection: [batch, seq, value_dim]
    auto z_proj = make_fc(
        layer_prefix + ".linear_attn.gate_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".linear_attn.gate_proj.qtype"));

    // Beta projection: [batch, seq, num_v_heads] -> sigmoid
    auto beta_proj = make_fc(
        layer_prefix + ".linear_attn.beta_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".linear_attn.beta_proj.qtype"));
    auto beta = sigmoid_op(beta_proj);

    // Alpha projection: [batch, seq, num_v_heads]
    auto alpha_proj = make_fc(
        layer_prefix + ".linear_attn.alpha_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".linear_attn.alpha_proj.qtype"));

    // Alpha + dt_bias -> softplus -> gate (decay)
    auto dt_bias_tensor = consts.at(layer_prefix + ".linear_attn.dt_bias");
    dt_bias_tensor.set_shape(ov::Shape{1, 1, (size_t)ssm_dt_rank});
    auto dt_bias_const = std::make_shared<v0::Constant>(dt_bias_tensor);
    auto dt_bias_f32 = std::make_shared<v0::Convert>(dt_bias_const, element::f32);
    auto alpha_biased = std::make_shared<v1::Add>(alpha_proj, dt_bias_f32, AutoBroadcastType::NUMPY);
    auto alpha_softplus = softplus(alpha_biased);

    // gate = softplus(alpha + bias) * A_log (decay)
    auto a_log_tensor = consts.at(layer_prefix + ".linear_attn.a_log");
    a_log_tensor.set_shape(ov::Shape{1, 1, (size_t)ssm_dt_rank});
    auto a_log_const = std::make_shared<v0::Constant>(a_log_tensor);
    auto a_log_f32 = std::make_shared<v0::Convert>(a_log_const, element::f32);
    auto decay_gate = std::make_shared<v1::Multiply>(alpha_softplus, a_log_f32, AutoBroadcastType::NUMPY);

    // Transpose QKV for conv: [batch, seq, conv_dim] -> [batch, conv_dim, seq]
    auto qkv_transposed = std::make_shared<v1::Transpose>(qkv_proj,
        std::make_shared<v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 1}));

    // 1D causal convolution
    auto [conv_output, conv_sinks] = build_conv1d_with_state(
        qkv_transposed, consts,
        layer_prefix + ".linear_attn.conv1d",
        conv_dim, ssm_d_conv, layer_idx, batch_dim, beam_idx);

    // Transpose back: [batch, conv_dim, seq] -> [batch, seq, conv_dim]
    auto conv_out_transposed = std::make_shared<v1::Transpose>(conv_output,
        std::make_shared<v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 1}));

    // SiLU activation
    auto conv_silu = std::make_shared<v4::Swish>(conv_out_transposed);

    // Split conv output into Q, K, V along last dimension
    // Q: [batch, seq, key_dim], K: [batch, seq, key_dim], V: [batch, seq, value_dim]
    auto axis_last = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto q_start = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto q_end = std::make_shared<v0::Constant>(element::i64, Shape{1}, key_dim);
    auto k_start = q_end;
    auto k_end = std::make_shared<v0::Constant>(element::i64, Shape{1}, key_dim * 2);
    auto v_start = k_end;
    auto v_end = std::make_shared<v0::Constant>(element::i64, Shape{1}, key_dim * 2 + value_dim);
    auto step_1 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 1);

    auto q_conv = std::make_shared<ov::opset13::Slice>(conv_silu, q_start, q_end, step_1, axis_last);
    auto k_conv = std::make_shared<ov::opset13::Slice>(conv_silu, k_start, k_end, step_1, axis_last);
    auto v_conv = std::make_shared<ov::opset13::Slice>(conv_silu, v_start, v_end, step_1, axis_last);

    // Reshape Q, K, V to multi-head first, then L2 normalize per head
    // [batch, seq, dim] -> [batch, seq, num_heads, head_dim]
    auto q_reshape_4d = std::make_shared<v0::Constant>(
        element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_k_heads, head_k_dim});
    auto k_reshape_4d = std::make_shared<v0::Constant>(
        element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_k_heads, head_k_dim});
    auto v_reshape_4d = std::make_shared<v0::Constant>(
        element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_v_heads, head_v_dim});

    auto q_4d_raw = std::make_shared<v1::Reshape>(q_conv, q_reshape_4d, true);
    auto k_4d_raw = std::make_shared<v1::Reshape>(k_conv, k_reshape_4d, true);
    auto v_4d = std::make_shared<v1::Reshape>(v_conv, v_reshape_4d, true);

    // L2 normalize Q and K per head (along last axis = head_dim)
    auto q_4d = l2_normalize(q_4d_raw, rms_norm_eps);
    auto k_4d = l2_normalize(k_4d_raw, rms_norm_eps);

    // If num_k_heads != num_v_heads, repeat Q and K to match V heads
    Output<Node> q_final = q_4d;
    Output<Node> k_final = k_4d;
    if (num_k_heads != num_v_heads) {
        int repeat_factor = num_v_heads / num_k_heads;
        // Tile Q along head dimension
        auto tile_shape = std::make_shared<v0::Constant>(
            element::i64, Shape{4}, std::vector<int64_t>{1, 1, repeat_factor, 1});
        q_final = std::make_shared<v0::Tile>(q_4d, tile_shape);
        k_final = std::make_shared<v0::Tile>(k_4d, tile_shape);
    }

    // Transpose to [batch, heads, seq, head_dim]
    auto perm_0213 = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto q_t = std::make_shared<v1::Transpose>(q_final, perm_0213);
    auto k_t = std::make_shared<v1::Transpose>(k_final, perm_0213);
    auto v_t = std::make_shared<v1::Transpose>(v_4d, perm_0213);

    // --- Recurrent state (delta net) ---
    // State shape: [batch, num_v_heads, head_v_dim, head_k_dim]
    std::string state_name = "ssm_state." + std::to_string(layer_idx);
    auto state_var_info = ov::op::util::VariableInfo{
        ov::PartialShape{-1, num_v_heads, head_v_dim, head_k_dim},
        ov::element::f32,
        state_name
    };
    auto state_var = std::make_shared<ov::op::util::Variable>(state_var_info);
    auto zero_c = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.0f);
    auto ssm_state_shape = std::make_shared<ov::opset13::Concat>(OutputVector{
        batch_dim,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, num_v_heads),
        std::make_shared<v0::Constant>(element::i64, Shape{1}, head_v_dim),
        std::make_shared<v0::Constant>(element::i64, Shape{1}, head_k_dim)
    }, 0);
    auto state_default = std::make_shared<v3::Broadcast>(zero_c, ssm_state_shape, BroadcastType::NUMPY);
    auto read_state = std::make_shared<v6::ReadValue>(state_default, state_var);
    auto state_gathered = std::make_shared<v8::Gather>(read_state, beam_idx,
        std::make_shared<v0::Constant>(element::i64, Shape{}, 0), 0);

    // Scale q by 1/sqrt(head_k_dim)
    float scale = 1.0f / std::sqrt((float)head_k_dim);
    auto scale_const = std::make_shared<v0::Constant>(element::f32, Shape{}, scale);
    auto q_scaled = std::make_shared<v1::Multiply>(q_t, scale_const, AutoBroadcastType::NUMPY);

    // Reshape gate and beta: [batch, seq, num_v_heads] -> [batch, num_v_heads, seq]
    auto decay_gate_t = std::make_shared<v1::Transpose>(decay_gate,
        std::make_shared<v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 1}));
    auto beta_t = std::make_shared<v1::Transpose>(beta,
        std::make_shared<v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 1}));

    // Unsqueeze gate and beta to [batch, num_v_heads, seq, 1, 1] and [batch, num_v_heads, seq, 1, 1]
    auto unsq_ax_3 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 3);
    auto unsq_ax_4 = std::make_shared<v0::Constant>(element::i64, Shape{1}, 4);
    auto gate_5d = std::make_shared<v0::Unsqueeze>(
        std::make_shared<v0::Unsqueeze>(decay_gate_t, unsq_ax_3), unsq_ax_4);
    auto beta_5d = std::make_shared<v0::Unsqueeze>(
        std::make_shared<v0::Unsqueeze>(beta_t, unsq_ax_3), unsq_ax_4);

    // Delta net recurrence using v5::Loop to iterate over seq dimension.
    // For each timestep t:
    //   state = exp(gate_t) * state
    //   sk = state @ k_t^T   (matrix-vector product)
    //   delta = beta_t * (v_t^T - sk)
    //   state = state + delta @ k_t   (rank-1 update)
    //   output_t = state @ q_t^T

    // Get trip count (seq_len) as a scalar
    auto q_shape = std::make_shared<v3::ShapeOf>(q_scaled);
    auto seq_axis_idx = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto zero_axis = std::make_shared<v0::Constant>(element::i64, Shape{}, 0);
    auto trip_count_1d = std::make_shared<v8::Gather>(q_shape, seq_axis_idx, zero_axis);
    auto trip_count_scalar = std::make_shared<v0::Squeeze>(trip_count_1d,
        std::make_shared<v0::Constant>(element::i64, Shape{1}, 0));

    auto exec_cond = std::make_shared<v0::Constant>(element::boolean, Shape{}, true);

    // Build Loop body
    // Body parameters:
    //   0: iteration counter (i64 scalar)
    //   1: execution condition (bool scalar)
    //   2: state [batch, num_v_heads, head_v_dim, head_k_dim] (carried across iterations)
    auto body_iter = std::make_shared<v0::Parameter>(element::i64, PartialShape{});
    auto body_cond = std::make_shared<v0::Parameter>(element::boolean, PartialShape{});
    auto body_state = std::make_shared<v0::Parameter>(element::f32,
        PartialShape{-1, num_v_heads, head_v_dim, head_k_dim});

    // Sliced inputs (one slice per iteration along seq dim=2):
    //   3: gate_t [batch, num_v_heads, 1, 1, 1]
    //   4: beta_t [batch, num_v_heads, 1, 1, 1]
    //   5: q_t [batch, num_v_heads, 1, head_k_dim]
    //   6: k_t [batch, num_v_heads, 1, head_k_dim]
    //   7: v_t [batch, num_v_heads, 1, head_v_dim]
    auto body_gate = std::make_shared<v0::Parameter>(element::f32,
        PartialShape{-1, num_v_heads, 1, 1, 1});
    auto body_beta = std::make_shared<v0::Parameter>(element::f32,
        PartialShape{-1, num_v_heads, 1, 1, 1});
    auto body_q = std::make_shared<v0::Parameter>(element::f32,
        PartialShape{-1, num_v_heads, 1, (int64_t)head_k_dim});
    auto body_k = std::make_shared<v0::Parameter>(element::f32,
        PartialShape{-1, num_v_heads, 1, (int64_t)head_k_dim});
    auto body_v = std::make_shared<v0::Parameter>(element::f32,
        PartialShape{-1, num_v_heads, 1, (int64_t)head_v_dim});

    // Body computation:
    // 1. Decay: state = exp(gate) * state
    auto body_exp_gate = std::make_shared<v0::Exp>(body_gate);
    // Squeeze gate from [batch, H, 1, 1, 1] to [batch, H, 1, 1] for broadcast with state [batch, H, Sv, Sk]
    auto body_exp_gate_4d = std::make_shared<v0::Squeeze>(body_exp_gate, unsq_ax_4);
    auto body_decayed = std::make_shared<v1::Multiply>(body_state, body_exp_gate_4d, AutoBroadcastType::NUMPY);

    // 2. sk = state @ k^T: [batch, H, Sv, Sk] @ [batch, H, Sk, 1] = [batch, H, Sv, 1]
    auto body_k_T = std::make_shared<v0::MatMul>(body_decayed,
        std::make_shared<v1::Transpose>(body_k,
            std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 1, 3, 2})),
        false, false);

    // 3. delta = beta * (v^T - sk)
    auto body_v_T = std::make_shared<v1::Transpose>(body_v,
        std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 1, 3, 2}));
    // v^T: [batch, H, Sv, 1], sk: [batch, H, Sv, 1]
    auto body_diff = std::make_shared<v1::Subtract>(body_v_T, body_k_T, AutoBroadcastType::NUMPY);
    // beta: [batch, H, 1, 1, 1] -> squeeze to [batch, H, 1, 1]
    auto body_beta_4d = std::make_shared<v0::Squeeze>(body_beta, unsq_ax_4);
    auto body_delta = std::make_shared<v1::Multiply>(body_diff, body_beta_4d, AutoBroadcastType::NUMPY);

    // 4. state += delta @ k: [batch, H, Sv, 1] @ [batch, H, 1, Sk] = [batch, H, Sv, Sk]
    auto body_kd = std::make_shared<v0::MatMul>(body_delta, body_k, false, false);
    auto body_new_state = std::make_shared<v1::Add>(body_decayed, body_kd, AutoBroadcastType::NUMPY);

    // 5. output = state @ q^T: [batch, H, Sv, Sk] @ [batch, H, Sk, 1] = [batch, H, Sv, 1]
    auto body_q_T = std::make_shared<v1::Transpose>(body_q,
        std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 1, 3, 2}));
    auto body_out = std::make_shared<v0::MatMul>(body_new_state, body_q_T, false, false);
    // Transpose output: [batch, H, Sv, 1] -> [batch, H, 1, Sv]
    auto body_out_t = std::make_shared<v1::Transpose>(body_out,
        std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 1, 3, 2}));

    // Body outputs: continue condition, new state, output slice
    auto body_cond_out = std::make_shared<v0::Constant>(element::boolean, Shape{}, true);

    // Create body model
    auto body_out_state = std::make_shared<v0::Result>(body_new_state);
    auto body_out_cond = std::make_shared<v0::Result>(body_cond_out);
    auto body_out_result = std::make_shared<v0::Result>(body_out_t);

    auto body = std::make_shared<ov::Model>(
        OutputVector{body_out_cond->output(0), body_out_state->output(0), body_out_result->output(0)},
        ParameterVector{body_iter, body_cond, body_state, body_gate, body_beta, body_q, body_k, body_v});

    // Create the Loop op
    auto loop = std::make_shared<v5::Loop>(trip_count_scalar, exec_cond);
    loop->set_function(body);

    // Set up inputs:
    // Merged (carried) input: state
    loop->set_merged_input(body_state, state_gathered, body_out_state);
    // Invariant inputs: none (all are sliced)
    // Sliced inputs: gate, beta, q, k, v along axis=2
    loop->set_sliced_input(body_gate, gate_5d, 0, 1, 1, -1, 2);
    loop->set_sliced_input(body_beta, beta_5d, 0, 1, 1, -1, 2);
    loop->set_sliced_input(body_q, q_scaled, 0, 1, 1, -1, 2);
    loop->set_sliced_input(body_k, k_t, 0, 1, 1, -1, 2);
    loop->set_sliced_input(body_v, v_t, 0, 1, 1, -1, 2);

    // Special body inputs
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{0, 0});

    // Outputs:
    // 1. Final state (last value of carried state)
    auto loop_final_state = loop->get_iter_value(body_out_state, -1);
    // 2. Concatenated outputs along axis=2: [batch, H, seq, Sv]
    auto loop_outputs = loop->get_concatenated_slices(body_out_result, 0, 1, 1, -1, 2);

    auto new_state = loop_final_state;
    auto attn_out_transposed = loop_outputs;

    // Assign new state
    auto state_assign = std::make_shared<ov::opset13::Assign>(new_state, state_var);

    // Gated normalization: rms_norm(attn_output) * silu(z)
    // attn_out_transposed: [batch, num_v_heads, seq, head_v_dim]
    // Apply RMS norm per head
    auto norm_weight_key = layer_prefix + ".linear_attn.norm";
    auto norm_out = make_rms_norm_qwen3(norm_weight_key, attn_out_transposed, consts, rms_norm_eps);

    // z_proj: [batch, seq, value_dim] -> reshape to [batch, seq, num_v_heads, head_v_dim]
    // -> transpose to [batch, num_v_heads, seq, head_v_dim]
    auto z_reshape = std::make_shared<v0::Constant>(
        element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_v_heads, head_v_dim});
    auto z_4d = std::make_shared<v1::Reshape>(z_proj, z_reshape, true);
    auto z_transposed = std::make_shared<v1::Transpose>(z_4d, perm_0213);

    // silu(z) * norm_out
    auto z_silu = std::make_shared<v4::Swish>(z_transposed);
    auto gated_output = std::make_shared<v1::Multiply>(norm_out, z_silu, AutoBroadcastType::NUMPY);

    // Reshape: [batch, num_v_heads, seq, head_v_dim] -> [batch, seq, num_v_heads * head_v_dim]
    auto final_transpose = std::make_shared<v1::Transpose>(gated_output,
        std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3}));
    auto final_reshape = std::make_shared<v0::Constant>(
        element::i64, Shape{3}, std::vector<int64_t>{0, 0, -1});
    auto linear_attn_out = std::make_shared<v1::Reshape>(final_transpose, final_reshape, true);

    // Output projection
    auto out_proj = make_fc(
        layer_prefix + ".linear_attn.out_proj",
        linear_attn_out,
        consts,
        qtypes.at(layer_prefix + ".linear_attn.out_proj.qtype"));

    // Residual connection
    auto attn_residual = std::make_shared<v1::Add>(hidden_states, out_proj, AutoBroadcastType::NUMPY);

    // Post-attention LayerNorm
    auto post_attn_norm = make_rms_norm(
        layer_prefix + ".post_attention_layernorm",
        attn_residual,
        consts,
        rms_norm_eps);

    // MLP (SwiGLU)
    auto gate_proj_mlp = make_fc(
        layer_prefix + ".mlp.gate_proj",
        post_attn_norm,
        consts,
        qtypes.at(layer_prefix + ".mlp.gate_proj.qtype"));
    auto silu_mlp = std::make_shared<v4::Swish>(gate_proj_mlp);
    auto up_proj = make_fc(
        layer_prefix + ".mlp.up_proj",
        post_attn_norm,
        consts,
        qtypes.at(layer_prefix + ".mlp.up_proj.qtype"));
    auto mul_mlp = std::make_shared<v1::Multiply>(silu_mlp, up_proj, AutoBroadcastType::NUMPY);
    auto down_proj = make_fc(
        layer_prefix + ".mlp.down_proj",
        mul_mlp,
        consts,
        qtypes.at(layer_prefix + ".mlp.down_proj.qtype"));

    // Final residual
    auto output = std::make_shared<v1::Add>(attn_residual, down_proj, AutoBroadcastType::NUMPY);

    // Collect all sinks
    ov::SinkVector sinks = conv_sinks;
    sinks.push_back(state_assign);

    return {output, sinks};
}

} // anonymous namespace

inline std::shared_ptr<ov::Model> create_qwen35_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {

    int layer_num = std::get<int>(configs.at("layer_num"));
    int full_attn_interval = 4;
    if (configs.count("full_attn_interval")) {
        full_attn_interval = std::get<int>(configs.at("full_attn_interval"));
    }

    // Create input parameters
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        element::i64, PartialShape{-1, -1});
    set_name_q35(input_ids, "input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        element::i64, PartialShape{-1, -1});
    set_name_q35(attention_mask, "attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        element::i64, PartialShape{-1, -1});
    set_name_q35(position_ids, "position_ids");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        element::i32, PartialShape{-1});
    set_name_q35(beam_idx, "beam_idx");

    // Embedding layer
    auto [inputs_embeds, embeddings] = make_embedding(
        "model.embed_tokens",
        input_ids->output(0),
        consts,
        qtypes.at("model.embed_tokens.qtype"));

    auto hidden_states = inputs_embeds;

    // Initialize RoPE
    auto rope_const = init_rope(
        std::get<int>(configs.at("head_size")),
        std::get<int>(configs.at("max_position_embeddings")),
        std::get<float>(configs.at("rope_freq_base")));

    // Get batch size
    auto input_shape = std::make_shared<v3::ShapeOf>(input_ids);
    auto batch_axis = std::make_shared<v0::Constant>(element::i64, Shape{1}, 0);
    auto batch_size = std::make_shared<v8::Gather>(input_shape, batch_axis, batch_axis);

    auto hidden_dim = std::make_shared<v0::Constant>(element::i64, Shape{1}, 3);

    // Process layers
    ov::SinkVector sinks;
    ov::Output<ov::Node> causal_mask_cached;
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> cos_sin_cached;
    std::shared_ptr<ov::Node> output_shape_cached = nullptr;

    for (int i = 0; i < layer_num; ++i) {
        bool is_recurrent = ((i + 1) % full_attn_interval != 0);

        if (is_recurrent) {
            // Linear attention (gated delta net) layer
            auto [new_hidden, layer_sinks] = qwen35_linear_attn_layer(
                configs, consts, qtypes, i, hidden_states, beam_idx, batch_size);

            hidden_states = new_hidden;
            sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
        } else {
            // Full attention layer
            auto [new_hidden, layer_sinks, new_mask, new_cos_sin, new_shape] =
                qwen35_full_attn_layer(
                    configs, consts, qtypes, i, hidden_states,
                    attention_mask, causal_mask_cached, position_ids,
                    rope_const, beam_idx, batch_size, hidden_dim,
                    cos_sin_cached, output_shape_cached);

            hidden_states = new_hidden;
            causal_mask_cached = new_mask;
            cos_sin_cached = new_cos_sin;
            output_shape_cached = new_shape;
            sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
        }
    }

    // Final layer norm
    auto final_norm = make_rms_norm(
        "model.norm",
        hidden_states,
        consts,
        std::get<float>(configs.at("rms_norm_eps")));

    // LM head
    auto embed_out = make_lm_head(
        "lm_head",
        final_norm,
        consts,
        embeddings,
        qtypes.at("lm_head.qtype"));

    // Create result
    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    set_name_q35(logits, "logits");

    // Create model
    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    auto model = std::make_shared<ov::Model>(OutputVector({logits->output(0)}), sinks, inputs);

    // Set runtime options
    if (std::get<int>(configs.at("file_type")) == 1 || std::get<int>(configs.at("file_type")) == 0) {
        model->set_rt_info(element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return model;
}
