// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <math.h>
#include <iostream>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"

#include "gguf_utils/building_blocks.hpp"

using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

static const size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

Output<ov::Node> causal_mask(
    const Output<ov::Node>& attention_mask,
    const Output<ov::Node>& keys,
    const Output<ov::Node>& hidden_dim,
    const Output<ov::Node>& input_shape) {

    // Extract shape of attention mask
    auto t130 = std::make_shared<v3::ShapeOf>(attention_mask, element::i64);
    auto t131 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t132 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t133 = std::make_shared<v8::Gather>(t130, t131, t132);

    // Reshape and construct new shapes
    auto t134 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t135 = std::make_shared<v1::Reshape>(t133, t134, false);
    auto t40 = input_shape;
    auto index_1 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto axis_0 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t127 = std::make_shared<v8::Gather>(t40, index_1, axis_0);
    auto t129 = std::make_shared<v1::Reshape>(t127, t134, false);
    auto t136 = std::make_shared<v0::Concat>(OutputVector{t129, t135}, 0);
    auto min_shape_val = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 1});
    auto t137 = std::make_shared<v1::Maximum>(min_shape_val, t136, AutoBroadcastType::NUMPY);
    auto const_neg65504 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, -65504.0f);
    auto t138 = std::make_shared<v3::Broadcast>(const_neg65504, t137, BroadcastType::NUMPY);

    // Create upper triangular mask for causal masking
    auto t139 = std::make_shared<v3::ShapeOf>(t138, element::i32);
    auto t140 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t141 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t142 = std::make_shared<v8::Gather>(t139, t140, t141, 0);
    auto t143 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);

    // Define ranges for the causal mask
    auto zero_const = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
    auto t144 = std::make_shared<v4::Range>(zero_const, t142, t143, element::i32);
    auto axes_zero = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0);
    auto t145 = std::make_shared<v0::Unsqueeze>(t144, axes_zero);
    auto t146 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);
    auto t147 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
    auto t148 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);

    // Broadcast causal mask
    auto t149 = std::make_shared<v8::Gather>(t139, t147, t148);
    auto t150 = std::make_shared<v1::Add>(t149, t146, AutoBroadcastType::NUMPY);
    auto t151 = std::make_shared<v4::Range>(t146, t150, t143, element::i32);
    auto t152 = std::make_shared<v0::Unsqueeze>(t151, t143);
    auto t153 = std::make_shared<v1::GreaterEqual>(t145, t152, AutoBroadcastType::NUMPY);

    // Create a causal mask using a selective operation
    auto t154 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, 0.0f);
    auto t155 = std::make_shared<v1::Select>(t153, t138, t154, AutoBroadcastType::NUMPY);

    // Next branch
    auto t156 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
    auto t157 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);
    auto t158 = std::make_shared<v4::Range>(t156, t133, t157, element::f32);
    auto t159 = std::make_shared<v0::Convert>(t158, element::i64);
    auto t160 = std::make_shared<v0::Convert>(t159, element::f32);
    auto t161 = std::make_shared<v3::ShapeOf>(keys, element::i64);
    auto t162 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 2);
    auto t163 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t164 = std::make_shared<v8::Gather>(t161, t162, t163, 0);
    auto t165 = std::make_shared<v1::Add>(t164, t127, AutoBroadcastType::NUMPY);
    auto t166 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 1);
    auto t167 = std::make_shared<v4::Range>(t164, t165, t166, element::f32);
    auto t168 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
    auto t169 = std::make_shared<v1::Reshape>(t167, t168, false);
    auto t170 = std::make_shared<v1::Greater>(t160, t169, AutoBroadcastType::NUMPY);
    auto t171 = std::make_shared<v0::Convert>(t170, element::f32);

    auto t172 = std::make_shared<v1::Multiply>(t155, t171, AutoBroadcastType::NUMPY);
    auto t173 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t174 = std::make_shared<v0::Unsqueeze>(t172, t173);
    auto t48 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t175 = std::make_shared<v0::Unsqueeze>(t174, t48);
    auto t41 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t42 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t43 = std::make_shared<v8::Gather>(input_shape, t41, t42);
    auto t176 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t177 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t178 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t179 = std::make_shared<v0::Concat>(OutputVector{t43, t176, t177, t178}, 0);
    auto t180 = std::make_shared<v3::Broadcast>(t175, t179, BroadcastType::BIDIRECTIONAL);
    auto t181 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto t182 = std::make_shared<v1::Reshape>(t180, t181, false);
    auto t183 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 0);
    auto t184 = std::make_shared<v3::ShapeOf>(t180, element::i64);
    auto t185 = std::make_shared<v1::ReduceProd>(t184, t183, false);
    auto t186 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1);
    auto t187 = std::make_shared<v4::Range>(t183, t185, t186, element::i64);
    auto t188 = std::make_shared<v1::Reshape>(t187, t184, false);
    auto t189 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t190 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t191 = std::make_shared<ov::opset13::Slice>(t188, t189, t135, t190, hidden_dim);
    auto t192 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
    auto t193 = std::make_shared<v1::Reshape>(t191, t192, false);
    auto t194 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t195 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t196 = std::make_shared<ov::opset13::Slice>(t180, t194, t135, t195, hidden_dim);

    auto t197 = std::make_shared<v0::Unsqueeze>(attention_mask, t48);
    auto t198 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 2);
    auto t199 = std::make_shared<v0::Unsqueeze>(t197, t198);
    auto t200 = std::make_shared<v0::Convert>(t199, element::f32);
    auto t201 = std::make_shared<v1::Add>(t196, t200, AutoBroadcastType::NUMPY);
    auto t202 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1,1,1,1}, std::vector<float>{0.0f});
    auto t203 = std::make_shared<v1::Equal>(t201, t202, AutoBroadcastType::NUMPY);
    auto t204 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, -65504.0f);
    auto t205 = std::make_shared<v1::Select>(t203, t204, t196, AutoBroadcastType::NUMPY);
    auto t206 = std::make_shared<v3::ShapeOf>(t196, element::i64);
    auto t207 = std::make_shared<v3::Broadcast>(t205, t206, BroadcastModeSpec(BroadcastType::NUMPY));
    auto t208 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto t209 = std::make_shared<v1::Reshape>(t207, t208, false);
    auto t210 = std::make_shared<v15::ScatterNDUpdate>(t182, t193, t209);
    auto t211 = std::make_shared<v1::Reshape>(t210, t184, false);
    auto t212 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t213 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t214 = std::make_shared<v1::Reshape>(t164, t213, false);
    auto t215 = std::make_shared<v1::Add>(t214, t129, AutoBroadcastType::NUMPY);
    auto t216 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t217 = std::make_shared<ov::opset13::Slice>(t211, t212, t215, t216, hidden_dim);

    return t217;
}

// Rotate half the hidden dimensions of the input tensor
Output<ov::Node> rotate_half(const Output<ov::Node>& x, int64_t head_size, const Output<Node>& axis) {
    auto t58 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{head_size / 2});
    auto t59 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{9223372036854775807});
    auto t60 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});

    // Slice second half
    auto t62 = std::make_shared<ov::opset13::Slice>(x, t58, t59, t60, axis);
    
    // Multiply by -1
    auto t63 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1,1,1,1}, std::vector<float>{-1.0f});
    auto t64 = std::make_shared<v1::Multiply>(t62, t63);
    
    // Slice first half
    auto t65 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t66 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{head_size / 2});
    auto t67 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t68 = std::make_shared<ov::opset13::Slice>(x, t65, t66, t67, axis);
    auto rotated = std::make_shared<v0::Concat>(ov::OutputVector{t64, t68}, -1);

    return rotated;
}

// Apply Rotary Position Embedding to query and key tensors
std::tuple<Output<ov::Node>, Output<ov::Node>, std::pair<Output<ov::Node>, Output<ov::Node>>> 
    apply_rotary_pos_emb(
        const Output<ov::Node>& q, 
        const Output<ov::Node>& k,
        const Output<ov::Node>& cos,
        const Output<ov::Node>& sin,
        int64_t head_size,
        const Output<Node>& hidden_dim,
        const std::pair<Output<ov::Node>, Output<ov::Node>>& cos_sin_cached,
        int64_t unsqueeze_dim=1) {
    
    // Handle unsqueeze or cached values
    Output<ov::Node> cos_unsqueezed, sin_unsqueezed;
    
    if (cos_sin_cached.first.get_node() == nullptr) {
        auto unsqueeze_axes1 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, unsqueeze_dim);
        cos_unsqueezed = std::make_shared<v0::Unsqueeze>(cos, unsqueeze_axes1);
        auto unsqueeze_axes2 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, unsqueeze_dim);
        sin_unsqueezed = std::make_shared<v0::Unsqueeze>(sin, unsqueeze_axes2);
    } else {
        cos_unsqueezed = cos_sin_cached.first;
        sin_unsqueezed = cos_sin_cached.second;
    }

    // Apply rotation
    auto q_rot = std::make_shared<v1::Add>(
        std::make_shared<v1::Multiply>(q, cos_unsqueezed),
        std::make_shared<v1::Multiply>(rotate_half(q, head_size, hidden_dim), sin_unsqueezed)
    );

    auto k_rot = std::make_shared<v1::Add>(
        std::make_shared<v1::Multiply>(k, cos_unsqueezed),
        std::make_shared<v1::Multiply>(rotate_half(k, head_size, hidden_dim), sin_unsqueezed)
    );

    return {q_rot, k_rot, {cos_unsqueezed, sin_unsqueezed}};
}

// Generate Rotary Position Embedding components
std::pair<Output<ov::Node>, Output<ov::Node>> rope_emb(
    const Output<ov::Node>& x,
    const Output<ov::Node>& rope_const,
    const Output<ov::Node>& position_ids,
    const Output<ov::Node>& batch_dim) {
    
    // Process position IDs
    auto position_expanded = std::make_shared<v0::Convert>(
        std::make_shared<v0::Unsqueeze>(position_ids, 
            std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 1)),
        element::f32
    );

    // Broadcast rope constants
    auto target_shape = std::make_shared<v0::Concat>(OutputVector{
        batch_dim,
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1),
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1)
    }, 0);

    auto inv_freq_expanded = std::make_shared<v3::Broadcast>(
        rope_const, target_shape, BroadcastType::BIDIRECTIONAL
    );

    // Compute frequencies
    auto freqs = std::make_shared<v0::MatMul>(
        inv_freq_expanded, position_expanded,
        false, false
    );

    auto freqs_transposed = std::make_shared<v1::Transpose>(
        freqs, 
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 2, 1})
    );

    // Concatenate and compute trigonometric values
    auto emb = std::make_shared<ov::opset13::Concat>(
        ov::NodeVector{freqs_transposed, freqs_transposed}, -1
    );

    return {
        std::make_shared<ov::opset13::Cos>(emb),
        std::make_shared<ov::opset13::Sin>(emb)
    };
}


ov::Output<ov::Node> make_rms_norm_qwen3(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    float rms_norm_eps) {
    auto eps_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1,1,1,1}, rms_norm_eps);
    auto square = std::make_shared<ov::op::v1::Power>(
        input, 
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 2.0f));
    
    auto variance = std::make_shared<ov::op::v1::ReduceMean>(
        square, 
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, -1),
        true);

    auto add_eps = std::make_shared<ov::op::v1::Add>(variance, eps_node);
    auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(add_eps);
    auto reciprocal = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v0::Constant>(
            ov::element::f32, ov::Shape{}, 1.0f),
        sqrt_node);

    std::shared_ptr<ov::Node> mul = std::make_shared<ov::op::v1::Multiply>(
        reciprocal, input, AutoBroadcastType::NUMPY);

    auto weight_tensor = weights.at(key + ".weight");
    // Check if all elements are 1.0
    bool all_ones = true;
    if (weight_tensor.get_element_type() == ov::element::f32) {
        const float* data = weight_tensor.data<float>();
        for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
            if (data[i] != 1.0f) {
                all_ones = false;
                break;
            }
        }
    } else if (weight_tensor.get_element_type() == ov::element::f16) {
        const uint16_t* data = weight_tensor.data<uint16_t>();
        const uint16_t one_in_fp16 = 0x3C00;
        for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
            if (data[i] != one_in_fp16) {
                all_ones = false;
                break;
            }
        }
    } else {
        OPENVINO_THROW("Unsupported weight type ", weight_tensor.get_element_type());
    }

    if (!all_ones) {
        weight_tensor.set_shape(ov::Shape{1, 1, 1, weight_tensor.get_shape()[0]});
        auto weights_const = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        auto weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        mul = std::make_shared<ov::op::v1::Multiply>(mul, weights_f32, AutoBroadcastType::NUMPY);
    }

    return mul;
}

// Helper function to split heads
// There are q_norm k_norm in Qwen3, if key_name + ".self_attn.q_norm" + ".weight" exists, a rms_norm will be built, if not it will go to else branch.
std::shared_ptr<v1::Transpose> split_heads(const Output<Node>& x,
                                            int num_h,
                                            int  head_dim,
                                            float rms_norm_eps,
                                            const std::string& key,
                                            const std::unordered_map<std::string, ov::Tensor>& weights) {
    auto shape = std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_h, head_dim});
    auto reshaped = std::make_shared<v1::Reshape>(x, shape, true);
    if (weights.count(key + ".weight")) { //Qwen3 rms_norm
        auto mul = make_rms_norm_qwen3(key, reshaped, weights, rms_norm_eps);
        auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
        
        return std::make_shared<v1::Transpose>(mul, transpose_order);
    } else { //none-Qwen3 architecture
        auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
        return std::make_shared<v1::Transpose>(reshaped, transpose_order);
    } 
};

std::tuple<Output<Node>, ov::SinkVector, std::pair<Output<Node>, Output<Node>>, Output<Node>>
multi_head_attention(
    const Output<Node>& query,
    const Output<Node>& key,
    const Output<Node>& value,
    const std::string& key_name,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::map<std::string, GGUFMetaData>& configs,
    const Output<Node>& batch_dim,
    int layer_idx,
    const Output<Node>& hidden_dim,
    const Output<Node>& input_shape,
    const Output<Node>& output_shape,
    const Output<Node>& attention_mask,
    const Output<Node>& mask,
    const Output<Node>& position_ids,
    const Output<Node>& rope_const,
    const Output<Node>& beam_idx,
    const std::pair<Output<Node>, Output<Node>>& cos_sin_cached) {
    int num_heads = std::get<int>(configs.at("head_num"));
    int head_dim = std::get<int>(configs.at("head_size"));
    int num_heads_kv = std::get<int>(configs.at("head_num_kv"));
    float rms_norm_eps = std::get<float>(configs.at("rms_norm_eps"));
    
    // 1. Split heads
    // There are q_norm k_norm in Qwen3, if key_name + ".self_attn.q_norm" + ".weight" exists, a rms_norm will be built.
    auto q_split = split_heads(query, num_heads, head_dim, rms_norm_eps, key_name + ".self_attn.q_norm", consts);
    auto k_split = split_heads(key, num_heads_kv, head_dim, rms_norm_eps, key_name  + ".self_attn.k_norm", consts);
    auto v_split = split_heads(value, num_heads_kv, head_dim, rms_norm_eps, key_name + ".self_attn.v_norm", consts);

    // 2. Apply rotary embeddings
    Output<Node> cos, sin;
    if (cos_sin_cached.first.get_node() == nullptr) {
        std::tie(cos, sin) = rope_emb(v_split, rope_const, position_ids, batch_dim);
    }

    auto [q_rot, k_rot, new_cos_sin] = apply_rotary_pos_emb(
        q_split, k_split, cos, sin, head_dim, hidden_dim, cos_sin_cached
    );

    // 3. Handle cache
    auto create_cache = [&](const std::string& name, const Output<Node>& init_value) {
        auto var_info = ov::op::util::VariableInfo{
                ov::PartialShape{-1, num_heads_kv, -1, head_dim},
                ov::element::f32,
                name
            };
        auto var = std::make_shared<ov::op::util::Variable>(var_info);
        auto read_value = std::make_shared<v6::ReadValue>(init_value, var);
        auto gathered = std::make_shared<v8::Gather>(read_value, beam_idx, 
            std::make_shared<v0::Constant>(element::i64, Shape{}, 0), 0);
        return std::make_pair(var, gathered);
    };

    auto zero_const = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.0f);
    auto k_cache_default = std::make_shared<v3::Broadcast>(zero_const, 
        std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0));

    auto v_cache_default = std::make_shared<v3::Broadcast>(zero_const, 
        std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0));

    auto k_cache = create_cache(
        "past_key_values." + std::to_string(layer_idx) + ".keypresent." + std::to_string(layer_idx) + ".key",
        k_cache_default
    );
    auto v_cache = create_cache(
        "past_key_values." + std::to_string(layer_idx) + ".valuepresent." + std::to_string(layer_idx) + ".key",
        v_cache_default
    );

    auto k_combined = std::make_shared<ov::opset13::Concat>(OutputVector{k_cache.second, k_rot}, 2);
    auto v_combined = std::make_shared<ov::opset13::Concat>(OutputVector{v_cache.second, v_split}, 2);

    auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined, k_cache.first); //->get_variable_id()
    auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined, v_cache.first);

    // 4. Handle group query attention
    Output<Node> k_reshaped = k_combined;
    Output<Node> v_reshaped = v_combined;
    if (num_heads != num_heads_kv) {
        int kv_per_head = num_heads / num_heads_kv;
        auto unsqueeze_axes1 = std::make_shared<v0::Constant>(element::i64, Shape{}, 2);
        auto k_unsq = std::make_shared<v0::Unsqueeze>(k_combined, unsqueeze_axes1);
        auto unsqueeze_axes2 = std::make_shared<v0::Constant>(element::i64, Shape{}, 2);
        auto v_unsq = std::make_shared<v0::Unsqueeze>(v_combined, unsqueeze_axes2);

        auto broadcast_shape1 = std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, kv_per_head),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0);

        k_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(k_unsq, broadcast_shape1, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true
        );


        auto broadcast_shape2 = std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, kv_per_head),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0);
        v_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(v_unsq, broadcast_shape2, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true
        );
    }

    // 5. Create causal mask if needed
    Output<Node> final_mask = mask;
    if (mask.get_node() == nullptr) {
        final_mask = causal_mask(attention_mask, k_cache.second, hidden_dim, input_shape);
    }

    // 6. Scaled dot product attention
    auto attention = std::make_shared<ScaledDotProductAttention>(
        q_rot, k_reshaped, v_reshaped, final_mask, false);

    // 7. Reshape output
    auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
    auto context_transposed = std::make_shared<v1::Transpose>(attention, transpose_order);
    auto output = std::make_shared<v1::Reshape>(context_transposed, output_shape, false);

    return {
        output,
        {k_assign, v_assign},
        new_cos_sin,
        final_mask
    };
}

// TODO: can be issues with allocated memory
// TODO: rewrite without doubling a memory
ov::Tensor reorder_interleaved_format(const ov::Tensor& weights, int head_size) {
    ov::Shape input_shape = weights.get_shape();
    if (input_shape.empty() || input_shape[0] % head_size != 0) {
        throw std::invalid_argument("Invalid input dimensions");
    }

    size_t num_heads = input_shape[0] / head_size;
    size_t total_rows = input_shape[0];
    std::vector<size_t> permutation(total_rows);

    // Precompute permutation indices
    for (size_t i = 0; i < total_rows; ++i) {
        size_t head = i / head_size;
        size_t row_in_head = i % head_size;
        size_t new_row_in_head = (row_in_head < head_size/2)
            ? row_in_head * 2
            : (row_in_head - head_size/2) * 2 + 1;
        permutation[i] = head * head_size + new_row_in_head;
    }

    // Create output tensor
    ov::Tensor reordered(weights.get_element_type(), input_shape);

    // Calculate row size in bytes
    size_t row_size = weights.get_byte_size() / total_rows;
    const char* src_data = (const char*)weights.data();
    char* dst_data = (char*)reordered.data();

    // Perform permutation copy
    for (size_t i = 0; i < total_rows; ++i) {
        std::memcpy(dst_data + i * row_size,
                   src_data + permutation[i] * row_size,
                   row_size);
    }

    return reordered;
}

ov::Output<ov::Node> make_fp16_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    bool reorder,
    int head_size) {

    auto it = consts.find(key + ".weight");
    OPENVINO_ASSERT(it != consts.end(), "Weight not found: ", key);
    ov::Tensor weight_f16 = it->second;    

    // Apply reordering
    if (reorder) {
        weight_f16 = reorder_interleaved_format(weight_f16, head_size);
    }

    // Create FP16 constant and convert to FP32
    auto weights_node = std::make_shared<v0::Constant>(weight_f16);
    weights_node->set_friendly_name(key + ".weight");
    return std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f32);
}

// Retrieve tensors
ov::Tensor get_tensor(const std::unordered_map<std::string, ov::Tensor>& consts,
                    const std::string& key) {
    auto it = consts.find(key);
    OPENVINO_ASSERT(it != consts.end(), "Missing tensor: ", key);
    return it->second;
};

ov::Output<ov::Node> make_int8_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    bool reorder,
    int head_size,
    size_t group_size = GGML_QUANTIZATION_GROUP_SIZE) {

    ov::Tensor weight = get_tensor(consts, key + ".weight");
    ov::Tensor scales = get_tensor(consts, key + ".scales");
    ov::Tensor biases = get_tensor(consts, key + ".biases");

    // Reshape weight to (num_heads, -1, group_size)
    ov::Shape orig_shape = weight.get_shape();
    orig_shape[1] *= sizeof(uint32_t) / sizeof(uint8_t);
    size_t num_groups = orig_shape[1] / group_size;

    // Expand dimensions for scales and biases
    auto scale_shape = scales.get_shape();
    scale_shape.push_back(1);
    scales.set_shape(scale_shape);
    biases.set_shape(scale_shape);

    // Apply reordering
    if (reorder) {
        weight = reorder_interleaved_format(weight, head_size);
        scales = reorder_interleaved_format(scales, head_size);
        biases = reorder_interleaved_format(biases, head_size);
    }

    // Create graph nodes
    auto weights_node = std::make_shared<v0::Constant>(ov::element::u8, ov::Shape{orig_shape[0], num_groups, group_size}, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);
    ov::Tensor biases_u8(ov::element::u8, scale_shape);

    // Calculate zero point
    const ov::float16* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const ov::float16* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    uint8_t* bias_u8_data = biases_u8.data<uint8_t>();
    for (size_t i = 0; i < biases_u8.get_size(); ++i) {
        bias_u8_data[i] = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i]) / static_cast<float>(scale_data[i]));
    }

    auto zero_point = std::make_shared<ov::op::v0::Constant>(biases_u8);

    // Quantization operations
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
    auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);

    auto w_zp = std::make_shared<ov::op::v1::Subtract>(
        weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY
    );
    auto w_zp_s = std::make_shared<ov::op::v1::Multiply>(
        w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY
    );

    // Reshape back to original dimensions
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape
    );
    auto w_zp_s_r = std::make_shared<ov::op::v1::Reshape>(
        w_zp_s, final_shape, false
    );

    return std::make_shared<ov::op::v0::Convert>(w_zp_s_r, ov::element::f32);
}

ov::Output<ov::Node> make_int4_weights(
    const std::string& key,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    bool reorder,
    int head_size,
    size_t group_size = 32) { // Assuming GGML_QUANTIZATION_GROUP_SIZE = 32

    ov::Tensor weight = get_tensor(consts, key + ".weight");

    // Convert weight to uint8 view and adjust shape
    ov::Shape orig_weight_shape = weight.get_shape();
    orig_weight_shape[1] *= sizeof(uint32_t) / sizeof(uint8_t) * 2; // Double number of columns for 4-bit representation

    // Retrieve scales and biases
    ov::Tensor scales = get_tensor(consts, key + ".scales");
    ov::Tensor biases = get_tensor(consts, key + ".biases");

    // Expand dimensions for scales and biases
    ov::Shape scale_bias_shape = scales.get_shape();
    scale_bias_shape.push_back(1); // Add new axis at the end
    scales.set_shape(scale_bias_shape);
    biases.set_shape(scale_bias_shape);

    // Apply reordering if needed
    if (reorder) {
        weight = reorder_interleaved_format(weight, head_size);
        scales = reorder_interleaved_format(scales, head_size);
        biases = reorder_interleaved_format(biases, head_size);
    }

    // Create INT4 weight tensor
    ov::Shape packed_shape = {
        orig_weight_shape[0],
        orig_weight_shape[1] / group_size,
        group_size
    };

    auto weights_node = std::make_shared<v0::Constant>(ov::element::u4, packed_shape, static_cast<uint8_t*>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holde"] = weight;
    auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

    // Pack zero points: two subsequent values into one
    const ov::float16* bias_data = biases.data<ov::element_type_traits<ov::element::f16>::value_type>();
    const ov::float16* scale_data = scales.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::Tensor zero_point_tensor(ov::element::u4, scale_bias_shape);
    uint8_t* zero_point_data = static_cast<uint8_t*>(zero_point_tensor.data());
    for (size_t i = 0; i < zero_point_tensor.get_byte_size(); ++i) {
        uint8_t bias1 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2]) / static_cast<float>(scale_data[i * 2]));
        uint8_t bias2 = (uint8_t)std::round(-1.f * static_cast<float>(bias_data[i * 2 + 1]) / static_cast<float>(scale_data[i * 2 + 1]));
        zero_point_data[i] = (bias2 << 4) | (bias1 & 0x0F);
    }

    auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zero_point_tensor);
    auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_node, ov::element::f16);

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    // Perform dequantization
    auto w_zp = std::make_shared<ov::op::v1::Subtract>(
        weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);

    auto w_zp_s = std::make_shared<ov::op::v1::Multiply>(
        w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);

    // Reshape back to original shape
    auto final_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{orig_weight_shape.size()}, orig_weight_shape);

    auto w_zp_s_r = std::make_shared<ov::op::v1::Reshape>(
        w_zp_s, final_shape, false);

    return std::make_shared<ov::op::v0::Convert>(w_zp_s_r, ov::element::f32);
}

ov::Output<ov::Node> make_weights_subgraph(const std::string& key,
                                           const std::unordered_map<std::string, ov::Tensor>& consts,
                                           gguf_tensor_type qtype,
                                           bool reorder,
                                           int head_size) {
    switch (qtype) {
    case gguf_tensor_type::GGUF_TYPE_F16:
        return make_fp16_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q8_0:
        return make_int8_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q4_0:
        return make_int4_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q4_1:
        return make_int4_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q4_K:
        return make_int4_weights(key, consts, reorder, head_size);
    case gguf_tensor_type::GGUF_TYPE_Q6_K:
        return make_int8_weights(key, consts, reorder, head_size, 16);
    default:
        OPENVINO_THROW("Unsupported quantization type");
    }
}

ov::Output<ov::Node> make_fc(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    gguf_tensor_type qtype,
    bool reorder = false,
    int head_size = -1) {
    auto w_f32 = make_weights_subgraph(key, consts, qtype, reorder, head_size);
    std::shared_ptr<ov::Node> output = std::make_shared<ov::op::v0::MatMul>(input, w_f32, false, true);

    // Add post-MatMul Add operation if exists
    if (consts.count(key + ".bias")) {
        auto add_tensor = get_tensor(consts, key + ".bias");
        auto add_const = std::make_shared<v0::Constant>(add_tensor);
        auto add_convert = std::make_shared<ov::op::v0::Convert>(add_const, ov::element::f32);
        output = std::make_shared<ov::op::v1::Add>(
                                    output, add_convert, ov::op::AutoBroadcastType::NUMPY);
    }
    return output;
}

ov::Output<ov::Node> make_lm_head(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const ov::Output<ov::Node>& embeddings_node,
    gguf_tensor_type qtype) {

    ov::Output<ov::Node> w_f32;
    if (consts.count(key + ".weight")) {
        gguf_tensor_type lm_qtype = qtype;
        if (!consts.count(key + ".scales")) {
            lm_qtype = gguf_tensor_type::GGUF_TYPE_F16;
        }
        w_f32 = make_weights_subgraph(key, consts, lm_qtype, false, -1);
    } else {
        w_f32 = embeddings_node;
    }
    return std::make_shared<ov::op::v0::MatMul>(
        input, w_f32, false, true);
}

ov::Output<ov::Node> make_rms_norm(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    float epsilon) {

    auto eps_node = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{1,1,1}, epsilon);
    auto square = std::make_shared<ov::op::v1::Power>(
        input, 
        std::make_shared<ov::op::v0::Constant>(
            ov::element::f32, ov::Shape{1,1,1}, 2.0f));
    
    auto variance = std::make_shared<ov::op::v1::ReduceMean>(
        square, 
        std::make_shared<ov::op::v0::Constant>(
            ov::element::i32, ov::Shape{1}, -1),
        true);

    auto add_eps = std::make_shared<ov::op::v1::Add>(variance, eps_node);
    auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(add_eps);
    auto reciprocal = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v0::Constant>(
            ov::element::f32, ov::Shape{1,1,1}, 1.0f),
        sqrt_node);

    std::shared_ptr<ov::Node> mul = std::make_shared<ov::op::v1::Multiply>(
        reciprocal, input, AutoBroadcastType::NUMPY);

    if (consts.count(key + ".weight")) {
        auto weight_tensor = consts.at(key + ".weight");
        // Check if all elements are 1.0
        bool all_ones = true;
        if (weight_tensor.get_element_type() == ov::element::f32) {
            const float* data = weight_tensor.data<float>();
            for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
                if (data[i] != 1.0f) {
                    all_ones = false;
                    break;
                }
            }
        } else if (weight_tensor.get_element_type() == ov::element::f16) {
            const uint16_t* data = weight_tensor.data<uint16_t>();
            const uint16_t one_in_fp16 = 0x3C00;
            for (size_t i = 0; i < weight_tensor.get_size(); ++i) {
                if (data[i] != one_in_fp16) {
                    all_ones = false;
                    break;
                }
            }
        } else {
            OPENVINO_THROW("Unsupported weight type ", weight_tensor.get_element_type());
        }

        if (!all_ones) {
            weight_tensor.set_shape(ov::Shape{1, 1, weight_tensor.get_shape()[0]});
            auto weights_const = std::make_shared<ov::op::v0::Constant>(
                weight_tensor);
            auto weights_f32 = std::make_shared<ov::op::v0::Convert>(
                weights_const, ov::element::f32);
            mul = std::make_shared<ov::op::v1::Multiply>(
                mul, weights_f32, AutoBroadcastType::NUMPY);
        }
    }

    return mul;
}

std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>> make_embedding(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    gguf_tensor_type qtype) {
        
    auto embedding_type = qtype;
    // Detmbedding_type = qtype;
    if (consts.count(key + ".scales") == 0) {
        embedding_type = gguf_tensor_type::GGUF_TYPE_F16;
    }

    // Create embedding weights
    auto embed_f32 = make_weights_subgraph(key, consts, embedding_type, false, -1);

    // Convert input to int32 indices
    auto input_int32 = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);

    // Gather embeddings
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
    auto embeddings = std::make_shared<ov::op::v8::Gather>(embed_f32, input_int32, axis);

    return {embeddings, embed_f32};
}

std::tuple<ov::Output<ov::Node>, 
           ov::SinkVector,
           ov::Output<ov::Node>,
           std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>,
           std::shared_ptr<ov::Node>> 
    layer(const std::map<std::string, GGUFMetaData>& configs,
        std::unordered_map<std::string, ov::Tensor>& consts,
        std::unordered_map<std::string, gguf_tensor_type>& qtypes,
        int layer_idx,
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& attn_mask,
        const ov::Output<ov::Node>& causal_mask,
        const ov::Output<ov::Node>& position_ids,
        const ov::Output<ov::Node>& rope_const,
        const ov::Output<ov::Node>& beam_idx,
        const ov::Output<ov::Node>& batch_dim,
        const ov::Output<ov::Node>& hidden_dim,
        const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& cos_sin_cached,
        const std::shared_ptr<ov::Node>& output_shape) {

    std::string name_suffix = ".layer" + std::to_string(layer_idx);
    std::string name_prefix = "model.layers.self_attn";
    std::string layer_prefix = format("model.layers[%d]", layer_idx);

    // LayerNorm
    auto input_layernorm = make_rms_norm(layer_prefix + ".input_layernorm",
                                         hidden_states,
                                         consts,
                                         std::get<float>(configs.at("rms_norm_eps")));

    // Attention projections
    // check if it's llama structure, if so, reorder= true
    bool reorder = false;
    if (std::get<std::string>(configs.at("architecture")).find("llama") != std::string::npos) {
        reorder = true;
    }
    auto q = make_fc(
        layer_prefix + ".self_attn.q_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.q_proj.qtype"),
        reorder,
        std::get<int>(configs.at("head_size")));
    
    auto k = make_fc(
        layer_prefix + ".self_attn.k_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.k_proj.qtype"),
        reorder,
        std::get<int>(configs.at("head_size")));

    auto v = make_fc(
        layer_prefix + ".self_attn.v_proj",
        input_layernorm,
        consts,
        qtypes.at(layer_prefix + ".self_attn.v_proj.qtype"));

    // Handle output shape
    std::shared_ptr<ov::Node> final_output_shape = output_shape;
    if (!output_shape) {
        auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_layernorm);
        auto indices = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1});
        auto gathered = std::make_shared<ov::op::v8::Gather>(
            input_shape, indices, 
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0));
        auto minus_one = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{1}, -1);
        final_output_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{gathered, minus_one}, 0);
    }

    // Multi-head attention
    auto [attn_output, sinks, new_cos_sin, new_causal_mask] = multi_head_attention(
        q, k, v,
        layer_prefix,
        consts,
        configs,
        batch_dim,
        layer_idx,
        hidden_dim,
        std::make_shared<ov::op::v3::ShapeOf>(input_layernorm),
        final_output_shape,
        attn_mask,
        causal_mask,
        position_ids,
        rope_const,
        beam_idx,
        cos_sin_cached);

    // Output projection
    auto o_proj = make_fc(
        layer_prefix + ".self_attn.o_proj",
        attn_output,
        consts,
        qtypes.at(layer_prefix + ".self_attn.o_proj.qtype"));

    // Residual connection
    auto attn_add = std::make_shared<ov::op::v1::Add>(
        hidden_states, o_proj, ov::op::AutoBroadcastType::NUMPY);
    attn_add->set_friendly_name(name_prefix + ".add0" + name_suffix);

    // Post-attention Layernorm
    auto post_attn_norm = make_rms_norm(
        layer_prefix + ".post_attention_layernorm",
        attn_add,
        consts,
        std::get<float>(configs.at("rms_norm_eps")));

    // MLP block
    auto gate_proj = make_fc(
        layer_prefix + ".mlp.gate_proj",
        post_attn_norm,
        consts,
        qtypes.at(layer_prefix + ".mlp.gate_proj.qtype"));
    auto silu = std::make_shared<ov::op::v4::Swish>(gate_proj);
    auto up_proj = make_fc(
        layer_prefix + ".mlp.up_proj",
        post_attn_norm,
        consts,
        qtypes.at(layer_prefix + ".mlp.up_proj.qtype"));
    auto mul = std::make_shared<ov::op::v1::Multiply>(
        silu, up_proj, ov::op::AutoBroadcastType::NUMPY);
    mul->set_friendly_name(name_prefix + ".mlp.mul" + name_suffix);
    auto down_proj = make_fc(
        layer_prefix + ".mlp.down_proj",
        mul,
        consts,
        qtypes.at(layer_prefix + ".mlp.down_proj.qtype"));

    // Final residual connection
    auto output = std::make_shared<ov::op::v1::Add>(
        attn_add, down_proj, ov::op::AutoBroadcastType::NUMPY);
    output->set_friendly_name(name_prefix + ".add1" + name_suffix);

    return {output, sinks, new_causal_mask, new_cos_sin, final_output_shape};
}

ov::Output<ov::Node> init_rope(
    int64_t head_dim,
    int64_t max_position_embeddings,
    float base,
    float scaling_factor) {

    // Calculate inverse frequencies
    size_t num_elements = head_dim / 2;
    std::vector<float> inv_freq_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        float idx = static_cast<float>(2 * i);  // Matches Python's step=2
        float exponent = idx / static_cast<float>(head_dim);
        inv_freq_data[i] = 1.0f / std::pow(base, exponent);
        
        // Apply scaling factor if needed (from original Python signature)
        if (scaling_factor != 1.0f) {
            inv_freq_data[i] *= scaling_factor;
        }
    }

    // Create OpenVINO constant with shape [1, num_elements, 1]
    ov::Shape const_shape = {1, static_cast<unsigned long>(num_elements), 1};
    auto rope_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, const_shape, inv_freq_data);

    return rope_const;
}