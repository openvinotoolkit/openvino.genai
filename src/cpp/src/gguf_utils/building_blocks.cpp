#include <vector>
#include <openvino/openvino.hpp>

#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"


using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

Output<ov::Node> causal_mask(
    const Output<ov::Node>& attention_mask,
    const Output<ov::Node>& keys,
    int64_t hidden_dim,
    const Output<ov::Node>& input_shape) {

    auto hidden_dim_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, hidden_dim);
    // Extract shape of attention mask
    auto t130 = std::make_shared<v3::ShapeOf>(attention_mask, element::i64);
    auto t131 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1);
    auto t132 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0);
    auto t133 = std::make_shared<v8::Gather>(t130, t131, t132);

    // Reshape and construct new shapes
    auto t134 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t135 = std::make_shared<v1::Reshape>(t133, t134, false);
    auto t40 = input_shape;
    auto index_1 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1);
    auto axis_0 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0);
    auto t127 = std::make_shared<v8::Gather>(t40, index_1, axis_0);
    auto t129 = std::make_shared<v1::Reshape>(t127, t134, false);
    auto t136 = std::make_shared<v0::Concat>(OutputVector{t129, t135}, 0);
    auto min_shape_val = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 1});
    auto t137 = std::make_shared<v1::Maximum>(min_shape_val, t136, AutoBroadcastType::NUMPY);
    auto const_neg65504 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, -65504);
    auto t138 = std::make_shared<v3::Broadcast>(const_neg65504, t137, BroadcastType::NUMPY);

    // Create upper triangular mask for causal masking
    auto t139 = std::make_shared<v3::ShapeOf>(t138, element::i32);
    auto t140 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1);
    auto t142 = std::make_shared<v8::Gather>(t139, t140, t132);
    auto t143 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, 1);
    auto zero_const = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, 0);
    auto t144 = std::make_shared<v4::Range>(zero_const, t142, t143, element::i32);
    auto axes_zero = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0);
    auto t145 = std::make_shared<v0::Unsqueeze>(t144, axes_zero);
    auto t146 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, 1);
    auto t149 = std::make_shared<v8::Gather>(t139, t132, t132);
    auto t150 = std::make_shared<v1::Add>(t149, t146, AutoBroadcastType::NUMPY);
    auto t151 = std::make_shared<v4::Range>(t146, t150, t143, element::i32);
    auto axes_one = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t152 = std::make_shared<v0::Unsqueeze>(t151, axes_one);
    auto t153 = std::make_shared<v1::GreaterEqual>(t145, t152, AutoBroadcastType::NUMPY);
    auto t154 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, 0.0f);
    auto t155 = std::make_shared<v1::Select>(t153, t138, t154, AutoBroadcastType::NUMPY);

    // Next branch
    auto t156 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, 0);
    auto t157 = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, 1);
    auto t158 = std::make_shared<v4::Range>(t156, t133, t157, element::f32);
    auto t159 = std::make_shared<v0::Convert>(t158, element::i64);
    auto t160 = std::make_shared<v0::Convert>(t159, element::f32);
    auto t161 = std::make_shared<v3::ShapeOf>(keys, element::i64);
    auto t162 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 2);
    auto t164 = std::make_shared<v8::Gather>(t161, t162, axis_0);
    auto t165 = std::make_shared<v1::Add>(t164, t127, AutoBroadcastType::NUMPY);
    auto t167 = std::make_shared<v4::Range>(t164, t165, t146, element::f32);
    auto t168 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
    auto t169 = std::make_shared<v1::Reshape>(t167, t168, false);
    auto t170 = std::make_shared<v1::Greater>(t160, t169, AutoBroadcastType::NUMPY);
    auto t171 = std::make_shared<v0::Convert>(t170, element::f32);
    auto t172 = std::make_shared<v1::Multiply>(t155, t171, AutoBroadcastType::NUMPY);

    auto t173 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0);
    auto unsqueeze_axes_0 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t174 = std::make_shared<v0::Unsqueeze>(t172, unsqueeze_axes_0);
    auto t48 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1);
    auto t175 = std::make_shared<v0::Unsqueeze>(t174, t48);
    auto t41 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t43 = std::make_shared<v8::Gather>(input_shape, t41, axis_0);
    auto t176 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t177 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t178 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t179 = std::make_shared<v0::Concat>(OutputVector{t43, t176, t177, t178}, 0);
    auto t180 = std::make_shared<v3::Broadcast>(t175, t179, BroadcastType::BIDIRECTIONAL);
    auto t181 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1});
    auto t182 = std::make_shared<v1::Reshape>(t180, t181, false);
    auto t183 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0);
    auto t184 = std::make_shared<v3::ShapeOf>(t180, element::i64);
    auto t185 = std::make_shared<v1::ReduceProd>(t184, t183, false);
    auto t186 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1);
    auto t187 = std::make_shared<v4::Range>(t183, t185, t186, element::i64);
    auto t188 = std::make_shared<v1::Reshape>(t187, t184, false);
    auto t189 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t190 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t191 = std::make_shared<ov::opset13::Slice>(t188, t189, t135, t190, hidden_dim_const);
    auto t192 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{-1, 1});
    auto t193 = std::make_shared<v1::Reshape>(t191, t192, false);
    auto t194 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto t195 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto t196 = std::make_shared<ov::opset13::Slice>(t180, t194, t135, t195, hidden_dim_const);

    auto t197 = std::make_shared<v0::Unsqueeze>(attention_mask, t48);
    auto t198 = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 2);
    auto t199 = std::make_shared<v0::Unsqueeze>(t197, t198);
    auto t200 = std::make_shared<v0::Convert>(t199, element::f32);
    auto t201 = std::make_shared<v1::Add>(t196, t200, AutoBroadcastType::NUMPY);
    auto t202 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1,1,1,1}, std::vector<float>{0.0});
    auto t203 = std::make_shared<v1::Equal>(t201, t202, AutoBroadcastType::NUMPY);
    auto t204 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, -65504.0);
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
    auto t217 = std::make_shared<ov::opset13::Slice>(t211, t212, t215, t216, hidden_dim_const);

    return t217->output(0);
}

// Rotate half the hidden dimensions of the input tensor
Output<ov::Node> rotate_half(const Output<ov::Node>& x, int64_t head_size, int64_t axis) {
    auto axis_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, axis);
    auto half_head = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, head_size / 2);
    auto max_int = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 9223372036854775807);
    auto step = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1);

    // Slice second half
    auto second_half = std::make_shared<ov::opset13::Slice>(x, half_head, max_int, step, axis_const);
    
    // Multiply by -1
    auto neg_one = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, -1.0f);
    auto rotated_half = std::make_shared<v1::Multiply>(second_half, neg_one);
    
    // Slice first half
    auto first_half = std::make_shared<ov::opset13::Slice>(x, 
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 0),
        half_head,
        step,
        axis_const);
    
    // Concatenate rotated half and first half
    return std::make_shared<v0::Concat>(ov::OutputVector{rotated_half, first_half}, axis);
}

// Apply Rotary Position Embedding to query and key tensors
std::tuple<Output<ov::Node>, Output<ov::Node>, std::pair<Output<ov::Node>, Output<ov::Node>>> 
apply_rotary_pos_emb(
    const Output<ov::Node>& q, 
    const Output<ov::Node>& k,
    const Output<ov::Node>& cos,
    const Output<ov::Node>& sin,
    int64_t head_size,
    int64_t hidden_dim,
    const std::pair<Output<ov::Node>, Output<ov::Node>>& cos_sin_cached,
    int64_t unsqueeze_dim) {
    
    // Handle unsqueeze or cached values
    Output<ov::Node> cos_unsqueezed, sin_unsqueezed;
    if (cos_sin_cached.first.get_node() == nullptr) {
        auto unsqueeze_axes = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, unsqueeze_dim);
        cos_unsqueezed = std::make_shared<v0::Unsqueeze>(cos, unsqueeze_axes);
        sin_unsqueezed = std::make_shared<v0::Unsqueeze>(sin, unsqueeze_axes);
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
            std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 1)),
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

//std::tuple<Output<Node>, std::vector<std::shared_ptr<VariableState>>, std::pair<Output<Node>, Output<Node>>, Output<Node>>
std::tuple<Output<Node>, std::vector<Output<Node>>, std::pair<Output<Node>, Output<Node>>, Output<Node>>
multi_head_attention(
    const Output<Node>& query,
    const Output<Node>& key,
    const Output<Node>& value,
    const std::map<std::string, int>& configs,
    const Output<Node>& batch_dim,
    int layer_idx,
    int hidden_dim,
    const Output<Node>& input_shape,
    const Output<Node>& output_shape,
    const Output<Node>& attention_mask,
    const Output<Node>& mask,
    const Output<Node>& position_ids,
    const Output<Node>& rope_const,
    const Output<Node>& beam_idx,
    const std::pair<Output<Node>, Output<Node>>& cos_sin_cached) {

    int num_heads = configs.at("head_num");
    int head_dim = configs.at("head_size");
    int num_heads_kv = configs.at("head_num_kv");

    // Helper function to split heads
    auto split_heads = [&](const Output<Node>& x, int num_h) {
        auto shape = std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, 0, num_h, head_dim});
        auto reshaped = std::make_shared<v1::Reshape>(x, shape, true);
        auto transpose_order = std::make_shared<v0::Constant>(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3});
        return std::make_shared<v1::Transpose>(reshaped, transpose_order);
    };

    // 1. Split heads
    auto q_split = split_heads(query, num_heads);
    auto k_split = split_heads(key, num_heads_kv);
    auto v_split = split_heads(value, num_heads_kv);

    // 2. Apply rotary embeddings
    Output<Node> cos, sin;
    if (cos_sin_cached.first.get_node() == nullptr) {
        std::tie(cos, sin) = rope_emb(v_split, rope_const, position_ids, batch_dim);
    } else {
        cos = cos_sin_cached.first;
        sin = cos_sin_cached.second;
    }

    auto [q_rot, k_rot, new_cos_sin] = apply_rotary_pos_emb(
        q_split, k_split, cos, sin, head_dim, hidden_dim, cos_sin_cached, 1
    );

    // 3. Handle cache
    auto create_cache = [&](const std::string& name, const Output<Node>& init_value) {
        // auto default_shape = std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{
        //     -1, num_heads_kv, 0, head_dim});
        //auto var = std::make_shared<v6::ReadValue>({init_value}, default_shape, element::f32, name);
        //auto var = makeOP<v6::ReadValue>({init_value}, {{"variable_id", name}, {"variable_type", "f32"}, {"variable_shape", default_shape}});
        // auto var = std::make_shared<v6::ReadValue>(init_value name);
        // var->visit_attributes();
        // var->constructor_validate_and_infer_types();
        auto var_info = ov::op::util::VariableInfo{
                ov::PartialShape{-1, num_heads_kv, 0, head_dim},
                ov::element::f32,
                name
            };
        auto var = std::make_shared<ov::op::util::Variable>(var_info);
        auto read_value = std::make_shared<v6::ReadValue>(var);
        auto gathered = std::make_shared<v8::Gather>(read_value, beam_idx, 
            std::make_shared<v0::Constant>(element::i64, Shape{}, 0));
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
        auto unsqueeze_axes = std::make_shared<v0::Constant>(element::i64, Shape{1}, 2);
        auto k_unsq = std::make_shared<v0::Unsqueeze>(k_combined, unsqueeze_axes);
        auto v_unsq = std::make_shared<v0::Unsqueeze>(v_combined, unsqueeze_axes);

        auto broadcast_shape = std::make_shared<ov::opset13::Concat>(OutputVector{
            batch_dim,
            std::make_shared<v0::Constant>(element::i64, Shape{1}, num_heads_kv),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, kv_per_head),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, 0),
            std::make_shared<v0::Constant>(element::i64, Shape{1}, head_dim)
        }, 0);

        k_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(k_unsq, broadcast_shape, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true
        );

        v_reshaped = std::make_shared<v1::Reshape>(
            std::make_shared<v3::Broadcast>(v_unsq, broadcast_shape, BroadcastType::BIDIRECTIONAL),
            std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{0, num_heads, -1, head_dim}),
            true
        );
    }

    // 5. Create causal mask if needed
    Output<Node> final_mask = mask;
    if (mask.get_node() == nullptr) {
        final_mask = causal_mask(attention_mask, k_combined, hidden_dim, input_shape);
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