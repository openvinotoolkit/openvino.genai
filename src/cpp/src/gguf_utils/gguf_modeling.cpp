// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <stdexcept>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"

#include "gguf_utils/building_blocks.hpp"
#include "gguf_utils/gguf_modeling.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

std::vector<int64_t> get_mrope_section_from_config(const std::map<std::string, GGUFMetaData>& configs) {
    std::vector<int64_t> mrope_section;
    if (!configs.count("mrope_section")) {
        return mrope_section;
    }

    const auto& tensor = std::get<ov::Tensor>(configs.at("mrope_section"));
    if (tensor.get_element_type() == ov::element::i32) {
        const auto* data = tensor.data<int32_t>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            if (data[i] > 0) {
                mrope_section.push_back(static_cast<int64_t>(data[i]));
            }
        }
    } else if (tensor.get_element_type() == ov::element::i64) {
        const auto* data = tensor.data<int64_t>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            if (data[i] > 0) {
                mrope_section.push_back(data[i]);
            }
        }
    } else {
        OPENVINO_THROW("Unsupported mrope_section type: ", tensor.get_element_type());
    }

    return mrope_section;
}

std::shared_ptr<ov::Model> create_language_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    // Create input parameters
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(input_ids, "input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(attention_mask, "attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(position_ids, "position_ids");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::PartialShape{-1});
    set_name(beam_idx, "beam_idx");

    // Create embedding layer
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

    // Get input shape components
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_ids);
    auto batch_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 0);
    auto batch_size = std::make_shared<ov::op::v8::Gather>(
        input_shape, batch_axis, batch_axis);

    auto hidden_dim = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 3);

    // Process layers
    ov::SinkVector sinks;
    ov::Output<ov::Node> causal_mask;
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> cos_sin_cached;
    std::shared_ptr<ov::Node> output_shape = nullptr;

    for (int i = 0; i < std::get<int>(configs.at("layer_num")); ++i) {
        auto [new_hidden, layer_sinks, new_mask, new_cos_sin, new_shape] = layer(
            configs,
            consts,
            qtypes,
            i,
            hidden_states,
            attention_mask,
            causal_mask,
            position_ids,
            rope_const,
            beam_idx,
            batch_size,
            hidden_dim,
            cos_sin_cached,
            output_shape);

        hidden_states = new_hidden;
        causal_mask = new_mask;
        cos_sin_cached = new_cos_sin;
        output_shape = new_shape;

        sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
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

    // Create results
    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    set_name(logits, "logits");

    // Create model
    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    auto model = std::make_shared<ov::Model>(ov::OutputVector({logits->output(0)}), sinks, inputs);

    if (std::get<int>(configs.at("file_type")) == 1 || std::get<int>(configs.at("file_type")) == 0) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return model;
}

std::shared_ptr<ov::Model> create_vlm_language_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    
    const int hidden_size = std::get<int>(configs.at("hidden_size"));
    const int n_deepstack_layers = configs.count("n_deepstack_layers") ?
        std::get<int>(configs.at("n_deepstack_layers")) : 0;
    std::vector<int64_t> mrope_section = get_mrope_section_from_config(configs);
    OPENVINO_ASSERT(!mrope_section.empty(), "[create_vlm_language_model] Missing rope.dimension_sections in GGUF metadata.");
    
    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(attention_mask, "attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{static_cast<int64_t>(mrope_section.size()), -1, -1});
    set_name(position_ids, "position_ids");

    auto inputs_embeds = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, -1, hidden_size});
    set_name(inputs_embeds, "inputs_embeds");

    auto visual_pos_masks = std::make_shared<ov::op::v0::Parameter>(
        ov::element::boolean, ov::PartialShape{-1, -1});
    set_name(visual_pos_masks, "visual_pos_masks");

    auto deepstack_visual_embeds = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, -1, hidden_size});
    set_name(deepstack_visual_embeds, "deepstack_visual_embeds");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::PartialShape{-1});
    set_name(beam_idx, "beam_idx");

    auto hidden_states = inputs_embeds->output(0);

    auto rope_const = init_rope(
        std::get<int>(configs.at("head_size")),
        std::get<int>(configs.at("max_position_embeddings")),
        std::get<float>(configs.at("rope_freq_base")));

    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(inputs_embeds);
    auto batch_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 0);
    auto batch_size = std::make_shared<ov::op::v8::Gather>(
        input_shape, batch_axis, batch_axis);

    auto hidden_dim = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 3);

    ov::SinkVector sinks;
    ov::Output<ov::Node> causal_mask;
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> cos_sin_cached;
    std::shared_ptr<ov::Node> output_shape = nullptr;

    const int layer_num = std::get<int>(configs.at("layer_num"));

    for (int i = 0; i < layer_num; ++i) {
        auto [new_hidden, layer_sinks, new_mask, new_cos_sin, new_shape] = layer(
            configs,
            consts,
            qtypes,
            i,
            hidden_states,
            attention_mask,
            causal_mask,
            position_ids,
            rope_const,
            beam_idx,
            batch_size,
            hidden_dim,
            cos_sin_cached,
            output_shape,
            true,
            mrope_section);

        hidden_states = new_hidden;
        causal_mask = new_mask;
        cos_sin_cached = new_cos_sin;
        output_shape = new_shape;

        if (i < n_deepstack_layers) {
            hidden_states = inject_visual_embeds(
                hidden_states,
                deepstack_visual_embeds,
                visual_pos_masks,
                i);
        }

        sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
    }

    auto final_norm = make_rms_norm(
        "model.norm",
        hidden_states,
        consts,
        std::get<float>(configs.at("rms_norm_eps")));

    ov::Output<ov::Node> lm_head_embeddings;
    if (!consts.count("lm_head.weight")) {
        auto dummy_input_ids = std::make_shared<ov::op::v0::Parameter>(
            ov::element::i64, ov::PartialShape{1, 1});
        auto [unused_inputs_embeds, embeddings] = make_embedding(
            "model.embed_tokens",
            dummy_input_ids,
            consts,
            qtypes.at("model.embed_tokens.qtype"));
        lm_head_embeddings = embeddings;
    }

    auto embed_out = make_lm_head(
        "lm_head",
        final_norm,
        consts,
        lm_head_embeddings,
        qtypes.at("lm_head.qtype"));

    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    set_name(logits, "logits");

    ov::ParameterVector inputs{
        beam_idx,
        deepstack_visual_embeds,
        visual_pos_masks,
        inputs_embeds,
        position_ids,
        attention_mask
    };

    auto model = std::make_shared<ov::Model>(ov::OutputVector({logits->output(0)}), sinks, inputs);

    const int file_type = std::get<int>(configs.at("file_type"));

    if (file_type == 1 || file_type == 0) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return model;
}

std::shared_ptr<ov::Model> create_text_embeddings_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {

    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(input, "input");

    auto [inputs_embeds, embeddings_table] = make_embedding(
        "model.embed_tokens",
        input->output(0),
        consts,
        qtypes.at("model.embed_tokens.qtype"));

    auto result = std::make_shared<ov::op::v0::Result>(inputs_embeds);
    set_name(result, "inputs_embeds");

    ov::ParameterVector inputs{input};
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector{result->output(0)},
        inputs);

    return model;
}

std::shared_ptr<ov::Model> create_vision_embeddings_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    // Create input parameter
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, -1});
    set_name(hidden_states, "hidden_states");

    // Get patch embedding weights
    auto weight_0 = consts.at("self.proj.weight.raw0");
    auto weight_1 = consts.at("self.proj.weight.raw1");
    auto bias = consts.at("self.proj.bias");

    // Merge two 4D weights [1024, 3, 16, 16] into one 5D kernel [1024, 3, 2, 16, 16]
    ov::Shape raw_shape = weight_0.get_shape();
    OPENVINO_ASSERT(raw_shape.size() == 4, "self.proj.weight.raw0 rank must be 4");
    OPENVINO_ASSERT(weight_1.get_shape() == raw_shape, "self.proj.weight.raw1 shape mismatch");

    ov::Shape merged_shape = {raw_shape[0], raw_shape[1], 2, raw_shape[2], raw_shape[3]};
    ov::Tensor merged_weight(weight_0.get_element_type(), merged_shape);

    size_t plane_size = raw_shape[2] * raw_shape[3];
    size_t channel_plane = raw_shape[1] * plane_size;
    size_t out_plane = 2 * channel_plane;

    if (weight_0.get_element_type() == ov::element::f32) {
        const float* src0 = weight_0.data<float>();
        const float* src1 = weight_1.data<float>();
        float* dst = merged_weight.data<float>();

        for (size_t oc = 0; oc < raw_shape[0]; ++oc) {
            for (size_t ic = 0; ic < raw_shape[1]; ++ic) {
                size_t src_offset = oc * channel_plane + ic * plane_size;
                size_t dst_offset_0 = oc * out_plane + ic * 2 * plane_size;
                size_t dst_offset_1 = dst_offset_0 + plane_size;

                std::memcpy(dst + dst_offset_0, src0 + src_offset, plane_size * sizeof(float));
                std::memcpy(dst + dst_offset_1, src1 + src_offset, plane_size * sizeof(float));
            }
        }
    } else if (weight_0.get_element_type() == ov::element::f16) {
        const ov::float16* src0 = weight_0.data<ov::float16>();
        const ov::float16* src1 = weight_1.data<ov::float16>();
        ov::float16* dst = merged_weight.data<ov::float16>();

        for (size_t oc = 0; oc < raw_shape[0]; ++oc) {
            for (size_t ic = 0; ic < raw_shape[1]; ++ic) {
                size_t src_offset = oc * channel_plane + ic * plane_size;
                size_t dst_offset_0 = oc * out_plane + ic * 2 * plane_size;
                size_t dst_offset_1 = dst_offset_0 + plane_size;

                std::memcpy(dst + dst_offset_0, src0 + src_offset, plane_size * sizeof(ov::float16));
                std::memcpy(dst + dst_offset_1, src1 + src_offset, plane_size * sizeof(ov::float16));
            }
        }
    } else {
        OPENVINO_THROW("Unsupported self.proj.weight.raw element type");
    }

    auto weight_const = std::make_shared<ov::op::v0::Constant>(merged_weight);
    set_name(weight_const, "self.proj.weight");

    std::shared_ptr<ov::Node> weight_f32 = weight_const;
    if (merged_weight.get_element_type() != ov::element::f32) {
        weight_f32 = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f32);
    }

    auto bias_const = std::make_shared<ov::op::v0::Constant>(bias);
    set_name(bias_const, "self.proj.bias");
    auto bias_f32 = std::make_shared<ov::op::v0::Convert>(bias_const, ov::element::f32);

    // Reshape input from [N, ?] to [-1, 3, 2, 16, 16]
    int patch_size = std::get<int>(configs.at("patch_size"));
    auto input_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{5},
        std::vector<int64_t>{-1, 3, 2, patch_size, patch_size});

    auto reshaped_input = std::make_shared<ov::op::v1::Reshape>(
        hidden_states,
        input_shape,
        false);

    // Convolution
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        reshaped_input,
        weight_f32,
        ov::Strides{2, static_cast<size_t>(patch_size), static_cast<size_t>(patch_size)},
        ov::CoordinateDiff{0, 0, 0},
        ov::CoordinateDiff{0, 0, 0},
        ov::Strides{1, 1, 1});

    // Add bias
    auto bias_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{5},
        std::vector<int64_t>{1, static_cast<int64_t>(bias.get_shape()[0]), 1, 1, 1});

    auto bias_reshape = std::make_shared<ov::op::v1::Reshape>(
        bias_f32,
        bias_shape,
        false);

    auto conv_add = std::make_shared<ov::op::v1::Add>(
        conv,
        bias_reshape,
        ov::op::AutoBroadcastType::NUMPY);

    // Reshape output to [-1, 1024]
    auto output_shape = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64,
        ov::Shape{2},
        std::vector<int64_t>{-1, static_cast<int64_t>(bias.get_shape()[0])});

    auto last_hidden_state = std::make_shared<ov::op::v1::Reshape>(
        conv_add,
        output_shape,
        false);

    auto result = std::make_shared<ov::op::v0::Result>(last_hidden_state);
    set_name(result, "last_hidden_state");

    ov::ParameterVector inputs{hidden_states};
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector({result->output(0)}),
        inputs);

    return model;
}

std::shared_ptr<ov::Model> create_vision_embeddings_pos_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{4, -1});
    set_name(input, "input");

    auto [last_hidden_state, embeddings] = make_embedding(
        "embeddings",
        input->output(0),
        consts,
        gguf_tensor_type::GGUF_TYPE_F16);

    auto result = std::make_shared<ov::op::v0::Result>(last_hidden_state);
    set_name(result, "last_hidden_state");

    ov::ParameterVector inputs{input};
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector({result->output(0)}),
        inputs);

    return model;
}

std::shared_ptr<ov::Model> create_vision_embeddings_merger_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes) {
    const int hidden_size = std::get<int>(configs.at("hidden_size"));
    const int num_heads = std::get<int>(configs.at("head_num"));
    const int head_dim = std::get<int>(configs.at("head_size"));
    const int layer_num = std::get<int>(configs.at("layer_num"));
    const float epsilon = std::get<float>(configs.at("rms_norm_eps"));

    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, hidden_size});
    set_name(hidden_states, "hidden_states");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{1, -1, -1});
    set_name(attention_mask, "attention_mask");

    auto rotary_pos_emb = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, 32});
    set_name(rotary_pos_emb, "rotary_pos_emb");

    ov::Output<ov::Node> current = hidden_states;
    std::vector<ov::Output<ov::Node>> deepstack_outputs;

    for (int i = 0; i < layer_num; ++i) {
        auto norm1 = make_layer_norm_with_bias(
            format("self.blocks.%d.norm1", i),
            current,
            consts,
            epsilon);

        auto [attn_out, attn_probs] = make_vision_attention(
            format("self.blocks.%d", i),
            norm1,
            attention_mask,
            rotary_pos_emb,
            consts,
            qtypes,
            num_heads,
            head_dim);

        auto attn_add = std::make_shared<ov::op::v1::Add>(
            current,
            attn_out,
            ov::op::AutoBroadcastType::NUMPY);

        auto norm2 = make_layer_norm_with_bias(
            format("self.blocks.%d.norm2", i),
            attn_add,
            consts,
            epsilon);

        auto mlp_out = make_vision_mlp(
            format("self.blocks.%d.mlp", i),
            norm2,
            consts,
            qtypes);

        current = std::make_shared<ov::op::v1::Add>(
            attn_add,
            mlp_out,
            ov::op::AutoBroadcastType::NUMPY);

        if (i == 5 || i == 11 || i == 17) {
            auto reshape_4096 = std::make_shared<ov::op::v1::Reshape>(
                current,
                std::make_shared<ov::op::v0::Constant>(
                    ov::element::i64,
                    ov::Shape{2},
                    std::vector<int64_t>{-1, 4096}),
                false);

            int deep_idx = 0;
            if (i == 11) {
                deep_idx = 1;
            } else if (i == 17) {
                deep_idx = 2;
            }

            auto deep_out = make_merger_block(
                format("self.deepstack_merger_list.%d", deep_idx),
                reshape_4096,
                consts,
                qtypes,
                epsilon);

            deepstack_outputs.push_back(deep_out);
        }
    }

    auto final_norm = make_layer_norm_with_bias(
        "self.norm",
        current,
        consts,
        epsilon);

    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(
        final_norm,
        std::make_shared<ov::op::v0::Constant>(
            ov::element::i64,
            ov::Shape{2},
            std::vector<int64_t>{-1, 4096}),
        false);

    auto last_hidden_state = make_fc(
        "self.merger.linear_fc1",
        final_reshape,
        consts,
        qtypes.at("self.merger.linear_fc1.qtype"),
        false,
        -1);

    auto last_hidden_state_gelu = std::make_shared<ov::op::v7::Gelu>(
        last_hidden_state,
        ov::op::GeluApproximationMode::ERF);

    auto last_hidden_state_out = make_fc(
        "self.merger.linear_fc2",
        last_hidden_state_gelu,
        consts,
        qtypes.at("self.merger.linear_fc2.qtype"),
        false,
        -1);


    OPENVINO_ASSERT(deepstack_outputs.size() == 3, "deepstack outputs must be 3");

    auto unsq_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

    auto deep0 = std::make_shared<ov::op::v0::Unsqueeze>(deepstack_outputs[0], unsq_axis);
    auto deep1 = std::make_shared<ov::op::v0::Unsqueeze>(deepstack_outputs[1], unsq_axis);
    auto deep2 = std::make_shared<ov::op::v0::Unsqueeze>(deepstack_outputs[2], unsq_axis);

    auto deepstack_feature_lists = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{deep0, deep1, deep2},
        0);

    auto result0 = std::make_shared<ov::op::v0::Result>(last_hidden_state_out);
    set_name(result0, "last_hidden_state");

    auto result1 = std::make_shared<ov::op::v0::Result>(deepstack_feature_lists);
    set_name(result1, "deepstack_feature_lists");

    ov::ParameterVector inputs{hidden_states, attention_mask, rotary_pos_emb};
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector({result0->output(0), result1->output(0)}),
        inputs);

    return model;
}

} // namespace

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path, const bool enable_save_ov_model) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::stringstream ss;
    ss << "Loading and unpacking model from: " << model_path;
    ov::genai::utils::print_gguf_debug_info(ss.str());
    auto [config, consts, qtypes] = load_gguf(model_path);
    auto load_finish_time = std::chrono::high_resolution_clock::now();

    ss.str("");
    ss << "Loading and unpacking model done. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(load_finish_time - start_time).count() << "ms";
    ov::genai::utils::print_gguf_debug_info(ss.str());

    std::shared_ptr<ov::Model> model;
    const std::string model_arch = std::get<std::string>(config.at("architecture"));
    ss.str("");
    ss << "Start generating OpenVINO model...";
    ov::genai::utils::print_gguf_debug_info(ss.str());
    if (!model_arch.compare("llama") || !model_arch.compare("qwen2") || !model_arch.compare("qwen3")) {
        model = create_language_model(config, consts, qtypes);
        if (enable_save_ov_model){
            std::filesystem::path gguf_model_path(model_path);
            std::filesystem::path save_path = gguf_model_path.parent_path() / "openvino_model.xml";
            ov::genai::utils::save_openvino_model(model, save_path.string(), true);
        }
    } else if (!model_arch.compare("qwen3vl")) {
        std::shared_ptr<ov::Model> vlm_llm_model = create_vlm_language_model(config, consts, qtypes);
        std::shared_ptr<ov::Model> text_embeddings_model = create_text_embeddings_model(config, consts, qtypes);

        if (enable_save_ov_model) {
            std::filesystem::path gguf_model_path(model_path);

            std::filesystem::path lm_save_path =
                gguf_model_path.parent_path() / "openvino_language_model.xml";
            ov::genai::utils::save_openvino_model(vlm_llm_model, lm_save_path.string(), true);
            
            std::filesystem::path text_emb_save_path =
                gguf_model_path.parent_path() / "openvino_text_embeddings_model.xml";
            ov::genai::utils::save_openvino_model(text_embeddings_model, text_emb_save_path.string(), true);
        }
    } else if (!model_arch.compare("clip")) {

        std::shared_ptr<ov::Model> vision_embeddings_model = create_vision_embeddings_model(config, consts, qtypes);
        std::shared_ptr<ov::Model> vision_embeddings_pos_model = create_vision_embeddings_pos_model(config, consts, qtypes);
        std::shared_ptr<ov::Model> vision_embeddings_merger_model = create_vision_embeddings_merger_model(config, consts, qtypes);

        if (enable_save_ov_model) {
            std::filesystem::path gguf_model_path(model_path);

            std::filesystem::path vision_emb_save_path =
                gguf_model_path.parent_path() / "openvino_vision_embeddings_model.xml";
            ov::genai::utils::save_openvino_model(vision_embeddings_model, vision_emb_save_path.string(), true);

            std::filesystem::path vision_pos_save_path =
                gguf_model_path.parent_path() / "openvino_vision_embeddings_pos_model.xml";
            ov::genai::utils::save_openvino_model(vision_embeddings_pos_model, vision_pos_save_path.string(), true);

            std::filesystem::path vision_merger_save_path =
                gguf_model_path.parent_path() / "openvino_vision_embeddings_merger_model.xml";
            ov::genai::utils::save_openvino_model(vision_embeddings_merger_model, vision_merger_save_path.string(), true);
        }
    } else {
        OPENVINO_THROW("Unsupported model architecture '", model_arch, "'");
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - load_finish_time).count();
    ss.str("");
    ss << "Model generation done. Time: " << duration << "ms";
    ov::genai::utils::print_gguf_debug_info(ss.str());

    return model;
}
