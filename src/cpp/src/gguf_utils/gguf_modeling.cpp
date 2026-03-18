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
