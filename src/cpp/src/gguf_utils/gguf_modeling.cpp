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
#include "openvino/pass/manager.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"

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

    // Set runtime options
    if (std::get<int>(configs.at("file_type")) == 1 || std::get<int>(configs.at("file_type")) == 0) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return model;
}

} // namespace

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path,
                                            bool enable_save_ov_model,
                                            bool use_legacy_reader) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::stringstream ss;

    // Escape hatch while the legacy reader is still around: gguf_reader("LEGACY") uses the old
    // hand-written builder instead of the GGUF frontend. Supports only llama/qwen2/qwen3 and
    // ignores some metadata (e.g. rope_freqs.weight). Remove with create_language_model().
    if (use_legacy_reader) {
        ss << "ov::genai::gguf_reader is '" << ov::genai::LEGACY_GGUF_READER
           << "': converting with the legacy GGUF reader instead of the OpenVINO GGUF frontend: " << model_path;
        ov::genai::utils::print_gguf_debug_info(ss.str());

        auto [config, consts, qtypes] = load_gguf(model_path);
        const std::string model_arch = std::get<std::string>(config.at("architecture"));
        OPENVINO_ASSERT(model_arch == "llama" || model_arch == "qwen2" || model_arch == "qwen3",
                        "The legacy GGUF reader does not support architecture '",
                        model_arch,
                        "'. Set the ov::genai::gguf_reader property to '",
                        ov::genai::FRONTEND_GGUF_READER,
                        "' (the default) to use the GGUF frontend.");
        auto legacy_model = create_language_model(config, consts, qtypes);
        if (enable_save_ov_model) {
            std::filesystem::path gguf_model_path(model_path);
            std::filesystem::path save_path = gguf_model_path.parent_path() / "openvino_model.xml";
            ov::genai::utils::save_openvino_model(legacy_model, save_path.string(), true);
        }
        ss.str("");
        ss << "Legacy GGUF conversion done. Time: "
           << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::high_resolution_clock::now() - start_time)
                  .count()
           << "ms";
        ov::genai::utils::print_gguf_debug_info(ss.str());
        return legacy_model;
    }

    ss << "Loading and converting GGUF model via the OpenVINO GGUF frontend: " << model_path;
    ov::genai::utils::print_gguf_debug_info(ss.str());

    // The GGUF frontend always converts to a stateless graph (explicit KV-cache input/output
    // pairs, like optimum-intel before its own make-stateful pass). Register GGUFMakeStateful as
    // a transformation extension so the frontend turns each cache into a ReadValue/Concat/Assign
    // state during conversion; scoped to this call rather than global via ov::Core::add_extension.
    // AdaptToGenAI then rewires IO to what StatefulLLMPipeline expects (input_ids /
    // attention_mask / position_ids / beam_idx -> logits).
    ov::frontend::gguf::FrontEnd frontend;
    frontend.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
        ov::frontend::gguf::pass::GGUFMakeStateful()));
    auto model = frontend.convert(frontend.load(model_path));

    ov::pass::Manager manager;
    manager.register_pass<ov::frontend::gguf::pass::AdaptToGenAI>();
    manager.run_passes(model);

    if (enable_save_ov_model) {
        std::filesystem::path gguf_model_path(model_path);
        std::filesystem::path save_path = gguf_model_path.parent_path() / "openvino_model.xml";
        ov::genai::utils::save_openvino_model(model, save_path.string(), true);
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    ss.str("");
    ss << "GGUF model conversion done. Time: " << duration << "ms";
    ov::genai::utils::print_gguf_debug_info(ss.str());

    return model;
}
