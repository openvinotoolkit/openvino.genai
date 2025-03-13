#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <stdexcept>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"

#include "building_blocks.hpp"
#include "gguf_modeling.hpp"


using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

// Also valid for other models, e.g. SmolLMs
std::shared_ptr<ov::Model> create_llama_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts) {

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Start generating OV model..." << std::endl;

    // Create input parameters
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    input_ids->output(0).set_names({"input_ids"});

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    attention_mask->output(0).set_names({"attention_mask"});

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    position_ids->output(0).set_names({"position_ids"});

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::PartialShape{-1});
    beam_idx->output(0).set_names({"beam_idx"});

    // Create embedding layer
    auto [inputs_embeds, embeddings] = make_embedding(
        "model.embed_tokens",
        input_ids->output(0),
        consts,
        static_cast<QType>(std::get<int>(configs.at("qtype"))));

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
        static_cast<QType>(std::get<int>(configs.at("qtype"))));

    // Create results
    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    logits->output(0).set_names({"logits"});

    // Create model
    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    auto model = std::make_shared<ov::Model>(ov::OutputVector({logits->output(0)}), sinks, inputs);

    // Set runtime options
    model->set_rt_info("f16", {"runtime_options", "KV_CACHE_PRECISION"});
    model->set_rt_info("8.0", {"runtime_options", "ACTIVATIONS_SCALE_FACTOR"});

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "Model generation done. Time: " << duration << "s" << std::endl;

    return model;
}

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path) {
    auto [config, consts] = load_gguf(model_path);

    std::shared_ptr<ov::Model> model;

    auto model_arch = std::get<std::string>(config.at("architecture"));
    std::cout << "Creating model with architecture:" << model_arch;
    

    if (!model_arch.compare("llama")) {
        model = create_llama_model(config, consts);
    }
    else {
        throw std::runtime_error(std::string("Unsupported model architecture") + model_arch);
    }

    return model;
}
