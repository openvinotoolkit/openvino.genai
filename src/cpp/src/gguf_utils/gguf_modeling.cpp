#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <stdexcept>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"

#include "gguf.h"
#include "building_blocks.h"
#include "gguf_modeling.h"


using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

// Also valid for other models, e.g. SmolLMs
std::shared_ptr<ov::Model> create_llama_model(
    const std::map<std::string, float>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts) {

    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Start generating OV model..." << std::endl;

    // Create input parameters
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    attention_mask->set_friendly_name("attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    position_ids->set_friendly_name("position_ids");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");

    // Create embedding layer
    auto [inputs_embeds, embeddings] = make_embedding(
        "model.embed_tokens",
        input_ids->output(0),
        consts,
        static_cast<QType>(configs.at("qtype")));

    auto hidden_states = inputs_embeds;

    // Initialize RoPE
    auto rope_const = init_rope(
        static_cast<int64_t>(configs.at("head_size")),
        static_cast<int64_t>(configs.at("max_position_embeddings")),
        static_cast<float>(configs.at("rope_freq_base")));

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

    for (int i = 0; i < (int)configs.at("layer_num"); ++i) {
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
        configs.at("rms_norm_eps"));

    // LM head
    auto embed_out = make_lm_head(
        "lm_head",
        final_norm,
        consts,
        embeddings,
        static_cast<QType>(configs.at("qtype")));

    // Create results
    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    logits->set_friendly_name("logits");

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

QType get_quantization_type(int gguf_type) {
    switch(gguf_type) {
        case 0:
        case 1:
            std::cout << "Working with FP16 model" << std::endl;
            return QType::FP16;
            
        case 2:
        case 3:
            std::cout << "Working with INT4 quantized model" << std::endl;
            return QType::INT4;
            
        case 7:
            std::cout << "Working with INT8 quantized model" << std::endl;
            return QType::INT8;
            
        default:
            throw std::invalid_argument(
                "Unsupported GGUF quantization type: " + std::to_string(gguf_type));
    }
}

float metadata_to_float(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    auto tensor = std::get<ov::Tensor>(metadata.at(key));
    return *(tensor.data<ov::element_type_traits<ov::element::f32>::value_type>());
}

std::map<std::string, float> config_from_meta(const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, float> config;
    config["layer_num"] = metadata_to_float(metadata, "llama.block_count");
    config["head_num"] = metadata_to_float(metadata, "llama.attention.head_count");
    config["head_size"] = metadata_to_float(metadata, "llama.embedding_length") / 
                     metadata_to_float(metadata, "llama.attention.head_count");
    config["head_num_kv"] = metadata.count("llama.attention.head_count_kv") ?
            metadata_to_float(metadata, "llama.attention.head_count_kv") :
            metadata_to_float(metadata, "llama.attention.head_count");
    config["hidden_size"] = metadata_to_float(metadata, "llama.embedding_length");
    config["max_position_embeddings"] = metadata.count("llama.context_length") ?
            metadata_to_float(metadata, "llama.context_length") : 2048;
    config["rotary_dims"] = metadata_to_float(metadata, "llama.rope.dimension_count");
    config["rms_norm_eps"] = metadata_to_float(metadata, "llama.attention.layer_norm_rms_epsilon");
    config["rope_freq_base"] =metadata.count("llama.rope.freq_base") ?
            metadata_to_float(metadata, "llama.rope.freq_base") : 10000.0f;
    config["qtype"] = (float)get_quantization_type((int)metadata_to_float(metadata, "general.file_type"));

    return config;
}

std::unordered_map<std::string, ov::Tensor> consts_from_weights(const std::map<std::string, float>& config,
                                                            const std::unordered_map<std::string, ov::Tensor>& weights) {
    std::unordered_map<std::string, ov::Tensor> consts;

    consts["model.embed_tokens.weight"] = weights.at("token_embd.weight");
    consts["model.norm.weight"] = weights.at("output_norm.weight");
    consts["lm_head.weight"] = weights.count("output.weight") ? 
        weights.at("output.weight") : ov::Tensor();

    // Handle quantization scales and biases
    if (weights.count("token_embd.scales")) {
        consts["model.embed_tokens.scales"] = weights.at("token_embd.scales");
        consts["model.embed_tokens.biases"] = weights.at("token_embd.biases");
    }
    if (weights.count("output.scales")) {
        consts["lm_head.scales"] = weights.at("output.scales");
        consts["lm_head.biases"] = weights.at("output.biases");
    }

    // Process layer weights
    for (int i = 0; i < config.at("layer_num"); ++i) {
        consts[format("model.layers[{}].input_layernorm.weight", i)] = weights.at(format("blk.{}.attn_norm.weight", i));
        consts[format("model.layers[{}].post_attention_layernorm.weight", i)] = weights.at(format("blk.{}.ffn_norm.weight", i));
        
        // Attention weights
        consts[format("model.layers[{}].self_attn.q_proj.weight", i)] = weights.at(format("blk.{}.attn_q.weight", i));
        consts[format("model.layers[{}].self_attn.k_proj.weight", i)] = weights.at(format("blk.{}.attn_k.weight", i));
        consts[format("model.layers[{}].self_attn.v_proj.weight", i)] = weights.at(format("blk.{}.attn_v.weight", i));
        consts[format("model.layers[{}].self_attn.o_proj.weight", i)] = weights.at(format("blk.{}.attn_output.weight", i));

        // MLP weights
        consts[format("model.layers[{}].mlp.gate_proj.weight", i)] = weights.at(format("blk.{}.ffn_gate.weight", i));
        consts[format("model.layers[{}].mlp.up_proj.weight", i)] = weights.at(format("blk.{}.ffn_up.weight", i));
        consts[format("model.layers[{}].mlp.down_proj.weight", i)] = weights.at(format("blk.{}.ffn_down.weight", i));

        // Quantization parameters
        if (QType((int)config.at("qtype")) != QType::FP16) {
            consts[format("model.layers[{}].self_attn.q_proj.scales", i)] = weights.at(format("blk.{}.attn_q.scales", i));
            consts[format("model.layers[{}].self_attn.k_proj.scales", i)] = weights.at(format("blk.{}.attn_k.scales", i));
            consts[format("model.layers[{}].self_attn.v_proj.scales", i)] = weights.at(format("blk.{}.attn_v.scales", i));
            consts[format("model.layers[{}].self_attn.o_proj.scales", i)] = weights.at(format("blk.{}.attn_output.scales", i));
            consts[format("model.layers[{}].mlp.gate_proj.scales", i)] = weights.at(format("blk.{}.ffn_gate.scales", i));
            consts[format("model.layers[{}].mlp.up_proj.scales", i)] = weights.at(format("blk.{}.ffn_up.scales", i));
            consts[format("model.layers[{}].mlp.down_proj.scales", i)] = weights.at(format("blk.{}.ffn_down.scales", i));

            consts[format("model.layers[{}].self_attn.q_proj.biases", i)] = weights.at(format("blk.{}.attn_q.biases", i));
            consts[format("model.layers[{}].self_attn.k_proj.biases", i)] = weights.at(format("blk.{}.attn_k.biases", i));
            consts[format("model.layers[{}].self_attn.v_proj.biases", i)] = weights.at(format("blk.{}.attn_v.biases", i));
            consts[format("model.layers[{}].self_attn.o_proj.biases", i)] = weights.at(format("blk.{}.attn_output.biases", i));
            consts[format("model.layers[{}].mlp.gate_proj.biases", i)] = weights.at(format("blk.{}.ffn_gate.biases", i));
            consts[format("model.layers[{}].mlp.up_proj.biases", i)] = weights.at(format("blk.{}.ffn_up.biases", i));
            consts[format("model.layers[{}].mlp.down_proj.biases", i)] = weights.at(format("blk.{}.ffn_down.biases", i));
        }
    }

    return consts;
}


std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path) {
    auto gguf_data = load_gguf(model_path);
    auto weights = gguf_data.first;
    auto metadata = gguf_data.second;

    auto config = config_from_meta(metadata);
    auto consts = consts_from_weights(config, weights);
    std::shared_ptr<ov::Model> model;

    auto model_arch = std::get<std::string>(metadata["general.architecture"]);

    if (metadata.find("general.architecture") != metadata.end()) {
        
        std::cout << "Creating model with architecture:" << model_arch;
    }
    else {
        throw std::runtime_error(std::string("Unsupported model architecture") + model_arch);
    }

    if (!model_arch.compare("llama")) {
        model = create_llama_model(config, consts);
    }

    return model;
}
