// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

size_t get_vae_scale_factor(const std::filesystem::path& vae_config_path);

SD3Transformer2DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "sample_size", sample_size);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "joint_attention_dim", joint_attention_dim);
}

SD3Transformer2DModel::SD3Transformer2DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model((root_dir / "openvino_model.xml").string());
    m_vae_scale_factor = get_vae_scale_factor(root_dir.parent_path() / "vae_decoder" / "config.json");
}

SD3Transformer2DModel::SD3Transformer2DModel(const std::filesystem::path& root_dir,
                                             const std::string& device,
                                             const ov::AnyMap& properties)
    : SD3Transformer2DModel(root_dir) {
    compile(device, properties);
}

SD3Transformer2DModel::SD3Transformer2DModel(const std::string& model,
                                             const Tensor& weights,
                                             const Config& config,
                                             const size_t vae_scale_factor) :
    m_config(config), m_vae_scale_factor(vae_scale_factor) {
    ov::Core core = utils::singleton_core();
    m_model = core.read_model(model, weights);
}

SD3Transformer2DModel::SD3Transformer2DModel(const std::string& model,
                                             const Tensor& weights,
                                             const Config& config,
                                             const size_t vae_scale_factor,
                                             const std::string& device,
                                             const ov::AnyMap& properties) :
    SD3Transformer2DModel(model, weights, config, vae_scale_factor) {
    compile(device, properties);
}

SD3Transformer2DModel::SD3Transformer2DModel(const SD3Transformer2DModel&) = default;

const SD3Transformer2DModel::Config& SD3Transformer2DModel::get_config() const {
    return m_config;
}

SD3Transformer2DModel& SD3Transformer2DModel::reshape(int batch_size,
                                                      int height,
                                                      int width,
                                                      int tokenizer_model_max_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    // hidden_states=latent_model_input,
    // timestep=timestep,
    // encoder_hidden_states=prompt_embeds,
    // pooled_projections=pooled_prompt_embeds,

    height /= m_vae_scale_factor;
    width /= m_vae_scale_factor;

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "timestep") {
            name_to_shape[input_name][0] = 1;
        } else if (input_name == "hidden_states") {
            name_to_shape[input_name] = {batch_size, name_to_shape[input_name][1], height, width};
        } else if (input_name == "encoder_hidden_states") {
            name_to_shape[input_name][0] = batch_size;
            name_to_shape[input_name][1] =
                tokenizer_model_max_length *
                2;  // x2 is necessary because of the concatenation of prompt_embeds and t5_prompt_embeds
        } else if (input_name == "pooled_projections") {
            name_to_shape[input_name][0] = batch_size;
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}

SD3Transformer2DModel& SD3Transformer2DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

void SD3Transformer2DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, encoder_hidden_states);
}

ov::Tensor SD3Transformer2DModel::infer(const ov::Tensor latent_model_input, const ov::Tensor timestep) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", latent_model_input);
    m_request.set_tensor("timestep", timestep);
    m_request.infer();

    return m_request.get_output_tensor();
}

}  // namespace genai
}  // namespace ov
