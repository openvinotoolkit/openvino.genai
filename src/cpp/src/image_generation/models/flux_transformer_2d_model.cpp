// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/flux_transformer_2d_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

FluxTransformer2DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    // read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "in_channels", in_channels);
    // read_json_param(data, "num_layers", num_layers);
    // read_json_param(data, "attention_head_dim", attention_head_dim);
    // read_json_param(data, "num_attention_heads", num_attention_heads);
    // read_json_param(data, "joint_attention_dim", joint_attention_dim);
    // read_json_param(data, "pooled_projection_dim", pooled_projection_dim);
    // read_json_param(data, "guidance_embeds", guidance_embeds);
    // read_json_param(data, "num_single_layers", num_single_layers);


    file.close();

    // block_out_channels should be read from VAE encoder / decoder config to compute proper m_vae_scale_factor
    std::filesystem::path vae_config_path = config_path.parent_path().parent_path() / "vae_decoder" / "config.json";
    file.open(vae_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", vae_config_path);
    data = nlohmann::json::parse(file);
    read_json_param(data, "block_out_channels", block_out_channels);
}

FluxTransformer2DModel::FluxTransformer2DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model((root_dir / "openvino_model.xml").string());

    // compute VAE scale factor
    m_vae_scale_factor = std::pow(2, m_config.block_out_channels.size());
}

FluxTransformer2DModel::FluxTransformer2DModel(const std::filesystem::path& root_dir,
                                             const std::string& device,
                                             const ov::AnyMap& properties)
    : FluxTransformer2DModel(root_dir) {
    compile(device, properties);
}

FluxTransformer2DModel::FluxTransformer2DModel(const FluxTransformer2DModel&) = default;

const FluxTransformer2DModel::Config& FluxTransformer2DModel::get_config() const {
    return m_config;
}

FluxTransformer2DModel& FluxTransformer2DModel::reshape(int batch_size,
                                                      int height,
                                                      int width,
                                                      int tokenizer_model_max_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    // hidden_states=latent_model_input,
    // timestep=timestep,
    // encoder_hidden_states=prompt_embeds,
    // pooled_projections=pooled_prompt_embeds,

    // height /= m_vae_scale_factor;
    // width /= m_vae_scale_factor;

    // std::map<std::string, ov::PartialShape> name_to_shape;

    // for (auto&& input : m_model->inputs()) {
    //     std::string input_name = input.get_any_name();
    //     name_to_shape[input_name] = input.get_partial_shape();
    //     if (input_name == "timestep") {
    //         name_to_shape[input_name][0] = batch_size;
    //     } else if (input_name == "hidden_states") {
    //         name_to_shape[input_name] = {batch_size, name_to_shape[input_name][1], height, width};
    //     } else if (input_name == "encoder_hidden_states") {
    //         name_to_shape[input_name][0] = batch_size;
    //         name_to_shape[input_name][1] =
    //             tokenizer_model_max_length *
    //             2;  // x2 is necessary because of the concatenation of prompt_embeds and t5_prompt_embeds
    //     } else if (input_name == "pooled_projections") {
    //         name_to_shape[input_name][0] = batch_size;
    //     }
    // }

    // m_model->reshape(name_to_shape);

    return *this;
}

FluxTransformer2DModel& FluxTransformer2DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

void FluxTransformer2DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, encoder_hidden_states);
}

size_t FluxTransformer2DModel::get_vae_scale_factor() const {
    return m_vae_scale_factor;
}

ov::Tensor FluxTransformer2DModel::infer(const ov::Tensor latent_model_input, const ov::Tensor timestep) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", latent_model_input);
    m_request.set_tensor("timestep", timestep);
    m_request.infer();

    return m_request.get_output_tensor();
}

}  // namespace genai
}  // namespace ov
