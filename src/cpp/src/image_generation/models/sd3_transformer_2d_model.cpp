// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/sd3_transformer_2d_model.hpp"
#include "image_generation/models/sd3transformer_2d_inference_dynamic.hpp"
#include "image_generation/models/sd3transformer_2d_inference_static_bs1.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

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
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
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
    m_model = utils::singleton_core().read_model(model, weights);
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

SD3Transformer2DModel SD3Transformer2DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ (m_impl != nullptr), "SD3Transformer2DModel must have exactly one of m_model or m_impl initialized");

    SD3Transformer2DModel cloned = *this;
    
    if (m_model) {
        cloned.m_model = m_model->clone();
    } else if (m_impl) {
        cloned.m_impl = m_impl->clone();
    }

    return cloned;
}

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

    SD3Transformer2DModel::Inference::reshape(m_model, batch_size, height, width, tokenizer_model_max_length);

    return *this;
}

SD3Transformer2DModel& SD3Transformer2DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("transformer"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }

    if (device.find("NPU") != std::string::npos) {
        m_impl = std::make_shared<SD3Transformer2DModel::InferenceStaticBS1>();
    }
    else {
        m_impl = std::make_shared<SD3Transformer2DModel::InferenceDynamic>();
    }

    m_impl->compile(m_model, device, *filtered_properties);

    // release the original model
    m_model.reset();

    return *this;
}

void SD3Transformer2DModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_impl, "Transformer model must be compiled first");
    if(adapters) {
        m_impl->set_adapters(m_adapter_controller, *adapters);
    }
}

void SD3Transformer2DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_impl, "Transformer model must be compiled first");
    m_impl->set_hidden_states(tensor_name, encoder_hidden_states);
}

ov::Tensor SD3Transformer2DModel::infer(const ov::Tensor latent_model_input, const ov::Tensor timestep) {
    OPENVINO_ASSERT(m_impl, "Transformer model must be compiled first. Cannot infer non-compiled model");
    return m_impl->infer(latent_model_input, timestep);
}

}  // namespace genai
}  // namespace ov
