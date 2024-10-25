// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/unet2d_condition_model.hpp"
#include "text2image/models/unet_inference_dynamic.hpp"
#include "text2image/models/unet_inference_static_bs1.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "lora_helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

UNet2DConditionModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "sample_size", sample_size);
    read_json_param(data, "time_cond_proj_dim", time_cond_proj_dim);

    file.close();

    // block_out_channels should be read from VAE encoder / decoder config to compute proper m_vae_scale_factor
    std::filesystem::path vae_config_path = config_path.parent_path().parent_path() / "vae_decoder" / "config.json";
    file.open(vae_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", vae_config_path);
    data = nlohmann::json::parse(file);
    read_json_param(data, "block_out_channels", block_out_channels);
}

UNet2DConditionModel::UNet2DConditionModel(const std::filesystem::path& root_dir) :
    m_config(root_dir / "config.json") {
    ov::Core core = utils::singleton_core();
    m_model = core.read_model((root_dir / "openvino_model.xml").string());
    // compute VAE scale factor
    m_vae_scale_factor = std::pow(2, m_config.block_out_channels.size() - 1);
}

UNet2DConditionModel::UNet2DConditionModel(const std::filesystem::path& root_dir,
                                           const std::string& device,
                                           const ov::AnyMap& properties) :
    UNet2DConditionModel(root_dir) {
    AdapterConfig adapters;
    if (auto filtered_properties = extract_adapters_from_properties(properties, &adapters)) {
        adapters.set_tensor_name_prefix(adapters.get_tensor_name_prefix().value_or("lora_unet"));
        m_adapter_controller = AdapterController(m_model, adapters, device);
        compile(device, *filtered_properties);
    } else {
        compile(device, properties);
    }
}

UNet2DConditionModel::UNet2DConditionModel(const UNet2DConditionModel&) = default;

const UNet2DConditionModel::Config& UNet2DConditionModel::get_config() const {
    return m_config;
}

size_t UNet2DConditionModel::get_vae_scale_factor() const {
    return m_vae_scale_factor;
}

UNet2DConditionModel& UNet2DConditionModel::reshape(int batch_size, int height, int width, int tokenizer_model_max_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    height /= m_vae_scale_factor;
    width /= m_vae_scale_factor;

    UNetInference::reshape(m_model, batch_size, height, width, tokenizer_model_max_length);

    return *this;
}

UNet2DConditionModel& UNet2DConditionModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");

    if (device == "NPU") {
        m_impl = std::make_shared<UNet2DConditionModel::UNetInferenceStaticBS1>();
    }
    else {
        m_impl = std::make_shared<UNet2DConditionModel::UNetInferenceDynamic>();
    }

    m_impl->compile(m_model, device, properties);

    // release the original model
    m_model.reset();

    return *this;
}

void UNet2DConditionModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_impl, "UNet model must be compiled first");
    m_impl->set_hidden_states(tensor_name, encoder_hidden_states);
}

void UNet2DConditionModel::set_adapters(const AdapterConfig& adapters) {
    OPENVINO_ASSERT(m_impl, "UNet model must be compiled first");
    m_impl->set_adapters(m_adapter_controller, adapters);
}

ov::Tensor UNet2DConditionModel::infer(ov::Tensor sample, ov::Tensor timestep) {
    OPENVINO_ASSERT(m_impl, "UNet model must be compiled first. Cannot infer non-compiled model");
    return m_impl->infer(sample, timestep);
}

} // namespace genai
} // namespace ov
