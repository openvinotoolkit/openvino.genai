// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/unet2d_condition_model.hpp"
#include "image_generation/models/unet_inference_dynamic.hpp"
#include "image_generation/models/unet_inference_static_bs1.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

size_t get_vae_scale_factor(const std::filesystem::path& vae_config_path);

UNet2DConditionModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "sample_size", sample_size);
    read_json_param(data, "time_cond_proj_dim", time_cond_proj_dim);
}

UNet2DConditionModel::UNet2DConditionModel(const std::filesystem::path& root_dir) :
    m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    m_vae_scale_factor = get_vae_scale_factor(root_dir.parent_path() / "vae_decoder" / "config.json");
}

UNet2DConditionModel::UNet2DConditionModel(const std::filesystem::path& root_dir,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : m_config(root_dir / "config.json") {
    m_vae_scale_factor = get_vae_scale_factor(root_dir.parent_path() / "vae_decoder" / "config.json");

    const auto [properties_without_blob, blob_path] = utils::extract_export_properties(properties);

    if (blob_path.has_value()) {
        import_model(*blob_path, device, properties_without_blob);
        return;
    }

    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    compile(device, properties_without_blob);
}

UNet2DConditionModel::UNet2DConditionModel(const std::string& model,
                                           const Tensor& weights,
                                           const Config& config,
                                           const size_t vae_scale_factor) :
    m_config(config), m_vae_scale_factor(vae_scale_factor) {
    m_model = utils::singleton_core().read_model(model, weights);
}

UNet2DConditionModel::UNet2DConditionModel(const std::string& model,
                                           const Tensor& weights,
                                           const Config& config,
                                           const size_t vae_scale_factor,
                                           const std::string& device,
                                           const ov::AnyMap& properties) :
    UNet2DConditionModel(model, weights, config, vae_scale_factor) {
    compile(device, properties);
}

UNet2DConditionModel::UNet2DConditionModel(const UNet2DConditionModel&) = default;

UNet2DConditionModel UNet2DConditionModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ (m_impl != nullptr), "UNet2DConditionModel must have exactly one of m_model or m_impl initialized");

    UNet2DConditionModel cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_impl = m_impl->clone();
    }

    return cloned;
}

const UNet2DConditionModel::Config& UNet2DConditionModel::get_config() const {
    return m_config;
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
    } else {
        m_impl = std::make_shared<UNet2DConditionModel::UNetInferenceDynamic>();
    }

    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("lora_unet"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    m_impl->compile(m_model, device, *filtered_properties);

    // release the original model
    m_model.reset();

    return *this;
}

void UNet2DConditionModel::import_model(const std::filesystem::path& blob_path, const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(!m_impl, "Model has been already compiled. Cannot re-compile already compiled model");

    if (device == "NPU") {
        m_impl = std::make_shared<UNet2DConditionModel::UNetInferenceStaticBS1>();
    } else {
        m_impl = std::make_shared<UNet2DConditionModel::UNetInferenceDynamic>();
    }

    m_impl->import_model(blob_path, device, properties);
}

void UNet2DConditionModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_impl, "UNet model must be compiled first");
    m_impl->set_hidden_states(tensor_name, encoder_hidden_states);
}

void UNet2DConditionModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_impl, "UNet model must be compiled first");
    if(adapters) {
        m_impl->set_adapters(m_adapter_controller, *adapters);
    }
}

ov::Tensor UNet2DConditionModel::infer(ov::Tensor sample, ov::Tensor timestep) {
    OPENVINO_ASSERT(m_impl, "UNet model must be compiled first. Cannot infer non-compiled model");
    return m_impl->infer(sample, timestep);
}

void UNet2DConditionModel::export_model(const std::filesystem::path& blob_path) {
    OPENVINO_ASSERT(m_impl, "UNet model must be compiled first. Cannot infer non-compiled model");
    m_impl->export_model(blob_path);
}

} // namespace genai
} // namespace ov
