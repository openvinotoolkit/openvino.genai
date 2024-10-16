// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/unet2d_condition_model.hpp"

#include <fstream>

#include "openvino/runtime/core.hpp"

#include "utils.hpp"
#include "lora_helper.hpp"

namespace ov {
namespace genai {

UNet2DConditionModel::Config::Config(const std::string& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "sample_size", sample_size);
    read_json_param(data, "block_out_channels", block_out_channels);
    read_json_param(data, "time_cond_proj_dim", time_cond_proj_dim);
}

UNet2DConditionModel::UNet2DConditionModel(const std::string root_dir) :
    m_config(root_dir + "/config.json") {
    m_model = ov::Core().read_model(root_dir + "/openvino_model.xml");
    // compute VAE scale factor
    m_vae_scale_factor = std::pow(2, m_config.block_out_channels.size() - 1);
}

UNet2DConditionModel::UNet2DConditionModel(const std::string& root_dir,
                        const std::string& device,
                        const ov::AnyMap& properties) :
    UNet2DConditionModel(root_dir) {
    AdapterConfig adapters;
    if(auto filtered_properties = extract_adapters_from_properties(properties, &adapters)) {
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

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto && input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "timestep") {
            name_to_shape[input_name][0] = 1;
        } else if (input_name == "sample") {
            name_to_shape[input_name] = {batch_size, name_to_shape[input_name][1], height, width};
        } else if (input_name == "time_ids" || input_name == "text_embeds") {
            name_to_shape[input_name][0] = batch_size;
        } else if (input_name == "encoder_hidden_states") {
            name_to_shape[input_name][0] = batch_size;
            name_to_shape[input_name][1] = tokenizer_model_max_length;
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}

UNet2DConditionModel& UNet2DConditionModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = ov::Core().compile_model(m_model, device, properties);
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

void UNet2DConditionModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "UNet model must be compiled first");
    m_request.set_tensor(tensor_name, encoder_hidden_states);
}

void UNet2DConditionModel::set_adapters(const AdapterConfig& adapters) {
    m_adapter_controller.apply(m_request, adapters);
}

ov::Tensor UNet2DConditionModel::infer(ov::Tensor sample, ov::Tensor timestep) {
    OPENVINO_ASSERT(m_request, "UNet model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("sample", sample);
    m_request.set_tensor("timestep", timestep);

    m_request.infer();

    return m_request.get_output_tensor();
}

} // namespace genai
} // namespace ov
