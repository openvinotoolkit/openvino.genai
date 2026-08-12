// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/zimage_transformer_2d_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

namespace ov {
namespace genai {

size_t get_vae_scale_factor(const std::filesystem::path& vae_config_path);

ZImageTransformer2DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "sample_size", sample_size);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    m_vae_scale_factor = ov::genai::get_vae_scale_factor(root_dir.parent_path() / "vae_decoder" / "config.json");
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::filesystem::path& root_dir,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties)
    : ZImageTransformer2DModel(root_dir) {
    compile(device, properties);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::string& model,
                                                  const Tensor& weights,
                                                  const Config& config,
                                                  const size_t vae_scale_factor) :
    m_config(config), m_vae_scale_factor(vae_scale_factor) {
    m_model = utils::singleton_core().read_model(model, weights);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::string& model,
                                                  const Tensor& weights,
                                                  const Config& config,
                                                  const size_t vae_scale_factor,
                                                  const std::string& device,
                                                  const ov::AnyMap& properties) :
    ZImageTransformer2DModel(model, weights, config, vae_scale_factor) {
    compile(device, properties);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const ZImageTransformer2DModel&) = default;

ZImageTransformer2DModel ZImageTransformer2DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request), "ZImageTransformer2DModel must have exactly one of m_model or m_request initialized");

    ZImageTransformer2DModel cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const ZImageTransformer2DModel::Config& ZImageTransformer2DModel::get_config() const {
    return m_config;
}

ZImageTransformer2DModel& ZImageTransformer2DModel::reshape(int batch_size,
                                                           int height,
                                                           int width,
                                                           int tokenizer_model_max_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    height /= m_vae_scale_factor;
    width /= m_vae_scale_factor;

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "timestep") {
            name_to_shape[input_name][0] = 1;
        } else if (input_name == "sample") {
            name_to_shape[input_name] = {batch_size, m_config.in_channels, height, width};
        } else if (input_name == "encoder_hidden_states") {
            name_to_shape[input_name] = {batch_size, 128, name_to_shape[input_name][2]};
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}

ZImageTransformer2DModel& ZImageTransformer2DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("transformer"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "ZImage Transformer 2D model");
    m_request = compiled_model.create_infer_request();
    m_model.reset();

    return *this;
}

void ZImageTransformer2DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, encoder_hidden_states);
}

void ZImageTransformer2DModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    if(adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

ov::Tensor ZImageTransformer2DModel::step(ov::Tensor sample, ov::Tensor timestep, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    
    m_request.set_tensor("hidden_states", sample);
    m_request.set_tensor("timestep", timestep);
    m_request.set_tensor("encoder_hidden_states", encoder_hidden_states);
    
    m_request.infer();
    
    return m_request.get_output_tensor();
}

}  // namespace genai
}  // namespace ov
