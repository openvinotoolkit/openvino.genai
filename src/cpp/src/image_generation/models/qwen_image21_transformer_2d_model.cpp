// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/qwen_image21_transformer_2d_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

QwenImage21Transformer2DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "context_in_dim", context_in_dim);
    read_json_param(data, "attention_head_dim", attention_head_dim);
    read_json_param(data, "num_layers", num_layers);
    if (data.contains("axes_dims_rope")) {
        axes_dims_rope = data["axes_dims_rope"].get<std::vector<size_t>>();
    }
}

QwenImage21Transformer2DModel::QwenImage21Transformer2DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

QwenImage21Transformer2DModel::QwenImage21Transformer2DModel(const std::filesystem::path& root_dir,
                                                             const std::string& device,
                                                             const ov::AnyMap& properties)
    : QwenImage21Transformer2DModel(root_dir) {
    compile(device, properties);
}

QwenImage21Transformer2DModel::QwenImage21Transformer2DModel(const QwenImage21Transformer2DModel&) = default;

QwenImage21Transformer2DModel QwenImage21Transformer2DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "QwenImage21Transformer2DModel must have exactly one of m_model or m_request initialized");

    QwenImage21Transformer2DModel cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const QwenImage21Transformer2DModel::Config& QwenImage21Transformer2DModel::get_config() const {
    return m_config;
}

QwenImage21Transformer2DModel& QwenImage21Transformer2DModel::compile(const std::string& device,
                                                                      const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("transformer"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "QwenImage 2.1 Transformer 2D model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

void QwenImage21Transformer2DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor tensor) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, tensor);
}

void QwenImage21Transformer2DModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

ov::Tensor QwenImage21Transformer2DModel::infer(const ov::Tensor latent, const ov::Tensor timestep) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", latent);
    m_request.set_tensor("timestep", timestep);
    m_request.infer();

    return m_request.get_output_tensor();
}

}  // namespace genai
}  // namespace ov
