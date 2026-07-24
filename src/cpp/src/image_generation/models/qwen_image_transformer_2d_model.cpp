// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/qwen_image_transformer_2d_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

namespace ov {
namespace genai {

namespace {

size_t get_qwen_image_vae_scale_factor(const std::filesystem::path& vae_config_path) {
    std::ifstream file(vae_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", vae_config_path);
    nlohmann::json data = nlohmann::json::parse(file);

    OPENVINO_ASSERT(data.contains("temperal_downsample"),
                    "QwenImage VAE config must contain 'temperal_downsample'");
    std::vector<bool> temperal_downsample = data["temperal_downsample"].get<std::vector<bool>>();
    return static_cast<size_t>(std::pow(2, temperal_downsample.size()));
}

}  // namespace

QwenImageTransformer2DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "guidance_embeds", guidance_embeds);
    read_json_param(data, "joint_attention_dim", joint_attention_dim);
    read_json_param(data, "attention_head_dim", attention_head_dim);
    read_json_param(data, "patch_size", patch_size);
    if (data.contains("axes_dims_rope")) {
        axes_dims_rope = data["axes_dims_rope"].get<std::vector<size_t>>();
    }
    if (data.contains("sample_size")) {
        default_sample_size = data["sample_size"].get<size_t>();
    }
}

QwenImageTransformer2DModel::QwenImageTransformer2DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    m_vae_scale_factor = get_qwen_image_vae_scale_factor(root_dir.parent_path() / "vae_decoder" / "config.json");
}

QwenImageTransformer2DModel::QwenImageTransformer2DModel(const std::filesystem::path& root_dir,
                                                         const std::string& device,
                                                         const ov::AnyMap& properties)
    : QwenImageTransformer2DModel(root_dir) {
    compile(device, properties);
}

QwenImageTransformer2DModel::QwenImageTransformer2DModel(const QwenImageTransformer2DModel&) = default;

QwenImageTransformer2DModel QwenImageTransformer2DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "QwenImageTransformer2DModel must have exactly one of m_model or m_request initialized");

    QwenImageTransformer2DModel cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const QwenImageTransformer2DModel::Config& QwenImageTransformer2DModel::get_config() const {
    return m_config;
}

QwenImageTransformer2DModel& QwenImageTransformer2DModel::reshape(int batch_size,
                                                                   int height,
                                                                   int width,
                                                                   int tokenizer_model_max_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    height /= m_vae_scale_factor;
    width /= m_vae_scale_factor;

    // QwenImage uses patchify: (B, C, H, W) -> (B, (H/2)*(W/2), C*4) packed
    const size_t image_seq_len = (height / 2) * (width / 2);

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "timestep") {
            name_to_shape[input_name] = {batch_size};
        } else if (input_name == "hidden_states") {
            // (B, image_seq_len, in_channels)
            name_to_shape[input_name] = {batch_size, image_seq_len, name_to_shape[input_name][2]};
        } else if (input_name == "encoder_hidden_states") {
            // (B, text_seq_len, joint_attention_dim)
            name_to_shape[input_name] = {batch_size, tokenizer_model_max_length, name_to_shape[input_name][2]};
        } else if (input_name == "encoder_hidden_states_mask") {
            // (B, text_seq_len)
            name_to_shape[input_name] = {batch_size, tokenizer_model_max_length};
        } else if (input_name == "img_cos" || input_name == "img_sin") {
            // (image_seq_len, rotary_dim)
            name_to_shape[input_name][0] = image_seq_len;
        } else if (input_name == "txt_cos" || input_name == "txt_sin") {
            // (text_seq_len, rotary_dim)
            name_to_shape[input_name][0] = tokenizer_model_max_length;
        } else if (input_name == "guidance") {
            name_to_shape[input_name] = {batch_size};
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}

QwenImageTransformer2DModel& QwenImageTransformer2DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("transformer"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "QwenImage Transformer 2D model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

void QwenImageTransformer2DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor tensor) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, tensor);
}

void QwenImageTransformer2DModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

ov::Tensor QwenImageTransformer2DModel::infer(const ov::Tensor latent_model_input, const ov::Tensor timestep) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", latent_model_input);
    m_request.set_tensor("timestep", timestep);
    m_request.infer();

    return m_request.get_output_tensor();
}

}  // namespace genai
}  // namespace ov
