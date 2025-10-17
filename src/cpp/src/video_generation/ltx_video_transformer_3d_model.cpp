// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx_video_transformer_3d_model.hpp"
// #include "image_generation/models/sd3transformer_2d_inference_dynamic.hpp"
// #include "image_generation/models/sd3transformer_2d_inference_static_bs1.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

namespace ov {
namespace genai {

LTXVideoTransformer3DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "patch_size_t", patch_size_t);
}

LTXVideoTransformer3DModel::LTXVideoTransformer3DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

LTXVideoTransformer3DModel::LTXVideoTransformer3DModel(const std::filesystem::path& root_dir,
                                             const std::string& device,
                                             const ov::AnyMap& properties)
    : LTXVideoTransformer3DModel(root_dir) {
    compile(device, properties);
}

// LTXVideoTransformer3DModel::LTXVideoTransformer3DModel(const std::string& model,
//                                              const Tensor& weights,
//                                              const Config& config,
//                                              const size_t vae_scale_factor) :
//     m_config(config), m_vae_scale_factor(vae_scale_factor) {
//     m_model = utils::singleton_core().read_model(model, weights);
// }

// LTXVideoTransformer3DModel::LTXVideoTransformer3DModel(const std::string& model,
//                                              const Tensor& weights,
//                                              const Config& config,
//                                              const size_t vae_scale_factor,
//                                              const std::string& device,
//                                              const ov::AnyMap& properties) :
//     LTXVideoTransformer3DModel(model, weights, config, vae_scale_factor) {
//     compile(device, properties);
// }

LTXVideoTransformer3DModel::LTXVideoTransformer3DModel(const LTXVideoTransformer3DModel&) = default;

// LTXVideoTransformer3DModel LTXVideoTransformer3DModel::clone() {
//     OPENVINO_ASSERT((m_model != nullptr) ^ (m_impl != nullptr), "LTXVideoTransformer3DModel must have exactly one of m_model or m_impl initialized");

//     LTXVideoTransformer3DModel cloned = *this;
    
//     if (m_model) {
//         cloned.m_model = m_model->clone();
//     } else {
//         cloned.m_impl = m_impl->clone();
//     }

//     return cloned;
// }

const LTXVideoTransformer3DModel::Config& LTXVideoTransformer3DModel::get_config() const {
    return m_config;
}

// LTXVideoTransformer3DModel& LTXVideoTransformer3DModel::reshape(int batch_size,
//                                                       int height,
//                                                       int width,
//                                                       int tokenizer_model_max_length) {
//     OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

//     // hidden_states=latent_model_input,
//     // timestep=timestep,
//     // encoder_hidden_states=prompt_embeds,
//     // pooled_projections=pooled_prompt_embeds,

//     height /= m_vae_scale_factor;
//     width /= m_vae_scale_factor;

//     LTXVideoTransformer3DModel::Inference::reshape(m_model, batch_size, height, width, tokenizer_model_max_length);

//     return *this;
// }

LTXVideoTransformer3DModel& LTXVideoTransformer3DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    OPENVINO_ASSERT(!adapters, "Adapters are not currently supported for Video Generation Pipeline.");

    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Flux Transformer 2D model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

void LTXVideoTransformer3DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, encoder_hidden_states);
}

ov::Tensor LTXVideoTransformer3DModel::infer(const ov::Tensor latent_model_input, const ov::Tensor timestep) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", latent_model_input);
    m_request.set_tensor("timestep", timestep);
    m_request.infer();

    return m_request.get_output_tensor();
}


}  // namespace genai
}  // namespace ov
