// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/ltx_video_transformer_3d_model.hpp"

#include <fstream>
#include <numeric>

#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

using namespace ov::genai;

namespace {

std::pair<int64_t, int64_t> get_compression_ratio(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);
    nlohmann::json data = nlohmann::json::parse(file);

    std::vector<bool> spatio_temporal_scaling;
    int64_t patch_size, patch_size_t;

    utils::read_json_param(data, "spatio_temporal_scaling", spatio_temporal_scaling);
    utils::read_json_param(data, "patch_size", patch_size);
    utils::read_json_param(data, "patch_size_t", patch_size_t);

    const auto compression_factor = std::pow(2, std::accumulate(spatio_temporal_scaling.begin(), spatio_temporal_scaling.end(), 0));
    const int64_t spatial_compression_ratio = patch_size * compression_factor;
    const int64_t temporal_compression_ratio = patch_size_t * compression_factor;

    return {spatial_compression_ratio, temporal_compression_ratio};
}

} // namespace

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
    std::tie(m_spatial_compression_ratio, m_temporal_compression_ratio) = get_compression_ratio(root_dir.parent_path() / "vae_decoder" / "config.json");
}

LTXVideoTransformer3DModel::LTXVideoTransformer3DModel(const std::filesystem::path& root_dir,
                                             const std::string& device,
                                             const ov::AnyMap& properties)
    : LTXVideoTransformer3DModel(root_dir) {
    compile(device, properties);
}

LTXVideoTransformer3DModel::LTXVideoTransformer3DModel(const LTXVideoTransformer3DModel&) = default;

const LTXVideoTransformer3DModel::Config& LTXVideoTransformer3DModel::get_config() const {
    return m_config;
}

LTXVideoTransformer3DModel& LTXVideoTransformer3DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("transformer"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX Video Transformer 3D model");
    m_request = compiled_model.create_infer_request();
    const auto& input_shape = compiled_model.input(0).get_partial_shape();
    m_expected_batch_size = input_shape.is_static() ? input_shape[0].get_length() : 0;
    // release the original model
    m_model.reset();

    return *this;
}

void LTXVideoTransformer3DModel::set_hidden_states(const std::string& tensor_name, const ov::Tensor& encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, encoder_hidden_states);
}

void LTXVideoTransformer3DModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

ov::Tensor LTXVideoTransformer3DModel::infer(const ov::Tensor& latent_model_input, const ov::Tensor& timestep) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", latent_model_input);
    m_request.set_tensor("timestep", timestep);
    m_request.infer();

    return m_request.get_output_tensor();
}

size_t LTXVideoTransformer3DModel::get_expected_batch_size() const {
    return m_expected_batch_size;
}

size_t LTXVideoTransformer3DModel::get_request_input_batch() {
    if (!m_request) {
        return 0;
    }
    const ov::Shape shape = m_request.get_input_tensor(0).get_shape();
    if (shape.empty()) {
        return 0;
    }
    return shape[0];
}

LTXVideoTransformer3DModel& LTXVideoTransformer3DModel::reshape(int64_t batch_size,
                                                            int64_t num_frames,
                                                            int64_t height,
                                                            int64_t width,
                                                            int64_t tokenizer_model_max_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    // hidden_states=latent_model_input,
    // timestep=timestep,
    // encoder_hidden_states=prompt_embeds,
    // pooled_projections=pooled_prompt_embeds,

    size_t patch_size = get_config().patch_size;
    size_t patch_size_t = get_config().patch_size_t;

    num_frames = ((num_frames - 1) / m_temporal_compression_ratio + 1) / patch_size_t;
    height /=  (m_spatial_compression_ratio * patch_size);
    width /=  (m_spatial_compression_ratio * patch_size);

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "timestep") {
            name_to_shape[input_name][0] = 1;
        } else if (input_name == "encoder_hidden_states") {
            name_to_shape[input_name] = {batch_size, tokenizer_model_max_length, name_to_shape[input_name][2]};
        } else if (input_name == "hidden_states") {
            name_to_shape[input_name] = {batch_size, num_frames * height * width, name_to_shape[input_name][2]};
        } else if (input_name == "encoder_attention_mask") {
            name_to_shape[input_name] = {batch_size, tokenizer_model_max_length};
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}
