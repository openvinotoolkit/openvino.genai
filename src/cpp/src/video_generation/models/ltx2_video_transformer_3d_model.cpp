// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "video_generation/models/ltx2_video_transformer_3d_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

#include "json_utils.hpp"
#include "utils.hpp"
#include "video_generation/video_generation_utils.hpp"

using namespace ov::genai;

namespace {

std::pair<int64_t, int64_t> get_vae_compression_ratio(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);
    nlohmann::json data = nlohmann::json::parse(file);

    int64_t spatial_compression_ratio = 0, temporal_compression_ratio = 0;
    utils::read_json_param(data, "spatial_compression_ratio", spatial_compression_ratio);
    utils::read_json_param(data, "temporal_compression_ratio", temporal_compression_ratio);

    if (spatial_compression_ratio == 0 || temporal_compression_ratio == 0) {
        std::vector<bool> spatio_temporal_scaling;
        int64_t patch_size, patch_size_t;
        utils::read_json_param(data, "spatio_temporal_scaling", spatio_temporal_scaling);
        utils::read_json_param(data, "patch_size", patch_size);
        utils::read_json_param(data, "patch_size_t", patch_size_t);
        const auto compression_factor =
            std::pow(2, std::accumulate(spatio_temporal_scaling.begin(), spatio_temporal_scaling.end(), 0));
        spatial_compression_ratio = patch_size * compression_factor;
        temporal_compression_ratio = patch_size_t * compression_factor;
    }

    return {spatial_compression_ratio, temporal_compression_ratio};
}

bool has_exact_input(const ov::CompiledModel& compiled_model, const std::string& name) {
    for (const auto& input : compiled_model.inputs()) {
        if (input.get_names().count(name)) {
            return true;
        }
    }
    return false;
}

}  // namespace

LTX2VideoTransformer3DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "audio_in_channels", audio_in_channels);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "patch_size_t", patch_size_t);
    read_json_param(data, "vae_scale_factors", vae_scale_factors);
    read_json_param(data, "audio_scale_factor", audio_scale_factor);
    read_json_param(data, "causal_offset", causal_offset);
    read_json_param(data, "audio_sampling_rate", audio_sampling_rate);
    read_json_param(data, "audio_hop_length", audio_hop_length);
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    std::tie(m_spatial_compression_ratio, m_temporal_compression_ratio) =
        get_vae_compression_ratio(root_dir.parent_path() / "vae_decoder" / "config.json");
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir,
                                                         const std::string& device,
                                                         const ov::AnyMap& properties)
    : LTX2VideoTransformer3DModel(root_dir) {
    compile(device, properties);
}

LTX2VideoTransformer3DModel::LTX2VideoTransformer3DModel(const LTX2VideoTransformer3DModel&) = default;

std::shared_ptr<LTX2VideoTransformer3DModel> LTX2VideoTransformer3DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "LTX2VideoTransformer3DModel must have exactly one of m_model or m_request initialized");

    std::shared_ptr<LTX2VideoTransformer3DModel> cloned = std::make_shared<LTX2VideoTransformer3DModel>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const LTX2VideoTransformer3DModel::Config& LTX2VideoTransformer3DModel::get_config() const {
    return m_config;
}

LTX2VideoTransformer3DModel& LTX2VideoTransformer3DModel::compile(const std::string& device,
                                                                  const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "LTX2 Video Transformer 3D model");
    m_request = compiled_model.create_infer_request();
    const auto& input_shape = compiled_model.input("hidden_states").get_partial_shape();
    m_expected_batch_size = input_shape[0].is_static() ? input_shape[0].get_length() : 0;
    m_model.reset();

    return *this;
}

void LTX2VideoTransformer3DModel::set_hidden_states(const std::string& tensor_name, const ov::Tensor& tensor) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    const auto input_type = m_request.get_compiled_model().input(tensor_name).get_element_type();
    m_request.set_tensor(tensor_name, video_generation_utils::convert_tensor(tensor, input_type));
}

size_t LTX2VideoTransformer3DModel::get_expected_batch_size() const {
    return m_expected_batch_size;
}

size_t LTX2VideoTransformer3DModel::get_request_input_batch() {
    if (!m_request) {
        return 0;
    }
    const ov::Shape shape = m_request.get_tensor("hidden_states").get_shape();
    if (shape.empty()) {
        return 0;
    }
    return shape[0];
}

size_t LTX2VideoTransformer3DModel::get_timestep_rank() {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot query non-compiled model");
    return m_request.get_compiled_model().input("timestep").get_partial_shape().rank().get_length();
}

std::pair<ov::Tensor, ov::Tensor> LTX2VideoTransformer3DModel::infer(const ov::Tensor& video_latent,
                                                                     const ov::Tensor& audio_latent,
                                                                     float timestep) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first. Cannot infer non-compiled model");

    m_request.set_tensor("hidden_states", video_latent);
    m_request.set_tensor("audio_hidden_states", audio_latent);

    const ov::Shape& latent_shape = video_latent.get_shape();
    OPENVINO_ASSERT(latent_shape.size() == 3, "Packed latents must be rank-3 [B, S, C], got rank ", latent_shape.size());

    // Legacy exports take a rank-1 [B] timestep, current ones a rank-2 [B, S] per-token timestep
    const auto timestep_rank = get_timestep_rank();
    OPENVINO_ASSERT(timestep_rank == 1 || timestep_rank == 2,
                    "LTX2 transformer expects a rank-1 or rank-2 'timestep' input, got rank ", timestep_rank);
    ov::Shape timestep_shape{latent_shape[0]};
    if (timestep_rank == 2) {
        timestep_shape = {latent_shape[0], latent_shape[1]};
    }
    ov::Tensor timestep_tensor(ov::element::f32, timestep_shape);
    std::fill_n(timestep_tensor.data<float>(), timestep_tensor.get_size(), timestep);
    m_request.set_tensor("timestep", timestep_tensor);

    if (has_exact_input(m_request.get_compiled_model(), "audio_timestep")) {
        ov::Tensor audio_timestep(ov::element::f32, {audio_latent.get_shape()[0]});
        std::fill_n(audio_timestep.data<float>(), audio_timestep.get_size(), timestep);
        m_request.set_tensor("audio_timestep", audio_timestep);
    }

    m_request.infer();

    return {m_request.get_tensor("out_sample"), m_request.get_tensor("audio_out_sample")};
}

LTX2VideoTransformer3DModel& LTX2VideoTransformer3DModel::reshape(int64_t batch_size,
                                                                  int64_t num_frames,
                                                                  int64_t height,
                                                                  int64_t width,
                                                                  int64_t audio_num_frames) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    const int64_t patch_size = m_config.patch_size;
    const int64_t patch_size_t = m_config.patch_size_t;

    const int64_t latent_num_frames = ((num_frames - 1) / m_temporal_compression_ratio + 1) / patch_size_t;
    const int64_t latent_height = height / (m_spatial_compression_ratio * patch_size);
    const int64_t latent_width = width / (m_spatial_compression_ratio * patch_size);
    const int64_t video_sequence_length = latent_num_frames * latent_height * latent_width;

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "hidden_states") {
            name_to_shape[input_name] = {batch_size, video_sequence_length, name_to_shape[input_name][2]};
        } else if (input_name == "audio_hidden_states") {
            name_to_shape[input_name] = {batch_size, audio_num_frames, name_to_shape[input_name][2]};
        } else if (input_name == "timestep") {
            const auto timestep_rank = name_to_shape[input_name].rank().get_length();
            OPENVINO_ASSERT(timestep_rank == 1 || timestep_rank == 2,
                            "LTX2 transformer expects a rank-1 or rank-2 'timestep' input, got rank ", timestep_rank);
            if (timestep_rank == 2) {
                name_to_shape[input_name] = {batch_size, video_sequence_length};
            } else {
                name_to_shape[input_name] = {batch_size};
            }
        } else if (input_name == "audio_timestep") {
            name_to_shape[input_name] = {batch_size};
        } else if (input_name == "video_coords") {
            name_to_shape[input_name] = {batch_size, 3, video_sequence_length, 2};
        } else if (input_name == "audio_coords") {
            name_to_shape[input_name] = {batch_size, 1, audio_num_frames, 2};
        } else if (input_name == "encoder_hidden_states" || input_name == "audio_encoder_hidden_states") {
            // The connector output sequence length is model-internal, keep it dynamic
            name_to_shape[input_name] = {batch_size, -1, name_to_shape[input_name][2]};
        } else if (input_name == "encoder_attention_mask" || input_name == "audio_encoder_attention_mask") {
            name_to_shape[input_name] = {batch_size, -1};
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}
