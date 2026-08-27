// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "video_generation/models/autoencoder_kl_ltx2_video.hpp"

#include <cmath>
#include <fstream>
#include <numeric>

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"

#include "json_utils.hpp"
#include "utils.hpp"

using namespace ov::genai;

AutoencoderKLLTX2Video::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "latent_channels", latent_channels);
    read_json_param(data, "scaling_factor", scaling_factor);
    read_json_param(data, "timestep_conditioning", timestep_conditioning);
    read_json_param(data, "latents_mean_data", latents_mean_data);
    read_json_param(data, "latents_std_data", latents_std_data);
    read_json_param(data, "audio_latents_mean_data", audio_latents_mean_data);
    read_json_param(data, "audio_latents_std_data", audio_latents_std_data);

    int64_t spatial = 0, temporal = 0;
    read_json_param(data, "spatial_compression_ratio", spatial);
    read_json_param(data, "temporal_compression_ratio", temporal);
    if (spatial == 0 || temporal == 0) {
        std::vector<bool> spatio_temporal_scaling;
        int64_t patch_size, patch_size_t;
        read_json_param(data, "spatio_temporal_scaling", spatio_temporal_scaling);
        read_json_param(data, "patch_size", patch_size);
        read_json_param(data, "patch_size_t", patch_size_t);
        const auto compression_factor =
            std::pow(2, std::accumulate(spatio_temporal_scaling.begin(), spatio_temporal_scaling.end(), 0));
        spatial = patch_size * compression_factor;
        temporal = patch_size_t * compression_factor;
    }
    spatial_compression_ratio = spatial;
    temporal_compression_ratio = temporal;

    if (latents_mean_data.empty()) {
        latents_mean_data.assign(latent_channels, 0.0f);
    }
    if (latents_std_data.empty()) {
        latents_std_data.assign(latent_channels, 1.0f);
    }
}

AutoencoderKLLTX2Video::AutoencoderKLLTX2Video(const std::filesystem::path& vae_decoder_path)
    : m_config(vae_decoder_path / "config.json") {
    m_decoder_model = utils::singleton_core().read_model(vae_decoder_path / "openvino_model.xml");
    merge_vae_video_post_processing();
}

AutoencoderKLLTX2Video::AutoencoderKLLTX2Video(const std::filesystem::path& vae_decoder_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : AutoencoderKLLTX2Video(vae_decoder_path) {
    compile(device, properties);
}

AutoencoderKLLTX2Video::AutoencoderKLLTX2Video(const AutoencoderKLLTX2Video&) = default;

std::shared_ptr<AutoencoderKLLTX2Video> AutoencoderKLLTX2Video::clone() {
    OPENVINO_ASSERT((m_decoder_model != nullptr) ^ static_cast<bool>(m_decoder_request),
                    "AutoencoderKLLTX2Video must have exactly one of m_decoder_model or m_decoder_request initialized");

    std::shared_ptr<AutoencoderKLLTX2Video> cloned = std::make_shared<AutoencoderKLLTX2Video>(*this);

    if (m_decoder_model) {
        cloned->m_decoder_model = m_decoder_model->clone();
    } else {
        cloned->m_decoder_request = m_decoder_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const AutoencoderKLLTX2Video::Config& AutoencoderKLLTX2Video::get_config() const {
    return m_config;
}

AutoencoderKLLTX2Video& AutoencoderKLLTX2Video::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_decoder_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Auto encoder KL LTX2 video decoder model");
    m_decoder_request = compiled_model.create_infer_request();
    m_decoder_model.reset();

    return *this;
}

AutoencoderKLLTX2Video& AutoencoderKLLTX2Video::reshape(int64_t batch_size,
                                                        int64_t num_frames,
                                                        int64_t height,
                                                        int64_t width) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");

    const int64_t latent_num_frames = (num_frames - 1) / m_config.temporal_compression_ratio + 1;
    const int64_t latent_height = height / m_config.spatial_compression_ratio;
    const int64_t latent_width = width / m_config.spatial_compression_ratio;

    ov::PartialShape input_shape = m_decoder_model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{
        {0, {batch_size, input_shape[1], latent_num_frames, latent_height, latent_width}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

ov::Tensor AutoencoderKLLTX2Video::decode(const ov::Tensor& latent) {
    OPENVINO_ASSERT(m_decoder_request, "VAE decoder model must be compiled first. Cannot infer non-compiled model");

    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
}

void AutoencoderKLLTX2Video::merge_vae_video_post_processing() const {
    ov::preprocess::PrePostProcessor ppp(m_decoder_model);

    if (m_config.scaling_factor != 1.0f)
        ppp.input().preprocess().scale(m_config.scaling_factor);

    // (x / 2 + 0.5) -> clamp(0..1) -> *255 -> round() -> u8
    ppp.output().postprocess().custom([](const ov::Output<ov::Node>& port) {
        auto c_0_5 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 0.5f);
        auto c_255 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 255.0f);

        auto scaled = std::make_shared<ov::op::v1::Multiply>(port, c_0_5);
        auto shifted = std::make_shared<ov::op::v1::Add>(scaled, c_0_5);
        auto clamped = std::make_shared<ov::op::v0::Clamp>(shifted, 0.0f, 1.0f);

        auto scaled_255 = std::make_shared<ov::op::v1::Multiply>(clamped, c_255);
        return std::make_shared<ov::op::v5::Round>(scaled_255, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    });

    ppp.output().postprocess().convert_element_type(ov::element::u8);

    // [B, C, F, H, W] -> [B, F, H, W, C]
    ppp.output().model().set_layout("NCDHW");
    ppp.output().tensor().set_layout("NDHWC");

    ppp.build();
}
