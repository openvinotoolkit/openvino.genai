// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/autoencoder_kl_ltx2_audio.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"

using namespace ov::genai;

AutoencoderKLLTX2Audio::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "latent_channels", latent_channels);
    read_json_param(data, "mel_bins", mel_bins);
    read_json_param(data, "sample_rate", sample_rate);
    read_json_param(data, "mel_hop_length", mel_hop_length);
    read_json_param(data, "mel_compression_ratio", mel_compression_ratio);
    read_json_param(data, "temporal_compression_ratio", temporal_compression_ratio);
    read_json_param(data, "latents_mean_data", latents_mean_data);
    read_json_param(data, "latents_std_data", latents_std_data);
}

AutoencoderKLLTX2Audio::AutoencoderKLLTX2Audio(const std::filesystem::path& decoder_path)
    : m_config(decoder_path / "config.json") {
    m_decoder_model = utils::singleton_core().read_model(decoder_path / "openvino_model.xml");
}

AutoencoderKLLTX2Audio::AutoencoderKLLTX2Audio(const std::filesystem::path& decoder_path,
                                               const std::string& device,
                                               const ov::AnyMap& properties)
    : AutoencoderKLLTX2Audio(decoder_path) {
    compile(device, properties);
}

AutoencoderKLLTX2Audio::AutoencoderKLLTX2Audio(const AutoencoderKLLTX2Audio&) = default;

AutoencoderKLLTX2Audio AutoencoderKLLTX2Audio::clone() {
    OPENVINO_ASSERT((m_decoder_model != nullptr) ^ static_cast<bool>(m_decoder_request),
                    "AutoencoderKLLTX2Audio must have exactly one of m_decoder_model or m_decoder_request initialized");

    AutoencoderKLLTX2Audio cloned = *this;

    if (m_decoder_model) {
        cloned.m_decoder_model = m_decoder_model->clone();
    } else {
        cloned.m_decoder_request = m_decoder_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const AutoencoderKLLTX2Audio::Config& AutoencoderKLLTX2Audio::get_config() const {
    return m_config;
}

AutoencoderKLLTX2Audio& AutoencoderKLLTX2Audio::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_decoder_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Auto encoder KL LTX2 audio decoder model");
    m_decoder_request = compiled_model.create_infer_request();
    m_decoder_model.reset();

    return *this;
}

AutoencoderKLLTX2Audio& AutoencoderKLLTX2Audio::reshape(int64_t batch_size, int64_t audio_num_frames) {
    OPENVINO_ASSERT(m_decoder_model, "Model has been already compiled. Cannot reshape already compiled model");

    const int64_t latent_mel_bins = m_config.mel_bins / m_config.mel_compression_ratio;
    std::map<size_t, ov::PartialShape> idx_to_shape{
        {0, {batch_size, static_cast<int64_t>(m_config.latent_channels), audio_num_frames, latent_mel_bins}}};
    m_decoder_model->reshape(idx_to_shape);

    return *this;
}

ov::Tensor AutoencoderKLLTX2Audio::decode(const ov::Tensor& latent) {
    OPENVINO_ASSERT(m_decoder_request, "Audio VAE decoder model must be compiled first. Cannot infer non-compiled model");

    m_decoder_request.set_input_tensor(latent);
    m_decoder_request.infer();
    return m_decoder_request.get_output_tensor();
}
