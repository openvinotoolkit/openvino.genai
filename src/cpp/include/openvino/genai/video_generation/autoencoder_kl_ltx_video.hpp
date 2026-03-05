// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS AutoencoderKLLTXVideo {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 3;
        size_t latent_channels = 128;
        size_t out_channels = 3;
        float scaling_factor = 1.0f;
        std::vector<size_t> block_out_channels = {128, 256, 512, 512};
        size_t patch_size = 4;
        std::vector<bool> spatio_temporal_scaling{true, true, true, false};
        size_t patch_size_t = 1;
        std::vector<float> latents_mean_data, latents_std_data;
        bool timestep_conditioning = false;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit AutoencoderKLLTXVideo(const std::filesystem::path& vae_decoder_path);

    AutoencoderKLLTXVideo(const std::filesystem::path& vae_encoder_path,
                  const std::filesystem::path& vae_decoder_path);

    AutoencoderKLLTXVideo(const std::filesystem::path& vae_decoder_path,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    AutoencoderKLLTXVideo(const std::filesystem::path& vae_encoder_path,
                  const std::filesystem::path& vae_decoder_path,
                  const std::string& device,
                  const ov::AnyMap& properties = {});

    AutoencoderKLLTXVideo& compile(const std::string& device, const ov::AnyMap& properties = {});

    ov::Tensor decode(const ov::Tensor& latent);

    const Config& get_config() const;

    size_t get_vae_scale_factor() const;

    AutoencoderKLLTXVideo& reshape(int64_t batch_size, int64_t num_frames, int64_t height, int64_t width);

private:
    void merge_vae_video_post_processing() const;

    Config m_config;
    ov::InferRequest m_encoder_request, m_decoder_request;
    std::shared_ptr<ov::Model> m_encoder_model = nullptr, m_decoder_model = nullptr;

    int64_t m_transformer_patch_size = -1, m_transformer_patch_size_t = -1;
};

} // namespace ov::genai
