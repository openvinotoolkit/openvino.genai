// Copyright (C) 2023-2025 Intel Corporation
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
        size_t latent_channels = 4;
        size_t out_channels = 3;
        float shift_factor = 0.0f;
        float scaling_factor = 1.0f;
        std::vector<size_t> block_out_channels = { 64 };
        
        size_t patch_size = 4;  // TODO: read from vae_decoder/config.json
        std::vector<bool> spatio_temporal_scaling{true, true, true, false};  // TODO: read from vae_decoder/config.json. I use it only to compute sum over it so far, so it may be removed
        size_t patch_size_t = 1;  // TODO: read from vae_decoder/config.json

        // latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        // latents_std = torch.ones((latent_channels,), requires_grad=False)
        std::vector<float> latents_mean_data; // TODO: set default value
        std::vector<float> latents_std_data;  // TODO: set default value

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

    ov::Tensor decode(ov::Tensor latent);

    const Config& get_config() const;

    size_t get_vae_scale_factor() const;

    void get_compression_ratio(const std::filesystem::path& config_path, int64_t& spatial_compression_ratio, int64_t& temporal_compression_ratio);

    AutoencoderKLLTXVideo& reshape(int64_t batch_size, int64_t num_frames, int64_t height, int64_t width);

private:
    void merge_vae_image_post_processing() const;

    Config m_config;
    ov::InferRequest m_encoder_request, m_decoder_request;
    std::shared_ptr<ov::Model> m_encoder_model = nullptr, m_decoder_model = nullptr;

    int64_t m_patch_size, m_patch_size_t;
};

} // namespace ov::genai
