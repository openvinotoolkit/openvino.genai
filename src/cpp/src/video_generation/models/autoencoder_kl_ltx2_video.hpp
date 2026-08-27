// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

class AutoencoderKLLTX2Video {
public:
    struct Config {
        size_t latent_channels = 128;
        float scaling_factor = 1.0f;
        int64_t spatial_compression_ratio = 32;
        int64_t temporal_compression_ratio = 8;
        bool timestep_conditioning = false;
        std::vector<float> latents_mean_data;
        std::vector<float> latents_std_data;
        std::vector<float> audio_latents_mean_data;
        std::vector<float> audio_latents_std_data;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit AutoencoderKLLTX2Video(const std::filesystem::path& vae_decoder_path);

    AutoencoderKLLTX2Video(const std::filesystem::path& vae_decoder_path,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    AutoencoderKLLTX2Video(const AutoencoderKLLTX2Video&);

    std::shared_ptr<AutoencoderKLLTX2Video> clone();

    const Config& get_config() const;

    AutoencoderKLLTX2Video& compile(const std::string& device, const ov::AnyMap& properties = {});

    AutoencoderKLLTX2Video& reshape(int64_t batch_size, int64_t num_frames, int64_t height, int64_t width);

    ov::Tensor decode(const ov::Tensor& latent);

private:
    void merge_vae_video_post_processing() const;

    Config m_config;
    ov::InferRequest m_decoder_request;
    std::shared_ptr<ov::Model> m_decoder_model;
};

}  // namespace ov::genai
