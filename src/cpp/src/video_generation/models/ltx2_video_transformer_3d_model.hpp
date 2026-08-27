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

class LTX2VideoTransformer3DModel {
public:
    struct Config {
        size_t in_channels = 128;
        size_t audio_in_channels = 128;
        size_t patch_size = 1;
        size_t patch_size_t = 1;
        std::vector<int64_t> vae_scale_factors = {8, 32, 32};
        int64_t audio_scale_factor = 4;
        int64_t causal_offset = 1;
        int64_t audio_sampling_rate = 16000;
        int64_t audio_hop_length = 160;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir);

    LTX2VideoTransformer3DModel(const std::filesystem::path& root_dir,
                                const std::string& device,
                                const ov::AnyMap& properties = {});

    LTX2VideoTransformer3DModel(const LTX2VideoTransformer3DModel&);

    std::shared_ptr<LTX2VideoTransformer3DModel> clone();

    const Config& get_config() const;

    LTX2VideoTransformer3DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    void set_hidden_states(const std::string& tensor_name, const ov::Tensor& tensor);

    /// Builds the 'timestep' input matching the compiled model rank (rank-1 [B] or rank-2 [B, S])
    /// and runs joint video + audio denoising.
    std::pair<ov::Tensor, ov::Tensor> infer(const ov::Tensor& video_latent, const ov::Tensor& audio_latent, float timestep);

    LTX2VideoTransformer3DModel& reshape(int64_t batch_size,
                                         int64_t num_frames,
                                         int64_t height,
                                         int64_t width,
                                         int64_t audio_num_frames);

    size_t get_expected_batch_size() const;
    size_t get_request_input_batch();
    size_t get_timestep_rank();

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    size_t m_expected_batch_size = 0;
    int64_t m_spatial_compression_ratio, m_temporal_compression_ratio;
};

}  // namespace ov::genai
