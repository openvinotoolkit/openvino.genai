// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS AutoencoderKLLTX2Audio {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t latent_channels = 8;
        int64_t mel_bins = 64;
        int64_t sample_rate = 16000;
        int64_t mel_hop_length = 160;
        int64_t mel_compression_ratio = 4;
        int64_t temporal_compression_ratio = 4;
        std::vector<float> latents_mean_data;
        std::vector<float> latents_std_data;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit AutoencoderKLLTX2Audio(const std::filesystem::path& decoder_path);

    AutoencoderKLLTX2Audio(const std::filesystem::path& decoder_path,
                           const std::string& device,
                           const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    AutoencoderKLLTX2Audio(const std::filesystem::path& decoder_path,
                           const std::string& device,
                           Properties&&... properties)
        : AutoencoderKLLTX2Audio(decoder_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    AutoencoderKLLTX2Audio(const AutoencoderKLLTX2Audio&);

    AutoencoderKLLTX2Audio clone();

    const Config& get_config() const;

    AutoencoderKLLTX2Audio& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<AutoencoderKLLTX2Audio&, Properties...> compile(const std::string& device,
                                                                                   Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    AutoencoderKLLTX2Audio& reshape(int64_t batch_size, int64_t audio_num_frames);

    /// @brief Decodes audio latents [B, C, L, M] to a mel spectrogram
    ov::Tensor decode(const ov::Tensor& latent);

private:
    Config m_config;
    ov::InferRequest m_decoder_request;
    std::shared_ptr<ov::Model> m_decoder_model;
};

}  // namespace ov::genai
