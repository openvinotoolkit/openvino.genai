// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS LTX2Vocoder {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        int64_t output_sampling_rate = 24000;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit LTX2Vocoder(const std::filesystem::path& root_dir);

    LTX2Vocoder(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    LTX2Vocoder(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : LTX2Vocoder(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    LTX2Vocoder(const LTX2Vocoder&);

    LTX2Vocoder clone();

    const Config& get_config() const;

    LTX2Vocoder& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<LTX2Vocoder&, Properties...> compile(const std::string& device,
                                                                        Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    LTX2Vocoder& reshape(int64_t batch_size);

    /// @brief Converts a mel spectrogram to an audio waveform [B, num_channels, num_samples]
    ov::Tensor infer(const ov::Tensor& mel_spectrogram);

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

}  // namespace ov::genai
