// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

class LTX2Vocoder {
public:
    struct Config {
        int64_t output_sampling_rate = 24000;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit LTX2Vocoder(const std::filesystem::path& root_dir);

    LTX2Vocoder(const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties = {});

    LTX2Vocoder(const LTX2Vocoder&);

    std::shared_ptr<LTX2Vocoder> clone();

    const Config& get_config() const;

    LTX2Vocoder& compile(const std::string& device, const ov::AnyMap& properties = {});

    LTX2Vocoder& reshape(int64_t batch_size);

    ov::Tensor infer(const ov::Tensor& mel_spectrogram);

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
};

}  // namespace ov::genai
