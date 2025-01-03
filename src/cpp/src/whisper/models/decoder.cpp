// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <filesystem>

#include "statefull_decoder.hpp"
#include "utils.hpp"
#include "with_past_decoder.hpp"

namespace ov::genai {
std::shared_ptr<WhisperDecoder> WhisperDecoder::from_path(const std::filesystem::path& models_path,
                                                          const std::string& device,
                                                          const ov::AnyMap& properties) {
    bool has_decoder_with_past = std::filesystem::exists(models_path / "openvino_decoder_with_past_model.xml");

    if (has_decoder_with_past) {
        // todo: add deprecation notice
        return std::make_shared<WhisperWithPastDecoder>(models_path, device, properties);
    }

    return std::make_shared<WhisperStatefullDecoder>(models_path, device, properties);
}

WhisperDecoder::~WhisperDecoder() = default;
}  // namespace ov::genai
