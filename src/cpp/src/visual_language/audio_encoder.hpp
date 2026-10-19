// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief Encodes a mono 16 kHz f32 waveform into [1, tokens, hidden_size] f32 chunks.
using AudioEncode = std::function<std::vector<ov::Tensor>(const ov::Tensor&)>;

enum class AudioEncoderType {
    GEMMA4,          // log-mel spectrogram input, 30 s chunks
    GEMMA4_UNIFIED,  // raw 640-sample waveform frames
    LFM2,            // normalized log-mel features plus relative position embeddings
};

struct AudioEncoderConfig {
    AudioEncoderType type = AudioEncoderType::GEMMA4;
    size_t num_mel_bins = 0;
};

/// @brief Compiles `model` and wraps it with the feature extraction its input layout expects.
AudioEncode create_audio_encoder(const std::shared_ptr<ov::Model>& model,
                                 const AudioEncoderConfig& config,
                                 const std::string& device,
                                 const ov::AnyMap& properties);

}  // namespace ov::genai
