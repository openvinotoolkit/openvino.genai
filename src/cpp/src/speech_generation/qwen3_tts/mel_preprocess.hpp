// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>

#include "openvino/openvino.hpp"

namespace ov {
namespace genai {

// Builds an OpenVINO model computing log-mel spectrogram features from a raw waveform,
// matching the Qwen3-TTS speaker encoder's expected input. Assumes a 24000 Hz waveform.
std::shared_ptr<ov::Model> build_qwen3_mel_preprocess_model(size_t mel_dim);

}  // namespace genai
}  // namespace ov
