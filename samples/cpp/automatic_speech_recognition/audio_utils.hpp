// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

namespace utils {
namespace audio {
ov::genai::RawSpeechInput read_wav(const std::string& filename);

/// @brief Read a WAV file into a 1-D f32 ov::Tensor without copying the decoded samples.
/// The PCM buffer produced by read_wav() is moved into the tensor's allocator, so the tensor
/// owns the samples directly.
ov::Tensor read_wav_as_tensor(const std::string& filename);
}  // namespace audio
}  // namespace utils
