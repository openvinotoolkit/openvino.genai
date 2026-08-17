// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

namespace utils {
namespace audio {
ov::genai::RawSpeechInput read_wav(const std::string& filename);

/// @brief Read a WAV file into a 1-D f32 ov::Tensor without copying the decoded samples.
/// The PCM buffer produced by read_wav() is moved into the tensor's allocator, so the tensor
/// owns the samples directly.
ov::Tensor read_wav_as_tensor(const std::string& filename);

/// @brief Save a mono f32 waveform to a WAV file.
/// @param waveform_ptr Pointer to the float samples.
/// @param waveform_size Number of samples.
/// @param file_path Destination WAV path.
/// @param bits_per_sample Bit depth to store each sample with.
/// @param sample_rate Sample rate in Hz.
void save_to_wav(const float* waveform_ptr,
                 size_t waveform_size,
                 const std::filesystem::path& file_path,
                 uint32_t bits_per_sample,
                 uint32_t sample_rate);
}  // namespace audio
}  // namespace utils
