// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

namespace utils {
namespace audio {

inline constexpr uint32_t COMMON_SAMPLE_RATE = 16000;

/**
 * Reads a mono or stereo WAV file and returns it as mono float samples.
 *
 * @param filename Path to the WAV file, a WAV buffer, or "-" to read from stdin
 * @param expected_sample_rate Expected sample rate of the WAV file in Hz.
 *                             Optional; defaults to COMMON_SAMPLE_RATE (16000).
 */
ov::genai::RawSpeechInput read_wav(const std::string& filename,
                                   uint32_t expected_sample_rate = COMMON_SAMPLE_RATE);

/**
 * Reads a WAV file into a 1-D f32 ov::Tensor without copying the decoded samples.
 *
 * The PCM buffer produced by read_wav() is moved into the tensor's allocator, so the tensor
 * owns the samples directly.
 *
 * @param filename Path to the WAV file
 * @param expected_sample_rate Expected sample rate of the WAV file in Hz.
 *                             Optional; defaults to COMMON_SAMPLE_RATE (16000).
 */
ov::Tensor read_wav_as_tensor(const std::string& filename,
                              uint32_t expected_sample_rate = COMMON_SAMPLE_RATE);

/**
 * This function saves an audio waveform, provided as an array of floating-point samples, to a WAV file.
 *
 * @param waveform_ptr Pointer to the array of float samples representing the audio waveform
 * @param waveform_size The number of samples in the waveform array
 * @param file_path The name (and path) of the WAV file to be created
 * @param bits_per_sample The bit depth used to store each sample in the WAV file
 */
void save_to_wav(const float* waveform_ptr,
                 size_t waveform_size,
                 const std::filesystem::path& file_path,
                 uint32_t bits_per_sample,
                 uint32_t sample_rate = COMMON_SAMPLE_RATE);

/**
 * Reads a binary file of float32 values and returns an ov::Tensor with the given shape.
 *
 * @param file_path  Path to the binary file.
 * @param shape      Expected tensor shape, as returned by
 *                   Text2SpeechPipeline::get_speaker_embedding_shape().
 * @returns ov::Tensor{f32, shape} with data loaded from the file.
 */
ov::Tensor read_speaker_embedding(const std::filesystem::path& file_path,
                                  const ov::Shape& shape);

}  // namespace audio
}  // namespace utils
