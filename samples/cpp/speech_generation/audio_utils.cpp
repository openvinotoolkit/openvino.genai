// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.hpp"

#include <fstream>
#include <iostream>
#include <vector>

#include "openvino/core/except.hpp"

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

namespace utils {
namespace audio {

void save_to_wav(const float* waveform_ptr,
                 size_t waveform_size,
                 const std::filesystem::path& file_path,
                 uint32_t bits_per_sample,
                 uint32_t sample_rate) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = sample_rate;
    format.bitsPerSample = bits_per_sample;

    drwav wav;
    OPENVINO_ASSERT(drwav_init_file_write(&wav, file_path.string().c_str(), &format, nullptr),
                    "Failed to initialize WAV writer");

    size_t total_samples = waveform_size * format.channels;

    drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, total_samples, waveform_ptr);
    OPENVINO_ASSERT(frames_written == total_samples, "Failed to write not all frames");

    drwav_uninit(&wav);
}

ov::Tensor read_speaker_embedding(const std::filesystem::path& file_path, const ov::Shape& shape) {
    std::ifstream input(file_path, std::ios::binary);
    OPENVINO_ASSERT(input, "Failed to open speaker embedding file: " + file_path.string());

    input.seekg(0, std::ios::end);
    const size_t buffer_size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    OPENVINO_ASSERT(buffer_size % sizeof(float) == 0,
                    "Speaker embedding file size is not a multiple of sizeof(float).");
    const size_t num_floats = buffer_size / sizeof(float);

    // Validate that file size matches expected shape.
    size_t expected_floats = 1;
    for (size_t d : shape) expected_floats *= d;
    OPENVINO_ASSERT(num_floats == expected_floats,
                    "Speaker embedding file contains ", num_floats, " float32 values "
                    "but expected shape ", shape, " (total ", expected_floats, " floats).");

    ov::Tensor tensor(ov::element::f32, shape);
    input.read(reinterpret_cast<char*>(tensor.data()), buffer_size);
    OPENVINO_ASSERT(input, "Failed to read all data from speaker embedding file.");
    return tensor;
}

}  // namespace audio
}  // namespace utils
