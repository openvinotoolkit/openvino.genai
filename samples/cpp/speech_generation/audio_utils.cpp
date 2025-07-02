// Copyright (C) 2023-2025 Intel Corporation
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
                 uint32_t bits_per_sample) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = 16000;  // assume it is always 16 KHz
    format.bitsPerSample = bits_per_sample;

    drwav wav;
    OPENVINO_ASSERT(drwav_init_file_write(&wav, file_path.string().c_str(), &format, nullptr),
                    "Failed to initialize WAV writer");

    size_t total_samples = waveform_size * format.channels;

    drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, total_samples, waveform_ptr);
    OPENVINO_ASSERT(frames_written == total_samples, "Failed to write not all frames");

    drwav_uninit(&wav);
}

ov::Tensor read_speaker_embedding(const std::filesystem::path& file_path) {
    std::ifstream input(file_path, std::ios::binary);
    OPENVINO_ASSERT(input, "Failed to open file: " + file_path.string());

    // Get file size
    input.seekg(0, std::ios::end);
    size_t buffer_size = static_cast<size_t>(input.tellg());
    input.seekg(0, std::ios::beg);

    // Check size is multiple of float
    OPENVINO_ASSERT(buffer_size % sizeof(float) == 0, "File size is not a multiple of float size.");
    size_t num_floats = buffer_size / sizeof(float);
    OPENVINO_ASSERT(num_floats == 512, "File must contain speaker embedding including 512 32-bit floats.");

    OPENVINO_ASSERT(input, "Failed to read all data from file.");
    ov::Tensor floats_tensor(ov::element::f32, ov::Shape{1, num_floats});
    input.read(reinterpret_cast<char*>(floats_tensor.data()), buffer_size);

    return floats_tensor;
}

}  // namespace audio
}  // namespace utils
