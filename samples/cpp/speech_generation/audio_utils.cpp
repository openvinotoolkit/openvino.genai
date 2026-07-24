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

ov::Tensor read_wav_mono_f32(const std::filesystem::path& file_path, uint32_t expected_sample_rate) {
    drwav wav;
    OPENVINO_ASSERT(drwav_init_file(&wav, file_path.string().c_str(), nullptr),
                    "Failed to open WAV file: ",
                    file_path.string());

    OPENVINO_ASSERT(wav.channels == 1 || wav.channels == 2,
                    "WAV file must be mono or stereo: ",
                    file_path.string());
    OPENVINO_ASSERT(wav.sampleRate == expected_sample_rate,
                    "WAV file sample rate must be ",
                    expected_sample_rate,
                    " Hz. Got ",
                    wav.sampleRate,
                    " Hz for ",
                    file_path.string(),
                    ". OV GenAI does not resample reference audio.");

    const uint64_t num_frames = wav.totalPCMFrameCount;
    const uint32_t channels = wav.channels;
    std::vector<float> interleaved(num_frames * wav.channels, 0.0f);
    const uint64_t frames_read = drwav_read_pcm_frames_f32(&wav, num_frames, interleaved.data());
    drwav_uninit(&wav);

    OPENVINO_ASSERT(frames_read == num_frames,
                    "Failed to read full WAV payload from ",
                    file_path.string());

    ov::Tensor waveform(ov::element::f32, ov::Shape{static_cast<size_t>(num_frames)});
    float* out = waveform.data<float>();
    if (channels == 1) {
        std::copy_n(interleaved.data(), static_cast<size_t>(num_frames), out);
    } else {
        for (uint64_t i = 0; i < num_frames; ++i) {
            out[i] = 0.5f * (interleaved[2 * i] + interleaved[2 * i + 1]);
        }
    }

    return waveform;
}

}  // namespace audio
}  // namespace utils
