// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "save_audio.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

void save_audio(const std::string& filename, const ov::Tensor& audio_tensor, uint32_t sample_rate) {
    const ov::Shape shape = audio_tensor.get_shape();  // [B, C, S]
    if (shape.size() != 3) {
        throw std::runtime_error("save_audio(): expected audio tensor of shape [B, C, S]");
    }
    const size_t B = shape[0], C = shape[1], S = shape[2];
    const float* data = audio_tensor.data<const float>();

    for (size_t b = 0; b < B; ++b) {
        std::string out = filename;
        if (B > 1) {
            std::filesystem::path p(filename);
            const std::string ext = p.has_extension() ? p.extension().string() : ".wav";
            out = (p.parent_path() / (p.stem().string() + "_b" + std::to_string(b) + ext)).string();
        }

        // [C, S] -> interleaved [S, C]
        std::vector<float> interleaved(C * S);
        for (size_t c = 0; c < C; ++c) {
            for (size_t i = 0; i < S; ++i) {
                interleaved[i * C + c] = data[(b * C + c) * S + i];
            }
        }

        drwav_data_format format;
        format.container = drwav_container_riff;
        format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
        format.channels = static_cast<drwav_uint32>(C);
        format.sampleRate = sample_rate;
        format.bitsPerSample = 32;

        drwav wav;
        if (!drwav_init_file_write(&wav, out.c_str(), &format, nullptr)) {
            throw std::runtime_error("save_audio(): failed to open " + out);
        }
        const drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, S, interleaved.data());
        drwav_uninit(&wav);
        if (frames_written != S) {
            throw std::runtime_error("save_audio(): failed to write " + out);
        }
        std::cout << "Wrote " << out << " (" << S << " samples, " << C << " channels @ " << sample_rate << " Hz)\n";
    }
}
