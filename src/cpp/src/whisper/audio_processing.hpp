// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160

namespace ov {
namespace genai {
namespace utils {
namespace audio {

std::vector<std::vector<float>> mel_filter_bank(const int64_t num_frequency_bins,
                                                const int64_t num_mel_filters,
                                                const int64_t sampling_rate,
                                                const float min_frequency = 0.0f,
                                                const float max_frequency = 8000.0f);

void fill_sin_cos_table();

std::vector<float> mel_spectrogram_convert_audio(const std::vector<float> pcmf32,
                                                 const int sample_rate,
                                                 const int frame_size,
                                                 const int frame_step,
                                                 const int n_threads);
}  // namespace audio
}  // namespace utils
}  // namespace genai
}  // namespace ov
