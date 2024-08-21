// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160

namespace ov {
namespace genai {
namespace utils {
namespace audio {

void fill_sin_cos_table();

void mel_spectrogram_convert_audio(const float* samples,
                                   const int n_samples,
                                   const int sample_rate,
                                   const int frame_size,
                                   const int frame_step,
                                   const int n_threads,
                                   std::vector<float>& outMelData);
}  // namespace audio
}  // namespace utils
}  // namespace genai
}  // namespace ov
