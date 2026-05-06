// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

namespace ov::genai::audio_utils {

// Mel-frequency scale constants (slaney normalization)
constexpr float MEL_BREAK_FREQUENCY_HERTZ = 1000.0f;
constexpr float MEL_BREAK_VALUE = 15.0f;
// 27.0 / log(6.4) = inverse of log(6.4) / 27.0
constexpr float MEL_HIGH_FREQUENCY_Q = 27.0f;
constexpr float MEL_DIVISOR = 6.4f;

/// Convert frequency in Hertz to mel scale (slaney normalization).
float hertz_to_mel(float freq);

/// Convert mel scale value to frequency in Hertz (slaney normalization).
float mel_to_hertz(float mel);

/// Compute Hann window of given length.
/// @param length Window size.
/// @return Hann window coefficients.
std::vector<float> hann_window(size_t length);

/// Compute pre-calculated sine table for FFT.
std::vector<float> build_sin_table(size_t n_fft);

/// Compute pre-calculated cosine table for FFT.
std::vector<float> build_cos_table(size_t n_fft);

/// Naive Discrete Fourier Transform (real input, complex output interleaved).
void dft(const std::vector<float>& in,
         std::vector<float>& out,
         const std::vector<float>& sin_vals,
         const std::vector<float>& cos_vals,
         size_t n_fft);

/// Cooley-Tukey FFT (real input, complex output interleaved).
void fft(const std::vector<float>& in,
         std::vector<float>& out,
         const std::vector<float>& sin_vals,
         const std::vector<float>& cos_vals,
         size_t n_fft);

/// Build flattened mel filter bank [num_mel_bins * num_frequency_bins] with slaney normalization.
std::vector<float> build_mel_filter(size_t num_frequency_bins, size_t num_mel_bins, size_t sampling_rate);

/// Worker thread for parallel mel spectrogram computation.
/// Each thread processes frames [ith, ith + n_threads, ith + 2*n_threads, ...].
void mel_worker(int ith,
                const std::vector<float>& hann,
                const std::vector<float>& samples,
                int n_samples,
                int frame_size,
                int frame_step,
                int n_threads,
                const std::vector<float>& mel_filter,
                size_t feature_size,
                size_t n_frames,
                std::vector<float>& output,
                const std::vector<float>& sin_vals,
                const std::vector<float>& cos_vals);

}  // namespace ov::genai::audio_utils
