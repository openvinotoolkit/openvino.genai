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

/// Log-mel spectrogram extractor shared by Whisper and Qwen3 Omni audio pipelines.
/// Output is a flat [num_mel_bins, n_frames] row-major buffer with Whisper-style
/// max-8 clamp and (x+4)/4 normalization.
class MelSpectrogramExtractor {
public:
    MelSpectrogramExtractor(size_t num_mel_bins, size_t sampling_rate, size_t n_fft, size_t hop_length);

    /// Log-mel extraction with optional Whisper-style min-length padding.
    /// raw_speech must have size > n_fft/2. n_frames is set to the produced frame count.
    /// @param min_length When non-zero, raw_speech is padded up to min_length before the
    ///        n_fft/2 reflect pad — the Whisper 30 s pre-pad. When zero (default), no extra
    ///        padding is applied (Qwen3 Omni semantics).
    /// @param n_active_frames When non-null, receives raw_speech.size() / hop_length — the count
    ///        of frames backed by real audio, used for long-form offset bookkeeping. When null
    ///        (default), the active-frame count is not computed.
    std::vector<float> extract(const std::vector<float>& raw_speech,
                               size_t& n_frames,
                               size_t min_length = 0,
                               size_t* n_active_frames = nullptr) const;

    size_t num_mel_bins() const {
        return m_num_mel_bins;
    }
    size_t sampling_rate() const {
        return m_sampling_rate;
    }
    size_t n_fft() const {
        return m_n_fft;
    }
    size_t hop_length() const {
        return m_hop_length;
    }

private:
    std::vector<float> pad_with_reflect(const std::vector<float>& raw_speech, size_t min_length) const;
    std::vector<float> compute_mel(const std::vector<float>& padded, size_t n_samples, size_t n_frames) const;

    const size_t m_num_mel_bins;
    const size_t m_sampling_rate;
    const size_t m_n_fft;
    const size_t m_hop_length;
    const std::vector<float> m_sin_vals;
    const std::vector<float> m_cos_vals;
    const std::vector<float> m_mel_filter;
};

}  // namespace ov::genai::audio_utils
