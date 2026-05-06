// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef _WIN32
#    define _USE_MATH_DEFINES
#endif

#include "audio_utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ov::genai::audio_utils {

float hertz_to_mel(float freq) {
    const float logstep = MEL_HIGH_FREQUENCY_Q / logf(MEL_DIVISOR);
    float mel = 3.0f * freq / 200.0f;
    if (freq >= MEL_BREAK_FREQUENCY_HERTZ) {
        mel = MEL_BREAK_VALUE + logf(freq / MEL_BREAK_FREQUENCY_HERTZ) * logstep;
    }
    return mel;
}

float mel_to_hertz(float mel) {
    const float logstep = logf(MEL_DIVISOR) / MEL_HIGH_FREQUENCY_Q;
    float freq = 200.0f * mel / 3.0f;
    if (mel >= MEL_BREAK_VALUE) {
        freq = MEL_BREAK_FREQUENCY_HERTZ * expf(logstep * (mel - MEL_BREAK_VALUE));
    }
    return freq;
}

std::vector<float> hann_window(size_t length) {
    std::vector<float> output(length);
    for (size_t i = 0; i < length; i++) {
        output[i] = 0.5f * (1.0f - cosf((2.0f * static_cast<float>(M_PI) * i) / length));
    }
    return output;
}

std::vector<float> build_sin_table(size_t n_fft) {
    std::vector<float> sin_vals(n_fft);
    for (size_t i = 0; i < n_fft; i++) {
        const double theta = (2.0 * M_PI * i) / n_fft;
        sin_vals[i] = sinf(theta);
    }
    return sin_vals;
}

std::vector<float> build_cos_table(size_t n_fft) {
    std::vector<float> cos_vals(n_fft);
    for (size_t i = 0; i < n_fft; i++) {
        const double theta = (2.0 * M_PI * i) / n_fft;
        cos_vals[i] = cosf(theta);
    }
    return cos_vals;
}

void dft(const std::vector<float>& in,
         std::vector<float>& out,
         const std::vector<float>& sin_vals,
         const std::vector<float>& cos_vals,
         size_t n_fft) {
    const auto N = static_cast<int>(in.size());
    out.resize(N * 2);
    const auto sin_cos_step = static_cast<int>(n_fft / N);

    for (int k = 0; k < N; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            const auto idx = (k * n * sin_cos_step) % static_cast<int>(n_fft);
            re += in[n] * cos_vals[idx];
            im -= in[n] * sin_vals[idx];
        }
        out[k * 2 + 0] = re;
        out[k * 2 + 1] = im;
    }
}

void fft(const std::vector<float>& in,
         std::vector<float>& out,
         const std::vector<float>& sin_vals,
         const std::vector<float>& cos_vals,
         size_t n_fft) {
    out.resize(in.size() * 2);
    const auto N = static_cast<int>(in.size());

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N % 2 == 1) {
        dft(in, out, sin_vals, cos_vals, n_fft);
        return;
    }

    std::vector<float> even, odd;
    even.reserve(N / 2);
    odd.reserve(N / 2);

    for (int i = 0; i < N; i++) {
        (i % 2 == 0 ? even : odd).push_back(in[i]);
    }

    std::vector<float> even_fft, odd_fft;
    fft(even, even_fft, sin_vals, cos_vals, n_fft);
    fft(odd, odd_fft, sin_vals, cos_vals, n_fft);

    const auto sin_cos_step = static_cast<int>(n_fft / N);
    for (int k = 0; k < N / 2; k++) {
        const auto idx = k * sin_cos_step;
        const float re = cos_vals[idx];
        const float im = -sin_vals[idx];
        const float re_odd = odd_fft[2 * k + 0];
        const float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + N / 2) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + N / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

std::vector<float> build_mel_filter(size_t num_frequency_bins, size_t num_mel_bins, size_t sampling_rate) {
    const float max_frequency = static_cast<float>(sampling_rate) / 2.0f;
    const float mel_min = hertz_to_mel(0.0f);
    const float mel_max = hertz_to_mel(max_frequency);
    const float mel_step = (mel_max - mel_min) / static_cast<float>(num_mel_bins + 1);

    std::vector<float> filter_freqs(num_mel_bins + 2);
    for (size_t i = 0; i < filter_freqs.size(); i++) {
        filter_freqs[i] = mel_to_hertz(mel_min + i * mel_step);
    }

    std::vector<float> fft_freqs(num_frequency_bins);
    const float fft_freq_step = max_frequency / static_cast<float>(num_frequency_bins - 1);
    for (size_t i = 0; i < num_frequency_bins; i++) {
        fft_freqs[i] = i * fft_freq_step;
    }

    // Triangular filter bank with slaney normalization
    std::vector<float> result(num_mel_bins * num_frequency_bins, 0.0f);
    for (size_t mel = 0; mel < num_mel_bins; mel++) {
        const float f_left = filter_freqs[mel];
        const float f_center = filter_freqs[mel + 1];
        const float f_right = filter_freqs[mel + 2];
        const float enorm = 2.0f / (f_right - f_left);

        for (size_t fft_idx = 0; fft_idx < num_frequency_bins; fft_idx++) {
            const float f = fft_freqs[fft_idx];
            const float up_slope = (f - f_left) / (f_center - f_left);
            const float down_slope = (f_right - f) / (f_right - f_center);
            const float val = std::max(0.0f, std::min(up_slope, down_slope));
            result[mel * num_frequency_bins + fft_idx] = val * enorm;
        }
    }
    return result;
}

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
                const std::vector<float>& cos_vals) {
    std::vector<float> fft_in(frame_size, 0.0f);
    std::vector<float> fft_out(2 * frame_size);
    const int n_fft = 1 + (frame_size / 2);

    for (int i = ith; i < static_cast<int>(n_frames); i += n_threads) {
        const auto offset = i * frame_step;

        // Apply Hanning window
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0f);
        }

        fft(fft_in, fft_out, sin_vals, cos_vals, frame_size);

        // Power spectrum
        for (int j = 0; j < n_fft; j++) {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }

        // Apply mel filters
        for (size_t j = 0; j < feature_size; j++) {
            double sum = 0.0;
            for (int k = 0; k < n_fft; k++) {
                sum += fft_out[k] * mel_filter[j * n_fft + k];
            }
            // Log scale (Qwen3-Omni uses log10, same as Whisper)
            output[j * n_frames + i] = static_cast<float>(log10(std::max(sum, 1e-10)));
        }
    }
}

}  // namespace ov::genai::audio_utils
