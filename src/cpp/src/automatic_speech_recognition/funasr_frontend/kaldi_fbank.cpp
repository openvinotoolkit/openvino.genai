// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/funasr_frontend/kaldi_fbank.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::genai::funasr_frontend {

namespace {

constexpr size_t FFT_SIZE = 512;
constexpr float PI = 3.14159265358979323846f;
// FunASR hardcodes this scale to convert normalized audio to the signed 16-bit PCM range.
constexpr float PCM16_SCALE = 32768.0f;
// FunASR hardcodes the Kaldi pre-emphasis filter as y[n] = x[n] - 0.97 * x[n - 1].
constexpr float PREEMPHASIS_COEFFICIENT = 0.97f;
// FunASR hardcodes the standard Hamming window as 0.54 - 0.46 * cos(2 * PI * n / (N - 1)).
constexpr float HAMMING_OFFSET = 0.54f;
constexpr float HAMMING_COSINE_SCALE = 0.46f;

float mel_scale(const float frequency) {
    return 1127.0f * std::log1p(frequency / 700.0f);
}

void fft(std::vector<std::complex<float>>& values) {
    const size_t size = values.size();
    for (size_t index = 1, reversed = 0; index < size; ++index) {
        size_t bit = size >> 1;
        for (; reversed & bit; bit >>= 1) {
            reversed ^= bit;
        }
        reversed ^= bit;
        if (index < reversed) {
            std::swap(values[index], values[reversed]);
        }
    }
    for (size_t length = 2; length <= size; length <<= 1) {
        const std::complex<float> step = std::polar(1.0f, -2.0f * PI / static_cast<float>(length));
        for (size_t offset = 0; offset < size; offset += length) {
            std::complex<float> factor(1.0f, 0.0f);
            for (size_t index = 0; index < length / 2; ++index) {
                const std::complex<float> even = values[offset + index];
                const std::complex<float> odd = values[offset + index + length / 2] * factor;
                values[offset + index] = even + odd;
                values[offset + index + length / 2] = even - odd;
                factor *= step;
            }
        }
    }
}

std::vector<float> make_mel_filters() {
    constexpr size_t frequency_bins = FFT_SIZE / 2 + 1;
    std::vector<float> filters(MEL_BINS * frequency_bins, 0.0f);
    const float low_mel = mel_scale(20.0f);
    const float high_mel = mel_scale(0.5f * SAMPLING_RATE);
    const float mel_delta = (high_mel - low_mel) / static_cast<float>(MEL_BINS + 1);
    const float fft_bin_width = static_cast<float>(SAMPLING_RATE) / FFT_SIZE;
    for (size_t mel_bin = 0; mel_bin < MEL_BINS; ++mel_bin) {
        const float left_mel = low_mel + static_cast<float>(mel_bin) * mel_delta;
        const float center_mel = low_mel + static_cast<float>(mel_bin + 1) * mel_delta;
        const float right_mel = low_mel + static_cast<float>(mel_bin + 2) * mel_delta;
        for (size_t fft_bin = 0; fft_bin + 1 < frequency_bins; ++fft_bin) {
            const float fft_bin_mel = mel_scale(static_cast<float>(fft_bin) * fft_bin_width);
            const float up_slope = (fft_bin_mel - left_mel) / (center_mel - left_mel);
            const float down_slope = (right_mel - fft_bin_mel) / (right_mel - center_mel);
            filters[mel_bin * frequency_bins + fft_bin] = std::max(0.0f, std::min(up_slope, down_slope));
        }
    }
    return filters;
}

}  // namespace

void compute_fbank(const float* samples,
                   const size_t frame_count,
                   const size_t frame_length,
                   const float dither,
                   float* output) {
    OPENVINO_ASSERT(frame_length >= 2 && frame_length <= FRAME_LENGTH,
                    "Filter-bank frame length must be in [2, ",
                    FRAME_LENGTH,
                    "], got ",
                    frame_length);
    static const std::vector<float> mel_filters = make_mel_filters();
    constexpr size_t frequency_bins = FFT_SIZE / 2 + 1;
    std::vector<std::complex<float>> spectrum(FFT_SIZE);

    std::vector<float> scaled(frame_length);
    std::mt19937 generator;
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    if (dither != 0.0f) {
        generator.seed(std::random_device{}());
    }

    for (size_t frame = 0; frame < frame_count; ++frame) {
        const size_t offset = frame * FRAME_SHIFT;
        double mean_sum = 0.0;
        for (size_t sample = 0; sample < frame_length; ++sample) {
            float value = samples[offset + sample] * PCM16_SCALE;
            if (dither == 0.0f) {
                // Preserve the original double-precision accumulation to avoid intermediate float rounding when dither
                // is disabled.
                mean_sum += static_cast<double>(samples[offset + sample]) * PCM16_SCALE;
            } else {
                value += dither * gaussian(generator);
                mean_sum += static_cast<double>(value);
            }
            scaled[sample] = value;
        }
        const float mean = static_cast<float>(mean_sum / static_cast<double>(frame_length));
        std::fill(spectrum.begin(), spectrum.end(), std::complex<float>{});
        for (size_t sample = 0; sample < frame_length; ++sample) {
            const float current = scaled[sample] - mean;
            const float previous = sample == 0 ? current : scaled[sample - 1] - mean;
            const float window =
                HAMMING_OFFSET - HAMMING_COSINE_SCALE * std::cos(2.0f * PI * sample / (frame_length - 1));
            spectrum[sample] = (current - PREEMPHASIS_COEFFICIENT * previous) * window;
        }
        fft(spectrum);
        for (size_t mel = 0; mel < MEL_BINS; ++mel) {
            float energy = 0.0f;
            for (size_t bin = 0; bin < frequency_bins; ++bin) {
                energy += std::norm(spectrum[bin]) * mel_filters[mel * frequency_bins + bin];
            }
            output[frame * MEL_BINS + mel] = std::log(std::max(energy, std::numeric_limits<float>::epsilon()));
        }
    }
}

}  // namespace ov::genai::funasr_frontend
