// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

#include "openvino/core/except.hpp"

namespace {

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
    constexpr size_t frequency_bins = ov::genai::FunASRFeatureExtractor::fft_size / 2 + 1;
    std::vector<float> filters(ov::genai::FunASRFeatureExtractor::mel_bins * frequency_bins, 0.0f);
    const float low_mel = mel_scale(20.0f);
    const float high_mel = mel_scale(0.5f * ov::genai::FunASRFeatureExtractor::sampling_rate);
    const float mel_delta = (high_mel - low_mel) / static_cast<float>(ov::genai::FunASRFeatureExtractor::mel_bins + 1);
    const float fft_bin_width = static_cast<float>(ov::genai::FunASRFeatureExtractor::sampling_rate) /
                                ov::genai::FunASRFeatureExtractor::fft_size;
    for (size_t mel_bin = 0; mel_bin < ov::genai::FunASRFeatureExtractor::mel_bins; ++mel_bin) {
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

namespace ov::genai {

ov::Tensor FunASRFeatureExtractor::extract(const std::vector<float>& audio) const {
    OPENVINO_ASSERT(audio.size() >= 2, "Fun-ASR input audio must contain at least 2 samples");
    const size_t effective_frame_length = std::min(frame_length, audio.size());
    const size_t frame_count =
        audio.size() < effective_frame_length ? 0 : 1 + (audio.size() - effective_frame_length) / frame_shift;
    OPENVINO_ASSERT(frame_count > 0, "Fun-ASR input audio is too short to extract a frame");

    static const std::vector<float> mel_filters = make_mel_filters();
    constexpr size_t frequency_bins = fft_size / 2 + 1;
    std::vector<float> fbank(frame_count * mel_bins);
    std::vector<std::complex<float>> spectrum(fft_size);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        const size_t offset = frame * frame_shift;
        double mean_sum = 0.0;
        for (size_t sample = 0; sample < effective_frame_length; ++sample) {
            mean_sum += static_cast<double>(audio[offset + sample]) * PCM16_SCALE;
        }
        const float mean = static_cast<float>(mean_sum / static_cast<double>(effective_frame_length));
        std::fill(spectrum.begin(), spectrum.end(), std::complex<float>{});
        for (size_t sample = 0; sample < effective_frame_length; ++sample) {
            const float current = audio[offset + sample] * PCM16_SCALE - mean;
            const float previous = sample == 0 ? current : audio[offset + sample - 1] * PCM16_SCALE - mean;
            const float window =
                HAMMING_OFFSET - HAMMING_COSINE_SCALE * std::cos(2.0f * PI * sample / (effective_frame_length - 1));
            spectrum[sample] = (current - PREEMPHASIS_COEFFICIENT * previous) * window;
        }
        fft(spectrum);
        for (size_t mel = 0; mel < mel_bins; ++mel) {
            float energy = 0.0f;
            for (size_t bin = 0; bin < frequency_bins; ++bin) {
                energy += std::norm(spectrum[bin]) * mel_filters[mel * frequency_bins + bin];
            }
            fbank[frame * mel_bins + mel] = std::log(std::max(energy, std::numeric_limits<float>::epsilon()));
        }
    }

    const size_t lfr_frames = (frame_count + lfr_stride - 1) / lfr_stride;
    ov::Tensor result(ov::element::f32, {1, lfr_frames, mel_bins * lfr_window});
    float* output = result.data<float>();
    constexpr size_t left_context = (lfr_window - 1) / 2;
    for (size_t frame = 0; frame < lfr_frames; ++frame) {
        for (size_t context = 0; context < lfr_window; ++context) {
            const int64_t source = static_cast<int64_t>(frame * lfr_stride + context) - left_context;
            const size_t source_frame = static_cast<size_t>(std::clamp<int64_t>(source, 0, frame_count - 1));
            std::copy_n(fbank.data() + source_frame * mel_bins, mel_bins, output);
            output += mel_bins;
        }
    }
    return result;
}

}  // namespace ov::genai
