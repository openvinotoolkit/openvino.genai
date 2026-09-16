// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speech_generation/qwen3_tts/mel_preprocess.hpp"

#include <algorithm>
#include <cmath>

#include <openvino/opsets/opset15.hpp>

namespace ov {
namespace genai {

namespace {

constexpr float PI_F = 3.14159265358979323846f;

float hertz_to_mel(const float freq) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    const float logstep = 27.0f / std::log(6.4f);
    float mel = 3.0f * freq / 200.0f;

    if (freq >= min_log_hertz) {
        mel = min_log_mel + std::log(freq / min_log_hertz) * logstep;
    }
    return mel;
}

float mel_to_hertz(const float mel) {
    constexpr float min_log_hertz = 1000.0f;
    constexpr float min_log_mel = 15.0f;
    const float logstep = std::log(6.4f) / 27.0f;
    float freq = 200.0f * mel / 3.0f;

    if (mel >= min_log_mel) {
        freq = min_log_hertz * std::exp(logstep * (mel - min_log_mel));
    }
    return freq;
}

std::vector<std::vector<float>> create_triangular_filter_bank(const std::vector<float>& fft_freqs,
                                                              const std::vector<float>& filter_freqs) {
    std::vector<float> filter_diff(filter_freqs.size() - 1);
    for (size_t i = 0; i < filter_diff.size(); ++i) {
        filter_diff[i] = filter_freqs[i + 1] - filter_freqs[i];
    }

    std::vector<std::vector<float>> slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size()));
    for (size_t row = 0; row < slopes.size(); ++row) {
        for (size_t col = 0; col < slopes[0].size(); ++col) {
            slopes[row][col] = filter_freqs[col] - fft_freqs[row];
        }
    }

    std::vector<std::vector<float>> down_slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < down_slopes.size(); ++row) {
        for (size_t col = 0; col < down_slopes[0].size(); ++col) {
            down_slopes[row][col] = -slopes[row][col] / filter_diff[col];
        }
    }

    std::vector<std::vector<float>> up_slopes(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < up_slopes.size(); ++row) {
        for (size_t col = 0; col < up_slopes[0].size(); ++col) {
            up_slopes[row][col] = slopes[row][col + 2] / filter_diff[col + 1];
        }
    }

    std::vector<std::vector<float>> result(fft_freqs.size(), std::vector<float>(filter_freqs.size() - 2));
    for (size_t row = 0; row < result.size(); ++row) {
        for (size_t col = 0; col < result[0].size(); ++col) {
            result[row][col] = std::max(0.0f, std::min(down_slopes[row][col], up_slopes[row][col]));
        }
    }

    return result;
}

std::vector<std::vector<float>> mel_filter_bank(const int64_t num_frequency_bins,
                                                const int64_t num_mel_filters,
                                                const int64_t sampling_rate,
                                                const float min_frequency,
                                                const float max_frequency) {
    OPENVINO_ASSERT(max_frequency <= (sampling_rate / 2.0f),
                    "max_frequency should be less or equal sampling_rate / 2");

    const float mel_min = hertz_to_mel(min_frequency);
    const float mel_max = hertz_to_mel(max_frequency);

    const float mel_freqs_step = (mel_max - mel_min) / static_cast<float>(num_mel_filters + 1);
    std::vector<float> filter_freqs(num_mel_filters + 2);
    for (size_t i = 0; i < filter_freqs.size(); ++i) {
        filter_freqs[i] = mel_to_hertz(mel_min + static_cast<float>(i) * mel_freqs_step);
    }

    std::vector<float> fft_freqs(num_frequency_bins);
    const float fft_freq_step = (sampling_rate / 2.0f) / static_cast<float>(num_frequency_bins - 1);
    for (size_t i = 0; i < fft_freqs.size(); ++i) {
        fft_freqs[i] = static_cast<float>(i) * fft_freq_step;
    }

    auto mel_filters = create_triangular_filter_bank(fft_freqs, filter_freqs);
    std::vector<float> enorm(num_mel_filters);
    for (size_t i = 0; i < enorm.size(); ++i) {
        enorm[i] = 2.0f / (filter_freqs[i + 2] - filter_freqs[i]);
    }

    for (size_t row = 0; row < mel_filters.size(); ++row) {
        for (size_t col = 0; col < mel_filters[0].size(); ++col) {
            mel_filters[row][col] *= enorm[col];
        }
    }
    return mel_filters;
}

std::vector<float> hann_window(size_t length) {
    std::vector<float> out(length, 0.0f);
    for (size_t i = 0; i < length; ++i) {
        out[i] = 0.5f * (1.0f - std::cos(2.0f * PI_F * static_cast<float>(i) / static_cast<float>(length)));
    }
    return out;
}

}  // namespace

std::shared_ptr<ov::Model> build_qwen3_mel_preprocess_model(size_t mel_dim) {
    constexpr int64_t n_fft = 1024;
    constexpr int64_t hop_size = 256;
    constexpr int64_t frame_size = 1024;
    constexpr int64_t padding = (n_fft - hop_size) / 2;

    auto waveform = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
    waveform->set_friendly_name("waveform");
    waveform->output(0).set_names({"waveform"});

    auto axis_1 = ov::opset15::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto unsqueezed = std::make_shared<ov::opset15::Unsqueeze>(waveform, axis_1);  // [B, 1, T]

    auto pads_begin = ov::opset15::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, padding});
    auto pads_end = ov::opset15::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, padding});
    auto pad_value = ov::opset15::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
    auto padded = std::make_shared<ov::opset15::Pad>(unsqueezed,
                                                     pads_begin,
                                                     pads_end,
                                                     pad_value,
                                                     ov::op::PadMode::REFLECT);

    auto squeezed = std::make_shared<ov::opset15::Squeeze>(padded, axis_1);  // [B, T']

    const auto hann = hann_window(static_cast<size_t>(frame_size));
    auto window = ov::opset15::Constant::create(ov::element::f32,
                                                ov::Shape{static_cast<size_t>(frame_size)},
                                                hann);
    auto frame_size_c = ov::opset15::Constant::create(ov::element::i32,
                                                      ov::Shape{},
                                                      std::vector<int32_t>{static_cast<int32_t>(frame_size)});
    auto frame_step_c = ov::opset15::Constant::create(ov::element::i32,
                                                      ov::Shape{},
                                                      std::vector<int32_t>{static_cast<int32_t>(hop_size)});

    auto stft = std::make_shared<ov::opset15::STFT>(squeezed, window, frame_size_c, frame_step_c, false);

    auto power_2 = ov::opset15::Constant::create(ov::element::f32,
                                                 ov::Shape{1, 1, 1, 1},
                                                 std::vector<float>{2.0f});
    auto squared = std::make_shared<ov::opset15::Power>(stft, power_2);
    auto imag_axis = ov::opset15::Constant::create(ov::element::i64,
                                                   ov::Shape{},
                                                   std::vector<int64_t>{-1});
    auto power_sum = std::make_shared<ov::opset15::ReduceSum>(squared, imag_axis, false);
    auto magnitude = std::make_shared<ov::opset15::Sqrt>(power_sum);  // [B, F, Frames]

    const auto mel_filter_2d = mel_filter_bank(1 + n_fft / 2,
                                               static_cast<int64_t>(mel_dim),
                                               24000,
                                               0.0f,
                                               12000.0f);
    std::vector<float> mel_filter_flat;
    mel_filter_flat.reserve(mel_dim * (1 + n_fft / 2));
    for (size_t m = 0; m < mel_dim; ++m) {
        for (size_t f = 0; f < (1 + n_fft / 2); ++f) {
            mel_filter_flat.push_back(mel_filter_2d[f][m]);
        }
    }
    auto mel_filter = ov::opset15::Constant::create(ov::element::f32,
                                                    ov::Shape{1, mel_dim, static_cast<size_t>(1 + n_fft / 2)},
                                                    mel_filter_flat);

    auto mel = std::make_shared<ov::opset15::MatMul>(mel_filter, magnitude, false, true);  // [B, M, Frames]
    auto min_clip = ov::opset15::Constant::create(ov::element::f32,
                                                  ov::Shape{1, 1, 1},
                                                  std::vector<float>{1e-5f});
    auto clipped = std::make_shared<ov::opset15::Maximum>(mel, min_clip);
    auto log_mel = std::make_shared<ov::opset15::Log>(clipped);

    auto order = ov::opset15::Constant::create(ov::element::i64,
                                               ov::Shape{3},
                                               std::vector<int64_t>{0, 2, 1});
    auto output = std::make_shared<ov::opset15::Transpose>(log_mel, order);  // [B, Frames, M]
    output->set_friendly_name("log_mel_features");

    auto result = std::make_shared<ov::opset15::Result>(output);
    result->set_friendly_name("log_mel_features");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{waveform}, "qwen3_mel_preprocess");
}

}  // namespace genai
}  // namespace ov
