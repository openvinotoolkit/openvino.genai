// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma4/audio_feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <vector>

#include "json_utils.hpp"
#include "nlohmann/json.hpp"
#include "openvino/openvino.hpp"

namespace ov::genai {

namespace {

constexpr size_t PAD_TO_MULTIPLE_OF = 128;
constexpr size_t MAX_AUDIO_DURATION_SECONDS = 30;
constexpr double PI = 3.14159265358979323846;

double hertz_to_mel(double frequency) {
    return 2595.0 * std::log10(1.0 + frequency / 700.0);
}

double mel_to_hertz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

std::vector<double> create_mel_filters(size_t sampling_rate,
                                       size_t feature_size,
                                       size_t fft_length,
                                       double min_frequency,
                                       double max_frequency) {
    const size_t fft_bins = fft_length / 2 + 1;
    const double min_mel = hertz_to_mel(min_frequency);
    const double max_mel = hertz_to_mel(max_frequency);
    std::vector<double> filter_frequencies(feature_size + 2);
    for (size_t i = 0; i < filter_frequencies.size(); ++i) {
        const double mel =
            min_mel + (max_mel - min_mel) * static_cast<double>(i) / static_cast<double>(feature_size + 1);
        filter_frequencies[i] = mel_to_hertz(mel);
    }

    std::vector<double> filters(fft_bins * feature_size, 0.0);
    for (size_t frequency_bin = 0; frequency_bin < fft_bins; ++frequency_bin) {
        const double frequency = static_cast<double>(frequency_bin * sampling_rate) / static_cast<double>(fft_length);
        for (size_t mel_bin = 0; mel_bin < feature_size; ++mel_bin) {
            const double lower = filter_frequencies[mel_bin];
            const double center = filter_frequencies[mel_bin + 1];
            const double upper = filter_frequencies[mel_bin + 2];
            const double rising = (frequency - lower) / (center - lower);
            const double falling = (upper - frequency) / (upper - center);
            filters[frequency_bin * feature_size + mel_bin] = std::max(0.0, std::min(rising, falling));
        }
    }
    return filters;
}

void fft(std::vector<std::complex<double>>& values) {
    const size_t size = values.size();
    for (size_t i = 1, j = 0; i < size; ++i) {
        size_t bit = size >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }

    for (size_t length = 2; length <= size; length <<= 1) {
        const std::complex<double> root = std::polar(1.0, -2.0 * PI / static_cast<double>(length));
        for (size_t offset = 0; offset < size; offset += length) {
            std::complex<double> factor{1.0, 0.0};
            for (size_t i = 0; i < length / 2; ++i) {
                const std::complex<double> even = values[offset + i];
                const std::complex<double> odd = values[offset + i + length / 2] * factor;
                values[offset + i] = even + odd;
                values[offset + i + length / 2] = even - odd;
                factor *= root;
            }
        }
    }
}

}  // namespace

Gemma4AudioFeatureExtractor::Gemma4AudioFeatureExtractor() {
    validate_parameters();
    m_mel_filters = create_mel_filters(m_sampling_rate, m_feature_size, m_fft_length, m_min_frequency, m_max_frequency);
}

Gemma4AudioFeatureExtractor::Gemma4AudioFeatureExtractor(const std::filesystem::path& config_dir_path) {
    const std::filesystem::path config_path = config_dir_path / "processor_config.json";
    std::ifstream stream(config_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", config_path, "' with Gemma4 processor config");
    const nlohmann::json parsed = nlohmann::json::parse(stream);
    OPENVINO_ASSERT(parsed.contains("feature_extractor"), "Gemma4 processor config must contain feature_extractor");
    const nlohmann::json& feature_extractor = parsed.at("feature_extractor");
    utils::read_json_param(feature_extractor, "sampling_rate", m_sampling_rate);
    utils::read_json_param(feature_extractor, "feature_size", m_feature_size);
    utils::read_json_param(feature_extractor, "frame_length", m_frame_length);
    utils::read_json_param(feature_extractor, "hop_length", m_hop_length);
    utils::read_json_param(feature_extractor, "fft_length", m_fft_length);
    utils::read_json_param(feature_extractor, "min_frequency", m_min_frequency);
    utils::read_json_param(feature_extractor, "max_frequency", m_max_frequency);
    utils::read_json_param(feature_extractor, "mel_floor", m_mel_floor);
    validate_parameters();
    m_mel_filters = create_mel_filters(m_sampling_rate, m_feature_size, m_fft_length, m_min_frequency, m_max_frequency);
}

void Gemma4AudioFeatureExtractor::validate_parameters() const {
    OPENVINO_ASSERT(m_sampling_rate > 0, "Gemma4 audio sampling_rate must be positive");
    OPENVINO_ASSERT(m_feature_size > 0, "Gemma4 audio feature_size must be positive");
    OPENVINO_ASSERT(m_frame_length > 0, "Gemma4 audio frame_length must be positive");
    OPENVINO_ASSERT(m_hop_length > 0, "Gemma4 audio hop_length must be positive");
    OPENVINO_ASSERT(m_fft_length >= m_frame_length,
                    "Gemma4 audio fft_length must be greater than or equal to frame_length");
    OPENVINO_ASSERT((m_fft_length & (m_fft_length - 1)) == 0, "Gemma4 audio fft_length must be a power of two");
    OPENVINO_ASSERT(m_min_frequency >= 0.0, "Gemma4 audio min_frequency must be non-negative");
    OPENVINO_ASSERT(m_max_frequency > m_min_frequency, "Gemma4 audio max_frequency must be greater than min_frequency");
    OPENVINO_ASSERT(m_max_frequency <= static_cast<double>(m_sampling_rate) / 2.0,
                    "Gemma4 audio max_frequency must not exceed the Nyquist frequency");
    OPENVINO_ASSERT(m_mel_floor > 0.0, "Gemma4 audio mel_floor must be positive");
}

void Gemma4AudioFeatureExtractor::validate_audio_input(const ov::Tensor& audio) const {
    OPENVINO_ASSERT(audio.get_element_type() == ov::element::f32,
                    "Gemma4 audio input must be float32 PCM, got ",
                    audio.get_element_type());
    OPENVINO_ASSERT(audio.get_shape().size() == 1,
                    "Gemma4 audio input must be a 1-D tensor of mono PCM samples, got rank ",
                    audio.get_shape().size());
    const size_t minimum_samples = m_frame_length / 2 + 1;
    OPENVINO_ASSERT(audio.get_size() >= minimum_samples,
                    "Gemma4 audio input must contain at least ",
                    minimum_samples,
                    " samples, got ",
                    audio.get_size());
    const size_t max_audio_samples = MAX_AUDIO_DURATION_SECONDS * m_sampling_rate;
    OPENVINO_ASSERT(audio.get_size() <= max_audio_samples,
                    "Gemma4 audio input exceeds the 30 second limit of ",
                    max_audio_samples,
                    " samples at ",
                    m_sampling_rate,
                    " Hz, got ",
                    audio.get_size());
}

Gemma4AudioFeatures Gemma4AudioFeatureExtractor::extract(const ov::Tensor& audio) const {
    validate_audio_input(audio);
    const size_t audio_length = audio.get_size();
    const size_t padded_length = ((audio_length + PAD_TO_MULTIPLE_OF - 1) / PAD_TO_MULTIPLE_OF) * PAD_TO_MULTIPLE_OF;
    const size_t left_padding = m_frame_length / 2;
    const size_t frame_window = m_frame_length + 1;
    const size_t num_frames = (left_padding + padded_length - frame_window) / m_hop_length + 1;
    const size_t fft_bins = m_fft_length / 2 + 1;

    ov::Tensor input_features(ov::element::f32, {1, num_frames, m_feature_size});
    ov::Tensor input_features_mask(ov::element::boolean, {1, num_frames});
    float* feature_data = input_features.data<float>();
    bool* mask_data = input_features_mask.data<bool>();
    std::fill(feature_data, feature_data + input_features.get_size(), 0.0f);

    const float* audio_data = audio.data<const float>();
    std::vector<double> window(m_frame_length);
    for (size_t i = 0; i < m_frame_length; ++i) {
        window[i] = 0.5 - 0.5 * std::cos(2.0 * PI * static_cast<double>(i) / static_cast<double>(m_frame_length));
    }

    std::vector<std::complex<double>> spectrum(m_fft_length);
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const size_t frame_start = frame * m_hop_length;
        const size_t frame_end = frame_start + m_frame_length;
        const bool valid = frame_end < left_padding + audio_length;
        mask_data[frame] = valid;
        if (!valid) {
            continue;
        }

        std::fill(spectrum.begin(), spectrum.end(), std::complex<double>{0.0, 0.0});
        for (size_t sample = 0; sample < m_frame_length; ++sample) {
            const size_t padded_index = frame_start + sample;
            if (padded_index >= left_padding) {
                spectrum[sample] = static_cast<double>(audio_data[padded_index - left_padding]) * window[sample];
            }
        }
        fft(spectrum);

        for (size_t mel_bin = 0; mel_bin < m_feature_size; ++mel_bin) {
            double mel_value = 0.0;
            for (size_t frequency_bin = 0; frequency_bin < fft_bins; ++frequency_bin) {
                mel_value +=
                    std::abs(spectrum[frequency_bin]) * m_mel_filters[frequency_bin * m_feature_size + mel_bin];
            }
            feature_data[frame * m_feature_size + mel_bin] = static_cast<float>(std::log(mel_value + m_mel_floor));
        }
    }

    return {std::move(input_features), std::move(input_features_mask)};
}

}  // namespace ov::genai
