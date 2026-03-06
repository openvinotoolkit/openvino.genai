// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/modules/models/whisper/feature_extraction_whisper.hpp"

#include <algorithm>
#include <cstring>
#include <random>

#include <openvino/core/except.hpp>

namespace ov::genai::module {

WhisperFeatureExtractor::WhisperFeatureExtractor(const std::filesystem::path& model_path)
	: m_impl(model_path / "preprocessor_config.json") {}

WhisperFeatureExtractorOutput WhisperFeatureExtractor::extract(const ov::Tensor& raw_speech,
                                                               std::optional<size_t> sampling_rate,
                                                               bool return_attention_mask,
                                                               float dither) {
    if (sampling_rate.has_value() && sampling_rate.value() != m_impl.sampling_rate) {
        OPENVINO_THROW("WhisperFeatureExtractor: expected sampling_rate=",
                       m_impl.sampling_rate,
                       ", got ",
                       sampling_rate.value());
    }

    const ov::Shape shape = raw_speech.get_shape();
    if (!(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1))) {
        OPENVINO_THROW("WhisperFeatureExtractor: expected raw_speech shape [L] or [1, L], got ", shape);
    }

    const size_t input_samples = (shape.size() == 2) ? shape[1] : shape[0];
    const size_t valid_samples = input_samples;

    std::vector<float> waveform;
    waveform.reserve(valid_samples);
    waveform.insert(waveform.end(), raw_speech.data<float>(), raw_speech.data<float>() + valid_samples);

    if (dither != 0.0f) {
        std::mt19937 rng{std::random_device{}()};
        std::normal_distribution<float> normal{0.0f, 1.0f};
        for (auto& sample : waveform) {
            sample += dither * normal(rng);
        }
    }

    ov::genai::WhisperFeatures features = m_impl.extract(waveform, /*pad_to_30s*/ false);

    ov::Tensor input_features(ov::element::f32, ov::Shape{features.feature_size, features.n_frames});
    std::memcpy(input_features.data(), features.data.data(), features.data.size() * sizeof(float));

    WhisperFeatureExtractorOutput out{std::move(input_features), std::nullopt, 0};

    // torch.stft(center=true) with reflect pad and dropping the last frame yields (L // hop_length) frames.
    out.num_frames = features.n_frames;

    if (return_attention_mask) {
        ov::Tensor mask(ov::element::i32, ov::Shape{features.n_frames});
        auto* mask_data = mask.data<int32_t>();
        std::fill(mask_data, mask_data + features.n_frames, 1);
        out.attention_mask = std::move(mask);
    }

    return out;
}

size_t WhisperFeatureExtractor::feature_size() const noexcept {
	return m_impl.feature_size;
}

size_t WhisperFeatureExtractor::sampling_rate() const noexcept {
	return m_impl.sampling_rate;
}

size_t WhisperFeatureExtractor::hop_length() const noexcept {
	return m_impl.hop_length;
}

size_t WhisperFeatureExtractor::n_fft() const noexcept {
	return m_impl.n_fft;
}

size_t WhisperFeatureExtractor::n_samples() const noexcept {
	return m_impl.n_samples;
}


} // namespace ov::genai::module
