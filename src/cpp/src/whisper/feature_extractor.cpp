// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper/feature_extractor.hpp"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <openvino/core/except.hpp>

#include "json_utils.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

std::vector<float> WhisperFeatures::get_data_with_offset(const size_t frame_offset, const size_t min_frames) {
    OPENVINO_ASSERT(n_frames > frame_offset);

    size_t copy_size = std::min(n_frames - frame_offset, min_frames);
    std::vector<float> offset_data;

    for (size_t i = 0; i < feature_size; i++) {
        size_t offset = frame_offset + (i * n_frames);
        std::copy(data.begin() + offset, data.begin() + offset + copy_size, std::back_inserter(offset_data));
        if (copy_size < min_frames) {
            std::fill_n(std::back_inserter(offset_data), min_frames - copy_size, 0);
        }
    }

    return offset_data;
}

WhisperFeatureExtractor::WhisperFeatureExtractor(const std::filesystem::path& preprocessor_json_path) {
    init_parameters(preprocessor_json_path);
    // emplace after init_parameters so the extractor sees JSON-overridden config values.
    m_extractor.emplace(feature_size, sampling_rate, n_fft, hop_length);
}

void WhisperFeatureExtractor::init_parameters(const std::filesystem::path& preprocessor_json_path) {
    // preprocessor_config.json not found. Skip parameters initialization from file, use defaults.
    if (!std::filesystem::exists(preprocessor_json_path)) {
        return;
    }

    using ov::genai::utils::read_json_param;

    std::ifstream f(preprocessor_json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", preprocessor_json_path, "' with preprocessor config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "feature_size", feature_size);
    read_json_param(data, "sampling_rate", sampling_rate);
    read_json_param(data, "hop_length", hop_length);
    read_json_param(data, "n_fft", n_fft);
    read_json_param(data, "chunk_length", chunk_length);
    read_json_param(data, "n_samples", n_samples);
    read_json_param(data, "nb_max_frames", nb_max_frames);
};

WhisperFeatures WhisperFeatureExtractor::extract(const std::vector<float>& raw_speech) {
    WhisperFeatures features;
    features.feature_size = feature_size;
    // sampling_rate * chunk_length respects config-driven chunk_length (default 30 matches
    // the prior hardcoded sampling_rate * 30 for byte-for-byte identity on default configs).
    features.data =
        m_extractor->extract(raw_speech, features.n_frames, sampling_rate * chunk_length, &features.n_active_frames);
    return features;
}

}  // namespace genai
}  // namespace ov
