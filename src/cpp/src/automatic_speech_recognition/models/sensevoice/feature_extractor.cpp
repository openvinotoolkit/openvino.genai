// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/sensevoice/feature_extractor.hpp"

#include <fstream>
#include <sstream>
#include <string>

#include "openvino/core/except.hpp"

namespace {

std::vector<float> parse_bracketed_values(const std::string& line) {
    std::istringstream token_stream(line);
    std::vector<float> values;
    std::string token;
    bool inside = false;
    while (token_stream >> token) {
        if (token == "[") {
            inside = true;
        } else if (token == "]") {
            break;
        } else if (inside) {
            values.push_back(std::stof(token));
        }
    }
    return values;
}

std::pair<std::vector<float>, std::vector<float>> load_cmvn(const std::filesystem::path& cmvn_file) {
    std::ifstream stream(cmvn_file);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open CMVN statistics file '", cmvn_file.string(), "'");

    enum class Pending { none, shifts, scales };
    Pending pending = Pending::none;
    std::vector<float> shifts;
    std::vector<float> scales;

    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("<AddShift>") != std::string::npos) {
            pending = Pending::shifts;
        } else if (line.find("<Rescale>") != std::string::npos) {
            pending = Pending::scales;
        } else if (pending == Pending::shifts) {
            shifts = parse_bracketed_values(line);
            pending = Pending::none;
        } else if (pending == Pending::scales) {
            scales = parse_bracketed_values(line);
            pending = Pending::none;
        }
    }

    OPENVINO_ASSERT(!shifts.empty() && !scales.empty(),
                    "Malformed am.mvn: missing '<AddShift>' or '<Rescale>' CMVN statistics");
    return {std::move(shifts), std::move(scales)};
}

}  // namespace

namespace ov::genai {

SenseVoiceSmallFeatureExtractor::SenseVoiceSmallFeatureExtractor(const std::filesystem::path& cmvn_file) {
    auto [shifts, scales] = load_cmvn(cmvn_file);
    OPENVINO_ASSERT(shifts.size() == feature_size && scales.size() == feature_size,
                    "am.mvn CMVN statistics do not match the expected ",
                    feature_size,
                    "-dimensional post-LFR features");
    m_shifts = std::move(shifts);
    m_scales = std::move(scales);
}

ov::Tensor SenseVoiceSmallFeatureExtractor::extract(const std::vector<float>& audio) const {
    ov::Tensor features = m_fbank_lfr.extract(audio);
    const ov::Shape shape = features.get_shape();

    float* data = features.data<float>();
    const size_t frame_count = shape[0] * shape[1];
    for (size_t frame = 0; frame < frame_count; ++frame) {
        float* row = data + frame * feature_size;
        for (size_t dim = 0; dim < feature_size; ++dim) {
            row[dim] = (row[dim] + m_shifts[dim]) * m_scales[dim];
        }
    }
    return features;
}

}  // namespace ov::genai
