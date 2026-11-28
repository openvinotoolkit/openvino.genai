// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/funasr_frontend/cmvn.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

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
                    "Malformed CMVN statistics file '",
                    cmvn_file.string(),
                    "': missing '<AddShift>' or '<Rescale>' statistics");
    return {std::move(shifts), std::move(scales)};
}

}  // namespace

namespace ov::genai::funasr_frontend {

CMVN::CMVN(const std::filesystem::path& cmvn_file, const size_t feature_size) {
    auto [shifts, scales] = load_cmvn(cmvn_file);
    OPENVINO_ASSERT(shifts.size() == feature_size && scales.size() == feature_size,
                    "CMVN statistics do not match the expected ",
                    feature_size,
                    "-dimensional features");
    m_shifts = std::move(shifts);
    m_scales = std::move(scales);
}

void CMVN::apply(float* rows, const size_t row_count) const {
    const size_t feature_size = m_shifts.size();
    for (size_t frame = 0; frame < row_count; ++frame) {
        float* row = rows + frame * feature_size;
        for (size_t dim = 0; dim < feature_size; ++dim) {
            row[dim] = (row[dim] + m_shifts[dim]) * m_scales[dim];
        }
    }
}

}  // namespace ov::genai::funasr_frontend
