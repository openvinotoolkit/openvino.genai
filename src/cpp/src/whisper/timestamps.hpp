// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "whisper.hpp"

namespace ov {
namespace genai {

std::pair<std::vector<int64_t>, std::vector<ov::genai::Segment>> extract_segments(
    const std::vector<int64_t>& tokens,
    const ov::genai::WhisperGenerationConfig& config,
    const float time_precision);

}  // namespace genai
}  // namespace ov
