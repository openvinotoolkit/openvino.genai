// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/whisper_pipeline.hpp"

namespace utils {
namespace audio {
ov::genai::RawSpeechInput read_wav(const std::string& filename);
}  // namespace audio
}  // namespace utils
