// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <vector>

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace audio {
bool OPENVINO_GENAI_EXPORTS read_wav(const std::string& fname,
                                     std::vector<float>& pcmf32,
                                     std::vector<std::vector<float>>& pcmf32s,
                                     bool stereo = false);
}  // namespace audio
}  // namespace utils
}  // namespace genai
}  // namespace ov
