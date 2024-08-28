// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <vector>

namespace utils {
namespace audio {
std::vector<float> read_wav(const std::string& filename);
}  // namespace audio
}  // namespace utils
