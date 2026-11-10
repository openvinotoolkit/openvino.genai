// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <vector>

namespace ov::genai {

using AudiosView = std::vector<std::reference_wrapper<const std::vector<float>>>;

}  // namespace ov::genai
