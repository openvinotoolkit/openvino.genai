// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

struct HiddenStatesPayload {
    // Outer vector: per return sequence. Inner vector: one tensor per generation step.
    std::vector<std::vector<ov::Tensor>> hidden_states;
    std::vector<std::vector<ov::Tensor>> intermediate_hidden_states;
    // Full prompt token IDs (needed for talker input construction).
    std::vector<int64_t> prompt_ids;
};

}  // namespace ov::genai
