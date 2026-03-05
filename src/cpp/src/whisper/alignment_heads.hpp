// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/runtime/core.hpp"

namespace ov::genai {

std::vector<ov::Tensor> get_whisper_alignments_heads_qks(ov::InferRequest& request,
                                                         const std::vector<std::pair<size_t, size_t>>& alignment_heads);

}  // namespace ov::genai
