// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "openvino/genai/classification/classification_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

ClassificationActivation default_activation(const std::string& problem_type, size_t num_labels);

/// Applies the activation to each row of logits [batch, num_labels]; returns scores[batch][num_labels].
std::vector<std::vector<float>> apply_activation(const ov::Tensor& logits, ClassificationActivation activation_type);

}  // namespace ov::genai
