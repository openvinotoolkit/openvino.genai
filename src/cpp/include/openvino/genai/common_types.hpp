// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include "openvino/core/core.hpp"

namespace ov {
namespace genai {

/// @brief A map of models for VLMPipeline constructor. 
/// Key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
/// and value is a pair of model IR as string and weights as tensor.
using ModelsMap = std::map<std::string, std::pair<std::string, ov::Tensor>>;

}
}