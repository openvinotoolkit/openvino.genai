// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "openvino/genai/visual_language/processor.hpp"

namespace ov::genai {

class InputsEmbedder;

/**
 * Internal bridge to expose shared InputsEmbedder from VLMProcessor to VLMPipeline for
 * vision/embeddings models to be loaded once and reused
 */
std::shared_ptr<InputsEmbedder> get_shared_inputs_embedder(const VLMProcessor& processor);

}  // namespace ov::genai
