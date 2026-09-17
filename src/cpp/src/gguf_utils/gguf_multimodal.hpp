// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "visual_language/processor_config.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {
struct GGUFMultimodalModels {
    std::shared_ptr<ov::Model> language, text_embeddings, vision;
    Tokenizer tokenizer;
    VLMConfig config;
    ProcessorConfig processor;
};

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties);
}  // namespace ov::genai
