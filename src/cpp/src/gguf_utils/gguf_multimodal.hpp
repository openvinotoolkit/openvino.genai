// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>

#include "openvino/core/model.hpp"
#include "openvino/genai/extensions.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "visual_language/vlm_models.hpp"

namespace ov::genai {
class InputsEmbedder;
struct GGUFMultimodalModels {
    std::shared_ptr<ov::Model> language, audio;
    VLMModels vlm;
    Tokenizer tokenizer;
    std::set<int64_t> stop_token_ids;
};

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties,
                                          const ExtensionList& extensions = {});
void attach_gguf_audio_encoder(InputsEmbedder& embedder,
                               const GGUFMultimodalModels& models,
                               const std::string& device,
                               const ov::AnyMap& properties);
}  // namespace ov::genai
