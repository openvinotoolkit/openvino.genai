// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>

#include "openvino/core/model.hpp"
#include "openvino/genai/extensions.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "visual_language/vlm_models.hpp"

namespace ov::genai {
struct GGUFMultimodalModels {
    std::shared_ptr<ov::Model> language;
    VLMModels vlm;
    Tokenizer tokenizer;
    std::set<int64_t> stop_token_ids;

    /// Default generation config: the vocabulary's EOS and end-of-generation tokens.
    GenerationConfig generation_config() const {
        GenerationConfig config;
        config.set_eos_token_id(tokenizer.get_eos_token_id());
        config.stop_token_ids.insert(stop_token_ids.begin(), stop_token_ids.end());
        return config;
    }
};

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties,
                                          const ExtensionList& extensions = {});
}  // namespace ov::genai
