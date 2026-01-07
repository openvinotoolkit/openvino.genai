// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "visual_language/vlm_config.hpp"
#include "openvino/genai/tokenizer.hpp"


namespace ov {
namespace genai {
namespace module {

class EmbeddingMergerModule : public IBaseModule {
    DeclareModuleConstructor(EmbeddingMergerModule);

private:
    bool initialize();
    Tokenizer m_tokenizer;
    VLMConfig m_vlm_config;
    std::map<std::string, int64_t> m_vision_token_ids;
    void encode_vision_placeholder_tokens();
    ov::Tensor merge_text_and_video_image_embeddings(const ov::Tensor& input_ids,
                                                     const ov::Tensor& text_embeds,
                                                     const ov::Tensor& processed_image_embeds,
                                                     const ov::Tensor& processed_video_embeds,
                                                     const int64_t image_pad_token_id,
                                                     const int64_t video_pad_token_id);
};

REGISTER_MODULE_CONFIG(EmbeddingMergerModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
