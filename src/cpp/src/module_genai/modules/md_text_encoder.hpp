// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "tokenizer/tokenizer_impl.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov {
namespace genai {
namespace module {

class TextEncoderModule : public IBaseModule {
    DeclareModuleConstructor(TextEncoderModule);

private:
    std::shared_ptr<Tokenizer::TokenizerImpl> m_tokenizer_impl;
    ov::AnyMap m_tokenization_params = {};
    VLMConfig m_vlm_config;
    ProcessorConfig m_processor_config;
    size_t m_merge_length;

    bool initialize();
    std::pair<TokenizedInputs, std::vector<int>> run(const std::vector<std::string>& prompts, 
                        const std::vector<ov::Tensor>& encoded_images,
                        const std::vector<std::vector<int>>& source_sizes,
                        bool has_encoded_image = false);
    TokenizedInputs run(const std::vector<std::string>& prompts, std::optional<ov::Tensor>& grid_thw);
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_image_id,
                                      size_t base_video_id,
                                      const std::vector<ov::Tensor>& encoded_images,
                                      const std::vector<ov::Tensor>& encoded_videos,
                                      const std::vector<std::vector<int>>& source_sizes);
    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_image_id,
                                      size_t base_video_id,
                                      const ov::Tensor& grid_thw);
    std::pair<std::string, std::vector<size_t>> normalize(
            const std::string& prompt,
            const std::string& native_tag,
            const std::string& automatic_tag,
            size_t base_id,
            size_t n_images);
    size_t calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const;

    ov::Tensor calc_thw(const std::vector<std::vector<int>>& source_sizes);
};

REGISTER_MODULE_CONFIG(TextEncoderModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
