// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "module_genai/transformer_config.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "tokenizer/tokenizer_impl.hpp"


namespace ov {
namespace genai {
namespace module {

class ClipTextEncoderModule : public IBaseModule {
    DeclareModuleConstructor(ClipTextEncoderModule);

private:
    bool initialize();
    bool do_classifier_free_guidance(float guidance_scale);
    DiffusionModelType m_model_type;
    TransformerConfig m_encoder_config;
    ov::InferRequest m_request;
    std::shared_ptr<Tokenizer::TokenizerImpl> m_tokenizer_impl;
    std::shared_ptr<minja::chat_template> m_minja_template;
    std::pair<ov::Tensor, ov::Tensor> run(
        const std::vector<std::string>& prompts,
        const std::vector<std::string>& negative_prompts,
        const ImageGenerationConfig &generation_config);
    ov::Tensor encode_prompt(const std::vector<std::string>& prompts, const ImageGenerationConfig &generation_config);
};

REGISTER_MODULE_CONFIG(ClipTextEncoderModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
