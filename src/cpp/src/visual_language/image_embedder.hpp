// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

class InputsEmbedder {
public:
    InputsEmbedder(const VLMConfig& vlm_config,
                   const std::filesystem::path& model_dir,
                   const std::string& device,
                   const ov::AnyMap device_config,
                   ov::InferRequest embedding,
                   // looks like dirty parameters
                   Tokenizer tokenizer,
                   bool& is_chat_conversation,
                   ChatHistory& history,
                   std::string& templated_chat_history);

    // compute input embedding for prompt and multiple images
    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images);

private:
    class IInputsEmbedder;
    std::shared_ptr<IInputsEmbedder> m_impl;

    friend class InputsEmbedderMiniCPM;
    friend class InputsEmbedderLLaVA;
    friend class InputsEmbedderLLaVANext;
};

} // namespace ov::genai
