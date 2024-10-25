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
#include "visual_language/embedding_model.hpp"

namespace ov::genai {

class InputsEmbedder {
public:
    InputsEmbedder(const VLMConfig& vlm_config,
                   const std::filesystem::path& model_dir,
                   const std::string& device,
                   const ov::AnyMap device_config);

    // compute input embedding for prompt and multiple images
    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images);

    // returns embedding model which converts token_id(s) to embedding vectors
    std::shared_ptr<EmbeddingsModel> get_embedding_model() const;

    // returns tokenizer
    Tokenizer get_tokenizer() const;

    // starts chat and adds optional system_message to chat history
    void start_chat(const std::string& system_message);
    // adds currently generated text to chat history
    void update_chat_history(const std::string& decoded_results);
    // finishes chat and clears a chat history 
    void finish_chat();
private:
    class IInputsEmbedder;
    std::shared_ptr<IInputsEmbedder> m_impl;

    friend class InputsEmbedderMiniCPM;
    friend class InputsEmbedderLLaVA;
    friend class InputsEmbedderLLaVANext;
    friend class InputsEmbedderInternVLChat;
};

} // namespace ov::genai
