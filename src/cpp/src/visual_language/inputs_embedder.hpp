// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "utils.hpp"
#include "lm_encoding.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "visual_language/vlm_config.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/vision_encoder.hpp"

namespace ov::genai {
struct VLMPerfMetrics;

class InputsEmbedder {
public:
    InputsEmbedder(const std::filesystem::path& model_dir,
                   const std::string& device,
                   const ov::AnyMap device_config);

    InputsEmbedder(const ModelsMap& models_map,
                   const Tokenizer& tokenizer,
                   const std::filesystem::path& config_dir_path,
                   const std::string& device,
                   const ov::AnyMap device_config);

    // compute input embedding for prompt and multiple images
    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics);

    // compute position ids for language model input
    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size);

    // returns embedding model which converts token_id(s) to embedding vectors
    EmbeddingsModel get_embedding_model() const;

    // returns tokenizer
    Tokenizer get_tokenizer() const;

    // get reflection of tokens contained in the kv cache
    KVCacheState& get_kv_cache_state();

    // returns amount of elements, which need to remove from the end of the KV cache
    size_t get_num_tokens_to_remove_from_hist() const;

    // starts chat and adds optional system_message to chat history
    void start_chat(const std::string& system_message);

    // adds currently generated text to chat history
    void update_chat_history(const std::string& decoded_results);

    // set the apply_chat_template flag, which determines whether chat template should be applied for non-chat scenarios
    void set_apply_chat_template_status(bool apply_chat_template);

    // finishes chat and clears a chat history
    void finish_chat();

private:
    class IInputsEmbedder {
    protected:
        // VLM config
        VLMConfig m_vlm_config;
        // An encoder to infer embeddings of an image.
        VisionEncoder::Ptr m_vision_encoder;
        // A model to compute token embeddings.
        // Input shape: [N, conversation length].
        // Output shape: [1, conversation length, hidden_size].
        EmbeddingsModel m_embedding;
        // A tokenizer encoding a prompt.
        Tokenizer m_tokenizer;
        // True if chat mode is activated to save conversation
        // history between generate() calls.
        bool m_is_chat_conversation = false;
        // Chat history
        ChatHistory m_history;
        // If sequence contains some symbols, which could be ambiguous encoded by tokenizer, we need to trim kv cache
        // If we use beam search sampling with chat mode we need to remove last answer of the model from kv cache and add best answer to history
        // so, let's keep info about amount of tokens to trim from kv cache and amount of tokens to keep in history
        ov::genai::KVCacheTrimManager m_kv_history_trim_manager = {0, 2};
        // True if chat template should be applied for non-chat scenario
        bool m_apply_chat_template = true;
        // Finish reason of last generation for chat scenario
        ov::genai::GenerationStatus m_chat_generation_finish_status = ov::genai::GenerationStatus::RUNNING;
        // reflection of tokens contained in the kv cache
        KVCacheState m_kv_cache_state;
    public:
        virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::Tensor>& images, ov::genai::VLMPerfMetrics& metrics) = 0;

        virtual std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size);

        EmbeddingsModel get_embedding_model() const {
            return m_embedding;
        }

        Tokenizer get_tokenizer() const {
            return m_tokenizer;
        }

        KVCacheState& get_kv_cache_state() {
            return m_kv_cache_state;
        }

        size_t get_num_tokens_to_remove_from_hist() const {
            return m_kv_history_trim_manager.num_tokens_to_trim;
        }

        void set_apply_chat_template_status(bool apply_chat_template) {
            m_apply_chat_template = apply_chat_template;
        }

        virtual void start_chat(const std::string& system_message);

        void update_chat_history(const std::string& decoded_results);

        virtual void finish_chat();

    protected:
        IInputsEmbedder(
            const VLMConfig& vlm_config,
            const std::filesystem::path& model_dir,
            const std::string& device,
            const ov::AnyMap device_config);

        IInputsEmbedder(
            const VLMConfig& vlm_config,
            const ModelsMap& models_map,
            const Tokenizer& tokenizer,
            const std::filesystem::path& config_dir_path,
            const std::string& device,
            const ov::AnyMap device_config);

        ov::Tensor apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics);

        ov::Tensor update_history(const ov::Tensor& new_chat_tokens);

        ov::Tensor get_encoded_input_ids(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics);

        /**
        * @brief Converts a vector of batched images ([NHWC]) into a vector of individual image tensors ([1HWC]).
        *
        * @param images A vector of tensors representing the images. Each tensor can have a shape of either [NHWC] or [HWC].
        * @return A vector of tensors where each tensor represents a single image with a shape of [1, H, W, C].
        */
        std::vector<ov::Tensor> to_single_image_tensors(const std::vector<ov::Tensor>& images);
    };

    std::shared_ptr<IInputsEmbedder> m_impl;

    friend class InputsEmbedderMiniCPM;
    friend class InputsEmbedderLLaVA;
    friend class InputsEmbedderLLaVANext;
    friend class InputsEmbedderInternVLChat;
    friend class InputsEmbedderPhi3V;
    friend class InputsEmbedderQwen2VL;
};

} // namespace ov::genai
