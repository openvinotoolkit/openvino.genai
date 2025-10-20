// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <regex>

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
const static std::regex UNIVERSAL_PATTERN{R"(<ov_genai_image_(\d+)>)"};

struct NormlizedPrompt {
    std::string unified_prompt;
    std::vector<size_t> images_sequence;
    std::vector<size_t> videos_sequence;
};

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
    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {});

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                            const std::vector<ov::genai::EncodedImage>& images,
                            const std::vector<ov::genai::EncodedVideo>& videos,
                            ov::genai::VLMPerfMetrics& metrics,
                            bool recalculate_merged_embeddings = true,
                            const std::vector<size_t>& image_sequence = {},
                            const std::vector<size_t>& videos_sequence = {});

    // compute input embedding and token_type_ids
    std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(const std::string& prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {});

    bool has_token_type_ids() const;
    
    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images);

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos);

    // compute position ids for language model input
    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size);

    // returns embedding model which converts token_id(s) to embedding vectors
    EmbeddingsModel::Ptr get_embedding_model() const;

    // returns tokenizer
    Tokenizer get_tokenizer() const;

    // get reflection of tokens contained in the kv cache
    utils::KVCacheState& get_kv_cache_state();

    // starts chat and adds optional system_message to chat history
    void start_chat(const std::string& system_message);

    // adds currently generated text to chat history
    void update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status);

    // set the apply_chat_template flag, which determines whether chat template should be applied for non-chat scenarios
    void set_apply_chat_template_status(bool apply_chat_template);

    // finishes chat and clears a chat history 
    void finish_chat();

    virtual NormlizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const;

    virtual NormlizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_image_id,
        size_t base_video_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos) const;


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
        EmbeddingsModel::Ptr m_embedding;
        // A tokenizer encoding a prompt.
        Tokenizer m_tokenizer;
        // True if chat mode is activated to save conversation
        // history between generate() calls.
        bool m_is_chat_conversation = false;
        // Chat history
        // True if chat template should be applied for non-chat scenario
        bool m_apply_chat_template = true;
        // Finish reason of last generation for chat scenario
        ov::genai::GenerationStatus m_chat_generation_finish_status = ov::genai::GenerationStatus::RUNNING;
        // reflection of tokens contained in the kv cache
        utils::KVCacheState m_kv_cache_state;
        // length of attention_mask/kv cache at the beginning of generation()
        size_t m_prev_hist_length = 0;
        // True if tokenizer should add special tokens
        bool m_add_special_tokens = true;
        // True, if m_add_special_tokens was set, otherwise default behaviour is used
        bool m_add_special_tokens_is_set = false;
        virtual ~IInputsEmbedder() = default;

    public:
        virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) = 0;

        virtual ov::Tensor get_inputs_embeds(const std::string& prompt,
                                             const std::vector<ov::genai::EncodedImage>& images,
                                             const std::vector<ov::genai::EncodedVideo>& videos,
                                             ov::genai::VLMPerfMetrics& metrics,
                                             bool recalculate_merged_embeddings = true,
                                             const std::vector<size_t>& image_sequence = {},
                                             const std::vector<size_t>& videos_sequence = {});

        virtual std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {});

        virtual bool has_token_type_ids() const;

        virtual std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images);

        virtual std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos);
    
        virtual std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size);
    
        EmbeddingsModel::Ptr get_embedding_model() const {
            return m_embedding;
        }
    
        Tokenizer get_tokenizer() const {
            return m_tokenizer;
        }
    
        utils::KVCacheState& get_kv_cache_state() {
            return m_kv_cache_state;
        }
    
        void set_apply_chat_template_status(bool apply_chat_template) {
            m_apply_chat_template = apply_chat_template;
        }

        void set_add_special_tokens(bool value) {
            m_add_special_tokens = value;
            m_add_special_tokens_is_set = true;
        }
    
        virtual void start_chat(const std::string& system_message);
    
        virtual void update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status);
    
        virtual void finish_chat();

        virtual NormlizedPrompt normalize_prompt(
            const std::string& prompt,
            size_t base_id,
            const std::vector<EncodedImage>& images
        ) const = 0;
    
        virtual NormlizedPrompt normalize_prompt(
            const std::string& prompt,
            size_t base_image_id,
            size_t base_video_id,
            const std::vector<EncodedImage>& images,
            const std::vector<EncodedVideo>& videos) const;
    
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
    
        virtual ov::Tensor apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics);
    
        ov::Tensor update_history(const ov::Tensor& new_chat_tokens);

        ov::Tensor get_encoded_input_ids(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics);

        std::pair<std::string, std::vector<size_t>> normalize(
            const std::string& prompt,
            const std::string& native_tag,
            const std::string& automatic_tag,
            size_t base_id,
            size_t n_images
        ) const;

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
    friend class InputsEmbedderNanoLLaVA;
    friend class InputsEmbedderLLaVANext;
    friend class InputsEmbedderInternVLChat;
    friend class InputsEmbedderPhi3V;
    friend class InputsEmbedderPhi4MM;
    friend class InputsEmbedderQwen2VL;
    friend class InputsEmbedderQwen2_5_VL;
    friend class InputsEmbedderGemma3;
};

template <typename Func>
std::pair<std::string, std::vector<size_t>> universal_to_native(
    const std::string& prompt,
    const Func& write_native
) {
    std::stringstream stream;
    std::vector<size_t> image_sequence;
    std::smatch match;
    std::regex_search(prompt, match, UNIVERSAL_PATTERN);
    auto search_begin = prompt.begin();
    while (!match.empty()) {
        stream.write(&*search_begin, match.position());
        image_sequence.push_back(std::stoul(match.str(1)));
        write_native(stream, image_sequence.back());
        search_begin = match.suffix().first;
        std::regex_search(search_begin, prompt.end(), match, UNIVERSAL_PATTERN);
    }
    stream.write(&*search_begin, prompt.end() - search_begin);
    return {stream.str(), std::move(image_sequence)};
}

void verify_ids(const std::vector<size_t>& image_ids, size_t base_id, size_t n_images);

/// @brief 1. Verify native and universal tags aren't mixed.
/// 2. Replace universal tags with native and save image order.
/// 3. If there were no universal tags, restore image order from native.
/// 4. If no tags were found, prepend native tags and assume incremental
/// ordering.
/// @param automatic_tag MiniCPM-V-2_6 inserts
/// <image>./</image>\n per image but it only replaces
/// <image>./</image> leaving \n untouched.
/// automatic_tag allows to handle this by being separated
/// from native_tag param.
std::pair<std::string, std::vector<size_t>> normalize_prompt(
    const std::string& prompt,
    const std::string& native_tag,
    const std::string& automatic_tag,
    size_t base_id,
    size_t n_images
);

} // namespace ov::genai
