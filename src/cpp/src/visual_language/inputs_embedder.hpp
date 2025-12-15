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
#include "visual_language/cdpruner/cdpruner_config.hpp"

namespace ov::genai {
struct VLMPerfMetrics;
const static std::regex UNIVERSAL_PATTERN{R"(<ov_genai_image_(\d+)>)"};

struct NormalizedPrompt {
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
                                 const std::vector<size_t>& videos_sequence = {},
                                 const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {});

    // compute input embedding and token_type_ids
    std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(const std::string& prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {});

    std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        const std::vector<ov::genai::EncodedVideo>& videos,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {});

    bool has_token_type_ids() const;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images);

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos);

    // compute position ids for language model input
    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size);

    void set_position_ids(const ov::Tensor& position_ids);

    void set_rope_delta(int64_t rope_delta);

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta);

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

    // set CDPruner setting
    virtual void set_visual_token_pruning_config(size_t pruning_ratio,
                                                 float relevance_weight);
    virtual NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const;

    virtual NormalizedPrompt normalize_prompt(
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
        // position ids
        ov::Tensor m_position_ids;
        int64_t m_rope_delta = 0;
        virtual ~IInputsEmbedder() = default;

    public:
        virtual ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) = 0;

        virtual ov::Tensor get_inputs_embeds(const std::string& prompt,
                                             const std::vector<ov::genai::EncodedImage>& images,
                                             const std::vector<ov::genai::EncodedVideo>& videos,
                                             ov::genai::VLMPerfMetrics& metrics,
                                             bool recalculate_merged_embeddings = true,
                                             const std::vector<size_t>& image_sequence = {},
                                             const std::vector<size_t>& videos_sequence = {},
                                             const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {});

        virtual std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {});
        virtual std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(
            const std::string& prompt,
            const std::vector<ov::genai::EncodedImage>& images,
            const std::vector<ov::genai::EncodedVideo>& videos,
            ov::genai::VLMPerfMetrics& metrics,
            bool recalculate_merged_embeddings = true,
            const std::vector<size_t>& image_sequence = {},
            const std::vector<size_t>& videos_sequence = {},
            const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {});

        virtual bool has_token_type_ids() const;

        virtual std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images);

        virtual std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos);

        virtual std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size);
        
        void set_position_ids(const ov::Tensor& position_ids) {
            m_position_ids = position_ids;
        }

        void set_rope_delta(int64_t rope_delta) {
            m_rope_delta = rope_delta;
        }

        virtual std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta);

        EmbeddingsModel::Ptr get_embedding_model() const {
            return m_embedding;
        }

        Tokenizer get_tokenizer() const {
            return m_tokenizer;
        }

        virtual void set_visual_token_pruning_config(size_t pruning_ratio, float relevance_weight) {
            if (!m_vision_encoder)
                return;
            auto pruner_config = m_vision_encoder->get_pruning_config();
            pruner_config->pruning_ratio = pruning_ratio;
            pruner_config->relevance_weight = relevance_weight;
            m_vision_encoder->set_pruning_config(pruner_config.value());
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

        virtual NormalizedPrompt normalize_prompt(
            const std::string& prompt,
            size_t base_id,
            const std::vector<EncodedImage>& images
        ) const = 0;
        virtual NormalizedPrompt normalize_prompt(
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

        /**
         * @brief Result structure for CDPruner visual token pruning pipeline.
         * Contains all necessary information about the pruning operation and its results.
         */
        struct PruningResult {
            bool is_pruned = false;                                ///< Whether pruning was actually applied
            size_t original_visual_tokens = 0;                     ///< Original number of visual tokens before pruning
            size_t pruned_visual_tokens = 0;                       ///< Number of visual tokens after pruning
            ov::Tensor pruned_embeddings;                          ///< Pruned visual embeddings tensor
            ov::Tensor pruned_input_ids;                           ///< Input IDs with pruned visual tokens removed
            std::vector<std::vector<bool>> keep_flags_per_region;  ///< Keep flags for each visual region
        };

        /**
         * @brief Extract text features for CDPruner relevance calculation.
         * Default implementation returns empty tensor. Models supporting CDPruner should override.
         */
        virtual ov::Tensor extract_text_features_for_pruning(const ov::Tensor& text_embeds,
                                                             const ov::Tensor& input_ids,
                                                             int64_t vision_start_token_id,
                                                             int64_t vision_end_token_id) const;

        /**
         * @brief Convert visual features to CDPruner format.
         * Default implementation returns single tensor in vector. Models supporting CDPruner should override.
         * @param vision_embeds The visual embeddings to convert
         * @param chunk_count Number of chunks for processing (for frame-based chunking)
         * @param images_grid_thw Grid information [T, H, W] for each image (empty for default behavior)
         * @return Vector of visual feature tensors
         */
        virtual std::vector<ov::Tensor> convert_visual_features_for_pruning(
            const ov::Tensor& vision_embeds,
            size_t chunk_count,
            const std::vector<std::array<size_t, 3>>& images_grid_thw = {}) const;

        /**
         * @brief Adjust position IDs after visual token pruning.
         * Default implementation does nothing. Models supporting CDPruner should override.
         * @param position_ids_inout The position IDs to adjust (modified in-place)
         * @param input_ids The input token IDs
         * @param vision_start_token_id Vision region start token ID
         * @param image_pad_token_id Image padding token ID
         * @param images_grid_thw Grid dimensions for each image
         * @param images_sequence Image sequence
         * @param keep_flags_per_region_out Output parameter for keep flags
         */
        virtual void adjust_position_ids_after_pruning(ov::Tensor& position_ids_inout,
                                                       const ov::Tensor& input_ids,
                                                       int64_t vision_start_token_id,
                                                       int64_t image_pad_token_id,
                                                       const std::vector<std::array<size_t, 3>>& images_grid_thw,
                                                       const std::vector<size_t>& images_sequence,
                                                       std::vector<std::vector<bool>>& keep_flags_per_region_out) const;

        /**
         * @brief Merge text and visual embeddings after pruning.
         * Default implementation throws error. Models supporting CDPruner must override.
         */
        virtual ov::Tensor merge_text_visual_embeddings_with_pruning(const ov::Tensor& text_embeds,
                                                                     const ov::Tensor& pruned_vision_embeds,
                                                                     const ov::Tensor& adjusted_position_ids,
                                                                     int64_t image_pad_token_id) const;

        /**
         * @brief Generate pruned input_ids based on keep_flags.
         * Default implementation returns input as-is. Models supporting CDPruner should override.
         */
        virtual ov::Tensor generate_pruned_input_ids(const ov::Tensor& input_ids,
                                                     const std::vector<std::vector<bool>>& keep_flags_per_region,
                                                     int64_t image_pad_token_id,
                                                     int64_t vision_start_token_id,
                                                     int64_t vision_end_token_id) const;

        /**
         * @brief Check if CDPruner should be active for current configuration.
         * @param images Vector of encoded images (empty check)
         * @return true if CDPruner is available, enabled, and has images to process
         */
        bool is_cdpruner_active(const std::vector<ov::genai::EncodedImage>& images) const;

        /**
         * @brief Execute the full CDPruner pipeline (Template Method).
         * This method orchestrates the entire pruning workflow:
         * 1. Extract text features
         * 2. Convert visual features
         * 3. Apply CDPruner
         * 4. Adjust position IDs
         * 5. Generate pruned input_ids
         *
         * Implementation in base class calls virtual functions that derived classes can override.
         */
        virtual PruningResult execute_cdpruner_pipeline(const ov::Tensor& input_ids,
                                                        const ov::Tensor& text_embeds,
                                                        const ov::Tensor& merged_visual_embeddings,
                                                        const std::vector<ov::genai::EncodedImage>& images,
                                                        const std::vector<std::array<size_t, 3>>& images_grid_thw,
                                                        const std::vector<size_t>& images_sequence,
                                                        int64_t image_pad_token_id,
                                                        int64_t vision_start_token_id,
                                                        int64_t vision_end_token_id);
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
