// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

class VisionEncoderGemma4 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;

private:
    EncodedImage encode_with_config(const ov::Tensor& image, const ProcessorConfig& config);
};

class InputsEmbedderGemma4 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderGemma4(const VLMConfig& vlm_config,
                         const std::filesystem::path& model_dir,
                         const std::string& device,
                         const ov::AnyMap device_config);

    InputsEmbedderGemma4(const VLMConfig& vlm_config,
                         const ModelsMap& models_map,
                         const Tokenizer& tokenizer,
                         const std::filesystem::path& config_dir_path,
                         const std::string& device,
                         const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {}) override;

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 const std::vector<ov::genai::EncodedVideo>& videos,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {},
                                 const std::vector<size_t>& videos_sequence = {},
                                 const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {}) override;

    std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        const std::vector<ov::genai::EncodedVideo>& videos,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    bool has_token_type_ids() const override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    std::vector<ov::genai::EncodedVideo> encode_videos(
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata = {}) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_image_id,
        size_t base_video_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos
    ) const override;

    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override;

    std::function<ov::Tensor(const ov::Tensor& new_input_ids)> get_per_layer_embeddings_callback() override {
        if (!has_per_layer_embeddings()) {
            return nullptr;
        }

        return [this](const ov::Tensor& input_ids) {
            return get_per_layer_embeddings(input_ids);
        };
    }

private:
    // Per-layer text embeddings model (Gemma4-specific)
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_per_layer_embeddings_requests = nullptr;

    // Extra inputs to pass to the language model
    std::unordered_map<std::string, ov::Tensor> m_lm_extra_inputs;

    void expand_video_tags_in_prompt(
        std::string& unified_prompt,
        const std::vector<EncodedVideo>& encoded_videos,
        const std::vector<size_t>& videos_sequence,
        size_t video_base_id
    ) const;

    ov::Tensor get_per_layer_embeddings(const ov::Tensor& input_ids);

    bool has_per_layer_embeddings() const {
        return m_vlm_config.hidden_size_per_layer_input > 0;
    }

    /// @brief Compute merged text+image embeddings together with the encoded input_ids.
    /// Shared implementation behind get_inputs_embeds() and get_inputs_embeds_with_token_type_ids().
    /// @return A pair of (inputs_embeds, input_ids).
    std::pair<ov::Tensor, ov::Tensor> compute_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos,
        VLMPerfMetrics& metrics,
        const std::vector<size_t>& images_sequence,
        const std::vector<size_t>& videos_sequence
    );

    ov::Tensor get_token_type_ids(const ov::Tensor& input_ids);

    int64_t m_image_token_id = -1;
    int64_t m_video_token_id = -1;
    std::once_flag m_vision_token_ids_once_flag;

    void encode_vision_token_ids();
};

}  // namespace ov::genai
