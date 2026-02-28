// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <array>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace videochat_flash_utils {
    ov::Tensor transpose_video_features(const ov::Tensor& src_tensor, const size_t mm_local_num_frames);
    ov::Tensor transpose_image_features(const ov::Tensor& src_tensor);
    ov::Tensor preprocess(const ov::Tensor& input_nhwc_u8,
                                    const size_t target_h = 224,
                                    const size_t target_w = 224,
                                    const std::array<float, 3>& image_mean = {0.485f, 0.456f, 0.406f},
                                    const std::array<float, 3>& image_std = {0.229f, 0.224f, 0.225f});
}

class VisionEncoderVideoChat_Flash : public VisionEncoder {
public:
    VisionEncoderVideoChat_Flash(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderVideoChat_Flash(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);
    
    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

    CircularBufferQueueElementGuard<ov::InferRequest> get_vision_encoder() {
        return m_ireq_queue_vision_encoder.get();
    }

    CircularBufferQueueElementGuard<ov::InferRequest> get_vision_projection() {
        return m_ireq_queue_vision_projection.get();
    }

    CircularBufferQueueElementGuard<ov::InferRequest> get_merge_model() {
        return m_ireq_queue_merge_model.get();
    }

    size_t get_mm_local_num_frames() const {
        return m_vlm_config.mm_local_num_frames;
    }

    ov::Tensor& get_pos_emb() {
        return m_pos_emb;
    }
protected:
    /// @brief  Infer requests queue for video projection model.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_projection;

    /// @brief Infer requests queue for vision merge model.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_merge_model;

    /// @brief A config to follow.
    VLMConfig m_vlm_config;

    /// @brief pos_emb Tensor.
    ov::Tensor m_pos_emb;
};

class InputsEmbedderVideoChat_Flash : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderVideoChat_Flash(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    );

    InputsEmbedderVideoChat_Flash(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos) override;
    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos, const ov::AnyMap& config_map);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 const std::vector<ov::genai::EncodedVideo>& videos,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {},
                                 const std::vector<size_t>& videos_sequence = {},
                                 const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    void update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

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
        const std::vector<EncodedVideo>& videos) const override;

private:
    std::vector<size_t> m_tokens_per_images;
    std::vector<size_t> m_prev_tokens_per_images;
};

} // namespace ov::genai
