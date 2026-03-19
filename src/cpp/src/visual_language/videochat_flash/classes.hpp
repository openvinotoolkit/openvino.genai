// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <array>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderVideoChatFlashQwen : public VisionEncoder {
public:
    VisionEncoderVideoChatFlashQwen(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderVideoChatFlashQwen(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
    
    CircularBufferQueueElementGuard<ov::InferRequest> get_vision_encoder() {
        OPENVINO_ASSERT(m_ireq_queue_vision_encoder, "Vision encoder infer request queue is not initialized");
        return m_ireq_queue_vision_encoder->get();
    }

    CircularBufferQueueElementGuard<ov::InferRequest> get_vision_projection() {
        OPENVINO_ASSERT(m_ireq_queue_vision_projection, "Vision projection infer request queue is not initialized");
        return m_ireq_queue_vision_projection->get();
    }

    CircularBufferQueueElementGuard<ov::InferRequest> get_merge_model() {
        OPENVINO_ASSERT(m_ireq_queue_merge_model, "Vision merge infer request queue is not initialized");
        return m_ireq_queue_merge_model->get();
    }

    const ov::Tensor& get_pos_emb() const {
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

class InputsEmbedderVideoChatFlashQwen : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderVideoChatFlashQwen(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    );

    InputsEmbedderVideoChatFlashQwen(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos) override;

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
