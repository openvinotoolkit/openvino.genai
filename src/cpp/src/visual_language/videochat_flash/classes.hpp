// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace videochat_flash_utils {
    ov::Tensor transpose_video_features(const ov::Tensor& src_tensor, const size_t mm_local_num_frames);
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
    // EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map) override;
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

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    // std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos) override;

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    void update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    std::pair<std::string, std::vector<size_t>> normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

private:
    std::vector<size_t> m_tokens_per_images;
    std::vector<size_t> m_prev_tokens_per_images;
};

} // namespace ov::genai
