// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <filesystem>
#include <mutex>

#include "circular_buffer_queue.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

struct clip_image_u8;

namespace ov::genai {

class VisionEncoderMiniCPMv4_7 : public VisionEncoder {
public:
    VisionEncoderMiniCPMv4_7(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties
    );

    VisionEncoderMiniCPMv4_7(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config
    );

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;

private:
    friend class InputsEmbedderMiniCPMv4_7;

    void update_vision_config(const VLMConfig& vlm_config);

    // Runs the vision graph once per crop and concatenates the outputs into a single
    // [total_tokens, hidden_size] tensor. Shared by the image and video paths.
    ov::Tensor encode_crops(
        const std::vector<clip_image_u8>& crops,
        const std::vector<ImageSize>& crop_sizes,
        const ProcessorConfig& config
    );
};

class InputsEmbedderMiniCPMv4_7 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMiniCPMv4_7(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config
    );

    InputsEmbedderMiniCPMv4_7(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config
    );

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {}
    ) override;

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        const std::vector<ov::genai::EncodedVideo>& videos,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}
    ) override;

    std::vector<ov::genai::EncodedVideo>
    encode_videos(const std::vector<ov::Tensor>& videos, const std::vector<VideoMetadata>& videos_metadata) override;

    std::pair<ov::Tensor, std::optional<int64_t>>
    get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(
        const size_t inputs_embeds_size,
        const size_t history_size,
        int64_t rope_delta
    ) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override {
        return normalize_prompt(prompt, base_id, 0, images, {});
    }

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_image_id,
        size_t base_video_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos
    ) const override;

private:
    void encode_vision_token_ids();

    std::once_flag m_vision_token_ids_once_flag;
    int64_t m_image_token_id = -1;
    int64_t m_video_token_id = -1;
};

}  // namespace ov::genai
