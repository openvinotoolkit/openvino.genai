// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderQwen2VL : public VisionEncoder {
    // A model for merging image embeddings (hidden states), rotary_pos_emb and attension_mask.
    // Inputs:
    //  - hidden_states: [N, embed_dim]
    //  - rotary_pos_emb: [?, 40]
    //  - attention_mask: [1, ?, ?]
    // Output: [N, hidden_size]
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_merger;

    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw);

public:
    VisionEncoderQwen2VL(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);


    VisionEncoderQwen2VL(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor run_image_embeddings_merger(const std::vector<EncodedImage>& images, const std::string& prompt, size_t image_id, const VLMConfig& vlm_config);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderQwen2VL : public InputsEmbedder::IInputsEmbedder {
    ov::Tensor m_position_ids;
    int64_t m_rope_delta = 0;

public:
    InputsEmbedderQwen2VL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderQwen2VL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer, 
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor run_image_embeddings_merger(const std::vector<EncodedImage>& images, const std::string& prompt) const override;

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        ov::genai::VLMPerfMetrics& metrics,
        std::optional<ov::Tensor> merged_image_embeddings = std::nullopt // The result of image embeddings merger, can be passed to avoid redundant recalculation.
    ) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    bool prompt_has_image_tag(const std::string& prompt) const override;

protected:
    ov::Tensor merge_text_and_image_embeddings_qwen2vl(
        const ov::Tensor& input_ids,
        const ov::Tensor& text_embeds, 
        const ov::Tensor& merged_image_embeds,
        const int64_t image_pad_token_id
    );

    ov::Tensor create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const int64_t vision_start_token_id
    );
};

} // namespace ov::genai
