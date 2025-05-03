// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderQwen2VL : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderQwen2VL : public InputsEmbedder::IInputsEmbedder {
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

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    bool prompt_has_image_tag(const std::string& prompt) const override;

    std::pair<std::string, std::vector<size_t>> normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        size_t n_images
    ) const override;

protected:
    // A model for merging image embeddings (hidden states), rotary_pos_emb and attension_mask.
    // Inputs:
    //  - hidden_states: [N, embed_dim]
    //  - rotary_pos_emb: [?, 40]
    //  - attention_mask: [1, ?, ?]
    // Output: [N, hidden_size]
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_merger;

    ov::Tensor m_position_ids;
    int64_t m_rope_delta = 0;
    ov::Tensor m_merged_image_embeddings;

    virtual ov::Tensor run_image_embeddings_merger(
        const std::vector<EncodedImage>& images, 
        const std::vector<size_t>& images_sequence, 
        const size_t image_id);

    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw);

    ov::Tensor create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const std::vector<size_t>& images_sequence,
        const size_t image_id,
        const int64_t vision_start_token_id
    );
};

namespace qwen2_vl_utils {

std::pair<std::vector<ov::Tensor>, std::vector<std::array<size_t, 3>>> reorder_image_embeds_and_grid_thw(
    const std::vector<EncodedImage>& encoded_images,
    const std::vector<size_t>& images_sequence,
    const size_t image_id
);

ov::Tensor get_attention_mask(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw);

ov::Tensor concatenate_image_embeds(const std::vector<ov::Tensor>& reordered_image_embeds);

} // namespace qwen2vl_utils

} // namespace ov::genai
