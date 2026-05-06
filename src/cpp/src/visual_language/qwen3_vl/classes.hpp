// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen2vl/classes.hpp"

namespace ov::genai {

class VisionEncoderQwen3VL : public VisionEncoderQwen2VL {
public:
    using VisionEncoderQwen2VL::VisionEncoderQwen2VL;

    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map) override;
};

class InputsEmbedderQwen3VL : public InputsEmbedderQwen2VL {
public:
    InputsEmbedderQwen3VL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderQwen3VL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        const std::vector<ov::genai::EncodedVideo>& videos,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

protected:
    // Vision embeddings position model
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_pos;
    
    // Cached extra inputs for language model
    std::unordered_map<std::string, ov::Tensor> m_lm_extra_inputs{
        {"deepstack_visual_embeds", ov::Tensor()},
        {"visual_pos_masks", ov::Tensor()}
    };

    bool has_lm_extra_input(const std::string& input_name) const;

    void expand_video_tags_in_prompt(
        std::string& unified_prompt,
        const std::vector<EncodedVideo>& encoded_videos,
        const std::vector<size_t>& videos_sequence,
        size_t video_base_id
    ) const override;

    /**
     * @brief Run vision embeddings merger with position interpolation.
     */
    std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence,
        const std::vector<EncodedVideo>& videos,
        const std::vector<size_t>& videos_sequence) override;

    /**
     * @brief Computes interpolated position embeddings.
     * 
     * Calculates position interpolation indices and weights, runs vision_embeddings_pos model,
     * applies bilinear interpolation weights, sums corners, permutes for spatial merge.
     */
    ov::Tensor get_interpolated_pos_embeds(const std::vector<std::array<size_t, 3>>& grids_thw);

    std::vector<std::array<size_t, 3>> get_vision_grid_thw_for_position_ids(
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const std::vector<size_t>& images_sequence,
        const size_t image_id,
        const std::vector<std::array<size_t, 3>>& videos_grid_thw,
        const std::vector<size_t>& videos_sequence,
        const size_t video_id,
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
    ) const override;
};

} // namespace ov::genai
