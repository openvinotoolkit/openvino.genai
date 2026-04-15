// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen3_vl/classes.hpp"

namespace ov::genai {

class VisionEncoderQwen3_5 : public VisionEncoderQwen3VL {
public:
    using VisionEncoderQwen3VL::VisionEncoderQwen3VL;
};

class InputsEmbedderQwen3_5 : public InputsEmbedderQwen3VL {
public:
    InputsEmbedderQwen3_5(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderQwen3_5(
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

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(
        const size_t inputs_embeds_size,
        const size_t history_size,
        int64_t rope_delta
    ) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

protected:
    /**
     * @brief Run vision embeddings merger with position interpolation.
     */
    std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence,
        const std::vector<EncodedVideo>& videos,
        const std::vector<size_t>& videos_sequence) override;
    
    std::pair<ov::Tensor, int64_t> create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const std::vector<size_t>& images_sequence,
        const size_t image_id,
        const std::vector<std::array<size_t, 3>>& videos_grid_thw,
        const std::vector<size_t>& videos_sequence,
        const size_t video_id,
        const int64_t vision_start_token_id,
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
    ) override;
};

} // namespace ov::genai
